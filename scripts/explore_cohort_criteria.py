"""Diagnóstico exploratório de critérios de elegibilidade de coorte no LIDC-IDRI.

AVISO SOBRE DADOS MÉDICOS
-------------------------
Este script materializa, em ``scripts/output/``, um CSV derivado das anotações do
LIDC-IDRI (identificadores de paciente, diâmetros e escores de malignidade).
Esse conteúdo é DADO DERIVADO DE IMAGEM MÉDICA e NÃO DEVE SER COMMITADO no
repositório. Na primeira execução o script cria ``scripts/output/.gitignore``
com ``*`` justamente para impedir o versionamento acidental desses arquivos.

ESCOPO
------
Script de DIAGNÓSTICO, não de produção. Serve para quantificar o impacto de
critérios candidatos de elegibilidade de coorte e de consolidação da
variável-alvo ANTES de fixar o protocolo formal da Sprint 2. Nada aqui deve ser
importado por ``src/``: quando o protocolo estiver decidido, ele será
reimplementado de forma definitiva no pacote do projeto.

Os critérios candidatos NÃO são aplicados como filtro durante a coleta; eles são
gravados como colunas booleanas independentes, para permitir comparar cenários
sobre o mesmo CSV bruto.

Execução:
    python scripts/explore_cohort_criteria.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pylidc as pl
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


# --------------------------------------------------------------------------- #
# Parâmetros do experimento (todos explícitos, nenhum PatientID hardcoded)
# --------------------------------------------------------------------------- #

RANDOM_STATE: int = 42
MIN_ANNOTATIONS: int = 3
MIN_DIAMETER_MM: float = 3.0
VALID_MALIGNANCY_SCORES: frozenset[int] = frozenset({1, 2, 3, 4, 5})
AMBIGUOUS_MALIGNANCY_SCORE: float = 3.0
BENIGN_MEDIAN_MAX: float = 2.0
MALIGNANT_MEDIAN_MIN: float = 4.0

TRAIN_FRACTION: float = 0.70
VALIDATION_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15
MAX_KFOLD_SPLITS: int = 5
PROGRESS_EVERY: int = 100

OUTPUT_DIRECTORY: Path = Path(__file__).resolve().parent / "output"
OUTPUT_CSV: Path = OUTPUT_DIRECTORY / "cohort_diagnostic_raw.csv"

DATAFRAME_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "nodule_index",
    "n_annotations_total",
    "n_annotations_with_characteristics",
    "mean_diameter_mm",
    "malignancy_values",
    "malignancy_median",
    "passes_min_annotations_3",
    "passes_diameter_3mm",
    "target_binary",
    "is_ambiguous",
)

SEPARATOR: str = "=" * 78


# --------------------------------------------------------------------------- #
# Extração por nódulo
# --------------------------------------------------------------------------- #


def has_characteristics(annotation: pl.Annotation) -> bool:
    """Indica se a annotation traz um escore de malignidade utilizável (1 a 5).

    No LIDC-IDRI apenas nódulos >= 3mm recebem ``characteristics`` preenchidos;
    leituras sem esse bloco chegam ao pylidc com malignancy nulo ou fora da
    escala, e são tratadas aqui como "sem characteristics".
    """
    malignancy = getattr(annotation, "malignancy", None)
    return malignancy is not None and int(malignancy) in VALID_MALIGNANCY_SCORES


def annotation_diameter_mm(annotation: pl.Annotation) -> float | None:
    """Retorna o diâmetro estimado pelo pylidc, ou None se não for calculável."""
    try:
        diameter = float(annotation.diameter)
    except Exception:  # noqa: BLE001 - annotation individual não pode quebrar o cluster
        return None
    return diameter if np.isfinite(diameter) else None


def mean_diameter_mm(cluster: Sequence[pl.Annotation]) -> float | None:
    """Média dos diâmetros calculáveis das annotations do cluster."""
    diameters = [
        diameter
        for diameter in (annotation_diameter_mm(annotation) for annotation in cluster)
        if diameter is not None
    ]
    return float(np.mean(diameters)) if diameters else None


def binarize_malignancy(malignancy_median: float | None) -> float:
    """Consolida a mediana de malignidade em alvo binário.

    Mediana <= 2 vira 0 (benigno), >= 4 vira 1 (maligno) e o restante — incluindo
    a nota 3 e as medianas fracionárias de contagem par (2.5 / 3.5) — vira NaN,
    isto é, alvo indefinido sob o critério candidato.
    """
    if malignancy_median is None or not np.isfinite(malignancy_median):
        return float("nan")
    if malignancy_median <= BENIGN_MEDIAN_MAX:
        return 0.0
    if malignancy_median >= MALIGNANT_MEDIAN_MIN:
        return 1.0
    return float("nan")


def summarize_cluster(
    patient_id: str,
    nodule_index: int,
    cluster: Sequence[pl.Annotation],
) -> dict[str, Any]:
    """Constrói o registro de diagnóstico de um único nódulo (cluster).

    ``nodule_index`` é 0-based e corresponde à posição do cluster na lista
    devolvida por ``Scan.cluster_annotations()``.
    """
    scored_annotations = [
        annotation for annotation in cluster if has_characteristics(annotation)
    ]
    malignancy_values = [int(annotation.malignancy) for annotation in scored_annotations]
    malignancy_median = float(median(malignancy_values)) if malignancy_values else None
    average_diameter = mean_diameter_mm(cluster)

    return {
        "patient_id": patient_id,
        "nodule_index": nodule_index,
        "n_annotations_total": len(cluster),
        "n_annotations_with_characteristics": len(scored_annotations),
        "mean_diameter_mm": average_diameter,
        "malignancy_values": json.dumps(malignancy_values),
        "malignancy_median": malignancy_median,
        "passes_min_annotations_3": len(scored_annotations) >= MIN_ANNOTATIONS,
        "passes_diameter_3mm": (
            average_diameter is not None and average_diameter >= MIN_DIAMETER_MM
        ),
        "target_binary": binarize_malignancy(malignancy_median),
        "is_ambiguous": malignancy_median == AMBIGUOUS_MALIGNANCY_SCORE,
    }


def collect_nodule_records() -> tuple[list[dict[str, Any]], int, list[tuple[str, str]]]:
    """Percorre todos os scans locais e devolve (registros, n_scans, falhas).

    Falhas em um scan (ou em um cluster isolado) são capturadas e acumuladas em
    vez de interromper o loop, para que um caso patológico não invalide a
    varredura inteira.
    """
    records: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    scans = pl.query(pl.Scan).all()

    for position, scan in enumerate(scans, start=1):
        patient_id = str(scan.patient_id)
        try:
            clusters = scan.cluster_annotations()
        except Exception as exc:  # noqa: BLE001 - scan problemático não para a varredura
            failures.append((patient_id, f"cluster_annotations: {exc}"))
            continue

        for nodule_index, cluster in enumerate(clusters):
            try:
                records.append(summarize_cluster(patient_id, nodule_index, cluster))
            except Exception as exc:  # noqa: BLE001 - cluster problemático é pulado
                failures.append((patient_id, f"nódulo {nodule_index}: {exc}"))

        if position % PROGRESS_EVERY == 0 or position == len(scans):
            print(
                f"  ... {position}/{len(scans)} scans processados "
                f"({len(records)} nódulos até aqui)",
                file=sys.stderr,
                flush=True,
            )

    return records, len(scans), failures


def build_dataframe(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Monta o DataFrame em nível de nódulo com o esquema de colunas fixo."""
    frame = pd.DataFrame(list(records), columns=list(DATAFRAME_COLUMNS))
    for column in ("passes_min_annotations_3", "passes_diameter_3mm", "is_ambiguous"):
        frame[column] = frame[column].astype(bool)
    return frame


def write_output_csv(frame: pd.DataFrame) -> None:
    """Grava o CSV bruto, criando a pasta e o .gitignore protetivo se faltarem."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    gitignore_path = OUTPUT_DIRECTORY / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(
            "# Saídas de diagnóstico derivadas de dados médicos: não versionar.\n*\n",
            encoding="utf-8",
        )
    frame.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Relatório de console
# --------------------------------------------------------------------------- #


def eligible_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """Nódulos que passam nos dois critérios candidatos simultaneamente."""
    return frame[frame["passes_min_annotations_3"] & frame["passes_diameter_3mm"]]


def modelable_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """Elegíveis com alvo binário definido (o que de fato entraria no modelo)."""
    eligible = eligible_subset(frame)
    return eligible[eligible["target_binary"].notna()]


def print_section(title: str) -> None:
    """Imprime um cabeçalho de seção do relatório."""
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}")


def report_totals(frame: pd.DataFrame, n_scans: int) -> None:
    """(1) Volume bruto e (2) sobrevivência aos critérios candidatos."""
    print_section("1-2. VOLUME BRUTO E IMPACTO DOS CRITÉRIOS")
    print(f"Scans processados                                : {n_scans}")
    print(f"Nódulos (clusters) encontrados                   : {len(frame)}")

    passes_annotations = int(frame["passes_min_annotations_3"].sum())
    passes_diameter = int(frame["passes_diameter_3mm"].sum())
    eligible = eligible_subset(frame)
    print(
        f"Passam >= {MIN_ANNOTATIONS} annotations com characteristics : "
        f"{passes_annotations} ({share(passes_annotations, len(frame))})"
    )
    print(
        f"Passam diâmetro médio >= {MIN_DIAMETER_MM:.0f}mm                  : "
        f"{passes_diameter} ({share(passes_diameter, len(frame))})"
    )
    print(
        f"Passam AMBOS os critérios (elegíveis)            : "
        f"{len(eligible)} ({share(len(eligible), len(frame))})"
    )


def share(part: int, whole: int) -> str:
    """Formata uma proporção como percentual legível."""
    return f"{100.0 * part / whole:.1f}%" if whole else "n/a"


def report_malignancy_distribution(frame: pd.DataFrame) -> None:
    """(3) Distribuição da mediana de malignidade entre os elegíveis."""
    print_section("3. DISTRIBUIÇÃO DE malignancy_median (NÓDULOS ELEGÍVEIS)")
    eligible = eligible_subset(frame)
    if eligible.empty:
        print("Nenhum nódulo elegível: distribuição indisponível.")
        return

    counts = eligible["malignancy_median"].value_counts(dropna=False).sort_index()
    largest = int(counts.max())
    for value, count in counts.items():
        label = "NaN" if pd.isna(value) else f"{float(value):>4.1f}"
        bar = "#" * max(1, round(50 * count / largest))
        percentage = share(int(count), len(eligible))
        print(f"  mediana {label} | {count:>5} ({percentage:>6}) {bar}")


def report_ambiguity(frame: pd.DataFrame) -> None:
    """(4) Peso da zona cinzenta (nota 3) entre os elegíveis."""
    print_section("4. AMBIGUIDADE (MEDIANA == 3) ENTRE OS ELEGÍVEIS")
    eligible = eligible_subset(frame)
    if eligible.empty:
        print("Nenhum nódulo elegível.")
        return

    ambiguous = int(eligible["is_ambiguous"].sum())
    undefined = int(eligible["target_binary"].isna().sum())
    fractional = undefined - ambiguous
    modelable = modelable_subset(frame)

    print(f"Elegíveis                                       : {len(eligible)}")
    print(
        f"is_ambiguous=True (mediana exatamente 3)        : "
        f"{ambiguous} ({share(ambiguous, len(eligible))})"
    )
    print(
        f"target_binary indefinido (NaN) no total          : "
        f"{undefined} ({share(undefined, len(eligible))})"
    )
    print(
        f"  -> destes, medianas fracionárias 2.5/3.5      : {fractional}\n"
        "     (não marcados como is_ambiguous, mas também sem alvo — decidir "
        "no protocolo\n      se entram na zona cinzenta ou são arredondados)"
    )
    print(f"Elegíveis COM alvo definido (base modelável)    : {len(modelable)}")
    if not modelable.empty:
        class_counts = modelable["target_binary"].value_counts().sort_index()
        for value, count in class_counts.items():
            print(
                f"  classe {int(value)} : {count:>5} "
                f"({share(int(count), len(modelable))})"
            )


def report_patient_level(frame: pd.DataFrame) -> None:
    """(5) Estrutura por paciente — relevante para o split agrupado."""
    print_section("5. ESTRUTURA POR PACIENTE (RELEVANTE PARA O SPLIT)")
    eligible = eligible_subset(frame)
    if eligible.empty:
        print("Nenhum nódulo elegível.")
        return

    per_patient = eligible.groupby("patient_id").size()
    print(f"Pacientes únicos com >= 1 nódulo elegível        : {len(per_patient)}")
    print(
        f"Pacientes com mais de 1 nódulo elegível          : "
        f"{int((per_patient > 1).sum())} ({share(int((per_patient > 1).sum()), len(per_patient))})"
    )
    print(f"Nódulos elegíveis por paciente (máximo)          : {int(per_patient.max())}")
    print(f"Nódulos elegíveis por paciente (média)           : {per_patient.mean():.2f}")

    modelable = modelable_subset(frame)
    if not modelable.empty:
        per_patient_modelable = modelable.groupby("patient_id").size()
        print(
            f"Pacientes na base modelável (alvo definido)      : "
            f"{len(per_patient_modelable)}"
        )
        print(
            f"  destes, com mais de 1 nódulo modelável        : "
            f"{int((per_patient_modelable > 1).sum())}"
        )
        mixed = (
            modelable.groupby("patient_id")["target_binary"].nunique().gt(1).sum()
        )
        print(f"  pacientes com nódulos de AMBAS as classes     : {int(mixed)}")


def report_stratified_group_kfold(frame: pd.DataFrame) -> None:
    """(6a) Viabilidade via StratifiedGroupKFold agrupado por paciente."""
    print_section("6a. SIMULAÇÃO DE SPLIT: StratifiedGroupKFold (groups=patient_id)")
    modelable = modelable_subset(frame)
    if modelable.empty:
        print("Base modelável vazia: simulação impossível.")
        return

    targets = modelable["target_binary"].astype(int).to_numpy()
    groups = modelable["patient_id"].to_numpy()
    n_groups = len(np.unique(groups))
    smallest_class = int(np.bincount(targets).min()) if targets.size else 0
    n_splits = min(MAX_KFOLD_SPLITS, n_groups, smallest_class)

    if n_splits < 2:
        print(
            "Amostra insuficiente para validação cruzada agrupada "
            f"(grupos={n_groups}, menor classe={smallest_class}; seriam necessários "
            ">= 2 de cada)."
        )
        return

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    print(f"n_splits={n_splits} | shuffle=True | random_state={RANDOM_STATE}")
    try:
        folds = list(splitter.split(np.zeros(len(targets)), targets, groups=groups))
    except Exception as exc:  # noqa: BLE001 - sklearn falha com amostras degeneradas
        print(f"StratifiedGroupKFold falhou: {exc}")
        return

    for fold_index, (train_index, test_index) in enumerate(folds, start=1):
        train_counts = np.bincount(targets[train_index], minlength=2)
        test_counts = np.bincount(targets[test_index], minlength=2)
        print(
            f"  fold {fold_index}: treino n={len(train_index):>5} "
            f"(classe0={train_counts[0]:>4}, classe1={train_counts[1]:>4}, "
            f"pacientes={len(np.unique(groups[train_index])):>4}) | "
            f"teste n={len(test_index):>4} "
            f"(classe0={test_counts[0]:>4}, classe1={test_counts[1]:>4}, "
            f"pacientes={len(np.unique(groups[test_index])):>4})"
        )


def report_holdout_simulation(frame: pd.DataFrame) -> None:
    """(6b) Viabilidade da partição 70/15/15 por paciente via GroupShuffleSplit."""
    print_section("6b. SIMULAÇÃO DE SPLIT: 70/15/15 por paciente (GroupShuffleSplit)")
    modelable = modelable_subset(frame)
    if modelable.empty:
        print("Base modelável vazia: simulação impossível.")
        return

    targets = modelable["target_binary"].astype(int).to_numpy()
    groups = modelable["patient_id"].to_numpy()
    if len(np.unique(groups)) < 3:
        print(
            f"Apenas {len(np.unique(groups))} paciente(s) na base modelável: "
            "impossível formar três partições disjuntas por paciente."
        )
        return

    features = np.zeros((len(targets), 1))
    try:
        first_stage = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_FRACTION + TEST_FRACTION,
            random_state=RANDOM_STATE,
        )
        train_index, holdout_index = next(first_stage.split(features, targets, groups))

        second_stage = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
            random_state=RANDOM_STATE,
        )
        relative_validation, relative_test = next(
            second_stage.split(
                features[holdout_index],
                targets[holdout_index],
                groups[holdout_index],
            )
        )
    except Exception as exc:  # noqa: BLE001 - amostra pequena demais para o split
        print(f"GroupShuffleSplit falhou: {exc}")
        return

    partitions = {
        "treino": train_index,
        "validação": holdout_index[relative_validation],
        "teste": holdout_index[relative_test],
    }
    print(
        f"Alvo nominal: {TRAIN_FRACTION:.0%}/{VALIDATION_FRACTION:.0%}/"
        f"{TEST_FRACTION:.0%} | random_state={RANDOM_STATE} | grupos=patient_id"
    )
    for name, index in partitions.items():
        counts = np.bincount(targets[index], minlength=2)
        print(
            f"  {name:<10}: n={len(index):>5} ({share(len(index), len(targets)):>6}) | "
            f"classe0={counts[0]:>4} | classe1={counts[1]:>4} | "
            f"pacientes={len(np.unique(groups[index])):>4}"
        )

    overlap = set(groups[partitions["treino"]]) & (
        set(groups[partitions["validação"]]) | set(groups[partitions["teste"]])
    )
    print(f"  vazamento de paciente entre partições           : {len(overlap)}")


def report_failures(failures: Sequence[tuple[str, str]]) -> None:
    """Consolida as falhas toleradas durante a varredura."""
    print_section("FALHAS TOLERADAS DURANTE A VARREDURA")
    print(f"Ocorrências registradas: {len(failures)}")
    for patient_id, message in failures[:20]:
        print(f"  - {patient_id}: {message}")
    if len(failures) > 20:
        print(f"  ... e mais {len(failures) - 20} ocorrência(s).")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main() -> int:
    """Executa a varredura completa, grava o CSV e imprime o relatório."""
    print("Varrendo scans do LIDC-IDRI via pylidc...", file=sys.stderr, flush=True)
    try:
        records, n_scans, failures = collect_nodule_records()
    except Exception as exc:  # noqa: BLE001 - converte falha global em exit code 1
        print(f"ERRO: falha ao consultar o pylidc: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    frame = build_dataframe(records)
    write_output_csv(frame)

    print_section("DIAGNÓSTICO DE CRITÉRIOS DE COORTE - LIDC-IDRI")
    print(f"CSV bruto salvo em: {OUTPUT_CSV}")
    print("AVISO: conteúdo derivado de dados médicos. Não commitar.")

    report_totals(frame, n_scans)
    report_malignancy_distribution(frame)
    report_ambiguity(frame)
    report_patient_level(frame)
    report_stratified_group_kfold(frame)
    report_holdout_simulation(frame)
    report_failures(failures)

    print(f"\n{SEPARATOR}\nDiagnóstico concluído.\n{SEPARATOR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
