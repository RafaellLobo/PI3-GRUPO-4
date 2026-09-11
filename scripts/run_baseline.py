"""Baseline oficial da Sprint 3 — Regressão Logística sobre três features Shape.

AVISO SOBRE DADOS MÉDICOS
-------------------------
Lê ``data/modeling_table_sprint3.csv`` (dado derivado, não versionado) e
``reports/sprint3/baseline/patient_split.csv``. Escreve apenas métricas e
matrizes de confusão agregadas em ``reports/sprint3/baseline/`` — nenhum
identificador de paciente é gravado pelos artefatos deste script.

ESCOPO
------
Baseline formal, deliberadamente mínimo: três atributos de forma, um
classificador linear, nenhum tuning. Serve como piso de referência contra o qual
modelos posteriores devem ser comparados — não é o modelo final.

Não faz tuning, não faz seleção de features, não refaz o split e não altera a
tabela de modelagem.

A incerteza das métricas finais (partição ``test``) é estimada por bootstrap
percentil reamostrando PACIENTES, sobre o modelo já treinado — ver
``bootstrap_ic_por_paciente``.

PREVENÇÃO DE VAZAMENTO
----------------------
O ``StandardScaler`` vive DENTRO do ``Pipeline``, que é ajustado exclusivamente
na partição ``train``. Validação e teste passam apenas por ``transform``, com as
médias e desvios aprendidos no treino — nunca por um ``fit`` próprio. Isso
cumpre a Seção 6.3 do protocolo, que proíbe ajustar qualquer transformação
sobre o dataset completo antes do split. O JSON registra
``scaler.n_samples_seen_`` como evidência verificável de quantas linhas o scaler
efetivamente viu.

Execução:
    python scripts/run_baseline.py
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SCRIPTS_DIR: Path = Path(__file__).resolve().parent
for _path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import sklearn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


# --------------------------------------------------------------------------- #
# Caminhos e parâmetros
# --------------------------------------------------------------------------- #

MODELING_CSV: Path = REPO_ROOT / "data" / "modeling_table_sprint3.csv"
SPLIT_CSV: Path = REPO_ROOT / "reports" / "sprint3" / "baseline" / "patient_split.csv"
OUTPUT_DIR: Path = REPO_ROOT / "reports" / "sprint3" / "baseline"
METRICS_JSON: Path = OUTPUT_DIR / "baseline_metrics.json"
CONFUSION_CSV: Path = OUTPUT_DIR / "confusion_matrix.csv"

#: As três — e somente estas três — features do baseline.
BASELINE_FEATURES: tuple[str, ...] = (
    "original_shape_MeshVolume",
    "original_shape_Maximum3DDiameter",
    "original_shape_Sphericity",
)

TARGET_COLUMN: str = "target_binary"
PATIENT_COLUMN: str = "patient_id"
NODULE_COLUMN: str = "nodulo_id"

RANDOM_STATE: int = 42
THRESHOLD: float = 0.5

LOGREG_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "penalty": "l2",
    "class_weight": None,
    "random_state": RANDOM_STATE,
    "max_iter": 1000,
}

#: Bootstrap de incerteza das métricas finais (partição test).
#: A reamostragem é POR PACIENTE, nunca por nódulo: um paciente pode ter vários
#: nódulos e eles não são observações independentes. Reamostrar nódulos
#: individualmente subestimaria a largura do intervalo.
BOOTSTRAP_N: int = 1000
BOOTSTRAP_LOWER_PERCENTILE: float = 2.5
BOOTSTRAP_UPPER_PERCENTILE: float = 97.5

SPLIT_TRAIN: str = "train"
SPLIT_VALIDATION: str = "validation"
SPLIT_TEST: str = "test"
SPLIT_ORDER: tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)
EVAL_SPLITS: tuple[str, ...] = (SPLIT_VALIDATION, SPLIT_TEST)

# --------------------------------------------------------------------------- #
# Invariantes obrigatórias (checadas ANTES do treinamento)
# --------------------------------------------------------------------------- #

EXPECTED_NODULES: int = 607
EXPECTED_PATIENTS: int = 416
EXPECTED_CLASS_0: int = 310
EXPECTED_CLASS_1: int = 297

SEPARATOR: str = "=" * 78


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #


def carregar() -> pd.DataFrame:
    """Une tabela de modelagem e split, sem alterar nenhum dos dois arquivos."""
    for caminho, dica in (
        (MODELING_CSV, "python scripts/build_modeling_table.py"),
        (SPLIT_CSV, "python scripts/make_patient_split.py"),
    ):
        if not caminho.exists():
            raise FileNotFoundError(f"{caminho} não encontrado. Rode antes: {dica}")

    colunas = [NODULE_COLUMN, PATIENT_COLUMN, TARGET_COLUMN, *BASELINE_FEATURES]
    tabela = pd.read_csv(MODELING_CSV, usecols=colunas)
    split = pd.read_csv(SPLIT_CSV)

    unido = tabela.merge(split, on=PATIENT_COLUMN, how="left", validate="many_to_one")
    return unido.sort_values(NODULE_COLUMN, kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Validação — tudo aqui roda ANTES do treinamento e de qualquer escrita
# --------------------------------------------------------------------------- #


def _exigir(condicao: bool, mensagem: str) -> None:
    """Interrompe a execução se a invariante não vale."""
    if not condicao:
        raise AssertionError(mensagem)


def validar(frame: pd.DataFrame) -> list[str]:
    """Valida as invariantes do contrato; devolve o log de checagens."""
    log: list[str] = []

    def ok(rotulo: str, valor: Any, esperado: Any) -> None:
        _exigir(
            valor == esperado,
            f"INVARIANTE VIOLADA — {rotulo}: esperado {esperado}, obtido {valor}. "
            "Nada foi treinado nem escrito.",
        )
        log.append(f"  [OK] {rotulo:<56} {valor}")

    ok("nódulos", len(frame), EXPECTED_NODULES)
    ok("pacientes únicos", int(frame[PATIENT_COLUMN].nunique()), EXPECTED_PATIENTS)

    contagem = frame[TARGET_COLUMN].value_counts()
    ok("classe 0", int(contagem.get(0, 0)), EXPECTED_CLASS_0)
    ok("classe 1", int(contagem.get(1, 0)), EXPECTED_CLASS_1)

    # Todo nódulo pertence a exatamente uma partição.
    sem_particao = int(frame["split"].isna().sum())
    _exigir(
        sem_particao == 0,
        f"INVARIANTE VIOLADA — {sem_particao} nódulos sem partição. "
        "Nada foi treinado nem escrito.",
    )
    log.append(f"  [OK] {'nódulos sem partição':<56} 0")

    por_nodulo = frame.groupby(NODULE_COLUMN)["split"].nunique()
    ambiguos = por_nodulo[por_nodulo != 1].index.tolist()
    _exigir(
        not ambiguos,
        f"INVARIANTE VIOLADA — nódulos em mais de uma partição: {ambiguos[:5]}. "
        "Nada foi treinado nem escrito.",
    )
    log.append(f"  [OK] {'nódulos em exatamente 1 partição':<56} {len(por_nodulo)}")

    desconhecidas = set(frame["split"]) - set(SPLIT_ORDER)
    _exigir(
        not desconhecidas,
        f"INVARIANTE VIOLADA — partições desconhecidas: {sorted(desconhecidas)}. "
        "Nada foi treinado nem escrito.",
    )

    # Disjunção de pacientes entre partições — a checagem central de leakage.
    conjuntos = {
        nome: set(frame.loc[frame["split"] == nome, PATIENT_COLUMN])
        for nome in SPLIT_ORDER
    }
    for esquerda, direita in (
        (SPLIT_TRAIN, SPLIT_VALIDATION),
        (SPLIT_TRAIN, SPLIT_TEST),
        (SPLIT_VALIDATION, SPLIT_TEST),
    ):
        comum = conjuntos[esquerda] & conjuntos[direita]
        _exigir(
            not comum,
            f"INVARIANTE VIOLADA — pacientes compartilhados entre {esquerda} e "
            f"{direita}: {sorted(comum)[:5]}. Nada foi treinado nem escrito.",
        )
    log.append(f"  [OK] {'pacientes compartilhados entre partições':<56} 0")

    # NaN/Inf nas três features do baseline.
    for coluna in BASELINE_FEATURES:
        valores = pd.to_numeric(frame[coluna], errors="coerce").to_numpy(dtype=float)
        n_nan = int(np.isnan(valores).sum())
        n_inf = int(np.isinf(valores).sum())
        _exigir(
            n_nan == 0 and n_inf == 0,
            f"INVARIANTE VIOLADA — {coluna} contém {n_nan} NaN e {n_inf} Inf. "
            "Nada foi treinado nem escrito.",
        )
        log.append(f"  [OK] {coluna + ' (NaN/Inf)':<56} 0 / 0")

    return log


# --------------------------------------------------------------------------- #
# Treino e avaliação
# --------------------------------------------------------------------------- #


def construir_pipeline() -> Pipeline:
    """StandardScaler -> LogisticRegression, nesta ordem, num único estimador.

    Manter o scaler dentro do Pipeline é o que garante que ele seja ajustado
    somente no treino: ``fit`` no Pipeline propaga ``fit_transform`` ao scaler
    apenas nos dados passados, e ``predict_proba`` usa ``transform``.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(**LOGREG_PARAMS)),
        ]
    )


def avaliar(modelo: Pipeline, bloco: pd.DataFrame) -> dict[str, Any]:
    """Calcula as métricas de uma partição com threshold fixo."""
    X = bloco[list(BASELINE_FEATURES)].to_numpy(dtype=float)
    y = bloco[TARGET_COLUMN].to_numpy(dtype=int)

    proba = modelo.predict_proba(X)[:, 1]
    # Threshold explícito, não `predict()`: o corte faz parte do contrato.
    predito = (proba >= THRESHOLD).astype(int)

    matriz = confusion_matrix(y, predito, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matriz.ravel())

    return {
        "n_nodulos": int(len(bloco)),
        "n_pacientes": int(bloco[PATIENT_COLUMN].nunique()),
        "classe_0": int((y == 0).sum()),
        "classe_1": int((y == 1).sum()),
        "auc_roc": float(roc_auc_score(y, proba)),
        "recall_classe_1": float(recall_score(y, predito, pos_label=1, zero_division=0)),
        "especificidade_classe_0": float(
            recall_score(y, predito, pos_label=0, zero_division=0)
        ),
        "f1_classe_1": float(f1_score(y, predito, pos_label=1, zero_division=0)),
        "matriz_confusao": {
            "verdadeiro_negativo": tn,
            "falso_positivo": fp,
            "falso_negativo": fn,
            "verdadeiro_positivo": tp,
        },
    }


def bootstrap_ic_por_paciente(
    modelo: Pipeline,
    bloco: pd.DataFrame,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """IC 95% percentil das métricas de ``bloco``, reamostrando PACIENTES.

    O modelo já está treinado e NÃO é reajustado aqui: probabilidades e
    predições são calculadas uma única vez sobre o bloco inteiro, e cada réplica
    apenas reindexa esses vetores. Nenhum ``fit`` de ``StandardScaler`` ou
    ``LogisticRegression`` ocorre dentro do laço.

    A unidade de reamostragem é ``patient_id``: sorteia-se, com reposição, o
    mesmo número de pacientes que a partição possui, e cada paciente sorteado
    entra com TODOS os seus nódulos. Um paciente sorteado k vezes contribui com
    seus nódulos k vezes.

    Réplicas em que o sorteio produz uma única classe não permitem calcular
    AUC/recall/especificidade; elas são descartadas e contabilizadas em
    ``n_bootstrap_valid``. O dataset não é alterado para evitar esse caso.
    """
    X = bloco[list(BASELINE_FEATURES)].to_numpy(dtype=float)
    y = bloco[TARGET_COLUMN].to_numpy(dtype=int)

    proba = modelo.predict_proba(X)[:, 1]
    predito = (proba >= THRESHOLD).astype(int)

    pacientes = bloco[PATIENT_COLUMN].to_numpy()
    unicos = np.unique(pacientes)
    linhas_do_paciente = {p: np.flatnonzero(pacientes == p) for p in unicos}

    rng = np.random.default_rng(seed)
    amostras: dict[str, list[float]] = {
        "auc_roc": [],
        "recall_classe_1": [],
        "especificidade_classe_0": [],
        "f1_classe_1": [],
    }
    validas = 0

    for _ in range(n_bootstrap):
        sorteados = rng.choice(unicos, size=len(unicos), replace=True)
        indices = np.concatenate([linhas_do_paciente[p] for p in sorteados])

        y_b = y[indices]
        if len(np.unique(y_b)) < 2:
            continue

        proba_b = proba[indices]
        pred_b = predito[indices]
        validas += 1

        amostras["auc_roc"].append(float(roc_auc_score(y_b, proba_b)))
        amostras["recall_classe_1"].append(
            float(recall_score(y_b, pred_b, pos_label=1, zero_division=0))
        )
        amostras["especificidade_classe_0"].append(
            float(recall_score(y_b, pred_b, pos_label=0, zero_division=0))
        )
        amostras["f1_classe_1"].append(
            float(f1_score(y_b, pred_b, pos_label=1, zero_division=0))
        )

    pontuais = {
        "auc_roc": float(roc_auc_score(y, proba)),
        "recall_classe_1": float(recall_score(y, predito, pos_label=1, zero_division=0)),
        "especificidade_classe_0": float(
            recall_score(y, predito, pos_label=0, zero_division=0)
        ),
        "f1_classe_1": float(f1_score(y, predito, pos_label=1, zero_division=0)),
    }

    metricas: dict[str, Any] = {}
    for nome, valores in amostras.items():
        metricas[nome] = {
            "estimativa_pontual": pontuais[nome],
            "ic_95_inferior": (
                float(np.percentile(valores, BOOTSTRAP_LOWER_PERCENTILE))
                if valores
                else None
            ),
            "ic_95_superior": (
                float(np.percentile(valores, BOOTSTRAP_UPPER_PERCENTILE))
                if valores
                else None
            ),
        }

    return {
        "metodo": "patient_level_percentile_bootstrap",
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid": int(validas),
        "n_bootstrap_descartadas": int(n_bootstrap - validas),
        "seed": int(seed),
        "unidade_reamostragem": "patient_id",
        "percentis": [BOOTSTRAP_LOWER_PERCENTILE, BOOTSTRAP_UPPER_PERCENTILE],
        "n_pacientes_reamostrados_por_replica": int(len(unicos)),
        "n_nodulos_test": int(len(bloco)),
        "modelo_retreinado_no_bootstrap": False,
        "metricas": metricas,
    }


def montar_confusao_csv(metricas: dict[str, Any]) -> pd.DataFrame:
    """Formato longo: uma linha por (partição, classe real, classe prevista)."""
    linhas = []
    for split in EVAL_SPLITS:
        matriz = metricas[split]["matriz_confusao"]
        for real, previsto, chave in (
            (0, 0, "verdadeiro_negativo"),
            (0, 1, "falso_positivo"),
            (1, 0, "falso_negativo"),
            (1, 1, "verdadeiro_positivo"),
        ):
            linhas.append(
                {
                    "split": split,
                    "true_label": real,
                    "pred_label": previsto,
                    "count": matriz[chave],
                }
            )
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #


def imprimir_metricas(metricas: dict[str, Any]) -> None:
    """Imprime métricas e matrizes de confusão das partições avaliadas."""
    print(f"\n{SEPARATOR}\nMÉTRICAS (threshold = {THRESHOLD})\n{SEPARATOR}")
    cabecalho = (
        f"  {'partição':<12} {'AUC-ROC':>9} {'Recall c1':>11} "
        f"{'Especif. c0':>13} {'F1 c1':>9}"
    )
    print(cabecalho)
    print(f"  {'-' * (len(cabecalho) - 2)}")
    for split in EVAL_SPLITS:
        m = metricas[split]
        print(
            f"  {split:<12} {m['auc_roc']:>9.4f} {m['recall_classe_1']:>11.4f} "
            f"{m['especificidade_classe_0']:>13.4f} {m['f1_classe_1']:>9.4f}"
        )

    print(f"\n{SEPARATOR}\nMATRIZES DE CONFUSÃO\n{SEPARATOR}")
    for split in EVAL_SPLITS:
        c = metricas[split]["matriz_confusao"]
        print(f"\n  {split}  (n={metricas[split]['n_nodulos']})")
        print(f"                    previsto 0   previsto 1")
        print(f"      real 0 {c['verdadeiro_negativo']:>12} {c['falso_positivo']:>12}")
        print(f"      real 1 {c['falso_negativo']:>12} {c['verdadeiro_positivo']:>12}")


def imprimir_ic(bootstrap: dict[str, Any]) -> None:
    """Imprime os intervalos de confiança de 95% da partição test."""
    print(f"\n{SEPARATOR}\nIC 95% — BOOTSTRAP POR PACIENTE (partição test)\n{SEPARATOR}")
    print(
        f"  método={bootstrap['metodo']} | réplicas válidas="
        f"{bootstrap['n_bootstrap_valid']}/{bootstrap['n_bootstrap_requested']} | "
        f"seed={bootstrap['seed']}"
    )
    print(
        f"  unidade={bootstrap['unidade_reamostragem']} | "
        f"{bootstrap['n_pacientes_reamostrados_por_replica']} pacientes por réplica | "
        f"modelo retreinado: {bootstrap['modelo_retreinado_no_bootstrap']}"
    )
    cabecalho = f"\n  {'métrica':<26} {'estimativa':>11} {'IC 95% inferior':>17} {'IC 95% superior':>17}"
    print(cabecalho)
    print(f"  {'-' * (len(cabecalho) - 4)}")
    for nome, valores in bootstrap["metricas"].items():
        print(
            f"  {nome:<26} {valores['estimativa_pontual']:>11.4f} "
            f"{valores['ic_95_inferior']:>17.4f} {valores['ic_95_superior']:>17.4f}"
        )


def imprimir_coeficientes(modelo: Pipeline) -> None:
    """Imprime coeficientes aprendidos, no espaço padronizado."""
    logreg = modelo.named_steps["logreg"]
    print(f"\n{SEPARATOR}\nCOEFICIENTES (espaço padronizado pelo StandardScaler)\n{SEPARATOR}")
    for nome, coef in zip(BASELINE_FEATURES, logreg.coef_[0]):
        print(f"  {nome:<40} {coef:>+10.6f}")
    print(f"  {'intercepto':<40} {logreg.intercept_[0]:>+10.6f}")


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #


def main() -> int:
    """Valida, treina no train, avalia em validation e test e grava os artefatos."""
    try:
        frame = carregar()
    except Exception as exc:  # noqa: BLE001 - converte falha de carga em exit code 1
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    print(f"{SEPARATOR}\nVALIDAÇÃO DE INVARIANTES (antes do treinamento)\n{SEPARATOR}")
    try:
        log = validar(frame)
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    print("\n".join(log))

    treino = frame[frame["split"] == SPLIT_TRAIN]
    X_treino = treino[list(BASELINE_FEATURES)].to_numpy(dtype=float)
    y_treino = treino[TARGET_COLUMN].to_numpy(dtype=int)

    modelo = construir_pipeline()

    avisos: list[str] = []
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        try:
            modelo.fit(X_treino, y_treino)
            metricas_por_split = {
                split: avaliar(modelo, frame[frame["split"] == split])
                for split in EVAL_SPLITS
            }
            # Incerteza das métricas finais; usa o modelo já treinado acima.
            bootstrap = bootstrap_ic_por_paciente(
                modelo, frame[frame["split"] == SPLIT_TEST]
            )
        except Exception as exc:  # noqa: BLE001 - falha de ajuste não escreve nada
            print(f"ERRO: falha ao treinar/avaliar: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1
        avisos = [f"{w.category.__name__}: {w.message}" for w in capturados]

    scaler = modelo.named_steps["scaler"]
    logreg = modelo.named_steps["logreg"]

    # Evidência verificável de que o scaler viu apenas o treino.
    n_vistos = int(np.atleast_1d(scaler.n_samples_seen_)[0])
    _exigir(
        n_vistos == len(treino),
        f"INVARIANTE VIOLADA — scaler viu {n_vistos} amostras, mas o treino tem "
        f"{len(treino)}. Nada foi escrito.",
    )

    composicao = {
        split: {
            "n_nodulos": int((frame["split"] == split).sum()),
            "n_pacientes": int(frame.loc[frame["split"] == split, PATIENT_COLUMN].nunique()),
            "classe_0": int(
                ((frame["split"] == split) & (frame[TARGET_COLUMN] == 0)).sum()
            ),
            "classe_1": int(
                ((frame["split"] == split) & (frame[TARGET_COLUMN] == 1)).sum()
            ),
        }
        for split in SPLIT_ORDER
    }

    resultado: dict[str, Any] = {
        "modelo": "LogisticRegression",
        "pipeline": ["StandardScaler", "LogisticRegression"],
        "features": list(BASELINE_FEATURES),
        "target": TARGET_COLUMN,
        "parametros_logreg": LOGREG_PARAMS,
        "threshold": THRESHOLD,
        "seed": RANDOM_STATE,
        "versao_sklearn": sklearn.__version__,
        "fontes": {
            "tabela_modelagem": str(MODELING_CSV.relative_to(REPO_ROOT)).replace("\\", "/"),
            "split": str(SPLIT_CSV.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "composicao_particoes": composicao,
        "distribuicao_classes_total": {
            "classe_0": EXPECTED_CLASS_0,
            "classe_1": EXPECTED_CLASS_1,
            "total": EXPECTED_NODULES,
        },
        "prevencao_vazamento": {
            "scaler_ajustado_somente_no_treino": True,
            "scaler_n_samples_seen": n_vistos,
            "n_nodulos_treino": int(len(treino)),
            "evidencia": (
                "StandardScaler vive dentro do Pipeline, ajustado apenas com a "
                "partição train; validation e test passam somente por transform. "
                "n_samples_seen_ == n_nodulos_treino comprova."
            ),
            "pacientes_compartilhados_entre_particoes": 0,
        },
        "coeficientes": {
            "espaco": "padronizado (StandardScaler), não interpretável em unidades originais",
            "por_feature": {
                nome: float(coef)
                for nome, coef in zip(BASELINE_FEATURES, logreg.coef_[0])
            },
            "intercepto": float(logreg.intercept_[0]),
            "scaler_mean": {
                nome: float(v) for nome, v in zip(BASELINE_FEATURES, scaler.mean_)
            },
            "scaler_scale": {
                nome: float(v) for nome, v in zip(BASELINE_FEATURES, scaler.scale_)
            },
            "n_iteracoes": int(np.atleast_1d(logreg.n_iter_)[0]),
        },
        "metricas": metricas_por_split,
        "bootstrap_ci_95_test": bootstrap,
        "warnings_sklearn": avisos,
    }

    imprimir_metricas(metricas_por_split)
    imprimir_ic(bootstrap)
    imprimir_coeficientes(modelo)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    montar_confusao_csv(metricas_por_split).to_csv(
        CONFUSION_CSV, index=False, encoding="utf-8"
    )

    print(f"\n{SEPARATOR}")
    print(f"Métricas gravadas em : {METRICS_JSON}")
    print(f"Confusão gravada em  : {CONFUSION_CSV}")
    if avisos:
        print(f"\nWarnings do scikit-learn ({len(avisos)}):")
        for aviso in avisos:
            print(f"  - {aviso}")
    else:
        print("\nWarnings do scikit-learn: nenhum.")
    print(SEPARATOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
