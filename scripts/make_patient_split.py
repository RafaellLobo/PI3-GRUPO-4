"""Gera e persiste o split de pacientes para o baseline da Sprint 3.

AVISO SOBRE DADOS MÉDICOS
-------------------------
Este script lê ``data/modeling_table_sprint3.csv`` (dado derivado, não
versionado) e escreve ``reports/sprint3/baseline/patient_split.csv``, que contém
apenas ``patient_id`` e a partição atribuída — nenhum escore, atributo ou
medida. O split é versionado de propósito: é ele que torna auditável qualquer
métrica reportada depois.

ESCOPO
------
Materializa a divisão treino/validação/teste POR PACIENTE definida em
``docs/protocolo_coorte_target_sprint2.md`` (Seção 6). Não treina modelo, não
escala, não seleciona features e não altera a tabela de modelagem.

MÉTODO
------
``GroupShuffleSplit`` em dois estágios, ``groups=patient_id``, conforme o
protocolo (Seção 6.1) e a simulação já registrada em
``scripts/explore_cohort_criteria.py``:

1. treino (70%)  x  holdout (30%)
2. holdout dividido ao meio  ->  validação (15%)  e  teste (15%)

A divisão é feita sobre as linhas de nódulo com ``groups=patient_id``, de modo
que as proporções mirem a massa de NÓDULOS, mas a unidade indivisível seja
sempre o PACIENTE — nenhum paciente pode cair em duas partições (leakage
anatômico/fisiológico, protocolo Seção 6).

LIMITAÇÃO CONHECIDA — não é um defeito deste script
---------------------------------------------------
``GroupShuffleSplit`` agrupa mas NÃO estratifica: ele ignora ``y``. Com apenas
uma fração dos pacientes possuindo nódulos de ambas as classes, a proporção de
classe 1 varia entre as partições. Isso está documentado no protocolo
(Seção 6.2) e deve ser refletido em intervalos de confiança ao reportar as
métricas finais, não corrigido aqui por reamostragem.

DETERMINISMO
------------
``random_state=42`` e ordenação canônica por ``nodulo_id`` antes do split: o
resultado não depende da ordem física das linhas do CSV de entrada.

Execução:
    python scripts/make_patient_split.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SCRIPTS_DIR: Path = Path(__file__).resolve().parent
for _path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

# Única definição das frações e da seed no projeto.
from explore_cohort_criteria import (  # noqa: E402
    RANDOM_STATE,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)


# --------------------------------------------------------------------------- #
# Caminhos e parâmetros
# --------------------------------------------------------------------------- #

INPUT_CSV: Path = REPO_ROOT / "data" / "modeling_table_sprint3.csv"
OUTPUT_CSV: Path = REPO_ROOT / "reports" / "sprint3" / "baseline" / "patient_split.csv"

SPLIT_TRAIN: str = "train"
SPLIT_VALIDATION: str = "validation"
SPLIT_TEST: str = "test"
SPLIT_ORDER: tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)

#: Colunas mínimas exigidas na tabela de modelagem.
REQUIRED_COLUMNS: tuple[str, ...] = ("nodulo_id", "patient_id", "target_binary")

#: Colunas do CSV de saída: uma linha por paciente, nada além do necessário.
OUTPUT_COLUMNS: tuple[str, ...] = ("patient_id", "split")

# --------------------------------------------------------------------------- #
# Invariantes estruturais obrigatórias
#
# Checadas ANTES da escrita: um split parcial ou com vazamento é pior que
# nenhum, porque contamina silenciosamente toda métrica reportada depois.
# --------------------------------------------------------------------------- #

EXPECTED_PATIENTS: int = 416
EXPECTED_NODULES: int = 607

SEPARATOR: str = "=" * 78


# --------------------------------------------------------------------------- #
# Geração do split
# --------------------------------------------------------------------------- #


def carregar_tabela(csv_path: Path) -> pd.DataFrame:
    """Carrega a tabela de modelagem em ordem canônica, sem modificá-la."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} não encontrado. Rode antes: "
            "python scripts/build_modeling_table.py"
        )

    frame = pd.read_csv(csv_path, usecols=list(REQUIRED_COLUMNS))
    for coluna in REQUIRED_COLUMNS:
        if coluna not in frame.columns:
            raise ValueError(f"coluna obrigatória ausente em {csv_path}: {coluna}")

    # Ordem canônica: torna o split independente da ordem física do arquivo.
    return frame.sort_values("nodulo_id", kind="mergesort").reset_index(drop=True)


def dividir_por_paciente(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Divide em treino/validação/teste agrupando por ``patient_id``.

    Devolve ``{particao: array de patient_id}``. O ``y`` é passado ao splitter
    apenas por conformidade de assinatura: ``GroupShuffleSplit`` não estratifica.
    """
    alvos = frame["target_binary"].astype(int).to_numpy()
    grupos = frame["patient_id"].astype(str).to_numpy()
    features = np.zeros((len(alvos), 1))

    holdout_fraction = VALIDATION_FRACTION + TEST_FRACTION

    primeiro = GroupShuffleSplit(
        n_splits=1, test_size=holdout_fraction, random_state=RANDOM_STATE
    )
    idx_treino, idx_holdout = next(primeiro.split(features, alvos, grupos))

    segundo = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_FRACTION / holdout_fraction,
        random_state=RANDOM_STATE,
    )
    rel_validacao, rel_teste = next(
        segundo.split(
            features[idx_holdout], alvos[idx_holdout], grupos[idx_holdout]
        )
    )

    return {
        SPLIT_TRAIN: np.unique(grupos[idx_treino]),
        SPLIT_VALIDATION: np.unique(grupos[idx_holdout[rel_validacao]]),
        SPLIT_TEST: np.unique(grupos[idx_holdout[rel_teste]]),
    }


def montar_saida(particoes: dict[str, np.ndarray]) -> pd.DataFrame:
    """Monta a tabela final: uma linha por paciente, colunas ``patient_id,split``."""
    linhas = [
        {"patient_id": patient_id, "split": nome}
        for nome in SPLIT_ORDER
        for patient_id in particoes[nome]
    ]
    saida = pd.DataFrame(linhas, columns=list(OUTPUT_COLUMNS))
    return saida.sort_values("patient_id", kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Validação — tudo aqui roda ANTES de qualquer escrita
# --------------------------------------------------------------------------- #


def _exigir(condicao: bool, mensagem: str) -> None:
    """Interrompe a execução se a invariante não vale."""
    if not condicao:
        raise AssertionError(mensagem)


def validar(
    frame: pd.DataFrame,
    particoes: dict[str, np.ndarray],
    saida: pd.DataFrame,
) -> list[str]:
    """Valida as invariantes estruturais; devolve o log de checagens."""
    log: list[str] = []

    def ok(rotulo: str, valor: Any, esperado: Any) -> None:
        _exigir(
            valor == esperado,
            f"INVARIANTE VIOLADA — {rotulo}: esperado {esperado}, obtido {valor}. "
            "Nenhum split foi gravado.",
        )
        log.append(f"  [OK] {rotulo:<56} {valor}")

    pacientes_entrada = set(frame["patient_id"].astype(str))
    ok("pacientes únicos de entrada", len(pacientes_entrada), EXPECTED_PATIENTS)
    ok("nódulos de entrada", len(frame), EXPECTED_NODULES)

    # Cobertura: todo paciente aparece exatamente uma vez na saída.
    ok("linhas no CSV de split (1 por paciente)", len(saida), EXPECTED_PATIENTS)
    duplicados = saida["patient_id"][saida["patient_id"].duplicated()].tolist()
    _exigir(
        not duplicados,
        f"INVARIANTE VIOLADA — patient_id duplicado no split: {duplicados[:5]}. "
        "Nenhum split foi gravado.",
    )
    log.append(f"  [OK] {'patient_id duplicado no split':<56} 0")

    faltantes = pacientes_entrada - set(saida["patient_id"].astype(str))
    sobrando = set(saida["patient_id"].astype(str)) - pacientes_entrada
    _exigir(
        not faltantes and not sobrando,
        f"INVARIANTE VIOLADA — cobertura de pacientes divergente. "
        f"Ausentes do split: {sorted(faltantes)[:5]}; "
        f"desconhecidos: {sorted(sobrando)[:5]}. Nenhum split foi gravado.",
    )
    log.append(f"  [OK] {'cobertura dos pacientes (exatamente 1x)':<56} {EXPECTED_PATIENTS}")

    # Disjunção par a par — a checagem central de leakage.
    conjuntos = {nome: set(map(str, ids)) for nome, ids in particoes.items()}
    for esquerda, direita in (
        (SPLIT_TRAIN, SPLIT_VALIDATION),
        (SPLIT_TRAIN, SPLIT_TEST),
        (SPLIT_VALIDATION, SPLIT_TEST),
    ):
        comum = conjuntos[esquerda] & conjuntos[direita]
        _exigir(
            not comum,
            f"INVARIANTE VIOLADA — leakage de paciente entre {esquerda} e "
            f"{direita}: {sorted(comum)[:5]}. Nenhum split foi gravado.",
        )
        log.append(f"  [OK] {f'interseção {esquerda}/{direita}':<56} 0")

    # Cada nódulo deve ser atribuível a exatamente uma partição após o join.
    unido = frame.merge(saida, on="patient_id", how="left", validate="many_to_one")
    ok("nódulos após o join com o split", len(unido), EXPECTED_NODULES)

    sem_particao = int(unido["split"].isna().sum())
    _exigir(
        sem_particao == 0,
        f"INVARIANTE VIOLADA — {sem_particao} nódulos sem partição após o join. "
        "Nenhum split foi gravado.",
    )
    log.append(f"  [OK] {'nódulos sem partição após o join':<56} 0")

    por_nodulo = unido.groupby("nodulo_id")["split"].nunique()
    ambiguos = por_nodulo[por_nodulo != 1].index.tolist()
    _exigir(
        not ambiguos,
        f"INVARIANTE VIOLADA — nódulos atribuídos a mais de uma partição: "
        f"{ambiguos[:5]}. Nenhum split foi gravado.",
    )
    log.append(f"  [OK] {'nódulos em exatamente 1 partição':<56} {len(por_nodulo)}")

    soma = sum(len(v) for v in conjuntos.values())
    ok("soma das partições == total de pacientes", soma, EXPECTED_PATIENTS)

    return log


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #


def relatar(frame: pd.DataFrame, saida: pd.DataFrame) -> None:
    """Imprime a composição do split por paciente, nódulo e classe."""
    unido = frame.merge(saida, on="patient_id", how="left")

    print(f"\n{SEPARATOR}\nCOMPOSIÇÃO DO SPLIT\n{SEPARATOR}")
    cabecalho = (
        f"  {'partição':<12} {'pacientes':>10} {'nódulos':>9} "
        f"{'classe0':>9} {'classe1':>9} {'% classe1':>11} {'% nódulos':>11}"
    )
    print(cabecalho)
    print(f"  {'-' * (len(cabecalho) - 2)}")

    for nome in SPLIT_ORDER:
        bloco = unido[unido["split"] == nome]
        n_pac = int(bloco["patient_id"].nunique())
        n_nod = len(bloco)
        c0 = int((bloco["target_binary"] == 0).sum())
        c1 = int((bloco["target_binary"] == 1).sum())
        prop = c1 / n_nod if n_nod else 0.0
        print(
            f"  {nome:<12} {n_pac:>10} {n_nod:>9} {c0:>9} {c1:>9} "
            f"{prop:>10.1%} {n_nod / len(unido):>10.1%}"
        )

    print(f"  {'-' * (len(cabecalho) - 2)}")
    total_c0 = int((unido["target_binary"] == 0).sum())
    total_c1 = int((unido["target_binary"] == 1).sum())
    print(
        f"  {'TOTAL':<12} {unido['patient_id'].nunique():>10} {len(unido):>9} "
        f"{total_c0:>9} {total_c1:>9} {total_c1 / len(unido):>10.1%} {1.0:>10.1%}"
    )


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #


def main() -> int:
    """Gera, valida e grava o split de pacientes."""
    try:
        frame = carregar_tabela(INPUT_CSV)
    except Exception as exc:  # noqa: BLE001 - converte falha de carga em exit code 1
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    print(f"Tabela de modelagem: {INPUT_CSV}", file=sys.stderr, flush=True)
    print(
        f"Split por paciente | GroupShuffleSplit 2 estágios | "
        f"{TRAIN_FRACTION:.0%}/{VALIDATION_FRACTION:.0%}/{TEST_FRACTION:.0%} | "
        f"random_state={RANDOM_STATE}",
        file=sys.stderr,
        flush=True,
    )

    try:
        particoes = dividir_por_paciente(frame)
    except Exception as exc:  # noqa: BLE001 - amostra degenerada não grava nada
        print(f"ERRO: falha ao gerar o split: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    saida = montar_saida(particoes)

    print(f"\n{SEPARATOR}\nVALIDAÇÃO DE INVARIANTES (antes de qualquer escrita)\n{SEPARATOR}")
    try:
        log = validar(frame, particoes, saida)
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    print("\n".join(log))
    print("\n  Vazamento de paciente entre partições: 0 (confirmado nas 3 interseções)")

    relatar(frame, saida)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    saida.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\n{SEPARATOR}")
    print(f"Split gravado em: {OUTPUT_CSV}")
    print(f"  {len(saida)} pacientes | colunas: {', '.join(OUTPUT_COLUMNS)}")
    print(SEPARATOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
