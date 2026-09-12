"""Materializa a tabela de modelagem da Sprint 3 a partir da base radiômica oficial.

AVISO SOBRE DADOS MÉDICOS
-------------------------
Este script lê ``data/base_radiomica_oficial_422.csv`` e escreve
``data/modeling_table_sprint3.csv``. Ambos são DADOS DERIVADOS DE IMAGEM MÉDICA
(identificadores de paciente, escores de malignidade e atributos radiômicos) e
NÃO DEVEM ser commitados. A pasta ``data/`` já é protegida pelo ``.gitignore``
do repositório.

ESCOPO
------
Associa cada linha da base radiômica oficial ao seu alvo binário, recuperando as
``Annotation`` reais do ``pylidc`` e aplicando a regra oficial de consolidação
descrita em ``docs/protocolo_coorte_target_sprint2.md`` (Seções 4, 5.2 e 5.3).

O script NÃO reextrai features: os atributos radiômicos são transportados da
base de entrada sem recálculo. Não decide split, não treina modelo e não lê
DICOM — apenas o banco de anotações (SQLite) embutido no ``pylidc``.

CHAVE DE ASSOCIAÇÃO — ponto crítico
-----------------------------------
O ``nodulo_id`` da base oficial tem o formato ``{patient_id}_N{pos}_scan{id}``.
O notebook de extração da Sprint 3 persistiu esse ``N{pos}`` a partir de::

    selecionados, _ = selection.selecionar_clusters(scan, clusters=clusters)
    for idx, (_, anns) in enumerate(selecionados):

Portanto ``N{pos}`` é a **posição dentro da lista filtrada** ``selecionados``,
NÃO o ``nodule_idx`` original do clustering. As duas numerações divergem em
46,8% das linhas, porque ``selecionar_clusters`` remove clusters inelegíveis e
o ``enumerate`` reindexa o que sobrou.

Em consequência, ``scripts/output/cohort_diagnostic_raw.csv`` — que indexa por
``nodule_index`` original e não registra o scan — **não pode** ser usado como
fonte de alvo para esta base: as linhas em que as duas numerações divergem —
quase metade delas — seriam associadas ao CLUSTER INCORRETO, podendo gerar
target incorreto. Nem toda associação equivocada troca o rótulo, já que dois
clusters diferentes podem coincidir na mesma classe, mas o vínculo estaria
errado de qualquer modo. O alvo é recuperado aqui diretamente das annotations
do scan correto, identificado sem ambiguidade pelo ``scan_id`` presente no
``nodulo_id``.

O formato desse ``nodulo_id`` e seu parser vivem em ``src/radiomics/ids.py``, e
não neste script: são os mesmos usados para validar os artefatos. A identidade
canônica do nódulo é ``ids.CHAVE_CANONICA`` — ``(scan_id, original_nodule_idx)``
—, e é ela, não a string, que identifica cientificamente o nódulo. Convenção
completa em ``docs/sprint3/convencao_identificacao_nodulos.md``.

A regra de annotation válida (``has_characteristics``) e a regra de binarização
(``binarize_malignancy``) são IMPORTADAS de ``scripts/explore_cohort_criteria.py``,
para que exista uma única definição de cada uma no projeto.

Execução:
    python scripts/build_modeling_table.py
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

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SCRIPTS_DIR: Path = Path(__file__).resolve().parent
for _path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pylidc as pl  # noqa: E402 - depende do sys.path ajustado acima

from src.radiomics import ids, selection  # noqa: E402 - idem

# Única definição de annotation válida e de regra de alvo no projeto.
from explore_cohort_criteria import (  # noqa: E402 - idem
    AMBIGUOUS_MALIGNANCY_SCORE,
    MIN_ANNOTATIONS,
    binarize_malignancy,
    has_characteristics,
)


# --------------------------------------------------------------------------- #
# Caminhos e parâmetros
# --------------------------------------------------------------------------- #

INPUT_CSV: Path = REPO_ROOT / "data" / "base_radiomica_oficial_422.csv"
OUTPUT_CSV: Path = REPO_ROOT / "data" / "modeling_table_sprint3.csv"

#: Medianas fracionárias produzidas por contagem par de observadores
#: (protocolo, Seção 5.3) — causa de indefinição distinta da nota 3.
FRACTIONAL_MEDIANS: frozenset[float] = frozenset({2.5, 3.5})

MOTIVO_INCLUIDO: str = "included"
MOTIVO_CONSENSO_INDEFINIDO: str = "consensus_indeterminate"
MOTIVO_MEDIANA_FRACIONARIA: str = "fractional_median_even_raters"
MOTIVO_LEITORES_INSUFICIENTES: str = selection.MOTIVO_LEITORES_INSUFICIENTES

#: Colunas da base de entrada que são metadados de processamento, não features.
PROCESS_METADATA_COLUMNS: tuple[str, ...] = (
    "teto_adaptativo_HU",
    "n_voxels_base",
    "n_voxels_resegmentado",
)

#: Colunas de identificação da base de entrada.
INPUT_IDENTITY_COLUMNS: tuple[str, ...] = ("nodulo_id", "patient_id")

#: Features Shape do baseline — não podem conter NaN/Inf na tabela final.
BASELINE_SHAPE_FEATURES: tuple[str, ...] = (
    "original_shape_MeshVolume",
    "original_shape_Maximum3DDiameter",
    "original_shape_Sphericity",
)

# --------------------------------------------------------------------------- #
# Invariantes obrigatórias (contrato da auditoria de reconciliação)
#
# Qualquer divergência interrompe a execução ANTES da escrita: uma tabela de
# modelagem parcial é pior que nenhuma, porque seria consumida silenciosamente
# pelo baseline.
# --------------------------------------------------------------------------- #

EXPECTED_INPUT_ROWS: int = 958
EXPECTED_MAPPED: int = 958
EXPECTED_MAPPING_FAILURES: int = 0
EXPECTED_INDETERMINATE: int = 351
EXPECTED_MODELABLE_ROWS: int = 607
EXPECTED_UNIQUE_PATIENTS: int = 416
EXPECTED_CLASS_0: int = 310
EXPECTED_CLASS_1: int = 297

SEPARATOR: str = "=" * 78


# --------------------------------------------------------------------------- #
# Consolidação do alvo
# --------------------------------------------------------------------------- #


def exclusion_reason(malignancy_median: float | None, n_scored: int) -> str:
    """Motivo de inclusão/exclusão do nódulo, conforme protocolo (Seção 5.3).

    As duas causas de indefinição são mantidas separadas de propósito: indecisão
    real de consenso (mediana 3.0) e efeito estrutural de contagem par de
    observadores (2.5 / 3.5) têm naturezas diferentes e são reportadas
    separadamente na Sprint 3.
    """
    if n_scored < MIN_ANNOTATIONS:
        return MOTIVO_LEITORES_INSUFICIENTES
    if malignancy_median is None:
        return MOTIVO_LEITORES_INSUFICIENTES
    if malignancy_median == AMBIGUOUS_MALIGNANCY_SCORE:
        return MOTIVO_CONSENSO_INDEFINIDO
    if malignancy_median in FRACTIONAL_MEDIANS:
        return MOTIVO_MEDIANA_FRACIONARIA
    return MOTIVO_INCLUIDO


# --------------------------------------------------------------------------- #
# Recuperação do alvo a partir das annotations reais
# --------------------------------------------------------------------------- #


def carregar_scans_por_id() -> dict[int, Any]:
    """Indexa todos os ``pylidc.Scan`` por ``Scan.id``.

    Uma única consulta ao banco de anotações; o ``scan_id`` embutido no
    ``nodulo_id`` é a chave que remove a ambiguidade de pacientes com mais de
    uma série reconstruída.
    """
    return {scan.id: scan for scan in pl.query(pl.Scan).all()}


def resolver_scan(scan: Any) -> list[tuple[int, Sequence[Any]]]:
    """Devolve ``selecionados`` para um scan, na mesma ordem da extração.

    ``cluster_annotations()`` é chamado SEM argumentos e o resultado é repassado
    a ``selecionar_clusters``, exatamente como no notebook de extração — a
    ordem e o conteúdo da lista são o que dá sentido à posição ``N{pos}``.
    """
    clusters = scan.cluster_annotations()
    selecionados, _ = selection.selecionar_clusters(scan, clusters=clusters)
    return selecionados


def consolidar_alvo(anns: Sequence[Any]) -> dict[str, Any]:
    """Consolida as annotations de um cluster no alvo binário oficial.

    Conta apenas annotations com escore de malignidade utilizável, conforme
    ``has_characteristics``; a mediana desses escores é binarizada por
    ``binarize_malignancy``. Ambas importadas de ``explore_cohort_criteria``.
    """
    scored = [ann for ann in anns if has_characteristics(ann)]
    valores = [int(ann.malignancy) for ann in scored]
    mediana = float(median(valores)) if valores else None

    if len(scored) < MIN_ANNOTATIONS or mediana is None:
        alvo: float = float("nan")
    else:
        alvo = binarize_malignancy(mediana)

    return {
        "n_annotations_total": len(anns),
        "n_annotations_with_characteristics": len(scored),
        "malignancy_values": json.dumps(valores),
        "malignancy_median": mediana,
        "target_binary": alvo,
        "exclusion_reason": exclusion_reason(mediana, len(scored)),
    }


def mapear_linhas(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Associa cada linha radiômica ao seu alvo, via scan e posição selecionada.

    Os scans são processados em blocos: ``cluster_annotations()`` é caro e roda
    uma única vez por scan, nunca uma vez por nódulo.

    Devolve ``(registros, falhas)``. Uma falha isolada não interrompe o laço —
    ela é acumulada e checada depois, contra a invariante de 0 falhas.
    """
    scans_por_id = carregar_scans_por_id()

    por_scan: dict[int, list[tuple[int, str, str, int]]] = {}
    for posicao_linha, nodulo_id, patient_id in zip(
        frame.index, frame["nodulo_id"], frame["patient_id"]
    ):
        pid, pos, scan_id = ids.decompor_id_extracao(str(nodulo_id))
        if pid != str(patient_id):
            raise ValueError(
                f"{nodulo_id}: prefixo do nodulo_id ({pid}) diverge da coluna "
                f"patient_id ({patient_id})"
            )
        por_scan.setdefault(scan_id, []).append(
            (int(posicao_linha), str(nodulo_id), pid, pos)
        )

    registros: list[dict[str, Any]] = []
    falhas: list[tuple[str, str]] = []
    total_scans = len(por_scan)

    for contador, (scan_id, itens) in enumerate(sorted(por_scan.items()), start=1):
        scan = scans_por_id.get(scan_id)
        if scan is None:
            for _, nodulo_id, _, _ in itens:
                falhas.append((nodulo_id, f"scan_id {scan_id} inexistente no pylidc"))
            continue

        if str(scan.patient_id) != itens[0][2]:
            for _, nodulo_id, pid, _ in itens:
                falhas.append(
                    (nodulo_id, f"scan {scan_id} pertence a {scan.patient_id}, não a {pid}")
                )
            continue

        try:
            selecionados = resolver_scan(scan)
        except Exception as exc:  # noqa: BLE001 - scan problemático não para a varredura
            for _, nodulo_id, _, _ in itens:
                falhas.append((nodulo_id, f"clustering falhou: {exc}"))
            continue

        for posicao_linha, nodulo_id, pid, pos in itens:
            if pos >= len(selecionados):
                falhas.append(
                    (
                        nodulo_id,
                        f"posição {pos} fora de selecionados (len={len(selecionados)})",
                    )
                )
                continue

            original_nodule_idx, anns = selecionados[pos]
            registro: dict[str, Any] = {
                "_row": posicao_linha,
                "scan_id": scan_id,
                "pos_selecionada": pos,
                "original_nodule_idx": int(original_nodule_idx),
            }
            registro.update(consolidar_alvo(anns))
            registros.append(registro)

        if contador % 100 == 0 or contador == total_scans:
            print(
                f"  ... {contador}/{total_scans} scans resolvidos "
                f"({len(registros)} nódulos mapeados)",
                file=sys.stderr,
                flush=True,
            )

    return registros, falhas


# --------------------------------------------------------------------------- #
# Montagem da tabela
# --------------------------------------------------------------------------- #


def montar_tabela(
    frame: pd.DataFrame,
    registros: Sequence[dict[str, Any]],
) -> tuple[pd.DataFrame, list[str]]:
    """Une alvo e features, devolvendo ``(tabela_completa, feature_columns)``.

    As features são transportadas da base de entrada sem qualquer recálculo,
    reindexadas pela posição original da linha.
    """
    feature_columns = [
        column
        for column in frame.columns
        if column not in INPUT_IDENTITY_COLUMNS
        and column not in PROCESS_METADATA_COLUMNS
    ]

    alvo = pd.DataFrame(list(registros)).set_index("_row").sort_index()
    completa = frame.join(alvo, how="inner")

    ordem = [
        "nodulo_id",
        "patient_id",
        "scan_id",
        "pos_selecionada",
        "original_nodule_idx",
        "n_annotations_total",
        "n_annotations_with_characteristics",
        "malignancy_values",
        "malignancy_median",
        "target_binary",
        "exclusion_reason",
        *PROCESS_METADATA_COLUMNS,
        *feature_columns,
    ]
    return completa[ordem], feature_columns


# --------------------------------------------------------------------------- #
# Validação — tudo aqui roda ANTES de qualquer escrita
# --------------------------------------------------------------------------- #


def _exigir(condicao: bool, mensagem: str) -> None:
    """Interrompe a execução se a invariante não vale."""
    if not condicao:
        raise AssertionError(mensagem)


def validar(
    frame: pd.DataFrame,
    registros: Sequence[dict[str, Any]],
    falhas: Sequence[tuple[str, str]],
    completa: pd.DataFrame,
    modelavel: pd.DataFrame,
    feature_columns: Sequence[str],
) -> list[str]:
    """Valida todas as invariantes do contrato; devolve o log de checagens.

    Levanta ``AssertionError`` na primeira violação, de modo que nenhuma tabela
    parcial chegue ao disco.
    """
    log: list[str] = []

    def ok(rotulo: str, valor: Any, esperado: Any) -> None:
        _exigir(
            valor == esperado,
            f"INVARIANTE VIOLADA — {rotulo}: esperado {esperado}, obtido {valor}. "
            "Nenhuma tabela foi gravada.",
        )
        log.append(f"  [OK] {rotulo:<52} {valor}")

    ok("linhas de entrada", len(frame), EXPECTED_INPUT_ROWS)
    ok("mapeamentos inequívocos", len(registros), EXPECTED_MAPPED)
    ok("falhas de associação", len(falhas), EXPECTED_MAPPING_FAILURES)

    indeterminados = int(completa["target_binary"].isna().sum())
    ok("targets indeterminados", indeterminados, EXPECTED_INDETERMINATE)

    ok("linhas modeláveis", len(modelavel), EXPECTED_MODELABLE_ROWS)
    ok(
        "pacientes únicos (base modelável)",
        int(modelavel["patient_id"].nunique()),
        EXPECTED_UNIQUE_PATIENTS,
    )

    contagem = modelavel["target_binary"].value_counts()
    ok("classe 0", int(contagem.get(0.0, 0)), EXPECTED_CLASS_0)
    ok("classe 1", int(contagem.get(1.0, 0)), EXPECTED_CLASS_1)

    # Duplicidade: nodulo_id é único; patient_id repete legitimamente (um
    # paciente pode ter vários nódulos), então o que se checa é a chave física
    # do nódulo, não o paciente.
    duplicados = modelavel["nodulo_id"][modelavel["nodulo_id"].duplicated()].tolist()
    _exigir(
        not duplicados,
        f"INVARIANTE VIOLADA — nodulo_id duplicado na base modelável: "
        f"{duplicados[:5]}. Nenhuma tabela foi gravada.",
    )
    log.append(f"  [OK] {'nodulo_id duplicado':<52} 0")

    # A unicidade da chave canônica é exigida já sobre os 958 registros
    # MAPEADOS, não apenas sobre o recorte modelável: uma colisão entre uma
    # linha incluída e uma excluída continuaria sendo duas linhas radiômicas
    # apontando para o mesmo cluster, e o recorte por target a esconderia.
    chave = list(ids.CHAVE_CANONICA)
    rotulo_chave = f"({', '.join(chave)})"
    for escopo, tabela, esperado in (
        ("mapeados", completa, EXPECTED_MAPPED),
        ("modeláveis", modelavel, EXPECTED_MODELABLE_ROWS),
    ):
        ok(f"chaves canônicas {rotulo_chave} — {escopo}", len(tabela), esperado)
        dup_chave = tabela[tabela.duplicated(subset=chave, keep=False)]
        _exigir(
            dup_chave.empty,
            f"INVARIANTE VIOLADA — chave canônica {rotulo_chave} duplicada em "
            f"{escopo}: {dup_chave['nodulo_id'].tolist()[:5]}. Duas linhas "
            "radiômicas apontam para o mesmo cluster. Nenhuma tabela foi gravada.",
        )
        log.append(f"  [OK] {f'chave canônica duplicada — {escopo}':<52} 0")

    pacientes_repetidos = int(
        modelavel["patient_id"].duplicated().sum()
    )
    log.append(
        f"  [--] {'linhas de pacientes com >1 nódulo (esperado)':<52} "
        f"{pacientes_repetidos}"
    )

    for coluna in BASELINE_SHAPE_FEATURES:
        _exigir(
            coluna in modelavel.columns,
            f"INVARIANTE VIOLADA — feature do baseline ausente: {coluna}. "
            "Nenhuma tabela foi gravada.",
        )
        valores = pd.to_numeric(modelavel[coluna], errors="coerce").to_numpy(dtype=float)
        n_nan = int(np.isnan(valores).sum())
        n_inf = int(np.isinf(valores).sum())
        _exigir(
            n_nan == 0 and n_inf == 0,
            f"INVARIANTE VIOLADA — {coluna} contém {n_nan} NaN e {n_inf} Inf. "
            "Nenhuma tabela foi gravada.",
        )
        log.append(f"  [OK] {coluna + ' (NaN/Inf)':<52} 0 / 0")

    # As features precisam chegar íntegras: mesma quantidade e mesmos nomes.
    entrada = [
        c
        for c in frame.columns
        if c not in INPUT_IDENTITY_COLUMNS and c not in PROCESS_METADATA_COLUMNS
    ]
    _exigir(
        list(feature_columns) == entrada,
        "INVARIANTE VIOLADA — conjunto de features divergente da base de entrada. "
        "Nenhuma tabela foi gravada.",
    )
    log.append(f"  [OK] {'features preservadas (sem recálculo)':<52} {len(entrada)}")

    return log


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #


def main() -> int:
    """Monta, valida e grava a tabela de modelagem da Sprint 3."""
    if not INPUT_CSV.exists():
        print(f"ERRO: {INPUT_CSV} não encontrado.", file=sys.stderr)
        return 1

    print(f"Lendo base radiômica oficial: {INPUT_CSV}", file=sys.stderr, flush=True)
    frame = pd.read_csv(INPUT_CSV)

    for coluna in INPUT_IDENTITY_COLUMNS:
        if coluna not in frame.columns:
            print(f"ERRO: coluna obrigatória ausente: {coluna}", file=sys.stderr)
            return 1

    print(
        "Resolvendo clusters no banco de anotações do pylidc (sem DICOM)...",
        file=sys.stderr,
        flush=True,
    )
    try:
        registros, falhas = mapear_linhas(frame)
    except Exception as exc:  # noqa: BLE001 - converte falha global em exit code 1
        print(f"ERRO: falha ao resolver os clusters: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    completa, feature_columns = montar_tabela(frame, registros)
    modelavel = completa[completa["target_binary"].notna()].copy()
    modelavel["target_binary"] = modelavel["target_binary"].astype(int)

    print(f"\n{SEPARATOR}\nVALIDAÇÃO DE INVARIANTES (antes de qualquer escrita)\n{SEPARATOR}")
    try:
        log = validar(frame, registros, falhas, completa, modelavel, feature_columns)
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print("\n".join(log))

    motivos = completa["exclusion_reason"].value_counts()
    print(f"\n{SEPARATOR}\nCOMPOSIÇÃO\n{SEPARATOR}")
    for motivo, quantidade in motivos.items():
        print(f"  {motivo:<40} {quantidade}")
    divergentes = int(
        (completa["pos_selecionada"] != completa["original_nodule_idx"]).sum()
    )
    print(
        f"\n  linhas com N{'{pos}'} != original_nodule_idx      "
        f"{divergentes} ({divergentes / len(completa):.1%})"
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    modelavel.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\n{SEPARATOR}")
    print(f"Tabela de modelagem gravada em: {OUTPUT_CSV}")
    print(f"  {len(modelavel)} nódulos | {modelavel['patient_id'].nunique()} pacientes "
          f"| {len(feature_columns)} features")
    print("AVISO: conteúdo derivado de dados médicos. Não commitar.")
    print(SEPARATOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
