"""Seleção de nódulos (clusters de annotations) do LIDC-IDRI.

Lógica extraída, sem reprojeto, da célula 20 de
``notebooks/sprint2/PI3_G4_sprint2_v2_extração_piloto.ipynb``, que produziu os
artefatos versionados em ``reports/sprint2/``. Contexto, divergências e
validação em ``docs/sprint2/pipeline_selecao_mascaras.md``.

Depende apenas de ``numpy`` e do banco de anotações do ``pylidc``; não requer
DICOM.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

MIN_ANOTADORES_PADRAO: int = 3
DIAMETRO_MIN_MM_PADRAO: float = 3.0

MOTIVO_LEITORES_INSUFICIENTES: str = "leitores_insuficientes"
MOTIVO_DIAMETRO_ABAIXO_DO_MINIMO: str = "diametro_abaixo_do_minimo"


def nodule_id(patient_id: str, nodule_idx: int) -> str:
    """Identificador do nódulo, no formato dos artefatos do piloto.

    ``nodule_idx`` é o índice original devolvido pelo clustering, nunca uma
    reindexação posterior aos filtros.
    """
    return f"{patient_id}_N{nodule_idx:02d}"


def diametro_medio_mm(anns: Sequence[Any]) -> float:
    """Média aritmética dos diâmetros das annotations do cluster, em mm.

    Média (não mediana, não máximo), sobre TODAS as annotations do cluster,
    sem filtrar por presença de ``characteristics``.
    """
    diams = [float(a.diameter) for a in anns]
    return float(np.mean(diams))


def motivo_descarte_cluster(
    anns: Sequence[Any],
    min_anotadores: int = MIN_ANOTADORES_PADRAO,
    diametro_min_mm: float = DIAMETRO_MIN_MM_PADRAO,
) -> Optional[str]:
    """Motivo de descarte do cluster, ou ``None`` se ele é elegível.

    A ORDEM de avaliação é parte do contrato científico: número de annotations
    primeiro, diâmetro depois — um cluster que falha nos dois é descartado como
    ``leitores_insuficientes``.

    As comparações são estritamente ``<``: exatamente ``min_anotadores``
    annotations passa, e diâmetro médio exatamente ``diametro_min_mm`` passa.
    """
    if len(anns) < min_anotadores:
        return MOTIVO_LEITORES_INSUFICIENTES
    if diametro_medio_mm(anns) < diametro_min_mm:
        return MOTIVO_DIAMETRO_ABAIXO_DO_MINIMO
    return None


def selecionar_clusters(
    scan: Any,
    min_anotadores: int = MIN_ANOTADORES_PADRAO,
    diametro_min_mm: float = DIAMETRO_MIN_MM_PADRAO,
    clusters: Optional[Sequence[Sequence[Any]]] = None,
) -> Tuple[List[Tuple[int, Sequence[Any]]], List[Tuple[str, str]]]:
    """Aplica os critérios de elegibilidade a todos os clusters de um scan.

    Devolve ``(selecionados, descartes)``: ``[(nodule_idx, anns)]`` elegíveis e
    ``[(nodule_id, motivo)]`` descartados, ambos na ordem original do
    clustering. O ``nodule_idx`` é a posição do cluster nessa lista, contando
    também os descartados — nunca reindexado depois dos filtros.

    ``clusters`` permite reaproveitar um ``cluster_annotations()`` já
    executado, nunca alterar os parâmetros do clustering: quando é ``None``,
    ``scan.cluster_annotations()`` é chamado SEM argumentos.
    """
    if clusters is None:
        clusters = scan.cluster_annotations()

    patient_id = str(scan.patient_id)

    selecionados: List[Tuple[int, Sequence[Any]]] = []
    descartes: List[Tuple[str, str]] = []

    for idx, anns in enumerate(clusters):
        motivo = motivo_descarte_cluster(
            anns,
            min_anotadores=min_anotadores,
            diametro_min_mm=diametro_min_mm,
        )
        if motivo is not None:
            descartes.append((nodule_id(patient_id, idx), motivo))
            continue
        selecionados.append((idx, anns))

    return selecionados, descartes
