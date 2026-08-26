"""Consolidação de máscara de consenso e recorte da ROI.

Lógica extraída, sem reprojeto, da célula 20 de
``notebooks/sprint2/PI3_G4_sprint2_v2_extração_piloto.ipynb``, que produziu os
artefatos versionados em ``reports/sprint2/``. Contexto e validação em
``docs/sprint2/pipeline_selecao_mascaras.md``.

As funções de geometria recebem o volume ou a sua forma, nunca um ``Scan``, o
que as mantém independentes de configuração de disco. A conversão para
SimpleITK e a extração de features ficam fora deste módulo.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from pylidc.utils import consensus

#: Nível de consenso por estratégia de máscara. ``consenso50`` é a usada no
#: piloto; as demais existiam no notebook e são preservadas como estão.
CLEVEL = {
    "consenso50": 0.5,
    "uniao": 0.01,
    "intersecao": 1.0,
}

MARGEM_CONTEXTO_VOXELS_PADRAO: int = 4


def montar_mascara(
    anns: Sequence[Any],
    modo: str = "consenso50",
) -> Tuple[Optional[np.ndarray], Optional[Tuple[slice, ...]]]:
    """Consolida as annotations do cluster em uma máscara e sua bounding box.

    Para ``consenso50``, ``pylidc.utils.consensus(anns, clevel=0.5)``. O
    argumento ``pad`` do ``consensus`` NÃO é usado: a margem de contexto é
    aplicada depois, por :func:`expandir_bbox`, cuja semântica de recorte nas
    bordas do volume é a validada no piloto.

    Devolve ``(cmask, cbbox)``: máscara booleana e tupla de 3 ``slice``
    indexável diretamente no volume. No modo ``leitorN`` com ``N`` fora da
    faixa de annotations, devolve ``(None, None)``, que o chamador trata como
    máscara vazia.
    """
    if modo in CLEVEL:
        cmask, cbbox, _ = consensus(anns, clevel=CLEVEL[modo])
        return cmask, cbbox
    if modo.startswith("leitor"):
        i = int(modo.replace("leitor", ""))
        if i >= len(anns):
            return None, None
        cmask, cbbox, _ = consensus([anns[i]], clevel=0.5)
        return cmask, cbbox
    raise ValueError(f"máscara desconhecida: {modo}")


def expandir_bbox(
    cbbox: Sequence[slice],
    vol_shape: Sequence[int],
    margem: int = MARGEM_CONTEXTO_VOXELS_PADRAO,
) -> Tuple[slice, ...]:
    """Expande a caixa envolvente do nódulo antes do recorte.

    A margem dá vizinhança à interpolação B-spline da reamostragem posterior;
    sem ela, a interpolação produz overshoot numérico nas bordas do recorte.

    A expansão é recortada contra os limites do volume::

        start = max(0, start - margem)
        stop  = min(dimensão, stop + margem)

    Uma tupla NOVA é devolvida; ``cbbox`` não é modificada, porque o offset do
    recorte é calculado a partir dela.
    """
    novo = []
    for eixo, tam in zip(cbbox, vol_shape):
        ini = max(0, eixo.start - margem)
        fim = min(tam, eixo.stop + margem)
        novo.append(slice(ini, fim))
    return tuple(novo)


def recortar_roi(
    vol: np.ndarray,
    cmask: np.ndarray,
    cbbox: Sequence[slice],
    margem: int = MARGEM_CONTEXTO_VOXELS_PADRAO,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recorta a ROI de CT e realoca a máscara no mesmo referencial.

    A CT é recortada pela bbox expandida; a máscara, que cobre apenas a bbox
    original, é reposicionada pelo offset entre as duas caixas. Os eixos não
    são transpostos.

    Devolve ``(sub, mask_exp)`` com a mesma shape e o mesmo alinhamento.
    ``mask_exp`` preserva o ``dtype`` de ``cmask``, e
    ``mask_exp.sum() == cmask.sum()`` — a expansão só acrescenta voxels de
    fundo.
    """
    cbbox_exp = expandir_bbox(cbbox, vol.shape, margem)
    sub = vol[cbbox_exp]

    mask_exp = np.zeros(sub.shape, dtype=cmask.dtype)
    offset = tuple(a.start - b.start for a, b in zip(cbbox, cbbox_exp))
    slices_orig = tuple(slice(o, o + s) for o, s in zip(offset, cmask.shape))
    mask_exp[slices_orig] = cmask

    return sub, mask_exp
