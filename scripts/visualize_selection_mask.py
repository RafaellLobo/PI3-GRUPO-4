"""Inspeção visual do resultado do pipeline de seleção e máscaras.

Ferramenta OPCIONAL de conferência: mostra, para ``LIDC-IDRI-0001`` / nódulo
``N00``, a ROI de CT, a máscara de consenso e a sobreposição das duas, na slice
em que a máscara tem mais voxels.

Nenhuma lógica científica é replicada: seleção, consenso e recorte vêm de
``src.radiomics``, e a carga do volume importa a função já validada em
``scripts/validate_selection_masks.py``, em vez de copiá-la.

Execução::

    python scripts/visualize_selection_mask.py
    python scripts/visualize_selection_mask.py --salvar figura.png

Por padrão nada é gravado em disco: apenas ``plt.show()``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt  # noqa: E402
import pylidc as pl  # noqa: E402

from src.radiomics import masks, selection  # noqa: E402

PACIENTE = "LIDC-IDRI-0001"
NODULE_IDX = 0
MARGEM = 4


def _carregar_helpers():
    """Importa o script de validação para reusar a carga de volume dele.

    ``scripts/`` não é um pacote, daí o carregamento pelo caminho. Importar em
    vez de copiar garante que a resolução de DICOM aqui é a mesma já
    exercitada na validação.
    """
    caminho = RAIZ / "scripts" / "validate_selection_masks.py"
    spec = importlib.util.spec_from_file_location("_validate_selection_masks", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--salvar",
        metavar="CAMINHO",
        default=None,
        help="grava a figura no caminho indicado (por padrão nada é salvo em disco)",
    )
    args = parser.parse_args()

    helpers = _carregar_helpers()

    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == PACIENTE).all()
    if not scans:
        print(f"ERRO: nenhum scan de {PACIENTE} na base do pylidc.")
        return 1
    scan = scans[0]

    selecionados, descartes = selection.selecionar_clusters(scan)
    cluster = {idx: anns for idx, anns in selecionados}.get(NODULE_IDX)
    if cluster is None:
        print(
            f"ERRO: nodule_idx {NODULE_IDX} não está entre os selecionados "
            f"{[idx for idx, _ in selecionados]}; descartes: {descartes}"
        )
        return 1

    nid = selection.nodule_id(PACIENTE, NODULE_IDX)

    cmask, cbbox = masks.montar_mascara(cluster, modo="consenso50")

    # volume carregado pela mesma estratégia da validação
    contador = helpers.Contador()
    vol = helpers.carregar_volume(scan, contador)
    if vol is None:
        print("\nDICOM indisponível localmente: nada a visualizar.")
        return 1

    sub, mask_exp = masks.recortar_roi(vol, cmask, cbbox, margem=MARGEM)
    cbbox_exp = masks.expandir_bbox(cbbox, vol.shape, margem=MARGEM)

    # slice local com mais voxels de máscara
    voxels_por_slice = mask_exp.sum(axis=(0, 1))
    z_local = int(np.argmax(voxels_por_slice))
    z_original = cbbox_exp[2].start + z_local
    voxels_na_slice = int(voxels_por_slice[z_local])

    print()
    print("=" * 68)
    print(f"Inspeção visual — {PACIENTE} / {nid}")
    print("=" * 68)
    print(f"volume original shape          : {vol.shape}")
    print(f"ROI shape                      : {sub.shape}")
    print(f"máscara shape                  : {mask_exp.shape}")
    print(f"bbox original                  : {[(s.start, s.stop) for s in cbbox]}")
    print(f"bbox expandida (margem={MARGEM})      : {[(s.start, s.stop) for s in cbbox_exp]}")
    print(f"slice local escolhida          : {z_local}")
    print(f"slice no volume original       : {z_original}")
    print(f"voxels totais da máscara       : {int(mask_exp.sum())}")
    print(f"voxels da máscara nessa slice  : {voxels_na_slice}")
    print("=" * 68)

    ct = sub[:, :, z_local]
    msk = mask_exp[:, :, z_local]

    fig, eixos = plt.subplots(1, 3, figsize=(14, 5.2))
    fig.suptitle(
        f"{PACIENTE} — {nid} | slice local {z_local} "
        f"(z={z_original} no volume) | {voxels_na_slice} voxels de máscara",
        fontsize=12,
    )

    eixos[0].imshow(ct, cmap="gray")
    eixos[0].set_title(f"A) CT ROI — {ct.shape[0]}x{ct.shape[1]}")

    eixos[1].imshow(msk, cmap="gray")
    eixos[1].set_title(f"B) Máscara consenso50 — {voxels_na_slice} voxels")

    eixos[2].imshow(ct, cmap="gray")
    eixos[2].imshow(
        np.ma.masked_where(~msk.astype(bool), msk.astype(float)),
        cmap="autumn",
        alpha=0.45,
        vmin=0.0,
        vmax=1.0,
    )
    eixos[2].set_title("C) Sobreposição")

    for eixo in eixos:
        eixo.set_xticks([])
        eixo.set_yticks([])

    fig.tight_layout()

    if args.salvar:
        fig.savefig(args.salvar, dpi=150, bbox_inches="tight")
        print(f"figura gravada em: {args.salvar}")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
