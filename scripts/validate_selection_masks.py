"""Validação de equivalência de ``src/radiomics`` com o piloto da Sprint 2.

Executa ``src.radiomics.selection`` e ``src.radiomics.masks`` sobre cinco
pacientes e confere o resultado contra os artefatos versionados que o notebook
do piloto produziu (``reports/sprint2/features/`` e
``reports/sprint2/validacao/``).

Os CSVs são a referência, e não há segunda cópia da lógica do notebook aqui:
``n_anotadores``, ``diametro_medio_mm`` e ``voxels_mascara`` derivam apenas do
banco de anotações do ``pylidc``, então quase toda a validação roda sem DICOM.
O recorte da ROI é o único bloco que precisa do volume; sem imagem local ele
vira SKIP informativo, sem derrubar a validação.

Execução::

    python scripts/validate_selection_masks.py

Sem pytest e sem dependência nova. Nenhum arquivo é criado ou modificado.
"""

from __future__ import annotations

import csv
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import pylidc as pl  # noqa: E402

from src.radiomics import masks, selection  # noqa: E402

PACIENTES: Tuple[str, ...] = (
    "LIDC-IDRI-0001",
    "LIDC-IDRI-0003",
    "LIDC-IDRI-0005",
    "LIDC-IDRI-0006",
    "LIDC-IDRI-0008",
)

PACIENTE_ROI = "LIDC-IDRI-0001"

CSV_FEATURES = RAIZ / "reports/sprint2/features/piloto_piloto_v2_consenso50.csv"
CSV_DESCARTES = RAIZ / "reports/sprint2/validacao/descartes_piloto_v2_consenso50.csv"

#: Diâmetro é float derivado de contornos; comparação relativa, não bit-a-bit.
TOLERANCIA_DIAMETRO_RELATIVA = 1e-9

#: Dimensões i/j do volume LIDC-IDRI, usadas para exercitar o recorte da
#: bbox contra os limites do volume quando não há DICOM em disco.
LINHAS_COLUNAS_VOLUME = 512

REPETICOES_DETERMINISMO = 3


class Contador:
    """Acumula o resultado das checagens para o resumo final."""

    def __init__(self) -> None:
        self.ok = 0
        self.skips: List[str] = []

    def checar(self, condicao: bool, descricao: str) -> None:
        assert condicao, descricao
        self.ok += 1

    def skip(self, descricao: str) -> None:
        self.skips.append(descricao)
        print(f"  SKIP  {descricao}")


def ler_referencia_features() -> Dict[str, Dict[str, str]]:
    """Linhas do CSV de features do piloto v2, indexadas por ``nodule_id``."""
    with open(CSV_FEATURES, newline="", encoding="utf-8") as fh:
        return {linha["nodule_id"]: linha for linha in csv.DictReader(fh)}


def ler_referencia_ordenada(patient_id: str) -> List[str]:
    """``nodule_id`` extraídos do paciente, na ordem em que estão no CSV."""
    with open(CSV_FEATURES, newline="", encoding="utf-8") as fh:
        return [
            linha["nodule_id"]
            for linha in csv.DictReader(fh)
            if linha["patient_id"] == patient_id
        ]


def ler_referencia_descartes(patient_id: str) -> List[Tuple[str, str]]:
    """Pares ``(nodule_id, motivo)`` do paciente, na ordem em que estão no CSV."""
    with open(CSV_DESCARTES, newline="", encoding="utf-8") as fh:
        return [
            (linha["nodule_id"], linha["motivo"])
            for linha in csv.DictReader(fh)
            if linha["nodule_id"].rsplit("_N", 1)[0] == patient_id
        ]


def scans_do_paciente(patient_id: str) -> List[Any]:
    return pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()


def descobrir_raiz_dicom(patient_id: str) -> Optional[Path]:
    """Localiza uma raiz de DICOM local que contenha ``patient_id``.

    Devolve o diretório PAI da pasta do paciente, que é o formato de raiz que o
    ``pylidc`` espera. Um ``~/.pylidcrc`` existente tem precedência sobre esta
    busca.
    """
    for candidato in (RAIZ / "data").glob(f"**/{patient_id}"):
        if candidato.is_dir() and any(candidato.glob("**/*.dcm")):
            return candidato.parent
    return None


def carregar_volume(scan: Any, contador: Contador) -> Optional[np.ndarray]:
    """Carrega o volume do scan, ou devolve ``None`` com um SKIP explicativo.

    Com ``~/.pylidcrc``, o ``pylidc`` resolve o caminho sozinho. Sem ele, e
    havendo DICOM sob ``data/``, a resolução de caminho do ``pylidc`` é
    apontada para essa raiz APENAS durante esta validação, sem escrever nenhum
    arquivo de configuração.
    """
    tem_config = Path(os.path.expanduser("~/.pylidcrc")).exists()
    raiz_local = None if tem_config else descobrir_raiz_dicom(scan.patient_id)

    if not tem_config and raiz_local is None:
        contador.skip(
            f"ROI de {scan.patient_id}: nenhum DICOM local encontrado "
            "(sem ~/.pylidcrc e sem imagens sob data/)"
        )
        return None

    # `pylidc.Scan` (o atributo) é a classe; a resolução de caminho é uma
    # função de módulo, alcançável só por sys.modules.
    modulo_scan = sys.modules["pylidc.Scan"]
    original = modulo_scan._get_dicom_file_path_from_config_file
    if raiz_local is not None:
        modulo_scan._get_dicom_file_path_from_config_file = lambda: str(raiz_local)
    try:
        return scan.to_volume(verbose=False)
    except Exception as exc:  # noqa: BLE001 - ausência de imagem vira SKIP
        contador.skip(f"ROI de {scan.patient_id}: volume indisponível ({exc})")
        return None
    finally:
        modulo_scan._get_dicom_file_path_from_config_file = original


def validar_selecao(patient_id: str, contador: Contador) -> List[Tuple[int, Any, Any]]:
    """Confere seleção, descartes e máscara de um paciente contra os CSVs.

    Devolve ``(nodule_idx, anns, cmask)`` dos elegíveis, para o bloco de ROI
    reaproveitar sem recomputar o consenso.
    """
    print(f"\n[{patient_id}]")

    referencia = ler_referencia_features()
    ids_esperados = ler_referencia_ordenada(patient_id)
    descartes_esperados = ler_referencia_descartes(patient_id)

    scans = scans_do_paciente(patient_id)
    contador.checar(len(scans) == 1, f"{patient_id}: esperado 1 scan, obtido {len(scans)}")
    scan = scans[0]

    clusters = scan.cluster_annotations()
    selecionados, descartes = selection.selecionar_clusters(scan, clusters=clusters)

    ids_obtidos = [selection.nodule_id(patient_id, idx) for idx, _ in selecionados]

    contador.checar(
        ids_obtidos == ids_esperados,
        f"{patient_id}: nodule_id/ordem divergem\n"
        f"  esperado: {ids_esperados}\n  obtido  : {ids_obtidos}",
    )
    print(f"  OK    nodule_id e ordem ({len(ids_obtidos)} extraídos): {ids_obtidos}")

    contador.checar(
        descartes == descartes_esperados,
        f"{patient_id}: descartes divergem\n"
        f"  esperado: {descartes_esperados}\n  obtido  : {descartes}",
    )
    print(f"  OK    descartes e motivos ({len(descartes)}): {descartes}")

    indices_esperados = [int(referencia[nid]["nodule_idx"]) for nid in ids_esperados]
    indices_obtidos = [idx for idx, _ in selecionados]
    contador.checar(
        indices_obtidos == indices_esperados,
        f"{patient_id}: nodule_idx reindexado\n"
        f"  esperado: {indices_esperados}\n  obtido  : {indices_obtidos}",
    )
    print(f"  OK    nodule_idx preservado: {indices_obtidos}")

    elegiveis: List[Tuple[int, Any, Any]] = []
    n_slices = len(scan.slice_zvals)
    vol_shape = (LINHAS_COLUNAS_VOLUME, LINHAS_COLUNAS_VOLUME, n_slices)

    for idx, anns in selecionados:
        nid = selection.nodule_id(patient_id, idx)
        linha = referencia[nid]

        contador.checar(
            len(anns) == int(linha["n_anotadores"]),
            f"{nid}: n_anotadores {len(anns)} != {linha['n_anotadores']} (CSV)",
        )

        diam = selection.diametro_medio_mm(anns)
        diam_ref = float(linha["diametro_medio_mm"])
        erro_rel = abs(diam - diam_ref) / abs(diam_ref)
        contador.checar(
            erro_rel <= TOLERANCIA_DIAMETRO_RELATIVA,
            f"{nid}: diametro_medio_mm {diam!r} != {diam_ref!r} "
            f"(erro relativo {erro_rel:.3e})",
        )
        contador.checar(
            diam >= selection.DIAMETRO_MIN_MM_PADRAO,
            f"{nid}: elegível com diâmetro {diam} abaixo do mínimo",
        )

        cmask, cbbox = masks.montar_mascara(anns, modo="consenso50")

        contador.checar(cmask is not None, f"{nid}: máscara de consenso nula")
        contador.checar(
            cmask.dtype == np.bool_,
            f"{nid}: dtype da máscara é {cmask.dtype}, esperado bool",
        )
        contador.checar(cmask.sum() > 0, f"{nid}: máscara de consenso vazia")

        voxels = int(cmask.sum())
        voxels_ref = int(linha["voxels_mascara"])
        contador.checar(
            voxels == voxels_ref,
            f"{nid}: voxels_mascara {voxels} != {voxels_ref} (CSV)",
        )

        # bbox tem de ser indexável no volume e casar com a shape da máscara
        contador.checar(len(cbbox) == 3, f"{nid}: bbox com {len(cbbox)} eixos, esperado 3")
        for eixo, (sl, tam) in enumerate(zip(cbbox, vol_shape)):
            contador.checar(
                isinstance(sl, slice) and sl.step is None,
                f"{nid}: eixo {eixo} não é slice contíguo",
            )
            contador.checar(
                0 <= sl.start < sl.stop <= tam,
                f"{nid}: eixo {eixo} fora do volume: {sl} (dimensão {tam})",
            )
        contador.checar(
            cmask.shape == tuple(sl.stop - sl.start for sl in cbbox),
            f"{nid}: shape da máscara {cmask.shape} não casa com a bbox {cbbox}",
        )

        # expansão de contexto: fórmula, clipping e bbox original intacta
        cbbox_antes = tuple((sl.start, sl.stop) for sl in cbbox)
        expandida = masks.expandir_bbox(cbbox, vol_shape, margem=4)
        contador.checar(
            tuple((sl.start, sl.stop) for sl in cbbox) == cbbox_antes,
            f"{nid}: expandir_bbox modificou a bbox original",
        )
        for eixo, (orig, exp, tam) in enumerate(zip(cbbox, expandida, vol_shape)):
            contador.checar(
                exp.start == max(0, orig.start - 4) and exp.stop == min(tam, orig.stop + 4),
                f"{nid}: eixo {eixo} expandido incorretamente: {orig} -> {exp}",
            )
            contador.checar(
                0 <= exp.start <= orig.start and orig.stop <= exp.stop <= tam,
                f"{nid}: eixo {eixo} expandido fora do volume: {exp} (dimensão {tam})",
            )

        print(
            f"  OK    {nid}: n_anotadores={len(anns)} "
            f"diametro={diam:.6f}mm voxels={voxels} "
            f"bbox={[(s.start, s.stop) for s in cbbox]}"
        )
        elegiveis.append((idx, anns, cmask))

    return elegiveis


def validar_determinismo(contador: Contador) -> None:
    """Repete seleção e consenso e exige resultado idêntico (célula 31 do notebook)."""
    print(f"\n[determinismo] {REPETICOES_DETERMINISMO} repetições sobre os {len(PACIENTES)} pacientes")

    assinaturas = []
    for _ in range(REPETICOES_DETERMINISMO):
        assinatura = []
        for patient_id in PACIENTES:
            scan = scans_do_paciente(patient_id)[0]
            selecionados, descartes = selection.selecionar_clusters(scan)
            for idx, anns in selecionados:
                cmask, cbbox = masks.montar_mascara(anns, modo="consenso50")
                assinatura.append(
                    (
                        selection.nodule_id(patient_id, idx),
                        len(anns),
                        selection.diametro_medio_mm(anns),
                        int(cmask.sum()),
                        tuple((sl.start, sl.stop) for sl in cbbox),
                    )
                )
            assinatura.extend(descartes)
        assinaturas.append(assinatura)

    for n, assinatura in enumerate(assinaturas[1:], start=2):
        contador.checar(
            assinatura == assinaturas[0],
            f"repetição {n} divergiu da primeira execução",
        )
    print(f"  OK    resultado idêntico nas {REPETICOES_DETERMINISMO} repetições "
          f"({len(assinaturas[0])} registros por execução)")


def validar_roi(contador: Contador, elegiveis: List[Tuple[int, Any, Any]]) -> None:
    """Recorte de CT + máscara, quando o volume estiver disponível localmente."""
    print(f"\n[{PACIENTE_ROI}] ROI (requer DICOM local)")

    scan = scans_do_paciente(PACIENTE_ROI)[0]
    vol = carregar_volume(scan, contador)
    if vol is None:
        return

    print(f"  volume carregado: shape={vol.shape} dtype={vol.dtype}")

    for idx, anns, cmask in elegiveis:
        nid = selection.nodule_id(PACIENTE_ROI, idx)
        _, cbbox = masks.montar_mascara(anns, modo="consenso50")

        sub, mask_exp = masks.recortar_roi(vol, cmask, cbbox, margem=4)

        contador.checar(
            sub.shape == mask_exp.shape,
            f"{nid}: ROI {sub.shape} e máscara {mask_exp.shape} com shapes diferentes",
        )
        contador.checar(
            mask_exp.dtype == cmask.dtype,
            f"{nid}: dtype da máscara mudou: {cmask.dtype} -> {mask_exp.dtype}",
        )
        contador.checar(
            int(mask_exp.sum()) == int(cmask.sum()),
            f"{nid}: voxels da máscara mudaram no recorte: "
            f"{int(cmask.sum())} -> {int(mask_exp.sum())}",
        )
        contador.checar(
            sub.dtype == vol.dtype,
            f"{nid}: dtype da CT mudou: {vol.dtype} -> {sub.dtype}",
        )

        # A máscara realocada tem de cair sobre a mesma região física: os
        # valores de CT sob a máscara recortada são os mesmos que sob a
        # máscara original na bbox original.
        contador.checar(
            np.array_equal(sub[mask_exp.astype(bool)], vol[tuple(cbbox)][cmask]),
            f"{nid}: máscara realocada não cobre a mesma região da CT",
        )

        esperado = tuple(
            min(t, sl.stop + 4) - max(0, sl.start - 4)
            for sl, t in zip(cbbox, vol.shape)
        )
        contador.checar(
            sub.shape == esperado,
            f"{nid}: ROI {sub.shape} não corresponde à bbox expandida {esperado}",
        )

        print(
            f"  OK    {nid}: ROI shape={sub.shape} dtype={sub.dtype} | "
            f"máscara dtype={mask_exp.dtype} voxels={int(mask_exp.sum())}"
        )


def main() -> int:
    contador = Contador()

    print("=" * 72)
    print("Validação de src/radiomics contra os artefatos do piloto v2")
    print("=" * 72)
    print(f"referência (features) : {CSV_FEATURES.relative_to(RAIZ)}")
    print(f"referência (descartes): {CSV_DESCARTES.relative_to(RAIZ)}")
    print(f"pacientes             : {', '.join(PACIENTES)}")

    try:
        elegiveis_roi: List[Tuple[int, Any, Any]] = []
        for patient_id in PACIENTES:
            elegiveis = validar_selecao(patient_id, contador)
            if patient_id == PACIENTE_ROI:
                elegiveis_roi = elegiveis

        validar_determinismo(contador)
        validar_roi(contador, elegiveis_roi)
    except AssertionError as erro:
        print("\n" + "=" * 72)
        print(f"VALIDAÇÃO FALHOU após {contador.ok} checagem(ns) bem-sucedida(s)")
        print("=" * 72)
        print(erro)
        return 1
    except Exception:  # noqa: BLE001 - erro inesperado também reprova
        print("\n" + "=" * 72)
        print(f"ERRO INESPERADO após {contador.ok} checagem(ns) bem-sucedida(s)")
        print("=" * 72)
        traceback.print_exc()
        return 1

    print("\n" + "=" * 72)
    print(f"VALIDAÇÃO OK — {contador.ok} checagens, {len(contador.skips)} skip(s)")
    for descricao in contador.skips:
        print(f"  SKIP: {descricao}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
