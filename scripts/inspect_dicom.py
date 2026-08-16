"""Inspeciona o primeiro arquivo DICOM encontrado no diretório data/."""

from __future__ import annotations

import sys
from pathlib import Path

import pydicom


METADATA_FIELDS = (
    "PatientID",
    "Modality",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "InstanceNumber",
    "SliceThickness",
    "PixelSpacing",
    "Rows",
    "Columns",
)


def find_dicom_files(data_directory: Path) -> list[Path]:
    """Localiza recursivamente arquivos com extensão .dcm."""
    return sorted(
        (
            path
            for path in data_directory.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".dcm"
        ),
        key=lambda path: str(path).casefold(),
    )


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    data_directory = repository_root / "data"

    try:
        dicom_files = find_dicom_files(data_directory)
        if not dicom_files:
            raise FileNotFoundError(
                f"nenhum arquivo .dcm foi encontrado em {data_directory}"
            )

        dicom_path = dicom_files[0]
        dataset = pydicom.dcmread(dicom_path)

        print(f"Caminho do arquivo: {dicom_path}")
        print(f"Quantidade total de DICOMs encontrados: {len(dicom_files)}")
        for field in METADATA_FIELDS:
            print(f"{field}: {getattr(dataset, field, '<ausente>')}")

        modality = str(getattr(dataset, "Modality", "")).strip().upper()
        if modality != "CT":
            raise ValueError(
                "modalidade inválida: esperado CT, "
                f"encontrado {modality or '<ausente>'}"
            )

        pixels = dataset.pixel_array
        print(f"pixel_array shape: {pixels.shape}")
        print(f"pixel_array dtype: {pixels.dtype}")
        print(f"pixel_array valor mínimo: {pixels.min().item()}")
        print(f"pixel_array valor máximo: {pixels.max().item()}")
    except Exception as exc:  # noqa: BLE001 - converte qualquer falha em exit code 1
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    print("DICOM validado com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
