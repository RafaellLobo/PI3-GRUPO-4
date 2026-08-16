"""Inspeciona, em modo somente leitura, anotações XML do LIDC-IDRI."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


CHARACTERISTIC_FIELDS = (
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
    "malignancy",
)


def normalized_tag(tag: object) -> str:
    """Retorna o nome local normalizado de uma tag, sem namespace/prefixo."""
    if not isinstance(tag, str):
        return ""
    local_name = tag.rsplit("}", maxsplit=1)[-1].rsplit(":", maxsplit=1)[-1]
    return "".join(character for character in local_name if character.isalnum()).casefold()


def find_elements(parent: ET.Element, name: str) -> list[ET.Element]:
    """Localiza descendentes pelo nome local da tag."""
    expected_name = normalized_tag(name)
    return [
        element
        for element in parent.iter()
        if normalized_tag(element.tag) == expected_name
    ]


def first_element(parent: ET.Element, name: str) -> ET.Element | None:
    """Retorna o primeiro descendente com o nome local informado."""
    expected_name = normalized_tag(name)
    return next(
        (
            element
            for element in parent.iter()
            if normalized_tag(element.tag) == expected_name
        ),
        None,
    )


def first_text(parent: ET.Element, name: str) -> str | None:
    """Retorna o primeiro texto não vazio de uma tag descendente."""
    for element in find_elements(parent, name):
        if element.text and element.text.strip():
            return element.text.strip()
    return None


def find_xml_files(data_directory: Path) -> list[Path]:
    """Localiza recursivamente arquivos com extensão .xml."""
    return sorted(
        (
            path
            for path in data_directory.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".xml"
        ),
        key=lambda path: str(path).casefold(),
    )


def print_optional(label: str, value: str | None) -> None:
    """Exibe um valor encontrado ou informa explicitamente sua ausência."""
    print(f"{label}: {value if value is not None else '<não encontrado>'}")


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    data_directory = repository_root / "data"

    try:
        xml_files = find_xml_files(data_directory)
        if not xml_files:
            raise FileNotFoundError(
                f"nenhum arquivo .xml foi encontrado em {data_directory}"
            )

        xml_path = xml_files[0]
        root = ET.parse(xml_path).getroot()

        reading_sessions = find_elements(root, "readingSession")
        unblinded_nodules = find_elements(root, "unblindedReadNodule")
        small_nodules = find_elements(root, "smallNodule")
        non_nodules = find_elements(root, "nonNodule")

        print(f"Caminho do XML: {xml_path}")
        print(f"Quantidade total de XMLs encontrados: {len(xml_files)}")
        print_optional("StudyInstanceUID", first_text(root, "StudyInstanceUID"))
        print_optional("SeriesInstanceUID", first_text(root, "SeriesInstanceUID"))
        print(f"Quantidade de readingSession: {len(reading_sessions)}")
        print(
            "Quantidade total de unblindedReadNodule: "
            f"{len(unblinded_nodules)}"
        )
        print(f"Quantidade de smallNodule: {len(small_nodules)}")
        print(f"Quantidade de nonNodule: {len(non_nodules)}")

        selected_nodule: ET.Element | None = None
        characteristics: ET.Element | None = None
        for nodule in unblinded_nodules:
            candidate = first_element(nodule, "characteristics")
            if candidate is not None:
                selected_nodule = nodule
                characteristics = candidate
                break

        if selected_nodule is None or characteristics is None:
            print(
                "Nenhum unblindedReadNodule com characteristics "
                "foi encontrado no XML analisado."
            )
            return 0

        print("\nPrimeiro unblindedReadNodule com characteristics:")
        nodule_id = first_text(selected_nodule, "noduleID")
        if nodule_id is not None:
            print(f"noduleID: {nodule_id}")

        for field in CHARACTERISTIC_FIELDS:
            value = first_text(characteristics, field)
            if value is not None:
                print(f"{field}: {value}")

        rois = find_elements(selected_nodule, "roi")
        edge_points = find_elements(selected_nodule, "edgeMap")
        print(f"Quantidade de ROI: {len(rois)}")
        print(f"Quantidade total de edgeMap/pontos: {len(edge_points)}")
        print_optional(
            "Primeiro imageZposition",
            first_text(selected_nodule, "imageZposition"),
        )
        print_optional(
            "Primeiro imageSOP_UID",
            first_text(selected_nodule, "imageSOP_UID"),
        )
    except Exception as exc:  # noqa: BLE001 - converte falhas em exit code 1
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    print("\nXML do LIDC-IDRI inspecionado com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
