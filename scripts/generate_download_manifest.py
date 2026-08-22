"""Gera o manifesto .tcia de download das séries CT da base modelável do LIDC-IDRI.

AVISO SOBRE DADOS MÉDICOS
-------------------------
Este script lê e escreve em ``scripts/output/``, que contém dados derivados do
LIDC-IDRI (identificadores de paciente e UIDs de série). Esses arquivos NÃO DEVEM
ser commitados; a pasta já é protegida por ``scripts/output/.gitignore``.

ESCOPO
------
A partir de ``cohort_diagnostic_raw.csv`` (produzido por
``scripts/explore_cohort_criteria.py``), seleciona os pacientes da "base
modelável" — aqueles com pelo menos um nódulo de ``target_binary`` definido — e
consulta a API REST pública da TCIA para descobrir quais séries CT primárias
precisam ser baixadas. O resultado é um manifesto no formato aceito pelo NBIA
Data Retriever.

O script NÃO decide qual reconstrução CT é a "correta" quando um paciente tem
mais de uma: todas são mantidas e o caso é logado. Essa escolha pertence à etapa
de segmentação, não à de download.

Idempotente: reexecutar sobrescreve o manifesto anterior, sem duplicar entradas.

Execução:
    python scripts/generate_download_manifest.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests


# --------------------------------------------------------------------------- #
# Parâmetros
# --------------------------------------------------------------------------- #

OUTPUT_DIRECTORY: Path = Path(__file__).resolve().parent / "output"
INPUT_CSV: Path = OUTPUT_DIRECTORY / "cohort_diagnostic_raw.csv"
MANIFEST_PATH: Path = OUTPUT_DIRECTORY / "lidc_manifest_422.tcia"

EXPECTED_PATIENT_COUNT: int = 422

TCIA_GET_SERIES_URL: str = (
    "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"
)
COLLECTION: str = "LIDC-IDRI"
TARGET_MODALITY: str = "CT"

RATE_LIMIT_SECONDS: float = 0.2
REQUEST_TIMEOUT_SECONDS: float = 30.0
MAX_ATTEMPTS: int = 3
BACKOFF_BASE_SECONDS: float = 1.0
PROGRESS_EVERY: int = 50

DOWNLOAD_SERVER_URL: str = (
    "https://services.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet"
)
DATABASKET_ID: str = "manifest-lidc-422-modelable.tcia"
MANIFEST_VERSION: str = "3.0"
INCLUDE_ANNOTATION: str = "true"
NUMBER_OF_RETRIES: str = "4"

BYTES_PER_GIGABYTE: int = 1024**3
SEPARATOR: str = "=" * 78


# --------------------------------------------------------------------------- #
# Seleção de pacientes
# --------------------------------------------------------------------------- #


REQUIRED_CSV_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "target_binary",
    "passes_min_annotations_3",
    "passes_diameter_3mm",
)


def load_modelable_patient_ids(csv_path: Path) -> list[str]:
    """Extrai os patient_id únicos da base modelável do diagnóstico de coorte.

    "Base modelável" é o mesmo conjunto definido em
    ``scripts/explore_cohort_criteria.py``: nódulos que passam nos DOIS critérios
    candidatos (>= 3 annotations com characteristics E diâmetro médio >= 3mm) E
    têm ``target_binary`` definido. Filtrar apenas por ``target_binary`` não nulo
    daria 637 pacientes, não 422, porque inclui nódulos de leitura única que o
    protocolo descarta.

    Levanta ``AssertionError`` se a contagem divergir de ``EXPECTED_PATIENT_COUNT``,
    para impedir que um manifesto silenciosamente desalinhado do diagnóstico seja
    gerado.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} não encontrado. Rode antes: "
            "python scripts/explore_cohort_criteria.py"
        )

    frame = pd.read_csv(csv_path)
    for column in REQUIRED_CSV_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"coluna obrigatória ausente no CSV: {column}")

    is_eligible = frame["passes_min_annotations_3"].astype(bool) & frame[
        "passes_diameter_3mm"
    ].astype(bool)
    modelable = frame[is_eligible & frame["target_binary"].notna()]
    patient_ids = sorted(modelable["patient_id"].astype(str).unique())

    assert len(patient_ids) == EXPECTED_PATIENT_COUNT, (
        f"esperados {EXPECTED_PATIENT_COUNT} patient_id com target_binary definido, "
        f"mas o CSV {csv_path} produziu {len(patient_ids)}. O diagnóstico de coorte "
        "e este manifesto estão desalinhados: regenere o CSV ou revise o critério "
        "antes de baixar qualquer imagem."
    )
    return patient_ids


# --------------------------------------------------------------------------- #
# Consulta à API da TCIA
# --------------------------------------------------------------------------- #


def fetch_patient_series(
    session: requests.Session,
    patient_id: str,
) -> list[dict[str, Any]]:
    """Consulta getSeries para um paciente, com retry e backoff exponencial.

    Levanta a última exceção após ``MAX_ATTEMPTS`` tentativas; cabe ao chamador
    decidir tolerar a falha e seguir para o próximo paciente.
    """
    parameters = {"Collection": COLLECTION, "PatientID": patient_id}
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(
                TCIA_GET_SERIES_URL,
                params=parameters,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"resposta inesperada (não é lista): {type(payload)}")
            return payload
        except Exception as exc:  # noqa: BLE001 - qualquer falha de rede é retentável
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                backoff = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                print(
                    f"  [retry] {patient_id}: tentativa {attempt}/{MAX_ATTEMPTS} "
                    f"falhou ({exc}); aguardando {backoff:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(backoff)

    raise RuntimeError(f"{MAX_ATTEMPTS} tentativas falharam: {last_error}")


def select_ct_series(series: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mantém apenas séries de modalidade CT, descartando SEG, CR, DX, SR etc."""
    return [
        entry
        for entry in series
        if str(entry.get("Modality", "")).strip().upper() == TARGET_MODALITY
    ]


def series_file_size(entry: dict[str, Any]) -> int | None:
    """Lê o campo FileSize da série em bytes, ou None se ausente/inválido."""
    raw_value = entry.get("FileSize")
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def collect_ct_series(
    patient_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, int], list[tuple[str, str]]]:
    """Percorre os pacientes e devolve (séries CT, multi-CT, falhas).

    ``multi_ct`` mapeia patient_id -> quantidade de séries CT, apenas para os
    pacientes com mais de uma. ``falhas`` acumula (patient_id, motivo) sem
    interromper a varredura.
    """
    collected: list[dict[str, Any]] = []
    multi_ct: dict[str, int] = {}
    failures: list[tuple[str, str]] = []

    with requests.Session() as session:
        for position, patient_id in enumerate(patient_ids, start=1):
            if position > 1:
                time.sleep(RATE_LIMIT_SECONDS)

            try:
                series = fetch_patient_series(session, patient_id)
            except Exception as exc:  # noqa: BLE001 - falha de paciente não aborta o script
                failures.append((patient_id, str(exc)))
                print(f"  [FALHA] {patient_id}: {exc}", file=sys.stderr, flush=True)
                continue

            ct_series = select_ct_series(series)
            if not ct_series:
                failures.append((patient_id, "nenhuma série CT retornada pela API"))
                print(
                    f"  [AVISO] {patient_id}: nenhuma série CT encontrada",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            if len(ct_series) > 1:
                multi_ct[patient_id] = len(ct_series)
                print(
                    f"  [AVISO] {patient_id}: {len(ct_series)} séries CT "
                    "(reconstruções múltiplas) — todas mantidas, escolha adiada "
                    "para a etapa de segmentação",
                    file=sys.stderr,
                    flush=True,
                )

            collected.extend(ct_series)

            if position % PROGRESS_EVERY == 0 or position == len(patient_ids):
                print(
                    f"  ... {position}/{len(patient_ids)} pacientes consultados "
                    f"({len(collected)} séries CT até aqui)",
                    file=sys.stderr,
                    flush=True,
                )

    return collected, multi_ct, failures


# --------------------------------------------------------------------------- #
# Escrita do manifesto
# --------------------------------------------------------------------------- #


def unique_series_uids(series: Sequence[dict[str, Any]]) -> list[str]:
    """Extrai SeriesInstanceUID sem duplicatas, preservando a ordem de coleta."""
    seen: set[str] = set()
    uids: list[str] = []
    for entry in series:
        uid = str(entry.get("SeriesInstanceUID", "")).strip()
        if uid and uid not in seen:
            seen.add(uid)
            uids.append(uid)
    return uids


def write_manifest(manifest_path: Path, series_uids: Sequence[str]) -> None:
    """Escreve (sobrescrevendo) o manifesto no formato do NBIA Data Retriever."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"downloadServerUrl={DOWNLOAD_SERVER_URL}",
        f"includeAnnotation={INCLUDE_ANNOTATION}",
        f"noOfrRetry={NUMBER_OF_RETRIES}",
        f"databasketId={DATABASKET_ID}",
        f"manifestVersion={MANIFEST_VERSION}",
        "ListOfSeriesToDownload=",
    ]
    content = "\n".join([*header, *series_uids]) + "\n"
    # newline="\n" evita CRLF no Windows; Path.write_text só aceita esse argumento
    # a partir do Python 3.10, e o ambiente do projeto é 3.9.
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest_file:
        manifest_file.write(content)


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #


def report(
    patient_ids: Sequence[str],
    series: Sequence[dict[str, Any]],
    series_uids: Sequence[str],
    multi_ct: dict[str, int],
    failures: Sequence[tuple[str, str]],
) -> None:
    """Imprime o resumo da geração do manifesto."""
    sizes = [series_file_size(entry) for entry in series]
    known_sizes = [size for size in sizes if size is not None]
    total_bytes = sum(known_sizes)
    missing_sizes = len(sizes) - len(known_sizes)

    print(f"\n{SEPARATOR}\nMANIFESTO DE DOWNLOAD - LIDC-IDRI (BASE MODELÁVEL)\n{SEPARATOR}")
    print(f"Manifesto salvo em: {MANIFEST_PATH}")
    print("AVISO: conteúdo derivado de dados médicos. Não commitar.\n")

    print(f"Pacientes na base modelável (entrada)           : {len(patient_ids)}")
    print(f"Pacientes consultados com sucesso               : {len(patient_ids) - len(failures)}")
    print(f"Séries CT encontradas                           : {len(series)}")
    print(f"UIDs únicos gravados no manifesto               : {len(series_uids)}")
    print(f"Pacientes com mais de 1 série CT                : {len(multi_ct)}")

    if multi_ct:
        distribution: dict[int, int] = {}
        for count in multi_ct.values():
            distribution[count] = distribution.get(count, 0) + 1
        for count in sorted(distribution):
            print(f"  {count} séries CT : {distribution[count]} paciente(s)")

    print(
        f"\nTamanho estimado total                          : "
        f"{total_bytes / BYTES_PER_GIGABYTE:.2f} GB "
        f"({total_bytes:,} bytes)"
    )
    if missing_sizes:
        print(
            f"  ATENÇÃO: {missing_sizes} série(s) sem FileSize na API; "
            "o total acima é um piso, não o valor final."
        )

    print(f"\nPacientes com falha                             : {len(failures)}")
    for patient_id, message in failures[:20]:
        print(f"  - {patient_id}: {message}")
    if len(failures) > 20:
        print(f"  ... e mais {len(failures) - 20} falha(s).")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main() -> int:
    """Seleciona a coorte, consulta a TCIA, grava o manifesto e reporta."""
    try:
        patient_ids = load_modelable_patient_ids(INPUT_CSV)
    except AssertionError as exc:
        print(f"ERRO DE VALIDAÇÃO: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - falha de entrada vira exit code 1
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    print(
        f"Consultando a TCIA para {len(patient_ids)} pacientes "
        f"(rate limit {RATE_LIMIT_SECONDS}s)...",
        file=sys.stderr,
        flush=True,
    )
    try:
        series, multi_ct, failures = collect_ct_series(patient_ids)
    except Exception as exc:  # noqa: BLE001 - converte falha global em exit code 1
        print(f"ERRO: falha ao consultar a TCIA: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    series_uids = unique_series_uids(series)
    if not series_uids:
        print(
            "ERRO: nenhuma série CT coletada; manifesto não foi gravado.",
            file=sys.stderr,
        )
        return 1

    write_manifest(MANIFEST_PATH, series_uids)
    report(patient_ids, series, series_uids, multi_ct, failures)

    print(f"\n{SEPARATOR}\nManifesto gerado.\n{SEPARATOR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
