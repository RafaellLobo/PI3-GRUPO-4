"""Validação da convenção de identificação de nódulos.

Confere que os dois formatos textuais de ID do projeto continuam consistentes
com os artefatos que os carregam, que os parsers se rejeitam mutuamente e que a
identidade canônica ``(scan_id, original_nodule_idx)`` é única.

A convenção está descrita em ``docs/sprint3/convencao_identificacao_nodulos.md``.

ALCANCE DESTE VALIDADOR
-----------------------
Esta é a camada ESTRUTURAL e OFFLINE da validação: ela depende apenas da stdlib
e dos CSVs já presentes, e **não exige ``pylidc`` nem DICOM**. Nenhum arquivo é
criado ou modificado.

A unicidade direta de ``(scan_id, original_nodule_idx)`` sobre os 958 registros
mapeados NÃO é provada aqui, e sim em ``scripts/build_modeling_table.py``:
``data/base_radiomica_oficial_422.csv`` não carrega ``original_nodule_idx``, que
precisa ser recuperado do banco de anotações do ``pylidc`` scan a scan. Sobre as
958 linhas, este script prova a unicidade de ``(scan_id, posição)``; sobre as
607 linhas modeláveis, em que o índice original está persistido, prova a da
chave canônica.

Execução::

    python scripts/validate_nodule_ids.py

Sem pytest e sem dependência nova. As bases sob ``data/`` são dado derivado de
imagem médica e não versionado; quando ausentes, os blocos correspondentes
viram SKIP informativo, sem derrubar a validação.
"""

from __future__ import annotations

import csv
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.radiomics import ids, selection  # noqa: E402

#: Artefatos versionados do piloto da Sprint 2, no formato
#: ``{patient_id}_N{nodule_idx:02d}``. ``colunas`` indica se o CSV carrega
#: ``patient_id``/``nodule_idx`` separados — ``piloto_v3`` não carrega, e por
#: isso só recebe a checagem de formato.
ARTEFATOS_PILOTO = (
    ("reports/sprint2/features/piloto_piloto_v2_consenso50.csv", True, 44),
    ("reports/sprint2/features/piloto_piloto_v3_consenso50.csv", False, 44),
    ("reports/sprint2/validacao/descartes_piloto_v2_consenso50.csv", False, 30),
    ("reports/sprint2/historico/v1/piloto_piloto_v1_consenso50.csv", True, 43),
    ("reports/sprint2/historico/v1/descartes_piloto_v1_consenso50.csv", False, 31),
    (
        "reports/sprint2/historico/v3_oficial/piloto_piloto_v3_oficial_consenso50.csv",
        True,
        44,
    ),
    (
        "reports/sprint2/historico/v3_oficial/descartes_piloto_v3_oficial_consenso50.csv",
        False,
        30,
    ),
)

BASE_OFICIAL = RAIZ / "data" / "base_radiomica_oficial_422.csv"
TABELA_MODELAGEM = RAIZ / "data" / "modeling_table_sprint3.csv"

EXPECTED_LINHAS_BASE = 958
EXPECTED_SCANS_BASE = 420
EXPECTED_LARGURA_N = 2

EXPECTED_LINHAS_MODELAGEM = 607
EXPECTED_PACIENTES_MODELAGEM = 416

SEPARADOR = "=" * 72


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


def ler_csv(caminho: Path) -> List[Dict[str, str]]:
    with open(caminho, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _rejeita(funcao, valor: str) -> bool:
    """Verdadeiro se ``funcao`` recusa ``valor`` com ``ValueError``."""
    try:
        funcao(valor)
    except ValueError:
        return True
    return False


def validar_round_trip(contador: Contador) -> None:
    """Ida e volta dos dois formatos, incluindo a fronteira de largura do ``:02d``."""
    print("\n[round-trip dos formatos]")

    paciente = "LIDC-IDRI-0042"
    for indice in (0, 1, 9, 10, 99, 100):
        texto = selection.nodule_id(paciente, indice)
        contador.checar(
            ids.decompor_id_piloto(texto) == (paciente, indice),
            f"round-trip piloto falhou para idx {indice}: {texto!r}",
        )
    print("  OK    piloto: nodule_id -> decompor_id_piloto para idx 0,1,9,10,99,100")

    for posicao in (0, 1, 9, 10, 99, 100):
        for scan_id in (1, 12, 1018):
            texto = ids.formatar_id_extracao(paciente, posicao, scan_id)
            contador.checar(
                ids.decompor_id_extracao(texto) == (paciente, posicao, scan_id),
                f"round-trip extração falhou para pos {posicao}/scan {scan_id}: {texto!r}",
            )
    print("  OK    extração: formatar_id_extracao -> decompor_id_extracao (18 casos)")


def validar_rejeicao_mutua(contador: Contador) -> None:
    """Os dois parsers têm de se recusar mutuamente — o guard central da convenção.

    Ler um ID de extração como se fosse piloto (ou o contrário) associaria a
    linha ao cluster errado sem violar nenhuma invariante de contagem.
    """
    print("\n[rejeição mútua dos parsers]")

    paciente = "LIDC-IDRI-0042"
    for indice in (0, 3, 15):
        piloto = selection.nodule_id(paciente, indice)
        extracao = ids.formatar_id_extracao(paciente, indice, 12)
        contador.checar(
            _rejeita(ids.decompor_id_extracao, piloto),
            f"decompor_id_extracao aceitou um ID de piloto: {piloto!r}",
        )
        contador.checar(
            _rejeita(ids.decompor_id_piloto, extracao),
            f"decompor_id_piloto aceitou um ID de extração: {extracao!r}",
        )

    for lixo in ("", "LIDC-IDRI-0042", "LIDC-IDRI-0042_N", "0042_N00", "LIDC-IDRI-0042_N00_scan"):
        contador.checar(
            _rejeita(ids.decompor_id_piloto, lixo)
            and _rejeita(ids.decompor_id_extracao, lixo),
            f"parser aceitou entrada malformada: {lixo!r}",
        )
    print("  OK    cada parser rejeita o outro formato e entradas malformadas")


def validar_artefatos_piloto(contador: Contador) -> None:
    """Todo ``nodule_id`` versionado da Sprint 2 continua no formato do piloto."""
    print("\n[artefatos versionados da Sprint 2]")

    for caminho_rel, tem_colunas, esperado in ARTEFATOS_PILOTO:
        caminho = RAIZ / caminho_rel
        linhas = ler_csv(caminho)
        contador.checar(
            len(linhas) == esperado,
            f"{caminho_rel}: {len(linhas)} linhas, esperado {esperado}",
        )

        for linha in linhas:
            nid = linha["nodule_id"]
            patient_id, nodule_idx = ids.decompor_id_piloto(nid)
            contador.checar(
                _rejeita(ids.decompor_id_extracao, nid),
                f"{caminho_rel}: {nid!r} também casa o formato de extração",
            )
            if tem_colunas:
                contador.checar(
                    patient_id == linha["patient_id"],
                    f"{caminho_rel}: prefixo de {nid!r} != patient_id "
                    f"{linha['patient_id']!r}",
                )
                contador.checar(
                    nodule_idx == int(linha["nodule_idx"]),
                    f"{caminho_rel}: {nid!r} != nodule_idx {linha['nodule_idx']}",
                )
                contador.checar(
                    nid == selection.nodule_id(patient_id, int(linha["nodule_idx"])),
                    f"{caminho_rel}: {nid!r} não é reproduzido por selection.nodule_id",
                )

        sufixo = " + colunas patient_id/nodule_idx" if tem_colunas else ""
        print(f"  OK    {caminho_rel} ({len(linhas)} linhas): formato piloto{sufixo}")


def validar_base_oficial(contador: Contador) -> Optional[List[Dict[str, str]]]:
    """Conformidade das 958 linhas da base radiômica oficial da Sprint 3."""
    print("\n[base radiômica oficial da Sprint 3]")

    if not BASE_OFICIAL.exists():
        contador.skip(
            f"{BASE_OFICIAL.relative_to(RAIZ)} ausente (dado derivado, não versionado)"
        )
        return None

    linhas = ler_csv(BASE_OFICIAL)
    contador.checar(
        len(linhas) == EXPECTED_LINHAS_BASE,
        f"base oficial: {len(linhas)} linhas, esperado {EXPECTED_LINHAS_BASE}",
    )

    vistos = set()
    scans = set()
    pares_scan_posicao = set()
    for linha in linhas:
        nid = linha["nodulo_id"]
        patient_id, posicao, scan_id = ids.decompor_id_extracao(nid)

        # A checagem que fecha o ciclo: o formatter reproduz exatamente o ID
        # persistido, para todas as 958 linhas.
        contador.checar(
            ids.formatar_id_extracao(patient_id, posicao, scan_id) == nid,
            f"base oficial: formatar_id_extracao não reproduz {nid!r}",
        )
        contador.checar(
            patient_id == linha["patient_id"],
            f"base oficial: prefixo de {nid!r} != patient_id {linha['patient_id']!r}",
        )
        contador.checar(
            _rejeita(ids.decompor_id_piloto, nid),
            f"base oficial: {nid!r} também casa o formato do piloto",
        )
        contador.checar(nid not in vistos, f"base oficial: nodulo_id duplicado: {nid!r}")
        vistos.add(nid)
        scans.add(scan_id)

        # Unicidade de (scan_id, posição) sobre as 958: é o que dá para provar
        # aqui, offline. Como a posição indexa ``selecionados``, cujos itens
        # carregam ``original_nodule_idx`` distintos e em ordem, duas linhas com
        # pares (scan_id, posição) distintos não podem cair no mesmo cluster.
        par = (scan_id, posicao)
        contador.checar(
            par not in pares_scan_posicao,
            f"base oficial: (scan_id, posição) duplicado em {nid!r}: {par}",
        )
        pares_scan_posicao.add(par)

        largura = len(ids.PADRAO_ID_EXTRACAO.match(nid).group(2))
        contador.checar(
            largura == EXPECTED_LARGURA_N,
            f"base oficial: campo N de {nid!r} tem largura {largura}, "
            f"esperado {EXPECTED_LARGURA_N}",
        )

    contador.checar(
        len(scans) == EXPECTED_SCANS_BASE,
        f"base oficial: {len(scans)} scans distintos, esperado {EXPECTED_SCANS_BASE}",
    )
    print(
        f"  OK    {len(linhas)} linhas: formato, round-trip do formatter, "
        f"prefixo, unicidade, largura"
    )
    print(f"  OK    {len(scans)} scan_id distintos")
    print(
        f"  OK    (scan_id, posição) único em {len(pares_scan_posicao)} pares "
        f"sobre as {len(linhas)} linhas"
    )
    return linhas


def validar_chave_canonica(contador: Contador) -> None:
    """Unicidade de ``(scan_id, original_nodule_idx)`` na tabela de modelagem.

    Alcance deste bloco: as 607 linhas modeláveis, que são as únicas em que
    ``original_nodule_idx`` está persistido — a base oficial de 958 linhas não
    carrega essa coluna. A mesma unicidade sobre os 958 registros MAPEADOS é
    exigida em ``scripts/build_modeling_table.py``, onde os índices originais
    são recuperados do ``pylidc`` e existem para todas as linhas, incluindo as
    excluídas por indefinição de alvo.
    """
    print("\n[identidade canônica]")

    contador.checar(
        ids.CHAVE_CANONICA == ("scan_id", "original_nodule_idx"),
        f"CHAVE_CANONICA inesperada: {ids.CHAVE_CANONICA}",
    )

    if not TABELA_MODELAGEM.exists():
        contador.skip(
            f"{TABELA_MODELAGEM.relative_to(RAIZ)} ausente "
            "(rode: python scripts/build_modeling_table.py)"
        )
        return

    linhas = ler_csv(TABELA_MODELAGEM)
    contador.checar(
        len(linhas) == EXPECTED_LINHAS_MODELAGEM,
        f"tabela de modelagem: {len(linhas)} linhas, "
        f"esperado {EXPECTED_LINHAS_MODELAGEM}",
    )

    pacientes = {linha["patient_id"] for linha in linhas}
    contador.checar(
        len(pacientes) == EXPECTED_PACIENTES_MODELAGEM,
        f"tabela de modelagem: {len(pacientes)} pacientes, "
        f"esperado {EXPECTED_PACIENTES_MODELAGEM}",
    )

    chaves = [tuple(linha[coluna] for coluna in ids.CHAVE_CANONICA) for linha in linhas]
    contador.checar(
        len(set(chaves)) == len(chaves),
        f"chave canônica duplicada em {len(chaves) - len(set(chaves))} linha(s)",
    )
    print(f"  OK    chave canônica {ids.CHAVE_CANONICA} única em {len(chaves)} linhas")
    print(
        "  --    sobre os 958 mapeados: exigido em scripts/build_modeling_table.py "
        "(original_nodule_idx não é persistido para as linhas excluídas)"
    )

    # O ID textual é representação: a posição que ele carrega diverge do índice
    # canônico em boa parte das linhas, e isso é esperado, não um defeito.
    divergentes = sum(
        1
        for linha in linhas
        if int(linha["pos_selecionada"]) != int(linha["original_nodule_idx"])
    )
    print(
        f"  --    posição no ID != original_nodule_idx: {divergentes}/{len(linhas)} "
        f"({divergentes / len(linhas):.1%}) — divergência esperada"
    )

    # Informativo, nunca exigido: a unicidade de (patient_id, original_nodule_idx)
    # é acidente da cobertura atual, não propriedade do domínio. Exigi-la
    # proibiria justamente o caso multi-série que a chave canônica suporta.
    multi = sum(
        1
        for _, quantidade in _contar_scans_por_paciente(linhas).items()
        if quantidade > 1
    )
    print(f"  --    pacientes com mais de um scan_id: {multi} (informativo)")


def _contar_scans_por_paciente(linhas: List[Dict[str, str]]) -> Dict[str, int]:
    por_paciente: Dict[str, set] = {}
    for linha in linhas:
        por_paciente.setdefault(linha["patient_id"], set()).add(linha["scan_id"])
    return {paciente: len(scans) for paciente, scans in por_paciente.items()}


def main() -> int:
    contador = Contador()

    print(SEPARADOR)
    print("Validação da convenção de identificação de nódulos")
    print(SEPARADOR)
    print("convenção: docs/sprint3/convencao_identificacao_nodulos.md")
    print(f"formato piloto   : {ids.PADRAO_ID_PILOTO.pattern}")
    print(f"formato extração : {ids.PADRAO_ID_EXTRACAO.pattern}")
    print(f"chave canônica   : {ids.CHAVE_CANONICA}")

    try:
        validar_round_trip(contador)
        validar_rejeicao_mutua(contador)
        validar_artefatos_piloto(contador)
        validar_base_oficial(contador)
        validar_chave_canonica(contador)
    except AssertionError as erro:
        print("\n" + SEPARADOR)
        print(f"VALIDAÇÃO FALHOU após {contador.ok} checagem(ns) bem-sucedida(s)")
        print(SEPARADOR)
        print(erro)
        return 1
    except Exception:  # noqa: BLE001 - erro inesperado também reprova
        print("\n" + SEPARADOR)
        print(f"ERRO INESPERADO após {contador.ok} checagem(ns) bem-sucedida(s)")
        print(SEPARADOR)
        traceback.print_exc()
        return 1

    print("\n" + SEPARADOR)
    print(f"VALIDAÇÃO OK — {contador.ok} checagens, {len(contador.skips)} skip(s)")
    for descricao in contador.skips:
        print(f"  SKIP: {descricao}")
    print(SEPARADOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
