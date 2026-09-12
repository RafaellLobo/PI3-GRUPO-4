"""Identificadores de nódulo: formatos textuais e identidade canônica.

Este módulo NÃO cria uma convenção nova. Ele dá formatter e parser explícitos
aos dois formatos textuais que o projeto já produziu, e nomeia a identidade
canônica que os scripts já usavam sem nome. Contexto, decisões e lista de
artefatos congelados em ``docs/sprint3/convencao_identificacao_nodulos.md``.

REGRA CENTRAL
-------------
Os IDs textuais são REPRESENTAÇÃO, não chave científica. A identidade de um
nódulo é ``CHAVE_CANONICA`` — ``(scan_id, original_nodule_idx)``. Todo código
novo que precise identificar um nódulo usa essa tupla; os formatos textuais
existem para ler e escrever os artefatos que já os carregam.

OS DOIS FORMATOS
----------------
``{patient_id}_N{nodule_idx:02d}``            — formato do piloto da Sprint 2.
    ``Nxx`` é o ``nodule_idx`` ORIGINAL devolvido por ``cluster_annotations()``,
    contando também os clusters descartados. Produzido por
    ``selection.nodule_id``, que está CONGELADA: é a chave dos artefatos
    versionados em ``reports/sprint2/``.

``{patient_id}_N{selected_position:02d}_scan{scan_id}`` — formato histórico da
    extração da Sprint 3, presente em ``data/base_radiomica_oficial_422.csv``.
    Aqui ``Nxx`` é a POSIÇÃO na lista filtrada ``selecionados``, não o índice do
    clustering. As duas numerações divergem em 448 das 958 linhas (46,8%).

O mesmo prefixo ``LIDC-IDRI-XXXX_Ndd`` aparece nos dois formatos com significado
diferente. Por isso os dois padrões são ancorados em ``^`` e ``$`` e os dois
parsers se rejeitam mutuamente: um ID de extração nunca é lido como piloto, e
um ID de piloto nunca é lido como extração.

ALCANCE DESSA GARANTIA
----------------------
A rejeição mútua é SINTÁTICA. Ela impede a confusão entre os dois formatos, e
nada além disso. Em particular, ela NÃO detecta um ID de extração bem formado
cujo campo ``Nxx`` tenha sido preenchido com ``original_nodule_idx`` em vez de
``selected_position``: as duas variantes produzem exatamente a mesma string, e
nenhum parser consegue separá-las olhando só para o texto.

Essa parte é semântica e se apoia no contrato explícito destas funções, na
proveniência da base de 958 linhas e na reconstrução end-to-end feita por
``scripts/build_modeling_table.py``, que recupera os índices originais do
``pylidc`` e falha se uma posição cair fora de ``selecionados``. Detalhes em
``docs/sprint3/convencao_identificacao_nodulos.md``, Seção 5.1.
"""

from __future__ import annotations

import re
from typing import Tuple

#: ``{patient_id}_N{nodule_idx:02d}`` — formato do piloto da Sprint 2.
PADRAO_ID_PILOTO: re.Pattern[str] = re.compile(r"^(LIDC-IDRI-\d{4})_N(\d+)$")

#: ``{patient_id}_N{selected_position:02d}_scan{scan_id}`` — formato histórico
#: da extração da Sprint 3.
PADRAO_ID_EXTRACAO: re.Pattern[str] = re.compile(
    r"^(LIDC-IDRI-\d{4})_N(\d+)_scan(\d+)$"
)

#: Identidade canônica do nódulo, como nomes de coluna. ``scan_id`` remove a
#: ambiguidade de pacientes com mais de uma série reconstruída;
#: ``original_nodule_idx`` é estável sob filtragem, porque não é reindexado.
CHAVE_CANONICA: Tuple[str, str] = ("scan_id", "original_nodule_idx")


def formatar_id_extracao(
    patient_id: str,
    selected_position: int,
    scan_id: int,
) -> str:
    """Monta o ID no formato histórico da extração da Sprint 3.

    ``selected_position`` é a posição na lista ``selecionados`` devolvida por
    ``selection.selecionar_clusters`` — nunca o ``nodule_idx`` do clustering.
    Trocar um pelo outro produz um ID sintaticamente válido e semanticamente
    errado.

    Existe para que o formato tenha um produtor versionado; não regenera nem
    altera nenhum artefato.
    """
    return f"{patient_id}_N{selected_position:02d}_scan{scan_id}"


def decompor_id_extracao(nodulo_id: str) -> Tuple[str, int, int]:
    """Decompõe o formato da Sprint 3 em ``(patient_id, posição, scan_id)``.

    Levanta ``ValueError`` para qualquer coisa fora do padrão — inclusive para
    um ID no formato do piloto, que não carrega scan e cujo ``Nxx`` significa
    outra coisa.
    """
    match = PADRAO_ID_EXTRACAO.match(nodulo_id)
    if match is None:
        raise ValueError(
            f"nodulo_id fora do formato de extração da Sprint 3: {nodulo_id!r}"
        )
    return match.group(1), int(match.group(2)), int(match.group(3))


def decompor_id_piloto(nodule_id: str) -> Tuple[str, int]:
    """Decompõe o formato do piloto da Sprint 2 em ``(patient_id, nodule_idx)``.

    ``nodule_idx`` é o índice ORIGINAL do clustering. Levanta ``ValueError``
    para qualquer coisa fora do padrão — inclusive para um ID da extração da
    Sprint 3, cujo ``Nxx`` é posição filtrada.
    """
    match = PADRAO_ID_PILOTO.match(nodule_id)
    if match is None:
        raise ValueError(
            f"nodule_id fora do formato do piloto da Sprint 2: {nodule_id!r}"
        )
    return match.group(1), int(match.group(2))
