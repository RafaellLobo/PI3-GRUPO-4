"""Único ponto de contato do benchmark S6 com o código da Sprint 4.

Pré-registro P9 (``docs/sprint6/preregistro_p9_benchmark.md``), Seção 14.2:
o benchmark reutiliza a S4 SOMENTE por nomes públicos, todos importados aqui.
Nenhum outro módulo de ``src/benchmark_s6`` importa ``src.incremental``, e
nenhum nome com ``_`` é importado. ``src/incremental`` nunca é modificado.

O SHA-256 (CRLF → LF) dos arquivos da S4 de onde vêm esses nomes fica fixado
em ``ARQUIVOS_S4``. ``verificar_arquivos_s4`` recalcula e compara: se a S4
mudar, a verificação falha e exige revisão humana — nunca adaptação
automática.

Importar este módulo não lê dados: os módulos da S4 só definem constantes e
funções na importação.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Tuple, Union

from ..incremental.checkpoints import fingerprint_folds
from ..incremental.config import FEATURES_M1
from ..incremental.dados import (
    SPLIT_OFICIAL,
    TABELA_OFICIAL,
    ArquivoVerificado,
    fonte_oficial,
    verificar_estrutura,
)
from ..incremental.folds import Amostra, criar_amostra
from ..incremental.guardas import formatar_registro, hash_valor_decisao
from ..incremental.rastreabilidade import (
    calcular_sha256,
    estado_git,
    json_canonico,
    sha256_canonico,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Nomes públicos importados, por módulo da S4 (P9, Seção 14.2).
SIMBOLOS_S4: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "src/incremental/checkpoints.py": ("fingerprint_folds",),
    "src/incremental/config.py": ("FEATURES_M1",),
    "src/incremental/dados.py": ("SPLIT_OFICIAL", "TABELA_OFICIAL", "ArquivoVerificado",
                                 "fonte_oficial", "verificar_estrutura"),
    "src/incremental/folds.py": ("Amostra", "criar_amostra"),
    "src/incremental/guardas.py": ("formatar_registro", "hash_valor_decisao"),
    "src/incremental/rastreabilidade.py": ("calcular_sha256", "estado_git", "json_canonico",
                                           "sha256_canonico"),
})

#: SHA-256 (CRLF → LF) dos arquivos da S4 acima, no ``origin/main`` 7f3e1c2, exceto ``config.py``:
#: atualizado em 03/11/2026 para o commit 7112548 da S4 (``config.DECISOES`` com P1–P4 registradas).
ARQUIVOS_S4: Mapping[str, str] = MappingProxyType({
    "src/incremental/checkpoints.py": "6a19bb410b59b93a861980f43374eed6425923f77e584d047fdead24b3741b17",
    "src/incremental/config.py": "dbb102b465a9fb0ae7a8ca79fa133b00f46162a91e335656ab2be31cd89c92e9",
    "src/incremental/dados.py": "a12a6a9c9750bb4f5ca3648c26158c1bdc78a29dbc164d09469e68c3745782b7",
    "src/incremental/folds.py": "95ab321f90e14996329a150da9eb225482e06f858d99318fb3d4e40cbbd6cb09",
    "src/incremental/guardas.py": "34fbc1294548aa3ecd8c404fa76b6014a5ef5f84b3940884aacf05eef544ba25",
    "src/incremental/rastreabilidade.py": "884ba49689ee0f89fc81ef33efe6ae701ffe34141fc13d28c6152739632460e0",
})


class ArquivoS4Divergente(RuntimeError):
    """Um arquivo da S4 usado pelo benchmark mudou desde a fixação."""


def caminho_relativo_repo(caminho: Path) -> str:
    """Caminho POSIX relativo à raiz do repositório (para registros sem caminho pessoal)."""
    return Path(caminho).resolve().relative_to(REPO_ROOT).as_posix()


def verificar_arquivos_s4(raiz: Union[str, Path] = REPO_ROOT) -> Mapping[str, str]:
    """Recalcula o SHA-256 (LF) dos arquivos da S4 e exige igualdade com ``ARQUIVOS_S4``."""
    raiz = Path(raiz)
    divergentes = []
    for relativo, esperado in ARQUIVOS_S4.items():
        # Arquivo ausente conta como divergente aqui, sem depender da exceção da S4.
        if not (raiz / relativo).is_file() or calcular_sha256(raiz / relativo)[1] != esperado:
            divergentes.append(relativo)
    if divergentes:
        raise ArquivoS4Divergente(
            f"arquivos da S4 alterados desde a fixação: {divergentes}; revise o benchmark "
            "antes de atualizar ARQUIVOS_S4")
    return ARQUIVOS_S4


__all__ = [
    "Amostra", "ArquivoS4Divergente", "ArquivoVerificado", "ARQUIVOS_S4", "FEATURES_M1",
    "REPO_ROOT", "SIMBOLOS_S4", "SPLIT_OFICIAL", "TABELA_OFICIAL", "calcular_sha256",
    "caminho_relativo_repo", "criar_amostra", "estado_git", "fingerprint_folds",
    "fonte_oficial", "formatar_registro", "hash_valor_decisao", "json_canonico",
    "sha256_canonico", "verificar_arquivos_s4", "verificar_estrutura",
]
