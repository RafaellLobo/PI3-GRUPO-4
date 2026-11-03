"""Varredura de dados individuais, caminhos locais e vocabulário (P9, Seções 8.3, 12, 14.5 e 18-G).

Tudo o que é versionado (``reports/sprint6/benchmark``) tem de ser agregado. Este
módulo recusa, sem exceção, o que não for:

* **identificadores:** o padrão ``LIDC-IDRI-\\d{4}`` da configuração e qualquer
  ``nodulo_id`` ou ``patient_id`` da amostra, em texto, chaves ou valores;
* **dados individuais por nome:** chaves como ``nodulo_ids``, ``pontuacoes`` ou
  ``posicoes``, que só existem nos arquivos locais;
* **caminhos locais:** a pasta pessoal, a raiz do repositório e a pasta
  temporária (a P9 manda registrar caminhos relativos);
* **vocabulário vetado** (Seção 8.3), sem diferenciar maiúsculas de minúsculas,
  nos textos e nas chaves de relatório. A única exceção é ``config_benchmark.json``,
  que é a configuração canônica e TEM de listar o vocabulário vetado para a
  regra existir; os demais arquivos não têm exceção.

Falha de varredura é a saída 5 (``privacidade_ou_vocabulario``).
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from . import adaptadores_s4 as s4
from . import config as cfg

#: Arquivo que contém, por definição, o vocabulário vetado (a configuração canônica).
ARQUIVO_COM_VOCABULARIO_VETADO = "config_benchmark.json"
#: Nomes de campo que só existem em arquivos LOCAIS. Singulares ambíguos ficam de fora: pontuacao é o método
#: de pontuação de um modelo na configuração canônica (predict_proba), não um escore individual.
CHAVES_INDIVIDUAIS = frozenset({
    "nodulo_id", "nodulo_ids", "patient_id", "paciente_ids", "pontuacoes", "posicoes",
    "rotulos", "y_true", "y_score", "predicoes", "probabilidades_individuais", "linhas"})
EXTENSOES_TEXTO = frozenset({".json", ".csv", ".md", ".svg", ".txt"})


class PrivacidadeViolada(RuntimeError):
    """Dado individual, caminho local ou vocabulário vetado num artefato versionável. Saída 5."""

    chave_saida = "privacidade_ou_vocabulario"


def _padrao_id() -> "re.Pattern[str]":
    return re.compile(cfg.CONFIGURACAO.saidas.padrao_varredura)


def caminhos_locais() -> Tuple[str, ...]:
    """Prefixos que nunca podem aparecer: pasta pessoal, raiz do repositório e pasta temporária (minúsculos)."""
    textos = set()
    for caminho in (Path.home(), s4.REPO_ROOT, Path(tempfile.gettempdir())):
        try:
            resolvido = str(caminho.resolve())
        except OSError:
            continue
        textos.update({resolvido, resolvido.replace("\\", "/")})
    return tuple(sorted(t.lower() for t in textos if len(t) > 3))


def _contem_vetada(texto: str) -> Optional[str]:
    minusculo = texto.lower()
    return next((v for v in cfg.CONFIGURACAO.saidas.vocabulario_vetado if v.lower() in minusculo), None)


def varrer_texto(texto: str, onde: str, ids: Iterable[str] = (), vocabulario: bool = True) -> None:
    """Recusa ``texto`` (de um arquivo, uma chave ou um valor) com identificador, caminho local ou termo vetado."""
    if _padrao_id().search(texto):
        raise PrivacidadeViolada(f"{onde}: identificador no padrão {cfg.CONFIGURACAO.saidas.padrao_varredura}")
    for i in ids:
        if i and i in texto:
            raise PrivacidadeViolada(f"{onde}: identificador individual da amostra")
    minusculo = texto.lower()
    for prefixo in caminhos_locais():
        if prefixo in minusculo:
            raise PrivacidadeViolada(f"{onde}: caminho local")
    if vocabulario:
        vetada = _contem_vetada(texto)
        if vetada is not None:
            raise PrivacidadeViolada(f"{onde}: vocabulário vetado ({vetada!r})")


def _textos_do_json(valor: Any, caminho: str = "") -> Iterator[Tuple[str, str, bool]]:
    """``(onde, texto, é_chave)`` de todas as chaves e valores textuais."""
    if isinstance(valor, dict):
        for chave, item in valor.items():
            yield f"{caminho}/{chave}", str(chave), True
            yield from _textos_do_json(item, f"{caminho}/{chave}")
    elif isinstance(valor, (list, tuple)):
        for i, item in enumerate(valor):
            yield from _textos_do_json(item, f"{caminho}[{i}]")
    elif isinstance(valor, str):
        yield caminho, valor, False


def varrer_json(valor: Any, onde: str, ids: Iterable[str] = (), vocabulario: bool = True) -> None:
    """Varre chaves e valores de um JSON; recusa também as chaves de dados individuais."""
    ids = tuple(ids)
    for caminho, texto, e_chave in _textos_do_json(valor):
        if e_chave and texto in CHAVES_INDIVIDUAIS:
            raise PrivacidadeViolada(f"{onde}{caminho}: campo de dado individual ({texto!r})")
        varrer_texto(texto, f"{onde}{caminho}", ids, vocabulario)


def varrer_csv(texto: str, onde: str, ids: Iterable[str] = (), vocabulario: bool = True) -> None:
    cabecalho = texto.split("\n", 1)[0].split(",")
    individuais = [c for c in cabecalho if c.strip() in CHAVES_INDIVIDUAIS]
    if individuais:
        raise PrivacidadeViolada(f"{onde}: coluna de dado individual {individuais}")
    varrer_texto(texto, onde, ids, vocabulario)


def varrer_bytes(nome: str, bruto: bytes, ids: Iterable[str] = ()) -> None:
    """Varre o conteúdo de um artefato ANTES de gravá-lo: texto, JSON, CSV, SVG ou binário (PNG)."""
    caminho = Path(nome)
    vocabulario = caminho.name != ARQUIVO_COM_VOCABULARIO_VETADO
    sufixo = caminho.suffix.lower()
    if sufixo in EXTENSOES_TEXTO:
        try:
            texto = bruto.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PrivacidadeViolada(f"{nome}: não é UTF-8") from exc
        if sufixo == ".json":
            try:
                valor = json.loads(texto)
            except ValueError as exc:
                raise PrivacidadeViolada(f"{nome}: JSON inválido") from exc
            varrer_json(valor, nome, ids, vocabulario)
        elif sufixo == ".csv":
            varrer_csv(texto, nome, ids, vocabulario)
        else:
            varrer_texto(texto, nome, ids, vocabulario)
    else:
        varrer_texto(bruto.decode("latin-1"), nome, ids, vocabulario=False)


def varrer_arquivo(caminho: Path, ids: Iterable[str] = ()) -> None:
    """Varre um arquivo versionável já gravado."""
    varrer_bytes(caminho.name, caminho.read_bytes(), ids)


def varrer_diretorio(diretorio: Path, ids: Iterable[str] = ()) -> Tuple[str, ...]:
    """Varre TODOS os arquivos de ``diretorio`` (recursivo). Devolve os caminhos relativos varridos."""
    diretorio = Path(diretorio)
    if not diretorio.is_dir():
        raise PrivacidadeViolada("diretório de artefatos inexistente")
    ids = tuple(ids)
    varridos: List[str] = []
    for caminho in sorted(p for p in diretorio.rglob("*") if p.is_file()):
        varrer_arquivo(caminho, ids)
        varridos.append(caminho.relative_to(diretorio).as_posix())
    return tuple(varridos)


def identificadores_da_amostra(amostra: s4.Amostra) -> Tuple[str, ...]:
    """``nodulo_id`` e ``patient_id`` da amostra (para a varredura de saída)."""
    return tuple(sorted(set(amostra.ids.tolist()) | set(amostra.grupos.tolist())))


def resumo_de_regras() -> Dict[str, Any]:
    return {"padrao_identificador": cfg.CONFIGURACAO.saidas.padrao_varredura,
            "vocabulario_vetado": list(cfg.CONFIGURACAO.saidas.vocabulario_vetado),
            "chaves_individuais": sorted(CHAVES_INDIVIDUAIS), "excecao_de_vocabulario": ARQUIVO_COM_VOCABULARIO_VETADO}


__all__ = [
    "ARQUIVO_COM_VOCABULARIO_VETADO", "CHAVES_INDIVIDUAIS", "PrivacidadeViolada", "caminhos_locais",
    "identificadores_da_amostra", "resumo_de_regras", "varrer_arquivo", "varrer_bytes", "varrer_csv",
    "varrer_diretorio", "varrer_json", "varrer_texto",
]
