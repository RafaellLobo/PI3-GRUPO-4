"""Manifesto do benchmark S6: SHA-256 de entradas, configuração, código, ambiente e saídas (P9, Seção 14.1).

``manifesto_benchmark.json`` é o ÚLTIMO artefato gravado e o único que não se
lista a si mesmo. Cada saída entra com o SHA-256 dos bytes (``sha256``) e o
normalizado CRLF → LF (``sha256_lf``), porque o ``core.autocrlf`` do Windows
muda o arquivo de texto no checkout; a verificação aceita qualquer um dos dois
(mesma regra de ``rastreabilidade.exigir_hash`` da S4).

**Elegibilidade.** ``elegivel`` é verdadeiro só no modo ``definitivo`` com dados
oficiais, árvore git limpa, aprovações válidas e as 250 unidades concluídas
(quem monta o manifesto informa e justifica). O manifesto do modo
``implementacao`` é EXPLORATÓRIO e nunca elegível.

**Git.** O manifesto registra o ``HEAD`` da execução. Depois que os agregados são
versionados, o ``HEAD`` avança; ``verificar_git`` aceita esse avanço SE a árvore
estiver limpa e só arquivos de ``reports/sprint6/benchmark`` mudaram desde o
``HEAD`` registrado (nenhum arquivo de código).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from . import adaptadores_s4 as s4
from . import config as cfg

VERSAO_ESQUEMA = 1
ARQUIVO_MANIFESTO = "manifesto_benchmark.json"
PREFIXO_ARTEFATOS = "reports/sprint6/benchmark"


class ManifestoInvalido(RuntimeError):
    """Manifesto malformado, com hash divergente ou git incompatível. Saída 4 (integridade)."""

    chave_saida = "integridade"


class SobrescritaRecusada(ManifestoInvalido):
    """Arquivo já existe com conteúdo diferente."""


# --------------------------------------------------------------------------- #
# Gravação atômica
# --------------------------------------------------------------------------- #


def gravar_atomico(caminho: Path, conteudo: bytes) -> bool:
    """Grava ``conteudo`` por temporário + ``fsync`` + ``os.replace``.

    Se o arquivo já existir com os MESMOS bytes, não faz nada (devolve ``False``);
    com bytes diferentes, recusa: um artefato não é sobrescrito. Devolve ``True`` ao gravar.
    """
    caminho = Path(caminho)
    if caminho.exists():
        if caminho.read_bytes() == conteudo:
            return False
        raise SobrescritaRecusada(f"{caminho.name} já existe com conteúdo diferente; artefatos não são sobrescritos")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    descritor, temporario = tempfile.mkstemp(prefix=f".{caminho.name}.", suffix=".tmp", dir=caminho.parent)
    try:
        with os.fdopen(descritor, "wb") as arquivo:
            arquivo.write(conteudo)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, caminho)
    finally:
        if os.path.lexists(temporario):
            os.unlink(temporario)
    return True


def bytes_json(valor: Any) -> bytes:
    """JSON canônico da S4 (chaves ordenadas, sem NaN) com ``\\n`` final."""
    return (s4.json_canonico(valor) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- #
# Montagem
# --------------------------------------------------------------------------- #


def hashes_de_saidas(destino: Path, arquivos: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """``{caminho relativo: {sha256, sha256_lf, bytes}}`` dos arquivos de ``destino`` (menos o manifesto)."""
    destino = Path(destino)
    saidas: Dict[str, Dict[str, Any]] = {}
    for relativo in sorted(set(arquivos)):
        if relativo == ARQUIVO_MANIFESTO:
            raise ManifestoInvalido("o manifesto não lista a si mesmo")
        caminho = destino / relativo
        if not caminho.is_file():
            raise ManifestoInvalido(f"saída ausente: {relativo}")
        sha, sha_lf, total = s4.calcular_sha256(caminho)
        saidas[relativo] = {"sha256": sha, "sha256_lf": sha_lf, "bytes": total}
    return saidas


def montar_manifesto(*, modo: str, oficial: bool, elegivel: bool, motivos: Sequence[str], estado_git: Any,
                     sha256_configuracao: str, entradas: Mapping[str, str], codigo: Mapping[str, str],
                     ambiente: Mapping[str, Any], saidas: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Conteúdo do manifesto. ``elegivel`` exige modo ``definitivo``, dados oficiais e git limpo."""
    if elegivel and (modo != "definitivo" or not oficial or not estado_git.limpo):
        raise ManifestoInvalido("manifesto elegível exige modo definitivo, dados oficiais e árvore git limpa")
    if not elegivel and not motivos:
        raise ManifestoInvalido("manifesto não elegível exige ao menos um motivo")
    return {
        "versao_esquema": VERSAO_ESQUEMA, "modo": modo, "oficial": bool(oficial), "elegivel": bool(elegivel),
        "exploratorio": modo != "definitivo", "motivos_de_inelegibilidade": list(motivos),
        "git": {"head": estado_git.head, "branch": estado_git.branch, "limpo": bool(estado_git.limpo)},
        "configuracao": {"sha256": sha256_configuracao}, "entradas": dict(sorted(entradas.items())),
        "codigo": dict(sorted(codigo.items())), "ambiente": json.loads(s4.json_canonico(dict(ambiente))),
        "saidas": {k: dict(v) for k, v in sorted(saidas.items())},
    }


def escrever_manifesto(destino: Path, manifesto: Mapping[str, Any]) -> Path:
    """Grava ``manifesto_benchmark.json`` (último artefato)."""
    caminho = Path(destino) / ARQUIVO_MANIFESTO
    gravar_atomico(caminho, bytes_json(dict(manifesto)))
    return caminho


# --------------------------------------------------------------------------- #
# Verificação
# --------------------------------------------------------------------------- #

_CHAVES = {"versao_esquema", "modo", "oficial", "elegivel", "exploratorio", "motivos_de_inelegibilidade", "git",
           "configuracao", "entradas", "codigo", "ambiente", "saidas"}


def ler_manifesto(destino: Path) -> Dict[str, Any]:
    caminho = Path(destino) / ARQUIVO_MANIFESTO
    if not caminho.is_file():
        raise ManifestoInvalido(f"{ARQUIVO_MANIFESTO} ausente")
    bruto = caminho.read_bytes().replace(b"\r\n", b"\n")
    try:
        manifesto = json.loads(bruto.decode("utf-8"))
    except ValueError as exc:
        raise ManifestoInvalido(f"{ARQUIVO_MANIFESTO} ilegível: {exc}") from exc
    if not isinstance(manifesto, dict) or set(manifesto) != _CHAVES or manifesto["versao_esquema"] != VERSAO_ESQUEMA:
        raise ManifestoInvalido(f"{ARQUIVO_MANIFESTO}: chaves ou versão inválidas")
    if bytes_json(manifesto) != bruto:
        raise ManifestoInvalido(f"{ARQUIVO_MANIFESTO}: bytes fora da forma canônica")
    return manifesto


def verificar_saidas(destino: Path, manifesto: Mapping[str, Any], ignorar: Sequence[str] = ("implementacao",)) -> None:
    """Cada saída listada existe e tem o SHA-256 registrado (bruto ou LF); nenhum arquivo extra.

    ``ignorar`` são subdiretórios de OUTRA execução que convivem com esta em ``reports/sprint6/benchmark``
    (os artefatos exploratórios do modo ``implementacao``, com manifesto próprio).
    """
    destino = Path(destino)
    presentes = {p.relative_to(destino).as_posix() for p in destino.rglob("*") if p.is_file()} - {ARQUIVO_MANIFESTO}
    presentes = {n for n in presentes if n.split("/", 1)[0] not in set(ignorar) or "/" not in n}
    listados = set(manifesto["saidas"])
    if presentes != listados:
        raise ManifestoInvalido(f"arquivos diferentes dos listados: faltam {sorted(listados - presentes)}, "
                                f"sobram {sorted(presentes - listados)}")
    for relativo, registro in manifesto["saidas"].items():
        sha, sha_lf, total = s4.calcular_sha256(destino / relativo)
        if registro["sha256"] != sha and registro["sha256_lf"] != sha_lf:
            raise ManifestoInvalido(f"{relativo}: SHA-256 difere do manifesto")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "--no-replace-objects", "-C", str(repo), *args], capture_output=True, check=False,
                          timeout=120)


def verificar_git(manifesto: Mapping[str, Any], repo: Path = s4.REPO_ROOT, prefixo: str = PREFIXO_ARTEFATOS) -> str:
    """Árvore limpa e, desde o ``HEAD`` do manifesto, mudanças SÓ em ``prefixo``. Devolve o ``HEAD`` atual."""
    estado = s4.estado_git(repo)
    if not estado.limpo:
        raise ManifestoInvalido("árvore git suja")
    registrado = manifesto["git"]["head"]
    if estado.head == registrado:
        return estado.head
    if _git(repo, "merge-base", "--is-ancestor", registrado, estado.head).returncode != 0:
        raise ManifestoInvalido("o HEAD do manifesto não é ancestral do HEAD atual")
    mudou = _git(repo, "diff", "--name-only", f"{registrado}..{estado.head}")
    if mudou.returncode != 0:
        raise ManifestoInvalido("falha ao comparar o HEAD do manifesto com o atual")
    fora: List[str] = [n for n in mudou.stdout.decode("utf-8").splitlines() if n and not n.startswith(prefixo + "/")]
    if fora:
        raise ManifestoInvalido(f"arquivos fora de {prefixo} mudaram depois da execução: {fora[:5]}")
    return estado.head


def verificar_ambiente_e_codigo(manifesto: Mapping[str, Any], codigo_atual: Mapping[str, str],
                                ambiente_atual: Mapping[str, Any], sha256_configuracao: Optional[str] = None) -> None:
    """Código, ambiente e configuração atuais iguais aos do manifesto."""
    if dict(manifesto["codigo"]) != dict(codigo_atual):
        diferentes = sorted(k for k in set(manifesto["codigo"]) | set(codigo_atual)
                            if manifesto["codigo"].get(k) != codigo_atual.get(k))
        raise ManifestoInvalido(f"código diferente do registrado: {diferentes[:5]}")
    if manifesto["ambiente"] != json.loads(s4.json_canonico(dict(ambiente_atual))):
        raise ManifestoInvalido("ambiente diferente do registrado")
    esperado = sha256_configuracao or cfg.sha256_configuracao()
    if manifesto["configuracao"]["sha256"] != esperado:
        raise ManifestoInvalido("configuração diferente da registrada")


__all__ = [
    "ARQUIVO_MANIFESTO", "ManifestoInvalido", "PREFIXO_ARTEFATOS", "SobrescritaRecusada", "VERSAO_ESQUEMA",
    "bytes_json", "escrever_manifesto", "gravar_atomico", "hashes_de_saidas", "ler_manifesto", "montar_manifesto",
    "verificar_ambiente_e_codigo", "verificar_git", "verificar_saidas",
]
