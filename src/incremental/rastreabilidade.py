"""Manifestos e rastreabilidade experimental do Cartão 4 — Sprint 4 (Incremento 2C-A).

Registra a proveniência das execuções futuras (M0–M4, CV aninhada) num
manifesto JSON com três seções, cada uma com hash próprio:

* ``execucao`` — identidade DETERMINÍSTICA da execução: atividade,
  finalidade, autorização, hash da configuração, seeds, parâmetros, estado
  inicial do git, ambiente e SHA-256 das entradas e do código. Duas execuções
  idênticas têm o mesmo ``sha256_execucao``;
* ``resultado`` — o que deve se REPRODUZIR: status, warnings agregados,
  SHA-256 das saídas e estado final do git. ``sha256_reprodutivel`` cobre
  ``execucao`` + ``resultado``;
* ``tempo`` — metadados VARIÁVEIS: início, término e duração. Só entram em
  ``sha256_manifesto``, o hash de integridade do arquivo inteiro.

Regras:

* alterações locais no git são registradas, nunca ignoradas. Uma execução
  ``DEFINITIVA`` exige árvore limpa no início, nenhuma alteração além das
  saídas registradas no fim, o mesmo HEAD, o ambiente de
  ``config.AMBIENTE_ESPERADO`` e autorização do mesmo HEAD. Uma execução
  ``EXPLORATORIA`` registra tudo e fica marcada como não elegível.
  Atividades de ``config.ATIVIDADES_SOMENTE_EXPLORATORIAS`` (o piloto
  exploratório) nunca são definitivas: ``Execucao`` recusa a finalidade e
  ``validar_manifesto`` recusa o manifesto, mesmo com hashes recalculados;
* REGISTRO AUDITÁVEL ≠ RESULTADO UTILIZÁVEL. Todo manifesto íntegro é um
  registro auditável, inclusive de execuções que falharam ou foram
  interrompidas. ``execucao.elegivel_definitiva`` diz apenas que os
  requisitos de execução definitiva foram atendidos; não diz que o resultado
  pode ser usado. Resultado definitivo utilizável é só o de execução
  definitiva, elegível e com status ``concluida``
  (``resultado_definitivo_utilizavel``). Para consumi-lo, use
  ``exigir_resultado_definitivo``, que também confere as saídas no disco;
* caminhos são relativos à raiz do repositório e todo arquivo registrado tem
  de estar dentro dela, o que evita caminhos pessoais;
* nenhum manifesto é aceito com identificador de paciente ou nódulo, caminho
  absoluto ou chave de dado individual (``_exigir_sem_dados_individuais``),
  nem na escrita nem na leitura;
* a escrita é atômica (arquivo temporário + ``os.link``) e nunca sobrescreve.
  A leitura exige bytes canônicos (só CRLF → LF é tolerado) e hashes
  coerentes;
* erros de git, arquivos ausentes, hashes divergentes e manifestos
  corrompidos levantam exceção; nada é corrigido em silêncio.

O módulo não lê dados com pandas, não importa os scripts históricos e não
executa modelos.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from . import config as cfg
from . import guardas

REPO_ROOT: Path = guardas.REPO_ROOT

FORMATO: str = "manifesto-cartao4-sprint4"
VERSAO_FORMATO: int = 1
#: Distribuições cujas versões são registradas (nomes de ``AMBIENTE_ESPERADO``).
PACOTES: Tuple[str, ...] = ("numpy", "pandas", "scikit-learn", "scipy")

STATUS_CONCLUIDA = "concluida"
STATUS_FALHOU = "falhou"
STATUS_INTERROMPIDA = "interrompida"
STATUS_VALIDOS: Tuple[str, ...] = (STATUS_CONCLUIDA, STATUS_FALHOU, STATUS_INTERROMPIDA)

_TAMANHO_BLOCO = 1 << 20
_TIMEOUT_GIT_S = 60
_MAX_MENSAGEM_WARNING = 500
_HASH_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NOME_SEED = re.compile(r"[a-z][a-z0-9_]{0,63}")
_INSTANTE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")

#: Padrões de dado individual ou de caminho pessoal.
_PADROES_PROIBIDOS = (
    (re.compile(r"LIDC-IDRI-\d{4}"), "identificador de paciente ou nódulo"),
    (re.compile(r"(^|[\s'\"(=])[A-Za-z]:[\\/]"), "caminho absoluto do Windows"),
    (re.compile(r"(^|[\s'\"(=])\\\\[^\\\s]+\\"), "caminho de rede"),
    (re.compile(r"(^|[\s'\"(=])/(home|Users|root|mnt|tmp|var|private)/"), "caminho absoluto"),
)
#: Chaves que indicariam dado individual num manifesto.
CHAVES_PROIBIDAS = frozenset({
    cfg.COLUNA_NODULO, cfg.COLUNA_PACIENTE, cfg.COLUNA_PROIBIDA, cfg.COLUNA_ALVO,
    "nodule_id", "scan_id", "y", "y_true", "y_pred", "y_score", "y_proba",
    "predicoes", "predicao", "probabilidades", "probabilidade", "scores", "linhas",
})

# --------------------------------------------------------------------------- #
# Exceções
# --------------------------------------------------------------------------- #


class ErroRastreabilidade(Exception):
    """Base de todas as falhas explícitas deste módulo."""


class ErroGit(ErroRastreabilidade):
    """Git indisponível, repositório inválido ou comando com falha."""


class ArquivoAusente(ErroRastreabilidade):
    """Arquivo registrado inexistente, fora do repositório ou não regular."""


class HashIncompativel(ErroRastreabilidade):
    """SHA-256 diferente do esperado ou alterado durante a execução."""


class ManifestoInvalido(ErroRastreabilidade):
    """Manifesto corrompido, fora do formato ou com hashes incoerentes."""


class DadosIndividuais(ManifestoInvalido):
    """Manifesto com identificador, caminho pessoal ou dado individual."""


class SobrescritaRecusada(ErroRastreabilidade):
    """O destino já existe; manifestos nunca são sobrescritos."""


class ExecucaoNaoDefinitiva(ErroRastreabilidade):
    """Requisito de execução definitiva não atendido."""


class ResultadoNaoUtilizavel(ErroRastreabilidade):
    """Manifesto íntegro, mas que não fornece resultado científico definitivo."""


class Finalidade(str, Enum):
    DEFINITIVA = "definitiva"
    EXPLORATORIA = "exploratoria"


# --------------------------------------------------------------------------- #
# JSON canônico e dados individuais
# --------------------------------------------------------------------------- #


def json_canonico(obj: Any) -> str:
    """JSON determinístico: chaves ordenadas, sem espaços, sem NaN/Infinity."""
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ManifestoInvalido(f"conteúdo não serializável de forma canônica: {exc}") from exc


def sha256_canonico(obj: Any) -> str:
    return hashlib.sha256(json_canonico(obj).encode("utf-8")).hexdigest()


def _textos_proibidos() -> Tuple[str, ...]:
    """Prefixos locais que nunca podem aparecer: home e raiz do repositório."""
    textos = set()
    for caminho in (Path.home(), REPO_ROOT, Path(tempfile.gettempdir())):
        try:
            resolvido = str(caminho.resolve())
        except OSError:
            continue
        textos.update({resolvido, resolvido.replace("\\", "/")})
    return tuple(t.lower() for t in textos if len(t) > 3)


def _exigir_sem_dados_individuais(obj: Any, caminho: str = "manifesto") -> None:
    proibidos = _textos_proibidos()

    def verificar_texto(texto: str, onde: str) -> None:
        for padrao, motivo in _PADROES_PROIBIDOS:
            if padrao.search(texto):
                raise DadosIndividuais(f"{onde}: {motivo}")
        minusculo = texto.lower()
        if any(p in minusculo for p in proibidos):
            raise DadosIndividuais(f"{onde}: caminho local")

    def visitar(valor: Any, onde: str) -> None:
        if isinstance(valor, Mapping):
            for chave, item in valor.items():
                if not isinstance(chave, str):
                    raise ManifestoInvalido(f"{onde}: chave não textual")
                if chave.lower() in CHAVES_PROIBIDAS:
                    raise DadosIndividuais(f"{onde}.{chave}: chave de dado individual")
                verificar_texto(chave, f"{onde}.{chave}")
                visitar(item, f"{onde}.{chave}")
        elif isinstance(valor, (list, tuple)):
            for i, item in enumerate(valor):
                visitar(item, f"{onde}[{i}]")
        elif isinstance(valor, str):
            verificar_texto(valor, onde)

    visitar(obj, caminho)


# --------------------------------------------------------------------------- #
# Hashes de arquivos
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HashArquivo:
    """SHA-256 bruto e com CRLF → LF de um arquivo, com caminho relativo ao repositório."""

    caminho: str
    sha256: str
    sha256_lf: str
    bytes: int

    def como_dict(self) -> dict:
        return {"caminho": self.caminho, "sha256": self.sha256, "sha256_lf": self.sha256_lf,
                "bytes": self.bytes}


def caminho_relativo(caminho: Union[str, Path], raiz: Union[str, Path] = REPO_ROOT) -> str:
    """Caminho POSIX relativo à raiz; recusa o que estiver fora dela."""
    try:
        return Path(caminho).resolve().relative_to(Path(raiz).resolve()).as_posix()
    except ValueError as exc:
        raise ArquivoAusente(f"{Path(caminho).name}: fora do repositório; não é registrado") from exc


def calcular_sha256(caminho: Union[str, Path], tamanho_bloco: int = _TAMANHO_BLOCO) -> Tuple[str, str, int]:
    """``(sha256 bruto, sha256 com CRLF → LF, bytes)``, em blocos.

    A normalização é a da reprodução auditada (``replace(b"\\r\\n", b"\\n")``
    sobre o conteúdo inteiro). Um ``\\r`` no fim de um bloco espera o próximo.
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ArquivoAusente(f"{caminho.name}: arquivo inexistente ou não regular")
    bruto, lf, total, pendente = hashlib.sha256(), hashlib.sha256(), 0, b""
    try:
        with caminho.open("rb") as arquivo:
            for bloco in iter(lambda: arquivo.read(tamanho_bloco), b""):
                bruto.update(bloco)
                total += len(bloco)
                trecho = pendente + bloco
                pendente = b"\r" if trecho.endswith(b"\r") else b""
                if pendente:
                    trecho = trecho[:-1]
                lf.update(trecho.replace(b"\r\n", b"\n"))
    except OSError as exc:
        raise ArquivoAusente(f"{caminho.name}: falha de leitura: {exc}") from exc
    lf.update(pendente)
    return bruto.hexdigest(), lf.hexdigest(), total


def hash_arquivo(caminho: Union[str, Path], raiz: Union[str, Path] = REPO_ROOT) -> HashArquivo:
    relativo = caminho_relativo(caminho, raiz)
    sha, sha_lf, total = calcular_sha256(caminho)
    return HashArquivo(relativo, sha, sha_lf, total)


def exigir_hash(caminho: Union[str, Path], sha256: str, sha256_lf: Optional[str] = None) -> str:
    """Confere o arquivo contra o hash bruto ou, se dado, o normalizado em LF.

    Devolve qual conferiu (``"bruto"`` ou ``"lf"``); senão, ``HashIncompativel``.
    """
    sha, sha_lf, _ = calcular_sha256(caminho)
    if sha == sha256:
        return "bruto"
    if sha256_lf is not None and sha_lf == sha256_lf:
        return "lf"
    raise HashIncompativel(f"{Path(caminho).name}: SHA-256 diferente do esperado")


def conferir_arquivos(registros: Iterable[Mapping[str, Any]], raiz: Union[str, Path] = REPO_ROOT) -> None:
    """Recalcula os hashes registrados num manifesto; falha se algum mudou."""
    for registro in registros:
        atual = hash_arquivo(Path(raiz) / registro["caminho"], raiz)
        if (atual.sha256, atual.bytes) != (registro["sha256"], registro["bytes"]):
            raise HashIncompativel(f"{registro['caminho']}: conteúdo diferente do registrado")


def codigo_do_pacote() -> Dict[str, Path]:
    """Os módulos de ``src/incremental``, sempre registrados como código."""
    pasta = Path(__file__).resolve().parent
    return {f"src.incremental.{p.stem}": p for p in sorted(pasta.glob("*.py"))}


# --------------------------------------------------------------------------- #
# Git
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EstadoGit:
    head: str
    branch: Optional[str]
    alteracoes: Tuple[Tuple[str, str], ...]

    @property
    def limpo(self) -> bool:
        return not self.alteracoes

    def como_dict(self) -> dict:
        return {"head": self.head, "branch": self.branch, "limpo": self.limpo,
                "alteracoes": [{"codigo": c, "caminho": p} for c, p in self.alteracoes]}


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    executavel = shutil.which("git")
    if executavel is None:
        raise ErroGit("git indisponível")
    ambiente = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    try:
        return subprocess.run(
            [executavel, "--no-replace-objects", "--no-optional-locks", "-C", str(repo), *args],
            capture_output=True, env=ambiente, timeout=_TIMEOUT_GIT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ErroGit(f"falha ao executar git: {exc}") from exc


def estado_git(repo: Union[str, Path] = REPO_ROOT) -> EstadoGit:
    """HEAD, branch (``None`` se destacado) e alterações locais, inclusive não rastreadas."""
    repo = Path(repo)
    processo = _git(repo, "rev-parse", "--show-toplevel", "HEAD^{commit}")
    linhas = processo.stdout.decode("utf-8", errors="replace").splitlines()
    if processo.returncode != 0 or len(linhas) != 2 or not _HASH_COMMIT.fullmatch(linhas[1]):
        raise ErroGit(f"não foi possível ler o HEAD de {repo.name}")
    if os.path.normcase(Path(linhas[0]).resolve()) != os.path.normcase(repo.resolve()):
        raise ErroGit(f"{repo.name} não é a raiz de um repositório git")
    head = linhas[1]

    processo = _git(repo, "symbolic-ref", "--short", "-q", "HEAD")
    if processo.returncode == 0:
        branch: Optional[str] = processo.stdout.decode("utf-8").strip()
    elif processo.returncode == 1:
        branch = None
    else:
        raise ErroGit("falha ao ler a branch")

    processo = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if processo.returncode != 0:
        raise ErroGit("falha ao ler o estado da árvore de trabalho")
    try:
        partes = processo.stdout.decode("utf-8").split("\0")
    except UnicodeDecodeError as exc:
        raise ErroGit("saída do git status não é UTF-8") from exc
    alteracoes, i = [], 0
    while i < len(partes) and partes[i]:
        entrada = partes[i]
        codigo, caminho = entrada[:2], entrada[3:]
        alteracoes.append((codigo, caminho))
        i += 2 if codigo[0] in "RC" else 1  # renomeação: o caminho de origem vem em seguida
    return EstadoGit(head, branch, tuple(sorted(alteracoes, key=lambda a: (a[1], a[0]))))


# --------------------------------------------------------------------------- #
# Ambiente
# --------------------------------------------------------------------------- #


def ambiente() -> dict:
    """Versões de Python e pacotes, comparadas com ``config.AMBIENTE_ESPERADO``.

    Usa ``importlib.metadata``: não importa os pacotes.
    """
    versoes: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for pacote in PACOTES:
        try:
            versoes[pacote] = metadata.version(pacote)
        except metadata.PackageNotFoundError:
            versoes[pacote] = None
    divergencias = {
        nome: {"esperado": esperado, "obtido": versoes.get(nome)}
        for nome, esperado in sorted(cfg.AMBIENTE_ESPERADO.items()) if versoes.get(nome) != esperado
    }
    return {
        "versoes": versoes,
        "implementacao": platform.python_implementation(),
        "plataforma": sys.platform,
        "confere_com_esperado": not divergencias,
        "divergencias": divergencias,
    }


# --------------------------------------------------------------------------- #
# Validação do manifesto
# --------------------------------------------------------------------------- #

_CHAVES_TOPO = ("formato", "versao_formato", "execucao", "resultado", "tempo",
                "sha256_execucao", "sha256_reprodutivel", "sha256_manifesto")
_CHAVES_EXECUCAO = ("atividade", "finalidade", "elegivel_definitiva", "autorizacao",
                    "hash_configuracao", "seeds", "parametros", "git", "ambiente", "entradas", "codigo")
_CHAVES_RESULTADO = ("status", "erro", "warnings", "saidas", "git_final")
_CHAVES_TEMPO = ("inicio", "termino", "duracao_s")


def _exigir_chaves(obj: Any, chaves: Tuple[str, ...], onde: str) -> None:
    if not isinstance(obj, dict) or set(obj) != set(chaves):
        raise ManifestoInvalido(f"{onde}: chaves esperadas {sorted(chaves)}")


def _exigir_registros_de_arquivo(registros: Any, onde: str) -> None:
    if not isinstance(registros, dict):
        raise ManifestoInvalido(f"{onde}: deve ser objeto")
    for rotulo, registro in registros.items():
        _exigir_chaves(registro, ("caminho", "sha256", "sha256_lf", "bytes"), f"{onde}.{rotulo}")
        if not (isinstance(registro["caminho"], str) and registro["caminho"]
                and not registro["caminho"].startswith(("/", "..")) and ":" not in registro["caminho"]):
            raise ManifestoInvalido(f"{onde}.{rotulo}: caminho deve ser relativo ao repositório")
        for chave in ("sha256", "sha256_lf"):
            if not (isinstance(registro[chave], str) and _SHA256.fullmatch(registro[chave])):
                raise ManifestoInvalido(f"{onde}.{rotulo}: {chave} inválido")
        if type(registro["bytes"]) is not int or registro["bytes"] < 0:
            raise ManifestoInvalido(f"{onde}.{rotulo}: bytes inválido")


def _com_hashes(corpo: dict) -> dict:
    manifesto = dict(corpo)
    manifesto["sha256_execucao"] = sha256_canonico(corpo["execucao"])
    manifesto["sha256_reprodutivel"] = sha256_canonico(
        {"execucao": corpo["execucao"], "resultado": corpo["resultado"]})
    manifesto["sha256_manifesto"] = sha256_canonico(manifesto)
    return manifesto


def validar_manifesto(manifesto: Any) -> dict:
    """Valida forma, hashes e ausência de dados individuais; devolve uma cópia."""
    _exigir_chaves(manifesto, _CHAVES_TOPO, "manifesto")
    m = copy.deepcopy(manifesto)
    if m["formato"] != FORMATO or m["versao_formato"] != VERSAO_FORMATO:
        raise ManifestoInvalido("formato ou versão de manifesto desconhecidos")
    _exigir_chaves(m["execucao"], _CHAVES_EXECUCAO, "execucao")
    _exigir_chaves(m["resultado"], _CHAVES_RESULTADO, "resultado")
    _exigir_chaves(m["tempo"], _CHAVES_TEMPO, "tempo")
    execucao, resultado, tempo = m["execucao"], m["resultado"], m["tempo"]
    try:
        guardas.normalizar_atividade(execucao["atividade"])
        Finalidade(execucao["finalidade"])
    except ValueError as exc:
        raise ManifestoInvalido(f"execucao: {exc}") from exc
    if not _SHA256.fullmatch(str(execucao["hash_configuracao"])):
        raise ManifestoInvalido("execucao.hash_configuracao inválido")
    definitiva = execucao["finalidade"] == Finalidade.DEFINITIVA.value
    # A autorização registrada é da mesma atividade: reetiquetar só a execução
    # (por exemplo, o piloto como Etapa 0) não passa.
    if not isinstance(execucao["autorizacao"], dict) or \
            execucao["autorizacao"].get("atividade") != execucao["atividade"]:
        raise ManifestoInvalido("execucao.autorizacao: atividade diferente da execução")
    # Antes dos hashes: um manifesto coerente, com hashes recalculados, também é recusado.
    if execucao["atividade"] in cfg.ATIVIDADES_SOMENTE_EXPLORATORIAS and \
            (definitiva or execucao["elegivel_definitiva"] is not False):
        raise ManifestoInvalido(f"execucao: {execucao['atividade']} é somente exploratória; "
                                "não pode ser definitiva nem elegível")
    if execucao["elegivel_definitiva"] is not definitiva:
        raise ManifestoInvalido("execucao.elegivel_definitiva incoerente com a finalidade")
    try:
        _validar_seeds(execucao["seeds"])
    except ValueError as exc:
        raise ManifestoInvalido(f"execucao.seeds: {exc}") from exc
    if not isinstance(execucao["parametros"], dict) or not isinstance(execucao["ambiente"], dict):
        raise ManifestoInvalido("execucao.parametros ou execucao.ambiente com tipo inválido")
    git_inicial = execucao["git"] if isinstance(execucao["git"], dict) else {}
    git_final = resultado["git_final"] if isinstance(resultado["git_final"], dict) else {}
    if definitiva and not (git_inicial.get("limpo") is True
                           and execucao["ambiente"].get("confere_com_esperado") is True
                           and git_final.get("alteracoes_alem_das_saidas") == []
                           and git_final.get("head") == git_inicial.get("head")):
        raise ManifestoInvalido("execução definitiva sem árvore limpa, ambiente esperado ou HEAD estável")
    _exigir_registros_de_arquivo(execucao["entradas"], "execucao.entradas")
    _exigir_registros_de_arquivo(execucao["codigo"], "execucao.codigo")
    _exigir_registros_de_arquivo(resultado["saidas"], "resultado.saidas")
    if resultado["status"] not in STATUS_VALIDOS:
        raise ManifestoInvalido("resultado.status desconhecido")
    if (resultado["erro"] is None) != (resultado["status"] == STATUS_CONCLUIDA) or \
            not isinstance(resultado["erro"], (str, type(None))):
        raise ManifestoInvalido("resultado.erro incoerente com o status")
    if not isinstance(resultado["warnings"], list) or not isinstance(resultado["git_final"], dict):
        raise ManifestoInvalido("resultado.warnings ou resultado.git_final com tipo inválido")
    for chave, registro_git in (("execucao.git", execucao["git"]), ("resultado.git_final", resultado["git_final"])):
        if not (isinstance(registro_git, dict) and _HASH_COMMIT.fullmatch(str(registro_git.get("head")))):
            raise ManifestoInvalido(f"{chave}: head inválido")
    for chave in ("inicio", "termino"):
        if not (isinstance(tempo[chave], str) and _INSTANTE.fullmatch(tempo[chave])):
            raise ManifestoInvalido(f"tempo.{chave} fora do formato UTC")
    if not isinstance(tempo["duracao_s"], (int, float)) or tempo["duracao_s"] < 0:
        raise ManifestoInvalido("tempo.duracao_s inválida")
    _exigir_sem_dados_individuais({k: m[k] for k in ("execucao", "resultado", "tempo")})

    corpo = {k: m[k] for k in ("formato", "versao_formato", "execucao", "resultado", "tempo")}
    esperado = _com_hashes(corpo)
    for chave in ("sha256_execucao", "sha256_reprodutivel", "sha256_manifesto"):
        if m[chave] != esperado[chave]:
            raise ManifestoInvalido(f"{chave} não confere: manifesto corrompido ou alterado")
    return m


# --------------------------------------------------------------------------- #
# Escrita atômica e leitura
# --------------------------------------------------------------------------- #


def serializar(manifesto: Mapping[str, Any]) -> bytes:
    """Bytes canônicos do arquivo: JSON indentado, chaves ordenadas, LF, UTF-8."""
    try:
        texto = json.dumps(manifesto, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ManifestoInvalido(f"manifesto não serializável: {exc}") from exc
    return (texto + "\n").encode("utf-8")


def escrever_manifesto(manifesto: Mapping[str, Any], destino: Union[str, Path],
                       raiz: Union[str, Path] = REPO_ROOT) -> HashArquivo:
    """Grava o manifesto de forma atômica, sem sobrescrever.

    O conteúdo vai para um temporário no mesmo diretório, com ``fsync``, e é
    publicado com ``os.link``, que falha se o destino existir. O diretório tem
    de existir, dentro do repositório e fora de ``data/``. Depois da gravação,
    o arquivo é relido e validado.
    """
    m = validar_manifesto(manifesto)
    destino = Path(destino)
    relativo = caminho_relativo(destino, raiz)
    if relativo == "data" or relativo.startswith("data/"):
        raise SobrescritaRecusada("manifestos não são gravados em data/")
    registrados = {r["caminho"] for secao in (m["execucao"]["entradas"], m["execucao"]["codigo"],
                                              m["resultado"]["saidas"]) for r in secao.values()}
    if relativo in registrados:
        raise SobrescritaRecusada("o destino é um arquivo registrado no próprio manifesto")
    if not destino.parent.is_dir():
        raise ArquivoAusente("diretório de destino inexistente")
    if os.path.lexists(destino):
        raise SobrescritaRecusada(f"{destino.name} já existe; manifestos não são sobrescritos")

    conteudo = serializar(m)
    descritor, temporario = tempfile.mkstemp(prefix=f".{destino.name}.", suffix=".tmp", dir=destino.parent)
    try:
        with os.fdopen(descritor, "wb") as arquivo:
            arquivo.write(conteudo)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        try:
            os.link(temporario, destino)
        except FileExistsError as exc:
            raise SobrescritaRecusada(f"{destino.name} já existe; manifestos não são sobrescritos") from exc
        except OSError as exc:
            raise ErroRastreabilidade(f"publicação atômica indisponível: {exc}") from exc
    finally:
        try:
            os.unlink(temporario)
        except FileNotFoundError:
            pass
    lido = ler_manifesto(destino)
    if lido != m:
        raise ManifestoInvalido("manifesto relido difere do gravado")
    return hash_arquivo(destino, raiz)


def resultado_definitivo_utilizavel(manifesto: Mapping[str, Any]) -> bool:
    """Se o manifesto (validado por inteiro) fornece resultado definitivo utilizável.

    Exige finalidade ``definitiva``, ``elegivel_definitiva`` e status
    ``concluida``. Execuções com ``falhou`` ou ``interrompida`` continuam
    registros auditáveis, mas nunca resultados utilizáveis.
    """
    m = validar_manifesto(manifesto)
    return (m["execucao"]["finalidade"] == Finalidade.DEFINITIVA.value
            and m["execucao"]["elegivel_definitiva"] is True
            and m["resultado"]["status"] == STATUS_CONCLUIDA)


def exigir_resultado_definitivo(caminho: Union[str, Path], raiz: Union[str, Path] = REPO_ROOT) -> dict:
    """Única porta para consumir resultados definitivos.

    Lê e valida o manifesto, exige ``resultado_definitivo_utilizavel`` e
    confere que as saídas registradas não mudaram desde a execução.

    Os hashes detectam corrupção e edição incoerente, mesmo com hashes
    recalculados. Uma falsificação inteiramente coerente só é detectável com
    âncora externa (por exemplo, o manifesto versionado num commit).
    """
    m = ler_manifesto(caminho)
    if not resultado_definitivo_utilizavel(m):
        raise ResultadoNaoUtilizavel(
            f"execução {m['execucao']['finalidade']} com status {m['resultado']['status']}: "
            "registro auditável, não resultado definitivo utilizável")
    conferir_arquivos(m["resultado"]["saidas"].values(), raiz)
    return m


def _sem_duplicatas(pares: List[Tuple[str, Any]]) -> dict:
    chaves = [k for k, _ in pares]
    if len(chaves) != len(set(chaves)):
        raise ManifestoInvalido("chave JSON duplicada")
    return dict(pares)


def _constante_invalida(nome: str) -> Any:
    raise ManifestoInvalido(f"constante JSON não permitida: {nome}")


def ler_manifesto(caminho: Union[str, Path]) -> dict:
    """Lê e valida um manifesto. Recusa JSON inválido, chaves duplicadas,
    NaN/Infinity, bytes não canônicos, hashes incoerentes e dados individuais."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ArquivoAusente(f"{caminho.name}: manifesto inexistente")
    try:
        bruto = caminho.read_bytes()
    except OSError as exc:
        raise ArquivoAusente(f"{caminho.name}: falha de leitura: {exc}") from exc
    normalizado = bruto.replace(b"\r\n", b"\n")
    try:
        manifesto = json.loads(normalizado.decode("utf-8"), object_pairs_hook=_sem_duplicatas,
                               parse_constant=_constante_invalida)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestoInvalido(f"{caminho.name}: JSON inválido: {exc}") from exc
    m = validar_manifesto(manifesto)
    if serializar(m) != normalizado:
        raise ManifestoInvalido(f"{caminho.name}: bytes fora da forma canônica")
    return m


# --------------------------------------------------------------------------- #
# Execução rastreada
# --------------------------------------------------------------------------- #


def _instante(momento: _dt.datetime) -> str:
    if momento.tzinfo is None:
        raise ErroRastreabilidade("instante sem fuso horário")
    return momento.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _validar_seeds(seeds: Mapping[str, int]) -> dict:
    if not isinstance(seeds, Mapping):
        raise ValueError("seeds deve ser um mapeamento nome → inteiro")
    for nome, valor in seeds.items():
        if not (isinstance(nome, str) and _NOME_SEED.fullmatch(nome)):
            raise ValueError(f"nome de seed inválido: {nome!r}")
        if type(valor) is not int or not 0 <= valor < 2 ** 32:
            raise ValueError(f"seed {nome} deve ser inteiro em [0, 2^32)")
    return dict(seeds)


def _validar_parametros(parametros: Mapping[str, Any]) -> dict:
    if not isinstance(parametros, Mapping):
        raise ValueError("parametros deve ser um mapeamento")
    copia = json.loads(json_canonico(dict(parametros)))
    _exigir_sem_dados_individuais(copia, "parametros")
    return copia


_CAMINHO_ABSOLUTO_EM_TEXTO = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|root|mnt|tmp|var|private|usr|opt)/)[^\s'\"<>|]*")


def _sanear_mensagem(texto: str) -> str:
    """Troca caminhos absolutos por ``<caminho>`` em mensagens de warning.

    Identificadores de paciente ou nódulo NÃO são trocados: eles fazem a
    validação do manifesto falhar.
    """
    return _CAMINHO_ABSOLUTO_EM_TEXTO.sub("<caminho>", texto)[:_MAX_MENSAGEM_WARNING]


class Execucao:
    """Contexto que registra uma execução e produz seu manifesto.

    ``with Execucao(...) as ex: ...``; ao sair, ``ex.manifesto`` contém o
    manifesto validado (também se o bloco falhar, e a exceção segue
    propagando). Para gravar: ``escrever_manifesto(ex.manifesto, destino)``.

    Na entrada, a autorização é reemitida por ``guardas.exigir_autorizacao`` e
    tem de ser idêntica à informada. Execuções definitivas são recusadas já na
    entrada se o git não estiver limpo, se o ambiente divergir ou se a
    autorização não for do HEAD atual. Código (módulos do pacote e
    ``codigo``) é registrado relativo à raiz do projeto; entradas e saídas,
    relativas a ``repo``.
    """

    def __init__(self, atividade: guardas.Atividade, finalidade: Finalidade,
                 autorizacao: guardas.Autorizacao, entradas: Mapping[str, Union[str, Path]],
                 seeds: Mapping[str, int], parametros: Optional[Mapping[str, Any]] = None,
                 codigo: Optional[Mapping[str, Union[str, Path]]] = None,
                 decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
                 repo: Union[str, Path] = REPO_ROOT,
                 relogio_utc: Callable[[], _dt.datetime] = lambda: _dt.datetime.now(_dt.timezone.utc),
                 relogio_monotonico: Callable[[], float] = time.monotonic) -> None:
        self.atividade = guardas.normalizar_atividade(atividade)
        self.finalidade = Finalidade(finalidade)
        if self.finalidade is Finalidade.DEFINITIVA and self.atividade in cfg.ATIVIDADES_SOMENTE_EXPLORATORIAS:
            raise ExecucaoNaoDefinitiva(f"{self.atividade}: atividade somente exploratória; "
                                        "execução definitiva não é permitida")
        if not isinstance(autorizacao, guardas.Autorizacao):
            raise TypeError("autorizacao deve ser guardas.Autorizacao")
        if autorizacao.atividade != self.atividade:
            raise ErroRastreabilidade("a autorização é de outra atividade")
        self.hash_configuracao = cfg.hash_configuracao(decisoes)
        if autorizacao.hash_configuracao != self.hash_configuracao:
            raise ErroRastreabilidade("a autorização foi emitida para outra configuração")
        self.autorizacao = autorizacao
        self._decisoes = decisoes
        self.seeds = _validar_seeds(seeds)
        self.parametros = _validar_parametros(parametros or {})
        self.repo = Path(repo)
        self._entradas = self._caminhos(entradas, "entradas")
        # Código é sempre do repositório do projeto; entradas e saídas, de ``repo``.
        self._codigo = self._caminhos({**codigo_do_pacote(), **(codigo or {})}, "codigo")
        self._relogio_utc, self._relogio_monotonico = relogio_utc, relogio_monotonico
        self._saidas: Dict[str, Path] = {}
        self._estado = "criada"
        self.manifesto: Optional[dict] = None

    @staticmethod
    def _caminhos(arquivos: Mapping[str, Union[str, Path]], onde: str) -> Dict[str, Path]:
        if not isinstance(arquivos, Mapping):
            raise ValueError(f"{onde} deve ser um mapeamento rótulo → caminho")
        for rotulo in arquivos:
            if not (isinstance(rotulo, str) and rotulo):
                raise ValueError(f"{onde}: rótulo inválido {rotulo!r}")
        return {rotulo: Path(caminho) for rotulo, caminho in arquivos.items()}

    def _hashes(self, arquivos: Mapping[str, Path], raiz: Optional[Path] = None) -> Dict[str, dict]:
        raiz = self.repo if raiz is None else raiz
        return {rotulo: hash_arquivo(caminho, raiz).como_dict()
                for rotulo, caminho in sorted(arquivos.items())}

    def registrar_saida(self, rotulo: str, caminho: Union[str, Path]) -> None:
        """Declara uma saída; o hash é calculado no fim, quando ela já existe."""
        if self._estado != "em_execucao":
            raise ErroRastreabilidade("saídas só podem ser registradas durante a execução")
        if not (isinstance(rotulo, str) and rotulo) or rotulo in self._saidas:
            raise ValueError(f"rótulo de saída inválido ou repetido: {rotulo!r}")
        caminho = Path(caminho)
        caminho_relativo(caminho, self.repo)
        self._saidas[rotulo] = caminho

    def __enter__(self) -> "Execucao":
        if self._estado != "criada":
            raise ErroRastreabilidade("cada Execucao é usada uma única vez")
        # A autorização registrada tem de ser a que as guardas emitem agora:
        # um objeto montado à mão não entra no manifesto.
        if guardas.exigir_autorizacao(self.atividade, self._decisoes, self.repo) != self.autorizacao:
            raise ErroRastreabilidade("a autorização informada não confere com a emitida pelas guardas")
        self._git_inicial = estado_git(self.repo)
        self._ambiente = ambiente()
        self._hash_entradas = self._hashes(self._entradas)
        self._hash_codigo = self._hashes(self._codigo, REPO_ROOT)
        if self.finalidade is Finalidade.DEFINITIVA:
            problemas = []
            if not self._git_inicial.limpo:
                problemas.append("árvore de trabalho com alterações locais")
            if not self._ambiente["confere_com_esperado"]:
                problemas.append(f"ambiente divergente: {sorted(self._ambiente['divergencias'])}")
            if self.autorizacao.head is not None and self.autorizacao.head != self._git_inicial.head:
                problemas.append("autorização emitida para outro HEAD")
            if problemas:
                raise ExecucaoNaoDefinitiva("; ".join(problemas))
        self._inicio = self._relogio_utc()
        self._t0 = self._relogio_monotonico()
        self._captura = warnings.catch_warnings(record=True)
        self._avisos = self._captura.__enter__()
        warnings.simplefilter("always")
        self._estado = "em_execucao"
        return self

    def __exit__(self, tipo, excecao, _rastro) -> bool:
        duracao = self._relogio_monotonico() - self._t0
        termino = self._relogio_utc()
        self._captura.__exit__(None, None, None)
        self._estado = "finalizada"
        if tipo is None:
            status, erro = STATUS_CONCLUIDA, None
        elif issubclass(tipo, (KeyboardInterrupt, SystemExit)):
            status, erro = STATUS_INTERROMPIDA, tipo.__name__
        else:
            status, erro = STATUS_FALHOU, tipo.__name__
        self.manifesto = self._finalizar(status, erro, termino, duracao)
        return False

    def _warnings_agregados(self) -> List[dict]:
        contagem: Dict[Tuple[str, str, str, int], int] = {}
        for aviso in self._avisos:
            chave = (aviso.category.__name__, _sanear_mensagem(str(aviso.message)),
                     Path(aviso.filename).name, int(aviso.lineno))
            contagem[chave] = contagem.get(chave, 0) + 1
        return [{"categoria": c, "mensagem": m, "arquivo": a, "linha": l, "ocorrencias": n}
                for (c, m, a, l), n in sorted(contagem.items())]

    def _finalizar(self, status: str, erro: Optional[str], termino: _dt.datetime,
                   duracao: float) -> dict:
        if self._hashes(self._entradas) != self._hash_entradas:
            raise HashIncompativel("entrada alterada durante a execução")
        if self._hashes(self._codigo, REPO_ROOT) != self._hash_codigo:
            raise HashIncompativel("código alterado durante a execução")
        saidas = self._hashes(self._saidas)
        git_final = estado_git(self.repo)
        caminhos_saida = {r["caminho"] for r in saidas.values()}
        extras = tuple(a for a in git_final.alteracoes if a[1] not in caminhos_saida)
        if self.finalidade is Finalidade.DEFINITIVA:
            if git_final.head != self._git_inicial.head or git_final.branch != self._git_inicial.branch:
                raise ExecucaoNaoDefinitiva("HEAD ou branch mudou durante a execução")
            if extras:
                raise ExecucaoNaoDefinitiva("alterações locais além das saídas registradas")
        elegivel = (self.finalidade is Finalidade.DEFINITIVA)
        corpo = {
            "formato": FORMATO,
            "versao_formato": VERSAO_FORMATO,
            "execucao": {
                "atividade": self.atividade,
                "finalidade": self.finalidade.value,
                "elegivel_definitiva": elegivel,
                "autorizacao": self.autorizacao.como_dict(),
                "hash_configuracao": self.hash_configuracao,
                "seeds": self.seeds,
                "parametros": self.parametros,
                "git": self._git_inicial.como_dict(),
                "ambiente": self._ambiente,
                "entradas": self._hash_entradas,
                "codigo": self._hash_codigo,
            },
            "resultado": {
                "status": status,
                "erro": erro,
                "warnings": self._warnings_agregados(),
                "saidas": saidas,
                "git_final": {**git_final.como_dict(),
                              "alteracoes_alem_das_saidas": [{"codigo": c, "caminho": p} for c, p in extras]},
            },
            "tempo": {
                "inicio": _instante(self._inicio),
                "termino": _instante(termino),
                "duracao_s": round(max(duracao, 0.0), 6),
            },
        }
        return validar_manifesto(_com_hashes(corpo))
