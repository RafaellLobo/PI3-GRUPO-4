"""Guardas de aprovação e autorização do benchmark S6 (P9, Seções 14.3, 14.4 e 19).

A execução definitiva só é autorizada com DOIS registros no pré-registro
(``docs/sprint6/preregistro_p9_benchmark.md``), cada um numa linha própria
começando na coluna 0, fora de bloco de código e de comentário HTML, no
formato das guardas da S4 (``guardas.formatar_registro``)::

    APROVACAO-DECISAO codigo=P9 status=aprovada data=AAAA-MM-DD valor_sha256=<64 hex>

* **P9:** ``valor_sha256`` = SHA-256 da configuração canônica gerada pelo
  código (``config.sha256_configuracao``).
* **S4_E3_REGISTRADA:** ``valor_sha256`` = SHA-256 canônico
  (``rastreabilidade.sha256_canonico``, igual a ``guardas.hash_valor_decisao``)
  do manifesto definitivo, elegível e concluído da Etapa 3 da S4, versionado.

Regras de cada registro, todas verificadas a cada chamada, sem cache:

1. no HEAD, exatamente um registro do código, com status ``aprovada``;
   qualquer registro ``revogada`` do código bloqueia;
2. o commit que o introduziu é único, é ancestral do HEAD, contém o registro
   uma única vez e nenhum dos seus pais o contém;
3. a data do registro é igual ou anterior à data desse commit;
4. nenhum commit do histórico do HEAD que não seja anterior à aprovação
   conteve revogação do código, nem em branch paralelo mesclado (revogar e
   depois apagar a revogação não reabilita a aprovação);
5. se o chamador informar o commit da aprovação, ele tem de ser exatamente
   o commit introdutor e ancestral do HEAD.

Regras adicionais:

* **P9 (Seção 14.4):** em todos os pais do commit da aprovação, a
  configuração canônica já estava versionada, com o SHA-256 do registro, e
  anotada na Seção 19; no commit da aprovação e no HEAD, o arquivo continua
  idêntico; o código gera a mesma configuração.
* **S4_E3_REGISTRADA:** o caminho do manifesto é um ``.json`` relativo e
  normalizado em ``reports/sprint4/``, validado antes de consultar o git; o
  manifesto existe, idêntico, no commit do registro e no HEAD; é um manifesto
  da S4 válido, sem dados individuais nem caminhos pessoais, da atividade
  ``etapa3_validacao``, definitivo, elegível e ``concluida``; o HEAD da execução da Etapa 3 é
  ancestral do commit do registro; as saídas registradas estão versionadas
  no HEAD com o SHA-256 registrado.

Fail-closed: registro ausente, malformado, duplicado, revogado, divergente,
fora de ordem ou impossível de verificar (git indisponível, HEAD instável,
diretório que não é a raiz do repositório) levanta ``BloqueioAprovacao``.
Não há caminho que devolva autorização parcial.

Reimplementação (P9, Seção 14.2): as regras espelham
``guardas._verificar`` e ``rastreabilidade.validar_manifesto`` da S4, que
não são públicas ou estão presas à configuração da S4. Importar outros
símbolos da S4 mudaria a configuração canônica, que registra a lista de
símbolos importados; por isso o adaptador continua com os mesmos nomes.

O módulo lê apenas objetos do git (``git cat-file``), nunca a cópia de
trabalho, e não lê nada em ``data/``. Não escreve arquivos nem registros.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from . import adaptadores_s4 as s4
from . import config as cfg

REPO_ROOT: Path = s4.REPO_ROOT

#: Documento onde os registros são aceitos (P9, Seção 19).
DOCUMENTO_APROVACAO: str = cfg.PREREGISTRO
MARCADOR_REGISTRO: str = "APROVACAO-DECISAO"
CODIGO_P9: str = "P9"
CODIGO_S4_E3: str = "S4_E3_REGISTRADA"
#: Únicos códigos admitidos no pré-registro; qualquer outro bloqueia.
CODIGOS_REGISTRO: Tuple[str, ...] = (CODIGO_P9, CODIGO_S4_E3)
STATUS_APROVADA: str = "aprovada"
STATUS_REVOGADA: str = "revogada"
STATUS_REGISTRO: Tuple[str, ...] = (STATUS_APROVADA, STATUS_REVOGADA)

#: Anotação da Seção 19, exigida antes do registro P9 (P9, Seção 14.4).
_ANOTACAO_SHA = re.compile(r"\*\*SHA-256 da configuração canônica:\*\* `(?P<sha>[0-9a-f]{64})`")
_REGISTRO = re.compile(
    r"APROVACAO-DECISAO"
    r" codigo=(?P<codigo>[A-Z][A-Z0-9_]*)"
    r" status=(?P<status>[a-z_]+)"
    r" data=(?P<data>\d{4}-\d{2}-\d{2})"
    r" valor_sha256=(?P<valor_sha256>[0-9a-f]{64})"
)
_CERCA = re.compile(r"(?P<cerca>`{3,}|~{3,})(?P<info>.*)")
_HASH_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_INSTANTE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
_TIMEOUT_GIT_S = 60

# Manifesto da S4 (espelho de ``rastreabilidade``: FORMATO, VERSAO_FORMATO,
# STATUS_CONCLUIDA, Finalidade.DEFINITIVA e as chaves de ``validar_manifesto``).
FORMATO_MANIFESTO_S4: str = "manifesto-cartao4-sprint4"
VERSAO_MANIFESTO_S4: int = 1
ATIVIDADE_S4_E3: str = "etapa3_validacao"
FINALIDADE_S4_DEFINITIVA: str = "definitiva"
STATUS_S4_CONCLUIDA: str = "concluida"
_CHAVES_TOPO = ("formato", "versao_formato", "execucao", "resultado", "tempo",
                "sha256_execucao", "sha256_reprodutivel", "sha256_manifesto")
_CHAVES_EXECUCAO = ("atividade", "finalidade", "elegivel_definitiva", "autorizacao",
                    "hash_configuracao", "seeds", "parametros", "git", "ambiente", "entradas", "codigo")
_CHAVES_RESULTADO = ("status", "erro", "warnings", "saidas", "git_final")
_CHAVES_TEMPO = ("inicio", "termino", "duracao_s")
_CHAVES_ARQUIVO = ("caminho", "sha256", "sha256_lf", "bytes")
_NOME_SEED = re.compile(r"[a-z][a-z0-9_]{0,63}")
#: Espelho de ``rastreabilidade.CHAVES_PROIBIDAS`` (igualdade conferida por teste).
CHAVES_PROIBIDAS_S4: frozenset = frozenset({
    "nodulo_id", "patient_id", "original_nodule_idx", "target_binary",
    "nodule_id", "scan_id", "y", "y_true", "y_pred", "y_score", "y_proba",
    "predicoes", "predicao", "probabilidades", "probabilidade", "scores", "linhas",
})
#: Espelho de ``rastreabilidade._PADROES_PROIBIDOS``.
_PADROES_PROIBIDOS = (
    (re.compile(r"LIDC-IDRI-\d{4}"), "identificador de paciente ou nódulo"),
    (re.compile(r"(^|[\s'\"(=])[A-Za-z]:[\\/]"), "caminho absoluto do Windows"),
    (re.compile(r"(^|[\s'\"(=])\\\\[^\\\s]+\\"), "caminho de rede"),
    (re.compile(r"(^|[\s'\"(=])/(home|Users|root|mnt|tmp|var|private)/"), "caminho absoluto"),
)
#: O manifesto da Etapa 3 só é procurado aqui (convenção dos manifestos da S4).
PREFIXO_MANIFESTO_S4: str = "reports/sprint4/"
_CONTROLE = re.compile(r"[\x00-\x1f\x7f]")

# --------------------------------------------------------------------------- #
# Exceções: toda recusa é BloqueioAprovacao (fail-closed)
# --------------------------------------------------------------------------- #


class BloqueioAprovacao(RuntimeError):
    """Execução definitiva não autorizada.

    ``chave_saida`` é a chave de ``EspecExecucao.codigos_saida`` que o
    coordenador usa: ``bloqueio_aprovacao`` (2) por padrão.
    """

    chave_saida: str = "bloqueio_aprovacao"


class RegistroAusente(BloqueioAprovacao):
    """Registro exigido inexistente no HEAD (inclui execução prematura)."""


class RegistroInvalido(BloqueioAprovacao):
    """Registro malformado, duplicado, fora de ordem, datado depois do commit,
    não introduzido pelo commit informado ou de commit não ancestral."""


class RegistroRevogado(BloqueioAprovacao):
    """Registro revogado no HEAD ou no histórico depois da aprovação."""


class ManifestoS4Invalido(BloqueioAprovacao):
    """Manifesto da Etapa 3 da S4 ausente, alterado, inválido ou não definitivo."""


class VerificacaoIndisponivel(BloqueioAprovacao):
    """Não foi possível verificar (git indisponível, HEAD instável, raiz errada)."""


class DivergenciaAprovacao(BloqueioAprovacao):
    """Hash aprovado diferente do conteúdo verificado (P9, Seção 14.3: integridade)."""

    chave_saida = "integridade"


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Registro:
    codigo: str
    status: str
    data: str
    valor_sha256: str


@dataclass(frozen=True)
class Evidencia:
    """Registro aceito e o commit que o introduziu."""

    codigo: str
    commit: str
    documento: str
    data: str
    valor_sha256: str

    def como_dict(self) -> Dict[str, str]:
        return {"codigo": self.codigo, "commit": self.commit, "documento": self.documento,
                "data": self.data, "valor_sha256": self.valor_sha256}


@dataclass(frozen=True)
class AutorizacaoDefinitiva:
    """Autorização concedida no momento da chamada; não vale para outro HEAD."""

    head: str
    sha256_configuracao: str
    p9: Evidencia
    s4_e3: Evidencia
    manifesto_s4: str

    def como_dict(self) -> Dict[str, Any]:
        return {"head": self.head, "sha256_configuracao": self.sha256_configuracao,
                "p9": self.p9.como_dict(), "s4_e3": self.s4_e3.como_dict(),
                "manifesto_s4": self.manifesto_s4}


# --------------------------------------------------------------------------- #
# Git: somente leitura de objetos, sem cache entre chamadas públicas
# --------------------------------------------------------------------------- #


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Espelho de ``guardas._git``: sem variáveis GIT_* e sem refs/replace."""
    executavel = shutil.which("git")
    if executavel is None:
        raise VerificacaoIndisponivel("git indisponível: não é possível verificar aprovações")
    ambiente = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    try:
        return subprocess.run([executavel, "--no-replace-objects", "-C", str(repo), *args],
                              capture_output=True, env=ambiente, timeout=_TIMEOUT_GIT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VerificacaoIndisponivel(f"falha ao consultar o git: {exc}") from exc


def _saida(processo: subprocess.CompletedProcess) -> str:
    return processo.stdout.decode("utf-8", errors="replace").strip()


class _LeitorGit:
    """Leituras de uma única chamada pública; o cache morre com ela."""

    def __init__(self, repo: Union[str, Path]) -> None:
        self.repo = Path(repo)
        self._blobs: Dict[Tuple[str, str], Optional[bytes]] = {}

    def head(self) -> str:
        """HEAD de ``repo``, que tem de ser a raiz do repositório (como na S4)."""
        processo = _git(self.repo, "rev-parse", "--show-toplevel", "HEAD^{commit}")
        linhas = _saida(processo).splitlines()
        if processo.returncode != 0 or len(linhas) != 2 or not _HASH_COMMIT.fullmatch(linhas[1]):
            raise VerificacaoIndisponivel(f"não foi possível ler o HEAD de {self.repo.name}")
        if os.path.normcase(Path(linhas[0]).resolve()) != os.path.normcase(self.repo.resolve()):
            raise VerificacaoIndisponivel(f"{self.repo.name} não é a raiz de um repositório git")
        return linhas[1]

    def commit(self, referencia: str) -> str:
        if type(referencia) is not str or not _HASH_COMMIT.fullmatch(referencia):
            raise RegistroInvalido(f"commit informado inválido: {referencia!r}; exige 40 hex minúsculos")
        if not self.existe_commit(referencia):
            raise RegistroInvalido(f"commit {referencia} inexistente ou não é commit")
        return referencia

    def existe_commit(self, commit: str) -> bool:
        processo = _git(self.repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
        return processo.returncode == 0 and _saida(processo) == commit

    def ancestral(self, commit: str, descendente: str) -> bool:
        processo = _git(self.repo, "merge-base", "--is-ancestor", commit, descendente)
        if processo.returncode not in (0, 1):
            raise VerificacaoIndisponivel(f"falha ao verificar a ancestralidade de {commit}")
        return processo.returncode == 0

    def pais(self, commit: str) -> Tuple[str, ...]:
        processo = _git(self.repo, "rev-list", "--parents", "-n", "1", commit)
        partes = _saida(processo).split()
        if processo.returncode != 0 or not partes or partes[0] != commit:
            raise VerificacaoIndisponivel(f"falha ao ler os pais de {commit}")
        return tuple(partes[1:])

    def data(self, commit: str) -> _dt.date:
        processo = _git(self.repo, "log", "-1", "--no-show-signature", "--format=%cs", commit)
        try:
            if processo.returncode != 0:
                raise ValueError
            return _dt.date.fromisoformat(_saida(processo))
        except ValueError as exc:
            raise VerificacaoIndisponivel(f"falha ao ler a data do commit {commit}") from exc

    def commits_que_alteram(self, head: str, caminho: str) -> Tuple[str, ...]:
        """Commits ancestrais de ``head`` que alteram ``caminho`` em relação a algum pai."""
        processo = _git(self.repo, "log", "--full-history", "--no-show-signature", "--format=%H",
                        head, "--", caminho)
        commits = tuple(_saida(processo).split())
        if processo.returncode != 0 or not all(_HASH_COMMIT.fullmatch(c) for c in commits):
            raise VerificacaoIndisponivel(f"falha ao ler o histórico de {caminho}")
        return commits

    def blob(self, commit: str, caminho: str) -> Optional[bytes]:
        """Bytes versionados de ``caminho`` em ``commit``; ``None`` se não existir."""
        chave = (commit, caminho)
        if chave not in self._blobs:
            processo = _git(self.repo, "cat-file", "blob", f"{commit}:{caminho}")
            self._blobs[chave] = processo.stdout if processo.returncode == 0 else None
        return self._blobs[chave]

    def documento(self, commit: str, caminho: str = DOCUMENTO_APROVACAO) -> Optional[str]:
        bruto = self.blob(commit, caminho)
        if bruto is None:
            return None
        try:
            return bruto.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RegistroInvalido(f"{caminho} em {commit[:12]} não é UTF-8 válido") from exc


def _caminho_versionado(caminho: Any, rotulo: str, lido: bool = True) -> str:
    """Caminho POSIX relativo à raiz, normalizado e sem ``..``.

    Se o caminho for ``lido`` pelas guardas, também fora de ``data/``. As
    entradas do manifesto da S4 (por exemplo, a tabela em ``data/``) só são
    validadas como texto, nunca lidas.
    """
    if type(caminho) is not str or not caminho or "\\" in caminho or ":" in caminho \
            or _CONTROLE.search(caminho):
        raise ManifestoS4Invalido(f"{rotulo}: caminho relativo POSIX exigido: {caminho!r}")
    puro = PurePosixPath(caminho)
    if puro.is_absolute() or not puro.parts or any(p in (".", "..") for p in puro.parts) \
            or str(puro) != caminho:
        raise ManifestoS4Invalido(f"{rotulo}: caminho relativo normalizado exigido: {caminho!r}")
    if lido and puro.parts[0].lower() == "data":
        raise ManifestoS4Invalido(f"{rotulo}: nada em data/ é lido pelas guardas")
    return caminho


def _caminho_manifesto(caminho: Any) -> str:
    """Caminho do manifesto da Etapa 3: versionado, em ``reports/sprint4/``, ``.json``.

    Validado antes de qualquer consulta ao git. Como a leitura é sempre
    ``git cat-file blob <commit>:<caminho>``, só objetos versionados no
    repositório podem ser lidos; a cópia de trabalho nunca é aberta.
    """
    caminho = _caminho_versionado(caminho, "manifesto da S4")
    if not caminho.startswith(PREFIXO_MANIFESTO_S4) or not caminho.endswith(".json"):
        raise ManifestoS4Invalido(
            f"manifesto da S4: exige um arquivo .json em {PREFIXO_MANIFESTO_S4}: {caminho!r}")
    return caminho


# --------------------------------------------------------------------------- #
# Registros no documento
# --------------------------------------------------------------------------- #


def _fecha_cerca(linha: str, cerca: Tuple[str, int]) -> bool:
    """Fechamento CommonMark estrito (espelho de ``guardas._fecha_cerca``)."""
    expandida = linha.expandtabs(4)
    resto = expandida.lstrip(" ")
    if len(expandida) - len(resto) > 3:
        return False
    resto = resto.rstrip()
    caractere, comprimento = cerca
    return len(resto) >= comprimento and set(resto) == {caractere}


def _linhas_visiveis(texto: str) -> Iterator[str]:
    """Linhas fora de bloco cercado e de comentário HTML (espelho de
    ``guardas._linhas_de_registro``). Erros de interpretação só podem esconder
    linhas, e assim falhar fechado, nunca expor um exemplo."""
    cerca: Optional[Tuple[str, int]] = None
    em_comentario = False
    for linha in texto.splitlines():
        if cerca is not None:
            if _fecha_cerca(linha, cerca):
                cerca = None
            continue
        if em_comentario:
            if "-->" in linha:
                em_comentario = linha.rfind("<!--") > linha.rfind("-->")
            continue
        abertura = _CERCA.fullmatch(linha.strip())
        if abertura and not (abertura["cerca"][0] == "`" and "`" in abertura["info"]):
            cerca = (abertura["cerca"][0], len(abertura["cerca"]))
            continue
        if linha.rfind("<!--") > linha.rfind("-->"):
            em_comentario = True
            continue
        yield linha.rstrip()


def ler_registros(texto: str, contexto: str = DOCUMENTO_APROVACAO, estrito: bool = True) -> Tuple[Registro, ...]:
    """Registros na coluna 0, fora de blocos.

    No modo estrito, toda linha que começa pelo marcador tem de estar bem
    formada, com código de ``CODIGOS_REGISTRO``, status de ``STATUS_REGISTRO``
    e data válida. O modo tolerante só responde se um registro já existia.
    """
    registros = []
    for linha in _linhas_visiveis(texto):
        if not linha.startswith(MARCADOR_REGISTRO):
            continue
        encontrado = _REGISTRO.fullmatch(linha)
        motivo = None
        if encontrado is None:
            motivo = "registro malformado"
        elif encontrado["codigo"] not in CODIGOS_REGISTRO:
            motivo = f"código desconhecido {encontrado['codigo']!r}"
        elif encontrado["status"] not in STATUS_REGISTRO:
            motivo = f"status desconhecido {encontrado['status']!r}"
        else:
            try:
                _dt.date.fromisoformat(encontrado["data"])
            except ValueError:
                motivo = "data inválida"
        if motivo is not None:
            if estrito:
                raise RegistroInvalido(f"{contexto}: {motivo}: {linha!r}")
            continue
        registros.append(Registro(**encontrado.groupdict()))
    return tuple(registros)


def _do_codigo(leitor: _LeitorGit, commit: str, codigo: str, estrito: bool) -> Tuple[Registro, ...]:
    texto = leitor.documento(commit)
    if texto is None:
        return ()
    return tuple(r for r in ler_registros(texto, f"{DOCUMENTO_APROVACAO} em {commit[:12]}", estrito)
                 if r.codigo == codigo)


def _introduz(leitor: _LeitorGit, commit: str, registro: Registro) -> bool:
    if registro not in _do_codigo(leitor, commit, registro.codigo, estrito=False):
        return False
    return all(registro not in _do_codigo(leitor, pai, registro.codigo, estrito=False)
               for pai in leitor.pais(commit))


def _verificar_registro(leitor: _LeitorGit, head: str, codigo: str, valor_esperado: Optional[str],
                        commit_informado: Optional[str]) -> Evidencia:
    """Regras 1 a 5 do docstring do módulo, para um código."""
    if leitor.documento(head) is None:
        raise RegistroAusente(f"{codigo}: {DOCUMENTO_APROVACAO} não existe no HEAD {head[:12]}")
    no_head = _do_codigo(leitor, head, codigo, estrito=True)
    if any(r.status == STATUS_REVOGADA for r in no_head):
        raise RegistroRevogado(f"{codigo}: registro revogado no HEAD {head[:12]}")
    if not no_head:
        raise RegistroAusente(f"{codigo}: nenhum registro de aprovação no HEAD {head[:12]}")
    if len(no_head) > 1:
        raise RegistroInvalido(f"{codigo}: {len(no_head)} registros no HEAD; exige exatamente um")
    registro = no_head[0]
    if valor_esperado is not None and registro.valor_sha256 != valor_esperado:
        raise DivergenciaAprovacao(
            f"{codigo}: valor_sha256 aprovado {registro.valor_sha256[:12]}… difere do verificado "
            f"{valor_esperado[:12]}…")

    historico = leitor.commits_que_alteram(head, DOCUMENTO_APROVACAO)
    introdutores = [c for c in historico if _introduz(leitor, c, registro)]
    if len(introdutores) != 1:
        raise RegistroInvalido(
            f"{codigo}: o registro foi introduzido {len(introdutores)} vezes no histórico do HEAD; "
            "exige um único commit de aprovação")
    commit = introdutores[0]
    if commit_informado is not None:
        leitor.commit(commit_informado)
        if not leitor.ancestral(commit_informado, head):
            raise RegistroInvalido(f"{codigo}: commit {commit_informado} não é ancestral do HEAD {head[:12]}")
        if commit_informado != commit:
            raise RegistroInvalido(
                f"{codigo}: o registro não foi introduzido pelo commit {commit_informado}, e sim por {commit}")
    if not leitor.ancestral(commit, head):
        raise RegistroInvalido(f"{codigo}: commit de aprovação {commit} não é ancestral do HEAD")
    if _do_codigo(leitor, commit, codigo, estrito=True) != (registro,):
        raise RegistroInvalido(f"{codigo}: o commit {commit[:12]} não contém o registro uma única vez")
    if _dt.date.fromisoformat(registro.data) > leitor.data(commit):
        raise RegistroInvalido(f"{codigo}: registro datado de {registro.data}, depois do commit {commit[:12]}")
    # Só uma revogação ANTERIOR à aprovação (ancestral dela) foi superada pelo
    # novo registro. Qualquer outra, posterior ou de branch paralelo mesclado,
    # bloqueia, mesmo que já tenha sido apagada.
    for outro in historico:
        if not leitor.ancestral(outro, commit) and any(
                r.status == STATUS_REVOGADA for r in _do_codigo(leitor, outro, codigo, estrito=False)):
            raise RegistroRevogado(
                f"{codigo}: revogação em {outro[:12]}, que não é anterior à aprovação; exige novo registro")
    return Evidencia(codigo, commit, DOCUMENTO_APROVACAO, registro.data, registro.valor_sha256)


# --------------------------------------------------------------------------- #
# P9: configuração canônica (Seção 14.4)
# --------------------------------------------------------------------------- #


def _exigir_configuracao(leitor: _LeitorGit, commit: str, sha_esperado: str, onde: str) -> None:
    bruto = leitor.blob(commit, cfg.ARQUIVO_CONFIGURACAO)
    if bruto is None:
        raise RegistroInvalido(
            f"P9: {cfg.ARQUIVO_CONFIGURACAO} não está versionado {onde} ({commit[:12]}); a configuração "
            "é versionada e anotada antes do registro de aprovação (Seção 14.4)")
    try:
        conteudo = cfg.ler_conteudo(bruto, f"{cfg.ARQUIVO_CONFIGURACAO} em {commit[:12]}")
    except cfg.ConfiguracaoDivergente as exc:
        raise DivergenciaAprovacao(f"P9: {exc}") from exc
    if s4.sha256_canonico(conteudo) != sha_esperado:
        raise DivergenciaAprovacao(
            f"P9: configuração versionada {onde} ({commit[:12]}) difere da aprovada")


def _exigir_anotacao(leitor: _LeitorGit, commit: str, sha_esperado: str, onde: str) -> None:
    texto = leitor.documento(commit) or ""
    anotacoes = [a["sha"] for linha in _linhas_visiveis(texto) for a in [_ANOTACAO_SHA.match(linha)] if a]
    if anotacoes != [sha_esperado]:
        raise RegistroInvalido(
            f"P9: a Seção 19 {onde} ({commit[:12]}) não anota exatamente uma vez o SHA-256 aprovado "
            "(Seção 14.4)")


def _verificar_p9(leitor: _LeitorGit, head: str, configuracao: cfg.ConfiguracaoBenchmark,
                  commit_aprovacao: Optional[str]) -> Tuple[Evidencia, str]:
    try:
        sha_codigo = cfg.sha256_configuracao(configuracao)
    except (cfg.ConfiguracaoInvalida, TypeError) as exc:
        raise DivergenciaAprovacao(f"P9: configuração do código inválida: {exc}") from exc
    evidencia = _verificar_registro(leitor, head, CODIGO_P9, sha_codigo, commit_aprovacao)
    pais = leitor.pais(evidencia.commit)
    if not pais:
        raise RegistroInvalido("P9: o commit da aprovação não pode ser o primeiro do histórico (Seção 14.4)")
    for pai in pais:
        _exigir_configuracao(leitor, pai, sha_codigo, "antes da aprovação")
        _exigir_anotacao(leitor, pai, sha_codigo, "antes da aprovação")
    _exigir_configuracao(leitor, evidencia.commit, sha_codigo, "no commit da aprovação")
    _exigir_configuracao(leitor, head, sha_codigo, "no HEAD")
    _exigir_anotacao(leitor, head, sha_codigo, "no HEAD")
    return evidencia, sha_codigo


# --------------------------------------------------------------------------- #
# S4_E3_REGISTRADA: manifesto definitivo da Etapa 3 da S4
# --------------------------------------------------------------------------- #


def _sem_duplicatas(pares: List[Tuple[str, Any]]) -> dict:
    chaves = [k for k, _ in pares]
    if len(chaves) != len(set(chaves)):
        raise ManifestoS4Invalido("manifesto da S4 com chave JSON duplicada")
    return dict(pares)


def _constante_invalida(nome: str) -> Any:
    raise ManifestoS4Invalido(f"manifesto da S4 com constante não finita: {nome}")


def _exigir_chaves(obj: Any, chaves: Tuple[str, ...], onde: str) -> dict:
    if not isinstance(obj, dict) or set(obj) != set(chaves):
        raise ManifestoS4Invalido(f"manifesto da S4: {onde} com chaves diferentes de {sorted(chaves)}")
    return obj


def _exigir_arquivos(registros: Any, onde: str, lidos: bool) -> Dict[str, dict]:
    if not isinstance(registros, dict):
        raise ManifestoS4Invalido(f"manifesto da S4: {onde} deve ser objeto")
    for rotulo, registro in registros.items():
        _exigir_chaves(registro, _CHAVES_ARQUIVO, f"{onde}.{rotulo}")
        _caminho_versionado(registro["caminho"], f"{onde}.{rotulo}", lido=lidos)
        for chave in ("sha256", "sha256_lf"):
            if not (isinstance(registro[chave], str) and _SHA256.fullmatch(registro[chave])):
                raise ManifestoS4Invalido(f"manifesto da S4: {onde}.{rotulo}.{chave} inválido")
        if type(registro["bytes"]) is not int or registro["bytes"] < 0:
            raise ManifestoS4Invalido(f"manifesto da S4: {onde}.{rotulo}.bytes inválido")
    return registros


def _textos_locais() -> Tuple[str, ...]:
    """Espelho de ``rastreabilidade._textos_proibidos``: home, raiz e temporário."""
    textos = set()
    for caminho in (Path.home(), REPO_ROOT, Path(tempfile.gettempdir())):
        try:
            resolvido = str(caminho.resolve())
        except OSError:
            continue
        textos.update({resolvido, resolvido.replace("\\", "/")})
    return tuple(t.lower() for t in textos if len(t) > 3)


def _exigir_sem_dados_individuais(obj: Any, onde: str = "manifesto") -> None:
    """Espelho de ``rastreabilidade._exigir_sem_dados_individuais``.

    Recusa chave de dado individual (sem distinguir caixa), identificador
    ``LIDC-IDRI-dddd``, caminho absoluto, de rede ou local, em chaves e
    valores, em qualquer profundidade.
    """
    locais = _textos_locais()

    def verificar_texto(texto: str, local: str) -> None:
        for padrao, motivo in _PADROES_PROIBIDOS:
            if padrao.search(texto):
                raise ManifestoS4Invalido(f"manifesto da S4: {local}: {motivo}")
        if any(t in texto.lower() for t in locais):
            raise ManifestoS4Invalido(f"manifesto da S4: {local}: caminho local")

    def visitar(valor: Any, local: str) -> None:
        if isinstance(valor, dict):
            for chave, item in valor.items():
                if not isinstance(chave, str):
                    raise ManifestoS4Invalido(f"manifesto da S4: {local}: chave não textual")
                if chave.lower() in CHAVES_PROIBIDAS_S4:
                    raise ManifestoS4Invalido(f"manifesto da S4: {local}.{chave}: chave de dado individual")
                verificar_texto(chave, f"{local}.{chave}")
                visitar(item, f"{local}.{chave}")
        elif isinstance(valor, list):
            for i, item in enumerate(valor):
                visitar(item, f"{local}[{i}]")
        elif isinstance(valor, str):
            verificar_texto(valor, local)

    visitar(obj, onde)


def _exigir_sementes(sementes: Any) -> None:
    """Espelho de ``rastreabilidade._validar_seeds``."""
    if not isinstance(sementes, dict):
        raise ManifestoS4Invalido("manifesto da S4: execucao.seeds deve ser objeto")
    for nome, valor in sementes.items():
        if not (isinstance(nome, str) and _NOME_SEED.fullmatch(nome)):
            raise ManifestoS4Invalido(f"manifesto da S4: nome de semente inválido: {nome!r}")
        if type(valor) is not int or not 0 <= valor < 2 ** 32:
            raise ManifestoS4Invalido(f"manifesto da S4: semente {nome} fora de [0, 2^32)")


def validar_manifesto_s4(bruto: bytes) -> dict:
    """Manifesto da S4 definitivo, elegível e concluído da Etapa 3.

    Espelha ``rastreabilidade.ler_manifesto``, ``validar_manifesto`` e
    ``resultado_definitivo_utilizavel``: bytes canônicos (JSON indentado,
    chaves ordenadas, LF), chaves exatas, sementes, coerência da execução
    definitiva, varredura de dados individuais e os três hashes internos
    recalculados. Mais estrito que a S4: só a atividade ``etapa3_validacao``,
    só ``definitiva``, elegível e ``concluida``, e caminhos normalizados.
    """
    normalizado = bruto.replace(b"\r\n", b"\n")
    try:
        m = json.loads(normalizado.decode("utf-8"), object_pairs_hook=_sem_duplicatas,
                       parse_constant=_constante_invalida)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestoS4Invalido(f"manifesto da S4 ilegível: {exc}") from exc
    try:
        canonico = json.dumps(m, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ManifestoS4Invalido(f"manifesto da S4 não serializável: {exc}") from exc
    if canonico.encode("utf-8") != normalizado:
        raise ManifestoS4Invalido("manifesto da S4 fora da forma canônica")
    _exigir_chaves(m, _CHAVES_TOPO, "topo")
    if m["formato"] != FORMATO_MANIFESTO_S4 or m["versao_formato"] != VERSAO_MANIFESTO_S4:
        raise ManifestoS4Invalido("manifesto da S4 com formato ou versão desconhecidos")
    execucao = _exigir_chaves(m["execucao"], _CHAVES_EXECUCAO, "execucao")
    resultado = _exigir_chaves(m["resultado"], _CHAVES_RESULTADO, "resultado")
    tempo = _exigir_chaves(m["tempo"], _CHAVES_TEMPO, "tempo")

    if execucao["atividade"] != ATIVIDADE_S4_E3:
        raise ManifestoS4Invalido(f"manifesto da S4 da atividade {execucao['atividade']!r}, não da Etapa 3")
    if execucao["finalidade"] != FINALIDADE_S4_DEFINITIVA:
        raise ManifestoS4Invalido("manifesto da S4 não é de execução definitiva")
    if execucao["elegivel_definitiva"] is not True:
        raise ManifestoS4Invalido("manifesto da S4 não é elegível como definitivo")
    if not isinstance(execucao["autorizacao"], dict) or \
            execucao["autorizacao"].get("atividade") != execucao["atividade"]:
        raise ManifestoS4Invalido("manifesto da S4: autorização de outra atividade")
    if not (isinstance(execucao["hash_configuracao"], str) and _SHA256.fullmatch(execucao["hash_configuracao"])):
        raise ManifestoS4Invalido("manifesto da S4: hash_configuracao inválido")
    _exigir_sementes(execucao["seeds"])
    for chave in ("parametros", "ambiente"):
        if not isinstance(execucao[chave], dict):
            raise ManifestoS4Invalido(f"manifesto da S4: execucao.{chave} deve ser objeto")
    git_inicial, git_final = execucao["git"], resultado["git_final"]
    for onde, registro_git in (("execucao.git", git_inicial), ("resultado.git_final", git_final)):
        if not (isinstance(registro_git, dict) and isinstance(registro_git.get("head"), str)
                and _HASH_COMMIT.fullmatch(registro_git["head"])):
            raise ManifestoS4Invalido(f"manifesto da S4: {onde}.head inválido")
    if not (git_inicial.get("limpo") is True
            and execucao["ambiente"].get("confere_com_esperado") is True
            and git_final.get("alteracoes_alem_das_saidas") == []
            and git_final["head"] == git_inicial["head"]):
        raise ManifestoS4Invalido("manifesto da S4 sem árvore limpa, ambiente esperado ou HEAD estável")
    _exigir_arquivos(execucao["entradas"], "execucao.entradas", lidos=False)
    _exigir_arquivos(execucao["codigo"], "execucao.codigo", lidos=False)
    _exigir_arquivos(resultado["saidas"], "resultado.saidas", lidos=True)
    if resultado["status"] != STATUS_S4_CONCLUIDA or resultado["erro"] is not None:
        raise ManifestoS4Invalido(f"manifesto da S4 com status {resultado['status']!r}; exige 'concluida' sem erro")
    if not isinstance(resultado["warnings"], list):
        raise ManifestoS4Invalido("manifesto da S4: resultado.warnings deve ser lista")
    for chave in ("inicio", "termino"):
        if not (isinstance(tempo[chave], str) and _INSTANTE.fullmatch(tempo[chave])):
            raise ManifestoS4Invalido(f"manifesto da S4: tempo.{chave} fora do formato UTC")
    if type(tempo["duracao_s"]) not in (int, float) or tempo["duracao_s"] < 0:
        raise ManifestoS4Invalido("manifesto da S4: tempo.duracao_s inválida")
    _exigir_sem_dados_individuais({k: m[k] for k in ("execucao", "resultado", "tempo")})

    corpo ={k: m[k] for k in ("formato", "versao_formato", "execucao", "resultado", "tempo")}
    esperado = dict(corpo)
    esperado["sha256_execucao"] = s4.sha256_canonico(corpo["execucao"])
    esperado["sha256_reprodutivel"] = s4.sha256_canonico({"execucao": corpo["execucao"],
                                                          "resultado": corpo["resultado"]})
    esperado["sha256_manifesto"] = s4.sha256_canonico(esperado)
    for chave in ("sha256_execucao", "sha256_reprodutivel", "sha256_manifesto"):
        if m[chave] != esperado[chave]:
            raise ManifestoS4Invalido(f"manifesto da S4: {chave} não confere (corrompido ou alterado)")
    return m


def sha256_manifesto_s4(manifesto: Mapping[str, Any]) -> str:
    """``valor_sha256`` do registro S4_E3_REGISTRADA: SHA-256 canônico do manifesto inteiro."""
    return s4.sha256_canonico(dict(manifesto))


def _exigir_caminho_manifesto(manifesto: Optional[str]) -> str:
    if manifesto is None:
        raise RegistroAusente(
            f"{CODIGO_S4_E3}: informe o manifesto definitivo da Etapa 3 da S4; sem ele a execução "
            "definitiva não é autorizada")
    return _caminho_manifesto(manifesto)


def _verificar_s4_e3(leitor: _LeitorGit, head: str, caminho: str,
                     commit_registro: Optional[str]) -> Evidencia:
    """``caminho`` já validado por ``_exigir_caminho_manifesto``, antes do git."""
    evidencia = _verificar_registro(leitor, head, CODIGO_S4_E3, None, commit_registro)
    no_registro, no_head = leitor.blob(evidencia.commit, caminho), leitor.blob(head, caminho)
    if no_registro is None:
        raise ManifestoS4Invalido(f"{caminho} não está versionado no commit do registro {evidencia.commit[:12]}")
    if no_head is None:
        raise ManifestoS4Invalido(f"{caminho} não existe no HEAD {head[:12]}")
    if no_registro.replace(b"\r\n", b"\n") != no_head.replace(b"\r\n", b"\n"):
        raise ManifestoS4Invalido(f"{caminho} mudou depois do registro {evidencia.commit[:12]}")
    m = validar_manifesto_s4(no_head)
    if sha256_manifesto_s4(m) != evidencia.valor_sha256:
        raise DivergenciaAprovacao(
            f"{CODIGO_S4_E3}: valor_sha256 aprovado difere do SHA-256 canônico de {caminho}")
    execucao_head = m["execucao"]["git"]["head"]
    if not leitor.existe_commit(execucao_head):
        raise ManifestoS4Invalido(f"HEAD da execução da Etapa 3 ({execucao_head[:12]}) não existe no repositório")
    if not leitor.ancestral(execucao_head, evidencia.commit):
        raise ManifestoS4Invalido(
            f"HEAD da execução da Etapa 3 ({execucao_head[:12]}) não é ancestral do registro "
            f"{evidencia.commit[:12]}")
    for rotulo, registro in m["resultado"]["saidas"].items():
        bruto = leitor.blob(head, registro["caminho"])
        if bruto is None:
            raise ManifestoS4Invalido(f"saída {rotulo} ({registro['caminho']}) não versionada no HEAD")
        if hashlib.sha256(bruto.replace(b"\r\n", b"\n")).hexdigest() != registro["sha256_lf"]:
            raise ManifestoS4Invalido(f"saída {rotulo} ({registro['caminho']}) difere do registrado")
    return evidencia


# --------------------------------------------------------------------------- #
# Pontos de entrada
# --------------------------------------------------------------------------- #


def _exigir_head_estavel(leitor: _LeitorGit, head: str) -> None:
    if leitor.head() != head:
        raise VerificacaoIndisponivel("o HEAD mudou durante a verificação; repita a autorização")


def verificar_p9(repo: Union[str, Path] = REPO_ROOT,
                 configuracao: cfg.ConfiguracaoBenchmark = cfg.CONFIGURACAO,
                 commit_aprovacao: Optional[str] = None) -> Evidencia:
    """Evidência do registro P9 contra o HEAD de ``repo``; ``BloqueioAprovacao`` sem ela."""
    leitor = _LeitorGit(repo)
    head = leitor.head()
    evidencia, _ = _verificar_p9(leitor, head, configuracao, commit_aprovacao)
    _exigir_head_estavel(leitor, head)
    return evidencia


def verificar_s4_e3(manifesto: Optional[str], repo: Union[str, Path] = REPO_ROOT,
                    commit_registro: Optional[str] = None) -> Evidencia:
    """Evidência do registro S4_E3_REGISTRADA e do manifesto definitivo da Etapa 3."""
    caminho = _exigir_caminho_manifesto(manifesto)
    leitor = _LeitorGit(repo)
    head = leitor.head()
    evidencia = _verificar_s4_e3(leitor, head, caminho, commit_registro)
    _exigir_head_estavel(leitor, head)
    return evidencia


def autorizar_definitivo(manifesto_s4: Optional[str], repo: Union[str, Path] = REPO_ROOT,
                         configuracao: cfg.ConfiguracaoBenchmark = cfg.CONFIGURACAO,
                         commit_p9: Optional[str] = None,
                         commit_s4_e3: Optional[str] = None) -> AutorizacaoDefinitiva:
    """Autoriza a execução definitiva agora ou levanta ``BloqueioAprovacao``.

    Exige P9 e S4_E3_REGISTRADA, verificados contra o mesmo HEAD. Deve ser
    chamada imediatamente antes de ler a partição de validação; não há cache.
    """
    if not isinstance(configuracao, cfg.ConfiguracaoBenchmark):
        raise DivergenciaAprovacao("configuracao deve ser ConfiguracaoBenchmark")
    if tuple(configuracao.execucao.registros_exigidos) != CODIGOS_REGISTRO:
        raise DivergenciaAprovacao(
            f"registros exigidos pela configuração {configuracao.execucao.registros_exigidos} "
            f"diferem dos verificados {CODIGOS_REGISTRO}")
    caminho = _exigir_caminho_manifesto(manifesto_s4)
    leitor = _LeitorGit(repo)
    head = leitor.head()
    p9, sha_configuracao = _verificar_p9(leitor, head, configuracao, commit_p9)
    s4_e3 = _verificar_s4_e3(leitor, head, caminho, commit_s4_e3)
    _exigir_head_estavel(leitor, head)
    return AutorizacaoDefinitiva(head, sha_configuracao, p9, s4_e3, caminho)
