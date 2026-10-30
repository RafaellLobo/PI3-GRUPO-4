"""Checkpoints e retomada segura por unidade experimental (Incremento 2C-B).

Uma UNIDADE é ``(atividade, repetição, fold, modelo)`` da futura CV aninhada.
Seu checkpoint só é reaproveitado se a IDENTIDADE gravada for idêntica à
esperada. A identidade reúne:

* atividade, repetição, fold e modelo;
* hash da configuração (``config.hash_configuracao``) e seeds da unidade;
* ambiente (versões, implementação, plataforma);
* SHA-256 do código (normalizado em LF, imune a ``core.autocrlf``) e das
  entradas (bruto e LF);
* fingerprint da composição dos folds (``fingerprint_folds``).

O fingerprint guarda só um hash: os identificadores dos nódulos nunca são
gravados no registro.

Estrutura em disco, sempre num diretório local NÃO versionado (ignorado pelo
git ou fora de qualquer repositório), nunca em ``data/``::

    <diretorio>/<atividade>/r<rrr>-f<ff>-<modelo>/
        payload.json      resultado da unidade (pode ter dados individuais)
        concluida.json    registro de conclusão — publicado POR ÚLTIMO
        falha-<nnn>.json  registros de tentativas que falharam (histórico)

Regras:

* tudo é publicado de forma atômica (temporário + ``fsync`` + ``os.link``) e
  nada é sobrescrito. O payload vem antes do registro: sem
  ``concluida.json`` válido, a unidade NUNCA está concluída;
* na retomada, um slot está ``ausente``, ``falhou`` (só registros de falha,
  pode ser executado de novo) ou ``concluida``. Qualquer outra coisa aborta
  com exceção e nada é apagado nem reaproveitado: payload órfão, resíduo de
  escrita interrompida, arquivo desconhecido, registro corrompido, hash de
  payload divergente, identidade diferente, slot não esperado;
* toda operação (``estado``, ``carregar``, ``verificar``, ``salvar``,
  ``registrar_falha``) recalcula os hashes de código e entradas. Se mudaram
  desde a abertura da sessão, levanta ``HashIncompativel`` antes de olhar o
  disco: uma sessão desatualizada nunca reaproveita checkpoints;
* ``verificar`` diz quais unidades estão concluídas e quais faltam; NÃO
  declara a etapa científica concluída;
* checkpoints não substituem as guardas. A sessão reemite a autorização com
  ``guardas.exigir_autorizacao``; unidades de M2, M3 e M4 exigem P3 na
  autorização; a atividade faz parte da identidade, então um checkpoint de
  uma etapa nunca é carregado noutra. O módulo não lê dados nem dá acesso a
  partições;
* persistência só em JSON (sem pickle). Registros passam pela varredura de
  dados individuais de ``rastreabilidade``; payloads ficam apenas no
  diretório local.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from . import config as cfg
from . import guardas
from . import rastreabilidade as rst

REPO_ROOT: Path = rst.REPO_ROOT

FORMATO: str = "checkpoint-cartao4-sprint4"
VERSAO_FORMATO: int = 1

ESTADO_AUSENTE = "ausente"
ESTADO_FALHOU = "falhou"
ESTADO_CONCLUIDA = "concluida"

ARQUIVO_PAYLOAD = "payload.json"
ARQUIVO_CONCLUIDA = "concluida.json"
_FALHA = re.compile(r"falha-(\d{3})\.json")
_RESIDUO = re.compile(r"\..+\.tmp")
_SLOT = re.compile(r"r(\d{3})-f(\d{2})-(M[0-4]v?)")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHAVES_REGISTRO = ("formato", "versao_formato", "identidade", "sha256_identidade", "status",
                    "payload", "resumo", "erro", "metadados", "sha256_registro")
_CHAVES_IDENTIDADE = ("atividade", "repeticao", "fold", "modelo", "hash_configuracao", "seeds",
                      "ambiente", "codigo", "dados", "fingerprint_folds")
_CHAVES_CONTEXTO = ("atividade", "hash_configuracao", "ambiente", "codigo", "dados")

# --------------------------------------------------------------------------- #
# Exceções — todas herdam de rastreabilidade.ErroRastreabilidade
# --------------------------------------------------------------------------- #


class CheckpointAusente(rst.ErroRastreabilidade):
    """Nenhum checkpoint para a unidade."""


class CheckpointNaoConcluido(rst.ErroRastreabilidade):
    """A unidade só tem registros de falha; precisa ser executada."""


class CheckpointCorrompido(rst.ErroRastreabilidade):
    """Registro, payload ou slot em estado que não pode ser interpretado."""


class CheckpointIncompativel(rst.ErroRastreabilidade):
    """Identidade gravada diferente da esperada, ou slot não esperado."""


class DiretorioNaoLocal(rst.ErroRastreabilidade):
    """Diretório versionável, dentro de ``data/`` ou inexistente."""


# --------------------------------------------------------------------------- #
# Fingerprint dos folds e identidade
# --------------------------------------------------------------------------- #


def fingerprint_folds(composicao: Mapping[str, Iterable[str]]) -> str:
    """SHA-256 da composição dos folds, sem gravar identificadores.

    ``composicao`` mapeia um papel (por exemplo ``"treino_externo"``) aos
    identificadores das linhas. A ordem das linhas não importa; um
    identificador repetido dentro do mesmo papel é erro.
    """
    if not isinstance(composicao, Mapping) or not composicao:
        raise ValueError("composicao deve ser um mapeamento papel → identificadores, não vazio")
    normalizada = {}
    for papel, ids in composicao.items():
        if not (isinstance(papel, str) and papel):
            raise ValueError(f"papel inválido: {papel!r}")
        if isinstance(ids, (str, bytes)):
            raise TypeError(f"{papel}: informe uma coleção de identificadores, não uma string")
        lista = [str(i) for i in ids]
        if len(set(lista)) != len(lista):
            raise ValueError(f"{papel}: identificador repetido")
        normalizada[papel] = sorted(lista)
    return rst.sha256_canonico(normalizada)


@dataclass(frozen=True)
class IdentidadeUnidade:
    """Identidade determinística de uma unidade. Monte com ``SessaoCheckpoints.identidade``."""

    atividade: str
    repeticao: int
    fold: int
    modelo: str
    hash_configuracao: str
    seeds: Mapping[str, int]
    ambiente: Mapping[str, Any]
    codigo: Mapping[str, Any]
    dados: Mapping[str, Any]
    fingerprint_folds: str

    def como_dict(self) -> dict:
        return json.loads(rst.json_canonico({
            "atividade": self.atividade, "repeticao": self.repeticao, "fold": self.fold,
            "modelo": self.modelo, "hash_configuracao": self.hash_configuracao,
            "seeds": dict(self.seeds), "ambiente": dict(self.ambiente), "codigo": dict(self.codigo),
            "dados": dict(self.dados), "fingerprint_folds": self.fingerprint_folds,
        }))

    @property
    def sha256(self) -> str:
        return rst.sha256_canonico(self.como_dict())

    @property
    def slot(self) -> str:
        return f"r{self.repeticao:03d}-f{self.fold:02d}-{self.modelo}"


# --------------------------------------------------------------------------- #
# Persistência
# --------------------------------------------------------------------------- #


def _publicar(conteudo: bytes, destino: Path) -> None:
    """Publicação atômica sem sobrescrita — o mesmo mecanismo de
    ``rastreabilidade.escrever_manifesto``, que só aceita manifestos."""
    if os.path.lexists(destino):
        raise rst.SobrescritaRecusada(f"{destino.name} já existe; checkpoints não são sobrescritos")
    descritor, temporario = tempfile.mkstemp(prefix=f".{destino.name}.", suffix=".tmp", dir=destino.parent)
    try:
        with os.fdopen(descritor, "wb") as arquivo:
            arquivo.write(conteudo)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        try:
            os.link(temporario, destino)
        except FileExistsError as exc:
            raise rst.SobrescritaRecusada(f"{destino.name} já existe; checkpoints não são sobrescritos") from exc
        except OSError as exc:
            raise rst.ErroRastreabilidade(f"publicação atômica indisponível: {exc}") from exc
    finally:
        try:
            os.unlink(temporario)
        except FileNotFoundError:
            pass


def _dentro_de_repositorio(caminho: Path) -> bool:
    return any((p / ".git").exists() for p in (caminho, *caminho.parents))


def _bytes_payload(payload: Mapping[str, Any]) -> bytes:
    if not isinstance(payload, Mapping):
        raise TypeError("payload deve ser um mapeamento JSON")
    try:
        return (rst.json_canonico(dict(payload)) + "\n").encode("utf-8")
    except rst.ManifestoInvalido as exc:
        raise ValueError(f"payload não serializável em JSON canônico: {exc}") from exc


def _ler_json(caminho: Path, canonico: Callable[[Any], bytes]) -> Any:
    """JSON seguro: sem pickle, sem chave duplicada, sem NaN, bytes canônicos."""
    try:
        bruto = caminho.read_bytes().replace(b"\r\n", b"\n")
        valor = json.loads(bruto.decode("utf-8"), object_pairs_hook=rst._sem_duplicatas,
                           parse_constant=rst._constante_invalida)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, rst.ManifestoInvalido) as exc:
        raise CheckpointCorrompido(f"{caminho.parent.name}/{caminho.name}: ilegível: {exc}") from exc
    try:
        forma_canonica = canonico(valor)
    except (TypeError, ValueError, rst.ManifestoInvalido) as exc:
        raise CheckpointCorrompido(f"{caminho.parent.name}/{caminho.name}: {exc}") from exc
    if forma_canonica != bruto:
        raise CheckpointCorrompido(f"{caminho.parent.name}/{caminho.name}: bytes fora da forma canônica")
    return valor


def _com_hash(corpo: dict) -> dict:
    registro = dict(corpo)
    registro["sha256_registro"] = rst.sha256_canonico(corpo)
    return registro


def _validar_registro(registro: Any, onde: str) -> dict:
    """Forma, coerência de status e hashes de um registro (sem comparar identidade)."""
    if not isinstance(registro, dict) or set(registro) != set(_CHAVES_REGISTRO):
        raise CheckpointCorrompido(f"{onde}: chaves do registro inválidas")
    if registro["formato"] != FORMATO or registro["versao_formato"] != VERSAO_FORMATO:
        raise CheckpointCorrompido(f"{onde}: formato desconhecido")
    identidade = registro["identidade"]
    if not isinstance(identidade, dict) or set(identidade) != set(_CHAVES_IDENTIDADE):
        raise CheckpointCorrompido(f"{onde}: identidade com chaves inválidas")
    if registro["sha256_identidade"] != rst.sha256_canonico(identidade):
        raise CheckpointCorrompido(f"{onde}: sha256_identidade não confere")
    status, payload, erro = registro["status"], registro["payload"], registro["erro"]
    if status == ESTADO_CONCLUIDA:
        if not (isinstance(payload, dict) and set(payload) == {"arquivo", "sha256", "bytes"}
                and payload["arquivo"] == ARQUIVO_PAYLOAD and _SHA256.fullmatch(str(payload["sha256"]))
                and type(payload["bytes"]) is int and erro is None):
            raise CheckpointCorrompido(f"{onde}: registro concluído sem payload coerente")
    elif status == ESTADO_FALHOU:
        if payload is not None or not (isinstance(erro, str) and erro):
            raise CheckpointCorrompido(f"{onde}: registro de falha incoerente")
    else:
        raise CheckpointCorrompido(f"{onde}: status desconhecido")
    if not isinstance(registro["resumo"], dict) or not isinstance(registro["metadados"], dict):
        raise CheckpointCorrompido(f"{onde}: resumo ou metadados com tipo inválido")
    corpo = {k: v for k, v in registro.items() if k != "sha256_registro"}
    if registro["sha256_registro"] != rst.sha256_canonico(corpo):
        raise CheckpointCorrompido(f"{onde}: sha256_registro não confere")
    try:
        rst._exigir_sem_dados_individuais({k: v for k, v in corpo.items()}, onde)
    except rst.DadosIndividuais as exc:
        raise CheckpointCorrompido(f"{onde}: {exc}") from exc
    return registro


def _diferencas(gravada: Mapping[str, Any], esperada: Mapping[str, Any]) -> List[str]:
    return sorted(k for k in set(gravada) | set(esperada) if gravada.get(k) != esperada.get(k))


# --------------------------------------------------------------------------- #
# Relatório de retomada
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RelatorioRetomada:
    """Estado das unidades esperadas. NÃO declara a etapa científica concluída:
    só diz o que já tem checkpoint válido e o que falta executar."""

    concluidas: Tuple[str, ...]
    ausentes: Tuple[str, ...]
    falhas: Tuple[str, ...]

    @property
    def a_executar(self) -> Tuple[str, ...]:
        return tuple(sorted(self.ausentes + self.falhas))

    @property
    def n_esperadas(self) -> int:
        return len(self.concluidas) + len(self.ausentes) + len(self.falhas)


# --------------------------------------------------------------------------- #
# Sessão
# --------------------------------------------------------------------------- #


class SessaoCheckpoints:
    """Contexto de checkpoints de UMA atividade, aberto na entrada da etapa.

    Na abertura: reemite a autorização com as guardas (tem de ser idêntica à
    informada), valida o diretório local e calcula o contexto da identidade
    (configuração, ambiente, código, entradas). Cada gravação recalcula o
    contexto e falha se código ou entradas mudaram.
    """

    def __init__(self, diretorio: Union[str, Path], atividade: guardas.Atividade,
                 autorizacao: guardas.Autorizacao, entradas: Mapping[str, Union[str, Path]],
                 decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
                 repo: Union[str, Path] = REPO_ROOT,
                 codigo: Optional[Mapping[str, Union[str, Path]]] = None,
                 relogio_utc: Callable[[], _dt.datetime] = lambda: _dt.datetime.now(_dt.timezone.utc)) -> None:
        self.atividade = guardas.normalizar_atividade(atividade)
        if not isinstance(autorizacao, guardas.Autorizacao):
            raise TypeError("autorizacao deve ser guardas.Autorizacao")
        if autorizacao.atividade != self.atividade:
            raise rst.ErroRastreabilidade("a autorização é de outra atividade")
        self.repo = Path(repo)
        if guardas.exigir_autorizacao(self.atividade, decisoes, self.repo) != autorizacao:
            raise rst.ErroRastreabilidade("a autorização informada não confere com a emitida pelas guardas")
        self.autorizacao = autorizacao
        self._decisoes = decisoes
        self.diretorio = self._exigir_diretorio_local(Path(diretorio))
        if not isinstance(entradas, Mapping) or not entradas:
            raise ValueError("entradas deve mapear rótulo → caminho, sem ser vazio")
        self._entradas = {str(k): Path(v) for k, v in entradas.items()}
        self._codigo = {**rst.codigo_do_pacote(), **{str(k): Path(v) for k, v in (codigo or {}).items()}}
        self._relogio_utc = relogio_utc
        # Guardado como JSON canônico: cada acesso devolve uma cópia nova.
        self._contexto_json = rst.json_canonico(self._calcular_contexto())
        self._head = rst.estado_git(self.repo).head

    @property
    def contexto(self) -> dict:
        return json.loads(self._contexto_json)

    # -- diretório e contexto ------------------------------------------------

    def _exigir_diretorio_local(self, diretorio: Path) -> Path:
        if not diretorio.is_dir():
            raise DiretorioNaoLocal("diretório de checkpoints inexistente")
        resolvido = diretorio.resolve()
        for raiz in {REPO_ROOT.resolve(), self.repo.resolve()}:
            try:
                relativo = resolvido.relative_to(raiz).as_posix()
            except ValueError:
                continue
            if relativo == "data" or relativo.startswith("data/"):
                raise DiretorioNaoLocal("checkpoints não ficam em data/")
        # Dentro de qualquer repositório (detectado por .git nos ancestrais), o
        # diretório tem de ser ignorado por ele; falha do git nunca libera.
        if _dentro_de_repositorio(resolvido):
            sonda = resolvido / ".sonda-checkpoint"
            ignorado = rst._git(resolvido, "check-ignore", "-q", "--no-index", str(sonda))
            if ignorado.returncode == 1:
                raise DiretorioNaoLocal("o diretório de checkpoints não é ignorado pelo git (versionável)")
            if ignorado.returncode != 0:
                raise rst.ErroGit("falha ao verificar se o diretório de checkpoints é ignorado")
        return resolvido

    def _calcular_contexto(self) -> dict:
        amb = rst.ambiente()
        return {
            "atividade": self.atividade,
            "hash_configuracao": cfg.hash_configuracao(self._decisoes),
            "ambiente": {"versoes": amb["versoes"], "implementacao": amb["implementacao"],
                         "plataforma": amb["plataforma"]},
            "codigo": {rotulo: {"caminho": h.caminho, "sha256_lf": h.sha256_lf}
                       for rotulo, h in ((k, rst.hash_arquivo(p, REPO_ROOT))
                                         for k, p in sorted(self._codigo.items()))},
            "dados": {rotulo: rst.hash_arquivo(p, self.repo).como_dict()
                      for rotulo, p in sorted(self._entradas.items())},
        }

    def _exigir_contexto_inalterado(self) -> None:
        atual = json.loads(rst.json_canonico(self._calcular_contexto()))
        contexto = self.contexto
        for chave in ("codigo", "dados"):
            if atual[chave] != contexto[chave]:
                raise rst.HashIncompativel(f"{chave} alterado desde a abertura da sessão")
        if atual != contexto:
            raise CheckpointIncompativel("contexto da sessão mudou (configuração ou ambiente)")

    # -- identidade ----------------------------------------------------------

    def _validar_unidade(self, repeticao: Any, fold: Any, modelo: Any, seeds: Any, fingerprint: Any) -> dict:
        """Regras da unidade, aplicadas na criação E em toda operação.

        M2, M3 e M4 exigem P3 na autorização da sessão: o checkpoint não
        substitui as guardas, nem com uma ``IdentidadeUnidade`` montada à mão.
        """
        for nome, valor, limite in (("repeticao", repeticao, cfg.ESPEC_CV.n_repeticoes),
                                    ("fold", fold, cfg.ESPEC_CV.n_folds_externos)):
            if type(valor) is not int or not 0 <= valor < limite:
                raise ValueError(f"{nome} deve ser inteiro em [0, {limite})")
        if type(modelo) is not str or modelo not in cfg.MODELOS_FIXOS + cfg.MODELOS_VARIAVEIS:
            raise guardas.ModeloDesconhecido(f"modelo desconhecido: {modelo!r}")
        exigidas = guardas.decisoes_exigidas(self.atividade, (modelo,))
        faltam = [c for c in exigidas if c not in self.autorizacao.decisoes_exigidas]
        if faltam:
            raise cfg.BloqueioDecisao(f"{modelo}: a autorização da sessão não cobre {faltam}")
        if not (isinstance(fingerprint, str) and _SHA256.fullmatch(fingerprint)):
            raise ValueError("fingerprint deve vir de fingerprint_folds")
        return rst._validar_seeds(seeds)

    def identidade(self, repeticao: int, fold: int, modelo: str, seeds: Mapping[str, int],
                   fingerprint: str) -> IdentidadeUnidade:
        """Identidade de uma unidade desta sessão."""
        seeds = self._validar_unidade(repeticao, fold, modelo, seeds, fingerprint)
        contexto = self.contexto
        return IdentidadeUnidade(
            atividade=self.atividade, repeticao=repeticao, fold=fold, modelo=modelo,
            hash_configuracao=contexto["hash_configuracao"], seeds=seeds,
            ambiente=contexto["ambiente"], codigo=contexto["codigo"],
            dados=contexto["dados"], fingerprint_folds=fingerprint)

    def _exigir_da_sessao(self, identidade: IdentidadeUnidade) -> dict:
        if not isinstance(identidade, IdentidadeUnidade):
            raise TypeError("identidade deve ser IdentidadeUnidade")
        self._validar_unidade(identidade.repeticao, identidade.fold, identidade.modelo,
                              identidade.seeds, identidade.fingerprint_folds)
        esperada = identidade.como_dict()
        contexto = self.contexto
        diferentes = [k for k in _CHAVES_CONTEXTO if esperada[k] != contexto[k]]
        if diferentes:
            raise CheckpointIncompativel(f"identidade de outra sessão: difere em {diferentes}")
        return esperada

    def _pasta(self, identidade: IdentidadeUnidade) -> Path:
        return self.diretorio / self.atividade / identidade.slot

    # -- estado de um slot -----------------------------------------------------

    def estado(self, identidade: IdentidadeUnidade) -> str:
        """``ausente``, ``falhou`` ou ``concluida``; qualquer outra situação levanta exceção.

        Recalcula antes os hashes de código e entradas: se mudaram desde a
        abertura da sessão, nenhum checkpoint é reconhecido (``HashIncompativel``).
        """
        self._exigir_contexto_inalterado()
        return self._estado(identidade)

    def _estado(self, identidade: IdentidadeUnidade) -> str:
        """Classificação do slot; quem chama já conferiu o contexto."""
        esperada = self._exigir_da_sessao(identidade)
        pasta = self._pasta(identidade)
        if not pasta.exists():
            return ESTADO_AUSENTE
        if not pasta.is_dir():
            raise CheckpointCorrompido(f"{identidade.slot}: não é diretório")
        nomes = sorted(p.name for p in pasta.iterdir())
        residuos = [n for n in nomes if _RESIDUO.fullmatch(n)]
        if residuos:
            raise CheckpointCorrompido(f"{identidade.slot}: resíduo de escrita interrompida {residuos}")
        desconhecidos = [n for n in nomes
                         if n not in (ARQUIVO_PAYLOAD, ARQUIVO_CONCLUIDA) and not _FALHA.fullmatch(n)]
        if desconhecidos:
            raise CheckpointCorrompido(f"{identidade.slot}: arquivos desconhecidos {desconhecidos}")
        for nome in nomes:
            if _FALHA.fullmatch(nome):
                registro = self._ler_registro(pasta / nome, identidade.slot, esperada)
                if registro["status"] != ESTADO_FALHOU:
                    raise CheckpointCorrompido(f"{identidade.slot}/{nome}: não é registro de falha")
        if ARQUIVO_CONCLUIDA in nomes:
            registro = self._ler_registro(pasta / ARQUIVO_CONCLUIDA, identidade.slot, esperada)
            if registro["status"] != ESTADO_CONCLUIDA:
                raise CheckpointCorrompido(f"{identidade.slot}: concluida.json sem status concluída")
            self._conferir_payload(pasta, registro, identidade.slot)
            return ESTADO_CONCLUIDA
        if ARQUIVO_PAYLOAD in nomes:
            raise CheckpointCorrompido(
                f"{identidade.slot}: payload sem registro de conclusão (gravação interrompida); "
                "inspecione o slot — nada é apagado nem reaproveitado")
        return ESTADO_FALHOU if nomes else ESTADO_AUSENTE

    def _ler_registro(self, caminho: Path, slot: str, esperada: Mapping[str, Any]) -> dict:
        registro = _validar_registro(_ler_json(caminho, rst.serializar), f"{slot}/{caminho.name}")
        diferentes = _diferencas(registro["identidade"], esperada)
        if diferentes:
            raise CheckpointIncompativel(
                f"{slot}/{caminho.name}: identidade gravada difere da esperada em {diferentes}")
        return registro

    @staticmethod
    def _conferir_payload(pasta: Path, registro: Mapping[str, Any], slot: str) -> None:
        caminho = pasta / ARQUIVO_PAYLOAD
        if not caminho.is_file():
            raise CheckpointCorrompido(f"{slot}: concluida.json sem payload")
        sha, _, tamanho = rst.calcular_sha256(caminho)
        if (sha, tamanho) != (registro["payload"]["sha256"], registro["payload"]["bytes"]):
            raise CheckpointCorrompido(f"{slot}: payload difere do registrado")

    # -- verificação do conjunto -----------------------------------------------

    def verificar(self, esperadas: Iterable[IdentidadeUnidade]) -> RelatorioRetomada:
        """Classifica as unidades esperadas; aborta se houver slot inválido ou inesperado."""
        esperadas = list(esperadas)
        slots = [i.slot for i in esperadas]
        if len(set(slots)) != len(slots):
            raise ValueError("unidades esperadas repetidas")
        self._exigir_contexto_inalterado()  # uma vez para o conjunto, antes de qualquer slot
        problemas: List[str] = []
        estados: Dict[str, List[str]] = {ESTADO_AUSENTE: [], ESTADO_FALHOU: [], ESTADO_CONCLUIDA: []}
        for identidade in esperadas:
            try:
                estados[self._estado(identidade)].append(identidade.slot)
            except (CheckpointCorrompido, CheckpointIncompativel) as exc:
                problemas.append(str(exc))
        pasta_atividade = self.diretorio / self.atividade
        if pasta_atividade.is_dir():
            inesperados = sorted(p.name for p in pasta_atividade.iterdir() if p.name not in set(slots))
            if inesperados:
                problemas.append(f"slots não esperados nesta atividade: {inesperados}")
        if problemas:
            raise CheckpointIncompativel("retomada abortada: " + " | ".join(problemas))
        return RelatorioRetomada(*(tuple(sorted(estados[e]))
                                   for e in (ESTADO_CONCLUIDA, ESTADO_AUSENTE, ESTADO_FALHOU)))

    # -- gravação --------------------------------------------------------------

    def _registro(self, identidade: IdentidadeUnidade, status: str, payload: Optional[dict],
                  resumo: Mapping[str, Any], erro: Optional[str], tentativa: int) -> dict:
        if not isinstance(resumo, Mapping):
            raise TypeError("resumo deve ser um mapeamento de agregados")
        rst._exigir_sem_dados_individuais(dict(resumo), "resumo")  # DadosIndividuais, antes de gravar
        corpo = {
            "formato": FORMATO, "versao_formato": VERSAO_FORMATO,
            "identidade": identidade.como_dict(), "sha256_identidade": identidade.sha256,
            "status": status, "payload": payload, "resumo": json.loads(rst.json_canonico(dict(resumo))),
            "erro": erro,
            "metadados": {"registrado_em": rst._instante(self._relogio_utc()), "tentativa": tentativa,
                          "head_sessao": self._head},
        }
        return _validar_registro(_com_hash(corpo), identidade.slot)

    def salvar(self, identidade: IdentidadeUnidade, payload: Mapping[str, Any],
               resumo: Optional[Mapping[str, Any]] = None) -> dict:
        """Grava o payload e, POR ÚLTIMO, o registro de conclusão.

        Recusa slot concluído (sem sobrescrita), slot inconsistente e
        código/entradas alterados desde a abertura da sessão. ``resumo`` só
        aceita agregados (passa pela varredura de dados individuais).
        """
        estado = self.estado(identidade)  # inclui a conferência de código e entradas
        if estado == ESTADO_CONCLUIDA:
            raise rst.SobrescritaRecusada(f"{identidade.slot} já está concluída")
        conteudo = _bytes_payload(payload)
        registro_payload = {"arquivo": ARQUIVO_PAYLOAD,
                            "sha256": hashlib.sha256(conteudo).hexdigest(), "bytes": len(conteudo)}
        tentativa = len(self._falhas(identidade)) + 1
        registro = self._registro(identidade, ESTADO_CONCLUIDA, registro_payload, resumo or {}, None, tentativa)
        pasta = self._pasta(identidade)
        pasta.mkdir(parents=True, exist_ok=True)
        _publicar(conteudo, pasta / ARQUIVO_PAYLOAD)
        _publicar(rst.serializar(registro), pasta / ARQUIVO_CONCLUIDA)
        return registro

    def registrar_falha(self, identidade: IdentidadeUnidade, erro: Union[BaseException, type, str]) -> dict:
        """Grava ``falha-<n>.json`` com o TIPO do erro (nunca a mensagem)."""
        estado = self.estado(identidade)
        if estado == ESTADO_CONCLUIDA:
            raise rst.SobrescritaRecusada(f"{identidade.slot} já está concluída")
        if isinstance(erro, BaseException):
            nome_erro = type(erro).__name__
        elif isinstance(erro, type) and issubclass(erro, BaseException):
            nome_erro = erro.__name__
        elif isinstance(erro, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,99}", erro):
            nome_erro = erro
        else:
            raise ValueError("erro deve ser exceção, classe de exceção ou nome de tipo")
        tentativa = len(self._falhas(identidade)) + 1
        registro = self._registro(identidade, ESTADO_FALHOU, None, {}, nome_erro, tentativa)
        pasta = self._pasta(identidade)
        pasta.mkdir(parents=True, exist_ok=True)
        _publicar(rst.serializar(registro), pasta / f"falha-{tentativa:03d}.json")
        return registro

    def _falhas(self, identidade: IdentidadeUnidade) -> List[str]:
        pasta = self._pasta(identidade)
        return sorted(p.name for p in pasta.iterdir() if _FALHA.fullmatch(p.name)) if pasta.is_dir() else []

    # -- retomada --------------------------------------------------------------

    def carregar(self, identidade: IdentidadeUnidade) -> dict:
        """Payload de uma unidade CONCLUÍDA e íntegra; senão, exceção explícita."""
        estado = self.estado(identidade)
        if estado == ESTADO_AUSENTE:
            raise CheckpointAusente(f"{identidade.slot}: sem checkpoint")
        if estado == ESTADO_FALHOU:
            raise CheckpointNaoConcluido(f"{identidade.slot}: só há registros de falha")
        caminho = self._pasta(identidade) / ARQUIVO_PAYLOAD
        payload = _ler_json(caminho, _bytes_payload)
        if not isinstance(payload, dict):
            raise CheckpointCorrompido(f"{identidade.slot}: payload não é objeto JSON")
        return payload
