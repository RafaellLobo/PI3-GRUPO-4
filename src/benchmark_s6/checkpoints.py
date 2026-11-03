"""Checkpoints atômicos por unidade do benchmark S6 (P9, Seções 6 e 14.5).

Reimplementa, para a S6, a sessão de checkpoints da S4
(``src/incremental/checkpoints.py``, ``SessaoCheckpoints``), que está presa à
configuração e às guardas da S4 (P9, Seção 14.2). É o ÚNICO módulo do
pacote que grava resultados de unidades.

**Identidade.** Um checkpoint só é reaproveitado se a identidade gravada for
idêntica à esperada. A identidade reúne:

* a unidade ``(r, k, modelo)``, o método de pontuação e o fingerprint da dobra;
* o modo, ``oficial``, o SHA-256 da configuração e o fingerprint do plano;
* as entradas: SHA-256 dos arquivos lidos (``DadosBenchmark.sha256_entradas``),
  partições, características, número de linhas, ``Amostra.sha256`` (ids,
  rótulos e grupos) e o SHA-256 da matriz ``X`` em ``float64``;
* o código: SHA-256 (CRLF → LF) de todos os ``.py`` de ``src/benchmark_s6`` e
  dos arquivos da S4 importados (``adaptadores_s4.ARQUIVOS_S4``);
* o ambiente: Python, plataforma, processador e versões das bibliotecas,
  inclusive o XGBoost (``None`` se ausente).

**Um diretório, uma identidade.** Na abertura, a sessão grava (ou confere)
``sessao.json`` com a identidade da sessão. Reabrir o mesmo diretório com
código, ambiente, entradas, configuração ou plano diferentes é recusado
(``CheckpointIncompativel``): corrigir o código invalida TODOS os checkpoints,
e a execução recomeça inteira noutro diretório (P9, Seção 6). Toda operação
recalcula os hashes do código e o ambiente; se mudaram desde a abertura,
nada é lido nem gravado.

Estrutura, num diretório local não versionável: fora deste repositório (e,
portanto, de ``data/``) e, se estiver dentro de outro repositório git, ignorado
por ele (P9, Seção 14.5; ``exigir_diretorio_local``)::

    <diretorio>/benchmark_s6/<modo>/
        sessao.json              identidade da sessão
        r00-k0-lr_fixa/
            predicoes.json       PRIVADO: PredicoesPrivadas (nodulo_id, pontuações)
            registro.json        RegistroUnidade agregado (passa pela varredura)
            concluida.json       identidade + hashes dos dois acima, gravado POR ÚLTIMO
            falha-001.json       tentativas que falharam (histórico, nunca resultado)
    <diretorio>/benchmark_s6/quarentena/<modo>/<unidade>/<instante>-<id>/
        recuperacao.json         resíduos de uma gravação interrompida, preservados

**Regras.**

* Cada arquivo é gravado num temporário do mesmo diretório, com ``fsync``, e
  publicado por substituição atômica (``os.replace``); um arquivo existente
  nunca é sobrescrito. Sem ``concluida.json`` válido, a unidade NUNCA está
  concluída.
* Na leitura, um slot está ``ausente``, ``falhou`` (só registros de falha; pode
  ser executado de novo), ``incompleta`` (gravação interrompida, ver abaixo) ou
  ``concluida``. Qualquer outra coisa aborta com exceção, e nada é apagado nem
  reaproveitado: arquivo desconhecido, resíduo junto de ``concluida.json``,
  JSON fora da forma canônica, hash divergente ou identidade diferente.
* **Gravação interrompida.** Sem ``concluida.json``, o slot é ``incompleta``
  só se os arquivos presentes forem, inequivocamente, um prefixo da
  transação deste módulo: ``predicoes.json`` e depois ``registro.json``, na
  ordem de publicação, mais no máximo UM temporário do próximo arquivo da
  sequência (ou de ``falha-<n+1>.json``, sem resultado publicado). Os ``.json``
  presentes têm de ser canônicos e desta unidade e desta sessão; senão,
  aborta. Uma unidade ``incompleta`` nunca é carregada: ``salvar`` e
  ``registrar_falha`` movem os resíduos para a quarentena e gravam a nova
  tentativa. A quarentena fica em ``<diretorio>/benchmark_s6/quarentena/
  <modo>/<unidade>/<instante>-<id>/``, com ``recuperacao.json`` (hashes dos
  arquivos movidos), num diretório novo a cada recuperação; nada é apagado
  nem sobrescrito.
* ``carregar`` reconstrói ``ResultadoUnidade`` e confere: hashes, forma
  canônica, ida e volta exata do JSON e coerência com o plano e a amostra da
  sessão (posições, ``nodulo_id``, rótulos, contagens, pontuações finitas).
* Falhas (``FalhaRegistrada``) nunca são carregadas como resultado.
* Persistência só em JSON canônico, sem ``pickle``. O registro agregado passa
  por uma varredura de identificadores da amostra e do padrão da
  configuração antes de ser gravado.
* Um único escritor: a sessão vive no processo principal. Workers só devolvem
  ``ResultadoUnidade`` ou ``FalhaRegistrada``; o processo principal grava.

Nenhuma métrica é calculada aqui.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl

from . import adaptadores_s4 as s4
from . import config as cfg
from . import folds
from . import modelos
from . import unidades as un
from .dados import DadosBenchmark

FORMATO = "checkpoint-benchmark-s6"
VERSAO_FORMATO = 1

ESTADO_AUSENTE = "ausente"
ESTADO_FALHOU = "falhou"
ESTADO_INCOMPLETA = "incompleta"
ESTADO_CONCLUIDA = un.ESTADO_CONCLUIDA

SUBDIRETORIO = "benchmark_s6"
ARQUIVO_SESSAO = "sessao.json"
ARQUIVO_PREDICOES = "predicoes.json"
ARQUIVO_REGISTRO = "registro.json"
ARQUIVO_CONCLUIDA = "concluida.json"
ARQUIVO_RECUPERACAO = "recuperacao.json"
QUARENTENA = "quarentena"
#: Ordem de publicação de uma conclusão (``salvar``).
_ORDEM_PUBLICACAO = (ARQUIVO_PREDICOES, ARQUIVO_REGISTRO, ARQUIVO_CONCLUIDA)
_FALHA = re.compile(r"falha-(\d{3})\.json")
_RESIDUO = re.compile(r"\..+\.tmp")
#: Temporário de ``_publicar`` (``tempfile.mkstemp``: prefixo, 8 caracteres aleatórios, sufixo). Outro
#: formato não é reconhecido como resíduo deste módulo e aborta (lado seguro).
_RESIDUO_TRANSACAO = re.compile(r"\.(predicoes\.json|registro\.json|concluida\.json|falha-(\d{3})\.json)"
                                r"\.[a-z0-9_]{8}\.tmp")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHAVES_CONTROLE = ("formato", "versao_formato", "identidade", "sha256_identidade", "estado", "arquivos",
                    "falha", "metadados", "sha256_conteudo")
_CHAVES_MARCA = ("formato", "versao_formato", "identidade_sessao", "sha256_identidade_sessao")
_ORDEM_CONTAGEM = ("nodulos", "pacientes", "classe_0", "classe_1")

PACOTE: Path = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Exceções
# --------------------------------------------------------------------------- #


class ErroCheckpoint(RuntimeError):
    """Base; ``chave_saida`` é a chave de ``EspecExecucao.codigos_saida`` (P9, Seção 14.3)."""

    chave_saida = "integridade"


class CheckpointCorrompido(ErroCheckpoint):
    """Slot ou arquivo que não pode ser interpretado: parcial, resíduo, hash ou forma inválidos."""


class CheckpointIncompativel(ErroCheckpoint):
    """Identidade gravada diferente da esperada, ou resultado de outra sessão."""


class SobrescritaRecusada(ErroCheckpoint):
    """Unidade já concluída ou arquivo já existente: checkpoints não são sobrescritos."""


class CheckpointAusente(ErroCheckpoint):
    """Nenhum checkpoint para a unidade."""

    chave_saida = "falha_tecnica"


class CheckpointNaoConcluido(ErroCheckpoint):
    """A unidade só tem registros de falha: precisa ser executada."""

    chave_saida = "falha_tecnica"


class DiretorioInvalido(ErroCheckpoint):
    """Diretório de checkpoints inexistente."""

    chave_saida = "falha_tecnica"


class DiretorioNaoLocal(ErroCheckpoint):
    """Diretório dentro do repositório ou de qualquer repositório git (P9, Seção 14.5)."""

    chave_saida = "privacidade_ou_vocabulario"


class DadosIndividuaisNoRegistro(ErroCheckpoint):
    """Identificador individual no registro agregado."""

    chave_saida = "privacidade_ou_vocabulario"


# --------------------------------------------------------------------------- #
# Identidade: código, ambiente e entradas
# --------------------------------------------------------------------------- #


def _codigo() -> Dict[str, str]:
    """SHA-256 (CRLF → LF) dos ``.py`` do pacote e dos arquivos da S4 importados (fixados)."""
    s4.verificar_arquivos_s4()
    arquivos = sorted(PACOTE.glob("*.py")) + [s4.REPO_ROOT / relativo for relativo in sorted(s4.ARQUIVOS_S4)]
    return {s4.caminho_relativo_repo(p): s4.calcular_sha256(p)[1] for p in arquivos}


def _ambiente() -> Dict[str, Any]:
    disponivel, versao_xgboost = modelos.xgboost_disponivel()
    return {
        "python": platform.python_version(), "implementacao": platform.python_implementation(),
        "sistema": platform.system(), "maquina": platform.machine(), "processador": platform.processor(),
        "versoes": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                    "pandas": pd.__version__, "joblib": joblib.__version__,
                    "threadpoolctl": threadpoolctl.__version__,
                    "xgboost": versao_xgboost if disponivel else None},
    }


def _entradas(dados: DadosBenchmark, contexto: un.ContextoExecucao) -> Dict[str, Any]:
    X = np.ascontiguousarray(contexto.X, dtype="<f8")
    return {"arquivos": {str(k): str(v) for k, v in dados.sha256_entradas.items()},
            "particoes": list(dados.particoes), "caracteristicas": list(dados.caracteristicas),
            "n_linhas": int(contexto.amostra.n), "sha256_amostra": contexto.amostra.sha256,
            "sha256_caracteristicas": hashlib.sha256(X.tobytes()).hexdigest()}


def _diferencas(gravada: Any, esperada: Dict[str, Any]) -> List[str]:
    if not isinstance(gravada, dict):
        return sorted(esperada)
    return sorted(k for k in set(gravada) | set(esperada) if gravada.get(k) != esperada.get(k))


# --------------------------------------------------------------------------- #
# JSON canônico e publicação atômica
# --------------------------------------------------------------------------- #


def _bytes_json(valor: Any) -> bytes:
    try:
        return (s4.json_canonico(valor) + "\n").encode("utf-8")
    except Exception as exc:  # ManifestoInvalido da S4: NaN, Infinity ou tipo não JSON
        raise ValueError(f"conteúdo não serializável em JSON canônico: {exc}") from exc


def _sem_duplicatas(pares: List[Tuple[str, Any]]) -> dict:
    objeto: dict = {}
    for chave, valor in pares:
        if chave in objeto:
            raise ValueError(f"chave duplicada {chave!r}")
        objeto[chave] = valor
    return objeto


def _constante_invalida(nome: str) -> Any:
    raise ValueError(f"constante {nome} não permitida")


def _json_de_bytes(bruto: bytes, onde: str) -> Any:
    """JSON estrito: UTF-8, sem chave duplicada, sem NaN/Infinity e nos bytes da forma canônica."""
    try:
        valor = json.loads(bruto.decode("utf-8"), object_pairs_hook=_sem_duplicatas,
                           parse_constant=_constante_invalida)
        canonico = _bytes_json(valor)
    except (UnicodeDecodeError, ValueError) as exc:
        raise CheckpointCorrompido(f"{onde}: ilegível: {exc}") from exc
    if canonico != bruto:
        raise CheckpointCorrompido(f"{onde}: bytes fora da forma canônica")
    return valor


def _ler_bytes(caminho: Path, onde: str) -> bytes:
    if caminho.is_symlink() or not caminho.is_file():
        raise CheckpointCorrompido(f"{onde}: não é arquivo regular")
    try:
        return caminho.read_bytes()
    except OSError as exc:
        raise CheckpointCorrompido(f"{onde}: ilegível: {exc}") from exc


def _hash_arquivo(conteudo: bytes) -> Dict[str, Any]:
    return {"sha256": hashlib.sha256(conteudo).hexdigest(), "bytes": len(conteudo)}


def _publicar(conteudo: bytes, destino: Path) -> None:
    """Temporário no mesmo diretório + ``fsync`` + ``os.replace``; nunca sobrescreve."""
    if os.path.lexists(destino):
        raise SobrescritaRecusada(f"{destino.parent.name}/{destino.name} já existe; "
                                  "checkpoints não são sobrescritos")
    descritor, temporario = tempfile.mkstemp(prefix=f".{destino.name}.", suffix=".tmp", dir=destino.parent)
    try:
        with os.fdopen(descritor, "wb") as arquivo:
            arquivo.write(conteudo)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        if os.path.lexists(destino):
            raise SobrescritaRecusada(f"{destino.parent.name}/{destino.name} apareceu durante a gravação")
        os.replace(temporario, destino)
    finally:
        if os.path.lexists(temporario):
            os.unlink(temporario)


def _instante() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Diretório local e varredura do registro agregado
# --------------------------------------------------------------------------- #


def _dentro(caminho: Path, raiz: Path) -> bool:
    c, r = os.path.normcase(str(caminho)), os.path.normcase(str(raiz))
    return c == r or c.startswith(r.rstrip("\\/") + os.sep)


def _dentro_de_repositorio(caminho: Path) -> bool:
    return any(os.path.lexists(p / ".git") for p in (caminho, *caminho.parents))


def _ignorado_pelo_git(caminho: Path) -> bool:
    """``git check-ignore`` no repositório que contém ``caminho``; falha do git nunca libera."""
    executavel = shutil.which("git")
    if executavel is None:
        raise DiretorioNaoLocal("git indisponível para conferir se o diretório de checkpoints é versionável")
    ambiente = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    try:
        processo = subprocess.run(
            [executavel, "--no-replace-objects", "-C", str(caminho), "check-ignore", "-q", "--no-index",
             str(caminho / ".sonda-checkpoint")], capture_output=True, env=ambiente, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DiretorioNaoLocal(f"falha ao executar git: {exc}") from exc
    if processo.returncode not in (0, 1):
        raise DiretorioNaoLocal("falha do git ao conferir se o diretório de checkpoints é versionável")
    return processo.returncode == 0


def exigir_diretorio_local(diretorio: Union[str, Path]) -> Path:
    """Diretório existente e não versionável (P9, Seção 14.5).

    * Dentro DESTE repositório: sempre recusado, mesmo em pasta ignorada
      (inclui ``data/``), porque o repositório é publicado.
    * Dentro de outro repositório git (por exemplo, um home versionado): só
      se aquele repositório o ignora. Falha do git recusa.
    """
    caminho = Path(diretorio)
    if not caminho.is_dir():
        raise DiretorioInvalido("diretório de checkpoints inexistente")
    resolvido = caminho.resolve()
    if _dentro(resolvido, s4.REPO_ROOT.resolve()):
        raise DiretorioNaoLocal("checkpoints não ficam dentro do repositório (nem em data/ nem em pastas ignoradas)")
    if _dentro_de_repositorio(resolvido) and not _ignorado_pelo_git(resolvido):
        raise DiretorioNaoLocal("o diretório de checkpoints está num repositório git e não é ignorado "
                                "(versionável)")
    return resolvido


def _textos(valor: Any) -> Iterable[str]:
    if isinstance(valor, str):
        yield valor
    elif isinstance(valor, dict):
        for chave, item in valor.items():
            yield from _textos(chave)
            yield from _textos(item)
    elif isinstance(valor, (list, tuple)):
        for item in valor:
            yield from _textos(item)


def _exigir_sem_identificadores(valor: Any, amostra: s4.Amostra, onde: str) -> None:
    """Nenhum ``nodulo_id``, ``patient_id`` da amostra nem texto no padrão de varredura da configuração."""
    padrao = re.compile(cfg.CONFIGURACAO.saidas.padrao_varredura)
    proibidos = set(amostra.ids.tolist()) | set(amostra.grupos.tolist())
    for texto in _textos(valor):
        if padrao.search(texto) or any(p in texto for p in proibidos):
            raise DadosIndividuaisNoRegistro(f"{onde}: identificador individual no registro agregado")


# --------------------------------------------------------------------------- #
# Reconstrução dos resultados
# --------------------------------------------------------------------------- #


def _registro_de_dict(d: Any) -> un.RegistroUnidade:
    unidade = un.Unidade(d["repeticao"], d["dobra_externa"], d["modelo"])
    contagem = {}
    for nome in ("contagem_treino", "contagem_avaliacao"):
        if set(d[nome]) != set(_ORDEM_CONTAGEM):
            raise ValueError(f"{nome} com chaves inválidas")
        contagem[nome] = tuple((chave, d[nome][chave]) for chave in _ORDEM_CONTAGEM)
    return un.RegistroUnidade(
        unidade=unidade, modo=d["modo"], oficial=d["oficial"], sha256_configuracao=d["sha256_configuracao"],
        fingerprint_plano=d["fingerprint_plano"], fingerprint_dobra=d["fingerprint_dobra"],
        metodo_pontuacao=d["metodo_pontuacao"], configuracao=tuple(sorted(d["configuracao"].items())),
        indice_configuracao=d["indice_configuracao"], n_configuracoes=d["n_configuracoes"],
        borda=tuple((nome, b["valor"], b["posicao"], b["n_valores"], b["na_borda"])
                    for nome, b in sorted(d["borda"].items())),
        n_ajustes=d["n_ajustes"],
        avisos=tuple((a["fase"], a["categoria"], a["mensagem"], a["ocorrencias"]) for a in d["avisos"]),
        **contagem)


def _somente_leitura(valores: Any, dtype: type) -> np.ndarray:
    arr = np.array(valores, dtype=dtype)
    arr.setflags(write=False)
    return arr


def _privado_de_dict(d: Any) -> un.PredicoesPrivadas:
    partes = d["unidade"].split("-", 2)
    if len(partes) != 3:
        raise ValueError("rótulo de unidade inválido")
    unidade = un.Unidade(int(partes[0][1:]), int(partes[1][1:]), partes[2])
    return un.PredicoesPrivadas(
        unidade=unidade, fingerprint_dobra=d["fingerprint_dobra"], metodo_pontuacao=d["metodo_pontuacao"],
        posicoes=_somente_leitura(d["posicoes"], np.int64), nodulo_ids=tuple(d["nodulo_ids"]),
        y=_somente_leitura(d["y"], np.int64), pontuacoes=_somente_leitura(d["pontuacoes"], np.float64),
        medias_internas=tuple(float(m) for m in d["medias_internas"]))


def _contagem(amostra: s4.Amostra, posicoes: np.ndarray) -> Tuple[Tuple[str, int], ...]:
    y = amostra.y[posicoes]
    valores = {"nodulos": int(len(posicoes)), "pacientes": len(set(amostra.grupos[posicoes].tolist())),
               "classe_0": int(np.sum(y == 0)), "classe_1": int(np.sum(y == 1))}
    return tuple((chave, valores[chave]) for chave in _ORDEM_CONTAGEM)


# --------------------------------------------------------------------------- #
# Relatório de retomada
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RelatorioRetomada:
    """Estado das unidades esperadas (rótulos). Não declara a execução concluída nem calcula métricas."""

    concluidas: Tuple[str, ...]
    ausentes: Tuple[str, ...]
    falhas: Tuple[str, ...]
    #: Gravação interrompida antes de ``concluida.json``: será recuperada e executada de novo.
    incompletas: Tuple[str, ...] = ()

    @property
    def a_executar(self) -> Tuple[str, ...]:
        return tuple(sorted(self.ausentes + self.falhas + self.incompletas))

    @property
    def n_esperadas(self) -> int:
        return len(self.concluidas) + len(self.ausentes) + len(self.falhas) + len(self.incompletas)


# --------------------------------------------------------------------------- #
# Sessão
# --------------------------------------------------------------------------- #


class SessaoCheckpoints:
    """Checkpoints de UMA carga num diretório local. Só no processo principal (um único escritor).

    Monta o próprio ``ContextoExecucao`` a partir de ``dados`` (``contexto``),
    de modo que identidade, plano e dados vêm da mesma carga.
    """

    def __init__(self, diretorio: Union[str, Path], dados: DadosBenchmark) -> None:
        if not isinstance(dados, DadosBenchmark):
            raise TypeError("dados deve ser DadosBenchmark")
        raiz = exigir_diretorio_local(diretorio)
        self.contexto = un.ContextoExecucao.de_dados(dados)
        self.diretorio = raiz / SUBDIRETORIO / self.contexto.modo
        self._sessao_json = s4.json_canonico({
            "modo": self.contexto.modo, "oficial": self.contexto.oficial,
            "sha256_configuracao": self.contexto.sha256_configuracao,
            "fingerprint_plano": self.contexto.plano.fingerprint,
            "entradas": _entradas(dados, self.contexto), "codigo": _codigo(), "ambiente": _ambiente(),
        })
        self.diretorio.mkdir(parents=True, exist_ok=True)
        self._abrir_marca()

    @property
    def identidade_sessao(self) -> Dict[str, Any]:
        return json.loads(self._sessao_json)

    # -- sessão ----------------------------------------------------------------

    def _marca(self) -> Dict[str, Any]:
        identidade = self.identidade_sessao
        return {"formato": FORMATO, "versao_formato": VERSAO_FORMATO, "identidade_sessao": identidade,
                "sha256_identidade_sessao": s4.sha256_canonico(identidade)}

    def _abrir_marca(self) -> None:
        caminho = self.diretorio / ARQUIVO_SESSAO
        if not os.path.lexists(caminho):
            if any(self.diretorio.iterdir()):
                raise CheckpointCorrompido(
                    f"{self.contexto.modo}: conteúdo sem {ARQUIVO_SESSAO}; nada é reaproveitado")
            _publicar(_bytes_json(self._marca()), caminho)
        self._conferir_marca()

    def _conferir_marca(self) -> None:
        caminho = self.diretorio / ARQUIVO_SESSAO
        marca = _json_de_bytes(_ler_bytes(caminho, ARQUIVO_SESSAO), ARQUIVO_SESSAO)
        if not isinstance(marca, dict) or set(marca) != set(_CHAVES_MARCA) or marca["formato"] != FORMATO \
                or marca["versao_formato"] != VERSAO_FORMATO \
                or marca["sha256_identidade_sessao"] != s4.sha256_canonico(marca["identidade_sessao"]):
            raise CheckpointCorrompido(f"{ARQUIVO_SESSAO}: forma ou hash inválidos")
        diferentes = _diferencas(marca["identidade_sessao"], self.identidade_sessao)
        if diferentes:
            raise CheckpointIncompativel(
                f"o diretório pertence a outra identidade (difere em {diferentes}); os checkpoints não valem "
                "para esta execução. Use um diretório novo; nada é apagado")

    def _exigir_inalterada(self) -> None:
        """Código e ambiente recalculados: mudança desde a abertura impede ler ou gravar."""
        atual = self.identidade_sessao
        for chave, valor in (("codigo", _codigo()), ("ambiente", _ambiente())):
            if json.loads(s4.json_canonico(valor)) != atual[chave]:
                raise CheckpointIncompativel(f"{chave} alterado desde a abertura da sessão; nada é reaproveitado")

    # -- identidade da unidade ---------------------------------------------------

    def _dobra(self, unidade: un.Unidade) -> folds.DobraExterna:
        if not isinstance(unidade, un.Unidade):
            raise TypeError("unidade deve ser Unidade")
        try:
            return folds.dobra_para_modelo(self.contexto.plano, unidade.repeticao, unidade.dobra, unidade.modelo)
        except folds.DobrasInvalidas as exc:
            raise CheckpointIncompativel(f"{unidade.rotulo}: unidade fora do plano {self.contexto.modo}") from exc

    def identidade(self, unidade: un.Unidade) -> Dict[str, Any]:
        """Identidade completa da unidade nesta sessão (JSON)."""
        dobra = self._dobra(unidade)
        return {**self.identidade_sessao, "unidade": unidade.rotulo, "repeticao": unidade.repeticao,
                "dobra_externa": unidade.dobra, "modelo": unidade.modelo, "fingerprint_dobra": dobra.fingerprint,
                "metodo_pontuacao": modelos.especificacao(unidade.modelo).pontuacao}

    def _pasta(self, unidade: un.Unidade) -> Path:
        return self.diretorio / unidade.rotulo

    # -- registros de controle -----------------------------------------------------

    def _controle(self, unidade: un.Unidade, estado: str, arquivos: Optional[dict], falha: Optional[dict],
                  tentativa: int) -> Dict[str, Any]:
        identidade = self.identidade(unidade)
        corpo = {"formato": FORMATO, "versao_formato": VERSAO_FORMATO, "identidade": identidade,
                 "sha256_identidade": s4.sha256_canonico(identidade), "estado": estado, "arquivos": arquivos,
                 "falha": falha, "metadados": {"registrado_em": _instante(), "tentativa": tentativa}}
        return {**corpo, "sha256_conteudo": s4.sha256_canonico(corpo)}

    def _ler_controle(self, caminho: Path, unidade: un.Unidade) -> Dict[str, Any]:
        onde = f"{unidade.rotulo}/{caminho.name}"
        registro = _json_de_bytes(_ler_bytes(caminho, onde), onde)
        if not isinstance(registro, dict) or set(registro) != set(_CHAVES_CONTROLE):
            raise CheckpointCorrompido(f"{onde}: chaves inválidas")
        if registro["formato"] != FORMATO or registro["versao_formato"] != VERSAO_FORMATO:
            raise CheckpointCorrompido(f"{onde}: formato desconhecido")
        corpo = {k: v for k, v in registro.items() if k != "sha256_conteudo"}
        if registro["sha256_conteudo"] != s4.sha256_canonico(corpo):
            raise CheckpointCorrompido(f"{onde}: sha256_conteudo não confere")
        if registro["sha256_identidade"] != s4.sha256_canonico(registro["identidade"]):
            raise CheckpointCorrompido(f"{onde}: sha256_identidade não confere")
        metadados = registro["metadados"]
        if not (isinstance(metadados, dict) and set(metadados) == {"registrado_em", "tentativa"}
                and type(metadados["tentativa"]) is int and metadados["tentativa"] >= 1):
            raise CheckpointCorrompido(f"{onde}: metadados inválidos")
        estado, arquivos, falha = registro["estado"], registro["arquivos"], registro["falha"]
        if estado == ESTADO_CONCLUIDA:
            if falha is not None or not isinstance(arquivos, dict) \
                    or set(arquivos) != {ARQUIVO_PREDICOES, ARQUIVO_REGISTRO} \
                    or not all(isinstance(h, dict) and set(h) == {"sha256", "bytes"}
                               and isinstance(h["sha256"], str) and _SHA256.fullmatch(h["sha256"])
                               and type(h["bytes"]) is int for h in arquivos.values()):
                raise CheckpointCorrompido(f"{onde}: registro de conclusão sem arquivos coerentes")
        elif estado == ESTADO_FALHOU:
            chaves = set(cfg.CONFIGURACAO.execucao.codigos_saida) - {"concluido"}
            if arquivos is not None or not (isinstance(falha, dict)
                                            and set(falha) == {"chave_saida", "categoria", "mensagem"}
                                            and all(isinstance(v, str) for v in falha.values())
                                            and falha["chave_saida"] in chaves):
                raise CheckpointCorrompido(f"{onde}: registro de falha incoerente")
        else:
            raise CheckpointCorrompido(f"{onde}: estado desconhecido")
        diferentes = _diferencas(registro["identidade"], self.identidade(unidade))
        if diferentes:
            raise CheckpointIncompativel(f"{onde}: identidade gravada difere da esperada em {diferentes}")
        return registro

    # -- estado ----------------------------------------------------------------------

    def estado(self, unidade: un.Unidade) -> str:
        """``ausente``, ``falhou``, ``incompleta`` ou ``concluida``; outra situação levanta exceção."""
        self._exigir_inalterada()
        return self._estado(unidade)[0]

    def _estado(self, unidade: un.Unidade) -> Tuple[str, Optional[un.ResultadoUnidade], int]:
        """``(estado, resultado reconstruído, número de falhas)``; quem chama já conferiu a sessão.

        ``concluida`` exige a reconstrução COMPLETA (hashes, forma canônica,
        ida e volta e coerência com a sessão), não só a presença do controle.
        """
        self.identidade(unidade)  # unidade do plano desta sessão
        pasta = self._pasta(unidade)
        if not os.path.lexists(pasta):
            return ESTADO_AUSENTE, None, 0
        if pasta.is_symlink() or not pasta.is_dir():
            raise CheckpointCorrompido(f"{unidade.rotulo}: não é diretório")
        nomes = sorted(p.name for p in pasta.iterdir())
        residuos = [n for n in nomes if _RESIDUO.fullmatch(n)]
        desconhecidos = [n for n in nomes if n not in _ORDEM_PUBLICACAO and not _FALHA.fullmatch(n)
                         and n not in residuos]
        if desconhecidos:
            raise CheckpointCorrompido(f"{unidade.rotulo}: arquivos desconhecidos {desconhecidos}")
        falhas = [n for n in nomes if _FALHA.fullmatch(n)]
        for i, nome in enumerate(falhas, start=1):
            registro = self._ler_controle(pasta / nome, unidade)
            if registro["estado"] != ESTADO_FALHOU or nome != f"falha-{i:03d}.json" \
                    or registro["metadados"]["tentativa"] != i:
                raise CheckpointCorrompido(f"{unidade.rotulo}/{nome}: registro de falha fora de sequência")
        if ARQUIVO_CONCLUIDA in nomes:
            if residuos:
                raise CheckpointCorrompido(f"{unidade.rotulo}: resíduo {residuos} junto de {ARQUIVO_CONCLUIDA}; "
                                           "inspecione o slot — nada é apagado nem reaproveitado")
            controle = self._ler_controle(pasta / ARQUIVO_CONCLUIDA, unidade)
            if controle["estado"] != ESTADO_CONCLUIDA or controle["metadados"]["tentativa"] != len(falhas) + 1:
                raise CheckpointCorrompido(f"{unidade.rotulo}: {ARQUIVO_CONCLUIDA} incoerente")
            return ESTADO_CONCLUIDA, self._reconstruir(unidade, controle), len(falhas)
        if residuos or ARQUIVO_PREDICOES in nomes or ARQUIVO_REGISTRO in nomes:
            self._transacao_incompleta(unidade, len(falhas))
            return ESTADO_INCOMPLETA, None, len(falhas)
        return (ESTADO_FALHOU if falhas else ESTADO_AUSENTE), None, len(falhas)

    # -- gravação interrompida -------------------------------------------------------

    def _transacao_incompleta(self, unidade: un.Unidade, n_falhas: int) -> List[str]:
        """Arquivos de uma gravação interrompida DESTE módulo, na ordem segura de remoção; senão, exceção.

        Mover na ordem devolvida (resíduo, depois publicados de trás para a
        frente) mantém cada estado intermediário reconhecível como incompleto.
        """
        pasta = self._pasta(unidade)
        nomes = sorted(p.name for p in pasta.iterdir())
        residuos = [n for n in nomes if _RESIDUO.fullmatch(n)]
        publicados = [n for n in _ORDEM_PUBLICACAO if n in nomes]
        if ARQUIVO_CONCLUIDA in publicados or publicados != list(_ORDEM_PUBLICACAO[:len(publicados)]):
            raise CheckpointCorrompido(f"{unidade.rotulo}: {publicados} não é uma gravação interrompida deste "
                                       "módulo; inspecione o slot — nada é apagado nem reaproveitado")
        if len(residuos) > 1:
            raise CheckpointCorrompido(f"{unidade.rotulo}: mais de um resíduo de gravação {residuos}")
        if residuos:
            encontrado = _RESIDUO_TRANSACAO.fullmatch(residuos[0])
            if encontrado is None:
                raise CheckpointCorrompido(f"{unidade.rotulo}: resíduo desconhecido {residuos}")
            if encontrado.group(2) is not None:
                na_sequencia = not publicados and int(encontrado.group(2)) == n_falhas + 1
            else:
                na_sequencia = _ORDEM_PUBLICACAO.index(encontrado.group(1)) == len(publicados)
            if not na_sequencia:
                raise CheckpointCorrompido(f"{unidade.rotulo}: resíduo {residuos[0]} fora da sequência de gravação")
            caminho = pasta / residuos[0]
            if caminho.is_symlink() or not caminho.is_file():
                raise CheckpointCorrompido(f"{unidade.rotulo}: resíduo {residuos[0]} não é arquivo regular")
        for nome in publicados:
            self._exigir_da_unidade(pasta / nome, unidade)
        return residuos + publicados[::-1]

    def _exigir_da_unidade(self, caminho: Path, unidade: un.Unidade) -> None:
        """Arquivo publicado (completo) desta unidade e desta sessão; senão, exceção."""
        onde = f"{unidade.rotulo}/{caminho.name}"
        bruto = _ler_bytes(caminho, onde)
        valor = _json_de_bytes(bruto, onde)
        try:
            objeto = (_privado_de_dict if caminho.name == ARQUIVO_PREDICOES else _registro_de_dict)(valor)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise CheckpointCorrompido(f"{onde}: conteúdo não reconstrói o resultado: {exc}") from exc
        if _bytes_json(objeto.como_dict()) != bruto:
            raise CheckpointCorrompido(f"{onde}: reconstrução não reproduz o arquivo gravado")
        ctx, dobra = self.contexto, self._dobra(unidade)
        esperado = (unidade, dobra.fingerprint, modelos.especificacao(unidade.modelo).pontuacao)
        if (objeto.unidade, objeto.fingerprint_dobra, objeto.metodo_pontuacao) != esperado:
            raise CheckpointIncompativel(f"{onde}: arquivo de outra unidade ou de outra dobra")
        if isinstance(objeto, un.RegistroUnidade) and \
                (objeto.modo, objeto.oficial, objeto.sha256_configuracao, objeto.fingerprint_plano) != \
                (ctx.modo, ctx.oficial, ctx.sha256_configuracao, ctx.plano.fingerprint):
            raise CheckpointIncompativel(f"{onde}: registro de outra sessão (modo, oficial, configuração ou plano)")

    def _recuperar(self, unidade: un.Unidade, n_falhas: int) -> Path:
        """Move os resíduos de uma gravação interrompida para um diretório NOVO da quarentena."""
        pasta = self._pasta(unidade)
        arquivos = self._transacao_incompleta(unidade, n_falhas)
        instante = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destino = self.diretorio.parent / QUARENTENA / self.contexto.modo / unidade.rotulo / \
            f"{instante}-{uuid.uuid4().hex[:12]}"
        destino.mkdir(parents=True, exist_ok=False)
        recuperacao = {"formato": FORMATO, "versao_formato": VERSAO_FORMATO, "unidade": unidade.rotulo,
                       "sha256_identidade": s4.sha256_canonico(self.identidade(unidade)),
                       "motivo": f"gravação interrompida antes de {ARQUIVO_CONCLUIDA}", "registrado_em": _instante(),
                       "arquivos": {n: _hash_arquivo(_ler_bytes(pasta / n, n)) for n in arquivos}}
        _publicar(_bytes_json(recuperacao), destino / ARQUIVO_RECUPERACAO)
        for nome in arquivos:
            if os.path.lexists(destino / nome):
                raise SobrescritaRecusada(f"quarentena: {nome} já existe")
            os.replace(pasta / nome, destino / nome)
        estado = self._estado(unidade)[0]
        if estado not in (ESTADO_AUSENTE, ESTADO_FALHOU):
            raise CheckpointCorrompido(f"{unidade.rotulo}: estado {estado} depois da recuperação")
        return destino

    def _preparar_gravacao(self, unidade: un.Unidade, estado: str, n_falhas: int) -> None:
        if estado == ESTADO_CONCLUIDA:
            raise SobrescritaRecusada(f"{unidade.rotulo} já está concluída")
        if estado == ESTADO_INCOMPLETA:
            self._recuperar(unidade, n_falhas)

    @staticmethod
    def _ler_conferido(caminho: Path, controle: Dict[str, Any], unidade: un.Unidade) -> bytes:
        onde = f"{unidade.rotulo}/{caminho.name}"
        if not os.path.lexists(caminho):
            raise CheckpointCorrompido(f"{onde}: ausente com {ARQUIVO_CONCLUIDA} presente")
        conteudo = _ler_bytes(caminho, onde)
        if _hash_arquivo(conteudo) != controle["arquivos"][caminho.name]:
            raise CheckpointCorrompido(f"{onde}: hash ou tamanho difere do registrado")
        return conteudo

    # -- conferência do resultado contra a sessão --------------------------------------

    def _conferir_resultado(self, resultado: un.ResultadoUnidade, erro: Type[ErroCheckpoint]) -> None:
        registro, privado = resultado.registro, resultado.privado
        unidade = registro.unidade
        dobra = self._dobra(unidade)
        espec = modelos.especificacao(unidade.modelo)
        ctx, amostra = self.contexto, self.contexto.amostra
        avaliacao = dobra.externa.avaliacao

        def exigir(condicao: bool, mensagem: str) -> None:
            if not condicao:
                raise erro(f"{unidade.rotulo}: {mensagem}")

        exigir((registro.modo, registro.oficial, registro.sha256_configuracao, registro.fingerprint_plano)
               == (ctx.modo, ctx.oficial, ctx.sha256_configuracao, ctx.plano.fingerprint),
               "registro de outra sessão (modo, oficial, configuração ou plano)")
        exigir(privado.unidade == unidade, "predições de outra unidade")
        exigir(registro.fingerprint_dobra == privado.fingerprint_dobra == dobra.fingerprint, "dobra divergente")
        exigir(registro.metodo_pontuacao == privado.metodo_pontuacao == espec.pontuacao, "método de pontuação")
        exigir(registro.n_configuracoes == espec.n_configuracoes
               and type(registro.indice_configuracao) is int
               and 0 <= registro.indice_configuracao < registro.n_configuracoes, "posição na grade")
        exigir(registro.contagem_treino == _contagem(amostra, dobra.externa.treino)
               and registro.contagem_avaliacao == _contagem(amostra, avaliacao), "contagens divergentes")
        posicoes = np.asarray(privado.posicoes)
        exigir(posicoes.dtype == np.int64 and np.array_equal(posicoes, avaliacao), "posições da avaliação")
        exigir(privado.nodulo_ids == tuple(str(i) for i in amostra.ids[avaliacao]), "nodulo_id divergente")
        y = np.asarray(privado.y)
        exigir(y.dtype == np.int64 and np.array_equal(y, amostra.y[avaliacao]), "rótulos divergentes")
        pontuacoes = np.asarray(privado.pontuacoes)
        exigir(pontuacoes.dtype == np.float64 and pontuacoes.shape == (len(avaliacao),)
               and bool(np.all(np.isfinite(pontuacoes))), "pontuações com forma, tipo ou valor inválidos")
        if espec.pontuacao == "predict_proba":
            exigir(bool(np.all((pontuacoes >= 0.0) & (pontuacoes <= 1.0))), "probabilidade fora de [0, 1]")
        medias = privado.medias_internas
        exigir(len(medias) == (espec.n_configuracoes if espec.grade else 0)
               and all(type(m) is float and np.isfinite(m) for m in medias), "médias internas inválidas")

    # -- gravação ----------------------------------------------------------------------

    def salvar(self, resultado: un.ResultadoUnidade) -> Dict[str, Any]:
        """Grava predições e registro e, POR ÚLTIMO, ``concluida.json``. Recusa unidade já concluída.

        Unidade ``incompleta`` tem os resíduos movidos para a quarentena antes da gravação.
        """
        if not isinstance(resultado, un.ResultadoUnidade):
            raise TypeError("resultado deve ser ResultadoUnidade; falhas vão para registrar_falha")
        unidade = resultado.unidade
        self._exigir_inalterada()
        estado, _, n_falhas = self._estado(unidade)
        if estado == ESTADO_CONCLUIDA:
            raise SobrescritaRecusada(f"{unidade.rotulo} já está concluída")
        self._conferir_resultado(resultado, CheckpointIncompativel)
        registro_dict = resultado.registro.como_dict()
        _exigir_sem_identificadores(registro_dict, self.contexto.amostra, f"{unidade.rotulo}/{ARQUIVO_REGISTRO}")
        self._preparar_gravacao(unidade, estado, n_falhas)  # depois das conferências: recusa não move nada
        conteudos = {ARQUIVO_PREDICOES: _bytes_json(resultado.privado.como_dict()),
                     ARQUIVO_REGISTRO: _bytes_json(registro_dict)}
        controle = self._controle(unidade, ESTADO_CONCLUIDA,
                                  {nome: _hash_arquivo(c) for nome, c in conteudos.items()}, None, n_falhas + 1)
        pasta = self._pasta(unidade)
        pasta.mkdir(exist_ok=True)
        for nome, conteudo in conteudos.items():
            _publicar(conteudo, pasta / nome)
        _publicar(_bytes_json(controle), pasta / ARQUIVO_CONCLUIDA)
        return controle

    def registrar_falha(self, falha: un.FalhaRegistrada) -> Dict[str, Any]:
        """Grava ``falha-<n>.json`` (local). A unidade continua a executar; nunca vira resultado."""
        if not isinstance(falha, un.FalhaRegistrada):
            raise TypeError("falha deve ser FalhaRegistrada")
        unidade = falha.unidade
        self._exigir_inalterada()
        estado, _, n_falhas = self._estado(unidade)
        self._preparar_gravacao(unidade, estado, n_falhas)
        causa = {"chave_saida": falha.chave_saida, "categoria": falha.categoria, "mensagem": falha.mensagem}
        controle = self._controle(unidade, ESTADO_FALHOU, None, causa, n_falhas + 1)
        pasta = self._pasta(unidade)
        pasta.mkdir(exist_ok=True)
        _publicar(_bytes_json(controle), pasta / f"falha-{n_falhas + 1:03d}.json")
        return controle

    # -- leitura -----------------------------------------------------------------------

    def carregar(self, unidade: un.Unidade) -> un.ResultadoUnidade:
        """``ResultadoUnidade`` de uma unidade CONCLUÍDA e íntegra; senão, exceção explícita."""
        self._exigir_inalterada()
        estado, resultado, _ = self._estado(unidade)
        if estado == ESTADO_AUSENTE:
            raise CheckpointAusente(f"{unidade.rotulo}: sem checkpoint")
        if estado == ESTADO_FALHOU:
            raise CheckpointNaoConcluido(f"{unidade.rotulo}: só há registros de falha")
        if estado == ESTADO_INCOMPLETA:
            raise CheckpointNaoConcluido(f"{unidade.rotulo}: gravação interrompida; a unidade será executada "
                                         "de novo")
        return resultado

    def _reconstruir(self, unidade: un.Unidade, controle: Dict[str, Any]) -> un.ResultadoUnidade:
        pasta = self._pasta(unidade)
        brutos = {nome: self._ler_conferido(pasta / nome, controle, unidade)
                  for nome in (ARQUIVO_PREDICOES, ARQUIVO_REGISTRO)}
        try:
            privado = _privado_de_dict(_json_de_bytes(brutos[ARQUIVO_PREDICOES], unidade.rotulo))
            registro = _registro_de_dict(_json_de_bytes(brutos[ARQUIVO_REGISTRO], unidade.rotulo))
        except CheckpointCorrompido:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise CheckpointCorrompido(f"{unidade.rotulo}: conteúdo não reconstrói o resultado: {exc}") from exc
        if registro.unidade != unidade:
            raise CheckpointCorrompido(f"{unidade.rotulo}: registro de outra unidade")
        # Ida e volta exata: o objeto reconstruído serializa nos MESMOS bytes gravados.
        if _bytes_json(privado.como_dict()) != brutos[ARQUIVO_PREDICOES] \
                or _bytes_json(registro.como_dict()) != brutos[ARQUIVO_REGISTRO]:
            raise CheckpointCorrompido(f"{unidade.rotulo}: reconstrução não reproduz os arquivos gravados")
        resultado = un.ResultadoUnidade(registro, privado)
        self._conferir_resultado(resultado, CheckpointCorrompido)
        return resultado

    def verificar(self, esperadas: Optional[Iterable[un.Unidade]] = None) -> RelatorioRetomada:
        """Classifica as unidades esperadas (padrão: todas do modo); aborta se houver slot inválido ou inesperado.

        Só lê: unidades ``incompleta`` são relatadas, não recuperadas.
        """
        esperadas = list(self.contexto.unidades() if esperadas is None else esperadas)
        rotulos = [u.rotulo for u in esperadas]
        if len(set(rotulos)) != len(rotulos):
            raise ValueError("unidades esperadas repetidas")
        self._exigir_inalterada()
        self._conferir_marca()
        problemas: List[str] = []
        estados: Dict[str, List[str]] = {ESTADO_AUSENTE: [], ESTADO_FALHOU: [], ESTADO_INCOMPLETA: [],
                                         ESTADO_CONCLUIDA: []}
        for unidade in esperadas:
            try:
                estados[self._estado(unidade)[0]].append(unidade.rotulo)
            except (CheckpointCorrompido, CheckpointIncompativel) as exc:
                problemas.append(str(exc))
        inesperados = sorted(p.name for p in self.diretorio.iterdir()
                             if p.name != ARQUIVO_SESSAO and p.name not in set(rotulos))
        if inesperados:
            problemas.append(f"entradas não esperadas: {inesperados}")
        if problemas:
            raise CheckpointCorrompido("retomada abortada: " + " | ".join(problemas))
        return RelatorioRetomada(*(tuple(estados[e]) for e in (ESTADO_CONCLUIDA, ESTADO_AUSENTE, ESTADO_FALHOU,
                                                               ESTADO_INCOMPLETA)))


def obter_unidade(sessao: SessaoCheckpoints, unidade: un.Unidade
                  ) -> Tuple[Union[un.ResultadoUnidade, un.FalhaRegistrada], bool]:
    """Recupera a unidade concluída sem treinar; senão executa e grava. Devolve ``(resultado, reaproveitado)``.

    Execução no processo atual (sem paralelismo). Falha é gravada como
    ``falha-<n>.json`` e devolvida; uma nova chamada executa de novo. Unidade
    ``incompleta`` é executada de novo, e os resíduos vão para a quarentena.
    """
    if not isinstance(sessao, SessaoCheckpoints):
        raise TypeError("sessao deve ser SessaoCheckpoints")
    sessao._exigir_inalterada()
    estado, carregado, _ = sessao._estado(unidade)
    if estado == ESTADO_CONCLUIDA:
        return carregado, True
    resultado = un.executar_tarefa(sessao.contexto, sessao.contexto.tarefa(unidade))
    if isinstance(resultado, un.ResultadoUnidade):
        sessao.salvar(resultado)
    else:
        sessao.registrar_falha(resultado)
    return resultado, False


__all__ = [
    "ARQUIVO_CONCLUIDA", "ARQUIVO_PREDICOES", "ARQUIVO_RECUPERACAO", "ARQUIVO_REGISTRO", "ARQUIVO_SESSAO",
    "CheckpointAusente", "CheckpointCorrompido", "CheckpointIncompativel", "CheckpointNaoConcluido",
    "DadosIndividuaisNoRegistro", "DiretorioInvalido", "DiretorioNaoLocal", "ErroCheckpoint", "ESTADO_AUSENTE",
    "ESTADO_CONCLUIDA", "ESTADO_FALHOU", "ESTADO_INCOMPLETA", "exigir_diretorio_local", "obter_unidade",
    "QUARENTENA", "RelatorioRetomada", "SessaoCheckpoints", "SobrescritaRecusada",
]
