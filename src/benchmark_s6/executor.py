"""Executor das unidades do benchmark S6, com 1 ou N processos (P9, Seções 6 e 14.3).

Executa as unidades ``(r, k, modelo)`` pendentes de uma ``SessaoCheckpoints``
e grava cada resultado assim que chega. Não calcula métricas, não gera
relatórios e não é o coordenador.

**Processos.**

* ``processos=1``: as unidades rodam no processo principal, com o
  ``ContextoExecucao`` da própria sessão.
* ``processos=N``: ``ProcessPoolExecutor`` com contexto ``spawn`` (o único do
  Windows). Cada worker roda ``carregador()`` UMA vez, no inicializador, e
  monta o próprio ``ContextoExecucao``; nada do plano nem dos dados atravessa
  processos. Só ``TarefaUnidade`` (primitivos) vai para o worker, e só
  ``ResultadoUnidade`` ou ``FalhaRegistrada`` volta.
* Em qualquer caso, cada estimador usa ``n_jobs=1`` (``modelos``) e o BLAS
  fica limitado a ``threads_blas`` (1) por ``threadpoolctl``: no worker,
  durante toda a vida do processo; no ajuste e na predição, de novo por
  chamada (``tuning`` e ``unidades``). O resultado não depende de
  ``processos`` (P9, Seção 14.3).

**Conferências.**

* No worker, ``unidades.executar_tarefa`` confere modo, configuração,
  amostra, ``X`` (SHA-256), plano e dobra da tarefa ANTES do treinamento;
  divergência é falha de integridade.
* No processo principal, cada retorno é conferido contra a tarefa enviada
  (tipo, unidade, modo, ``oficial``, configuração, plano e dobra) e, ao
  gravar, contra a sessão (``SessaoCheckpoints.salvar``).

**Checkpoints.** Só o processo principal grava, sempre pela
``SessaoCheckpoints``; workers nunca escrevem. Unidades concluídas são
reaproveitadas sem treinar; ``incompleta`` (gravação interrompida) e
``falhou`` são executadas de novo. Falha da unidade, exceção no worker ou
morte do worker vira ``FalhaRegistrada`` gravada como ``falha-<n>.json``, e a
execução fica ``incompleta``: nunca é tratada como concluída. Checkpoint
corrompido, identidade divergente ou erro de gravação abortam a execução
(exceção), sem cancelar o que já foi gravado.

**Ordem.** As tarefas são enviadas na ordem canônica (``r``, ``k``, modelo) e
gravadas na ordem em que terminam; o relatório e os resultados voltam sempre
na ordem canônica, lidos dos checkpoints. A ordem de conclusão não afeta
nenhum byte gravado nem devolvido.

O ``carregador`` é uma função sem argumentos, serializável por ``pickle``
(função de módulo ou ``functools.partial``), que devolve a MESMA
``DadosBenchmark`` da sessão: por exemplo ``dados.carregar_implementacao`` ou
``functools.partial(dados.carregar_definitivo, manifesto_s4, commit_p9)``.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from threadpoolctl import threadpool_limits

from . import config as cfg
from . import unidades as un
from .checkpoints import SessaoCheckpoints
from .dados import DadosBenchmark

CONTEXTO_MULTIPROCESSAMENTO = "spawn"

#: Estado do worker: preenchido UMA vez por ``_inicializar_worker``.
_WORKER: Dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProcessoWorker:
    """Diagnóstico de um worker: inicializações (tem de ser 1) e tarefas atendidas."""

    pid: int
    inicializacoes: int
    tarefas: int


@dataclass(frozen=True, eq=False)
class RelatorioExecucao:
    """Resultado de uma chamada a ``executar``, na ordem canônica. Sem métricas."""

    modo: str
    processos: int
    #: Rótulos das unidades pedidas, na ordem canônica.
    unidades: Tuple[str, ...]
    #: Concluídas antes desta chamada: carregadas sem treinar.
    reaproveitadas: Tuple[str, ...]
    #: Concluídas nesta chamada.
    executadas: Tuple[str, ...]
    #: Falhas desta chamada (gravadas como ``falha-<n>.json``), na ordem canônica.
    falhas: Tuple[un.FalhaRegistrada, ...]
    #: Resultados de TODAS as unidades concluídas, lidos dos checkpoints, na ordem canônica.
    resultados: Tuple[un.ResultadoUnidade, ...]
    workers: Tuple[ProcessoWorker, ...] = ()
    #: (rótulo, segundos) de relógio de cada unidade executada ou que falhou nesta chamada, na ordem canônica.
    #: Só diagnóstico (relatorio_execucao.json): nunca entra em checkpoint, identidade nem resultado.
    duracoes: Tuple[Tuple[str, float], ...] = ()
    duracao_total_s: float = 0.0

    @property
    def concluida(self) -> bool:
        """Todas as unidades pedidas concluídas e nenhuma falha. Falha pendente nunca conta como concluída."""
        return not self.falhas and len(self.resultados) == len(self.unidades)

    @property
    def estado(self) -> str:
        return un.ESTADO_CONCLUIDA if self.concluida else "incompleta"


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #


def _inicializar_worker(carregador: Callable[[], DadosBenchmark]) -> None:
    """Uma vez por worker: limita o BLAS, carrega os dados e monta o ``ContextoExecucao``.

    Erro aqui não derruba o pool: fica guardado, e cada tarefa deste worker
    volta como ``FalhaRegistrada`` com a causa.
    """
    _WORKER["inicializacoes"] = _WORKER.get("inicializacoes", 0) + 1
    _WORKER["limites_blas"] = threadpool_limits(limits=cfg.CONFIGURACAO.execucao.threads_blas)
    _WORKER["contexto"], _WORKER["erro"] = None, None
    try:
        dados = carregador()
        if not isinstance(dados, DadosBenchmark):
            raise TypeError(f"o carregador devolveu {type(dados).__name__}, e não DadosBenchmark")
        _WORKER["contexto"] = un.ContextoExecucao.de_dados(dados)
    except Exception as exc:  # carga, guardas, plano: a causa vai para cada tarefa
        _WORKER["erro"] = (getattr(exc, "chave_saida", "falha_tecnica"), type(exc).__name__,
                           f"inicialização do worker: {exc}")


def _executar_no_worker(tarefa: un.TarefaUnidade) -> Tuple[Any, int, int, float]:
    """``(ResultadoUnidade | FalhaRegistrada, pid, inicializações deste worker, segundos)``."""
    inicio = time.perf_counter()
    _WORKER["tarefas"] = _WORKER.get("tarefas", 0) + 1
    if _WORKER.get("erro") is not None:
        chave, categoria, mensagem = _WORKER["erro"]
        resultado: Any = un.FalhaRegistrada(tarefa.unidade, chave, categoria, f"{tarefa.unidade.rotulo}: {mensagem}")
    else:
        resultado = un.executar_tarefa(_WORKER["contexto"], tarefa)
    return resultado, os.getpid(), _WORKER["inicializacoes"], time.perf_counter() - inicio


# --------------------------------------------------------------------------- #
# Processo principal
# --------------------------------------------------------------------------- #


def _falha_de_execucao(unidade: un.Unidade, exc: BaseException) -> un.FalhaRegistrada:
    """Exceção fora de ``executar_tarefa`` (no worker ou na comunicação): falha técnica da unidade."""
    return un.FalhaRegistrada(unidade, "falha_tecnica", type(exc).__name__, f"{unidade.rotulo}: execução: {exc}")


def _conferir_retorno(sessao: SessaoCheckpoints, tarefa: un.TarefaUnidade, retorno: Any) -> Any:
    """``ResultadoUnidade`` ou ``FalhaRegistrada`` DA tarefa; retorno divergente vira falha de integridade."""
    unidade = tarefa.unidade

    def divergente(motivo: str) -> un.FalhaRegistrada:
        return un.FalhaRegistrada(unidade, "integridade", "RetornoDivergente",
                                  f"{unidade.rotulo}: retorno do worker não corresponde à tarefa: {motivo}")

    if isinstance(retorno, un.FalhaRegistrada):
        if retorno.unidade != unidade:
            return divergente("falha de outra unidade")
        if retorno.chave_saida not in set(cfg.CONFIGURACAO.execucao.codigos_saida) - {"concluido"}:
            return divergente(f"chave de saída inválida {retorno.chave_saida!r}")
        return retorno
    if not isinstance(retorno, un.ResultadoUnidade):
        return divergente(f"tipo {type(retorno).__name__}")
    registro, ctx = retorno.registro, sessao.contexto
    if registro.unidade != unidade or retorno.privado.unidade != unidade:
        return divergente("resultado de outra unidade")
    obtido = (registro.modo, registro.oficial, registro.sha256_configuracao, registro.fingerprint_plano,
              registro.fingerprint_dobra, retorno.privado.fingerprint_dobra)
    esperado = (tarefa.modo, ctx.oficial, tarefa.sha256_configuracao, tarefa.fingerprint_plano,
                tarefa.fingerprint_dobra, tarefa.fingerprint_dobra)
    if obtido != esperado:
        return divergente("modo, origem, configuração, plano ou dobra")
    return retorno


def _unidades_pedidas(sessao: SessaoCheckpoints, unidades: Optional[Iterable[un.Unidade]]) -> List[un.Unidade]:
    todas = sessao.contexto.unidades()
    if unidades is None:
        return list(todas)
    pedidas = list(unidades)
    if not all(isinstance(u, un.Unidade) for u in pedidas):
        raise TypeError("unidades deve conter Unidade")
    if len(set(pedidas)) != len(pedidas):
        raise ValueError("unidades repetidas")
    fora = [u.rotulo for u in pedidas if u not in set(todas)]
    if fora:
        raise ValueError(f"unidades fora do plano {sessao.contexto.modo}: {fora}")
    return [u for u in todas if u in set(pedidas)]  # ordem canônica, não a de chegada


def _validar_processos(processos: Any) -> int:
    if type(processos) is not int or processos < 1:
        raise ValueError(f"processos deve ser inteiro ≥ 1: {processos!r}")
    return processos


def executar(sessao: SessaoCheckpoints, carregador: Optional[Callable[[], DadosBenchmark]] = None,
             processos: int = 1, unidades: Optional[Iterable[un.Unidade]] = None) -> RelatorioExecucao:
    """Executa as unidades pendentes (padrão: todas do modo) e grava cada resultado no processo principal.

    ``carregador`` só é usado (e exigido) com ``processos > 1``. Levanta
    exceção, sem treinar nada, se algum checkpoint estiver corrompido ou com
    identidade divergente (``SessaoCheckpoints.verificar``).
    """
    if not isinstance(sessao, SessaoCheckpoints):
        raise TypeError("sessao deve ser SessaoCheckpoints")
    processos = _validar_processos(processos)
    if processos > 1:
        if carregador is None or not callable(carregador):
            raise TypeError("com processos > 1, informe o carregador dos dados de cada worker")
        try:
            pickle.dumps(carregador)
        except Exception as exc:
            raise TypeError(f"o carregador precisa ser serializável para o spawn: {exc}") from exc
    pedidas = _unidades_pedidas(sessao, unidades)

    retomada = sessao.verificar()  # todas as unidades do modo; aborta se algo estiver corrompido
    ja_concluidas = set(retomada.concluidas)
    pendentes = [u for u in pedidas if u.rotulo not in ja_concluidas]
    tarefas = {u: sessao.contexto.tarefa(u) for u in pendentes}
    executadas: List[un.Unidade] = []
    falhas: Dict[un.Unidade, un.FalhaRegistrada] = {}
    workers: Dict[int, List[int]] = {}
    duracoes: Dict[un.Unidade, float] = {}
    inicio_total = time.perf_counter()

    def registrar(unidade: un.Unidade, retorno: Any) -> None:
        resultado = _conferir_retorno(sessao, tarefas[unidade], retorno)
        if isinstance(resultado, un.ResultadoUnidade):
            sessao.salvar(resultado)
            executadas.append(unidade)
        else:
            sessao.registrar_falha(resultado)
            falhas[unidade] = resultado

    if pendentes and processos == 1:
        for unidade in pendentes:
            inicio = time.perf_counter()
            try:
                retorno: Any = un.executar_tarefa(sessao.contexto, tarefas[unidade])
            except Exception as exc:  # o mesmo tratamento de uma exceção no worker
                retorno = _falha_de_execucao(unidade, exc)
            duracoes[unidade] = time.perf_counter() - inicio
            registrar(unidade, retorno)
    elif pendentes:
        contexto_mp = multiprocessing.get_context(CONTEXTO_MULTIPROCESSAMENTO)
        with ProcessPoolExecutor(max_workers=min(processos, len(pendentes)), mp_context=contexto_mp,
                                 initializer=_inicializar_worker, initargs=(carregador,)) as pool:
            futuros: Dict[Future, un.Unidade] = {pool.submit(_executar_no_worker, tarefas[u]): u for u in pendentes}
            try:
                for futuro in as_completed(futuros):
                    unidade = futuros[futuro]
                    try:
                        retorno, pid, inicializacoes, segundos = futuro.result()
                    except Exception as exc:  # exceção no worker, worker morto (BrokenProcessPool), pickle
                        retorno = _falha_de_execucao(unidade, exc)
                    else:
                        duracoes[unidade] = segundos
                        contagem = workers.setdefault(pid, [inicializacoes, 0])
                        contagem[0] = max(contagem[0], inicializacoes)
                        contagem[1] += 1
                    registrar(unidade, retorno)
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)  # o que já foi gravado fica
                raise

    concluidas = ja_concluidas | {u.rotulo for u in executadas}
    ordem = [u for u in pedidas if u.rotulo in concluidas]
    return RelatorioExecucao(
        modo=sessao.contexto.modo, processos=processos, unidades=tuple(u.rotulo for u in pedidas),
        reaproveitadas=tuple(u.rotulo for u in pedidas if u.rotulo in ja_concluidas),
        executadas=tuple(u.rotulo for u in pedidas if u in set(executadas)),
        falhas=tuple(falhas[u] for u in pedidas if u in falhas),
        resultados=tuple(sessao.carregar(u) for u in ordem),
        workers=tuple(ProcessoWorker(pid, n[0], n[1]) for pid, n in sorted(workers.items())),
        duracoes=tuple((u.rotulo, duracoes[u]) for u in pedidas if u in duracoes),
        duracao_total_s=time.perf_counter() - inicio_total)


__all__ = ["CONTEXTO_MULTIPROCESSAMENTO", "ProcessoWorker", "RelatorioExecucao", "executar"]
