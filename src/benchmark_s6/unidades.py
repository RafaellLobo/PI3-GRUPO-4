"""Execução de uma unidade ``(r, k, modelo)`` do benchmark S6 (P9, Seções 6 e 7.1).

Uma unidade ajusta o ``modelo`` no treino da dobra externa ``(r, k)``
(``tuning.ajustar``) e gera as pontuações da avaliação externa:

* ``predict_proba[:, 1]`` para ``lr_fixa``, ``lr_tuning``, ``rf`` e ``xgb``;
* ``decision_function`` para ``svm``.

O método vem de ``EspecModelo.pontuacao`` da configuração. As pontuações são
conferidas: uma por linha da avaliação externa, ``float64`` finitas e, nas
probabilidades, em [0, 1]. Exceção na predição ou pontuação ``NaN`` ou
infinita é falha técnica (P9, Seção 6). Nenhuma métrica é calculada aqui.

**Dois tipos de resultado, separados de propósito:**

* ``RegistroUnidade``: agregado, sem identificadores nem pontuações. Traz a
  configuração escolhida, posição e borda na grade, avisos, número de
  ajustes, contagens agregadas e fingerprints. Pode ir para relatórios
  depois da varredura de privacidade.
* ``PredicoesPrivadas``: individual. Traz ``nodulo_id``, posição, rótulo e
  pontuação de cada linha avaliada, mais as médias internas do ajuste. Fica
  SÓ no diretório local (P9, Seção 14.5); nunca é versionado.

Os dois contêm só tipos primitivos, tuplas e arrays ``numpy``: serializam
com ``pickle`` e com JSON (``como_dict``), sem depender de ``PlanoDobras``,
``DobraExterna`` ou ``ResultadoAjuste``.

**Desenho para workers (multiprocessing com ``spawn`` no Windows).** Nada do
plano atravessa processos. O processo principal monta ``TarefaUnidade``
(só primitivos: modo, ``r``, ``k``, modelo, configuração, fingerprints do
plano e da dobra e SHA-256 da amostra e de ``X``). Cada worker monta o PRÓPRIO
``ContextoExecucao`` uma vez, a partir da mesma carga
(``ContextoExecucao.de_dados``), regerando o plano de forma determinística, e
roda ``executar_tarefa``: amostra, ``X``, plano e dobra precisam coincidir com
os da tarefa ANTES do treinamento, senão a tarefa falha por integridade. O
resultado volta como ``ResultadoUnidade`` ou ``FalhaRegistrada``, ambos de
tipos primitivos. O executor paralelo fica em ``executor.py``.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from threadpoolctl import threadpool_limits

from . import adaptadores_s4 as s4
from . import config as cfg
from . import folds
from . import modelos
from . import tuning
from .dados import DadosBenchmark

ESTADO_CONCLUIDA = "concluida"
ESTADO_FALHA = "FT"
FASE_PREDICAO = "predicao"


class FalhaUnidade(RuntimeError):
    """Falha de uma unidade; ``chave_saida`` herda a da causa (falha técnica por padrão)."""

    def __init__(self, unidade: "Unidade", mensagem: str, chave_saida: str = "falha_tecnica"):
        super().__init__(f"{unidade.rotulo}: {mensagem}")
        self.unidade = unidade
        self.chave_saida = chave_saida


# --------------------------------------------------------------------------- #
# Identificação
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Unidade:
    """Unidade ``(r, k, modelo)`` da P9, Seção 6."""

    repeticao: int
    dobra: int
    modelo: str

    def __post_init__(self) -> None:
        espec = cfg.CONFIGURACAO.dobras
        if type(self.repeticao) is not int or not 0 <= self.repeticao < espec.n_repeticoes:
            raise ValueError(f"repetição fora de 0…{espec.n_repeticoes - 1}: {self.repeticao!r}")
        if type(self.dobra) is not int or not 0 <= self.dobra < espec.n_dobras_externas:
            raise ValueError(f"dobra fora de 0…{espec.n_dobras_externas - 1}: {self.dobra!r}")
        modelos.especificacao(self.modelo)

    @property
    def rotulo(self) -> str:
        return f"r{self.repeticao:02d}-k{self.dobra}-{self.modelo}"


def unidades_do_modo(modo: str) -> Tuple[Unidade, ...]:
    """Unidades na ordem ``r``, ``k``, modelo (da configuração): 25 no modo ``implementacao``, 250 no ``definitivo``."""
    return tuple(Unidade(r, k, m) for r in folds.repeticoes_do_modo(modo)
                 for k in range(cfg.CONFIGURACAO.dobras.n_dobras_externas) for m in modelos.codigos())


@dataclass(frozen=True)
class TarefaUnidade:
    """O que atravessa processos: só primitivos. O worker refaz o resto e confere os fingerprints."""

    modo: str
    repeticao: int
    dobra: int
    modelo: str
    fingerprint_plano: str
    fingerprint_dobra: str
    sha256_configuracao: str
    #: ``Amostra.sha256`` (ids, rótulos e grupos) e ``sha256_caracteristicas(X)`` da carga do processo
    #: principal; o worker confere os dois antes do treinamento.
    sha256_amostra: str
    sha256_caracteristicas: str

    @property
    def unidade(self) -> Unidade:
        return Unidade(self.repeticao, self.dobra, self.modelo)


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #


def _contagem(amostra: s4.Amostra, posicoes: np.ndarray) -> Tuple[Tuple[str, int], ...]:
    y = amostra.y[posicoes]
    return (("nodulos", int(len(posicoes))), ("pacientes", len(set(amostra.grupos[posicoes].tolist()))),
            ("classe_0", int(np.sum(y == 0))), ("classe_1", int(np.sum(y == 1))))


@dataclass(frozen=True)
class RegistroUnidade:
    """Registro agregado de uma unidade concluída: sem identificadores e sem pontuações."""

    unidade: Unidade
    modo: str
    oficial: bool
    sha256_configuracao: str
    fingerprint_plano: str
    fingerprint_dobra: str
    metodo_pontuacao: str
    #: ``tuning.ResultadoAjuste.registro()`` convertido em tuplas (configuração, posição, borda, avisos…).
    configuracao: Tuple[Tuple[str, Any], ...]
    indice_configuracao: int
    n_configuracoes: int
    borda: Tuple[Tuple[str, Any, int, int, bool], ...]
    n_ajustes: int
    #: ``(fase, categoria, mensagem, ocorrências)``, das fases de busca/ajuste e de predição.
    avisos: Tuple[Tuple[str, str, str, int], ...]
    contagem_treino: Tuple[Tuple[str, int], ...]
    contagem_avaliacao: Tuple[Tuple[str, int], ...]

    @property
    def estado(self) -> str:
        return ESTADO_CONCLUIDA

    @property
    def avisos_convergencia(self) -> int:
        return sum(n for _, categoria, _, n in self.avisos if categoria == "ConvergenceWarning")

    def como_dict(self) -> Dict[str, Any]:
        return {
            "unidade": self.unidade.rotulo, "repeticao": self.unidade.repeticao, "dobra_externa": self.unidade.dobra,
            "modelo": self.unidade.modelo, "estado": self.estado, "modo": self.modo, "oficial": self.oficial,
            "sha256_configuracao": self.sha256_configuracao, "fingerprint_plano": self.fingerprint_plano,
            "fingerprint_dobra": self.fingerprint_dobra, "metodo_pontuacao": self.metodo_pontuacao,
            "configuracao": dict(self.configuracao), "indice_configuracao": self.indice_configuracao,
            "n_configuracoes": self.n_configuracoes,
            "borda": {nome: {"valor": valor, "posicao": pos, "n_valores": n, "na_borda": borda}
                      for nome, valor, pos, n, borda in self.borda},
            "na_borda_da_grade": any(b[4] for b in self.borda), "n_ajustes": self.n_ajustes,
            "avisos": [{"fase": f, "categoria": c, "mensagem": m, "ocorrencias": n} for f, c, m, n in self.avisos],
            "avisos_convergencia": self.avisos_convergencia,
            "contagem_treino": dict(self.contagem_treino), "contagem_avaliacao": dict(self.contagem_avaliacao),
        }


@dataclass(frozen=True, eq=False)
class PredicoesPrivadas:
    """PRIVADO: predições individuais da avaliação externa. Só no diretório local (P9, Seção 14.5)."""

    unidade: Unidade
    fingerprint_dobra: str
    metodo_pontuacao: str
    #: Posições na amostra (ordem canônica), ``nodulo_id`` e rótulo de cada linha avaliada.
    posicoes: np.ndarray = field(repr=False)
    nodulo_ids: Tuple[str, ...] = field(repr=False)
    y: np.ndarray = field(repr=False)
    #: Pontuação ``float64`` da avaliação externa, na ordem de ``posicoes``.
    pontuacoes: np.ndarray = field(repr=False)
    #: Médias da AUC interna por configuração (ajuste interno). Também locais.
    medias_internas: Tuple[float, ...] = field(repr=False)

    def como_dict(self) -> Dict[str, Any]:
        return {"unidade": self.unidade.rotulo, "fingerprint_dobra": self.fingerprint_dobra,
                "metodo_pontuacao": self.metodo_pontuacao, "posicoes": self.posicoes.tolist(),
                "nodulo_ids": list(self.nodulo_ids), "y": self.y.tolist(), "pontuacoes": self.pontuacoes.tolist(),
                "medias_internas": list(self.medias_internas)}


@dataclass(frozen=True, eq=False)
class ResultadoUnidade:
    registro: RegistroUnidade
    privado: PredicoesPrivadas

    @property
    def unidade(self) -> Unidade:
        return self.registro.unidade

    @property
    def estado(self) -> str:
        return ESTADO_CONCLUIDA


@dataclass(frozen=True)
class FalhaRegistrada:
    """Falha técnica (ou de integridade) de uma unidade, com a causa (P9, Seção 6). Só primitivos."""

    unidade: Unidade
    chave_saida: str
    categoria: str
    mensagem: str

    @property
    def estado(self) -> str:
        return ESTADO_FALHA

    def como_dict(self) -> Dict[str, Any]:
        return {"unidade": self.unidade.rotulo, "estado": self.estado, "chave_saida": self.chave_saida,
                "categoria": self.categoria, "mensagem": self.mensagem}


# --------------------------------------------------------------------------- #
# Contexto de execução
# --------------------------------------------------------------------------- #


def sha256_caracteristicas(X: np.ndarray) -> str:
    """SHA-256 da matriz ``X`` em ``float64`` little-endian, contígua (mesma forma da identidade dos checkpoints)."""
    return hashlib.sha256(np.ascontiguousarray(X, dtype="<f8").tobytes()).hexdigest()


@dataclass(frozen=True, eq=False)
class ContextoExecucao:
    """Dados e plano de UM processo. Nunca atravessa processos: cada worker monta o seu."""

    modo: str
    oficial: bool
    X: np.ndarray = field(repr=False)
    amostra: s4.Amostra = field(repr=False)
    plano: folds.PlanoDobras = field(repr=False)
    sha256_configuracao: str
    sha256_caracteristicas: str

    @classmethod
    def de_dados(cls, dados: DadosBenchmark) -> "ContextoExecucao":
        """``X``, ``Amostra`` e plano da MESMA carga: alinhados por construção, plano determinístico."""
        if not isinstance(dados, DadosBenchmark):
            raise TypeError("dados deve ser DadosBenchmark")
        X = dados.X()
        X.setflags(write=False)
        amostra = dados.amostra()
        plano = folds.gerar_plano(amostra, dados.modo)
        return cls(dados.modo, bool(dados.oficial), X, amostra, plano, cfg.sha256_configuracao(cfg.CONFIGURACAO),
                   sha256_caracteristicas(X))

    def unidades(self) -> Tuple[Unidade, ...]:
        return unidades_do_modo(self.modo)

    def tarefa(self, unidade: Unidade) -> TarefaUnidade:
        dobra = self._dobra(unidade)
        return TarefaUnidade(self.modo, unidade.repeticao, unidade.dobra, unidade.modelo, self.plano.fingerprint,
                             dobra.fingerprint, self.sha256_configuracao, self.amostra.sha256,
                             self.sha256_caracteristicas)

    def tarefas(self) -> Tuple[TarefaUnidade, ...]:
        return tuple(self.tarefa(u) for u in self.unidades())

    def _dobra(self, unidade: Unidade) -> folds.DobraExterna:
        if not isinstance(unidade, Unidade):
            raise TypeError("unidade deve ser Unidade")
        try:
            return folds.dobra_para_modelo(self.plano, unidade.repeticao, unidade.dobra, unidade.modelo)
        except folds.DobrasInvalidas as exc:
            raise FalhaUnidade(unidade, f"unidade fora do plano {self.modo}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #


def _pontuar(estimador: Any, X_avaliacao: np.ndarray, metodo: str, unidade: Unidade) -> np.ndarray:
    n = len(X_avaliacao)
    saida = np.asarray(getattr(estimador, metodo)(X_avaliacao))
    if metodo == "predict_proba":
        if saida.shape != (n, 2):
            raise FalhaUnidade(unidade, f"predict_proba com forma {saida.shape}; esperada {(n, 2)}")
        saida = saida[:, 1]
    elif saida.shape != (n,):
        raise FalhaUnidade(unidade, f"decision_function com forma {saida.shape}; esperada {(n,)}")
    if saida.dtype.kind != "f":
        raise FalhaUnidade(unidade, f"pontuação não real: dtype {saida.dtype}")
    pontuacoes = np.array(saida, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(pontuacoes)):
        raise FalhaUnidade(unidade, "pontuação externa NaN ou infinita")
    if metodo == "predict_proba" and not np.all((pontuacoes >= 0.0) & (pontuacoes <= 1.0)):
        raise FalhaUnidade(unidade, "probabilidade fora de [0, 1]")
    pontuacoes.setflags(write=False)
    return pontuacoes


def executar_unidade(contexto: ContextoExecucao, unidade: Unidade) -> ResultadoUnidade:
    """Ajusta e pontua uma unidade no processo atual. Falha levanta ``FalhaUnidade``."""
    if not isinstance(contexto, ContextoExecucao):
        raise TypeError("contexto deve ser ContextoExecucao")
    dobra = contexto._dobra(unidade)
    espec = modelos.especificacao(unidade.modelo)
    amostra = contexto.amostra
    # Verificação prévia do ambiente: dentro do ajuste, a falha de construção viraria
    # falha técnica; XGBoost ausente ou incompatível é integridade (P9, Seção 14.3).
    try:
        modelos.criar_estimador(unidade.modelo)
    except (modelos.XGBoostIndisponivel, modelos.ModeloInvalido) as exc:
        raise FalhaUnidade(unidade, f"{type(exc).__name__}: {exc}", exc.chave_saida) from exc
    try:
        ajuste = tuning.ajustar(contexto.X, amostra, dobra, unidade.modelo)
    except Exception as exc:  # FalhaAjuste, DobrasInvalidas, XGBoostIndisponivel, ModeloInvalido…
        raise FalhaUnidade(unidade, f"{type(exc).__name__}: {exc}",
                           getattr(exc, "chave_saida", "falha_tecnica")) from exc
    if (ajuste.repeticao, ajuste.dobra, ajuste.modelo, ajuste.fingerprint_dobra) != \
            (unidade.repeticao, unidade.dobra, unidade.modelo, dobra.fingerprint):
        raise FalhaUnidade(unidade, "ajuste de outra unidade ou de outra dobra")

    avaliacao = dobra.externa.avaliacao
    X_avaliacao = np.array(contexto.X[avaliacao], dtype=np.float64, copy=True)
    if not np.all(np.isfinite(X_avaliacao)):
        raise FalhaUnidade(unidade, "característica NaN ou infinita na avaliação externa")
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as registrados:
        warnings.simplefilter("always")
        try:
            pontuacoes = _pontuar(ajuste.estimador, X_avaliacao, espec.pontuacao, unidade)
        except FalhaUnidade:
            raise
        except Exception as exc:  # P9, Seção 6: exceção na predição é falha técnica
            raise FalhaUnidade(unidade, f"predição: {type(exc).__name__}: {exc}") from exc
    contagem_avisos = Counter((w.category.__name__, str(w.message)) for w in registrados)
    avisos_predicao = tuple(tuning.AvisoAjuste(FASE_PREDICAO, categoria, mensagem, n)
                            for (categoria, mensagem), n in sorted(contagem_avisos.items()))

    reg = ajuste.registro()
    registro = RegistroUnidade(
        unidade=unidade, modo=contexto.modo, oficial=contexto.oficial,
        sha256_configuracao=contexto.sha256_configuracao, fingerprint_plano=contexto.plano.fingerprint,
        fingerprint_dobra=dobra.fingerprint, metodo_pontuacao=espec.pontuacao,
        configuracao=tuple(sorted(reg["configuracao"].items())), indice_configuracao=reg["indice_configuracao"],
        n_configuracoes=reg["n_configuracoes"],
        borda=tuple((nome, b["valor"], b["posicao"], b["n_valores"], b["na_borda"])
                    for nome, b in sorted(reg["borda"].items())),
        n_ajustes=reg["n_ajustes"],
        avisos=tuple((a.fase, a.categoria, a.mensagem, a.ocorrencias) for a in (*ajuste.avisos, *avisos_predicao)),
        contagem_treino=_contagem(amostra, dobra.externa.treino), contagem_avaliacao=_contagem(amostra, avaliacao))
    posicoes = np.array(avaliacao, dtype=np.int64, copy=True)
    y = np.array(amostra.y[avaliacao], dtype=np.int64, copy=True)
    for arr in (posicoes, y):
        arr.setflags(write=False)
    privado = PredicoesPrivadas(unidade, dobra.fingerprint, espec.pontuacao, posicoes,
                                tuple(str(i) for i in amostra.ids[avaliacao]), y, pontuacoes,
                                tuple(float(m) for m in ajuste.medias_internas))
    return ResultadoUnidade(registro, privado)


def executar_tarefa(contexto: ContextoExecucao, tarefa: TarefaUnidade) -> Union[ResultadoUnidade, FalhaRegistrada]:
    """Ponto de entrada do worker: confere a tarefa contra o contexto local e só então executa.

    Nunca deixa uma falha da unidade escapar: devolve ``FalhaRegistrada``.
    Tipos errados (erro de programação) ainda levantam ``TypeError``.
    """
    if not isinstance(contexto, ContextoExecucao):
        raise TypeError("contexto deve ser ContextoExecucao")
    if not isinstance(tarefa, TarefaUnidade):
        raise TypeError("tarefa deve ser TarefaUnidade")
    try:
        unidade = tarefa.unidade
    except ValueError as exc:
        raise TypeError(f"tarefa com unidade inválida: {exc}") from exc
    try:
        if tarefa.modo != contexto.modo:
            raise FalhaUnidade(unidade, f"tarefa do modo {tarefa.modo}; contexto {contexto.modo}", "integridade")
        if tarefa.sha256_configuracao != contexto.sha256_configuracao:
            raise FalhaUnidade(unidade, "configuração da tarefa difere da do processo", "integridade")
        if tarefa.sha256_amostra != contexto.amostra.sha256:
            raise FalhaUnidade(unidade, "amostra (ids, rótulos ou grupos) da tarefa difere da deste processo",
                               "integridade")
        if tarefa.sha256_caracteristicas != contexto.sha256_caracteristicas:
            raise FalhaUnidade(unidade, "características (X) da tarefa diferem das deste processo", "integridade")
        if tarefa.fingerprint_plano != contexto.plano.fingerprint:
            raise FalhaUnidade(unidade, "plano da tarefa difere do regerado neste processo", "integridade")
        if tarefa.fingerprint_dobra != contexto._dobra(unidade).fingerprint:
            raise FalhaUnidade(unidade, "dobra da tarefa difere da regerada neste processo", "integridade")
        return executar_unidade(contexto, unidade)
    except FalhaUnidade as falha:
        causa: Optional[BaseException] = falha.__cause__
        categoria = type(causa).__name__ if causa is not None else type(falha).__name__
        return FalhaRegistrada(unidade, falha.chave_saida, categoria, str(falha))


__all__ = [
    "ContextoExecucao", "ESTADO_CONCLUIDA", "ESTADO_FALHA", "FalhaRegistrada", "FalhaUnidade",
    "PredicoesPrivadas", "RegistroUnidade", "ResultadoUnidade", "TarefaUnidade", "Unidade", "executar_tarefa",
    "executar_unidade", "sha256_caracteristicas", "unidades_do_modo",
]
