"""Métricas por dobra do benchmark S6 (P9, Seções 6, 7 e 7.4).

Consome os resultados das unidades (``unidades.ResultadoUnidade``) SOMENTE no
processo e no diretório local: as predições individuais entram, e saem só
agregados por dobra (contagens, AUC, Brier…). Nada aqui lê dados, grava
arquivos ou depende do modo de execução além da guarda de ``BaseAvaliacao``.

**Desfechos (P9, Seção 7.2).**

* ``G``: AUC-ROC dentro de cada dobra externa, sobre todos os nódulos avaliados.
* ``F``: AUC-ROC dentro de cada dobra externa, só sobre os nódulos avaliados
  com ``12 ≤ D < 30`` mm, ``D = original_shape_Maximum3DDiameter``. O recorte é
  SÓ da avaliação: os modelos foram ajustados com todos os nódulos.

**Validade (P9, Seção 7.3).** A AUC de uma dobra num desfecho só é calculada
com pelo menos ``min_positivos`` nódulos distintos positivos e
``min_negativos`` negativos. Dobra sem isso é degenerada (DD): a AUC não é
calculada nem imputada. Como as linhas são as mesmas para todos os modelos, a
validade é a mesma para todos.

**Métricas descritivas (P9, Seção 7.4).** Só no desfecho G. Brier, log-loss,
intercepto e inclinação de calibração e sensibilidade/especificidade com
``p ≥ 0,5`` para as probabilidades; para o SVM, apenas sensibilidade e
especificidade com ``decision_function ≥ 0``, rotuladas "regra nativa, não
comparável". Nenhuma entra em contraste ou na seleção.

O intercepto e a inclinação vêm da regressão logística de ``y`` sobre
``logit(p)`` (``p`` cortada em ``[corte, 1 − corte]``), sem penalização, ajustada
por Newton-Raphson. Sem convergência (separação) ou com ``logit(p)`` constante,
o valor fica indefinido (``None``), nunca imputado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from . import adaptadores_s4 as s4
from . import config as cfg
from . import modelos
from . import unidades as un

DESFECHO_G = "G"
DESFECHO_F = "F"
ROTULO_REGRA_NATIVA = "regra nativa, não comparável"


class MetricasInvalidas(ValueError):
    """Entrada incoerente com a amostra ou com a P9. Chave de saída: integridade."""

    chave_saida = "integridade"


# --------------------------------------------------------------------------- #
# Base da avaliação
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class BaseAvaliacao:
    """O que as métricas precisam saber da amostra: faixa do desfecho F e paciente de cada linha."""

    modo: str
    amostra: s4.Amostra = field(repr=False)
    #: ``True`` onde ``12 ≤ D < 30`` mm (D = variável da faixa da configuração).
    na_faixa: np.ndarray = field(repr=False)
    #: Índice (0…P−1, na ordem alfabética dos ``patient_id``) do paciente de cada linha.
    paciente: np.ndarray = field(repr=False)
    n_pacientes: int = 0

    @classmethod
    def de_contexto(cls, contexto: un.ContextoExecucao) -> "BaseAvaliacao":
        """Base do modo ``definitivo``; o ``implementacao`` não calcula nenhuma métrica externa (P9, Seção 14.3)."""
        if not isinstance(contexto, un.ContextoExecucao):
            raise TypeError("contexto deve ser ContextoExecucao")
        if contexto.modo != "definitivo":
            raise MetricasInvalidas(
                f"métricas externas só existem no modo definitivo; o modo {contexto.modo!r} não as calcula "
                "(P9, Seção 14.3)")
        return cls._criar(contexto.modo, contexto.amostra, contexto.X)

    @classmethod
    def _criar(cls, modo: str, amostra: s4.Amostra, X: np.ndarray) -> "BaseAvaliacao":
        faixa = cfg.CONFIGURACAO.faixa
        caracteristicas = tuple(cfg.CONFIGURACAO.caracteristicas)
        if faixa.variavel not in caracteristicas:
            raise MetricasInvalidas(f"variável da faixa {faixa.variavel!r} fora das características")
        if X.shape != (amostra.n, len(caracteristicas)):
            raise MetricasInvalidas(f"X com forma {X.shape}; esperada {(amostra.n, len(caracteristicas))}")
        diametro = np.asarray(X[:, caracteristicas.index(faixa.variavel)], dtype=np.float64)
        na_faixa = (diametro >= faixa.inferior) & (diametro < faixa.superior)
        _, paciente = np.unique(np.asarray(amostra.grupos, dtype=object).astype(str), return_inverse=True)
        na_faixa.setflags(write=False)
        paciente = paciente.astype(np.int64)
        paciente.setflags(write=False)
        return cls(modo, amostra, na_faixa, paciente, int(paciente.max()) + 1)

    def linhas_do_desfecho(self, posicoes: np.ndarray, desfecho: str) -> np.ndarray:
        """Máscara, sobre ``posicoes`` (linhas da avaliação), das linhas que contam no desfecho."""
        if desfecho == DESFECHO_G:
            return np.ones(len(posicoes), dtype=bool)
        if desfecho == DESFECHO_F:
            return self.na_faixa[posicoes]
        raise MetricasInvalidas(f"desfecho desconhecido: {desfecho!r}")

    def contagem_faixa(self) -> Dict[str, int]:
        """Contagem do desfecho F na base inteira (conferida pelo carregador no modo definitivo)."""
        dentro = np.flatnonzero(self.na_faixa)
        y = self.amostra.y[dentro]
        return {"pacientes": len(set(self.paciente[dentro].tolist())), "nodulos": int(len(dentro)),
                "classe_0": int(np.sum(y == 0)), "classe_1": int(np.sum(y == 1))}


# --------------------------------------------------------------------------- #
# Denominadores e AUC por dobra
# --------------------------------------------------------------------------- #


class Denominador(NamedTuple):
    """Denominadores e validade de uma dobra num desfecho (P9, Seção 7.3). Iguais para todos os modelos."""

    repeticao: int
    dobra: int
    desfecho: str
    nodulos: int
    pacientes: int
    positivos: int
    negativos: int
    valida: bool


def valida(positivos: int, negativos: int) -> bool:
    """Regra da P9, Seção 7.3, sobre contagens de nódulos DISTINTOS."""
    v = cfg.CONFIGURACAO.validade
    return positivos >= v.min_positivos and negativos >= v.min_negativos


def auc_roc(y: np.ndarray, pontuacoes: np.ndarray) -> float:
    """AUC-ROC (``sklearn.metrics.roc_auc_score``); exige as duas classes. Empates contam 0,5."""
    y = np.asarray(y)
    if len(set(y.tolist())) != 2:
        raise MetricasInvalidas("AUC exige as duas classes")
    return float(roc_auc_score(y, np.asarray(pontuacoes, dtype=np.float64)))


def denominador(base: BaseAvaliacao, repeticao: int, dobra: int, posicoes: np.ndarray, desfecho: str) -> Denominador:
    linhas = posicoes[base.linhas_do_desfecho(posicoes, desfecho)]
    y = base.amostra.y[linhas]
    positivos, negativos = int(np.sum(y == 1)), int(np.sum(y == 0))
    return Denominador(repeticao, dobra, desfecho, int(len(linhas)), len(set(base.paciente[linhas].tolist())),
                       positivos, negativos, valida(positivos, negativos))


def auc_da_dobra(base: BaseAvaliacao, resultado: un.ResultadoUnidade, desfecho: str
                 ) -> Tuple[Denominador, Optional[float]]:
    """Denominadores e AUC de uma unidade num desfecho; ``None`` se a dobra for degenerada (DD), nunca imputada."""
    _exigir_coerente(base, resultado)
    privado = resultado.privado
    unidade = resultado.unidade
    posicoes = np.asarray(privado.posicoes)
    d = denominador(base, unidade.repeticao, unidade.dobra, posicoes, desfecho)
    if not d.valida:
        return d, None
    mascara = base.linhas_do_desfecho(posicoes, desfecho)
    return d, auc_roc(np.asarray(privado.y)[mascara], np.asarray(privado.pontuacoes)[mascara])


def _exigir_coerente(base: BaseAvaliacao, resultado: un.ResultadoUnidade) -> None:
    """Predições da unidade alinhadas à amostra da base (posições, rótulos e ``nodulo_id``)."""
    if not isinstance(resultado, un.ResultadoUnidade):
        raise TypeError("resultado deve ser ResultadoUnidade")
    privado = resultado.privado
    posicoes = np.asarray(privado.posicoes)
    amostra = base.amostra
    if posicoes.ndim != 1 or len(posicoes) == 0 or posicoes.min() < 0 or posicoes.max() >= amostra.n \
            or len(np.unique(posicoes)) != len(posicoes):
        raise MetricasInvalidas(f"{resultado.unidade.rotulo}: posições inválidas")
    if not np.array_equal(np.asarray(privado.y), amostra.y[posicoes]) or \
            tuple(privado.nodulo_ids) != tuple(str(i) for i in amostra.ids[posicoes]):
        raise MetricasInvalidas(f"{resultado.unidade.rotulo}: rótulos ou nodulo_id não conferem com a amostra")
    pontuacoes = np.asarray(privado.pontuacoes)
    if pontuacoes.shape != posicoes.shape or not np.all(np.isfinite(pontuacoes)):
        raise MetricasInvalidas(f"{resultado.unidade.rotulo}: pontuações com forma inválida ou não finitas")


# --------------------------------------------------------------------------- #
# Métricas descritivas (somente G)
# --------------------------------------------------------------------------- #


class Descritiva(NamedTuple):
    """Uma métrica descritiva de uma dobra no desfecho G. ``valor`` é ``None`` quando indefinida."""

    modelo: str
    repeticao: int
    dobra: int
    metrica: str
    valor: Optional[float]
    rotulo: str


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, corte: float) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), corte, 1.0 - corte)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def calibracao(y: np.ndarray, p: np.ndarray, corte: float, max_iteracoes: int = 100,
               tolerancia: float = 1e-10) -> Tuple[Optional[float], Optional[float]]:
    """``(intercepto, inclinação)`` da regressão logística de ``y`` sobre ``logit(p)``; ``(None, None)`` se indefinida.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), corte, 1.0 - corte)
    z = np.log(p) - np.log1p(-p)
    y = np.asarray(y, dtype=np.float64)
    if np.ptp(z) == 0.0 or y.min() == y.max():
        return None, None
    A = np.column_stack([np.ones_like(z), z])
    beta = np.zeros(2)
    for _ in range(max_iteracoes):
        mu = 1.0 / (1.0 + np.exp(-(A @ beta)))
        gradiente = A.T @ (y - mu)
        hessiana = (A * (mu * (1.0 - mu))[:, None]).T @ A
        if np.linalg.cond(hessiana) > 1e12:
            return None, None
        passo = np.linalg.solve(hessiana, gradiente)
        beta = beta + passo
        if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 1e6:
            return None, None
        if np.max(np.abs(passo)) < tolerancia:
            return float(beta[0]), float(beta[1])
    return None, None


def sensibilidade_especificidade(y: np.ndarray, previsto_positivo: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y)
    previsto = np.asarray(previsto_positivo, dtype=bool)
    positivos, negativos = int(np.sum(y == 1)), int(np.sum(y == 0))
    return (float(np.sum(previsto & (y == 1)) / positivos), float(np.sum(~previsto & (y == 0)) / negativos))


def descritivas_da_dobra(base: BaseAvaliacao, resultado: un.ResultadoUnidade) -> Tuple[Descritiva, ...]:
    """Métricas descritivas de uma unidade, só no desfecho G (P9, Seção 7.4); vazio se a dobra for degenerada."""
    _exigir_coerente(base, resultado)
    espec = cfg.CONFIGURACAO.metricas
    unidade, privado = resultado.unidade, resultado.privado
    y, s = np.asarray(privado.y), np.asarray(privado.pontuacoes)
    if not valida(int(np.sum(y == 1)), int(np.sum(y == 0))):
        return ()
    modelo = unidade.modelo
    linha = lambda metrica, valor, rotulo="": Descritiva(  # noqa: E731
        modelo, unidade.repeticao, unidade.dobra, metrica, None if valor is None else float(valor), rotulo)
    if modelos.especificacao(modelo).metricas_probabilisticas:
        intercepto, inclinacao = calibracao(y, s, espec.corte_log_loss)
        sens, esp = sensibilidade_especificidade(y, s >= espec.limiar_probabilidade)
        return (linha("brier", brier(y, s)), linha("log_loss", log_loss(y, s, espec.corte_log_loss)),
                linha("intercepto_calibracao", intercepto), linha("inclinacao_calibracao", inclinacao),
                linha("sensibilidade_p_0_5", sens), linha("especificidade_p_0_5", esp))
    sens, esp = sensibilidade_especificidade(y, s >= espec.limiar_decisao_nativa)
    return (linha("sensibilidade_regra_nativa", sens, ROTULO_REGRA_NATIVA),
            linha("especificidade_regra_nativa", esp, ROTULO_REGRA_NATIVA))


__all__ = [
    "BaseAvaliacao", "DESFECHO_F", "DESFECHO_G", "Denominador", "Descritiva", "MetricasInvalidas",
    "ROTULO_REGRA_NATIVA", "auc_da_dobra", "auc_roc", "brier", "calibracao", "denominador", "descritivas_da_dobra",
    "log_loss", "sensibilidade_especificidade", "valida",
]
