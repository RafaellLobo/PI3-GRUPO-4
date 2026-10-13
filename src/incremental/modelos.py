"""Fábrica compartilhada de modelos — Cartão 4, Sprint 4.

Constrói, ajusta e aplica os modelos da Seção 6 do protocolo a partir das
especificações de ``config`` (``EspecModelo``), em três passos separados:

* ``construir`` — autoriza o modelo pelas guardas e monta o estimador NÃO
  ajustado. Nada é ajustado na importação nem na construção;
* ``ajustar`` — reemite a autorização, clona o estimador e ajusta;
* ``ModeloAjustado.prever_proba`` / ``prever_classe`` — reemitem a
  autorização e aplicam o modelo ajustado.

Estimadores (Seção 6.1), com os parâmetros de ``config.ESPEC_AVALIACAO``:

* M0, M1, M2–M4, M3v, M4v e M2v com S_k não vazio: ``StandardScaler →
  LogisticRegression`` num único ``Pipeline``, com os nomes de passo do
  baseline da Sprint 3 (``scaler``, ``logreg``);
* M2v com S_k vazio: ``DummyClassifier(strategy=config.DUMMY_ESTRATEGIA)``,
  que prevê a prevalência do treino (Seção 6.3). M3v e M4v com S_k vazio
  continuam com a regressão logística sobre seus preditores de referência.

Autorização: toda operação chama ``guardas.autorizar_modelo``, que exige as
decisões da atividade e, para M2, M3 e M4, P3 aprovada com evidência no git,
e compara a composição com a derivada das decisões verificadas. Não há
caminho alternativo: a fábrica não cria aprovações nem aceita autorização
pronta, e uma especificação de M2–M4 montada só com decisões em memória é
recusada pela guarda.

Colunas: ``X`` é sempre um DataFrame cujas colunas são EXATAMENTE as
features do modelo, com os mesmos nomes e na mesma ordem
(``frame[list(modelo.features)]``). Nada é selecionado, reordenado ou
convertido em silêncio; ``NaN`` ou ``Inf`` abortam (Seção 6.1).

S_k chega pronto (``selecao.SelecaoFold``): a fábrica nunca executa a
seleção LASSO. O módulo não lê dados, não calcula métricas e não implementa
spline, orquestrador nem CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from . import guardas
from .selecao import SelecaoFold

PASSO_SCALER = "scaler"
PASSO_LOGREG = "logreg"
PASSO_DUMMY = "dummy"
CLASSES = (0, 1)


class ModeloInvalido(ValueError):
    """Entrada incompatível com o modelo ou uso fora do contrato da fábrica."""


# --------------------------------------------------------------------------- #
# Estimadores (sem autorização: uso interno e inspeção estrutural)
# --------------------------------------------------------------------------- #


def estimador_logistico() -> Pipeline:
    """``StandardScaler → LogisticRegression`` com ``ESPEC_AVALIACAO`` (não ajustado)."""
    return Pipeline([
        (PASSO_SCALER, StandardScaler()),
        (PASSO_LOGREG, LogisticRegression(**cfg.ESPEC_AVALIACAO.parametros())),
    ])


def estimador_dummy() -> Pipeline:
    """``DummyClassifier`` da prevalência do treino, para M2v com S_k vazio (não ajustado)."""
    return Pipeline([(PASSO_DUMMY, DummyClassifier(strategy=cfg.DUMMY_ESTRATEGIA))])


def _estimador(espec: cfg.EspecModelo) -> Pipeline:
    return estimador_dummy() if espec.usa_dummy else estimador_logistico()


# --------------------------------------------------------------------------- #
# Entradas
# --------------------------------------------------------------------------- #


def _matriz(X: Any, features: Tuple[str, ...], nome: str) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise ModeloInvalido(f"{nome}: X deve ser DataFrame com as colunas do modelo")
    colunas = list(X.columns)
    if colunas != list(features):
        raise ModeloInvalido(f"{nome}: colunas de X diferentes das features do modelo (nomes ou ordem); "
                             f"use frame[list(modelo.features)]")
    if len(X) == 0:
        raise ModeloInvalido(f"{nome}: X sem linhas")
    if not all(getattr(t, "kind", "O") in "iuf" for t in X.dtypes):
        raise ModeloInvalido(f"{nome}: X deve ter apenas colunas inteiras ou reais")
    matriz = X.to_numpy(dtype=np.float64)
    if not np.isfinite(matriz).all():
        raise ModeloInvalido(f"{nome}: NaN ou Inf em X")
    return matriz


def _rotulos(y: Any, X: pd.DataFrame, nome: str) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        raise ModeloInvalido(f"{nome}: y deve ser unidimensional")
    if isinstance(y, pd.Series) and not y.index.equals(X.index):
        raise ModeloInvalido(f"{nome}: índices de X e y desalinhados")
    rotulos = np.asarray(y)
    if rotulos.ndim != 1 or len(rotulos) != len(X):
        raise ModeloInvalido(f"{nome}: y deve ser unidimensional e ter uma entrada por linha de X")
    if rotulos.dtype.kind not in "iu":
        raise ModeloInvalido(f"{nome}: y deve ser inteiro 0/1; dtype {rotulos.dtype}")
    if set(np.unique(rotulos).tolist()) != set(CLASSES):
        raise ModeloInvalido(f"{nome}: y deve conter exatamente as classes {CLASSES}")
    return rotulos.astype(np.int64)


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class ModeloConstruido:
    """Modelo autorizado e NÃO ajustado. Use ``construir``; reutilizável entre ajustes."""

    espec: cfg.EspecModelo
    atividade: str
    autorizacao: guardas.Autorizacao
    estimador: Pipeline = field(repr=False)
    decisoes: Mapping[str, cfg.Decisao] = field(repr=False)
    repo: Path = field(repr=False)

    @property
    def nome(self) -> str:
        return self.espec.nome

    @property
    def features(self) -> Tuple[str, ...]:
        return self.espec.features

    def reautorizar(self) -> guardas.Autorizacao:
        """Reemite a autorização agora; tem de ser idêntica à da construção."""
        atual = guardas.autorizar_modelo(self.espec, self.atividade, self.decisoes, self.repo)
        if atual != self.autorizacao:
            raise guardas.AprovacaoNaoVerificada(
                f"{self.nome}: a autorização mudou desde a construção; construa o modelo de novo")
        return atual


@dataclass(frozen=True, eq=False)
class ModeloAjustado:
    """Modelo ajustado por ``ajustar``. A predição reemite a autorização."""

    construido: ModeloConstruido
    estimador: Pipeline = field(repr=False)
    n_treino: int

    @property
    def nome(self) -> str:
        return self.construido.nome

    @property
    def features(self) -> Tuple[str, ...]:
        return self.construido.features

    def prever_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidade da classe 1, uma por linha de ``X``."""
        self.construido.reautorizar()
        matriz = _matriz(X, self.features, self.nome)
        if tuple(self.estimador.classes_.tolist()) != CLASSES:
            raise ModeloInvalido(f"{self.nome}: classes ajustadas diferentes de {CLASSES}")
        return self.estimador.predict_proba(matriz)[:, 1]

    def prever_classe(self, X: pd.DataFrame) -> np.ndarray:
        """Classe com o threshold explícito ``config.THRESHOLD`` sobre ``prever_proba``."""
        return (self.prever_proba(X) >= cfg.THRESHOLD).astype(np.int64)


def construir(espec: cfg.EspecModelo, atividade: guardas.Atividade,
              decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
              repo: Union[str, Path] = guardas.REPO_ROOT) -> ModeloConstruido:
    """Autoriza ``espec`` na atividade e devolve o estimador não ajustado."""
    if not isinstance(espec, cfg.EspecModelo):
        raise TypeError("construir exige um config.EspecModelo")
    autorizacao = guardas.autorizar_modelo(espec, atividade, decisoes, repo)
    return ModeloConstruido(espec, autorizacao.atividade, autorizacao, _estimador(espec),
                            MappingProxyType(dict(decisoes)), Path(repo))


def construir_fixo(nome: str, atividade: guardas.Atividade,
                   decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
                   repo: Union[str, Path] = guardas.REPO_ROOT) -> ModeloConstruido:
    """M0 ou M1; M2, M3 ou M4 só com P3 aprovada e verificada."""
    return construir(cfg.modelo_fixo(nome, decisoes), atividade, decisoes, repo)


def modelos_do_fold(selecao: SelecaoFold, atividade: guardas.Atividade,
                    decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
                    repo: Union[str, Path] = guardas.REPO_ROOT) -> Mapping[str, ModeloConstruido]:
    """M2v, M3v e M4v do fold externo, com o S_k já calculado (sem refazer a seleção)."""
    if not isinstance(selecao, SelecaoFold):
        raise TypeError("modelos_do_fold exige um selecao.SelecaoFold")
    return MappingProxyType({nome: construir(espec, atividade, decisoes, repo)
                             for nome, espec in selecao.modelos().items()})


def ajustar(modelo: ModeloConstruido, X: pd.DataFrame, y: Any) -> ModeloAjustado:
    """Reautoriza, clona o estimador construído e o ajusta em ``X``/``y``."""
    if not isinstance(modelo, ModeloConstruido):
        raise TypeError("ajustar exige um ModeloConstruido (use construir)")
    modelo.reautorizar()
    matriz = _matriz(X, modelo.features, modelo.nome)
    rotulos = _rotulos(y, X, modelo.nome)
    estimador = clone(modelo.estimador).fit(matriz, rotulos)
    return ModeloAjustado(modelo, estimador, len(rotulos))
