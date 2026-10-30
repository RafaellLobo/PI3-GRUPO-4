"""Transformadores reutilizáveis da seleção radiômica — Cartão 4, Sprint 4.

Passos 2 e 3 do procedimento de seleção (protocolo, Seção 5.3), como
componentes do pipeline que a CV interna ajusta em cada TREINO interno
(Seção 5.4):

* variância — ``sklearn.feature_selection.VarianceThreshold`` com
  ``ESPEC_SELECAO.limiar_variancia``, sem reimplementação;
* correlação — ``FiltroCorrelacao``, com ``ESPEC_SELECAO.limiar_correlacao``.

Regra de correlação, idêntica ao código auditado
(``scripts/sprint4/reproduce_juan_feature_selection.py``)::

    corr = X_train[f_variancia].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundantes = [col for col in upper.columns if (upper[col] > 0.90).any()]

* a coluna ``j`` é removida se ``|r(i, j)| > limiar`` (estritamente) para
  ALGUMA coluna anterior ``i < j`` — inclusive uma ``i`` que também foi
  removida: não é um filtro guloso;
* o sinal não importa (valor absoluto); a primeira coluna nunca é removida,
  logo o filtro nunca esvazia o conjunto;
* correlação indefinida (NaN, por exemplo coluna constante) nunca remove
  nada, porque ``NaN > limiar`` é falso;
* o resultado depende da ordem das colunas, que é preservada.

A correlação é a mesma função do código auditado (``DataFrame.corr`` de
Pearson, ``min_periods=1``), para que valores próximos do limiar coincidam
bit a bit. ``fit`` aprende só a máscara ``support_`` a partir dos dados
recebidos e ignora ``y``; ``transform`` aplica a máscara e nunca recalcula
correlações.

O módulo não treina modelos, não lê dados e não implementa LASSO.
"""

from __future__ import annotations

import numbers
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted

from . import config as cfg

PASSO_VARIANCIA = "variancia"
PASSO_CORRELACAO = "correlacao"


def correlacao_absoluta(X: np.ndarray) -> np.ndarray:
    """``|r de Pearson|`` entre colunas, pela mesma função do código auditado.

    ``DataFrame.corr`` (Pearson, ``min_periods=1``) seguido de ``abs``.
    Coluna sem variância produz NaN nos seus pares.
    """
    return pd.DataFrame(X).corr(method="pearson", min_periods=1).abs().to_numpy()


def mascara_correlacao(corr_abs: np.ndarray, limiar: float) -> np.ndarray:
    """Máscara das colunas mantidas pela regra histórica.

    Remove a coluna ``j`` se ``corr_abs[i, j] > limiar`` para alguma ``i < j``,
    removida ou não; NaN nunca remove.
    """
    corr_abs = np.asarray(corr_abs, dtype=np.float64)
    if corr_abs.ndim != 2 or corr_abs.shape[0] != corr_abs.shape[1]:
        raise ValueError("a matriz de correlação deve ser quadrada")
    superior = np.triu(np.ones(corr_abs.shape, dtype=bool), k=1)
    with np.errstate(invalid="ignore"):
        acima = superior & (corr_abs > limiar)
    return ~acima.any(axis=0)


def _validar_limiar(limiar: Any) -> float:
    if isinstance(limiar, bool) or not isinstance(limiar, numbers.Real) or \
            not np.isfinite(limiar) or not 0 <= limiar < 1:
        raise ValueError(f"limiar deve ser real em [0, 1); recebido {limiar!r}")
    return float(limiar)


def _exigir_numerico(X: Any) -> None:
    """Recusa texto, objetos e complexos, que ``check_array`` poderia converter em silêncio."""
    tipos = getattr(X, "dtypes", None)
    if tipos is not None and not isinstance(tipos, np.dtype):
        ok = all(pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_complex_dtype(t) for t in tipos)
    else:
        ok = np.asarray(X).dtype.kind in "biuf"
    if not ok:
        raise ValueError("entrada deve ser numérica (booleano, inteiro ou real)")


def _nomes_colunas(X: Any) -> Optional[List[str]]:
    """Nomes das colunas de um DataFrame com nomes textuais (critério do sklearn); senão None."""
    colunas = getattr(X, "columns", None)
    if colunas is None:
        return None
    nomes = list(colunas)
    if not all(isinstance(nome, str) for nome in nomes):
        return None
    return nomes


def _recusar_esparsa(X: Any) -> None:
    if sparse.issparse(X):
        raise TypeError("matriz esparsa não é aceita")


class FiltroCorrelacao(SelectorMixin, BaseEstimator):
    """Remove colunas redundantes pela regra de correlação do código auditado.

    Compatível com ``Pipeline``, ``GridSearchCV`` e ``sklearn.clone``.
    ``get_support`` e ``get_feature_names_out`` (de ``SelectorMixin``) mapeiam
    as colunas mantidas às colunas de entrada, na ordem original.

    Atributos aprendidos: ``support_`` (somente leitura), ``limiar_``,
    ``n_features_in_`` e, para DataFrame com nomes textuais,
    ``feature_names_in_``.

    Entradas recusadas: esparsas, não numéricas, não finitas, não
    bidimensionais, sem colunas, com menos de duas linhas no ajuste (a
    correlação não é definida) ou com nomes repetidos. ``transform`` exige as
    mesmas colunas do ajuste — em número, nomes e ordem — e recusa a troca
    entre entrada com e sem nomes.
    """

    def __init__(self, limiar: float = cfg.ESPEC_SELECAO.limiar_correlacao):
        self.limiar = limiar

    def fit(self, X: Any, y: Any = None) -> "FiltroCorrelacao":
        """Aprende a máscara só com ``X``; ``y`` é ignorado."""
        limiar = _validar_limiar(self.limiar)
        _recusar_esparsa(X)
        nomes = _nomes_colunas(X)
        if nomes is not None and len(set(nomes)) != len(nomes):
            raise ValueError("nomes de colunas repetidos")
        _exigir_numerico(X)
        matriz = self._validate_data(X, dtype=np.float64, ensure_min_samples=2, reset=True)
        suporte = mascara_correlacao(correlacao_absoluta(matriz), limiar)
        suporte.setflags(write=False)
        self.support_ = suporte
        self.limiar_ = limiar
        return self

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.support_

    def transform(self, X: Any) -> Any:
        """Aplica a máscara aprendida; não recalcula correlações."""
        check_is_fitted(self, "support_")
        _recusar_esparsa(X)
        nomes = _nomes_colunas(X)
        esperados = getattr(self, "feature_names_in_", None)
        if esperados is None and nomes is not None:
            raise ValueError("ajustado sem nomes de colunas; transform recebeu colunas nomeadas")
        if esperados is not None and nomes != list(esperados):
            raise ValueError("colunas diferentes das do ajuste (nomes, ordem ou ausência de nomes)")
        _exigir_numerico(X)
        matriz = check_array(X, dtype="numeric")
        if matriz.shape[1] != self.n_features_in_:
            raise ValueError(f"{matriz.shape[1]} colunas; o ajuste usou {self.n_features_in_}")
        return super().transform(X)


def filtros_selecao() -> Pipeline:
    """Passos 2 e 3 da Seção 5.3, com os limiares de ``config``.

    ``VarianceThreshold`` do sklearn seguido de ``FiltroCorrelacao``. Feito
    para ser o início do pipeline que a CV interna ajusta em cada treino.
    """
    espec = cfg.ESPEC_SELECAO
    return Pipeline([
        (PASSO_VARIANCIA, VarianceThreshold(threshold=espec.limiar_variancia)),
        (PASSO_CORRELACAO, FiltroCorrelacao(limiar=espec.limiar_correlacao)),
    ])
