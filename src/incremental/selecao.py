"""Seleção radiômica LASSO ajustada para a CV aninhada — Cartão 4, Sprint 4.

Procedimento da Seção 5.4 do protocolo, aplicado a UM treino externo k
para produzir S_k (Seção 6.3):

1. folds de estabilidade e internos de ``folds.plano_repeticao``, todos
   agrupados por paciente e contidos no treino externo;
2. em cada fold de estabilidade j, ``GridSearchCV`` sobre o pipeline COMPLETO
   ``VarianceThreshold → FiltroCorrelacao → StandardScaler →
   LogisticRegression(L1)``, com a CV interna de 4 folds. Filtros e scaler
   fazem parte do estimador clonado em cada treino interno: são ajustados
   DENTRO de cada fold interno, nunca antes da CV (correção da Seção 5.3);
3. ``refit`` do pipeline no treino do fold de estabilidade com o ``C``
   escolhido; selecionadas as features com ``|coef| > limiar_coeficiente``,
   mapeadas às colunas originais pela composição das máscaras dos filtros;
4. S_k = features selecionadas em pelo menos ``limiar_estabilidade`` dos
   folds de estabilidade, na ordem de ``features`` (a do esquema oficial).
   S_k pode ser vazio; nenhuma feature substituta é inventada.

Grade de ``C``, solver, seeds, limiares e regra 3 de 5 vêm de
``config.ESPEC_SELECAO`` (lida a cada chamada); as seeds das CVs vêm das
divisões de ``folds``.

Somente as linhas do treino externo são lidas de ``X``: as linhas da
avaliação externa nunca são indexadas, nem para checar valores. As
divisões de ``folds`` usam posições GLOBAIS na amostra; a ``GridSearchCV``
recebe posições LOCAIS nas linhas do fold de estabilidade, convertidas por
``posicoes_locais``, que recusa qualquer posição fora do nível.

``GridSearchCV`` roda com ``n_jobs=1`` (warnings de processos paralelos não
chegam ao processo principal e seriam perdidos) e ``error_score="raise"``
(falha de ajuste interrompe, em vez de virar NaN em silêncio). Warnings não
são suprimidos: são contados por categoria e mensagem.

O procedimento histórico auditado
(``scripts/sprint4/reproduce_juan_feature_selection.py``) permanece intacto
e não é importado. Esta seleção corrigida NÃO reproduz seus resultados nem
seus hashes: filtros ajustados em cada treino interno e seleção restrita ao
treino externo mudam, por construção, os dados vistos em cada passo.

O módulo não lê dados, não ajusta modelos de avaliação e não calcula
métricas de validação ou teste.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from . import folds as fl
from . import rastreabilidade as rst
from . import transformadores as tf

PASSO_SCALER = "scaler"
PASSO_L1 = "clf"
PARAMETRO_C = f"{PASSO_L1}__C"


class SelecaoInvalida(ValueError):
    """Entrada incompatível ou violação de invariante da seleção."""


# --------------------------------------------------------------------------- #
# Estimadores
# --------------------------------------------------------------------------- #


def pipeline_selecao() -> Pipeline:
    """Filtros → scaler → L1, com os parâmetros de ``ESPEC_SELECAO``."""
    espec = cfg.ESPEC_SELECAO
    return Pipeline([
        *tf.filtros_selecao().steps,
        (PASSO_SCALER, StandardScaler()),
        (PASSO_L1, LogisticRegression(penalty=espec.penalty, solver=espec.solver,
                                      max_iter=espec.max_iter, random_state=espec.random_state)),
    ])


def busca_C(cv: Sequence[Tuple[np.ndarray, np.ndarray]]) -> GridSearchCV:
    """``GridSearchCV`` do pipeline completo sobre a grade de ``C``, com divisões explícitas."""
    espec = cfg.ESPEC_SELECAO
    return GridSearchCV(pipeline_selecao(), {PARAMETRO_C: list(espec.grade_C.valores())}, cv=list(cv),
                        scoring=espec.scoring, refit=True, n_jobs=1, error_score="raise")


# --------------------------------------------------------------------------- #
# Índices e entradas
# --------------------------------------------------------------------------- #


def posicoes_locais(base: np.ndarray, globais: np.ndarray) -> np.ndarray:
    """Posições de ``globais`` dentro de ``base`` (ambos crescentes); recusa o que não estiver em ``base``."""
    base = np.asarray(base)
    globais = np.asarray(globais)
    if base.ndim != 1 or globais.ndim != 1 or not (np.all(np.diff(base) > 0) and np.all(np.diff(globais) > 0)):
        raise SelecaoInvalida("posições devem ser vetores estritamente crescentes")
    locais = np.searchsorted(base, globais)
    if np.any(locais >= len(base)) or not np.array_equal(base[np.minimum(locais, len(base) - 1)], globais):
        raise SelecaoInvalida("posição fora das linhas do nível")
    return locais.astype(np.int64)


def _validar_features(features: Any) -> Tuple[str, ...]:
    if isinstance(features, (str, bytes)) or not isinstance(features, (tuple, list)):
        raise TypeError("features deve ser tupla ou lista de nomes, na ordem do esquema")
    features = cfg.validar_nomes(tuple(features), "features")
    if not features:
        raise SelecaoInvalida("nenhuma feature candidata")
    return features


def _matriz(X: Any, features: Tuple[str, ...], n: int) -> np.ndarray:
    colunas = getattr(X, "columns", None)
    if colunas is not None and list(colunas) != list(features):
        raise SelecaoInvalida("colunas de X diferentes de features (nomes ou ordem)")
    tipos = getattr(X, "dtypes", None)
    numerico = (all(getattr(t, "kind", "O") in "biuf" for t in tipos) if colunas is not None
                else np.asarray(X).dtype.kind in "biuf")
    if not numerico:
        raise SelecaoInvalida("X deve ser numérico")
    matriz = np.asarray(X, dtype=np.float64)
    if matriz.ndim != 2 or matriz.shape != (n, len(features)):
        raise SelecaoInvalida(f"X deve ter forma ({n}, {len(features)}); tem {matriz.shape}")
    return matriz


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FoldEstabilidade:
    """Resultado técnico agregado de um fold de estabilidade (sem identificadores)."""

    fold_estabilidade: int
    seed_interna: int
    n_nodulos: int
    n_pacientes: int
    C: float
    n_pos_variancia: int
    n_pos_correlacao: int
    #: Coeficientes do refit, uma entrada por feature que chegou ao L1, na ordem do esquema.
    coeficientes: Tuple[Tuple[str, float], ...]
    selecionadas: Tuple[str, ...]
    n_iter: int
    convergiu: bool
    #: (categoria, mensagem saneada, contagem), CV interna e refit juntos.
    warnings: Tuple[Tuple[str, str, int], ...]

    def como_dict(self) -> dict:
        return {
            "fold_estabilidade": self.fold_estabilidade, "seed_interna": self.seed_interna,
            "n_nodulos": self.n_nodulos, "n_pacientes": self.n_pacientes, "C": self.C,
            "n_pos_variancia": self.n_pos_variancia, "n_pos_correlacao": self.n_pos_correlacao,
            "coeficientes": [[nome, valor] for nome, valor in self.coeficientes],
            "selecionadas": list(self.selecionadas), "n_iter": self.n_iter, "convergiu": self.convergiu,
            "warnings": [{"categoria": c, "mensagem": m, "contagem": n} for c, m, n in self.warnings],
        }


@dataclass(frozen=True)
class SelecaoFold:
    """S_k de um fold externo, calculado uma vez e reutilizado por M2v, M3v e M4v."""

    repeticao: int
    fold_externo: int
    sha256_amostra: str
    fingerprint_folds: str
    hash_configuracao: str
    features: Tuple[str, ...]
    folds: Tuple[FoldEstabilidade, ...]
    frequencia: Mapping[str, int]
    s_k: Tuple[str, ...]

    @property
    def vazio(self) -> bool:
        return not self.s_k

    def modelos(self) -> Mapping[str, cfg.EspecModelo]:
        """M2v, M3v e M4v deste fold, todos com o MESMO S_k (Seção 6.3)."""
        return MappingProxyType({nome: cfg.modelo_variavel(nome, self.s_k) for nome in cfg.MODELOS_VARIAVEIS})

    def como_dict(self) -> dict:
        """Forma serializável (JSON canônico), sem identificadores individuais."""
        dados = {
            "repeticao": self.repeticao, "fold_externo": self.fold_externo,
            "sha256_amostra": self.sha256_amostra, "fingerprint_folds": self.fingerprint_folds,
            "hash_configuracao": self.hash_configuracao, "features": list(self.features),
            "folds_estabilidade": [f.como_dict() for f in self.folds],
            "frequencia": dict(self.frequencia), "s_k": list(self.s_k), "s_k_vazio": self.vazio,
        }
        _ = rst.json_canonico(dados)
        rst._exigir_sem_dados_individuais(dados, "selecao")
        return dados


# --------------------------------------------------------------------------- #
# Procedimento
# --------------------------------------------------------------------------- #


def consolidar(features: Sequence[str], selecionadas_por_fold: Iterable[Iterable[str]]
               ) -> Tuple[Mapping[str, int], Tuple[str, ...]]:
    """Frequência por feature e S_k (frequência ≥ ``limiar_estabilidade``), na ordem de ``features``."""
    espec = cfg.ESPEC_SELECAO
    features = _validar_features(features)
    conjuntos: List[set] = []
    for selecionadas in selecionadas_por_fold:
        if isinstance(selecionadas, (str, bytes)):
            raise TypeError("cada fold deve trazer uma coleção de nomes")
        lista = list(selecionadas)
        if len(set(lista)) != len(lista) or not set(lista) <= set(features):
            raise SelecaoInvalida("seleção com nome repetido ou fora das features")
        conjuntos.append(set(lista))
    if len(conjuntos) != espec.n_folds_estabilidade:
        raise SelecaoInvalida(f"{len(conjuntos)} folds; esperados {espec.n_folds_estabilidade}")
    frequencia = {f: sum(f in s for s in conjuntos) for f in features}
    s_k = tuple(f for f in features if frequencia[f] >= espec.limiar_estabilidade)
    return MappingProxyType(frequencia), s_k


def _resumir_warnings(capturados: Iterable[warnings.WarningMessage]) -> Tuple[Tuple[str, str, int], ...]:
    contagem = Counter((w.category.__name__, rst._sanear_mensagem(str(w.message))) for w in capturados)
    return tuple(sorted((c, m, n) for (c, m), n in contagem.items()))


def _ajustar_fold_estabilidade(X_est: np.ndarray, y_est: np.ndarray, n_pacientes: int,
                               splits: List[Tuple[np.ndarray, np.ndarray]], features: Tuple[str, ...],
                               estabilidade: fl.Divisao, seed_interna: int) -> FoldEstabilidade:
    espec = cfg.ESPEC_SELECAO
    busca = busca_C(splits)
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        busca.fit(X_est, y_est)
    melhor = busca.best_estimator_
    variancia = melhor.named_steps[tf.PASSO_VARIANCIA]
    correlacao = melhor.named_steps[tf.PASSO_CORRELACAO]
    # Coluna i que chega ao L1 = feature original indices[i] (máscaras compostas).
    indices = np.flatnonzero(variancia.get_support())[correlacao.get_support(indices=True)]
    nomes = tuple(features[i] for i in indices)
    pelos_filtros = correlacao.get_feature_names_out(variancia.get_feature_names_out(list(features)))
    if tuple(pelos_filtros) != nomes:
        raise SelecaoInvalida("mapeamento de colunas inconsistente entre filtros")
    classificador = melhor.named_steps[PASSO_L1]
    coef = np.asarray(classificador.coef_, dtype=np.float64).ravel()
    if coef.shape != (len(nomes),):
        raise SelecaoInvalida("coeficientes não correspondem às colunas filtradas")
    n_iter = int(np.max(classificador.n_iter_))
    return FoldEstabilidade(
        fold_estabilidade=estabilidade.fold_estabilidade,
        seed_interna=seed_interna,
        n_nodulos=len(y_est),
        n_pacientes=n_pacientes,
        C=float(busca.best_params_[PARAMETRO_C]),
        n_pos_variancia=int(variancia.get_support().sum()),
        n_pos_correlacao=len(nomes),
        coeficientes=tuple((nome, float(c)) for nome, c in zip(nomes, coef)),
        selecionadas=tuple(nome for nome, c in zip(nomes, coef) if abs(c) > espec.limiar_coeficiente),
        n_iter=n_iter,
        convergiu=n_iter < espec.max_iter,
        warnings=_resumir_warnings(capturados),
    )


def selecionar_fold_externo(amostra: fl.Amostra, X: Any, features: Sequence[str],
                            plano: fl.PlanoRepeticao, k: int) -> SelecaoFold:
    """S_k do treino externo ``k`` da repetição de ``plano``.

    ``X``: matriz (ou DataFrame com colunas = ``features``) alinhada por
    posição com ``amostra``. Só as linhas do treino externo são lidas. O
    plano recebido é conferido contra o plano canônico da amostra (mesma
    composição em todos os níveis); as divisões usadas são as canônicas.
    """
    if not isinstance(amostra, fl.Amostra):
        raise TypeError("amostra deve ser folds.Amostra")
    if not isinstance(plano, fl.PlanoRepeticao) or plano.sha256_amostra != amostra.sha256:
        raise SelecaoInvalida("plano de outra amostra")
    if type(k) is not int or k not in plano.estabilidade:
        raise SelecaoInvalida(f"fold externo inexistente: {k!r}")
    features = _validar_features(features)
    matriz = _matriz(X, features, amostra.n)

    canonico = fl.plano_repeticao(amostra, plano.repeticao)
    fingerprint = fl.fingerprint_unidade(amostra, canonico, k)
    if fl.fingerprint_unidade(amostra, plano, k) != fingerprint:
        raise SelecaoInvalida("plano diferente do canônico para esta amostra e repetição")

    externa = canonico.externos[k]
    X_ext = matriz[externa.treino]  # única leitura de X: linhas do treino externo
    if not np.isfinite(X_ext).all():
        raise SelecaoInvalida("valores não finitos no treino externo")
    pacientes_avaliacao = set(amostra.grupos[externa.avaliacao].tolist())

    resultados = []
    for est in canonico.estabilidade[k]:
        internas = canonico.internos[(k, est.fold_estabilidade)]
        if pacientes_avaliacao & set(amostra.grupos[est.treino].tolist()):
            raise SelecaoInvalida(f"{est.rotulo}: paciente da avaliação externa no treino de estabilidade")
        X_est = X_ext[posicoes_locais(externa.treino, est.treino)]
        splits = [(posicoes_locais(est.treino, d.treino), posicoes_locais(est.treino, d.avaliacao))
                  for d in internas]
        seeds = {d.seed for d in internas}
        if len(seeds) != 1:
            raise SelecaoInvalida(f"{est.rotulo}: seeds internas inconsistentes")
        resultados.append(_ajustar_fold_estabilidade(
            X_est, amostra.y[est.treino], len(set(amostra.grupos[est.treino].tolist())), splits, features,
            est, seeds.pop()))

    frequencia, s_k = consolidar(features, (r.selecionadas for r in resultados))
    return SelecaoFold(
        repeticao=canonico.repeticao, fold_externo=k, sha256_amostra=amostra.sha256,
        fingerprint_folds=fingerprint, hash_configuracao=cfg.hash_configuracao(), features=features,
        folds=tuple(resultados), frequencia=frequencia, s_k=s_k,
    )
