"""Ajuste interno de hiperparâmetros do benchmark S6 (P9, Seção 5.2).

Uma chamada de ``ajustar`` corresponde aos DADOS de uma unidade
``(r, k, modelo)`` (P9, Seção 6): só o treino da dobra externa ``k`` é usado.

* ``lr_tuning``, ``svm``, ``rf`` e ``xgb``: ``GridSearchCV`` com a grade
  explícita de ``modelos.grade_busca`` (um ponto por dicionário, na ordem de
  simplicidade), ``scoring="roc_auc"``, as cinco divisões internas de
  ``DobraExterna.cv_interno()``, ``n_jobs=1``, ``error_score="raise"``,
  ``return_train_score=False`` e ``refit`` com a função de escolha abaixo. O
  próprio ``GridSearchCV`` reajusta a configuração escolhida no treino
  externo inteiro, uma vez.
* ``lr_fixa``: um único ajuste no treino externo, sem busca.

**Escolha** (``escolher_configuracao``): a média da AUC das cinco dobras
internas; vale a maior média, e toda média a até ``1e-9`` do máximo empata
com ele; o empate vai para a configuração que vem primeiro na ordem de
simplicidade.

Nada da avaliação externa entra no ajuste: as características e os rótulos
da avaliação externa nem são lidos, e o módulo não prediz nem calcula
métricas externas. Os ajustes rodam com o BLAS limitado a uma thread
(``threadpoolctl``). Avisos são capturados e registrados, nunca descartados.
Exceção no ajuste, pontuação interna ``NaN`` ou infinita e invariante violada
levantam ``FalhaAjuste`` (falha técnica, P9, Seção 6). A dobra recebida é
revalidada pela API pública ``folds.validar_dobra``.

A AUC interna é a do ``scoring="roc_auc"`` do scikit-learn 1.3.2, que usa
``decision_function`` quando o estimador a tem (regressão logística e SVM) e
``predict_proba`` nos demais (RF e XGBoost). É o comportamento documentado
dessa versão; não há avaliador próprio.

``ResultadoAjuste.registro()`` traz configuração escolhida, posição na grade,
borda de cada hiperparâmetro, fingerprint da dobra e avisos, sem
identificadores e sem pontuações. As médias internas ficam só em memória
(``ResultadoAjuste.medias_internas``), para o diretório local; não são
relatadas (P9, Seção 14.3).
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from . import adaptadores_s4 as s4
from . import config as cfg
from . import folds
from . import modelos
from .dados import DadosBenchmark

FASE_BUSCA = "busca"
FASE_AJUSTE = "ajuste"


class FalhaAjuste(RuntimeError):
    """Falha técnica do ajuste interno (P9, Seção 6): a unidade não produz resultado."""

    chave_saida: str = "falha_tecnica"


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AvisoAjuste:
    """Aviso capturado num ajuste, agregado por fase, categoria e mensagem."""

    fase: str
    categoria: str
    mensagem: str
    ocorrencias: int

    @property
    def convergencia(self) -> bool:
        return self.categoria == "ConvergenceWarning"


@dataclass(frozen=True)
class BordaHiperparametro:
    """Posição do valor escolhido entre os valores declarados do hiperparâmetro."""

    valor: Any
    posicao: int
    n_valores: int

    @property
    def na_borda(self) -> bool:
        return self.posicao in (0, self.n_valores - 1)


@dataclass(frozen=True, eq=False)
class ResultadoAjuste:
    modelo: str
    repeticao: int
    dobra: int
    fingerprint_dobra: str
    #: Configuração escolhida, com os nomes da grade (sem prefixo do ``Pipeline``).
    configuracao: Mapping[str, Any]
    #: Posição da configuração na ordem de simplicidade (0 = mais simples).
    indice_configuracao: int
    n_configuracoes: int
    borda: Mapping[str, BordaHiperparametro]
    avisos: Tuple[AvisoAjuste, ...]
    n_ajustes: int
    #: Médias da AUC interna por configuração, na ordem da grade. Só em memória.
    medias_internas: Tuple[float, ...] = field(repr=False)
    #: Estimador reajustado no treino externo inteiro.
    estimador: Any = field(repr=False)

    @property
    def avisos_convergencia(self) -> int:
        return sum(a.ocorrencias for a in self.avisos if a.convergencia)

    def registro(self) -> Dict[str, Any]:
        """Registro JSON da escolha: sem identificadores, sem pontuações internas."""
        return {
            "modelo": self.modelo,
            "repeticao": self.repeticao,
            "dobra_externa": self.dobra,
            "fingerprint_dobra": self.fingerprint_dobra,
            "configuracao": dict(self.configuracao),
            "indice_configuracao": self.indice_configuracao,
            "n_configuracoes": self.n_configuracoes,
            "borda": {nome: {"valor": b.valor, "posicao": b.posicao, "n_valores": b.n_valores,
                             "na_borda": b.na_borda} for nome, b in self.borda.items()},
            "na_borda_da_grade": any(b.na_borda for b in self.borda.values()),
            "n_ajustes": self.n_ajustes,
            "avisos": [{"fase": a.fase, "categoria": a.categoria, "mensagem": a.mensagem,
                        "ocorrencias": a.ocorrencias} for a in self.avisos],
            "avisos_convergencia": self.avisos_convergencia,
        }


# --------------------------------------------------------------------------- #
# Escolha da configuração
# --------------------------------------------------------------------------- #


def _pontuacoes_por_dobra(cv_results: Mapping[str, Any], n_dobras: int, n_configuracoes: int) -> np.ndarray:
    try:
        colunas = [np.asarray(cv_results[f"split{i}_test_score"], dtype=np.float64) for i in range(n_dobras)]
    except (KeyError, TypeError, ValueError) as exc:
        raise FalhaAjuste(f"pontuações internas ausentes ou inválidas: {exc}") from exc
    if f"split{n_dobras}_test_score" in cv_results:
        raise FalhaAjuste(f"mais de {n_dobras} dobras internas pontuadas")
    if any(c.shape != (n_configuracoes,) for c in colunas):
        raise FalhaAjuste(f"pontuações internas com formas {[c.shape for c in colunas]}; "
                          f"esperadas {n_configuracoes} por dobra")
    # NaN ou infinito numa dobra torna a média não finita, recusada em ``escolher_configuracao``.
    return np.column_stack(colunas)


def escolher_configuracao(medias: Sequence[float]) -> int:
    """Índice da escolhida: a primeira, na ordem de simplicidade, a até ``tolerancia_empate`` da maior média.

    A tolerância (1e-9) vem sempre da configuração canônica; a comparação é
    inclusiva: média ``>= máximo − tolerância`` empata com o máximo.
    """
    valores = np.asarray(medias, dtype=np.float64)
    if valores.ndim != 1 or valores.size == 0:
        raise FalhaAjuste("médias internas devem ser um vetor não vazio")
    if not np.all(np.isfinite(valores)):
        raise FalhaAjuste("média interna NaN ou infinita")
    tolerancia = cfg.CONFIGURACAO.ajuste_interno.tolerancia_empate
    return int(np.flatnonzero(valores >= valores.max() - tolerancia)[0])


def _refit(configuracoes: Tuple[Mapping[str, Any], ...], n_dobras: int):
    """Função ``refit`` do ``GridSearchCV``: confere a grade e devolve a escolhida."""

    def escolher(cv_results: Mapping[str, Any]) -> int:
        params = [dict(p) for p in cv_results["params"]]
        if params != [dict(c) for c in configuracoes]:
            raise FalhaAjuste("candidatos do GridSearchCV fora da ordem de simplicidade")
        matriz = _pontuacoes_por_dobra(cv_results, n_dobras, len(configuracoes))
        return escolher_configuracao(matriz.mean(axis=1))

    return escolher


# --------------------------------------------------------------------------- #
# Entradas
# --------------------------------------------------------------------------- #


def _exigir_caracteristicas(X: Any, amostra: s4.Amostra, treino: np.ndarray) -> np.ndarray:
    """Matriz alinhada à amostra; só as linhas do treino externo são lidas (e conferidas)."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X deve ser numpy.ndarray (DadosBenchmark.X())")
    n_caracteristicas = len(cfg.CONFIGURACAO.caracteristicas)
    if X.ndim != 2 or X.shape != (amostra.n, n_caracteristicas):
        raise FalhaAjuste(f"X com forma {X.shape}; esperada {(amostra.n, n_caracteristicas)}, alinhada à amostra")
    if X.dtype != np.float64:
        raise FalhaAjuste(f"X deve ser float64; dtype {X.dtype}")
    X_treino = np.array(X[treino], dtype=np.float64, copy=True)
    if not np.all(np.isfinite(X_treino)):
        raise FalhaAjuste("característica NaN ou infinita no treino externo")
    return X_treino


# --------------------------------------------------------------------------- #
# Ajuste
# --------------------------------------------------------------------------- #


def _agregar_avisos(registrados: List[warnings.WarningMessage], fase: str) -> Tuple[AvisoAjuste, ...]:
    contagem = Counter((w.category.__name__, str(w.message)) for w in registrados)
    return tuple(AvisoAjuste(fase, categoria, mensagem, n) for (categoria, mensagem), n in sorted(contagem.items()))


def _executar(funcao, fase: str):
    """Roda ``funcao`` com o BLAS em uma thread, capturando TODOS os avisos e convertendo falhas."""
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as registrados:
        warnings.simplefilter("always")
        try:
            resultado = funcao()
        except FalhaAjuste:
            raise
        except Exception as exc:  # P9, Seção 6: exceção no ajuste é falha técnica
            raise FalhaAjuste(f"{fase}: {type(exc).__name__}: {exc}") from exc
    return resultado, _agregar_avisos(registrados, fase)


def _borda(espec: cfg.EspecModelo, configuracao: Mapping[str, Any]) -> Dict[str, BordaHiperparametro]:
    borda = {}
    for p in espec.grade:
        valores = p.materializar()
        borda[p.nome] = BordaHiperparametro(configuracao[p.nome], valores.index(configuracao[p.nome]), len(valores))
    return borda


def _buscar(estimador: Any, grade: List[Dict[str, List[Any]]], configuracoes: Tuple[Mapping[str, Any], ...],
            X_treino: np.ndarray, y_treino: np.ndarray, cv: Sequence[Tuple[np.ndarray, np.ndarray]]) -> GridSearchCV:
    ajuste = cfg.CONFIGURACAO.ajuste_interno
    busca = GridSearchCV(estimador, param_grid=grade, scoring=ajuste.scoring, cv=list(cv),
                         refit=_refit(configuracoes, len(cv)), n_jobs=ajuste.n_jobs, error_score=ajuste.error_score,
                         return_train_score=ajuste.return_train_score)
    return busca.fit(X_treino, y_treino)


def ajustar(X: np.ndarray, amostra: s4.Amostra, dobra: folds.DobraExterna, modelo: str) -> ResultadoAjuste:
    """Ajuste interno e reajuste do ``modelo`` no treino da ``dobra`` externa.

    ``X``: características ``float64`` na ordem da configuração, com uma linha
    por linha da ``amostra``.

    **Pré-condição: alinhamento posicional.** A linha ``i`` de ``X`` precisa ser
    o nódulo ``amostra.ids[i]``. A função confere forma, ``dtype`` e finitude,
    mas isso NÃO identifica uma permutação arbitrária das linhas: ``X`` em
    outra ordem passaria e treinaria com rótulos trocados. Para cargas
    oficiais, use ``ajustar_dados``, que tira ``X`` e ``Amostra`` da mesma
    carga e garante o alinhamento por construção.

    A dobra é revalidada por ``folds.validar_dobra``; violação levanta
    ``folds.DobrasInvalidas`` (falha técnica) ou ``folds.PlanoDivergente``
    (fingerprint divergente, integridade).
    """
    if not isinstance(amostra, s4.Amostra):
        raise TypeError("amostra deve ser a Amostra da S4 (DadosBenchmark.amostra())")
    espec = modelos.especificacao(modelo)
    folds.validar_dobra(amostra, dobra)
    treino = dobra.externa.treino
    X_treino = _exigir_caracteristicas(X, amostra, treino)
    y_treino = np.array(amostra.y[treino], dtype=np.int64, copy=True)
    n_internas = cfg.CONFIGURACAO.dobras.n_dobras_internas

    if not espec.grade:
        estimador, avisos = _executar(lambda: modelos.criar_estimador(modelo).fit(X_treino, y_treino), FASE_AJUSTE)
        indice, configuracao, medias = 0, {}, ()
    else:
        cv = dobra.cv_interno()  # cinco divisões, garantidas por ``folds.validar_dobra``
        configuracoes = modelos.configuracoes(modelo)
        busca, avisos = _executar(lambda: _buscar(modelos.criar_estimador(modelo), modelos.grade_busca(modelo),
                                                  configuracoes, X_treino, y_treino, cv), FASE_BUSCA)
        indice = int(busca.best_index_)
        matriz = _pontuacoes_por_dobra(busca.cv_results_, n_internas, len(configuracoes))
        if indice != escolher_configuracao(matriz.mean(axis=1)):
            raise FalhaAjuste("índice escolhido pelo GridSearchCV difere da regra de escolha")
        estimador = busca.best_estimator_
        escolhida = configuracoes[indice]
        parametros = estimador.get_params()
        if any(parametros[nome] != valor for nome, valor in escolhida.items()):
            raise FalhaAjuste("estimador reajustado com configuração diferente da escolhida")
        prefixo = modelos.PREFIXO_CLASSIFICADOR if espec.padronizacao else ""
        configuracao = {nome[len(prefixo):]: valor for nome, valor in escolhida.items()}
        medias = tuple(float(m) for m in matriz.mean(axis=1))

    try:
        check_is_fitted(estimador)
    except Exception as exc:
        raise FalhaAjuste(f"estimador não ajustado depois do reajuste: {exc}") from exc
    if list(getattr(estimador, "classes_", [])) != [0, 1]:
        raise FalhaAjuste("estimador reajustado sem as classes 0 e 1")

    return ResultadoAjuste(
        modelo=modelo, repeticao=dobra.repeticao, dobra=dobra.dobra, fingerprint_dobra=dobra.fingerprint,
        configuracao=MappingProxyType(configuracao), indice_configuracao=indice,
        n_configuracoes=espec.n_configuracoes, borda=MappingProxyType(_borda(espec, configuracao)),
        avisos=avisos, n_ajustes=espec.ajustes_por_dobra_externa(n_internas),
        medias_internas=medias, estimador=estimador)


def ajustar_dados(dados: DadosBenchmark, dobra: folds.DobraExterna, modelo: str) -> ResultadoAjuste:
    """``ajustar`` com ``X`` e ``Amostra`` tirados da mesma carga, alinhados por construção.

    Entrada preferencial para cargas oficiais: elimina a pré-condição de
    alinhamento posicional que ``ajustar`` não consegue verificar.
    """
    if not isinstance(dados, DadosBenchmark):
        raise TypeError("dados deve ser DadosBenchmark")
    return ajustar(dados.X(), dados.amostra(), dobra, modelo)


__all__ = [
    "AvisoAjuste", "BordaHiperparametro", "FASE_AJUSTE", "FASE_BUSCA", "FalhaAjuste", "ResultadoAjuste",
    "ajustar", "ajustar_dados", "escolher_configuracao",
]
