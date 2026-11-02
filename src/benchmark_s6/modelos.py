"""Fábrica dos cinco classificadores do benchmark S6 (P9, Seção 5.1).

Tudo vem de ``config.CONFIGURACAO.modelos``: a classe do estimador (pelo nome
registrado em ``EspecModelo.estimador``), os parâmetros fixos, a padronização
e a grade. Nenhum parâmetro é acrescentado aqui; os demais ficam no padrão da
biblioteca, fixada pela P9, Seção 13.

* ``lr_fixa``, ``lr_tuning`` e ``svm``: ``Pipeline`` com ``StandardScaler``
  (passo ``padronizacao``) seguido do classificador (passo ``classificador``).
  Os hiperparâmetros da grade ganham o prefixo ``classificador__``.
* ``rf`` e ``xgb``: o classificador sozinho, sem padronização.

As grades saem na ORDEM DE SIMPLICIDADE pré-registrada
(``EspecModelo.configuracoes``), como lista explícita de pontos para o
``param_grid`` do ``GridSearchCV``: a ordem dos candidatos é a ordem da grade,
o que o desempate da P9, Seção 5.2, exige.

Cada chamada devolve instâncias NOVAS, nunca ajustadas. Depois de construir,
a fábrica confere que os parâmetros fixos são exatamente os da
configuração, que o SVM não tem probabilidade nem calibração, que o XGBoost
não tem parada antecipada, que o paralelismo interno é de um processo e que
toda semente está fixada. As duas guardas do SVM e da semente valem mesmo
que a configuração diga o contrário.

O XGBoost é importado só quando ``xgb`` é pedido e precisa estar na versão
fixada (2.1.4). Ausente, incompatível ou com falha ao carregar a biblioteca
nativa, a fábrica levanta ``XGBoostIndisponivel``; nada é instalado nem
trocado. O módulo não lê dados, não treina e não calcula métricas.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg

PASSO_PADRONIZACAO = "padronizacao"
PASSO_CLASSIFICADOR = "classificador"
PREFIXO_CLASSIFICADOR = f"{PASSO_CLASSIFICADOR}__"

LOGREG = "sklearn.linear_model.LogisticRegression"
SVC = "sklearn.svm.SVC"
FLORESTA = "sklearn.ensemble.RandomForestClassifier"
XGB = "xgboost.XGBClassifier"
ESTIMADORES_SUPORTADOS: Tuple[str, ...] = (LOGREG, SVC, FLORESTA, XGB)

#: Versão exigida do XGBoost: a do ambiente da configuração (P9, Seção 13).
VERSAO_XGBOOST: str = cfg.CONFIGURACAO.execucao.ambiente["xgboost"]


class ModeloInvalido(ValueError):
    """Código desconhecido ou estimador construído diferente da configuração."""

    chave_saida: str = "integridade"


class XGBoostIndisponivel(RuntimeError):
    """XGBoost ausente, em versão diferente da fixada ou com a biblioteca nativa bloqueada."""

    chave_saida: str = "integridade"


# --------------------------------------------------------------------------- #
# Especificação e classes
# --------------------------------------------------------------------------- #


def codigos() -> Tuple[str, ...]:
    """Códigos dos modelos, na ordem da configuração."""
    return tuple(m.codigo for m in cfg.CONFIGURACAO.modelos)


def especificacao(codigo: str) -> cfg.EspecModelo:
    for espec in cfg.CONFIGURACAO.modelos:
        if espec.codigo == codigo:
            return espec
    raise ModeloInvalido(f"modelo fora da configuração: {codigo!r}; use {codigos()}")


def _classe_xgboost() -> type:
    """``xgboost.XGBClassifier`` sob demanda, só na versão fixada."""
    try:
        import xgboost
    except Exception as exc:  # ImportError, ou falha da biblioteca nativa (DLL bloqueada)
        raise XGBoostIndisponivel(
            f"xgboost {VERSAO_XGBOOST} indisponível ({type(exc).__name__}: {exc}); o modelo xgb não pode "
            "ser construído. Instale a versão fixada num ambiente revisado; nada é instalado aqui") from exc
    versao = getattr(xgboost, "__version__", None)
    if versao != VERSAO_XGBOOST:
        raise XGBoostIndisponivel(f"xgboost {versao!r} instalado; a P9 exige {VERSAO_XGBOOST}. "
                                  "Trocar de versão exige adendo")
    return xgboost.XGBClassifier


def _classe(estimador: str) -> type:
    if estimador == LOGREG:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression
    if estimador == SVC:
        from sklearn.svm import SVC as ClasseSVC
        return ClasseSVC
    if estimador == FLORESTA:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier
    if estimador == XGB:
        return _classe_xgboost()
    raise ModeloInvalido(f"estimador não suportado: {estimador!r}; use {ESTIMADORES_SUPORTADOS}")


def xgboost_disponivel() -> Tuple[bool, str]:
    """``(True, versão)`` se o XGBoost fixado carrega; senão ``(False, motivo)``. Não instala nada."""
    try:
        _classe_xgboost()
    except XGBoostIndisponivel as exc:
        return False, str(exc)
    return True, VERSAO_XGBOOST


# --------------------------------------------------------------------------- #
# Conferências depois da construção
# --------------------------------------------------------------------------- #


def _igual(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b


def _conferir(classificador: Any, espec: cfg.EspecModelo) -> None:
    parametros = classificador.get_params(deep=False)
    for nome, valor in espec.fixos.items():
        if nome not in parametros or not _igual(parametros[nome], valor):
            raise ModeloInvalido(f"{espec.codigo}: parâmetro fixo {nome}={parametros.get(nome)!r}; "
                                 f"a configuração exige {valor!r}")
    for p in espec.grade:
        if p.nome not in parametros:
            raise ModeloInvalido(f"{espec.codigo}: hiperparâmetro da grade inexistente no estimador: {p.nome}")
    # Paralelismo interno de um processo (P9, Seção 14.3). Na regressão
    # logística, ``n_jobs`` fica no padrão (``None``, um processo) porque a
    # configuração não o fixa e ele não tem efeito no problema binário.
    if "n_jobs" in parametros and parametros["n_jobs"] != 1 and \
            not (espec.estimador == LOGREG and parametros["n_jobs"] is None):
        raise ModeloInvalido(f"{espec.codigo}: n_jobs={parametros['n_jobs']!r}; exigido 1")
    if espec.estimador == SVC and parametros.get("probability") is not False:
        raise ModeloInvalido("svm: probability deve ser False (sem calibração; P9, D10)")
    if espec.estimador == XGB and parametros.get("early_stopping_rounds") is not None:
        raise ModeloInvalido("xgb: parada antecipada fora do escopo (P9, Seção 5.1)")
    # Sementes da P9, D12: todo estimador com ``random_state`` precisa de semente inteira fixada.
    if "random_state" in parametros and type(parametros["random_state"]) is not int:
        raise ModeloInvalido(f"{espec.codigo}: random_state={parametros['random_state']!r}; semente não fixada")


# --------------------------------------------------------------------------- #
# Fábrica e grades
# --------------------------------------------------------------------------- #


def criar_classificador(codigo: str) -> Any:
    """Classificador NOVO, sem padronização, com os parâmetros fixos da configuração."""
    espec = especificacao(codigo)
    classificador = _classe(espec.estimador)(**dict(espec.fixos))
    _conferir(classificador, espec)
    return classificador


def criar_estimador(codigo: str) -> Any:
    """Estimador NOVO e não ajustado do modelo: ``Pipeline`` com padronização para LR e SVM."""
    espec = especificacao(codigo)
    classificador = criar_classificador(codigo)
    if not espec.padronizacao:
        return classificador
    return Pipeline([(PASSO_PADRONIZACAO, StandardScaler()), (PASSO_CLASSIFICADOR, classificador)])


def nome_parametro(codigo: str, nome: str) -> str:
    """Nome do hiperparâmetro no estimador de ``criar_estimador`` (com prefixo no ``Pipeline``)."""
    espec = especificacao(codigo)
    if nome not in {p.nome for p in espec.grade}:
        raise ModeloInvalido(f"{codigo}: {nome!r} não está na grade")
    return f"{PREFIXO_CLASSIFICADOR}{nome}" if espec.padronizacao else nome


def configuracoes(codigo: str) -> Tuple[Mapping[str, Any], ...]:
    """Configurações da grade na ordem de simplicidade, com os nomes de ``criar_estimador``.

    A referência histórica (sem grade) tem uma única configuração, vazia.
    """
    espec = especificacao(codigo)
    prefixo = PREFIXO_CLASSIFICADOR if espec.padronizacao else ""
    return tuple(MappingProxyType({f"{prefixo}{nome}": valor for nome, valor in c.items()})
                 for c in espec.configuracoes())


def grade_busca(codigo: str) -> List[Dict[str, List[Any]]]:
    """``param_grid`` explícito para o ``GridSearchCV``: um ponto por dicionário, na ordem de simplicidade.

    Lista nova a cada chamada. Vazia para a referência histórica, que não tem
    ajuste interno.
    """
    if not especificacao(codigo).grade:
        return []
    return [{nome: [valor] for nome, valor in c.items()} for c in configuracoes(codigo)]


__all__ = [
    "ESTIMADORES_SUPORTADOS", "ModeloInvalido", "PASSO_CLASSIFICADOR", "PASSO_PADRONIZACAO",
    "PREFIXO_CLASSIFICADOR", "VERSAO_XGBOOST", "XGBoostIndisponivel", "codigos", "configuracoes",
    "criar_classificador", "criar_estimador", "especificacao", "grade_busca", "nome_parametro",
    "xgboost_disponivel",
]
