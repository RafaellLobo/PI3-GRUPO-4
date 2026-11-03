"""Análises de sensibilidade da Etapa 3 — spline, lista reproduzida e C = 0,1.

Especificações do protocolo (nada é acrescentado aqui):

* **Spline (Seção 10.1; P2).** M0s = spline do diâmetro; M3s = spline do diâmetro + as oito features
  fixas. ``SplineTransformer`` com os parâmetros de P2 (``config.spline_confirmado``), nós ajustados SÓ
  no treino, depois ``StandardScaler`` → a mesma ``LogisticRegression`` da Seção 6.1. Contraste
  M3s − M0s, com o mesmo bootstrap pareado.
* **Lista reproduzida (P3; Emenda 1, E8 e E11).** A lista da reprodução fiel
  (``config.PAINEL_REPRODUZIDO``) "é avaliada como sensibilidade, com os mesmos contrastes": M2r, M3r e
  M4r no lugar de M2, M3 e M4 (M0 e M1 não mudam), contrastes M3r − M0, M4r − M1 e M2r − M0.
* **C = 0,1 (P7).** M0–M4 com ``C = 0,1`` (L2) no lugar de ``C = 1``, mesmos contrastes.

Regra de isolamento (Seção 7.5; Emenda 1, E7): **cada análise altera um único fator** em relação à
análise principal e **nenhuma combinação** é calculada. Os modelos M0–M4 da análise principal não são
tocados (as sensibilidades recebem as suas probabilidades prontas, quando as usam).

Autorização e segurança: estes modelos não existem na taxonomia das guardas (M0–M4, M2v–M4v), por isso
NÃO passam por ``modelos.construir`` nem pelos checkpoints, e as guardas e a fábrica NÃO foram alteradas.
Em vez disso, ``avaliar_sensibilidades`` exige a ``Autorizacao`` da própria Etapa 3 (emitida pelas
guardas para ``etapa3_validacao`` com M0–M4, o que inclui P1, P2 e P3 verificadas no git) e as decisões
P2 e P3 aprovadas (``spline_confirmado`` e ``painel_fixo`` levantam ``BloqueioDecisao`` se pendentes).
Sem isso nada é ajustado. O painel reproduzido é uma constante de ``config`` já existente.

Os dados são os de ``train`` (ajuste) e ``validation`` (avaliação) que a Etapa 3 já carregou; o teste
nunca é visto. As unidades são baratas (regressão logística) e determinísticas: são recalculadas a cada
execução, sem checkpoint, e não guardam predições individuais.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from . import bootstrap as bs
from . import config as cfg
from . import guardas
from . import modelos
from . import rastreabilidade as rst

ATIVIDADE = cfg.Etapa.E3.value
#: ``C`` da sensibilidade P7.
C_SENSIBILIDADE = 0.1
ANALISES: Tuple[str, ...] = ("spline", "lista_reproduzida", "c_0_1")
DESCRICAO: Mapping[str, str] = MappingProxyType({
    "spline": "representação flexível do tamanho: spline do diâmetro (Seção 10.1; P2)",
    "lista_reproduzida": "lista de features da reprodução fiel no lugar da lista declarada (P3)",
    "c_0_1": "regularização C = 0,1 no lugar de C = 1 (P7)",
})
DECISOES_EXIGIDAS: Tuple[str, ...] = ("P2", "P3")


class SensibilidadeInvalida(RuntimeError):
    """Composição, autorização ou entrada incompatível com a especificação das sensibilidades."""


@dataclass(frozen=True)
class ModeloSensibilidade:
    nome: str
    analise: str
    features: Tuple[str, ...]
    C: float
    spline: Optional[cfg.EspecSpline] = None

    def estimador(self) -> Pipeline:
        """Pipeline não ajustado. Nós do spline e scaler são ajustados SÓ nas linhas passadas a ``fit``."""
        logistica = LogisticRegression(**{**cfg.ESPEC_AVALIACAO.parametros(), "C": self.C})
        etapas: List[Tuple[str, Any]] = []
        if self.spline is not None:
            s = self.spline
            if self.features[0] != s.coluna:
                raise SensibilidadeInvalida(f"{self.nome}: a primeira feature deve ser {s.coluna}")
            transformador = SplineTransformer(n_knots=s.n_knots, degree=s.degree, knots=s.knots,
                                              extrapolation=s.extrapolation, include_bias=s.include_bias)
            etapas.append(("spline", ColumnTransformer([("spline", transformador, [0])], remainder="passthrough")))
        etapas += [(modelos.PASSO_SCALER, StandardScaler()), (modelos.PASSO_LOGREG, logistica)]
        return Pipeline(etapas)


def features_necessarias() -> Tuple[str, ...]:
    """Features que as sensibilidades acrescentam à carga (a lista reproduzida); só nomes de ``config``."""
    return cfg.validar_nomes(cfg.PAINEL_REPRODUZIDO, "lista reproduzida")


def definir_modelos(decisoes: Mapping[str, cfg.Decisao]) -> Dict[str, Dict[str, ModeloSensibilidade]]:
    """Os modelos de cada análise. Levanta ``BloqueioDecisao`` enquanto P2 ou P3 estiverem pendentes."""
    spline = cfg.spline_confirmado(decisoes)
    painel = cfg.painel_fixo(decisoes)
    reproduzido = features_necessarias()
    if len(reproduzido) != cfg.N_PAINEL:
        raise SensibilidadeInvalida(f"a lista reproduzida deve ter {cfg.N_PAINEL} features")
    ref = cfg.REFERENCIA
    por_analise: Dict[str, Dict[str, ModeloSensibilidade]] = {
        "spline": {
            "M0s": ModeloSensibilidade("M0s", "spline", ref["M0"], cfg.ESPEC_AVALIACAO.C, spline),
            "M3s": ModeloSensibilidade("M3s", "spline", cfg.uniao_ordenada(ref["M3"], painel),
                                       cfg.ESPEC_AVALIACAO.C, spline),
        },
        "lista_reproduzida": {
            "M2r": ModeloSensibilidade("M2r", "lista_reproduzida", cfg.uniao_ordenada(ref["M2"], reproduzido),
                                       cfg.ESPEC_AVALIACAO.C),
            "M3r": ModeloSensibilidade("M3r", "lista_reproduzida", cfg.uniao_ordenada(ref["M3"], reproduzido),
                                       cfg.ESPEC_AVALIACAO.C),
            "M4r": ModeloSensibilidade("M4r", "lista_reproduzida", cfg.uniao_ordenada(ref["M4"], reproduzido),
                                       cfg.ESPEC_AVALIACAO.C),
        },
        "c_0_1": {},
    }
    base = {"M0": ref["M0"], "M1": ref["M1"], "M2": cfg.uniao_ordenada(ref["M2"], painel),
            "M3": cfg.uniao_ordenada(ref["M3"], painel), "M4": cfg.uniao_ordenada(ref["M4"], painel)}
    for nome, features in base.items():
        por_analise["c_0_1"][nome + "c"] = ModeloSensibilidade(nome + "c", "c_0_1", features, C_SENSIBILIDADE)
    return por_analise


#: Contrastes de cada análise: (A, B, contraste principal correspondente). Δ = A − B.
CONTRASTES: Mapping[str, Tuple[Tuple[str, str, str], ...]] = MappingProxyType({
    "spline": (("M3s", "M0s", "M3-M0"),),
    "lista_reproduzida": (("M3r", "M0", "M3-M0"), ("M4r", "M1", "M4-M1"), ("M2r", "M0", "M2-M0")),
    "c_0_1": (("M3c", "M0c", "M3-M0"), ("M4c", "M1c", "M4-M1"), ("M1c", "M0c", "M1-M0"), ("M2c", "M0c", "M2-M0")),
})
#: Modelos da análise principal reaproveitados (probabilidades prontas) por análise.
REUSO_DO_PRINCIPAL: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {"spline": (), "lista_reproduzida": ("M0", "M1"), "c_0_1": ()})


def _matriz(frame: Any, features: Tuple[str, ...], nome: str) -> np.ndarray:
    ausentes = [f for f in features if f not in frame.columns]
    if ausentes:
        raise SensibilidadeInvalida(f"{nome}: features ausentes do quadro: {ausentes}")
    return modelos._matriz(frame[list(features)], features, nome)


def ajustar_e_prever(modelo: ModeloSensibilidade, treino: Any, validacao: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Ajusta SÓ em ``treino`` e prevê em ``validacao``. Devolve probabilidades e agregados técnicos."""
    X, Xv = _matriz(treino, modelo.features, modelo.nome), _matriz(validacao, modelo.features, modelo.nome)
    y = modelos._rotulos(treino[cfg.COLUNA_ALVO], treino[list(modelo.features)], modelo.nome)
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        pipe = modelo.estimador().fit(X, y)
        proba = pipe.predict_proba(Xv)[:, 1]
    if not np.isfinite(proba).all():
        raise SensibilidadeInvalida(f"{modelo.nome}: probabilidades não finitas")
    scaler = pipe.named_steps[modelos.PASSO_SCALER]
    tecnico = {"n_features": len(modelo.features), "features": list(modelo.features), "C": modelo.C,
               "n_treino": int(np.atleast_1d(scaler.n_samples_seen_)[0]),
               "n_iter": int(np.max(pipe.named_steps[modelos.PASSO_LOGREG].n_iter_)),
               "warnings": [{"categoria": c, "mensagem": m, "contagem": n}
                            for (c, m), n in sorted(_contar(capturados).items())]}
    if modelo.spline is not None:
        tecnico["n_colunas_do_spline"] = int(pipe.named_steps["spline"].transformers_[0][1].n_features_out_)
    return proba, tecnico


def _contar(capturados: Any) -> Dict[Tuple[str, str], int]:
    contagem: Dict[Tuple[str, str], int] = {}
    for w in capturados:
        chave = (w.category.__name__, rst._sanear_mensagem(str(w.message)))
        contagem[chave] = contagem.get(chave, 0) + 1
    return contagem


def avaliar_sensibilidades(*, autorizacao: guardas.Autorizacao, decisoes: Mapping[str, cfg.Decisao], treino: Any,
                           validacao: Any, probas_principais: Mapping[str, np.ndarray],
                           contrastes_principais: Mapping[str, Mapping[str, Any]],
                           espec_bootstrap: cfg.EspecBootstrap = cfg.ESPEC_BOOTSTRAP) -> Dict[str, Any]:
    """Seção ``sensibilidades`` do relatório da Etapa 3. Nada é ajustado sem a autorização da etapa."""
    if not isinstance(autorizacao, guardas.Autorizacao) or autorizacao.atividade != ATIVIDADE:
        raise SensibilidadeInvalida("as sensibilidades exigem a autorização da Etapa 3 (etapa3_validacao)")
    faltam = [c for c in DECISOES_EXIGIDAS if c not in autorizacao.decisoes_exigidas]
    if faltam:
        raise SensibilidadeInvalida(f"a autorização da Etapa 3 não cobre {faltam}")
    por_analise = definir_modelos(decisoes)  # BloqueioDecisao se P2 ou P3 estiverem pendentes
    y = validacao[cfg.COLUNA_ALVO].to_numpy(dtype=np.int64)
    pacientes = validacao[cfg.COLUNA_PACIENTE].to_numpy()
    saida: Dict[str, Any] = {
        "regra": "cada análise altera um único fator em relação à análise principal; nenhuma combinação é calculada "
                 "(Emenda 1, E7; protocolo, Seção 7.5)",
        "leitura": "descritiva, sem inferência causal e sem a régua da Seção 9 (Seção 10.1)",
        "autorizacao": "mesma da Etapa 3; guardas e fábrica de modelos não alteradas",
    }
    for analise in ANALISES:
        probas: Dict[str, np.ndarray] = {}
        tecnicos: Dict[str, Any] = {}
        for nome in REUSO_DO_PRINCIPAL[analise]:
            probas[nome] = np.asarray(probas_principais[nome], dtype=np.float64)
        for nome, modelo in por_analise[analise].items():
            probas[nome], tecnicos[nome] = ajustar_e_prever(modelo, treino, validacao)
        pares = [(a, b) for a, b, _ in CONTRASTES[analise]]
        resultado = bs.bootstrap_pareado(pacientes, y, probas, pares, espec_bootstrap)
        modelos_rel = {}
        for nome in probas:
            modelos_rel[nome] = {"origem": "novo" if nome in tecnicos else "análise principal (sem reajuste)",
                                 "metricas": {m: resultado.por_modelo[m][nome].como_dict() for m in bs.METRICAS}}
            if nome in tecnicos:
                modelos_rel[nome].update(tecnicos[nome])
        contrastes_rel = {}
        for a, b, principal in CONTRASTES[analise]:
            par = f"{a}-{b}"
            contrastes_rel[par] = {
                "a": a, "b": b, "contraste_principal_correspondente": principal,
                "principal": contrastes_principais[principal]["auc"],
                **{m: resultado.contrastes[m][par].como_dict() for m in bs.METRICAS},
            }
        saida[analise] = {
            "descricao": DESCRICAO[analise], "modelos": modelos_rel, "contrastes": contrastes_rel,
            "bootstrap": {"n_replicas": resultado.n_replicas, "n_validas": resultado.n_validas,
                          "n_descartadas": resultado.n_descartadas, "seed": resultado.seed,
                          "mesma_amostra_de_pacientes_da_analise_principal": True},
        }
        if analise == "spline":
            saida[analise]["parametros_do_spline"] = {
                "coluna": por_analise[analise]["M0s"].spline.coluna, "n_knots": por_analise[analise]["M0s"].spline.n_knots,
                "degree": por_analise[analise]["M0s"].spline.degree, "knots": por_analise[analise]["M0s"].spline.knots,
                "extrapolation": por_analise[analise]["M0s"].spline.extrapolation,
                "include_bias": por_analise[analise]["M0s"].spline.include_bias, "nos_ajustados_somente_no_treino": True}
    return saida


__all__ = ["ANALISES", "C_SENSIBILIDADE", "CONTRASTES", "ModeloSensibilidade", "SensibilidadeInvalida",
           "ajustar_e_prever", "avaliar_sensibilidades", "definir_modelos", "features_necessarias"]
