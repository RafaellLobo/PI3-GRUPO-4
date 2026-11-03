"""Consolidação dos resultados das unidades (P9, Seções 6, 8.1 e 9).

Reúne as 250 unidades (25 se a base tivesse uma só repetição; aqui, as do
modo ``definitivo``), confere o PAREAMENTO e calcula, para cada modelo ``m`` e
desfecho ``d``:

* ``θ̂(m, d)``: média, entre as repetições válidas, da média das AUC das
  dobras válidas (P9, Seção 8.1);
* ``Δ̂(a, b, d) = θ̂(a, d) − θ̂(b, d)``, pareado por dobra. Como a validade é a
  mesma para todos os modelos, é a média, nas repetições válidas, da diferença
  entre as médias por repetição;
* a sensibilidade à partição: ``Δ̂`` de cada repetição, com média, mínimo,
  máximo e número de repetições com ``Δ > 0``.

**Estados (P9, Seção 6).** DD (dobra degenerada), RI (repetição inválida) e NE
(desfecho não estimável) são contados por desfecho. Nenhuma AUC é imputada. Em
desfecho NE, ``θ̂`` e ``Δ̂`` ficam ``None`` e só os valores por dobra são
relatados. FT não aparece aqui: com falha técnica pendente a execução é
``incompleta`` e a consolidação nem é chamada (``parear`` exige as unidades
todas).

**Privacidade.** As predições individuais entram em ``Pareados`` (só em
memória, nunca serializado). Tudo o que ``consolidar`` devolve é agregado:
contagens, AUC e médias por dobra, curvas médias e calibração por decis (cada
decil com pelo menos ``min_nodulos_por_decil`` nódulos).

Os contrastes da família (RF, XGB e SVM contra a LR com ajuste, em G e F) e os
descritivos vêm da configuração canônica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_curve

from . import config as cfg
from . import metricas as mt
from . import modelos
from . import unidades as un

GRUPO_FAMILIA = "familia"
GRUPO_DESCRITIVO = "descritivo"
PERCENTIS_CURVA = (2.5, 97.5)


class ConsolidacaoInvalida(ValueError):
    """Unidades incompletas, duplicadas ou não pareadas. Chave de saída: integridade."""

    chave_saida = "integridade"


# --------------------------------------------------------------------------- #
# Pareamento
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class Pareados:
    """Resultados completos e PAREADOS dos modelos. Contém predições individuais: só em memória e local."""

    base: mt.BaseAvaliacao
    repeticoes: Tuple[int, ...]
    n_dobras: int
    resultados: Dict[Tuple[str, int, int], un.ResultadoUnidade] = field(repr=False)

    def resultado(self, modelo: str, repeticao: int, dobra: int) -> un.ResultadoUnidade:
        return self.resultados[(modelo, repeticao, dobra)]


def parear(base: mt.BaseAvaliacao, resultados: Iterable[un.ResultadoUnidade]) -> Pareados:
    """Confere que há exatamente as unidades do modo e que os cinco modelos avaliam as MESMAS linhas por dobra."""
    if not isinstance(base, mt.BaseAvaliacao):
        raise TypeError("base deve ser BaseAvaliacao")
    esperadas = un.unidades_do_modo(base.modo)
    por_unidade: Dict[Tuple[str, int, int], un.ResultadoUnidade] = {}
    for r in resultados:
        if not isinstance(r, un.ResultadoUnidade):
            raise TypeError("resultados deve conter ResultadoUnidade; falhas não são resultados")
        chave = (r.unidade.modelo, r.unidade.repeticao, r.unidade.dobra)
        if chave in por_unidade:
            raise ConsolidacaoInvalida(f"{r.unidade.rotulo}: unidade repetida")
        por_unidade[chave] = r
    faltam = [u.rotulo for u in esperadas if (u.modelo, u.repeticao, u.dobra) not in por_unidade]
    if faltam:
        raise ConsolidacaoInvalida(
            f"{len(faltam)} unidade(s) ausente(s), por exemplo {faltam[:3]}: execução incompleta")
    if len(por_unidade) != len(esperadas):
        raise ConsolidacaoInvalida("unidades fora do plano do modo")
    sha, oficial = cfg.sha256_configuracao(), {r.registro.oficial for r in por_unidade.values()}
    for r in por_unidade.values():
        if r.registro.modo != base.modo or r.registro.sha256_configuracao != sha:
            raise ConsolidacaoInvalida(f"{r.unidade.rotulo}: modo ou configuração diferentes da base")
    if len(oficial) != 1 or len({r.registro.fingerprint_plano for r in por_unidade.values()}) != 1:
        raise ConsolidacaoInvalida("unidades de origens (oficial) ou planos diferentes")
    codigos = modelos.codigos()
    repeticoes = tuple(sorted({u.repeticao for u in esperadas}))
    n_dobras = cfg.CONFIGURACAO.dobras.n_dobras_externas
    for r in repeticoes:
        for k in range(n_dobras):
            referencia = por_unidade[(codigos[0], r, k)]
            for m in codigos:
                atual = por_unidade[(m, r, k)]
                mt._exigir_coerente(base, atual)
                if atual.registro.fingerprint_dobra != referencia.registro.fingerprint_dobra \
                        or not np.array_equal(atual.privado.posicoes, referencia.privado.posicoes):
                    raise ConsolidacaoInvalida(f"r{r:02d}-k{k}: {m} não avalia as mesmas linhas que {codigos[0]}")
    return Pareados(base, repeticoes, n_dobras, por_unidade)


# --------------------------------------------------------------------------- #
# Linhas de saída (todas agregadas)
# --------------------------------------------------------------------------- #


class AucDobra(NamedTuple):
    modelo: str
    repeticao: int
    dobra: int
    desfecho: str
    auc: Optional[float]
    valida: bool


class AucRepeticao(NamedTuple):
    modelo: str
    repeticao: int
    desfecho: str
    n_dobras_validas: int
    valida: bool
    auc: Optional[float]


class Estimativa(NamedTuple):
    """``θ̂(m, d)``; ``estimavel=False`` (NE) deixa ``theta`` ``None``."""

    modelo: str
    desfecho: str
    estimavel: bool
    n_repeticoes_validas: int
    theta: Optional[float]
    minimo_repeticoes: Optional[float]
    maximo_repeticoes: Optional[float]


class Contraste(NamedTuple):
    """``Δ̂(a, b, d) = θ̂(a, d) − θ̂(b, d)`` e sua sensibilidade à partição."""

    a: str
    b: str
    desfecho: str
    grupo: str
    estimavel: bool
    n_repeticoes_validas: int
    delta: Optional[float]
    #: Δ de cada repetição (índice = repetição); ``None`` onde a repetição é inválida.
    delta_por_repeticao: Tuple[Optional[float], ...]
    minimo: Optional[float]
    maximo: Optional[float]
    n_repeticoes_positivas: int
    fracao_repeticoes_positivas: Optional[float]


class DeltaDobra(NamedTuple):
    a: str
    b: str
    desfecho: str
    repeticao: int
    dobra: int
    delta: Optional[float]


class EstadosDesfecho(NamedTuple):
    """DD, RI e NE de um desfecho (P9, Seção 6)."""

    desfecho: str
    n_dobras: int
    dobras_degeneradas: int
    n_repeticoes: int
    repeticoes_invalidas: int
    repeticoes_validas: int
    nao_estimavel: bool


class AvisosModelo(NamedTuple):
    modelo: str
    n_unidades: int
    unidades_com_aviso: int
    avisos_convergencia: int
    unidades_com_convergencia: Tuple[str, ...]


class Escolha(NamedTuple):
    modelo: str
    repeticao: int
    dobra: int
    indice_configuracao: int
    n_configuracoes: int
    configuracao: Tuple[Tuple[str, object], ...]
    na_borda: bool
    ajustes: int


class CurvaRoc(NamedTuple):
    """Curva ROC média vertical (desfecho G) numa grade fixa de 1 − especificidade."""

    modelo: str
    n_dobras: int
    fpr: Tuple[float, ...]
    tpr_media: Tuple[float, ...]
    tpr_p025: Tuple[float, ...]
    tpr_p975: Tuple[float, ...]


class FaixaCalibracao(NamedTuple):
    modelo: str
    faixa: int
    nodulos: int
    probabilidade_media: float
    frequencia_observada: float


class ResumoDescritiva(NamedTuple):
    modelo: str
    metrica: str
    rotulo: str
    n_dobras: int
    media: Optional[float]
    minimo: Optional[float]
    maximo: Optional[float]


@dataclass(frozen=True, eq=False)
class Consolidacao:
    """Tudo o que se calcula SEM bootstrap. Apenas agregados."""

    modo: str
    repeticoes: Tuple[int, ...]
    n_dobras: int
    denominadores: Tuple[mt.Denominador, ...]
    auc_por_dobra: Tuple[AucDobra, ...]
    auc_por_repeticao: Tuple[AucRepeticao, ...]
    estimativas: Tuple[Estimativa, ...]
    contrastes: Tuple[Contraste, ...]
    deltas_por_dobra: Tuple[DeltaDobra, ...]
    estados: Tuple[EstadosDesfecho, ...]
    avisos: Tuple[AvisosModelo, ...]
    escolhas: Tuple[Escolha, ...]
    descritivas: Tuple[mt.Descritiva, ...]
    resumo_descritivas: Tuple[ResumoDescritiva, ...]
    curvas_roc: Tuple[CurvaRoc, ...]
    calibracao: Tuple[FaixaCalibracao, ...]
    contagem_faixa: Tuple[Tuple[str, int], ...]
    oficial: bool

    def estimativa(self, modelo: str, desfecho: str) -> Estimativa:
        return next(e for e in self.estimativas if (e.modelo, e.desfecho) == (modelo, desfecho))

    def contraste(self, a: str, b: str, desfecho: str) -> Contraste:
        return next(c for c in self.contrastes if (c.a, c.b, c.desfecho) == (a, b, desfecho))


# --------------------------------------------------------------------------- #
# Cálculo
# --------------------------------------------------------------------------- #

DESFECHOS = (mt.DESFECHO_G, mt.DESFECHO_F)


def pares_de_contraste() -> Tuple[Tuple[str, str, str], ...]:
    """``(a, b, grupo)`` da configuração: a família (desafiantes contra o comparador) e os descritivos."""
    c = cfg.CONFIGURACAO.contrastes
    return tuple((a, c.comparador, GRUPO_FAMILIA) for a in c.desafiantes) + \
        tuple((a, b, GRUPO_DESCRITIVO) for a, b in c.descritivos)


def _media(valores: List[float]) -> Optional[float]:
    return float(np.mean(valores)) if valores else None


TOLERANCIA_FPR = 1e-9


def tpr_na_grade(fpr: np.ndarray, tpr: np.ndarray, grade: np.ndarray) -> np.ndarray:
    """Sensibilidade da curva ROC em cada ``1 − especificidade`` da grade.

    ``fpr`` e ``tpr`` são os pontos de ``roc_curve(..., drop_intermediate=False)``, em ordem, e o caminho entre
    pontos consecutivos é linear (degraus quando os escores são distintos; diagonais nos empates). Num ``fpr``
    exatamente igual ao da grade (a menos de ``TOLERANCIA_FPR``) vale o MAIOR ``tpr`` do segmento vertical; entre
    pontos, a interpolação linear. Pontos de mesmo ``fpr`` NÃO são fundidos: isso trocaria os degraus por diagonais.
    """
    fpr, tpr = np.asarray(fpr, dtype=np.float64), np.asarray(tpr, dtype=np.float64)
    saida = np.empty(len(grade))
    # j = primeiro ponto estritamente à direita da grade (com a tolerância); fpr[j-1] <= g (e é o topo do segmento)
    direita = np.searchsorted(fpr, np.asarray(grade) + TOLERANCIA_FPR, side="right")
    for i, g in enumerate(grade):
        j = int(direita[i])
        if j >= len(fpr):
            saida[i] = tpr[-1]
        elif abs(fpr[j - 1] - g) <= TOLERANCIA_FPR:
            saida[i] = tpr[j - 1]
        else:
            x0, x1, y0, y1 = fpr[j - 1], fpr[j], tpr[j - 1], tpr[j]
            saida[i] = y0 + (y1 - y0) * (g - x0) / (x1 - x0)
    return saida


def _curvas_roc(p: Pareados) -> Tuple[CurvaRoc, ...]:
    grade = np.linspace(0.0, 1.0, cfg.CONFIGURACAO.saidas.pontos_roc)
    curvas = []
    for m in modelos.codigos():
        linhas = []
        for r in p.repeticoes:
            for k in range(p.n_dobras):
                res = p.resultado(m, r, k)
                y, s = np.asarray(res.privado.y), np.asarray(res.privado.pontuacoes)
                if not mt.valida(int(np.sum(y == 1)), int(np.sum(y == 0))):
                    continue
                fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
                linhas.append(tpr_na_grade(fpr, tpr, grade))
        if linhas:
            matriz = np.vstack(linhas)
            q = np.percentile(matriz, PERCENTIS_CURVA, axis=0)
            curvas.append(CurvaRoc(m, len(linhas), tuple(grade.tolist()), tuple(matriz.mean(axis=0).tolist()),
                                   tuple(q[0].tolist()), tuple(q[1].tolist())))
    return tuple(curvas)


def _fundir(grupos: List[np.ndarray], minimo: int) -> List[np.ndarray]:
    """Funde cada grupo com menos de ``minimo`` linhas com o vizinho menor, até não restar nenhum."""
    grupos = list(grupos)
    while len(grupos) > 1:
        tamanhos = [len(g) for g in grupos]
        i = int(np.argmin(tamanhos))
        if tamanhos[i] >= minimo:
            break
        vizinhos = [j for j in (i - 1, i + 1) if 0 <= j < len(grupos)]
        j = min(vizinhos, key=lambda v: (tamanhos[v], v))
        a, b = sorted((i, j))
        grupos[a:b + 1] = [np.concatenate([grupos[a], grupos[b]])]
    return grupos


def _calibracao(p: Pareados) -> Tuple[FaixaCalibracao, ...]:
    """Decis da probabilidade, nas predições fora da dobra de ``r = 0`` (P9, Seção 11, F6)."""
    saidas_cfg = cfg.CONFIGURACAO.saidas
    linhas = []
    for m in modelos.codigos():
        if not modelos.especificacao(m).metricas_probabilisticas:
            continue
        ys, ss = [], []
        for k in range(p.n_dobras):
            res = p.resultado(m, 0, k)
            ys.append(np.asarray(res.privado.y))
            ss.append(np.asarray(res.privado.pontuacoes))
        y, s = np.concatenate(ys), np.concatenate(ss)
        ordem = np.argsort(s, kind="mergesort")
        grupos = _fundir([g for g in np.array_split(ordem, saidas_cfg.decis_calibracao) if len(g)],
                         saidas_cfg.min_nodulos_por_decil)
        for i, g in enumerate(grupos, start=1):
            linhas.append(FaixaCalibracao(m, i, int(len(g)), float(np.mean(s[g])), float(np.mean(y[g]))))
    return tuple(linhas)


def consolidar(p: Pareados) -> Consolidacao:
    """AUC por dobra e repetição, ``θ̂``, ``Δ̂``, estados, avisos, escolhas e métricas descritivas (sem bootstrap)."""
    if not isinstance(p, Pareados):
        raise TypeError("p deve ser Pareados (use parear)")
    validade = cfg.CONFIGURACAO.validade
    codigos = modelos.codigos()
    denominadores: List[mt.Denominador] = []
    auc_dobra: List[AucDobra] = []
    auc_rep: List[AucRepeticao] = []
    estimativas: List[Estimativa] = []
    estados: List[EstadosDesfecho] = []
    medias_rep: Dict[Tuple[str, str], Dict[int, float]] = {}
    medias_dobra: Dict[Tuple[str, str, int, int], Optional[float]] = {}

    for desfecho in DESFECHOS:
        invalidas_dobra = 0
        reps_invalidas = 0
        validas_por_rep: Dict[int, bool] = {}
        for r in p.repeticoes:
            n_validas = 0
            for k in range(p.n_dobras):
                dens = set()
                for m in codigos:
                    d, auc = mt.auc_da_dobra(p.base, p.resultado(m, r, k), desfecho)
                    dens.add(d)
                    auc_dobra.append(AucDobra(m, r, k, desfecho, auc, d.valida))
                    medias_dobra[(m, desfecho, r, k)] = auc
                if len(dens) != 1:  # a validade e os denominadores não dependem do modelo
                    raise ConsolidacaoInvalida(f"r{r:02d}-k{k}: denominadores diferentes entre modelos")
                (d,) = dens
                denominadores.append(d)
                invalidas_dobra += 0 if d.valida else 1
                n_validas += 1 if d.valida else 0
            validas_por_rep[r] = n_validas >= validade.min_dobras_validas_por_repeticao
            reps_invalidas += 0 if validas_por_rep[r] else 1
            for m in codigos:
                aucs = [medias_dobra[(m, desfecho, r, k)] for k in range(p.n_dobras)
                        if medias_dobra[(m, desfecho, r, k)] is not None]
                media = _media(aucs) if validas_por_rep[r] else None
                auc_rep.append(AucRepeticao(m, r, desfecho, len(aucs), validas_por_rep[r], media))
                if media is not None:
                    medias_rep.setdefault((m, desfecho), {})[r] = media
        n_validas_rep = sum(validas_por_rep.values())
        nao_estimavel = n_validas_rep < validade.min_repeticoes_validas
        estados.append(EstadosDesfecho(desfecho, len(p.repeticoes) * p.n_dobras, invalidas_dobra, len(p.repeticoes),
                                       reps_invalidas, n_validas_rep, nao_estimavel))
        for m in codigos:
            por_rep = list(medias_rep.get((m, desfecho), {}).values())
            if nao_estimavel:
                estimativas.append(Estimativa(m, desfecho, False, n_validas_rep, None, None, None))
            else:
                estimativas.append(Estimativa(m, desfecho, True, n_validas_rep, float(np.mean(por_rep)),
                                              float(min(por_rep)), float(max(por_rep))))

    contrastes: List[Contraste] = []
    deltas_dobra: List[DeltaDobra] = []
    for a, b, grupo in pares_de_contraste():
        for desfecho in DESFECHOS:
            estado = next(e for e in estados if e.desfecho == desfecho)
            ra, rb = medias_rep.get((a, desfecho), {}), medias_rep.get((b, desfecho), {})
            por_rep = tuple((ra[r] - rb[r]) if (r in ra and r in rb) else None for r in p.repeticoes)
            validos = [d for d in por_rep if d is not None]
            positivos = sum(1 for d in validos if d > 0)
            if estado.nao_estimavel:
                contrastes.append(Contraste(a, b, desfecho, grupo, False, len(validos), None, por_rep, None, None,
                                            positivos, None))
            else:
                contrastes.append(Contraste(a, b, desfecho, grupo, True, len(validos), float(np.mean(validos)),
                                            por_rep, float(min(validos)), float(max(validos)), positivos,
                                            positivos / len(validos)))
            for r in p.repeticoes:
                for k in range(p.n_dobras):
                    x, y = medias_dobra[(a, desfecho, r, k)], medias_dobra[(b, desfecho, r, k)]
                    deltas_dobra.append(DeltaDobra(a, b, desfecho, r, k, None if x is None or y is None else x - y))

    avisos, escolhas, descritivas = [], [], []
    for m in codigos:
        registros = [p.resultado(m, r, k).registro for r in p.repeticoes for k in range(p.n_dobras)]
        com_conv = tuple(g.unidade.rotulo for g in registros if g.avisos_convergencia > 0)
        avisos.append(AvisosModelo(m, len(registros), sum(1 for g in registros if g.avisos), sum(
            g.avisos_convergencia for g in registros), com_conv))
        for g in registros:
            escolhas.append(Escolha(m, g.unidade.repeticao, g.unidade.dobra, g.indice_configuracao, g.n_configuracoes,
                                    g.configuracao, any(b[4] for b in g.borda), g.n_ajustes))
        for r in p.repeticoes:
            for k in range(p.n_dobras):
                descritivas.extend(mt.descritivas_da_dobra(p.base, p.resultado(m, r, k)))
    resumo: List[ResumoDescritiva] = []
    por_chave: Dict[Tuple[str, str, str], List[float]] = {}
    for d in descritivas:
        if d.valor is not None:
            por_chave.setdefault((d.modelo, d.metrica, d.rotulo), []).append(d.valor)
    for (m, metrica, rotulo), valores in por_chave.items():
        resumo.append(ResumoDescritiva(m, metrica, rotulo, len(valores), float(np.mean(valores)), float(min(valores)),
                                       float(max(valores))))
    return Consolidacao(
        modo=p.base.modo, repeticoes=p.repeticoes, n_dobras=p.n_dobras, denominadores=tuple(denominadores),
        auc_por_dobra=tuple(auc_dobra), auc_por_repeticao=tuple(auc_rep), estimativas=tuple(estimativas),
        contrastes=tuple(contrastes), deltas_por_dobra=tuple(deltas_dobra), estados=tuple(estados),
        avisos=tuple(avisos), escolhas=tuple(escolhas), descritivas=tuple(descritivas),
        resumo_descritivas=tuple(resumo), curvas_roc=_curvas_roc(p), calibracao=_calibracao(p),
        contagem_faixa=tuple(sorted(p.base.contagem_faixa().items())),
        oficial=bool(next(iter(p.resultados.values())).registro.oficial))


def contagem_estados(consolidacao: Consolidacao) -> Dict[str, Dict[str, int]]:
    """Contagem de DD, RI e NE por desfecho (para o relatório)."""
    return {e.desfecho: {"DD": e.dobras_degeneradas, "RI": e.repeticoes_invalidas, "NE": int(e.nao_estimavel)}
            for e in consolidacao.estados}


__all__ = [
    "AucDobra", "AucRepeticao", "AvisosModelo", "Consolidacao", "ConsolidacaoInvalida", "Contraste", "CurvaRoc",
    "DESFECHOS", "DeltaDobra", "Escolha", "EstadosDesfecho", "Estimativa", "FaixaCalibracao", "GRUPO_DESCRITIVO",
    "GRUPO_FAMILIA", "Pareados", "ResumoDescritiva", "consolidar", "contagem_estados", "pares_de_contraste", "parear",
    "tpr_na_grade",
]
