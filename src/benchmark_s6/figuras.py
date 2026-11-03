"""Figuras F1 a F8 (P9, Seção 11): PNG a 300 dpi, SVG e o CSV agregado que gerou cada uma.

Regras gerais da P9: só dados agregados (nenhum ponto por nódulo ou paciente);
``matplotlib`` do ambiente fixado; cor fixa por modelo na paleta Okabe–Ito;
eixos de AUC comuns entre painéis; legenda com a base, o desfecho e o tipo de
intervalo ou de faixa.

As funções devolvem BYTES (``{"figuras/F1.png": ..., "figuras/F1.svg": ...,
"figuras/dados/F1.csv": ...}``); quem grava é o coordenador. A saída é
determinística: o gerador de ids do SVG tem sal fixo, e as datas e a assinatura
do ``matplotlib`` são retiradas dos metadados, para o modo ``verificar`` comparar
SHA-256.

* F1: θ̂ por modelo, em G e F, com as 10 repetições, a média e o IC de 95%
  descritivo do bootstrap (SEM correção de multiplicidade) e a linha da LR fixa.
* F2: floresta dos seis contrastes da família (Bonferroni em traço grosso, 95% em
  traço fino, amplitude entre repetições, linhas em 0 e em δ).
* F3: Δ por dobra (até 50 pontos por contraste e desfecho), com caixa.
* F4: frequência das configurações escolhidas, com as bordas destacadas.
* F5: ROC média vertical (G); a faixa é "dispersão entre dobras — não é intervalo
  de confiança".
* F6: calibração por decis (LR fixa, LR com ajuste, RF e XGB; G; r = 0).
* F7: denominadores do desfecho F por dobra.
* F8: histograma de Δ* do bootstrap por contraste da família.
"""

from __future__ import annotations

import io
import textwrap
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from . import bootstrap as bs  # noqa: E402
from . import config as cfg  # noqa: E402
from . import consolidacao as co  # noqa: E402
from . import modelos  # noqa: E402
from . import relatorios as rl  # noqa: E402

#: Okabe–Ito. Cor fixa por modelo em TODAS as figuras.
COR = {"lr_fixa": "#000000", "lr_tuning": "#0072B2", "svm": "#E69F00", "rf": "#009E73", "xgb": "#D55E00"}
COR_NEUTRA = "#999999"
ROTULO_FAIXA_ROC = "dispersão entre dobras — não é intervalo de confiança"
HASH_SALT = "s6-p9-benchmark"
METADADOS = {"png": {"Software": None}, "svg": {"Date": None, "Creator": None}}
LARGURA_PADRAO = 7.0


def _estilo() -> Dict[str, Any]:
    return {"svg.hashsalt": HASH_SALT, "svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 8,
            "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
            "savefig.dpi": cfg.CONFIGURACAO.saidas.dpi,
            "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False}


def _jitter(n: int, largura: float = 0.12) -> np.ndarray:
    """Deslocamento determinístico (sem sorteio) em ``[-largura, largura]``."""
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-largura, largura, n)


def _bytes(fig: plt.Figure, nome: str, tabela: bytes) -> Dict[str, bytes]:
    saidas: Dict[str, bytes] = {}
    for formato in cfg.CONFIGURACAO.saidas.formatos_figura:
        buf = io.BytesIO()
        fig.savefig(buf, format=formato, dpi=cfg.CONFIGURACAO.saidas.dpi, metadata=METADADOS[formato])
        saidas[f"figuras/{nome}.{formato}"] = buf.getvalue()
    saidas[f"figuras/dados/{nome}.csv"] = tabela
    plt.close(fig)
    return saidas


def _legenda_base(fig: plt.Figure, base: Mapping[str, int], texto: str) -> None:
    linhas = textwrap.wrap(f"Base: {base['pacientes']} pacientes / {base['nodulos']} nódulos. {texto}", width=115)
    fig.text(0.01, 0.005, "\n".join(linhas), fontsize=6.5, color="#333333", va="bottom", ha="left")


def _nome(m: str) -> str:
    return rl.NOME_MODELO[m]


# --------------------------------------------------------------------------- #
# F1 a F4
# --------------------------------------------------------------------------- #


def f1(c: co.Consolidacao, b: bs.ResultadoBootstrap, base: Mapping[str, int]) -> Dict[str, bytes]:
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA_PADRAO, 3.6), sharey=True)
    linhas: List[Tuple[Any, ...]] = []
    reps = {(a.modelo, a.desfecho): [] for a in c.auc_por_repeticao}
    for a in c.auc_por_repeticao:
        reps[(a.modelo, a.desfecho)].append(a.auc)
    todos: List[float] = []
    for eixo, desfecho in zip(eixos, co.DESFECHOS):
        ref = c.estimativa("lr_fixa", desfecho).theta
        if ref is not None:
            eixo.axhline(ref, color=COR["lr_fixa"], ls="--", lw=0.8, label="referência: LR fixa")
        for x, m in enumerate(modelos.codigos()):
            e, ic = c.estimativa(m, desfecho), b.intervalo_theta(m, desfecho)
            valores = [v for v in reps[(m, desfecho)] if v is not None]
            todos += valores
            eixo.scatter(x + _jitter(len(valores)), valores, s=9, color=COR[m], alpha=0.55, zorder=2)
            if e.estimavel:
                eixo.scatter([x], [e.theta], marker="D", s=26, color=COR[m], edgecolor="white", zorder=4)
                if ic.relatado:
                    eixo.vlines(x, ic.limite_baixo, ic.limite_alto, color=COR[m], lw=2.2, zorder=3)
                else:
                    eixo.annotate(f"IC não relatado ({ic.motivo})", (x, e.theta), fontsize=6, ha="center",
                                  va="bottom", xytext=(0, 6), textcoords="offset points")
            else:
                eixo.annotate("não estimável", (x, 0.5), fontsize=6, ha="center")
            linhas.append((m, desfecho, e.theta, ic.limite_baixo, ic.limite_alto, ic.relatado, ic.motivo,
                           *reps[(m, desfecho)]))
        eixo.set_xticks(range(5), [_nome(m) for m in modelos.codigos()], rotation=30, ha="right")
        eixo.set_title(rl.NOME_DESFECHO[desfecho], fontsize=8)
    eixos[0].set_ylabel("AUC")
    if todos:
        margem = 0.01
        eixos[0].set_ylim(min(todos) - margem, min(1.0, max(todos) + margem))
    eixos[0].legend(loc="lower left", fontsize=6.5)
    fig.suptitle("F1 — AUC por modelo: repetições, média e IC de 95% descritivo (bootstrap, sem correção)",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _legenda_base(fig, base, "Pontos: 10 repetições; losango: θ̂; traço: IC de 95% descritivo, sem correção de "
                             "multiplicidade.")
    tabela = rl.csv_bytes(["modelo", "desfecho", "theta", "ic95_descritivo_baixo", "ic95_descritivo_alto",
                           "ic_relatado", "motivo_sem_ic"] + [f"auc_r{r:02d}" for r in c.repeticoes], linhas)
    return _bytes(fig, "F1", tabela)


def f2(c: co.Consolidacao, b: bs.ResultadoBootstrap, base: Mapping[str, int]) -> Dict[str, bytes]:
    familia = [x for x in c.contrastes if x.grupo == co.GRUPO_FAMILIA]
    fig, eixo = plt.subplots(figsize=(LARGURA_PADRAO, 3.6))
    delta_min = cfg.CONFIGURACAO.selecao.delta
    linhas = []
    for y, x in enumerate(reversed(familia)):
        ic = b.intervalo_contraste(x.a, x.b, x.desfecho)
        cor = COR[x.a]
        if x.estimavel:
            if x.minimo is not None:
                eixo.hlines(y, x.minimo, x.maximo, color=cor, lw=6, alpha=0.18, zorder=1)
            if ic.relatado:
                eixo.hlines(y, ic.bonferroni_baixo, ic.bonferroni_alto, color=cor, lw=3.2, zorder=3)
                eixo.hlines(y, ic.ic95_baixo, ic.ic95_alto, color="black", lw=0.9, zorder=4)
            else:
                eixo.annotate(f"IC não relatado ({ic.motivo})", (x.delta, y), fontsize=6, xytext=(4, 6),
                              textcoords="offset points")
            eixo.scatter([x.delta], [y], marker="D", s=22, color=cor, edgecolor="white", zorder=5)
        else:
            eixo.annotate("não estimável", (0, y), fontsize=6, ha="center")
        linhas.append((x.a, x.b, x.desfecho, x.delta, ic.bonferroni_baixo, ic.bonferroni_alto, ic.ic95_baixo,
                       ic.ic95_alto, x.minimo, x.maximo, ic.relatado, ic.motivo))
    eixo.axvline(0.0, color="black", lw=0.8)
    eixo.axvline(delta_min, color="black", lw=0.8, ls=":")
    eixo.set_yticks(range(len(familia)), [f"{_nome(x.a)} − {_nome(x.b)} ({x.desfecho})" for x in reversed(familia)])
    eixo.set_xlabel("Diferença estimada de AUC (Δ)")
    fig.suptitle("F2 — Seis contrastes da família contra a LR com ajuste", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    _legenda_base(fig, base, "Traço grosso: IC com Bonferroni; fino: IC de 95% sem correção; faixa clara: menor a "
                             f"maior repetição; linhas: 0 e δ = {delta_min}.")
    tabela = rl.csv_bytes(["a", "b", "desfecho", "delta", "bonferroni_baixo", "bonferroni_alto", "ic95_baixo",
                           "ic95_alto", "minimo_repeticoes", "maximo_repeticoes", "ic_relatado", "motivo_sem_ic"],
                          linhas)
    return _bytes(fig, "F2", tabela)


def f3(c: co.Consolidacao, base: Mapping[str, int]) -> Dict[str, bytes]:
    familia = [(a, "lr_tuning") for a in cfg.CONFIGURACAO.contrastes.desafiantes]
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA_PADRAO, 3.6), sharey=True)
    linhas = []
    for eixo, desfecho in zip(eixos, co.DESFECHOS):
        dados = []
        for a, bb in familia:
            deltas = [d for d in c.deltas_por_dobra if (d.a, d.b, d.desfecho) == (a, bb, desfecho)]
            validos = [d.delta for d in deltas if d.delta is not None]
            dados.append(validos)
            linhas += [(a, bb, desfecho, d.repeticao, d.dobra, d.delta) for d in deltas]
        posicoes = range(1, len(familia) + 1)
        if any(dados):
            eixo.boxplot([v if v else [np.nan] for v in dados], positions=list(posicoes), widths=0.5,
                         showfliers=False, medianprops={"color": "black"})
        for p, (a, _), v in zip(posicoes, familia, dados):
            eixo.scatter(p + _jitter(len(v), 0.2), v, s=7, color=COR[a], alpha=0.6, zorder=3)
            degeneradas = sum(1 for d in c.deltas_por_dobra if (d.a, d.b, d.desfecho) == (a, "lr_tuning", desfecho)
                              and d.delta is None)
            eixo.annotate(f"DD: {degeneradas}", (p, 1.0), xycoords=("data", "axes fraction"), fontsize=6, ha="center",
                          va="top")
        eixo.axhline(0.0, color="black", lw=0.8)
        eixo.set_xticks(list(posicoes), [_nome(a) for a, _ in familia])
        eixo.set_title(rl.NOME_DESFECHO[desfecho], fontsize=8)
    eixos[0].set_ylabel("Δ de AUC por dobra (contra a LR com ajuste)")
    fig.suptitle("F3 — Distribuição de Δ por dobra (até 50 por contraste e desfecho)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _legenda_base(fig, base, "Pontos: dobras externas válidas; DD: dobras degeneradas, sem AUC.")
    tabela = rl.csv_bytes(["a", "b", "desfecho", "repeticao", "dobra", "delta"], linhas)
    return _bytes(fig, "F3", tabela)


def _borda_da_configuracao(modelo: str, configuracao: Mapping[str, Any]) -> bool:
    for p in modelos.especificacao(modelo).grade:
        valores = p.materializar()
        if valores.index(configuracao[modelos.nome_parametro(modelo, p.nome)]) in (0, len(valores) - 1):
            return True
    return False


def f4(c: co.Consolidacao, base: Mapping[str, int]) -> Dict[str, bytes]:
    com_grade = [m for m in modelos.codigos() if modelos.especificacao(m).grade]
    fig, eixos = plt.subplots(1, len(com_grade), figsize=(LARGURA_PADRAO + 1.5, 3.4))
    linhas = []
    for eixo, m in zip(eixos, com_grade):
        configs = modelos.configuracoes(m)
        prefixo = modelos.PREFIXO_CLASSIFICADOR if modelos.especificacao(m).padronizacao else ""
        freq = [0] * len(configs)
        for e in c.escolhas:
            if e.modelo == m:
                freq[e.indice_configuracao] += 1
        bordas = [_borda_da_configuracao(m, cfgs) for cfgs in configs]
        cores = [COR_NEUTRA if bd else COR[m] for bd in bordas]
        eixo.bar(range(len(configs)), freq, color=cores, edgecolor=["black" if bd else "none" for bd in bordas],
                 linewidth=0.6)
        eixo.set_title(f"{_nome(m)}", fontsize=8)
        eixo.set_xlabel("configuração (ordem de simplicidade)")
        for i, (cfgs, f, bd) in enumerate(zip(configs, freq, bordas)):
            texto = ";".join(f"{k[len(prefixo):]}={v!r}" for k, v in cfgs.items())
            linhas.append((m, i, texto, f, bd))
    eixos[0].set_ylabel("dobras (de 50) que a escolheram")
    fig.suptitle("F4 — Frequência das configurações escolhidas (cinza com contorno: na borda da grade)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    _legenda_base(fig, base, "Borda: algum hiperparâmetro no primeiro ou no último valor da grade.")
    tabela = rl.csv_bytes(["modelo", "indice_configuracao", "configuracao", "frequencia", "na_borda"], linhas)
    return _bytes(fig, "F4", tabela)


# --------------------------------------------------------------------------- #
# F5 a F8
# --------------------------------------------------------------------------- #


def f5(c: co.Consolidacao, base: Mapping[str, int]) -> Dict[str, bytes]:
    fig, eixo = plt.subplots(figsize=(4.8, 4.6))
    linhas = []
    for k in c.curvas_roc:
        eixo.fill_between(k.fpr, k.tpr_p025, k.tpr_p975, color=COR[k.modelo], alpha=0.12, lw=0)
        eixo.plot(k.fpr, k.tpr_media, color=COR[k.modelo], lw=1.4, label=_nome(k.modelo))
        linhas += [(k.modelo, f, m, lo, hi) for f, m, lo, hi in zip(k.fpr, k.tpr_media, k.tpr_p025, k.tpr_p975)]
    eixo.plot([0, 1], [0, 1], color=COR_NEUTRA, ls=":", lw=0.8)
    eixo.set_xlabel("1 − especificidade")
    eixo.set_ylabel("Sensibilidade")
    eixo.set_xlim(0, 1)
    eixo.set_ylim(0, 1.01)
    eixo.fill_between([], [], [], color="#BBBBBB", alpha=0.3, label=f"faixa: {ROTULO_FAIXA_ROC}")
    eixo.legend(loc="lower right", fontsize=6.2)
    fig.suptitle("F5 — Curva ROC média vertical, desfecho G", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    _legenda_base(fig, base, "Média das curvas por dobra em grade fixa de 101 pontos; faixa: percentis 2,5 e 97,5 das "
                             "dobras.")
    tabela = rl.csv_bytes(["modelo", "um_menos_especificidade", "sensibilidade_media", "sensibilidade_p2_5",
                           "sensibilidade_p97_5"], linhas)
    return _bytes(fig, "F5", tabela)


def f6(c: co.Consolidacao, base: Mapping[str, int]) -> Dict[str, bytes]:
    fig, eixo = plt.subplots(figsize=(4.8, 4.6))
    linhas = []
    for m in (x for x in modelos.codigos() if modelos.especificacao(x).metricas_probabilisticas):
        faixas = [f for f in c.calibracao if f.modelo == m]
        eixo.plot([f.probabilidade_media for f in faixas], [f.frequencia_observada for f in faixas], marker="o", ms=3,
                  lw=1.0, color=COR[m], label=_nome(m))
        linhas += [(f.modelo, f.faixa, f.nodulos, f.probabilidade_media, f.frequencia_observada) for f in faixas]
    eixo.plot([0, 1], [0, 1], color=COR_NEUTRA, ls=":", lw=0.8)
    eixo.set_xlabel("Probabilidade prevista (média da faixa)")
    eixo.set_ylabel("Frequência observada de malignos")
    eixo.set_xlim(0, 1)
    eixo.set_ylim(0, 1)
    eixo.legend(loc="upper left", fontsize=6.5)
    fig.suptitle("F6 — Calibração por decis (G, predições fora da dobra, r = 0)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    _legenda_base(fig, base, "Decis com menos de 10 nódulos são fundidos com o vizinho; o SVM não tem probabilidade.")
    tabela = rl.csv_bytes(["modelo", "faixa", "nodulos", "probabilidade_media", "frequencia_observada"], linhas)
    return _bytes(fig, "F6", tabela)


def f7(c: co.Consolidacao, base: Mapping[str, int]) -> Dict[str, bytes]:
    dens = [d for d in c.denominadores if d.desfecho == "F"]
    fig, eixo = plt.subplots(figsize=(LARGURA_PADRAO, 3.6))
    x = np.arange(len(dens))
    pos = np.array([d.positivos for d in dens])
    neg = np.array([d.negativos for d in dens])
    valida = np.array([d.valida for d in dens])
    eixo.bar(x, pos, color="#D55E00", label="positivos (malignos)", width=0.8)
    eixo.bar(x, neg, bottom=pos, color="#56B4E9", label="negativos (benignos)", width=0.8)
    for xi in x[~valida]:
        eixo.annotate("DD", (xi, pos[xi] + neg[xi]), ha="center", va="bottom", fontsize=5.5, color="black")
    eixo.axhline(cfg.CONFIGURACAO.validade.min_positivos, color="black", ls=":", lw=0.7)
    eixo.set_xlabel("dobra externa (repetição × dobra, em ordem)")
    eixo.set_ylabel("nódulos na faixa de 12 a 30 mm")
    eixo.set_xlim(-1, len(dens))
    eixo.legend(loc="upper right", fontsize=6.5, ncol=2)
    fig.suptitle("F7 — Denominadores do desfecho F por dobra", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    _legenda_base(fig, base, "Linha pontilhada: mínimo de 5 de cada classe; DD: dobra degenerada (sem AUC).")
    tabela = rl.csv_bytes(["repeticao", "dobra", "nodulos", "positivos", "negativos", "valida"],
                          [(d.repeticao, d.dobra, d.nodulos, d.positivos, d.negativos, d.valida) for d in dens])
    return _bytes(fig, "F7", tabela)


def f8(c: co.Consolidacao, b: bs.ResultadoBootstrap, base: Mapping[str, int]) -> Dict[str, bytes]:
    familia = [x for x in c.contrastes if x.grupo == co.GRUPO_FAMILIA]
    fig, eixos = plt.subplots(2, 3, figsize=(LARGURA_PADRAO + 1.5, 5.0))
    linhas = []
    for eixo, x in zip(eixos.ravel(), familia):
        valores = b.delta_replicas[(x.a, x.b, x.desfecho)]
        validos = valores[~np.isnan(valores)]
        ic = b.intervalo_contraste(x.a, x.b, x.desfecho)
        descarte = b.descarte(x.desfecho)
        if validos.size:
            contagens, bordas = np.histogram(validos, bins=40)
            eixo.bar(bordas[:-1], contagens, width=np.diff(bordas), align="edge", color=COR[x.a], alpha=0.65)
            linhas += [(x.a, x.b, x.desfecho, float(bordas[i]), float(bordas[i + 1]), int(contagens[i]))
                       for i in range(len(contagens))]
        if ic.relatado:
            for v in (ic.bonferroni_baixo, ic.bonferroni_alto):
                eixo.axvline(v, color="black", lw=1.1)
            for v in (ic.ic95_baixo, ic.ic95_alto):
                eixo.axvline(v, color="black", lw=0.8, ls="--")
        eixo.axvline(0.0, color=COR_NEUTRA, lw=0.8, ls=":")
        eixo.set_title(f"{_nome(x.a)} − {_nome(x.b)} ({x.desfecho})", fontsize=7)
        eixo.text(0.02, 0.97, f"descarte: {100 * descarte.fracao:.2f}%{' (IQ)' if descarte.iq else ''}",
                  transform=eixo.transAxes, fontsize=6, va="top",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
        eixo.tick_params(labelsize=6)
        eixo.xaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.suptitle("F8 — Histograma de Δ* do bootstrap por contraste da família", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    _legenda_base(fig, base, "Linha cheia: percentis do IC com Bonferroni; tracejada: IC de 95% sem correção; "
                             "pontilhada: zero.")
    linhas = [(a_, b_, d_, "bin", lo, hi, n, None, None) for a_, b_, d_, lo, hi, n in linhas]
    linhas += [(x.a, x.b, x.desfecho, "descarte", None, None, b.descarte(x.desfecho).descartadas,
                b.descarte(x.desfecho).fracao, b.descarte(x.desfecho).iq) for x in familia]
    tabela = rl.csv_bytes(["a", "b", "desfecho", "tipo", "limite_baixo_do_bin", "limite_alto_do_bin", "replicas",
                           "fracao_descartada", "iq"], linhas)
    return _bytes(fig, "F8", tabela)


# --------------------------------------------------------------------------- #
# Conjunto
# --------------------------------------------------------------------------- #


def gerar_todas(c: co.Consolidacao, b: bs.ResultadoBootstrap, base: Mapping[str, int]) -> Dict[str, bytes]:
    """Todas as figuras F1…F8 (PNG, SVG e CSV) como bytes. ``base`` = ``{"pacientes": …, "nodulos": …}``."""
    esperadas = [f.codigo for f in cfg.CONFIGURACAO.saidas.figuras]
    if esperadas != [f"F{i}" for i in range(1, 9)]:
        raise ValueError("a configuração deve declarar F1…F8")
    with matplotlib.rc_context(_estilo()):
        saidas: Dict[str, bytes] = {}
        for gerar in (lambda: f1(c, b, base), lambda: f2(c, b, base), lambda: f3(c, base), lambda: f4(c, base),
                      lambda: f5(c, base), lambda: f6(c, base), lambda: f7(c, base), lambda: f8(c, b, base)):
            saidas.update(gerar())
    return saidas


def nomes_esperados() -> Tuple[str, ...]:
    formatos = cfg.CONFIGURACAO.saidas.formatos_figura
    return tuple(sorted([f"figuras/F{i}.{fmt}" for i in range(1, 9) for fmt in formatos]
                        + [f"figuras/dados/F{i}.csv" for i in range(1, 9)]))


__all__ = ["COR", "ROTULO_FAIXA_ROC", "gerar_todas", "nomes_esperados"]
