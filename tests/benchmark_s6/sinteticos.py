"""Resultados sintéticos de unidades para os testes de métricas, bootstrap, seleção e relatórios.

Nada é treinado: as pontuações vêm de um gerador controlado. Só dados sintéticos
(``oficial=False``); o contexto é do modo ``definitivo`` (10 repetições × 5 dobras × 5 modelos = 250 unidades).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.benchmark_s6 import config as cfg
from src.benchmark_s6 import dados, modelos
from src.benchmark_s6 import unidades as un
from test_folds import amostra_sintetica

CODIGOS = tuple(modelos.codigos())
QUALIDADE = {"lr_fixa": 2.0, "lr_tuning": 2.0, "svm": 2.0, "rf": 2.0, "xgb": 2.0}

Pontuador = Callable[[str, int, int, np.ndarray, np.ndarray, np.random.Generator], np.ndarray]


def carga_definitiva(n_pacientes: int = 250, semente: int = 0, modo: str = "definitivo") -> dados.DadosBenchmark:
    """Base sintética: diâmetro uniforme em [5, 45) mm (cerca de 45% na faixa 12 ≤ D < 30)."""
    a = amostra_sintetica(n_pacientes, semente=semente)
    rng = np.random.default_rng(semente + 1)
    X = np.column_stack([rng.lognormal(6, 1, a.n), rng.uniform(5.0, 45.0, a.n), rng.uniform(0.3, 0.9, a.n)])
    frame = pd.DataFrame({cfg.COLUNA_NODULO: a.ids, cfg.COLUNA_PACIENTE: a.grupos,
                          dados.COLUNA_PARTICAO: "train", cfg.COLUNA_ALVO: a.y})
    for i, nome in enumerate(cfg.CONFIGURACAO.caracteristicas):
        frame[nome] = X[:, i]
    return dados.DadosBenchmark(modo, dados.PARTICOES_POR_MODO[modo], False, tuple(cfg.CONFIGURACAO.caracteristicas),
                                {}, {}, None, frame)


def pontuador_padrao(qualidade: Optional[Dict[str, float]] = None) -> Pontuador:
    q = {**QUALIDADE, **(qualidade or {})}

    def pontuar(modelo, r, k, y, posicoes, rng):
        sinal = q[modelo] * (2.0 * y - 1.0) + rng.normal(size=len(y)) * 1.5
        if modelo == "svm":
            return sinal * 0.8 - 0.1  # decision_function: escala e viés arbitrários
        return 1.0 / (1.0 + np.exp(-sinal))

    return pontuar


def resultado(ctx: un.ContextoExecucao, unidade: un.Unidade, pontuacoes: np.ndarray, indice: int = 0,
              avisos=(), na_borda: bool = False) -> un.ResultadoUnidade:
    dobra = ctx._dobra(unidade)
    espec = modelos.especificacao(unidade.modelo)
    av = dobra.externa.avaliacao
    amostra = ctx.amostra
    registro = un.RegistroUnidade(
        unidade=unidade, modo=ctx.modo, oficial=ctx.oficial, sha256_configuracao=ctx.sha256_configuracao,
        fingerprint_plano=ctx.plano.fingerprint, fingerprint_dobra=dobra.fingerprint,
        metodo_pontuacao=espec.pontuacao,
        configuracao=tuple((p.nome, p.materializar()[indice % len(p.materializar())]) for p in espec.grade),
        indice_configuracao=indice % espec.n_configuracoes, n_configuracoes=espec.n_configuracoes,
        borda=tuple((p.nome, p.materializar()[0], 0, len(p.materializar()), na_borda) for p in espec.grade),
        n_ajustes=espec.ajustes_por_dobra_externa(5), avisos=tuple(avisos),
        contagem_treino=un._contagem(amostra, dobra.externa.treino), contagem_avaliacao=un._contagem(amostra, av))
    posicoes = np.array(av, dtype=np.int64)
    y = np.array(amostra.y[av], dtype=np.int64)
    s = np.array(pontuacoes, dtype=np.float64)
    for arr in (posicoes, y, s):
        arr.setflags(write=False)
    medias = tuple(0.9 for _ in range(espec.n_configuracoes if espec.grade else 0))
    privado = un.PredicoesPrivadas(unidade, dobra.fingerprint, espec.pontuacao, posicoes,
                                   tuple(str(i) for i in amostra.ids[av]), y, s, medias)
    return un.ResultadoUnidade(registro, privado)


def resultados_sinteticos(ctx: un.ContextoExecucao, pontuador: Optional[Pontuador] = None, semente: int = 0,
                          avisos: Optional[Dict[str, tuple]] = None) -> List[un.ResultadoUnidade]:
    pontuador = pontuador or pontuador_padrao()
    saida = []
    for u in ctx.unidades():
        av = ctx._dobra(u).externa.avaliacao
        # Semente por (r, k) e por modelo: ruídos independentes entre modelos, mesmos nódulos.
        rng = np.random.default_rng([semente, u.repeticao, u.dobra, CODIGOS.index(u.modelo)])
        y = ctx.amostra.y[av]
        s = pontuador(u.modelo, u.repeticao, u.dobra, y, av, rng)
        saida.append(resultado(ctx, u, s, indice=u.repeticao + u.dobra, avisos=(avisos or {}).get(u.modelo, ())))
    return saida


# --------------------------------------------------------------------------- #
# Grades reduzidas (SÓ testes) e carregadores serializáveis para workers ``spawn``
# --------------------------------------------------------------------------- #

import dataclasses  # noqa: E402
import functools  # noqa: E402

_ESPECIFICACAO_ORIGINAL = modelos.especificacao
_REDUCOES = {"svm": {"C": 2}, "rf": {"min_samples_leaf": 2}, "xgb": {"max_depth": 2}}


@functools.lru_cache(maxsize=None)
def especificacao_reduzida(codigo: str):
    """Especificação com grade reduzida (só nos testes): 2 configurações para ``svm``, ``rf`` e ``xgb``."""
    espec = _ESPECIFICACAO_ORIGINAL(codigo)
    if codigo not in _REDUCOES:
        return espec
    grade = tuple(dataclasses.replace(p, valores=tuple(p.materializar())[:_REDUCOES[codigo].get(p.nome, 1)])
                  for p in espec.grade)
    return dataclasses.replace(espec, grade=grade)


def usar_grades_reduzidas() -> None:
    """Troca ``modelos.especificacao`` NESTE processo (workers a chamam no carregador). Só testes."""
    modelos.especificacao = especificacao_reduzida


def carregar_implementacao_sintetica() -> dados.DadosBenchmark:
    """Carregador de worker: grades reduzidas + base sintética do modo ``implementacao`` (90 pacientes)."""
    usar_grades_reduzidas()
    return carga_definitiva(n_pacientes=90, modo="implementacao")


def carregar_implementacao_sintetica_x_alterado() -> dados.DadosBenchmark:
    """Como ``carregar_implementacao_sintetica``, mas com UM valor de X alterado em 1e-9: o hash de X diverge."""
    usar_grades_reduzidas()
    carga = carga_definitiva(n_pacientes=90, modo="implementacao")
    frame = carga.frame()
    coluna = cfg.CONFIGURACAO.caracteristicas[0]
    frame.loc[frame.index[0], coluna] += 1e-9
    return dataclasses.replace(carga, _frame=frame)
