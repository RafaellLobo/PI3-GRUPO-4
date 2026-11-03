"""Bootstrap de avaliação por paciente (P9, Seções 8.2, 9 e 6).

**O que é reamostrado.** Os PACIENTES, com reposição, sobre a base inteira
(``P`` pacientes, 353 no definitivo). Cada réplica sorteia ``P`` pacientes; o
peso de uma linha é o número de vezes que o seu paciente foi sorteado. Reamostrar
nódulos em vez de pacientes seria errado: os nódulos de um mesmo paciente
andam juntos (todos recebem o mesmo peso).

**O que NÃO é refeito.** Nenhum modelo é reajustado: as predições fora da dobra
são fixas (``Pareados``). O intervalo é, portanto, condicional às partições, às
configurações escolhidas e aos modelos já ajustados (P9, Seção 8.3).

**Geração.** ``numpy.random.default_rng(semente)``, UMA chamada
``rng.integers(0, P, size=(B, P))`` (ordem C: linha = réplica) com os pacientes
numerados na ordem alfabética dos ``patient_id``. Os pesos são a contagem de cada
paciente na linha. ``B = 10.000`` e a semente 42 vêm da configuração canônica; o
argumento ``n_replicas`` existe só para os testes e nenhum caminho do código
altera ``B``, a semente ou a grade por tempo de execução (P9, Seção 15).

**Em cada réplica**, para cada repetição, dobra e desfecho, a AUC é recalculada
sobre as linhas da dobra com os pesos da réplica, e a regra da Seção 7.3 vale
para nódulos DISTINTOS (peso > 0): menos de 5 positivos ou de 5 negativos
distintos invalida a dobra. Depois vêm a média por repetição (≥ 4 dobras
válidas), ``θ*`` (≥ 8 repetições válidas) e ``Δ* = θ*(a) − θ*(b)``. Como a
validade não depende do modelo, ``θ*(a) − θ*(b)`` é o contraste pareado por dobra.

**Descarte (RD) e IQ.** Réplica em que o desfecho não fica definido (menos de 8
repetições válidas) é descartada para aquele desfecho e contada. Com mais de
``max_fracao_descarte_bootstrap`` (1%) de descartes, o desfecho é IQ: nenhum IC
dele é relatado, e a elegibilidade do algoritmo não muda.

**Intervalos.** Por percentis (``numpy.percentile``, interpolação linear), sobre
as réplicas não descartadas do desfecho:

* ``θ̂`` de cada modelo: IC de 95% (2,5 e 97,5), descritivo, SEM correção;
* contrastes da família (6): Bonferroni, percentis ``100·α/(2·6)`` e o simétrico
  (0,41667 e 99,58333), mais o IC de 95% em anexo;
* contrastes descritivos: IC de 95%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from . import config as cfg
from . import consolidacao as co
from . import metricas as mt
from . import modelos

TAMANHO_BLOCO = 1000


class BootstrapInvalido(ValueError):
    """Entrada incoerente. Chave de saída: integridade."""

    chave_saida = "integridade"


# --------------------------------------------------------------------------- #
# Pesos e AUC ponderada
# --------------------------------------------------------------------------- #


def pesos_pacientes(n_pacientes: int, n_replicas: int, semente: int) -> np.ndarray:
    """Matriz ``(B, P)`` int32 de multiplicidades: quantas vezes cada paciente foi sorteado em cada réplica."""
    rng = np.random.default_rng(semente)
    sorteios = rng.integers(0, n_pacientes, size=(n_replicas, n_pacientes))
    pesos = np.zeros((n_replicas, n_pacientes), dtype=np.int32)
    for b in range(n_replicas):
        pesos[b] = np.bincount(sorteios[b], minlength=n_pacientes)
    return pesos


def matriz_pares(s_positivos: np.ndarray, s_negativos: np.ndarray) -> np.ndarray:
    """``C[i, j] = 1`` se o positivo ``i`` tem escore maior que o negativo ``j``, ``0,5`` no empate, ``0`` senão."""
    p = np.asarray(s_positivos, dtype=np.float64)[:, None]
    n = np.asarray(s_negativos, dtype=np.float64)[None, :]
    return (p > n).astype(np.float64) + 0.5 * (p == n)


def auc_ponderada(y: np.ndarray, pontuacoes: np.ndarray, pesos: np.ndarray) -> float:
    """AUC com pesos inteiros por linha (igual à AUC das linhas repetidas ``peso`` vezes). Exige as duas classes."""
    y, s, w = np.asarray(y), np.asarray(pontuacoes, dtype=np.float64), np.asarray(pesos, dtype=np.float64)
    wp, wn = w[y == 1], w[y == 0]
    if wp.sum() <= 0 or wn.sum() <= 0:
        raise BootstrapInvalido("AUC ponderada exige peso nas duas classes")
    return float(wp @ matriz_pares(s[y == 1], s[y == 0]) @ wn / (wp.sum() * wn.sum()))


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #


class DescarteDesfecho(NamedTuple):
    """RD e IQ de um desfecho (P9, Seção 6)."""

    desfecho: str
    n_replicas: int
    descartadas: int
    fracao: float
    iq: bool


class IntervaloTheta(NamedTuple):
    """IC de 95% descritivo de ``θ̂``, sem correção de multiplicidade. ``None`` se NE ou IQ."""

    modelo: str
    desfecho: str
    relatado: bool
    motivo: str
    theta: Optional[float]
    inferior: Optional[float]
    superior: Optional[float]


class IntervaloContraste(NamedTuple):
    """IC de 95% e, na família, IC com Bonferroni. ``None`` se NE ou IQ."""

    a: str
    b: str
    desfecho: str
    grupo: str
    relatado: bool
    motivo: str
    delta: Optional[float]
    ic95_inferior: Optional[float]
    ic95_superior: Optional[float]
    bonferroni_inferior: Optional[float]
    bonferroni_superior: Optional[float]


@dataclass(frozen=True, eq=False)
class ResultadoBootstrap:
    """Resultado do bootstrap. Só agregados globais (nenhum valor por paciente ou nódulo)."""

    n_replicas: int
    semente: int
    percentis_descritivos: Tuple[float, float]
    percentis_familia: Tuple[float, float]
    descartes: Tuple[DescarteDesfecho, ...]
    intervalos_theta: Tuple[IntervaloTheta, ...]
    intervalos_contraste: Tuple[IntervaloContraste, ...]
    #: ``θ*`` por (modelo, desfecho): vetor de B, ``nan`` onde a réplica foi descartada para o desfecho.
    theta_replicas: Dict[Tuple[str, str], np.ndarray] = field(repr=False)
    #: ``Δ*`` por (a, b, desfecho): vetor de B, ``nan`` nos descartes.
    delta_replicas: Dict[Tuple[str, str, str], np.ndarray] = field(repr=False)

    def descarte(self, desfecho: str) -> DescarteDesfecho:
        return next(d for d in self.descartes if d.desfecho == desfecho)

    def intervalo_theta(self, modelo: str, desfecho: str) -> IntervaloTheta:
        return next(i for i in self.intervalos_theta if (i.modelo, i.desfecho) == (modelo, desfecho))

    def intervalo_contraste(self, a: str, b: str, desfecho: str) -> IntervaloContraste:
        return next(i for i in self.intervalos_contraste if (i.a, i.b, i.desfecho) == (a, b, desfecho))


# --------------------------------------------------------------------------- #
# Cálculo
# --------------------------------------------------------------------------- #


class _Dobra(NamedTuple):
    """O que o bootstrap precisa de uma dobra num desfecho: pacientes dos positivos e dos negativos e as matrizes."""

    repeticao: int
    dobra: int
    pacientes_positivos: np.ndarray
    pacientes_negativos: np.ndarray
    comparacoes: Dict[str, np.ndarray]


def _preparar(p: co.Pareados, desfecho: str) -> List[_Dobra]:
    base, codigos = p.base, modelos.codigos()
    dobras = []
    for r in p.repeticoes:
        for k in range(p.n_dobras):
            ref = p.resultado(codigos[0], r, k).privado
            posicoes = np.asarray(ref.posicoes)
            mascara = base.linhas_do_desfecho(posicoes, desfecho)
            linhas = posicoes[mascara]
            y = np.asarray(ref.y)[mascara]
            comparacoes = {}
            for m in codigos:
                s = np.asarray(p.resultado(m, r, k).privado.pontuacoes)[mascara]
                comparacoes[m] = matriz_pares(s[y == 1], s[y == 0])
            dobras.append(_Dobra(r, k, base.paciente[linhas[y == 1]], base.paciente[linhas[y == 0]], comparacoes))
    return dobras


def _replicas_de_desfecho(p: co.Pareados, desfecho: str, pesos: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """``θ*`` por modelo (``nan`` onde descartada) e máscara de réplicas definidas, para um desfecho."""
    validade = cfg.CONFIGURACAO.validade
    codigos = modelos.codigos()
    n_rep, n_dobras = len(p.repeticoes), p.n_dobras
    B = pesos.shape[0]
    dobras = _preparar(p, desfecho)
    theta = {m: np.full(B, np.nan) for m in codigos}
    definida = np.zeros(B, dtype=bool)
    for inicio in range(0, B, TAMANHO_BLOCO):
        bloco = pesos[inicio:inicio + TAMANHO_BLOCO]
        b = bloco.shape[0]
        auc = {m: np.full((b, n_rep, n_dobras), np.nan) for m in codigos}
        valida = np.zeros((b, n_rep, n_dobras), dtype=bool)
        for i, d in enumerate(dobras):
            r, k = divmod(i, n_dobras)
            if d.pacientes_positivos.size == 0 or d.pacientes_negativos.size == 0:
                continue
            wp = bloco[:, d.pacientes_positivos].astype(np.float64)
            wn = bloco[:, d.pacientes_negativos].astype(np.float64)
            distintos_p, distintos_n = (wp > 0).sum(axis=1), (wn > 0).sum(axis=1)
            ok = (distintos_p >= validade.min_positivos) & (distintos_n >= validade.min_negativos)
            valida[:, r, k] = ok
            if not ok.any():
                continue
            denominador = wp.sum(axis=1) * wn.sum(axis=1)
            for m in codigos:
                numerador = ((wp @ d.comparacoes[m]) * wn).sum(axis=1)
                auc[m][:, r, k] = np.where(ok, numerador / np.where(ok, denominador, 1.0), np.nan)
        n_validas = valida.sum(axis=2)  # (b, R)
        rep_valida = n_validas >= validade.min_dobras_validas_por_repeticao
        ok_desfecho = rep_valida.sum(axis=1) >= validade.min_repeticoes_validas
        definida[inicio:inicio + b] = ok_desfecho
        for m in codigos:
            soma = np.nansum(auc[m], axis=2)
            media_rep = np.where(rep_valida, soma / np.maximum(n_validas, 1), 0.0)
            media = media_rep.sum(axis=1) / np.maximum(rep_valida.sum(axis=1), 1)
            theta[m][inicio:inicio + b] = np.where(ok_desfecho, media, np.nan)
    return theta, definida


def _percentis(valores: np.ndarray, percentis: Tuple[float, float]) -> Tuple[float, float]:
    inferior, superior = np.percentile(valores, percentis)
    return float(inferior), float(superior)


def bootstrap(p: co.Pareados, c: co.Consolidacao, n_replicas: Optional[int] = None) -> ResultadoBootstrap:
    """Bootstrap por paciente sobre predições fixas. ``n_replicas`` só deve ser informado nos testes."""
    if not isinstance(p, co.Pareados) or not isinstance(c, co.Consolidacao):
        raise TypeError("p deve ser Pareados e c, Consolidacao")
    espec = cfg.CONFIGURACAO.bootstrap
    B = espec.n_replicas if n_replicas is None else n_replicas
    if type(B) is not int or B < 2:
        raise BootstrapInvalido(f"número de réplicas inválido: {B!r}")
    if c.modo != p.base.modo:
        raise BootstrapInvalido("consolidação e pareamento de modos diferentes")
    pesos = pesos_pacientes(p.base.n_pacientes, B, espec.semente)
    descritivo = espec.percentis_descritivos()
    familia = cfg.CONFIGURACAO.contrastes.percentis_familia()
    teto = cfg.CONFIGURACAO.validade.max_fracao_descarte_bootstrap

    theta_replicas: Dict[Tuple[str, str], np.ndarray] = {}
    descartes: List[DescarteDesfecho] = []
    definidas: Dict[str, np.ndarray] = {}
    for desfecho in co.DESFECHOS:
        theta, definida = _replicas_de_desfecho(p, desfecho, pesos)
        definidas[desfecho] = definida
        for m, valores in theta.items():
            theta_replicas[(m, desfecho)] = valores
        descartadas = int(B - definida.sum())
        descartes.append(DescarteDesfecho(desfecho, B, descartadas, descartadas / B, descartadas / B > teto))

    delta_replicas: Dict[Tuple[str, str, str], np.ndarray] = {}
    for a, b, _ in co.pares_de_contraste():
        for desfecho in co.DESFECHOS:
            delta_replicas[(a, b, desfecho)] = theta_replicas[(a, desfecho)] - theta_replicas[(b, desfecho)]

    def situacao(desfecho: str) -> str:
        estado = next(e for e in c.estados if e.desfecho == desfecho)
        if estado.nao_estimavel:
            return "NE"
        return "IQ" if next(d for d in descartes if d.desfecho == desfecho).iq else ""

    intervalos_theta: List[IntervaloTheta] = []
    for desfecho in co.DESFECHOS:
        motivo = situacao(desfecho)
        for m in modelos.codigos():
            est = c.estimativa(m, desfecho)
            if motivo:
                intervalos_theta.append(IntervaloTheta(m, desfecho, False, motivo, est.theta, None, None))
            else:
                inf, sup = _percentis(theta_replicas[(m, desfecho)][definidas[desfecho]], descritivo)
                intervalos_theta.append(IntervaloTheta(m, desfecho, True, "", est.theta, inf, sup))

    intervalos_contraste: List[IntervaloContraste] = []
    for a, b, grupo in co.pares_de_contraste():
        for desfecho in co.DESFECHOS:
            motivo = situacao(desfecho)
            delta = c.contraste(a, b, desfecho).delta
            if motivo:
                intervalos_contraste.append(IntervaloContraste(a, b, desfecho, grupo, False, motivo, delta,
                                                               None, None, None, None))
                continue
            validos = delta_replicas[(a, b, desfecho)][definidas[desfecho]]
            ic95 = _percentis(validos, descritivo)
            bonf = _percentis(validos, familia) if grupo == co.GRUPO_FAMILIA else (None, None)
            intervalos_contraste.append(IntervaloContraste(a, b, desfecho, grupo, True, "", delta, ic95[0], ic95[1],
                                                           bonf[0], bonf[1]))
    return ResultadoBootstrap(B, espec.semente, descritivo, familia, tuple(descartes), tuple(intervalos_theta),
                              tuple(intervalos_contraste), theta_replicas, delta_replicas)


__all__ = [
    "BootstrapInvalido", "DescarteDesfecho", "IntervaloContraste", "IntervaloTheta", "ResultadoBootstrap",
    "auc_ponderada", "bootstrap", "matriz_pares", "pesos_pacientes",
]
