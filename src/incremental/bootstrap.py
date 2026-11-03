"""Bootstrap pareado por paciente da Sprint 4 (protocolo, Seção 8.2).

Implementação PRÓPRIA da S4, independente de ``src/benchmark_s6`` (outro desenho: B = 10.000,
Bonferroni, ponderação). Parâmetros de ``config.ESPEC_BOOTSTRAP``:

* unidade de reamostragem: ``patient_id``. Os pacientes são sorteados com reposição e cada um entra
  com TODOS os seus nódulos;
* pareamento: em cada réplica a MESMA amostra de pacientes vale para todos os modelos, e os contrastes
  são calculados dentro da réplica;
* B = 2000 réplicas, seed 42 (``numpy.random.default_rng``); os B sorteios são feitos de uma vez,
  em ordem, e não dependem dos modelos nem dos contrastes pedidos;
* IC percentil, limites 2,5 e 97,5 (``numpy.percentile``, interpolação linear);
* réplica com uma única classe é DESCARTADA e CONTADA (não é sorteada de novo);
* sem retreino: as probabilidades são as do modelo já ajustado.

Métricas: AUC-ROC (principal) e, para a Seção 10.2, Brier e log-loss, todas na mesma réplica.
Nenhum p-valor é calculado (Seção 8.3). O módulo não lê dados, não ajusta modelos e não grava nada.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as cfg

#: ``log_loss`` com probabilidades limitadas a [EPS, 1 - EPS], como o scikit-learn.
EPS = 1e-15
METRICAS: Tuple[str, ...] = ("auc", "brier", "log_loss")


class BootstrapInvalido(ValueError):
    """Entrada inconsistente com o bootstrap da Seção 8.2."""


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    q = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1.0 - q)))


_FUNCOES: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] = MappingProxyType(
    {"auc": _auc, "brier": _brier, "log_loss": _log_loss})


@dataclass(frozen=True)
class Intervalo:
    """Estimativa pontual na amostra completa e IC percentil das réplicas válidas."""

    ponto: float
    limite_inferior: float
    limite_superior: float

    def como_dict(self) -> dict:
        return {"ponto": self.ponto, "limite_inferior": self.limite_inferior,
                "limite_superior": self.limite_superior}


@dataclass(frozen=True)
class ResultadoBootstrap:
    """Resultado agregado (sem identificadores nem predições)."""

    n_replicas: int
    n_validas: int
    n_descartadas: int
    seed: int
    percentis: Tuple[float, float]
    n_pacientes: int
    n_nodulos: int
    #: ``{metrica: {modelo: Intervalo}}``
    por_modelo: Mapping[str, Mapping[str, Intervalo]]
    #: ``{metrica: {"A-B": Intervalo}}`` com Δ = metrica(A) − metrica(B), dentro da réplica.
    contrastes: Mapping[str, Mapping[str, Intervalo]]

    def como_dict(self) -> dict:
        return {
            "n_replicas": self.n_replicas, "n_validas": self.n_validas, "n_descartadas": self.n_descartadas,
            "seed": self.seed, "percentis": list(self.percentis), "n_pacientes": self.n_pacientes,
            "n_nodulos": self.n_nodulos, "unidade": cfg.ESPEC_BOOTSTRAP.unidade, "pareado": True,
            "retreino": False, "gerador": "numpy.random.default_rng", "ic": "percentil",
            "por_modelo": {m: {k: v.como_dict() for k, v in d.items()} for m, d in self.por_modelo.items()},
            "contrastes": {m: {k: v.como_dict() for k, v in d.items()} for m, d in self.contrastes.items()},
        }


def _validar(pacientes: Any, y: Any, probas: Mapping[str, Any], contrastes: Sequence[Tuple[str, str]]
             ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    pacientes = np.asarray(pacientes)
    y = np.asarray(y)
    if pacientes.ndim != 1 or y.ndim != 1 or len(pacientes) != len(y) or len(y) == 0:
        raise BootstrapInvalido("pacientes e y devem ser vetores do mesmo tamanho, não vazios")
    if y.dtype.kind not in "iu" or set(np.unique(y).tolist()) != {0, 1}:
        raise BootstrapInvalido("y deve ser inteiro 0/1 com as duas classes")
    if not isinstance(probas, Mapping) or not probas:
        raise BootstrapInvalido("probas deve mapear modelo → probabilidades")
    saida: Dict[str, np.ndarray] = {}
    for modelo, p in probas.items():
        p = np.asarray(p, dtype=np.float64)
        if p.shape != y.shape or not np.isfinite(p).all() or p.min() < 0.0 or p.max() > 1.0:
            raise BootstrapInvalido(f"{modelo}: probabilidades desalinhadas, não finitas ou fora de [0, 1]")
        saida[str(modelo)] = p
    for a, b in contrastes:
        if a not in saida or b not in saida or a == b:
            raise BootstrapInvalido(f"contraste inválido: {a}-{b}")
    return pacientes, y.astype(np.int64), saida


def bootstrap_pareado(pacientes: Any, y: Any, probas: Mapping[str, Any],
                      contrastes: Sequence[Tuple[str, str]],
                      espec: cfg.EspecBootstrap = cfg.ESPEC_BOOTSTRAP) -> ResultadoBootstrap:
    """Bootstrap por paciente com a mesma amostra para todos os modelos (Seção 8.2).

    ``pacientes``, ``y`` e cada ``probas[modelo]`` estão alinhados por nódulo. ``contrastes`` é uma
    sequência de pares ``(A, B)`` com Δ = A − B. Com a mesma ``espec``, o resultado é idêntico
    qualquer que seja o conjunto de modelos e contrastes pedido.
    """
    pacientes, y, probas = _validar(pacientes, y, probas, contrastes)
    if (espec.unidade, espec.pareado, espec.retreino, espec.descartar_replicas_com_uma_classe) != \
            (cfg.COLUNA_PACIENTE, True, False, True):
        raise BootstrapInvalido("espec diferente da Seção 8.2 (unidade, pareamento, retreino ou descarte)")
    nomes_pac, inverso = np.unique(pacientes, return_inverse=True)
    n_pac = len(nomes_pac)
    por_paciente: List[np.ndarray] = [np.flatnonzero(inverso == i) for i in range(n_pac)]
    rng = np.random.default_rng(espec.seed)
    sorteios = rng.integers(0, n_pac, size=(espec.n_replicas, n_pac))

    pares = [f"{a}-{b}" for a, b in contrastes]
    valores: Dict[str, Dict[str, List[float]]] = {m: {n: [] for n in probas} for m in METRICAS}
    deltas: Dict[str, Dict[str, List[float]]] = {m: {p: [] for p in pares} for m in METRICAS}
    descartadas = 0
    for linha in sorteios:
        indices = np.concatenate([por_paciente[i] for i in linha])
        y_b = y[indices]
        if y_b.min() == y_b.max():
            descartadas += 1
            continue
        for metrica in METRICAS:
            funcao = _FUNCOES[metrica]
            calculado = {n: funcao(y_b, p[indices]) for n, p in probas.items()}
            for n, v in calculado.items():
                valores[metrica][n].append(v)
            for (a, b), par in zip(contrastes, pares):
                deltas[metrica][par].append(calculado[a] - calculado[b])
    n_validas = espec.n_replicas - descartadas
    if n_validas == 0:
        raise BootstrapInvalido("todas as réplicas têm uma única classe")

    def intervalo(ponto: float, amostra: List[float]) -> Intervalo:
        baixo, alto = np.percentile(np.asarray(amostra), espec.percentis)
        return Intervalo(float(ponto), float(baixo), float(alto))

    pontos = {m: {n: _FUNCOES[m](y, p) for n, p in probas.items()} for m in METRICAS}
    por_modelo = {m: MappingProxyType({n: intervalo(pontos[m][n], valores[m][n]) for n in probas})
                  for m in METRICAS}
    por_contraste = {m: MappingProxyType({par: intervalo(pontos[m][a] - pontos[m][b], deltas[m][par])
                                          for (a, b), par in zip(contrastes, pares)}) for m in METRICAS}
    return ResultadoBootstrap(
        n_replicas=espec.n_replicas, n_validas=n_validas, n_descartadas=descartadas, seed=espec.seed,
        percentis=tuple(espec.percentis), n_pacientes=n_pac, n_nodulos=len(y),
        por_modelo=MappingProxyType(por_modelo), contrastes=MappingProxyType(por_contraste))


__all__ = ["BootstrapInvalido", "EPS", "Intervalo", "METRICAS", "ResultadoBootstrap", "bootstrap_pareado"]
