"""Invariante da LR fixa no treino oficial (P9, Seção 14.3, passo 10; modo ``implementacao``).

A LR fixa do benchmark (``modelos.criar_estimador("lr_fixa")``), ajustada em TODO
o ``train`` oficial, tem de reproduzir os parâmetros de ``baseline_metrics.json``
da Sprint 3: coeficientes e intercepto (``rtol`` 1e-6, ``atol`` 1e-8) e média e
escala do ``StandardScaler`` (``rtol`` 1e-10, ``atol`` 1e-12), as tolerâncias de
``scripts/sprint4/validate_reference_models.py``.

Isto NÃO é métrica de desempenho: só parâmetros ajustados são comparados. Nenhuma
predição é feita nem registrada, e nenhuma AUC, Brier ou taxa é calculada.
O resultado é agregado e não traz identificadores.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from . import adaptadores_s4 as s4
from . import config as cfg
from . import modelos

CAMINHO_BASELINE = s4.REPO_ROOT / "reports" / "sprint3" / "baseline" / "baseline_metrics.json"
TOLERANCIAS = {
    "scaler_mean": {"rtol": 1e-10, "atol": 1e-12},
    "scaler_scale": {"rtol": 1e-10, "atol": 1e-12},
    "coeficientes": {"rtol": 1e-6, "atol": 1e-8},
    "intercepto": {"rtol": 1e-6, "atol": 1e-8},
}


class InvarianteViolado(RuntimeError):
    """Parâmetros da LR fixa diferentes da referência da S3. Chave de saída: integridade."""

    chave_saida = "integridade"


def ajustar_lr_fixa(X: np.ndarray, y: np.ndarray) -> Any:
    """A LR fixa do benchmark ajustada em ``X``, ``y`` (sem predição)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return modelos.criar_estimador("lr_fixa").fit(X, y)


def _comparar(obtido: Mapping[str, float], referencia: Mapping[str, float], tol: Mapping[str, float]) -> Dict[str, Any]:
    if set(obtido) != set(referencia):
        raise InvarianteViolado("chaves diferentes de baseline_metrics.json")
    termos = {}
    for nome in sorted(obtido):
        o, e = float(obtido[nome]), float(referencia[nome])
        termos[nome] = {"obtido": o, "referencia": e, "diferenca_absoluta": abs(o - e),
                        "dentro_da_tolerancia": bool(np.isclose(o, e, rtol=tol["rtol"], atol=tol["atol"]))}
    return {"tolerancia": dict(tol), "termos": termos, "dentro_da_tolerancia": all(
        t["dentro_da_tolerancia"] for t in termos.values())}


def verificar(X: np.ndarray, y: np.ndarray, caminho: Path = CAMINHO_BASELINE) -> Dict[str, Any]:
    """Ajusta a LR fixa em ``X``, ``y`` e compara com a referência da S3. Levanta ``InvarianteViolado`` se falhar."""
    caracteristicas = list(cfg.CONFIGURACAO.caracteristicas)
    if X.shape != (len(y), len(caracteristicas)):
        raise InvarianteViolado(f"X com forma {X.shape} para {len(y)} linhas e {len(caracteristicas)} características")
    bruto = Path(caminho).read_bytes()
    referencia = json.loads(bruto.decode("utf-8"))
    if list(referencia["features"]) != caracteristicas:
        raise InvarianteViolado("características de baseline_metrics.json diferem da configuração")
    modelo = ajustar_lr_fixa(np.asarray(X, dtype=np.float64), np.asarray(y))
    escalador, classificador = modelo.named_steps["padronizacao"], modelo.named_steps["classificador"]
    n_vistos = int(np.atleast_1d(escalador.n_samples_seen_)[0])
    if n_vistos != referencia["prevencao_vazamento"]["scaler_n_samples_seen"] or n_vistos != len(y):
        raise InvarianteViolado(f"o padronizador viu {n_vistos} linhas; a referência, "
                                f"{referencia['prevencao_vazamento']['scaler_n_samples_seen']}")
    ref = referencia["coeficientes"]
    obtido = {
        "coeficientes": ({f: float(c) for f, c in zip(caracteristicas, classificador.coef_[0])}, ref["por_feature"]),
        "intercepto": ({"intercepto": float(classificador.intercept_[0])}, {"intercepto": ref["intercepto"]}),
        "scaler_mean": ({f: float(v) for f, v in zip(caracteristicas, escalador.mean_)}, ref["scaler_mean"]),
        "scaler_scale": ({f: float(v) for f, v in zip(caracteristicas, escalador.scale_)}, ref["scaler_scale"]),
    }
    grupos = {nome: _comparar(o, r, TOLERANCIAS[nome]) for nome, (o, r) in obtido.items()}
    resultado = {
        "versao_esquema": 1, "modelo": "lr_fixa", "n_linhas_de_ajuste": len(y),
        "sha256_baseline_metrics": s4.calcular_sha256(caminho)[1], "grupos": grupos,
        "dentro_da_tolerancia": all(g["dentro_da_tolerancia"] for g in grupos.values()),
        "observacao": "Compara só parâmetros ajustados; nenhuma predição nem métrica de desempenho é calculada.",
    }
    if not resultado["dentro_da_tolerancia"]:
        fora = [f"{g}/{n}" for g, d in grupos.items() for n, t in d["termos"].items() if not t["dentro_da_tolerancia"]]
        raise InvarianteViolado(f"parâmetros fora da tolerância: {fora}")
    return resultado


__all__ = ["CAMINHO_BASELINE", "InvarianteViolado", "TOLERANCIAS", "ajustar_lr_fixa", "verificar"]
