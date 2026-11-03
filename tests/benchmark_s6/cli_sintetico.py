"""Linha de comando do coordenador com dados SINTÉTICOS injetados, para o teste de ``spawn`` no Windows.

Roda como script (``python cli_sintetico.py --modo implementacao …``): os workers ``spawn`` reimportam este arquivo
como ``__mp_main__`` e, graças à guarda, NÃO reiniciam o coordenador. Só testes: a produção não tem como injetar dados.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sinteticos as sn  # noqa: E402
from src.benchmark_s6 import coordenador as co  # noqa: E402
from src.benchmark_s6 import executor as ex  # noqa: E402
from src.benchmark_s6 import invariante as inv  # noqa: E402


def baseline_sintetico(caminho: Path) -> None:
    carga = sn.carga_definitiva(n_pacientes=90, modo="implementacao")
    modelo = inv.ajustar_lr_fixa(carga.X(), carga.y())
    esc, clf = modelo.named_steps["padronizacao"], modelo.named_steps["classificador"]
    nomes = list(carga.caracteristicas)
    por = lambda v: {n: float(x) for n, x in zip(nomes, v)}  # noqa: E731
    caminho.write_text(json.dumps({
        "features": nomes, "prevencao_vazamento": {"scaler_n_samples_seen": len(carga.y())},
        "coeficientes": {"por_feature": por(clf.coef_[0]), "intercepto": float(clf.intercept_[0]),
                         "scaler_mean": por(esc.mean_), "scaler_scale": por(esc.scale_)}}), encoding="utf-8")


def dependencias(baseline: Path) -> co.Dependencias:
    return co.Dependencias(
        carregar=lambda modo, manifesto: sn.carregar_implementacao_sintetica(),
        carregador_dos_workers=lambda modo, manifesto: sn.carregar_implementacao_sintetica,
        estado_git=lambda: SimpleNamespace(head="a" * 40, branch="teste", limpo=True), executar=ex.executar,
        ambiente=co.ambiente_instalado, exigir_oficial=False, baseline=baseline)


if __name__ == "__main__":
    sn.usar_grades_reduzidas()
    pasta = Path(tempfile.mkdtemp(prefix="cli-s6-"))
    base = pasta / "baseline.json"
    baseline_sintetico(base)
    sys.exit(co.main(sys.argv[1:], dependencias(base)))
