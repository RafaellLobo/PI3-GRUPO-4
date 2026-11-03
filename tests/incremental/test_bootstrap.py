"""Testes do bootstrap pareado por paciente da S4 (``src/incremental/bootstrap.py``, Seção 8.2).

Dados sintéticos apenas. A referência é um cálculo independente: os sorteios de pacientes são refeitos
com ``numpy.random.default_rng`` e os nódulos de cada réplica são montados com pandas; a métrica vem do
scikit-learn. Nada é lido do disco.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/incremental -v
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.incremental import bootstrap as bs  # noqa: E402
from src.incremental import config as cfg  # noqa: E402

ESPEC_PEQUENA = dataclasses.replace(cfg.ESPEC_BOOTSTRAP, n_replicas=200)


def dados_sinteticos(n_pacientes=40, semente=5):
    rng = np.random.default_rng(semente)
    pacientes, y = [], []
    for i in range(n_pacientes):
        for _ in range(int(rng.integers(1, 5))):  # 1 a 4 nódulos por paciente
            pacientes.append(f"P{i:03d}")
            y.append(int(rng.random() < 0.45))
    y = np.array(y)
    sinal = y + rng.normal(0.0, 0.8, len(y))

    def para_proba(x):
        return 1.0 / (1.0 + np.exp(-x))

    return np.array(pacientes), y, {
        "A": para_proba(sinal), "B": para_proba(0.4 * sinal + rng.normal(0.0, 1.0, len(y))),
        "C": para_proba(rng.normal(0.0, 1.0, len(y)))}


def referencia(pacientes, y, probas, contrastes, espec):
    """Cálculo independente: pandas expande os nódulos de cada paciente sorteado."""
    quadro = pd.DataFrame({"p": pacientes, "y": y, **{m: v for m, v in probas.items()}})
    unicos = sorted(set(pacientes))
    por_pac = {p: g for p, g in quadro.groupby("p")}
    rng = np.random.default_rng(espec.seed)
    sorteios = rng.integers(0, len(unicos), size=(espec.n_replicas, len(unicos)))
    auc = {m: [] for m in probas}
    delta = {f"{a}-{b}": [] for a, b in contrastes}
    descartadas = 0
    for linha in sorteios:
        bloco = pd.concat([por_pac[unicos[i]] for i in linha])
        if bloco["y"].nunique() < 2:
            descartadas += 1
            continue
        valores = {m: roc_auc_score(bloco["y"], bloco[m]) for m in probas}
        for m, v in valores.items():
            auc[m].append(v)
        for a, b in contrastes:
            delta[f"{a}-{b}"].append(valores[a] - valores[b])
    return auc, delta, descartadas


class TestContraReferencia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pacientes, cls.y, cls.probas = dados_sinteticos()
        cls.contrastes = (("A", "B"), ("A", "C"))
        cls.res = bs.bootstrap_pareado(cls.pacientes, cls.y, cls.probas, cls.contrastes, ESPEC_PEQUENA)
        cls.auc, cls.delta, cls.descartadas = referencia(cls.pacientes, cls.y, cls.probas, cls.contrastes,
                                                         ESPEC_PEQUENA)

    def test_ic_de_auc_e_de_contrastes_batem_com_a_referencia(self):
        for m, amostra in self.auc.items():
            baixo, alto = np.percentile(amostra, [2.5, 97.5])
            intervalo = self.res.por_modelo["auc"][m]
            self.assertAlmostEqual(intervalo.limite_inferior, baixo, places=12)
            self.assertAlmostEqual(intervalo.limite_superior, alto, places=12)
            self.assertAlmostEqual(intervalo.ponto, roc_auc_score(self.y, self.probas[m]), places=12)
        for par, amostra in self.delta.items():
            baixo, alto = np.percentile(amostra, [2.5, 97.5])
            intervalo = self.res.contrastes["auc"][par]
            self.assertAlmostEqual(intervalo.limite_inferior, baixo, places=12)
            self.assertAlmostEqual(intervalo.limite_superior, alto, places=12)
            a, b = par.split("-")
            self.assertAlmostEqual(intervalo.ponto, roc_auc_score(self.y, self.probas[a])
                                   - roc_auc_score(self.y, self.probas[b]), places=12)

    def test_contagens_de_replicas(self):
        self.assertEqual((self.res.n_replicas, self.res.n_descartadas, self.res.n_validas),
                         (200, self.descartadas, 200 - self.descartadas))
        self.assertEqual(self.res.n_pacientes, len(set(self.pacientes.tolist())))
        self.assertEqual(self.res.n_nodulos, len(self.y))

    def test_parametros_da_secao_8_2(self):
        espec = cfg.ESPEC_BOOTSTRAP
        self.assertEqual((espec.n_replicas, espec.seed, espec.percentis, espec.unidade, espec.pareado,
                          espec.retreino), (2000, 42, (2.5, 97.5), "patient_id", True, False))
        d = self.res.como_dict()
        self.assertEqual((d["seed"], d["percentis"], d["unidade"], d["pareado"], d["retreino"]),
                         (42, [2.5, 97.5], "patient_id", True, False))

    def test_b_padrao_e_2000(self):
        res = bs.bootstrap_pareado(*dados_sinteticos(n_pacientes=12), (("A", "B"),))
        self.assertEqual((res.n_replicas, res.n_validas + res.n_descartadas), (2000, 2000))

    def test_brier_e_log_loss_na_mesma_replica(self):
        # Referência independente de Brier e log-loss: mesmos sorteios, mesma amostra para os modelos.
        quadro = pd.DataFrame({"p": self.pacientes, "y": self.y, **self.probas})
        unicos = sorted(set(self.pacientes))
        por_pac = {p: g for p, g in quadro.groupby("p")}
        rng = np.random.default_rng(ESPEC_PEQUENA.seed)
        deltas = []
        for linha in rng.integers(0, len(unicos), size=(200, len(unicos))):
            bloco = pd.concat([por_pac[unicos[i]] for i in linha])
            if bloco["y"].nunique() < 2:
                continue
            brier = {m: float(((bloco[m] - bloco["y"]) ** 2).mean()) for m in ("A", "B")}
            deltas.append(brier["A"] - brier["B"])
        baixo, alto = np.percentile(deltas, [2.5, 97.5])
        intervalo = self.res.contrastes["brier"]["A-B"]
        self.assertAlmostEqual((intervalo.limite_inferior, intervalo.limite_superior), (baixo, alto), places=12)
        # log-loss pelo scikit-learn
        from sklearn.metrics import log_loss
        ponto = log_loss(self.y, self.probas["A"]) - log_loss(self.y, self.probas["B"])
        self.assertAlmostEqual(self.res.contrastes["log_loss"]["A-B"].ponto, ponto, places=12)


class TestPareamentoEIndependencia(unittest.TestCase):
    def test_mesma_amostra_para_todos_os_modelos(self):
        pacientes, y, probas = dados_sinteticos()
        # B é cópia de A: com a MESMA amostra em cada réplica, Δ é exatamente 0 em todas.
        res = bs.bootstrap_pareado(pacientes, y, {"A": probas["A"], "B": probas["A"].copy()}, (("A", "B"),),
                                   ESPEC_PEQUENA)
        for metrica in bs.METRICAS:
            intervalo = res.contrastes[metrica]["A-B"]
            self.assertEqual((intervalo.ponto, intervalo.limite_inferior, intervalo.limite_superior), (0.0, 0.0, 0.0))

    def test_resultado_nao_depende_dos_outros_modelos_nem_contrastes_pedidos(self):
        pacientes, y, probas = dados_sinteticos()
        tudo = bs.bootstrap_pareado(pacientes, y, probas, (("A", "B"), ("A", "C"), ("B", "C")), ESPEC_PEQUENA)
        so_ab = bs.bootstrap_pareado(pacientes, y, {"A": probas["A"], "B": probas["B"]}, (("A", "B"),),
                                     ESPEC_PEQUENA)
        self.assertEqual(tudo.por_modelo["auc"]["A"], so_ab.por_modelo["auc"]["A"])
        self.assertEqual(tudo.contrastes["auc"]["A-B"], so_ab.contrastes["auc"]["A-B"])
        self.assertEqual(tudo.n_descartadas, so_ab.n_descartadas)

    def test_ordem_dos_nodulos_nao_altera_o_resultado(self):
        pacientes, y, probas = dados_sinteticos()
        ordem = np.random.default_rng(1).permutation(len(y))
        a = bs.bootstrap_pareado(pacientes, y, probas, (("A", "B"),), ESPEC_PEQUENA)
        b = bs.bootstrap_pareado(pacientes[ordem], y[ordem], {m: p[ordem] for m, p in probas.items()},
                                 (("A", "B"),), ESPEC_PEQUENA)
        self.assertEqual(a.contrastes["auc"]["A-B"].limite_inferior, b.contrastes["auc"]["A-B"].limite_inferior)
        self.assertEqual(a.contrastes["auc"]["A-B"].limite_superior, b.contrastes["auc"]["A-B"].limite_superior)

    def test_deterministico_e_sensivel_a_seed(self):
        pacientes, y, probas = dados_sinteticos()
        a = bs.bootstrap_pareado(pacientes, y, probas, (("A", "B"),), ESPEC_PEQUENA)
        b = bs.bootstrap_pareado(pacientes, y, probas, (("A", "B"),), ESPEC_PEQUENA)
        outra = bs.bootstrap_pareado(pacientes, y, probas, (("A", "B"),),
                                     dataclasses.replace(ESPEC_PEQUENA, seed=43))
        self.assertEqual(a.como_dict(), b.como_dict())
        self.assertNotEqual(a.contrastes["auc"]["A-B"], outra.contrastes["auc"]["A-B"])

    def test_paciente_entra_com_todos_os_seus_nodulos(self):
        # Um único paciente com todos os positivos: cada réplica ou o inclui inteiro ou não o inclui.
        pacientes = np.array(["X", "X", "X", "Y", "Z", "W"])
        y = np.array([1, 1, 1, 0, 0, 0])
        proba = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        res = bs.bootstrap_pareado(pacientes, y, {"A": proba, "B": 1 - proba}, (("A", "B"),), ESPEC_PEQUENA)
        auc, delta, descartadas = referencia(pacientes, y, {"A": proba, "B": 1 - proba}, (("A", "B"),),
                                             ESPEC_PEQUENA)
        self.assertEqual(res.n_descartadas, descartadas)
        self.assertGreater(descartadas, 0)  # réplicas sem X (sem positivos) são descartadas
        self.assertAlmostEqual(res.por_modelo["auc"]["A"].limite_inferior, np.percentile(auc["A"], 2.5), places=12)


class TestDegeneradas(unittest.TestCase):
    def test_replicas_com_uma_classe_sao_descartadas_e_contadas(self):
        pacientes = np.array(["A", "B", "C", "D", "E"])
        y = np.array([1, 0, 0, 0, 0])  # um único paciente positivo
        proba = np.array([0.9, 0.2, 0.3, 0.4, 0.1])
        res = bs.bootstrap_pareado(pacientes, y, {"A": proba, "B": 1 - proba}, (("A", "B"),), ESPEC_PEQUENA)
        _, _, descartadas = referencia(pacientes, y, {"A": proba, "B": 1 - proba}, (("A", "B"),), ESPEC_PEQUENA)
        self.assertEqual((res.n_descartadas, res.n_validas), (descartadas, 200 - descartadas))
        self.assertGreater(res.n_descartadas, 0)
        self.assertEqual(res.n_validas + res.n_descartadas, res.n_replicas)  # nenhuma réplica sorteada de novo


class TestEntradas(unittest.TestCase):
    def setUp(self):
        self.p, self.y, self.probas = dados_sinteticos(n_pacientes=10)

    def erro(self, **mudancas):
        argumentos = {"pacientes": self.p, "y": self.y, "probas": self.probas, "contrastes": (("A", "B"),)}
        argumentos.update(mudancas)
        with self.assertRaises(bs.BootstrapInvalido):
            bs.bootstrap_pareado(argumentos["pacientes"], argumentos["y"], argumentos["probas"],
                                 argumentos["contrastes"], ESPEC_PEQUENA)

    def test_recusas(self):
        self.erro(y=self.y[:-1])
        self.erro(y=np.zeros_like(self.y))
        self.erro(y=self.y.astype(float))
        self.erro(probas={})
        self.erro(probas={**self.probas, "A": self.probas["A"][:-1]})
        self.erro(probas={**self.probas, "A": self.probas["A"] * 2.0})
        self.erro(probas={**self.probas, "A": np.where(np.arange(len(self.y)) == 0, np.nan, self.probas["A"])})
        self.erro(contrastes=(("A", "Z"),))
        self.erro(contrastes=(("A", "A"),))
        self.erro(pacientes=self.p.reshape(-1, 1))

    def test_espec_diferente_da_secao_8_2_e_recusada(self):
        for mudanca in ({"unidade": "nodulo_id"}, {"pareado": False}, {"retreino": True},
                        {"descartar_replicas_com_uma_classe": False}):
            with self.assertRaises(bs.BootstrapInvalido):
                bs.bootstrap_pareado(self.p, self.y, self.probas, (("A", "B"),),
                                     dataclasses.replace(ESPEC_PEQUENA, **mudanca))


class TestContratoDoModulo(unittest.TestCase):
    def test_sem_p_valor_e_json_serializavel(self):
        res = bs.bootstrap_pareado(*dados_sinteticos(n_pacientes=15), (("A", "B"),), ESPEC_PEQUENA)
        texto = json.dumps(res.como_dict(), sort_keys=True, allow_nan=False)
        self.assertNotIn("p_valor", texto)
        self.assertNotIn("p-valor", texto)
        self.assertFalse(cfg.CALCULA_P_VALOR)

    def test_nao_importa_o_bootstrap_nem_qualquer_modulo_da_s6(self):
        arvore = ast.parse(inspect.getsource(bs))
        importados = set()
        for no in ast.walk(arvore):
            if isinstance(no, ast.Import):
                importados.update(a.name for a in no.names)
            elif isinstance(no, ast.ImportFrom):
                importados.add(("." * no.level) + (no.module or ""))
        self.assertFalse([m for m in importados if "benchmark_s6" in m])
        self.assertEqual({m for m in importados if m.startswith(".")}, {"."})  # só ``from . import config``
        self.assertNotIn("open", {n.func.id for n in ast.walk(arvore)
                                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
