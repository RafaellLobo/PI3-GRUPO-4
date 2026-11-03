"""Testes de ``src/benchmark_s6/bootstrap.py`` (P9, Seções 8.2, 9 e 6). Só dados sintéticos; nada é treinado.

A referência é lenta e independente: para cada réplica, pesos por paciente, ``roc_auc_score`` com
``sample_weight`` (e, numa réplica, com as linhas repetidas) e as regras de validade escritas do zero.
"""

from __future__ import annotations

import ast
import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sinteticos as sn  # noqa: E402
from src.benchmark_s6 import bootstrap as bs  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import consolidacao as co  # noqa: E402
from src.benchmark_s6 import metricas as mt  # noqa: E402
from src.benchmark_s6 import unidades as un  # noqa: E402

C = cfg.CONFIGURACAO
CODIGOS = sn.CODIGOS
MODULO = Path(bs.__file__)


def theta_referencia(res, base, pesos_pac, desfecho, modelo, expandir=False):
    """θ* de UMA réplica, do zero: pesos por paciente, validade por nódulos distintos, regras de 4/5 e 8/10."""
    por_rep = []
    for r in range(10):
        aucs = []
        for k in range(5):
            x = res[(modelo, r, k)].privado
            pos, y, s = np.asarray(x.posicoes), np.asarray(x.y), np.asarray(x.pontuacoes)
            if desfecho == "F":
                dentro = base.na_faixa[pos]
                pos, y, s = pos[dentro], y[dentro], s[dentro]
            w = pesos_pac[base.paciente[pos]]
            if ((y == 1) & (w > 0)).sum() >= 5 and ((y == 0) & (w > 0)).sum() >= 5:
                if expandir:
                    aucs.append(roc_auc_score(np.repeat(y, w), np.repeat(s, w)))
                else:
                    aucs.append(roc_auc_score(y, s, sample_weight=w))
        if len(aucs) >= 4:
            por_rep.append(np.mean(aucs))
    return float(np.mean(por_rep)) if len(por_rep) >= 8 else float("nan")


class BaseBootstrap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.carga = sn.carga_definitiva()
        cls.ctx = un.ContextoExecucao.de_dados(cls.carga)
        cls.base = mt.BaseAvaliacao.de_contexto(cls.ctx)
        lista = sn.resultados_sinteticos(cls.ctx)
        cls.res = {(r.unidade.modelo, r.unidade.repeticao, r.unidade.dobra): r for r in lista}
        cls.p = co.parear(cls.base, lista)
        cls.c = co.consolidar(cls.p)

    def com_faixa(self, mascara):
        base = dataclasses.replace(self.base, na_faixa=mascara)
        p = co.parear(base, list(self.res.values()))
        return base, p, co.consolidar(p)


class TestPesos(unittest.TestCase):
    def test_pesos_sao_multiplicidades_de_pacientes_e_deterministicos(self):
        pesos = bs.pesos_pacientes(40, 25, 42)
        self.assertEqual(pesos.shape, (25, 40))
        self.assertTrue((pesos.sum(axis=1) == 40).all())  # cada réplica sorteia P pacientes
        self.assertTrue((pesos >= 0).all())
        np.testing.assert_array_equal(pesos, bs.pesos_pacientes(40, 25, 42))
        self.assertFalse(np.array_equal(pesos, bs.pesos_pacientes(40, 25, 43)))
        esperado = np.random.default_rng(42).integers(0, 40, size=(25, 40))
        np.testing.assert_array_equal(pesos[3], np.bincount(esperado[3], minlength=40))
        self.assertEqual(pesos.dtype, np.int32)

    def test_reamostra_pacientes_e_nao_nodulos(self):
        carga = sn.carga_definitiva(n_pacientes=60)
        ctx = un.ContextoExecucao.de_dados(carga)
        base = mt.BaseAvaliacao.de_contexto(ctx)
        pesos = bs.pesos_pacientes(base.n_pacientes, 5, 42)
        por_linha = pesos[:, base.paciente]  # peso de cada nódulo = peso do seu paciente
        for b in range(5):
            for p in range(base.n_pacientes):
                self.assertEqual(len(set(por_linha[b][base.paciente == p].tolist())), 1)
        # Reamostrar nódulos daria pesos diferentes para nódulos do mesmo paciente (e somaria n, não P)
        self.assertTrue((por_linha.sum(axis=1) != base.amostra.n).any())

    def test_configuracao_e_semente_do_bootstrap(self):
        self.assertEqual((C.bootstrap.n_replicas, C.bootstrap.semente), (10_000, 42))
        self.assertEqual(C.bootstrap.reajuste, False)
        self.assertEqual(C.bootstrap.unidade, "patient_id")


class TestAucPonderada(unittest.TestCase):
    def test_contra_sklearn_e_contra_linhas_repetidas(self):
        rng = np.random.default_rng(3)
        y = (rng.random(50) < 0.5).astype(int)
        s = np.round(rng.normal(size=50) + y, 1)  # empates de propósito
        w = rng.integers(0, 4, size=50)
        w[0], w[1] = 1, 1
        esperado = roc_auc_score(y, s, sample_weight=w)
        self.assertAlmostEqual(bs.auc_ponderada(y, s, w), esperado, places=12)
        self.assertAlmostEqual(bs.auc_ponderada(y, s, w), roc_auc_score(np.repeat(y, w), np.repeat(s, w)), places=12)
        self.assertAlmostEqual(bs.auc_ponderada(y, s, np.ones(50)), roc_auc_score(y, s), places=12)

    def test_exige_peso_nas_duas_classes(self):
        y = np.array([1, 1, 0, 0])
        with self.assertRaises(bs.BootstrapInvalido):
            bs.auc_ponderada(y, np.arange(4.0), np.array([1, 1, 0, 0]))

    def test_matriz_de_pares_conta_empate_como_meio(self):
        c = bs.matriz_pares(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        np.testing.assert_array_equal(c, [[0.5, 0.0], [1.0, 0.0]])


class TestReplicas(BaseBootstrap):
    def test_theta_de_cada_replica_contra_referencia_lenta_em_g_e_f(self):
        B = 12
        r = bs.bootstrap(self.p, self.c, n_replicas=B)
        pesos = bs.pesos_pacientes(self.base.n_pacientes, B, 42)
        for desfecho in ("G", "F"):
            for m in CODIGOS:
                for b in range(B):
                    esperado = theta_referencia(self.res, self.base, pesos[b], desfecho, m)
                    self.assertAlmostEqual(r.theta_replicas[(m, desfecho)][b], esperado, places=12,
                                           msg=(desfecho, m, b))

    def test_referencia_com_linhas_repetidas_coincide(self):
        pesos = bs.pesos_pacientes(self.base.n_pacientes, 2, 42)
        r = bs.bootstrap(self.p, self.c, n_replicas=2)
        for desfecho in ("G", "F"):
            esperado = theta_referencia(self.res, self.base, pesos[1], desfecho, "rf", expandir=True)
            self.assertAlmostEqual(r.theta_replicas[("rf", desfecho)][1], esperado, places=12)

    def test_delta_e_a_diferenca_dos_theta_da_mesma_replica(self):
        r = bs.bootstrap(self.p, self.c, n_replicas=40)
        for (a, b, d), delta in r.delta_replicas.items():
            np.testing.assert_allclose(delta, r.theta_replicas[(a, d)] - r.theta_replicas[(b, d)], atol=1e-15)
        self.assertEqual(len(r.delta_replicas), (6 + 4 * 2))

    def test_replica_com_pesos_unitarios_reproduz_a_estimativa_observada(self):
        # Sem reamostragem (todo paciente com peso 1) o bootstrap devolve θ̂: usa o mesmo cálculo da réplica.
        pesos = np.ones((1, self.base.n_pacientes), dtype=np.int32)
        for desfecho in ("G", "F"):
            theta, definida = bs._replicas_de_desfecho(self.p, desfecho, pesos)
            self.assertTrue(definida[0])
            for m in CODIGOS:
                self.assertAlmostEqual(theta[m][0], self.c.estimativa(m, desfecho).theta, places=12)

    def test_validade_por_nodulos_distintos_na_replica(self):
        """A dobra (0, 0) tem EXATAMENTE 5 positivos na faixa; sem um deles (peso 0) ela sai da média da réplica."""
        mascara = self.base.na_faixa.copy()
        x = self.res[("lr_fixa", 0, 0)].privado
        pos, y = np.asarray(x.posicoes), np.asarray(x.y)
        positivos_f = pos[(y == 1) & mascara[pos]]
        mascara[positivos_f[5:]] = False  # sobram 5 positivos distintos na dobra (0, 0)
        base, p, c = self.com_faixa(mascara)
        d00 = next(d for d in c.denominadores if (d.repeticao, d.dobra, d.desfecho) == (0, 0, "F"))
        self.assertEqual((d00.positivos, d00.valida), (5, True))
        pesos = np.ones((3, base.n_pacientes), dtype=np.int32)
        pesos[1, base.paciente[positivos_f[0]]] = 0  # um dos 5 positivos some da réplica 1
        pesos[2, base.paciente[positivos_f[0]]] = 2  # ... e na réplica 2 aparece duas vezes (continua 5 distintos)
        theta, definida = bs._replicas_de_desfecho(p, "F", pesos)
        self.assertTrue(definida.all())
        for m in ("rf", "svm"):
            for b in range(3):
                self.assertAlmostEqual(theta[m][b], theta_referencia(self.res, base, pesos[b], "F", m), places=12)
        # A dobra (0, 0) só conta nas réplicas 0 e 2: tirá-la da média muda θ* da réplica 1 em relação à 0.
        self.assertNotAlmostEqual(theta["rf"][1], theta["rf"][0], places=6)
        cinco = base.paciente[positivos_f[:5]]  # pacientes dos 5 positivos que ficaram na faixa
        distintos = [len({int(q) for q in cinco if pesos[b][q] > 0}) for b in range(3)]
        self.assertLess(distintos[1], distintos[0])
        self.assertEqual(distintos[0], len(set(cinco.tolist())))
        self.assertEqual(distintos[2], distintos[0])

    def test_nao_ha_reajuste_nem_dependencia_do_tempo(self):
        arvore = ast.parse(MODULO.read_text(encoding="utf-8"))
        modulos = {a.name for n in ast.walk(arvore) if isinstance(n, ast.Import) for a in n.names} | \
                  {n.module or "" for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom)}
        for proibido in ("time", "datetime", "sklearn.model_selection", "sklearn.linear_model", "multiprocessing",
                         "pickle", "subprocess", "pandas"):
            self.assertNotIn(proibido, modulos)
        nomes = {n.id for n in ast.walk(arvore) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(arvore) if isinstance(n, ast.Attribute)}
        for proibido in ("fit", "predict_proba", "perf_counter", "clone", "GridSearchCV"):
            self.assertNotIn(proibido, nomes)

    def test_n_replicas_padrao_e_o_da_configuracao_e_invalidos_sao_recusados(self):
        for ruim in (1, 0, -3, 10.0, "100", True):
            with self.subTest(n=ruim), self.assertRaises(bs.BootstrapInvalido):
                bs.bootstrap(self.p, self.c, n_replicas=ruim)
        with self.assertRaises(TypeError):
            bs.bootstrap(self.res, self.c)  # type: ignore[arg-type]

    def test_determinismo(self):
        a = bs.bootstrap(self.p, self.c, n_replicas=60)
        b = bs.bootstrap(self.p, self.c, n_replicas=60)
        for chave in a.theta_replicas:
            np.testing.assert_array_equal(a.theta_replicas[chave], b.theta_replicas[chave])
        self.assertEqual(a.intervalos_contraste, b.intervalos_contraste)
        self.assertEqual(a.intervalos_theta, b.intervalos_theta)


class TestIntervalos(BaseBootstrap):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = bs.bootstrap(cls.p, cls.c, n_replicas=2000)

    def test_theta_ic95_descritivo_por_percentis_sem_correcao(self):
        self.assertEqual(self.r.percentis_descritivos, (2.5, 97.5))
        for i in self.r.intervalos_theta:
            valores = self.r.theta_replicas[(i.modelo, i.desfecho)]
            valores = valores[~np.isnan(valores)]
            self.assertEqual((i.inferior, i.superior), (float(np.percentile(valores, 2.5)),
                                                        float(np.percentile(valores, 97.5))))
            self.assertTrue(i.relatado)
            self.assertLess(i.inferior, i.theta)
            self.assertLess(i.theta, i.superior)

    def test_contrastes_da_familia_tem_bonferroni_e_ic95_e_os_descritivos_so_ic95(self):
        inf, sup = self.r.percentis_familia
        self.assertAlmostEqual(inf, 0.41667, places=4)
        self.assertAlmostEqual(sup, 99.58333, places=4)
        self.assertAlmostEqual(inf, 100 * 0.05 / 12, places=10)
        for i in self.r.intervalos_contraste:
            validos = self.r.delta_replicas[(i.a, i.b, i.desfecho)]
            validos = validos[~np.isnan(validos)]
            self.assertEqual((i.ic95_inferior, i.ic95_superior), (float(np.percentile(validos, 2.5)),
                                                                  float(np.percentile(validos, 97.5))))
            if i.grupo == co.GRUPO_FAMILIA:
                self.assertEqual((i.bonferroni_inferior, i.bonferroni_superior),
                                 (float(np.percentile(validos, inf)), float(np.percentile(validos, sup))))
                self.assertLessEqual(i.bonferroni_inferior, i.ic95_inferior)  # Bonferroni é mais largo
                self.assertGreaterEqual(i.bonferroni_superior, i.ic95_superior)
            else:
                self.assertEqual((i.bonferroni_inferior, i.bonferroni_superior), (None, None))
        self.assertEqual(sum(1 for i in self.r.intervalos_contraste if i.grupo == co.GRUPO_FAMILIA), 6)

    def test_bonferroni_so_nos_seis_contrastes_e_nunca_em_auc_individual(self):
        self.assertFalse(hasattr(self.r.intervalos_theta[0], "bonferroni_inferior"))
        self.assertEqual(co.pares_de_contraste()[:3], (("rf", "lr_tuning", co.GRUPO_FAMILIA),
                                                       ("xgb", "lr_tuning", co.GRUPO_FAMILIA),
                                                       ("svm", "lr_tuning", co.GRUPO_FAMILIA)))

    def test_sem_descartes_nos_dados_sintéticos_saudaveis(self):
        for d in self.r.descartes:
            self.assertEqual((d.n_replicas, d.descartadas, d.fracao, d.iq), (2000, 0, 0.0, False))

    def test_intervalo_cobre_a_estimativa_e_tem_largura_plausivel(self):
        for i in self.r.intervalos_theta:
            self.assertTrue(0.0 < i.superior - i.inferior < 0.05)
        centro = np.nanmean(self.r.theta_replicas[("lr_tuning", "G")])
        self.assertAlmostEqual(centro, self.c.estimativa("lr_tuning", "G").theta, delta=0.003)


class TestDescarteEIQ(BaseBootstrap):
    def mascara_fragil(self):
        """Faixa esparsa: observada estimável, mas muitas réplicas deixam o desfecho F indefinido."""
        return np.random.default_rng(11).random(self.base.amostra.n) < 0.2

    def test_iq_quando_mais_de_1_por_cento_das_replicas_sao_descartadas(self):
        base, p, c = self.com_faixa(self.mascara_fragil())
        f = next(e for e in c.estados if e.desfecho == "F")
        self.assertFalse(f.nao_estimavel)  # observada: estimável
        r = bs.bootstrap(p, c, n_replicas=400)
        d = r.descarte("F")
        self.assertGreater(d.fracao, C.validade.max_fracao_descarte_bootstrap)
        self.assertTrue(d.iq)
        self.assertEqual(d.fracao, d.descartadas / 400)
        for i in (x for x in r.intervalos_theta if x.desfecho == "F"):
            self.assertEqual((i.relatado, i.motivo, i.inferior, i.superior), (False, "IQ", None, None))
            self.assertIsNotNone(i.theta)  # a estimativa pontual continua relatada
        for i in (x for x in r.intervalos_contraste if x.desfecho == "F"):
            self.assertEqual((i.relatado, i.motivo, i.ic95_inferior, i.bonferroni_inferior), (False, "IQ", None, None))
            self.assertIsNotNone(i.delta)
        self.assertFalse(r.descarte("G").iq)  # o outro desfecho não é afetado
        self.assertTrue(all(x.relatado for x in r.intervalos_theta if x.desfecho == "G"))
        descartada = np.isnan(r.theta_replicas[("rf", "F")])
        self.assertEqual(int(descartada.sum()), d.descartadas)
        for m in CODIGOS:  # descarte é por desfecho: igual para todos os modelos
            np.testing.assert_array_equal(np.isnan(r.theta_replicas[(m, "F")]), descartada)

    def test_descarte_contra_referencia_lenta(self):
        base, p, c = self.com_faixa(self.mascara_fragil())
        B = 30
        r = bs.bootstrap(p, c, n_replicas=B)
        pesos = bs.pesos_pacientes(base.n_pacientes, B, 42)
        esperado = [np.isnan(theta_referencia(self.res, base, pesos[b], "F", "lr_fixa")) for b in range(B)]
        self.assertEqual(list(np.isnan(r.theta_replicas[("lr_fixa", "F")])), esperado)
        self.assertEqual(r.descarte("F").descartadas, sum(esperado))

    def test_fronteira_do_descarte_e_estritamente_maior_que_1_por_cento(self):
        # 1% de 400 = 4 descartes: 4 não é IQ; 5 é.
        teto = C.validade.max_fracao_descarte_bootstrap
        self.assertFalse(4 / 400 > teto)
        self.assertTrue(5 / 400 > teto)
        self.assertEqual(teto, 0.01)

    def test_desfecho_nao_estimavel_nao_tem_ic(self):
        base, p, c = self.com_faixa(np.zeros(self.base.amostra.n, dtype=bool))
        r = bs.bootstrap(p, c, n_replicas=50)
        for i in (x for x in r.intervalos_theta if x.desfecho == "F"):
            self.assertEqual((i.relatado, i.motivo, i.theta, i.inferior), (False, "NE", None, None))
        for i in (x for x in r.intervalos_contraste if x.desfecho == "F"):
            self.assertEqual((i.relatado, i.motivo, i.delta), (False, "NE", None))
        self.assertTrue(r.descarte("F").iq)  # todas as réplicas descartadas
        self.assertEqual(r.descarte("F").fracao, 1.0)
        self.assertTrue(all(x.relatado for x in r.intervalos_theta if x.desfecho == "G"))


class TestPrivacidade(BaseBootstrap):
    def test_resultado_sem_identificadores(self):
        r = bs.bootstrap(self.p, self.c, n_replicas=30)
        texto = repr(r.intervalos_theta) + repr(r.intervalos_contraste) + repr(r.descartes)
        for i in self.base.amostra.ids[::25]:
            self.assertNotIn(str(i), texto)
        for g in set(self.base.amostra.grupos.tolist()):
            self.assertNotIn(g, texto)
        self.assertEqual(bs.BootstrapInvalido.chave_saida, "integridade")


if __name__ == "__main__":
    unittest.main()
