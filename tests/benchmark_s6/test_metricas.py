"""Testes de ``src/benchmark_s6/metricas.py`` (P9, Seções 6, 7 e 7.4). Só dados sintéticos; nada é treinado."""

from __future__ import annotations

import ast
import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss as sk_log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sinteticos as sn  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import metricas as mt  # noqa: E402
from src.benchmark_s6 import unidades as un  # noqa: E402

MODULO = Path(mt.__file__)
C = cfg.CONFIGURACAO


def auc_por_pares(y, s) -> float:
    """Referência independente: fração de pares (positivo, negativo) ordenados corretamente; empate vale 0,5."""
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


class BaseMetricas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.carga = sn.carga_definitiva()
        cls.ctx = un.ContextoExecucao.de_dados(cls.carga)
        cls.base = mt.BaseAvaliacao.de_contexto(cls.ctx)
        cls.resultados = {(r.unidade.modelo, r.unidade.repeticao, r.unidade.dobra): r
                          for r in sn.resultados_sinteticos(cls.ctx)}


class TestAuc(BaseMetricas):
    def test_auc_igual_a_referencia_por_pares_com_e_sem_empates(self):
        rng = np.random.default_rng(1)
        for arredondar in (None, 1):  # arredondar força muitos empates
            y = (rng.random(60) < 0.5).astype(int)
            s = rng.normal(size=60) + y
            s = s if arredondar is None else np.round(s, arredondar)
            self.assertAlmostEqual(mt.auc_roc(y, s), auc_por_pares(y, s), places=12)

    def test_auc_exige_as_duas_classes(self):
        with self.assertRaises(mt.MetricasInvalidas):
            mt.auc_roc(np.ones(10, dtype=int), np.arange(10.0))

    def test_validade_nas_fronteiras(self):
        self.assertFalse(mt.valida(4, 50))
        self.assertFalse(mt.valida(50, 4))
        self.assertTrue(mt.valida(5, 5))
        self.assertTrue(mt.valida(C.validade.min_positivos, C.validade.min_negativos))

    def test_auc_da_dobra_g_usa_todas_as_linhas(self):
        res = self.resultados[("lr_tuning", 0, 0)]
        d, auc = mt.auc_da_dobra(self.base, res, "G")
        y, s = np.asarray(res.privado.y), np.asarray(res.privado.pontuacoes)
        self.assertEqual((d.nodulos, d.positivos, d.negativos), (len(y), int(y.sum()), int((1 - y).sum())))
        self.assertTrue(d.valida)
        self.assertAlmostEqual(auc, auc_por_pares(y, s), places=12)

    def test_auc_da_dobra_f_recorta_so_a_avaliacao(self):
        res = self.resultados[("rf", 1, 2)]
        d, auc = mt.auc_da_dobra(self.base, res, "F")
        posicoes = np.asarray(res.privado.posicoes)
        dentro = (self.carga.X()[posicoes, 1] >= 12.0) & (self.carga.X()[posicoes, 1] < 30.0)
        y, s = np.asarray(res.privado.y)[dentro], np.asarray(res.privado.pontuacoes)[dentro]
        self.assertEqual(d.nodulos, int(dentro.sum()))
        self.assertEqual(d.pacientes, len(set(self.base.paciente[posicoes[dentro]].tolist())))
        self.assertAlmostEqual(auc, auc_por_pares(y, s), places=12)

    def test_dobra_degenerada_nao_tem_auc_imputada(self):
        sem_faixa = dataclasses.replace(self.base, na_faixa=np.zeros(self.base.amostra.n, dtype=bool))
        d, auc = mt.auc_da_dobra(sem_faixa, self.resultados[("lr_fixa", 0, 0)], "F")
        self.assertEqual((d.nodulos, d.valida, auc), (0, False, None))
        quase = np.zeros(self.base.amostra.n, dtype=bool)
        res = self.resultados[("lr_fixa", 0, 0)]
        y = np.asarray(res.privado.y)
        posicoes = np.asarray(res.privado.posicoes)
        quase[posicoes[y == 1][:4]] = True  # 4 positivos
        quase[posicoes[y == 0][:30]] = True
        d, auc = mt.auc_da_dobra(dataclasses.replace(self.base, na_faixa=quase), res, "F")
        self.assertEqual((d.positivos, d.negativos, d.valida, auc), (4, 30, False, None))
        quase[posicoes[y == 1][4]] = True  # 5º positivo: passa a valer
        d, auc = mt.auc_da_dobra(dataclasses.replace(self.base, na_faixa=quase), res, "F")
        self.assertTrue(d.valida)
        self.assertIsNotNone(auc)

    def test_denominador_conta_nodulos_distintos(self):
        res = self.resultados[("lr_fixa", 0, 0)]
        posicoes = np.asarray(res.privado.posicoes)
        d = mt.denominador(self.base, 0, 0, posicoes, "G")
        self.assertEqual(d.nodulos, len(set(res.privado.nodulo_ids)))
        self.assertEqual(d.pacientes, len(set(self.base.amostra.grupos[posicoes].tolist())))


class TestBase(BaseMetricas):
    def test_faixa_semiaberta_12_inclusivo_30_exclusivo(self):
        a = self.base.amostra
        X = np.column_stack([np.ones(a.n), np.full(a.n, 20.0), np.ones(a.n)])
        for valor, esperado in ((11.999999, False), (12.0, True), (29.999999, True), (30.0, False)):
            X[0, 1] = valor
            base = mt.BaseAvaliacao._criar("definitivo", a, X)
            self.assertEqual(bool(base.na_faixa[0]), esperado, valor)

    def test_modo_implementacao_nao_calcula_metricas_externas(self):
        ctx = un.ContextoExecucao.de_dados(sn.carga_definitiva(modo="implementacao"))
        with self.assertRaisesRegex(mt.MetricasInvalidas, "só existem no modo definitivo"):
            mt.BaseAvaliacao.de_contexto(ctx)
        self.assertEqual(mt.MetricasInvalidas.chave_saida, "integridade")

    def test_contagem_da_faixa_confere_com_a_base(self):
        c = self.base.contagem_faixa()
        dentro = (self.carga.X()[:, 1] >= 12.0) & (self.carga.X()[:, 1] < 30.0)
        self.assertEqual(c["nodulos"], int(dentro.sum()))
        self.assertEqual(c["classe_0"] + c["classe_1"], c["nodulos"])
        self.assertEqual(c["pacientes"], len(set(self.carga.grupos()[dentro].tolist())))

    def test_entradas_incoerentes(self):
        res = self.resultados[("lr_fixa", 0, 0)]
        privado = res.privado
        casos = {
            "rotulo": dataclasses.replace(privado, y=1 - np.asarray(privado.y)),
            "nodulo_id": dataclasses.replace(privado, nodulo_ids=privado.nodulo_ids[::-1]),
            "posicao": dataclasses.replace(privado, posicoes=np.asarray(privado.posicoes) + 1),
            "nao_finita": dataclasses.replace(privado, pontuacoes=np.full(len(privado.y), np.nan)),
        }
        for nome, p in casos.items():
            with self.subTest(caso=nome), self.assertRaises(mt.MetricasInvalidas):
                mt.auc_da_dobra(self.base, un.ResultadoUnidade(res.registro, p), "G")
        with self.assertRaises(TypeError):
            mt.auc_da_dobra(self.base, "resultado", "G")  # type: ignore[arg-type]
        with self.assertRaises(mt.MetricasInvalidas):
            mt.auc_da_dobra(self.base, res, "H")


class TestDescritivas(BaseMetricas):
    def test_brier_e_log_loss_contra_sklearn(self):
        res = self.resultados[("xgb", 0, 1)]
        y, p = np.asarray(res.privado.y), np.asarray(res.privado.pontuacoes)
        self.assertAlmostEqual(mt.brier(y, p), brier_score_loss(y, p), places=12)
        self.assertAlmostEqual(mt.log_loss(y, p, C.metricas.corte_log_loss), sk_log_loss(y, p), places=10)

    def test_log_loss_corta_a_probabilidade(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.0, 1.0, 1.0, 0.0])  # os dois primeiros seriam infinitos sem o corte
        corte = C.metricas.corte_log_loss
        valor = mt.log_loss(y, p, corte)
        self.assertTrue(np.isfinite(valor))
        # dois erros máximos (p cortada em 1e-15 ou 1 − 1e-15) e dois acertos; mesma definição do sklearn
        self.assertAlmostEqual(valor, sk_log_loss(y, p, eps=corte, labels=[0, 1]), places=10)
        self.assertGreater(valor, 17.0)

    def test_calibracao_contra_regressao_logistica_sem_penalizacao(self):
        for nome in ("lr_tuning", "rf"):
            res = self.resultados[(nome, 2, 3)]
            y, p = np.asarray(res.privado.y), np.asarray(res.privado.pontuacoes)
            intercepto, inclinacao = mt.calibracao(y, p, C.metricas.corte_log_loss)
            z = np.log(p) - np.log1p(-p)
            ref = LogisticRegression(penalty=None, tol=1e-12, max_iter=10_000).fit(z.reshape(-1, 1), y)
            self.assertAlmostEqual(intercepto, float(ref.intercept_[0]), places=5)
            self.assertAlmostEqual(inclinacao, float(ref.coef_[0, 0]), places=5)

    def test_calibracao_indefinida_nunca_e_imputada(self):
        corte = C.metricas.corte_log_loss
        y = np.array([0, 0, 0, 1, 1, 1])
        self.assertEqual(mt.calibracao(y, np.array([.1, .2, .3, .7, .8, .9]), corte), (None, None))  # separação
        self.assertEqual(mt.calibracao(y, np.full(6, 0.5), corte), (None, None))  # logit constante
        self.assertEqual(mt.calibracao(np.ones(6, dtype=int), np.linspace(.1, .9, 6), corte), (None, None))

    def test_sensibilidade_e_especificidade(self):
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        previsto = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
        self.assertEqual(mt.sensibilidade_especificidade(y, previsto), (0.75, 0.75))

    def test_probabilisticos_so_para_quem_tem_probabilidade(self):
        for modelo in ("lr_fixa", "lr_tuning", "rf", "xgb"):
            linhas = mt.descritivas_da_dobra(self.base, self.resultados[(modelo, 0, 0)])
            self.assertEqual([d.metrica for d in linhas], list(C.metricas.descritivas_probabilisticas))
            self.assertTrue(all(d.rotulo == "" for d in linhas))
            res = self.resultados[(modelo, 0, 0)]
            y, p = np.asarray(res.privado.y), np.asarray(res.privado.pontuacoes)
            por_nome = {d.metrica: d.valor for d in linhas}
            self.assertEqual(por_nome["sensibilidade_p_0_5"], float(np.mean(p[y == 1] >= 0.5)))
            self.assertEqual(por_nome["especificidade_p_0_5"], float(np.mean(p[y == 0] < 0.5)))

    def test_svm_so_regra_nativa_rotulada_e_sem_metricas_de_probabilidade(self):
        res = self.resultados[("svm", 0, 0)]
        linhas = mt.descritivas_da_dobra(self.base, res)
        self.assertEqual([d.metrica for d in linhas], list(C.metricas.descritivas_regra_nativa))
        self.assertTrue(all(d.rotulo == "regra nativa, não comparável" for d in linhas))
        proibidas = set(C.metricas.descritivas_probabilisticas)
        self.assertFalse(proibidas & {d.metrica for d in linhas})
        y, s = np.asarray(res.privado.y), np.asarray(res.privado.pontuacoes)
        por_nome = {d.metrica: d.valor for d in linhas}
        self.assertEqual(por_nome["sensibilidade_regra_nativa"], float(np.mean(s[y == 1] >= 0.0)))
        self.assertEqual(por_nome["especificidade_regra_nativa"], float(np.mean(s[y == 0] < 0.0)))

    def test_descritivas_sao_so_do_desfecho_g(self):
        self.assertEqual(C.metricas.desfecho_das_descritivas, "G")
        arvore = ast.parse(MODULO.read_text(encoding="utf-8"))
        funcao = next(n for n in ast.walk(arvore)
                      if isinstance(n, ast.FunctionDef) and n.name == "descritivas_da_dobra")
        usados = {n.id for n in ast.walk(funcao) if isinstance(n, ast.Name)}
        self.assertNotIn("na_faixa", usados)
        self.assertNotIn("linhas_do_desfecho", usados)

    def test_descritivas_recusam_rotulos_que_nao_conferem_com_a_amostra(self):
        res = self.resultados[("lr_fixa", 0, 0)]
        p = dataclasses.replace(res.privado, y=np.zeros(len(res.privado.y), dtype=np.int64))
        with self.assertRaises(mt.MetricasInvalidas):  # rótulos que não conferem com a amostra
            mt.descritivas_da_dobra(self.base, un.ResultadoUnidade(res.registro, p))


class TestModulo(unittest.TestCase):
    def test_nao_le_dados_nao_grava_e_nao_treina(self):
        arvore = ast.parse(MODULO.read_text(encoding="utf-8"))
        modulos = {a.name for n in ast.walk(arvore) if isinstance(n, ast.Import) for a in n.names} | \
                  {n.module or "" for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom)}
        for proibido in ("pickle", "multiprocessing", "subprocess", "pandas", "sklearn.model_selection",
                         "src.benchmark_s6.dados", "sklearn.linear_model"):
            self.assertNotIn(proibido, modulos)
        nomes = {n.id for n in ast.walk(arvore) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(arvore) if isinstance(n, ast.Attribute)}
        for proibido in ("open", "write_text", "write_bytes", "fit", "predict_proba", "carregar_definitivo"):
            self.assertNotIn(proibido, nomes)

    def test_exportacoes(self):
        for nome in mt.__all__:
            self.assertTrue(hasattr(mt, nome), nome)


if __name__ == "__main__":
    unittest.main()
