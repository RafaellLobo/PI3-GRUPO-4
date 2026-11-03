"""Testes de ``src/benchmark_s6/consolidacao.py`` (P9, Seções 6, 8.1 e 9). Só dados sintéticos; nada é treinado.

As referências são escritas do zero, a partir das predições brutas, sem chamar o código testado.
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sinteticos as sn  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import consolidacao as co  # noqa: E402
from src.benchmark_s6 import metricas as mt  # noqa: E402
from src.benchmark_s6 import unidades as un  # noqa: E402

C = cfg.CONFIGURACAO
CODIGOS = sn.CODIGOS


def auc_por_pares(y, s) -> float:
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def referencia(res: Dict, mascara: Optional[np.ndarray], modelo: str):
    """``(θ̂, médias por repetição, dobras degeneradas, repetições inválidas)`` calculados do zero."""
    por_rep: Dict[int, float] = {}
    dd = 0
    for r in range(10):
        aucs: List[float] = []
        for k in range(5):
            x = res[(modelo, r, k)].privado
            posicoes, y, s = np.asarray(x.posicoes), np.asarray(x.y), np.asarray(x.pontuacoes)
            if mascara is not None:
                y, s = y[mascara[posicoes]], s[mascara[posicoes]]
            if (y == 1).sum() >= 5 and (y == 0).sum() >= 5:
                aucs.append(auc_por_pares(y, s))
            else:
                dd += 1
        if len(aucs) >= 4:
            por_rep[r] = float(np.mean(aucs))
    theta = float(np.mean(list(por_rep.values()))) if len(por_rep) >= 8 else None
    return theta, por_rep, dd, 10 - len(por_rep)


class BaseConsolidacao(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.carga = sn.carga_definitiva()
        cls.ctx = un.ContextoExecucao.de_dados(cls.carga)
        cls.base = mt.BaseAvaliacao.de_contexto(cls.ctx)
        cls.lista = sn.resultados_sinteticos(cls.ctx)
        cls.res = {(r.unidade.modelo, r.unidade.repeticao, r.unidade.dobra): r for r in cls.lista}
        cls.cons = co.consolidar(co.parear(cls.base, cls.lista))
        cls.mascara_f = cls.base.na_faixa


class TestPareamento(BaseConsolidacao):
    def test_exige_as_250_unidades_e_recusa_falhas_como_resultado(self):
        self.assertEqual(len(self.lista), 250)
        with self.assertRaisesRegex(co.ConsolidacaoInvalida, "ausente"):
            co.parear(self.base, self.lista[1:])
        with self.assertRaisesRegex(co.ConsolidacaoInvalida, "repetida"):
            co.parear(self.base, self.lista + self.lista[:1])
        falha = un.FalhaRegistrada(un.Unidade(0, 0, "svm"), "falha_tecnica", "X", "m")
        with self.assertRaises(TypeError):
            co.parear(self.base, self.lista[:5] + [falha])  # type: ignore[list-item]
        self.assertEqual(co.ConsolidacaoInvalida.chave_saida, "integridade")

    def test_recusa_modelos_que_nao_avaliam_as_mesmas_linhas(self):
        alterada = self.res[("rf", 3, 2)]
        outra = self.res[("rf", 3, 1)].privado  # linhas de outra dobra
        trocada = un.ResultadoUnidade(alterada.registro, dataclasses.replace(outra, unidade=alterada.unidade))
        lista = [trocada if r is alterada else r for r in self.lista]
        with self.assertRaises((co.ConsolidacaoInvalida, mt.MetricasInvalidas)):
            co.parear(self.base, lista)

    def test_recusa_configuracao_ou_modo_diferentes(self):
        r = self.lista[7]
        for campo, valor in (("sha256_configuracao", "0" * 64), ("modo", "implementacao")):
            lista = [un.ResultadoUnidade(dataclasses.replace(r.registro, **{campo: valor}), r.privado)
                     if x is r else x for x in self.lista]
            with self.subTest(campo=campo), self.assertRaisesRegex(co.ConsolidacaoInvalida, "modo ou configuração"):
                co.parear(self.base, lista)

    def test_base_do_implementacao_e_recusada(self):
        with self.assertRaises(mt.MetricasInvalidas):
            mt.BaseAvaliacao.de_contexto(un.ContextoExecucao.de_dados(sn.carga_definitiva(modo="implementacao")))

    def test_consolidar_exige_pareados(self):
        with self.assertRaises(TypeError):
            co.consolidar(self.lista)  # type: ignore[arg-type]


class TestEstimativas(BaseConsolidacao):
    def test_theta_contra_referencia_independente_em_g_e_f(self):
        for desfecho, mascara in (("G", None), ("F", self.mascara_f)):
            for m in CODIGOS:
                with self.subTest(modelo=m, desfecho=desfecho):
                    theta, por_rep, _, _ = referencia(self.res, mascara, m)
                    e = self.cons.estimativa(m, desfecho)
                    self.assertTrue(e.estimavel)
                    self.assertAlmostEqual(e.theta, theta, places=12)
                    self.assertEqual(e.n_repeticoes_validas, len(por_rep))
                    self.assertAlmostEqual(e.minimo_repeticoes, min(por_rep.values()), places=12)
                    self.assertAlmostEqual(e.maximo_repeticoes, max(por_rep.values()), places=12)

    def test_auc_por_dobra_e_por_repeticao_batem_com_a_referencia(self):
        for a in self.cons.auc_por_dobra[::37]:
            x = self.res[(a.modelo, a.repeticao, a.dobra)].privado
            y, s, pos = np.asarray(x.y), np.asarray(x.pontuacoes), np.asarray(x.posicoes)
            if a.desfecho == "F":
                y, s = y[self.mascara_f[pos]], s[self.mascara_f[pos]]
            self.assertAlmostEqual(a.auc, auc_por_pares(y, s), places=12)
        self.assertEqual(len(self.cons.auc_por_dobra), 5 * 50 * 2)
        self.assertEqual(len(self.cons.auc_por_repeticao), 5 * 10 * 2)
        for a in self.cons.auc_por_repeticao:
            aucs = [d.auc for d in self.cons.auc_por_dobra
                    if (d.modelo, d.repeticao, d.desfecho) == (a.modelo, a.repeticao, a.desfecho)]
            self.assertAlmostEqual(a.auc, float(np.mean(aucs)), places=12)
            self.assertEqual(a.n_dobras_validas, 5)

    def test_contrastes_sao_pareados_e_iguais_a_diferenca_de_thetas(self):
        familia = [c for c in self.cons.contrastes if c.grupo == co.GRUPO_FAMILIA]
        self.assertEqual(len(familia), 6)
        self.assertEqual({(c.a, c.b, c.desfecho) for c in familia},
                         {(a, "lr_tuning", d) for a in ("rf", "xgb", "svm") for d in ("G", "F")})
        for c in self.cons.contrastes:
            with self.subTest(contraste=(c.a, c.b, c.desfecho)):
                dif = self.cons.estimativa(c.a, c.desfecho).theta - self.cons.estimativa(c.b, c.desfecho).theta
                self.assertAlmostEqual(c.delta, dif, places=12)
                mascara = self.mascara_f if c.desfecho == "F" else None
                _, ra, _, _ = referencia(self.res, mascara, c.a)
                _, rb, _, _ = referencia(self.res, mascara, c.b)
                deltas = [ra[r] - rb[r] for r in range(10)]
                for obtido, esperado in zip(c.delta_por_repeticao, deltas):
                    self.assertAlmostEqual(obtido, esperado, places=12)
                self.assertAlmostEqual(c.delta, float(np.mean(deltas)), places=12)
                self.assertAlmostEqual(c.minimo, min(deltas), places=12)
                self.assertAlmostEqual(c.maximo, max(deltas), places=12)
                self.assertEqual(c.n_repeticoes_positivas, sum(1 for d in deltas if d > 0))
                self.assertAlmostEqual(c.fracao_repeticoes_positivas, c.n_repeticoes_positivas / 10)

    def test_contrastes_descritivos_vem_da_configuracao(self):
        descritivos = {(c.a, c.b) for c in self.cons.contrastes if c.grupo == co.GRUPO_DESCRITIVO}
        self.assertEqual(descritivos, set(C.contrastes.descritivos))
        self.assertEqual(len(self.cons.contrastes), (6 + 4 * 2))

    def test_deltas_por_dobra(self):
        d = next(x for x in self.cons.deltas_por_dobra if (x.a, x.b, x.desfecho, x.repeticao, x.dobra)
                 == ("rf", "lr_tuning", "G", 4, 1))
        a = next(x for x in self.cons.auc_por_dobra
                 if (x.modelo, x.repeticao, x.dobra, x.desfecho) == ("rf", 4, 1, "G"))
        b = next(x for x in self.cons.auc_por_dobra
                 if (x.modelo, x.repeticao, x.dobra, x.desfecho) == ("lr_tuning", 4, 1, "G"))
        self.assertAlmostEqual(d.delta, a.auc - b.auc, places=15)
        self.assertEqual(len([x for x in self.cons.deltas_por_dobra if x.desfecho == "G"
                              and (x.a, x.b) == ("rf", "lr_tuning")]), 50)  # até 50 por contraste e desfecho

    def test_modelo_igual_ao_comparador_tem_delta_zero(self):
        base = sn.pontuador_padrao()

        def pontuador(m, r, k, y, pos, rng):
            return base("lr_tuning", r, k, y, pos, np.random.default_rng([0, r, k, 1]))  # mesmo ruído do comparador

        cons = co.consolidar(co.parear(self.base, sn.resultados_sinteticos(self.ctx, lambda m, *a: pontuador(m, *a))))
        for c in cons.contrastes:
            self.assertAlmostEqual(c.delta, 0.0, places=15)
            self.assertEqual(c.n_repeticoes_positivas, 0)


class TestEstados(BaseConsolidacao):
    def consolidar_com_faixa(self, mascara: np.ndarray):
        base = dataclasses.replace(self.base, na_faixa=mascara)
        return co.consolidar(co.parear(base, self.lista)), base

    def test_sem_nenhuma_linha_na_faixa_o_desfecho_f_e_nao_estimavel_e_nada_e_imputado(self):
        cons, _ = self.consolidar_com_faixa(np.zeros(self.base.amostra.n, dtype=bool))
        f = next(e for e in cons.estados if e.desfecho == "F")
        self.assertEqual((f.dobras_degeneradas, f.repeticoes_invalidas, f.repeticoes_validas, f.nao_estimavel),
                         (50, 10, 0, True))
        g = next(e for e in cons.estados if e.desfecho == "G")
        self.assertEqual((g.dobras_degeneradas, g.repeticoes_invalidas, g.nao_estimavel), (0, 0, False))
        for m in CODIGOS:
            e = cons.estimativa(m, "F")
            self.assertEqual((e.estimavel, e.theta, e.minimo_repeticoes, e.maximo_repeticoes),
                             (False, None, None, None))
            self.assertTrue(cons.estimativa(m, "G").estimavel)
        for c in cons.contrastes:
            if c.desfecho == "F":
                self.assertEqual((c.estimavel, c.delta, c.minimo, c.maximo, c.fracao_repeticoes_positivas),
                                 (False, None, None, None, None))
            else:
                self.assertTrue(c.estimavel)
        self.assertTrue(all(a.auc is None and not a.valida for a in cons.auc_por_dobra if a.desfecho == "F"))
        self.assertTrue(all(d.delta is None for d in cons.deltas_por_dobra if d.desfecho == "F"))
        self.assertTrue(all(not d.valida for d in cons.denominadores if d.desfecho == "F"))

    def test_estados_dd_ri_ne_contra_referencia(self):
        # Faixa esparsa e sem relação com o alvo: mistura de dobras válidas e degeneradas.
        for semente, prevalencia in ((3, 0.09), (4, 0.11), (5, 0.07)):
            rng = np.random.default_rng(semente)
            mascara = rng.random(self.base.amostra.n) < prevalencia
            cons, _ = self.consolidar_com_faixa(mascara)
            f = next(e for e in cons.estados if e.desfecho == "F")
            for m in CODIGOS:
                theta, por_rep, dd, ri = referencia(self.res, mascara, m)
                with self.subTest(semente=semente, modelo=m):
                    self.assertEqual((f.dobras_degeneradas, f.repeticoes_invalidas), (dd, ri))
                    e = cons.estimativa(m, "F")
                    self.assertEqual(e.estimavel, theta is not None)
                    if theta is None:
                        self.assertIsNone(e.theta)
                    else:
                        self.assertAlmostEqual(e.theta, theta, places=12)
                    self.assertEqual(f.nao_estimavel, len(por_rep) < 8)
                    self.assertEqual(f.repeticoes_validas, len(por_rep))
            # a validade é a mesma para todos os modelos
            validade = {(d.repeticao, d.dobra): d.valida for d in cons.denominadores if d.desfecho == "F"}
            for a in cons.auc_por_dobra:
                if a.desfecho == "F":
                    self.assertEqual(a.valida, validade[(a.repeticao, a.dobra)])
                    self.assertEqual(a.auc is not None, a.valida)

    def test_ha_dobras_degeneradas_e_repeticoes_invalidas_no_cenario_esparso(self):
        mascara = np.random.default_rng(4).random(self.base.amostra.n) < 0.11
        cons, _ = self.consolidar_com_faixa(mascara)
        f = next(e for e in cons.estados if e.desfecho == "F")
        self.assertGreater(f.dobras_degeneradas, 0)
        self.assertGreaterEqual(f.repeticoes_invalidas, 0)
        invalidas = [a for a in cons.auc_por_repeticao if a.desfecho == "F" and not a.valida]
        self.assertTrue(all(a.auc is None for a in invalidas))  # a repetição inválida sai da estimativa

    def test_repeticao_com_menos_de_4_dobras_validas_e_invalida_e_com_4_vale(self):
        # Tira positivos de F até que EXATAMENTE uma dobra da repetição 0 fique degenerada; depois, duas.
        mascara = self.base.na_faixa.copy()
        alvo = self.res[("lr_fixa", 0, 0)].privado
        posicoes = np.asarray(alvo.posicoes)
        positivos = posicoes[np.asarray(alvo.y) == 1]
        mascara[positivos] = False  # dobra (0, 0) sem positivos na faixa → DD
        cons, _ = self.consolidar_com_faixa(mascara)
        rep0 = next(a for a in cons.auc_por_repeticao if (a.modelo, a.repeticao, a.desfecho) == ("rf", 0, "F"))
        self.assertEqual((rep0.n_dobras_validas, rep0.valida), (4, True))  # 4 de 5: ainda vale
        self.assertAlmostEqual(
            rep0.auc, float(np.mean([a.auc for a in cons.auc_por_dobra if (a.modelo, a.repeticao, a.desfecho)
                                     == ("rf", 0, "F") and a.auc is not None])), places=12)
        segunda = self.res[("lr_fixa", 0, 1)].privado
        mascara[np.asarray(segunda.posicoes)[np.asarray(segunda.y) == 1]] = False  # segunda dobra degenerada
        cons, _ = self.consolidar_com_faixa(mascara)
        rep0 = next(a for a in cons.auc_por_repeticao if (a.modelo, a.repeticao, a.desfecho) == ("rf", 0, "F"))
        self.assertEqual((rep0.n_dobras_validas, rep0.valida, rep0.auc), (3, False, None))
        f = next(e for e in cons.estados if e.desfecho == "F")
        self.assertGreaterEqual(f.repeticoes_invalidas, 1)

    def test_desfecho_com_8_repeticoes_validas_e_estimavel_e_com_7_nao(self):
        mascara = self.base.na_faixa.copy()
        for r in (0, 1):  # invalida as repetições 0 e 1 (duas dobras degeneradas em cada)
            for k in (0, 1):
                x = self.res[("lr_fixa", r, k)].privado
                mascara[np.asarray(x.posicoes)[np.asarray(x.y) == 1]] = False
        cons, _ = self.consolidar_com_faixa(mascara)
        f = next(e for e in cons.estados if e.desfecho == "F")
        invalidas = f.repeticoes_invalidas
        self.assertEqual(f.repeticoes_validas, 10 - invalidas)
        self.assertEqual(f.nao_estimavel, 10 - invalidas < 8)
        # Fronteira exata: com 3 repetições invalidadas restam 7 → NE; com 2 restam 8 → estimável.
        base2 = self.base.na_faixa.copy()
        for r in (0, 1, 2):
            for k in (0, 1):
                x = self.res[("lr_fixa", r, k)].privado
                base2[np.asarray(x.posicoes)[np.asarray(x.y) == 1]] = False
        cons3, _ = self.consolidar_com_faixa(base2)
        f3 = next(e for e in cons3.estados if e.desfecho == "F")
        self.assertGreaterEqual(f3.repeticoes_invalidas, 3)
        self.assertEqual(f3.nao_estimavel, f3.repeticoes_validas < 8)
        self.assertEqual(f.nao_estimavel, f.repeticoes_validas < 8)


class TestAvisosEscolhasEDescritivas(BaseConsolidacao):
    def test_avisos_de_convergencia_por_modelo(self):
        aviso = (("busca", "ConvergenceWarning", "não convergiu", 3), ("predicao", "UserWarning", "x", 1))
        cons = co.consolidar(co.parear(self.base, sn.resultados_sinteticos(self.ctx, avisos={"svm": aviso})))
        por_modelo = {a.modelo: a for a in cons.avisos}
        self.assertEqual(por_modelo["svm"].unidades_com_aviso, 50)
        self.assertEqual(por_modelo["svm"].avisos_convergencia, 150)
        self.assertEqual(len(por_modelo["svm"].unidades_com_convergencia), 50)
        for m in ("lr_fixa", "lr_tuning", "rf", "xgb"):
            self.assertEqual((por_modelo[m].unidades_com_aviso, por_modelo[m].avisos_convergencia), (0, 0))
        self.assertTrue(all(a.n_unidades == 50 for a in cons.avisos))

    def test_escolhas_de_hiperparametros_e_borda(self):
        self.assertEqual(len(self.cons.escolhas), 250)
        e = next(x for x in self.cons.escolhas if (x.modelo, x.repeticao, x.dobra) == ("svm", 2, 3))
        self.assertEqual(e.n_configuracoes, 30)
        self.assertEqual({nome for nome, _ in e.configuracao}, {"C", "gamma"})
        self.assertFalse(e.na_borda)
        cons = co.consolidar(co.parear(self.base, [
            sn.resultado(self.ctx, r.unidade, r.privado.pontuacoes, indice=1, na_borda=True) if r.unidade.modelo == "rf"
            else r for r in self.lista]))
        self.assertTrue(all(x.na_borda for x in cons.escolhas if x.modelo == "rf"))
        self.assertFalse(any(x.na_borda for x in cons.escolhas if x.modelo != "rf"))
        fixa = next(x for x in cons.escolhas if x.modelo == "lr_fixa")
        self.assertEqual((fixa.n_configuracoes, fixa.configuracao), (1, ()))

    def test_resumo_das_descritivas_so_para_g_e_sem_probabilidade_no_svm(self):
        metricas_svm = {d.metrica for d in self.cons.resumo_descritivas if d.modelo == "svm"}
        self.assertEqual(metricas_svm, set(C.metricas.descritivas_regra_nativa))
        for m in ("lr_fixa", "lr_tuning", "rf", "xgb"):
            self.assertEqual({d.metrica for d in self.cons.resumo_descritivas if d.modelo == m},
                             {x for x in C.metricas.descritivas_probabilisticas
                              if x not in ("intercepto_calibracao", "inclinacao_calibracao")} |
                             {x for x in ("intercepto_calibracao", "inclinacao_calibracao")
                              if any(d.modelo == m and d.metrica == x and d.valor is not None
                                     for d in self.cons.descritivas)})
        for d in self.cons.resumo_descritivas:
            self.assertEqual(d.n_dobras, sum(1 for x in self.cons.descritivas
                                             if (x.modelo, x.metrica) == (d.modelo, d.metrica) and x.valor is not None))
            self.assertLessEqual(d.minimo, d.media)
            self.assertLessEqual(d.media, d.maximo)
        self.assertEqual(len(self.cons.descritivas), 4 * 50 * 6 + 50 * 2)


class TestCurvasECalibracao(BaseConsolidacao):
    def test_curva_roc_media_vertical_em_101_pontos(self):
        self.assertEqual(len(self.cons.curvas_roc), 5)
        for c in self.cons.curvas_roc:
            self.assertEqual(c.n_dobras, 50)
            self.assertEqual(len(c.fpr), C.saidas.pontos_roc)
            self.assertEqual((c.fpr[0], c.fpr[-1]), (0.0, 1.0))
            self.assertAlmostEqual(c.tpr_media[-1], 1.0)
            self.assertTrue(all(b >= a - 1e-12 for a, b in zip(c.tpr_media, c.tpr_media[1:])))
            area = float(np.trapz(c.tpr_media, c.fpr))  # área da curva média ≈ θ̂ (G)
            self.assertAlmostEqual(area, self.cons.estimativa(c.modelo, "G").theta, delta=0.01)

    def test_curva_media_contra_calculo_independente(self):
        """Referência escrita do zero: pontos da curva a cada limiar e leitura em cada ``1 − especificidade``."""
        grade = np.linspace(0, 1, 101)

        def tpr_em(y, s, g):
            pontos = [(0.0, 0.0)] + [(np.mean(s[y == 0] >= t), np.mean(s[y == 1] >= t))
                                     for t in sorted(set(s.tolist()), reverse=True)]
            topo = [b for a, b in pontos if abs(a - g) < 1e-9]
            if topo:
                return max(topo)
            esq = max((p for p in pontos if p[0] < g), key=lambda p: (p[0], p[1]))
            dir_ = min((p for p in pontos if p[0] > g), key=lambda p: (p[0], p[1]))
            return esq[1] + (dir_[1] - esq[1]) * (g - esq[0]) / (dir_[0] - esq[0])

        linhas = []
        for r in range(3):
            for k in range(5):
                x = self.res[("lr_fixa", r, k)].privado
                y, s = np.asarray(x.y), np.asarray(x.pontuacoes)
                linhas.append([tpr_em(y, s, g) for g in grade])
        c = next(x for x in self.cons.curvas_roc if x.modelo == "lr_fixa")
        parciais = co.tpr_na_grade  # a mesma função aplicada às mesmas dobras reproduz a referência
        from sklearn.metrics import roc_curve
        obtidas = []
        for r in range(3):
            for k in range(5):
                x = self.res[("lr_fixa", r, k)].privado
                fpr, tpr, _ = roc_curve(np.asarray(x.y), np.asarray(x.pontuacoes), drop_intermediate=False)
                obtidas.append(parciais(fpr, tpr, grade))
        np.testing.assert_allclose(obtidas, linhas, atol=1e-12)
        self.assertEqual(len(c.tpr_media), 101)

    def test_degraus_nao_viram_diagonais(self):
        """ROC de 2 positivos × 2 negativos: sobe a 0,5, anda e sobe a 1,0; em 1 − esp = 0,25 a sensibilidade é 0,5."""
        from sklearn.metrics import roc_curve
        y = np.array([1, 1, 0, 0])
        s = np.array([0.9, 0.6, 0.7, 0.1])
        fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
        grade = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(co.tpr_na_grade(fpr, tpr, grade), [0.5, 0.5, 1.0, 1.0, 1.0])
        # empate entre positivo e negativo: segmento diagonal, interpolado
        y = np.array([1, 0])
        fpr, tpr, _ = roc_curve(y, np.array([0.5, 0.5]), drop_intermediate=False)
        np.testing.assert_allclose(co.tpr_na_grade(fpr, tpr, np.array([0.0, 0.5, 1.0])), [0.0, 0.5, 1.0])

    def test_curva_media_e_banda_dispersao(self):
        for c in self.cons.curvas_roc:
            self.assertTrue(all(lo <= hi + 1e-12 for lo, hi in zip(c.tpr_p025, c.tpr_p975)))
            self.assertTrue(all(0.0 <= v <= 1.0 + 1e-12 for v in c.tpr_media))

    def test_calibracao_por_decis_em_r0_so_para_quem_tem_probabilidade(self):
        modelos_cal = {f.modelo for f in self.cons.calibracao}
        self.assertEqual(modelos_cal, {"lr_fixa", "lr_tuning", "rf", "xgb"})  # sem SVM
        for m in modelos_cal:
            faixas = [f for f in self.cons.calibracao if f.modelo == m]
            self.assertEqual(len(faixas), 10)
            self.assertEqual(sum(f.nodulos for f in faixas), self.base.amostra.n)  # todas as linhas, uma vez (r = 0)
            self.assertTrue(all(f.nodulos >= C.saidas.min_nodulos_por_decil for f in faixas))
            self.assertTrue(all(0 <= f.frequencia_observada <= 1 and 0 < f.probabilidade_media < 1 for f in faixas))
            medias = [f.probabilidade_media for f in faixas]
            self.assertEqual(medias, sorted(medias))
            y = np.concatenate([np.asarray(self.res[(m, 0, k)].privado.y) for k in range(5)])
            esperado = sum(f.frequencia_observada * f.nodulos for f in faixas)
            self.assertAlmostEqual(esperado, float(y.sum()), places=9)

    def test_decis_pequenos_sao_fundidos_com_o_vizinho(self):
        grupos = [np.arange(5), np.arange(5, 25), np.arange(25, 28), np.arange(28, 60)]
        fundidos = co._fundir(grupos, 10)
        # o menor (3) funde com o vizinho menor (20) → 23; depois o 5 funde com o 23 → 28; o 32 fica
        self.assertEqual([len(g) for g in fundidos], [28, 32])
        self.assertTrue(all(len(g) >= 10 for g in fundidos))
        self.assertEqual(sum(len(g) for g in fundidos), 60)
        self.assertEqual(len(co._fundir([np.arange(3)], 10)), 1)  # um só grupo nunca some


class TestPrivacidade(BaseConsolidacao):
    def test_nenhuma_saida_agregada_carrega_identificador_ou_pontuacao_individual(self):
        textos = []
        for campo in dataclasses.fields(self.cons):
            valor = getattr(self.cons, campo.name)
            textos.append(repr(valor))
        junto = "\n".join(textos)
        for i in self.base.amostra.ids[::17]:
            self.assertNotIn(str(i), junto)
        for g in set(self.base.amostra.grupos.tolist()):
            self.assertNotIn(g, junto)
        self.assertNotIn("nodulo_ids", junto)
        self.assertEqual(self.cons.oficial, False)
        self.assertEqual(dict(self.cons.contagem_faixa)["nodulos"], int(self.base.na_faixa.sum()))


if __name__ == "__main__":
    unittest.main()
