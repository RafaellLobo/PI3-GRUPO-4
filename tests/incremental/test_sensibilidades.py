"""Testes das sensibilidades da Etapa 3 (``src/incremental/sensibilidades.py``).

Somente dados sintéticos em repositórios git temporários (ambiente de ``test_validacao``), com aprovações
FICTÍCIAS de P1–P3 no repositório temporário. As linhas de ``test`` têm valores inválidos: se fossem lidas,
a carga falharia. A referência numérica é um ajuste independente com scikit-learn direto.

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
for caminho in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.incremental import bootstrap as bs  # noqa: E402
from src.incremental import config as cfg  # noqa: E402
from src.incremental import dados  # noqa: E402
from src.incremental import guardas as g  # noqa: E402
from src.incremental import modelos as md  # noqa: E402
from src.incremental import sensibilidades as sens  # noqa: E402
from src.incremental import validacao  # noqa: E402
from test_guardas import VALOR_P3  # noqa: E402
from test_validacao import (ESPEC_PEQUENA, EXPLORATORIA, AmbienteE3, features_dos_modelos,  # noqa: E402
                            linhas_de)

PAINEL = tuple(VALOR_P3)


def matriz(linhas, features):
    return np.array([[float(l[f]) for f in features] for l in linhas])


def logistica(C):
    return LogisticRegression(C=C, penalty="l2", solver="lbfgs", class_weight=None, max_iter=1000, random_state=42)


def proba_independente(features, C=1.0, spline=False):
    """Ajuste em train e predição em validation, com sklearn direto (spline à mão)."""
    treino, val = linhas_de("train"), linhas_de("validation")
    X, Xv = matriz(treino, features), matriz(val, features)
    y = np.array([int(l["target_binary"]) for l in treino])
    if spline:
        espec = cfg.SPLINE_PROPOSTO
        st = SplineTransformer(n_knots=espec.n_knots, degree=espec.degree, knots=espec.knots,
                               extrapolation=espec.extrapolation, include_bias=espec.include_bias)
        base = st.fit_transform(X[:, [0]])
        X = np.hstack([base, X[:, 1:]])
        Xv = np.hstack([st.transform(Xv[:, [0]]), Xv[:, 1:]])
    sc = StandardScaler().fit(X)
    modelo = logistica(C).fit(sc.transform(X), y)
    return modelo.predict_proba(sc.transform(Xv))[:, 1]


class TestEspecificacao(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amb = AmbienteE3(cls.addClassCleanup)
        cls.defs = sens.definir_modelos(cls.amb.decisoes)
        cls.principal = features_dos_modelos()

    def test_spline_m0s_e_m3s(self):
        d = self.defs["spline"]
        self.assertEqual(set(d), {"M0s", "M3s"})
        self.assertEqual(d["M0s"].features, (cfg.DIAMETRO,))
        self.assertEqual(d["M3s"].features, (cfg.DIAMETRO, *PAINEL))
        for m in d.values():
            self.assertEqual(m.spline, cfg.SPLINE_PROPOSTO)  # parâmetros de P2
            self.assertEqual(m.C, cfg.ESPEC_AVALIACAO.C)

    def test_lista_reproduzida_troca_so_a_lista(self):
        d = self.defs["lista_reproduzida"]
        self.assertEqual(set(d), {"M2r", "M3r", "M4r"})
        self.assertEqual(d["M2r"].features, cfg.PAINEL_REPRODUZIDO)
        self.assertEqual(d["M3r"].features, (cfg.DIAMETRO, *[f for f in cfg.PAINEL_REPRODUZIDO if f != cfg.DIAMETRO]))
        self.assertEqual(len(d["M4r"].features), 10)  # Sphericity entra uma única vez
        self.assertEqual(len(set(d["M4r"].features)), 10)
        for m in d.values():
            self.assertIn("original_glcm_Imc1", m.features)
            self.assertIn("original_glcm_ClusterShade", m.features)
            self.assertNotIn("original_firstorder_Skewness", m.features)
            self.assertNotIn("original_ngtdm_Coarseness", m.features)
            self.assertEqual((m.C, m.spline), (cfg.ESPEC_AVALIACAO.C, None))

    def test_c_0_1_mesmas_features_dos_cinco_modelos(self):
        d = self.defs["c_0_1"]
        self.assertEqual(set(d), {"M0c", "M1c", "M2c", "M3c", "M4c"})
        for nome, m in d.items():
            self.assertEqual(m.features, tuple(self.principal[nome[:-1]]))
            self.assertEqual((m.C, m.spline), (0.1, None))

    def test_cada_analise_altera_um_unico_fator(self):
        # spline: features dos principais M0 e M3, mesmo C, mesma lista; só muda a representação do diâmetro.
        for nome, principal in (("M0s", "M0"), ("M3s", "M3")):
            self.assertEqual(self.defs["spline"][nome].features, tuple(self.principal[principal]))
        # lista reproduzida: C e representação do diâmetro do principal.
        # c_0_1: lista e representação do principal.
        self.assertTrue(all(m.spline is None for a in ("lista_reproduzida", "c_0_1") for m in self.defs[a].values()))
        self.assertTrue(all(m.C == 1.0 for a in ("spline", "lista_reproduzida") for m in self.defs[a].values()))
        self.assertEqual(sens.ANALISES, ("spline", "lista_reproduzida", "c_0_1"))  # sem combinações

    def test_modelos_m0_a_m4_nao_foram_tocados(self):
        self.assertEqual(cfg.ESPEC_AVALIACAO.C, 1.0)
        self.assertEqual(cfg.MODELOS_FIXOS, ("M0", "M1", "M2", "M3", "M4"))
        pipe = md.estimador_logistico()
        self.assertEqual(pipe.named_steps[md.PASSO_LOGREG].C, 1.0)
        self.assertEqual([n for n, _ in pipe.steps], [md.PASSO_SCALER, md.PASSO_LOGREG])

    def test_pendencia_de_p2_ou_p3_bloqueia_a_definicao(self):
        with self.assertRaises(cfg.BloqueioDecisao):
            sens.definir_modelos(dict(cfg.DECISOES))


class TestNumericoEIntegridade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amb = AmbienteE3(cls.addClassCleanup)
        cls.defs = sens.definir_modelos(cls.amb.decisoes)
        cls.carga = dados.carregar(sens.ATIVIDADE, ("train", "validation"),
                                   cfg.uniao_ordenada(*features_dos_modelos().values(), sens.features_necessarias()),
                                   cls.amb.decisoes, cls.amb.raiz, cls.amb.fonte)
        cls.treino, cls.validacao = cls.carga.frame("train"), cls.carga.frame("validation")

    def test_probabilidades_iguais_ao_ajuste_independente(self):
        for analise, modelos_ in self.defs.items():
            for nome, m in modelos_.items():
                proba, _ = sens.ajustar_e_prever(m, self.treino, self.validacao)
                esperado = proba_independente(m.features, C=m.C, spline=m.spline is not None)
                np.testing.assert_allclose(proba, esperado, rtol=0, atol=1e-12, err_msg=f"{analise}/{nome}")

    def test_spline_tem_cinco_colunas_e_nos_so_do_treino(self):
        m = self.defs["spline"]["M0s"]
        _, tecnico = sens.ajustar_e_prever(m, self.treino, self.validacao)
        self.assertEqual(tecnico["n_colunas_do_spline"], 5)  # n_knots + grau - 1 = 6, menos 1 sem viés (include_bias=False)
        self.assertEqual(tecnico["n_treino"], len(self.treino))
        pipe = m.estimador().fit(self.treino[[cfg.DIAMETRO]].to_numpy(), self.treino[cfg.COLUNA_ALVO].to_numpy())
        grau = cfg.SPLINE_PROPOSTO.degree
        nos = pipe.named_steps["spline"].transformers_[0][1].bsplines_[0].t[grau:-grau]  # nós internos da base
        esperado = np.percentile(self.treino[cfg.DIAMETRO].to_numpy(), np.linspace(0, 100, cfg.SPLINE_PROPOSTO.n_knots))
        np.testing.assert_allclose(nos, esperado)

    def test_ausencia_de_vazamento_a_validacao_nao_influencia_o_ajuste(self):
        # Mesmo treino, validações diferentes: as predições das linhas comuns não mudam; o ajuste só viu o treino.
        alterada = self.validacao.copy()
        alterada.loc[alterada.index[0], list(features_dos_modelos()["M3"])] = 9999.0
        alterada.loc[alterada.index[1], cfg.COLUNA_ALVO] = 1 - alterada.loc[alterada.index[1], cfg.COLUNA_ALVO]
        for analise, modelos_ in self.defs.items():
            for nome, m in modelos_.items():
                a, ta = sens.ajustar_e_prever(m, self.treino, self.validacao)
                b, tb = sens.ajustar_e_prever(m, self.treino, alterada)
                np.testing.assert_array_equal(a[2:], b[2:], err_msg=f"{analise}/{nome}")
                self.assertEqual((ta["n_treino"], tb["n_treino"]), (len(self.treino),) * 2)

    def test_entradas_invalidas(self):
        com_nan = self.treino.copy()
        com_nan.loc[com_nan.index[0], cfg.DIAMETRO] = np.nan
        with self.assertRaises(md.ModeloInvalido):
            sens.ajustar_e_prever(self.defs["c_0_1"]["M0c"], com_nan, self.validacao)
        with self.assertRaises(sens.SensibilidadeInvalida):
            sens.ajustar_e_prever(self.defs["c_0_1"]["M0c"], self.treino.drop(columns=[cfg.DIAMETRO]), self.validacao)


class TestAutorizacao(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amb = AmbienteE3(cls.addClassCleanup)
        cls.autorizacao = g.exigir_autorizacao(cfg.Etapa.E3, cls.amb.decisoes, cls.amb.raiz, modelos=validacao.MODELOS)
        cls.carga = dados.carregar(sens.ATIVIDADE, ("train", "validation"),
                                   cfg.uniao_ordenada(*features_dos_modelos().values(), sens.features_necessarias()),
                                   cls.amb.decisoes, cls.amb.raiz, cls.amb.fonte)

    def chamar(self, autorizacao, decisoes=None):
        val = self.carga.frame("validation")
        return sens.avaliar_sensibilidades(
            autorizacao=autorizacao, decisoes=decisoes or self.amb.decisoes, treino=self.carga.frame("train"),
            validacao=val, probas_principais={m: np.full(len(val), 0.5) for m in validacao.MODELOS},
            contrastes_principais={f"{a}-{b}": {"auc": {}} for a, b in validacao.CONTRASTES}, espec_bootstrap=ESPEC_PEQUENA)

    def test_autorizacao_de_outra_atividade_e_recusada(self):
        outra = g.exigir_autorizacao(cfg.Etapa.E0, self.amb.decisoes, self.amb.raiz)
        with self.assertRaises(sens.SensibilidadeInvalida):
            self.chamar(outra)
        with self.assertRaises(sens.SensibilidadeInvalida):
            self.chamar(None)

    def test_autorizacao_sem_p2_ou_p3_e_recusada(self):
        for sem in (("P1", "P3"), ("P1", "P2")):
            parcial = dataclasses.replace(self.autorizacao, decisoes_exigidas=sem)
            with self.assertRaises(sens.SensibilidadeInvalida):
                self.chamar(parcial)

    def test_decisoes_pendentes_bloqueiam_mesmo_com_autorizacao_valida(self):
        with self.assertRaises(cfg.BloqueioDecisao):
            self.chamar(self.autorizacao, decisoes=dict(cfg.DECISOES))


class TestIntegracaoNaEtapa3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amb = AmbienteE3(cls.addClassCleanup)
        cls.res, cls.ck_dir, _ = cls.amb.executar("sens", finalidade=EXPLORATORIA, espec_bootstrap=ESPEC_PEQUENA)
        cls.rel = json.loads(cls.res.caminho_relatorio.read_text(encoding="utf-8"))
        cls.s = cls.rel["sensibilidades"]
        cls.val = linhas_de("validation")
        cls.y = np.array([int(l["target_binary"]) for l in cls.val])
        cls.pac = [l["patient_id"] for l in cls.val]

    def test_tres_analises_com_os_contrastes_do_protocolo(self):
        self.assertEqual({k for k in self.s if k in sens.ANALISES}, set(sens.ANALISES))
        self.assertEqual(set(self.s["spline"]["contrastes"]), {"M3s-M0s"})
        self.assertEqual(set(self.s["lista_reproduzida"]["contrastes"]), {"M3r-M0", "M4r-M1", "M2r-M0"})
        self.assertEqual(set(self.s["c_0_1"]["contrastes"]), {"M3c-M0c", "M4c-M1c", "M1c-M0c", "M2c-M0c"})

    def test_auc_e_delta_iguais_ao_calculo_independente(self):
        m0s, m3s = proba_independente((cfg.DIAMETRO,), spline=True), proba_independente(
            (cfg.DIAMETRO, *PAINEL), spline=True)
        ponto = roc_auc_score(self.y, m3s) - roc_auc_score(self.y, m0s)
        self.assertAlmostEqual(self.s["spline"]["contrastes"]["M3s-M0s"]["auc"]["ponto"], ponto, places=10)
        m3r = proba_independente((cfg.DIAMETRO, *[f for f in cfg.PAINEL_REPRODUZIDO if f != cfg.DIAMETRO]))
        m0 = proba_independente((cfg.DIAMETRO,))
        self.assertAlmostEqual(self.s["lista_reproduzida"]["contrastes"]["M3r-M0"]["auc"]["ponto"],
                               roc_auc_score(self.y, m3r) - roc_auc_score(self.y, m0), places=10)
        c3, c0 = proba_independente((cfg.DIAMETRO, *PAINEL), C=0.1), proba_independente((cfg.DIAMETRO,), C=0.1)
        self.assertAlmostEqual(self.s["c_0_1"]["contrastes"]["M3c-M0c"]["auc"]["ponto"],
                               roc_auc_score(self.y, c3) - roc_auc_score(self.y, c0), places=10)

    def test_bootstrap_pareado_com_a_mesma_amostra_da_analise_principal(self):
        direto = bs.bootstrap_pareado(self.pac, self.y, {
            "M0s": proba_independente((cfg.DIAMETRO,), spline=True),
            "M3s": proba_independente((cfg.DIAMETRO, *PAINEL), spline=True)}, (("M3s", "M0s"),), ESPEC_PEQUENA)
        obtido = self.s["spline"]["contrastes"]["M3s-M0s"]["auc"]
        esperado = direto.contrastes["auc"]["M3s-M0s"]
        self.assertAlmostEqual(obtido["limite_inferior"], esperado.limite_inferior, places=9)
        self.assertAlmostEqual(obtido["limite_superior"], esperado.limite_superior, places=9)
        for analise in sens.ANALISES:  # mesmos sorteios de pacientes: mesma contagem de réplicas descartadas
            self.assertEqual(self.s[analise]["bootstrap"]["n_descartadas"], self.rel["bootstrap"]["n_descartadas"])
            self.assertTrue(self.s[analise]["bootstrap"]["mesma_amostra_de_pacientes_da_analise_principal"])

    def test_resultado_principal_acompanha_cada_contraste(self):
        for analise in sens.ANALISES:
            for c in self.s[analise]["contrastes"].values():
                self.assertEqual(c["principal"], self.rel["contrastes"][c["contraste_principal_correspondente"]]["auc"])

    def test_sensibilidades_nao_geram_checkpoints_nem_alteram_as_cinco_unidades(self):
        slots = sorted(p.name for p in (self.ck_dir / validacao.ATIVIDADE).iterdir())
        self.assertEqual(slots, [f"r000-f00-{m}" for m in validacao.MODELOS])

    def test_leitura_descritiva_sem_regua_da_secao_9_e_sem_vocabulario_inferencial(self):
        for analise in sens.ANALISES:
            for c in self.s[analise]["contrastes"].values():
                self.assertNotIn("leitura_secao_9", c)
        texto = self.res.caminho_relatorio.read_text(encoding="utf-8").lower()
        for proibido in ("p-valor", "p_valor", "significativ", "lidc"):
            self.assertNotIn(proibido, texto)

    def test_determinismo_e_isolamento_do_teste(self):
        outra, _, _ = self.amb.executar("sens_b", finalidade=EXPLORATORIA, espec_bootstrap=ESPEC_PEQUENA)
        self.assertEqual(outra.caminho_relatorio.read_bytes(), self.res.caminho_relatorio.read_bytes())
        self.assertEqual(self.rel["particoes_lidas"], ["train", "validation"])  # 'test' tem lixo e não foi lido


class TestContratoDoModulo(unittest.TestCase):
    def test_nao_toca_dados_nem_o_teste_e_nao_altera_guardas(self):
        arvore = ast.parse(inspect.getsource(sens))
        importados = {n.module or "" for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom)}
        nomes = {a.name for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom) for a in n.names}
        self.assertNotIn("dados", nomes)  # nunca carrega dados: recebe os quadros de train e validation
        self.assertFalse([m for m in importados if "benchmark_s6" in m])
        literais = {n.value for n in ast.walk(arvore) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertNotIn("test", literais)
        self.assertNotIn("test", sens.ATIVIDADE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
