"""Testes da fábrica de modelos (``src/incremental/modelos.py``).

Somente dados sintéticos; nenhuma leitura da base oficial e nenhuma métrica.
O baseline da Sprint 3 é lido por AST (``scripts/run_baseline.py`` não é
importado nem executado). Aprovações de P3 existem apenas em repositórios
git temporários, como nos testes das guardas.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/incremental -v
"""

from __future__ import annotations

import ast
import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

REPO_ROOT = Path(__file__).resolve().parents[2]
for caminho in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.incremental import config as cfg  # noqa: E402
from src.incremental import guardas as g  # noqa: E402
from src.incremental import modelos as md  # noqa: E402
from src.incremental import selecao as sel  # noqa: E402
from test_guardas import (  # noqa: E402
    HASH_CONFIG_INCREMENTO_1, VALOR_P2, VALOR_P3, RepoTemporario, aprovar, protocolo, registro,
)

E0, E1, E3 = cfg.Etapa.E0.value, cfg.Etapa.E1.value, cfg.Etapa.E3.value
BASELINE = REPO_ROOT / "scripts" / "run_baseline.py"
S_K = (cfg.PAINEL_DECLARADO[0], cfg.DIAMETRO, cfg.PAINEL_DECLARADO[3])  # contém o diâmetro
COLUNAS = cfg.uniao_ordenada(cfg.FEATURES_M1, cfg.PAINEL_DECLARADO, S_K)


def frame_sintetico(n=120, semente=3):
    """DataFrame com todas as colunas usadas nos testes e rótulo dependente de duas delas."""
    rng = np.random.RandomState(semente)
    dados = {nome: rng.normal(loc=10.0, scale=2.0, size=n) for nome in COLUNAS}
    logito = (dados[cfg.DIAMETRO] - 10.0) - 0.5 * (dados[cfg.PAINEL_DECLARADO[0]] - 10.0)
    y = (logito + rng.normal(size=n) > 0).astype(np.int64)
    return pd.DataFrame(dados), pd.Series(y)


FRAME, Y = frame_sintetico()


def selecao_com(s_k):
    """``SelecaoFold`` sintético com S_k dado (sem rodar a seleção)."""
    return sel.SelecaoFold(repeticao=0, fold_externo=0, sha256_amostra="0" * 64, fingerprint_folds="1" * 64,
                           hash_configuracao=cfg.hash_configuracao(), features=COLUNAS, folds=(),
                           frequencia={}, s_k=tuple(s_k))


# --------------------------------------------------------------------------- #
# Baseline da Sprint 3, lido por AST
# --------------------------------------------------------------------------- #


def _constantes_do_baseline():
    arvore = ast.parse(BASELINE.read_text(encoding="utf-8"))
    simples = {}
    for no in arvore.body:
        alvo = no.target if isinstance(no, ast.AnnAssign) else (no.targets[0] if isinstance(no, ast.Assign) else None)
        if isinstance(alvo, ast.Name) and no.value is not None:
            try:
                simples[alvo.id] = ast.literal_eval(no.value)
            except (ValueError, TypeError, SyntaxError):
                pass

    def valor(no):
        if isinstance(no, ast.Name):
            return simples[no.id]
        if isinstance(no, ast.Dict):
            return {valor(k): valor(v) for k, v in zip(no.keys, no.values)}
        return ast.literal_eval(no)

    params = None
    for no in arvore.body:
        alvo = no.target if isinstance(no, ast.AnnAssign) else None
        if isinstance(alvo, ast.Name) and alvo.id == "LOGREG_PARAMS":
            params = valor(no.value)
    funcao = next(n for n in arvore.body if isinstance(n, ast.FunctionDef) and n.name == "construir_pipeline")
    passos = [(t.elts[0].value, t.elts[1].func.id) for t in ast.walk(funcao)
              if isinstance(t, ast.Tuple) and len(t.elts) == 2 and isinstance(t.elts[0], ast.Constant)
              and isinstance(t.elts[1], ast.Call)]
    return simples["BASELINE_FEATURES"], params, passos


class TestEstimadores(unittest.TestCase):
    def test_hiperparametros_da_regressao_oficial(self):
        pipe = md.estimador_logistico()
        self.assertEqual([n for n, _ in pipe.steps], [md.PASSO_SCALER, md.PASSO_LOGREG])
        self.assertEqual([type(e) for _, e in pipe.steps], [StandardScaler, LogisticRegression])
        lr = pipe.named_steps[md.PASSO_LOGREG]
        self.assertEqual({k: lr.get_params()[k] for k in cfg.ESPEC_AVALIACAO.parametros()},
                         cfg.ESPEC_AVALIACAO.parametros())
        self.assertEqual(cfg.ESPEC_AVALIACAO.parametros(),
                         {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000,
                          "class_weight": None, "random_state": 42})
        dummy = md.estimador_dummy()
        self.assertEqual([type(e) for _, e in dummy.steps], [DummyClassifier])
        self.assertEqual(dummy.named_steps[md.PASSO_DUMMY].strategy, "prior")

    def test_equivalencia_estrutural_com_o_m1_da_sprint3(self):
        features, params, passos = _constantes_do_baseline()
        self.assertEqual(tuple(features), cfg.FEATURES_M1)
        self.assertEqual(passos, [(md.PASSO_SCALER, "StandardScaler"), (md.PASSO_LOGREG, "LogisticRegression")])
        baseline = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(**params))])
        nosso = md.estimador_logistico()
        # solver omitido no baseline = padrão lbfgs do scikit-learn 1.3.2: parâmetros idênticos.
        self.assertEqual(baseline.named_steps["logreg"].get_params(), nosso.named_steps[md.PASSO_LOGREG].get_params())
        m1 = md.construir_fixo("M1", E0)
        self.assertEqual(m1.features, tuple(features))
        X = FRAME[list(features)]
        baseline.fit(X.to_numpy(dtype=float), Y.to_numpy(dtype=int))
        ajustado = md.ajustar(m1, X, Y).estimador
        for atributo in ("mean_", "scale_"):
            np.testing.assert_array_equal(getattr(ajustado.named_steps[md.PASSO_SCALER], atributo),
                                          getattr(baseline.named_steps["scaler"], atributo))
        for atributo in ("coef_", "intercept_"):
            np.testing.assert_array_equal(getattr(ajustado.named_steps[md.PASSO_LOGREG], atributo),
                                          getattr(baseline.named_steps["logreg"], atributo))


# --------------------------------------------------------------------------- #
# Composição, construção e S_k
# --------------------------------------------------------------------------- #


class TestComposicao(unittest.TestCase):
    def test_fixos_de_referencia(self):
        self.assertEqual(md.construir_fixo("M0", E0).features, (cfg.DIAMETRO,))
        self.assertEqual(md.construir_fixo("M1", E0).features, cfg.FEATURES_M1)

    def test_variaveis_com_s_k_pronto_sem_rodar_a_selecao(self):
        def proibido(*_a, **_k):
            raise AssertionError("a fábrica não pode executar a seleção")

        with unittest.mock.patch.object(sel, "selecionar_fold_externo", proibido), \
                unittest.mock.patch.object(GridSearchCV, "fit", proibido), \
                unittest.mock.patch.object(sel, "busca_C", proibido):
            modelos = md.modelos_do_fold(selecao_com(S_K), E0)
        self.assertEqual(set(modelos), set(cfg.MODELOS_VARIAVEIS))
        self.assertEqual(modelos["M2v"].features, S_K)
        self.assertEqual(modelos["M3v"].features, (cfg.DIAMETRO, S_K[0], S_K[2]))  # diâmetro uma única vez
        self.assertEqual(modelos["M4v"].features, cfg.uniao_ordenada(cfg.FEATURES_M1, S_K))
        for m in modelos.values():
            self.assertIsInstance(m.estimador.steps[-1][1], LogisticRegression)

    def test_construcao_nao_ajusta_e_clona(self):
        m = md.construir_fixo("M1", E0)
        for _, passo in m.estimador.steps:
            with self.assertRaises(NotFittedError):
                check_is_fitted(passo)
        copia = clone(m.estimador)
        self.assertIsNot(copia, m.estimador)
        self.assertEqual(copia.get_params(deep=True).keys(), m.estimador.get_params(deep=True).keys())
        self.assertEqual(copia.named_steps[md.PASSO_LOGREG].get_params(),
                         m.estimador.named_steps[md.PASSO_LOGREG].get_params())
        ajustado = md.ajustar(m, FRAME[list(m.features)], Y)
        with self.assertRaises(NotFittedError):
            check_is_fitted(m.estimador.named_steps[md.PASSO_LOGREG])  # o construído segue intacto
        with self.assertRaises(NotFittedError):
            check_is_fitted(clone(ajustado.estimador).named_steps[md.PASSO_LOGREG])

    def test_entradas_de_construcao(self):
        with self.assertRaises(TypeError):
            md.construir(cfg.FEATURES_M1, E0)
        with self.assertRaises(TypeError):
            md.modelos_do_fold({"s_k": S_K}, E0)
        with self.assertRaises(ValueError):
            md.construir_fixo("M2v", E0)
        with self.assertRaises(g.AtividadeDesconhecida):
            md.construir_fixo("M0", "etapa_inexistente")


class TestSkVazio(unittest.TestCase):
    def setUp(self):
        self.modelos = md.modelos_do_fold(selecao_com(()), E0)

    def test_m2v_dummy_da_prevalencia_do_treino(self):
        m2v = self.modelos["M2v"]
        self.assertEqual(m2v.features, ())
        self.assertTrue(m2v.espec.usa_dummy)
        self.assertEqual([type(e) for _, e in m2v.estimador.steps], [DummyClassifier])
        treino = FRAME.iloc[:80]
        ajustado = md.ajustar(m2v, treino[[]], Y.iloc[:80])
        proba = ajustado.prever_proba(FRAME.iloc[80:][[]])
        np.testing.assert_allclose(proba, np.full(40, Y.iloc[:80].mean()))

    def test_m3v_e_m4v_continuam_treinaveis(self):
        m3v, m4v = self.modelos["M3v"], self.modelos["M4v"]
        self.assertEqual(m3v.features, (cfg.DIAMETRO,))
        self.assertEqual(m4v.features, cfg.FEATURES_M1)
        for m, fixo in ((m3v, "M0"), (m4v, "M1")):
            with self.subTest(modelo=m.nome):
                variavel = md.ajustar(m, FRAME[list(m.features)], Y).estimador.named_steps[md.PASSO_LOGREG]
                ref = md.construir_fixo(fixo, E0)
                referencia = md.ajustar(ref, FRAME[list(ref.features)], Y).estimador.named_steps[md.PASSO_LOGREG]
                np.testing.assert_array_equal(variavel.coef_, referencia.coef_)


# --------------------------------------------------------------------------- #
# Ajuste e predição
# --------------------------------------------------------------------------- #


class TestAjustePredicao(unittest.TestCase):
    def test_ajuste_e_predicao_sinteticos(self):
        m = md.modelos_do_fold(selecao_com(S_K), E0)["M4v"]
        treino, teste = FRAME.iloc[:90], FRAME.iloc[90:]
        ajustado = md.ajustar(m, treino[list(m.features)], Y.iloc[:90])
        self.assertEqual(ajustado.n_treino, 90)
        proba = ajustado.prever_proba(teste[list(m.features)])
        self.assertEqual(proba.shape, (30,))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        np.testing.assert_array_equal(ajustado.prever_classe(teste[list(m.features)]),
                                      (proba >= cfg.THRESHOLD).astype(int))
        np.testing.assert_array_equal(
            proba, ajustado.estimador.predict_proba(teste[list(m.features)].to_numpy(dtype=float))[:, 1])

    def test_deterministico_e_ajustes_independentes(self):
        m = md.construir_fixo("M1", E0)
        X = FRAME[list(m.features)]
        a, b = md.ajustar(m, X, Y), md.ajustar(m, X, Y)
        self.assertIsNot(a.estimador, b.estimador)
        np.testing.assert_array_equal(a.prever_proba(X), b.prever_proba(X))
        c = md.ajustar(m, X.iloc[:60], Y.iloc[:60])
        self.assertFalse(np.array_equal(a.prever_proba(X), c.prever_proba(X)))


class TestColunasERotulos(unittest.TestCase):
    def setUp(self):
        self.m = md.construir_fixo("M1", E0)
        self.X = FRAME[list(self.m.features)]

    def test_colunas_recusadas_sem_adaptacao(self):
        com_nan = self.X.copy()
        com_nan.iloc[0, 0] = np.nan
        com_inf = self.X.copy()
        com_inf.iloc[1, 2] = np.inf
        casos = {
            "ordem trocada": self.X[list(self.m.features[::-1])],
            "coluna a menos": self.X.iloc[:, 1:],
            "coluna a mais": FRAME[list(self.m.features) + [cfg.PAINEL_DECLARADO[0]]],
            "renomeada": self.X.rename(columns={cfg.DIAMETRO: "original_shape_Diametro"}),
            "array sem nomes": self.X.to_numpy(),
            "booleana": self.X.assign(**{cfg.DIAMETRO: self.X[cfg.DIAMETRO] > 10}),
            "texto": self.X.astype(str),
            "nan": com_nan,
            "inf": com_inf,
            "sem linhas": self.X.iloc[:0],
        }
        ajustado = md.ajustar(self.m, self.X, Y)
        for nome, X in casos.items():
            with self.subTest(caso=nome):
                with self.assertRaises(md.ModeloInvalido):
                    md.ajustar(self.m, X, Y.iloc[:len(X)] if hasattr(X, "__len__") else Y)
                with self.assertRaises(md.ModeloInvalido):
                    ajustado.prever_proba(X)

    def test_rotulos_recusados(self):
        y = Y.to_numpy()
        deslocado = Y.copy()
        deslocado.index = deslocado.index + 1
        casos = {
            "float": y.astype(float), "booleano": y.astype(bool), "uma classe": np.zeros_like(y),
            "valor 2": np.where(y == 1, 2, 0), "comprimento": y[:-1], "2-D": y.reshape(-1, 1),
            "série desalinhada": deslocado, "dataframe": Y.to_frame(),
        }
        for nome, rotulos in casos.items():
            with self.subTest(caso=nome):
                with self.assertRaises(md.ModeloInvalido):
                    md.ajustar(self.m, self.X, rotulos)
        with self.assertRaises(TypeError):
            md.ajustar(self.m.estimador, self.X, Y)


# --------------------------------------------------------------------------- #
# Bloqueios de P3 e autorização
# --------------------------------------------------------------------------- #


class TestBloqueiosP3(unittest.TestCase):
    def setUp(self):
        self.repo = RepoTemporario()
        self.addCleanup(self.repo.remover)
        r12 = [registro("P1"), registro("P2", VALOR_P2)]
        self.c1 = self.repo.commit(protocolo(*r12), "2026-10-02")
        self.e1 = aprovar(P1=(self.c1, None), P2=(self.c1, VALOR_P2))
        self.r12 = r12

    def test_p3_pendente_bloqueia_m2_m4(self):
        for nome in cfg.MODELOS_COM_PAINEL:
            for atividade in (E0, E1, E3, cfg.ATIVIDADE_CV_LISTA_FIXA):
                with self.subTest(modelo=nome, atividade=atividade):
                    with self.assertRaises(cfg.BloqueioDecisao):
                        md.construir_fixo(nome, atividade, self.e1, self.repo.raiz)

    def test_p3_so_em_memoria_nao_basta(self):
        # Decisão marcada como aprovada sem registro no git: modelo_fixo aceita, a guarda recusa.
        forjadas = aprovar(self.e1, P3=(self.c1, VALOR_P3))
        espec = cfg.modelo_fixo("M2", forjadas)
        with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
            md.construir(espec, E3, forjadas, self.repo.raiz)
        # Espec de M2 sem a marca de modelo_fixo nem chega a existir.
        with self.assertRaises(cfg.BloqueioDecisao):
            cfg.EspecModelo("M2", cfg.TipoModelo.FIXO, VALOR_P3)
        # ModeloConstruido montado à mão com autorização forjada não ajusta nem prevê.
        forjada = g.Autorizacao(E3, ("P1", "P2", "P3"), (), None, HASH_CONFIG_INCREMENTO_1)
        manual = md.ModeloConstruido(espec, E3, forjada, md.estimador_logistico(), forjadas, self.repo.raiz)
        with self.assertRaises(cfg.BloqueioDecisao):
            md.ajustar(manual, FRAME[list(espec.features)], Y)

    def test_p3_aprovada_e_verificada_libera_m2_m4(self):
        c2 = self.repo.commit(protocolo(*self.r12, registro("P3", VALOR_P3, "2026-10-03")), "2026-10-03")
        decisoes = aprovar(self.e1, P3=(c2, VALOR_P3))
        for nome in cfg.MODELOS_COM_PAINEL:
            with self.subTest(modelo=nome):
                m = md.construir_fixo(nome, E3, decisoes, self.repo.raiz)
                self.assertIn("P3", m.autorizacao.decisoes_exigidas)
                self.assertEqual(m.features, cfg.uniao_ordenada(cfg.REFERENCIA[nome], VALOR_P3))
                X = FRAME[list(m.features)]
                self.assertEqual(md.ajustar(m, X, Y).prever_proba(X).shape, (len(X),))

    def test_autorizacao_revalidada_em_cada_operacao(self):
        c2 = self.repo.commit(protocolo(*self.r12, registro("P3", VALOR_P3, "2026-10-03")), "2026-10-03")
        decisoes = aprovar(self.e1, P3=(c2, VALOR_P3))
        m = md.construir_fixo("M2", E3, decisoes, self.repo.raiz)
        X = FRAME[list(m.features)]
        ajustado = md.ajustar(m, X, Y)
        # Novo commit: o HEAD da autorização muda e as operações exigem reconstrução.
        self.repo.commit({"outro.txt": "x\n"}, "2026-10-04")
        with self.assertRaises(g.AprovacaoNaoVerificada):
            md.ajustar(m, X, Y)
        with self.assertRaises(g.AprovacaoNaoVerificada):
            ajustado.prever_proba(X)
        # Registro de P3 removido do protocolo: a reconstrução é bloqueada.
        self.repo.commit(protocolo(*self.r12), "2026-10-05")
        with self.assertRaises(cfg.BloqueioDecisao):
            md.construir_fixo("M2", E3, decisoes, self.repo.raiz)

    def test_decisoes_da_atividade_tambem_sao_exigidas(self):
        for nome in ("M0", "M1"):
            with self.subTest(modelo=nome):
                with self.assertRaises(cfg.BloqueioDecisao):
                    md.construir_fixo(nome, E1)  # P1 e P2 pendentes no estado real
        with self.assertRaises(cfg.BloqueioDecisao):
            md.modelos_do_fold(selecao_com(S_K), E1)
        self.assertEqual(md.construir_fixo("M1", E1, self.e1, self.repo.raiz).autorizacao.decisoes_exigidas,
                         ("P1", "P2"))


# --------------------------------------------------------------------------- #
# Módulo
# --------------------------------------------------------------------------- #


class TestModulo(unittest.TestCase):
    def setUp(self):
        self.fonte = (REPO_ROOT / "src" / "incremental" / "modelos.py").read_text(encoding="utf-8")
        self.arvore = ast.parse(self.fonte)

    def test_hiperparametros_nao_duplicados(self):
        constantes = [n.value for n in ast.walk(self.arvore) if isinstance(n, ast.Constant)]
        for proibido in (1.0, 1000, 42, 0.5, "lbfgs", "l2", "prior", "l1", "liblinear"):
            self.assertNotIn(proibido, [c for c in constantes if type(c) is type(proibido)], proibido)
        for nome in ("ESPEC_AVALIACAO.parametros()", "DUMMY_ESTRATEGIA", "THRESHOLD", "autorizar_modelo"):
            self.assertIn(nome, self.fonte)

    def test_sem_selecao_leitura_ou_ajuste_na_importacao(self):
        chamadas = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
                    for n in ast.walk(self.arvore) if isinstance(n, ast.Call)}
        self.assertFalse(chamadas & {"selecionar_fold_externo", "busca_C", "pipeline_selecao", "consolidar",
                                     "GridSearchCV", "read_csv", "carregar", "exigir_autorizacao"})
        nivel_modulo = [n for n in self.arvore.body if not isinstance(
            n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Expr))]
        self.assertTrue(all(isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Tuple, ast.JoinedStr))
                            for n in nivel_modulo))
        importados = {no.module for no in ast.walk(self.arvore) if isinstance(no, ast.ImportFrom)}
        self.assertFalse({m for m in importados if m and ("dados" in m or "scripts" in m)})

    def test_configuracao_inalterada(self):
        self.assertEqual(cfg.hash_configuracao(), HASH_CONFIG_INCREMENTO_1)


if __name__ == "__main__":
    unittest.main()
