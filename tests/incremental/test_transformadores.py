"""Testes dos transformadores da seleção (``src/incremental/transformadores.py``).

Somente dados sintéticos; nenhuma leitura da base oficial e nenhum modelo.
A referência histórica abaixo é a regra do código auditado, copiada sem
alteração (um teste confere que as linhas continuam no script).

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
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV

REPO_ROOT = Path(__file__).resolve().parents[2]
for caminho in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.incremental import config as cfg  # noqa: E402
from src.incremental import transformadores as tf  # noqa: E402

LIMIAR = cfg.ESPEC_SELECAO.limiar_correlacao
SCRIPT_AUDITADO = REPO_ROOT / "scripts" / "sprint4" / "reproduce_juan_feature_selection.py"
LINHAS_AUDITADAS = (
    "vt = VarianceThreshold(threshold=0.0)",
    "f_variancia = X_train.columns[vt.get_support()].tolist()",
    "corr = X_train[f_variancia].corr().abs()",
    "upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))",
    "redundantes = [col for col in upper.columns if (upper[col] > 0.90).any()]",
    "f_reduzidas = [f for f in f_variancia if f not in redundantes]",
)


# --------------------------------------------------------------------------- #
# Referência histórica (cópia literal das linhas auditadas)
# --------------------------------------------------------------------------- #


def historico_correlacao(X_train: pd.DataFrame, limiar: float = 0.90) -> list:
    f_variancia = X_train.columns.tolist()
    corr = X_train[f_variancia].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundantes = [col for col in upper.columns if (upper[col] > limiar).any()]
    return [f for f in f_variancia if f not in redundantes]


def historico_filtros(X_train: pd.DataFrame) -> list:
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_train)
    f_variancia = X_train.columns[vt.get_support()].tolist()
    corr = X_train[f_variancia].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundantes = [col for col in upper.columns if (upper[col] > 0.90).any()]
    return [f for f in f_variancia if f not in redundantes]


# --------------------------------------------------------------------------- #
# Dados sintéticos
# --------------------------------------------------------------------------- #


def sintetico(semente: int, n: int = 40, p: int = 14) -> pd.DataFrame:
    """Blocos de colunas correlacionadas (sinais mistos), ruído, constantes e inteiros."""
    rng = np.random.RandomState(semente)
    fatores = rng.normal(size=(n, 3))
    colunas = {}
    for j in range(p):
        tipo = rng.randint(6)
        if tipo == 0:
            valores = np.full(n, rng.normal())  # constante: correlação indefinida
        elif tipo == 1:
            valores = rng.normal(size=n)
        elif tipo == 2:
            valores = rng.randint(0, 4, size=n)  # inteiro
        else:
            sinal = rng.choice([-1.0, 1.0])
            ruido = rng.uniform(0.05, 0.8)  # cobre a vizinhança de |r| = 0,90
            valores = sinal * fatores[:, rng.randint(3)] + ruido * rng.normal(size=n)
        colunas[f"feat_{j:02d}"] = valores
    return pd.DataFrame(colunas)


def ortonormais(n: int, k: int, semente: int = 0) -> np.ndarray:
    """k vetores centrados e ortonormais em R^n (correlações controladas por ângulos)."""
    rng = np.random.RandomState(semente)
    m = rng.normal(size=(n, k))
    m -= m.mean(axis=0)
    q, _ = np.linalg.qr(m)
    return q


def cadeia(n: int = 50, sinal_c: float = 1.0) -> pd.DataFrame:
    """r(A,B) = r(B,C) = cos 18° ≈ 0,951 > 0,90 e r(A,C) = cos 36° ≈ 0,809 < 0,90."""
    e1, e2 = ortonormais(n, 2).T
    t = np.deg2rad(18.0)
    a = e1
    b = np.cos(t) * e1 + np.sin(t) * e2
    c = sinal_c * (np.cos(2 * t) * e1 + np.sin(2 * t) * e2)
    return pd.DataFrame({"A": a, "B": b, "C": c})


def nomes_mantidos(filtro) -> list:
    return list(filtro.get_feature_names_out())


# --------------------------------------------------------------------------- #
# Equivalência com a regra histórica
# --------------------------------------------------------------------------- #


class TestEquivalenciaHistorica(unittest.TestCase):
    def test_linhas_auditadas_continuam_no_script(self):
        fonte = SCRIPT_AUDITADO.read_text(encoding="utf-8")
        for linha in LINHAS_AUDITADAS:
            self.assertIn(linha, fonte)

    def test_filtro_igual_a_regra_historica(self):
        removidas = 0
        for semente in range(300):
            X = sintetico(semente)
            with self.subTest(semente=semente):
                f = tf.FiltroCorrelacao().fit(X)
                esperado = historico_correlacao(X)
                self.assertEqual(nomes_mantidos(f), esperado)
                # mesma máscara com array numpy (sem nomes)
                np.testing.assert_array_equal(tf.FiltroCorrelacao().fit(X.to_numpy()).get_support(),
                                              f.get_support())
                removidas += X.shape[1] - len(esperado)
        self.assertGreater(removidas, 0)

    def test_pipeline_igual_aos_filtros_historicos(self):
        for semente in range(300):
            X = sintetico(semente)
            with self.subTest(semente=semente):
                pipe = tf.filtros_selecao().fit(X)
                self.assertEqual(list(pipe.get_feature_names_out()), historico_filtros(X))

    def test_vizinhanca_do_limiar_bit_a_bit(self):
        # Correlações calculadas pela mesma função: o empate estrito é respeitado.
        X = cadeia()
        r = tf.correlacao_absoluta(X.to_numpy())[0, 1]
        self.assertEqual(r, X.corr().abs().iloc[0, 1])
        no_limiar = tf.FiltroCorrelacao(limiar=r).fit(X[["A", "B"]])
        self.assertEqual(nomes_mantidos(no_limiar), ["A", "B"])  # |r| == limiar: mantém
        abaixo = tf.FiltroCorrelacao(limiar=float(np.nextafter(r, 0))).fit(X[["A", "B"]])
        self.assertEqual(nomes_mantidos(abaixo), ["A"])
        for limiar in (r, float(np.nextafter(r, 0))):
            self.assertEqual(nomes_mantidos(tf.FiltroCorrelacao(limiar=limiar).fit(X[["A", "B"]])),
                             historico_correlacao(X[["A", "B"]], limiar))

    def test_limiar_padrao_vem_da_configuracao(self):
        self.assertEqual(tf.FiltroCorrelacao().get_params(), {"limiar": LIMIAR})
        pipe = tf.filtros_selecao()
        self.assertEqual(pipe.named_steps[tf.PASSO_VARIANCIA].threshold, cfg.ESPEC_SELECAO.limiar_variancia)
        self.assertIsInstance(pipe.named_steps[tf.PASSO_VARIANCIA], VarianceThreshold)
        self.assertEqual(pipe.named_steps[tf.PASSO_CORRELACAO].limiar, LIMIAR)


# --------------------------------------------------------------------------- #
# Regra: sinal, colunas já removidas, ordem, correlação indefinida
# --------------------------------------------------------------------------- #


class TestRegra(unittest.TestCase):
    def test_correlacao_positiva_e_negativa(self):
        e1, e2 = ortonormais(30, 2).T
        X = pd.DataFrame({"x": e1, "pos": 0.95 * e1 + 0.3 * e2, "neg": -0.95 * e1 - 0.3 * e2, "outra": e2})
        f = tf.FiltroCorrelacao().fit(X)
        self.assertEqual(nomes_mantidos(f), ["x", "outra"])
        self.assertEqual(nomes_mantidos(f), historico_correlacao(X))

    def test_comparacao_com_coluna_ja_removida(self):
        for sinal in (1.0, -1.0):
            with self.subTest(sinal_c=sinal):
                X = cadeia(sinal_c=sinal)
                corr = X.corr().abs()
                self.assertGreater(corr.loc["A", "B"], LIMIAR)
                self.assertGreater(corr.loc["B", "C"], LIMIAR)
                self.assertLess(corr.loc["A", "C"], LIMIAR)
                # C sai por causa de B, mesmo B tendo saído: um filtro guloso manteria C.
                self.assertEqual(nomes_mantidos(tf.FiltroCorrelacao().fit(X)), ["A"])
                self.assertEqual(historico_correlacao(X), ["A"])

    def test_ordem_das_colunas_decide_a_representante(self):
        X = cadeia()
        for ordem, esperado in ((["A", "B", "C"], ["A"]), (["C", "B", "A"], ["C"]),
                                (["B", "A", "C"], ["B"]), (["A", "C", "B"], ["A", "C"])):
            with self.subTest(ordem=ordem):
                f = tf.FiltroCorrelacao().fit(X[ordem])
                self.assertEqual(nomes_mantidos(f), esperado)
                self.assertEqual(historico_correlacao(X[ordem]), esperado)

    def test_correlacao_indefinida_nao_remove(self):
        e1, e2 = ortonormais(30, 2).T
        X = pd.DataFrame({"const1": np.full(30, 2.5), "x": e1, "const2": np.full(30, -1.0),
                          "dup": e1.copy(), "y": e2})
        corr = tf.correlacao_absoluta(X.to_numpy())
        self.assertTrue(np.isnan(corr[0, 1]) and np.isnan(corr[0, 2]))
        f = tf.FiltroCorrelacao().fit(X)
        self.assertEqual(nomes_mantidos(f), ["const1", "x", "const2", "y"])
        self.assertEqual(nomes_mantidos(f), historico_correlacao(X))
        # no pipeline, as constantes saem antes, pelo VarianceThreshold
        self.assertEqual(list(tf.filtros_selecao().fit(X).get_feature_names_out()), ["x", "y"])

    def test_nunca_esvazia_e_primeira_coluna_fica(self):
        e1 = ortonormais(20, 1)[:, 0]
        X = np.column_stack([e1, 2 * e1, -e1, e1 + 1])
        f = tf.FiltroCorrelacao().fit(X)
        np.testing.assert_array_equal(f.get_support(indices=True), [0])

    def test_limiar_padrao_e_estrito(self):
        # Dados inteiros centrados com r = 9/10: o pandas devolve exatamente 0.9.
        a = np.array([2.0, -2.0, 1.0, -1.0, 0.0])
        b = np.array([2.0, -2.0, 1.0, 0.0, -1.0])
        casos = (("abaixo", 1e-6, ["a", "b"]), ("igual", 0.0, ["a", "b"]), ("acima", -1e-6, ["a"]))
        for nome, perturbacao, esperado in casos:
            for sinal in (1.0, -1.0):
                with self.subTest(caso=nome, sinal=sinal):
                    bb = b.copy()
                    bb[3] += perturbacao
                    X = pd.DataFrame({"a": a, "b": sinal * bb})
                    r = tf.correlacao_absoluta(X.to_numpy())[0, 1]
                    self.assertEqual(r, abs(X.corr().iloc[0, 1]))
                    self.assertEqual(np.sign(r - LIMIAR), {"abaixo": -1, "igual": 0, "acima": 1}[nome])
                    self.assertEqual(nomes_mantidos(tf.FiltroCorrelacao().fit(X)), esperado)
                    self.assertEqual(historico_correlacao(X), esperado)

    def test_mascara_isolada(self):
        corr = np.array([[1.0, 0.95, np.nan], [0.95, 1.0, 0.91], [np.nan, 0.91, 1.0]])
        np.testing.assert_array_equal(tf.mascara_correlacao(corr, LIMIAR), [True, False, False])
        np.testing.assert_array_equal(tf.mascara_correlacao(corr, 0.95), [True, True, True])
        with self.assertRaises(ValueError):
            tf.mascara_correlacao(np.ones((2, 3)), LIMIAR)


# --------------------------------------------------------------------------- #
# Nomes, suporte e mapeamento para as features originais
# --------------------------------------------------------------------------- #


class TestNomesESuporte(unittest.TestCase):
    def setUp(self):
        self.X = sintetico(7)  # remoções intercaladas com colunas mantidas
        self.filtro = tf.FiltroCorrelacao().fit(self.X)

    def test_suporte_e_nomes_na_ordem_original(self):
        mascara = self.filtro.get_support()
        nomes = self.filtro.get_feature_names_out()
        self.assertEqual(list(nomes), [c for c, m in zip(self.X.columns, mascara) if m])
        np.testing.assert_array_equal(self.filtro.get_support(indices=True), np.flatnonzero(mascara))
        self.assertLess(len(nomes), self.X.shape[1])
        self.assertEqual(list(self.filtro.feature_names_in_), list(self.X.columns))

    def test_saida_corresponde_as_colunas_originais(self):
        # Coluna i da saída = feature get_feature_names_out()[i]: é o que liga coeficientes a features.
        saida = self.filtro.transform(self.X)
        np.testing.assert_array_equal(saida, self.X[self.filtro.get_feature_names_out()].to_numpy())
        pipe = tf.filtros_selecao().fit(self.X)
        np.testing.assert_array_equal(pipe.transform(self.X), self.X[pipe.get_feature_names_out()].to_numpy())

    def test_saida_pandas_e_array_sem_nomes(self):
        pipe = tf.filtros_selecao().set_output(transform="pandas").fit(self.X)
        saida = pipe.transform(self.X)
        self.assertEqual(list(saida.columns), historico_filtros(self.X))
        pd.testing.assert_frame_equal(saida, self.X[historico_filtros(self.X)])
        f = tf.FiltroCorrelacao().fit(self.X.to_numpy())
        self.assertEqual(list(f.get_feature_names_out()),
                         [f"x{i}" for i in np.flatnonzero(self.filtro.get_support())])
        self.assertEqual(list(f.get_feature_names_out(list(self.X.columns))),
                         list(self.filtro.get_feature_names_out()))

    def test_suporte_somente_leitura(self):
        with self.assertRaises(ValueError):
            self.filtro.support_[0] = not self.filtro.support_[0]
        with self.assertRaises(ValueError):
            self.filtro.get_support()[0] = False


# --------------------------------------------------------------------------- #
# sklearn: clone, Pipeline e GridSearchCV
# --------------------------------------------------------------------------- #


class TestSklearn(unittest.TestCase):
    def test_clone_e_parametros(self):
        original = tf.FiltroCorrelacao(limiar=0.8).fit(sintetico(1))
        copia = clone(original)
        self.assertEqual(copia.get_params(), {"limiar": 0.8})
        self.assertFalse(hasattr(copia, "support_"))
        with self.assertRaises(NotFittedError):
            copia.transform(sintetico(1))
        self.assertIs(copia.set_params(limiar=0.7), copia)
        self.assertEqual(copia.limiar, 0.7)
        pipe = clone(tf.filtros_selecao())
        pipe.set_params(**{f"{tf.PASSO_CORRELACAO}__limiar": 0.85})
        self.assertEqual(pipe.named_steps[tf.PASSO_CORRELACAO].limiar, 0.85)
        self.assertEqual(tf.filtros_selecao().get_params()[f"{tf.PASSO_CORRELACAO}__limiar"], LIMIAR)

    def test_grid_search_ajusta_so_no_treino_de_cada_split(self):
        X = sintetico(5, n=48)
        y = np.tile([0, 1], 24)
        rng = np.random.RandomState(0)
        splits = []
        for _ in range(3):
            perm = rng.permutation(len(X))
            splits.append((np.sort(perm[:32]), np.sort(perm[32:])))
        chamadas = []
        original = tf.correlacao_absoluta

        def espiao(matriz):
            chamadas.append(np.array(matriz, copy=True))
            return original(matriz)

        def pontuacao(estimador, X_aval, y_aval):
            estimador.transform(X_aval)  # aplica o estado aprendido, sem recalcular
            return 0.0

        busca = GridSearchCV(tf.filtros_selecao(), {f"{tf.PASSO_CORRELACAO}__limiar": [LIMIAR]},
                             cv=splits, scoring=pontuacao, refit=False)
        with unittest.mock.patch.object(tf, "correlacao_absoluta", side_effect=espiao):
            busca.fit(X, y)
        self.assertEqual(len(chamadas), len(splits))  # uma por ajuste; nenhuma no transform
        for recebida, (treino, _) in zip(chamadas, splits):
            X_treino = X.iloc[treino]
            esperado = VarianceThreshold(0.0).fit(X_treino).transform(X_treino)
            np.testing.assert_array_equal(recebida, esperado)
        # Com refit, o melhor estimador mapeia os nomes às colunas originais.
        busca = GridSearchCV(tf.filtros_selecao(), {f"{tf.PASSO_CORRELACAO}__limiar": [LIMIAR]},
                             cv=splits, scoring=pontuacao, refit=True).fit(X, y)
        nomes = list(busca.best_estimator_.get_feature_names_out())
        self.assertEqual(nomes, historico_filtros(X))
        np.testing.assert_array_equal(busca.transform(X), X[nomes].to_numpy())


# --------------------------------------------------------------------------- #
# Ausência de aprendizado com dados de avaliação
# --------------------------------------------------------------------------- #


class TestSemVazamento(unittest.TestCase):
    def setUp(self):
        self.treino = cadeia(n=40)
        # Avaliação com estrutura oposta: A, B, C independentes entre si.
        e = ortonormais(20, 3, semente=4)
        self.avaliacao = pd.DataFrame(e, columns=["A", "B", "C"])

    def test_transform_nao_recalcula_nem_reaprende(self):
        f = tf.FiltroCorrelacao().fit(self.treino)
        suporte = f.support_
        with unittest.mock.patch.object(tf, "correlacao_absoluta") as corr:
            saida = f.transform(self.avaliacao)
            f.get_feature_names_out()
        corr.assert_not_called()
        self.assertIs(f.support_, suporte)
        np.testing.assert_array_equal(saida, self.avaliacao[["A"]].to_numpy())
        # Se aprendesse com a avaliação, manteria as três colunas:
        self.assertEqual(nomes_mantidos(tf.FiltroCorrelacao().fit(self.avaliacao)), ["A", "B", "C"])

    def test_novo_fit_substitui_todo_o_estado(self):
        f = tf.FiltroCorrelacao().fit(self.avaliacao).fit(self.treino)
        novo = tf.FiltroCorrelacao().fit(self.treino)
        np.testing.assert_array_equal(f.get_support(), novo.get_support())
        f.fit(self.treino.to_numpy())
        self.assertFalse(hasattr(f, "feature_names_in_"))

    def test_y_ignorado_e_entrada_intacta(self):
        X = sintetico(9)
        copia = X.copy()
        base = tf.FiltroCorrelacao().fit(X).get_support()
        for y in (None, np.zeros(len(X)), np.arange(len(X)) % 2, np.random.RandomState(1).normal(size=len(X))):
            np.testing.assert_array_equal(tf.FiltroCorrelacao().fit(X, y).get_support(), base)
        pd.testing.assert_frame_equal(X, copia)
        tf.filtros_selecao().fit(X).transform(X)
        pd.testing.assert_frame_equal(X, copia)


# --------------------------------------------------------------------------- #
# Entradas inválidas, colunas incompatíveis e conjuntos vazios
# --------------------------------------------------------------------------- #


class TestEntradasInvalidas(unittest.TestCase):
    def setUp(self):
        self.X = sintetico(2)

    def test_ajuste_recusa(self):
        X = self.X.to_numpy()
        com_nan, com_inf = X.copy(), X.copy()
        com_nan[0, 0], com_inf[1, 1] = np.nan, np.inf
        casos = {
            "nan": com_nan, "inf": com_inf, "uma linha": X[:1], "sem linhas": X[:0],
            "sem colunas": X[:, :0], "1-D": X[:, 0], "texto": np.array([["a", "b"], ["c", "d"]]),
            "nomes repetidos": pd.DataFrame(X[:, :2], columns=["f", "f"]),
            "complexo": X.astype(complex),
        }
        for nome, entrada in casos.items():
            with self.subTest(caso=nome):
                with self.assertRaises(ValueError):
                    tf.FiltroCorrelacao().fit(entrada)
        with self.assertRaises(TypeError):
            tf.FiltroCorrelacao().fit(sparse.csr_matrix(X))

    def test_limiar_invalido(self):
        for limiar in (-0.1, 1.0, 1.5, np.nan, np.inf, True, "0.9", None):
            with self.subTest(limiar=limiar):
                with self.assertRaises(ValueError):
                    tf.FiltroCorrelacao(limiar=limiar).fit(self.X)
        self.assertEqual(tf.FiltroCorrelacao(limiar=np.float64(0.0)).fit(self.X).limiar_, 0.0)

    def test_transform_recusa_colunas_incompativeis(self):
        f = tf.FiltroCorrelacao().fit(self.X)
        renomeado = self.X.rename(columns={self.X.columns[0]: "outra"})
        com_nan = self.X.astype(float)
        com_nan.iloc[0, 0] = np.nan
        casos = {
            "reordenado": self.X[self.X.columns[::-1]],
            "renomeado": renomeado,
            "coluna a menos": self.X.iloc[:, 1:],
            "coluna a mais": self.X.assign(extra=1.0),
            "array sem nomes": self.X.to_numpy(),
            "nan": com_nan,
            "sem linhas": self.X.iloc[:0],
        }
        for nome, entrada in casos.items():
            with self.subTest(caso=nome):
                with self.assertRaises(ValueError):
                    f.transform(entrada)
        with self.assertRaises(TypeError):
            f.transform(sparse.csr_matrix(self.X.to_numpy()))
        g = tf.FiltroCorrelacao().fit(self.X.to_numpy())
        for nome, entrada in {"nomes após ajuste sem nomes": self.X,
                              "colunas a menos": self.X.to_numpy()[:, 1:],
                              "texto": self.X.to_numpy().astype(str).astype(object)}.items():
            with self.subTest(caso=nome):
                with self.assertRaises(ValueError):
                    g.transform(entrada)
        self.assertEqual(g.transform(self.X.to_numpy()[:1]).shape[0], 1)

    def test_nao_ajustado(self):
        f = tf.FiltroCorrelacao()
        for metodo in (lambda: f.transform(self.X), f.get_support, f.get_feature_names_out):
            with self.assertRaises(NotFittedError):
                metodo()


# --------------------------------------------------------------------------- #
# Módulo
# --------------------------------------------------------------------------- #


class TestModulo(unittest.TestCase):
    def setUp(self):
        self.fonte = (REPO_ROOT / "src" / "incremental" / "transformadores.py").read_text(encoding="utf-8")
        self.arvore = ast.parse(self.fonte)

    def test_limiares_nao_duplicados(self):
        reais = {n.value for n in ast.walk(self.arvore)
                 if isinstance(n, ast.Constant) and type(n.value) is float}
        self.assertFalse(reais & {LIMIAR, cfg.ESPEC_SELECAO.limiar_variancia}, reais)
        for nome in ("limiar_correlacao", "limiar_variancia"):
            self.assertIn(f"espec.{nome}", self.fonte)

    def test_sem_variancia_reimplementada_modelos_ou_dados(self):
        importados, chamadas = set(), set()
        for no in ast.walk(self.arvore):
            if isinstance(no, ast.Import):
                importados.update(a.name for a in no.names)
            elif isinstance(no, ast.ImportFrom):
                importados.add(("." * no.level) + (no.module or ""))
                importados.update(a.name for a in no.names)
            elif isinstance(no, ast.Call):
                chamadas.add(no.func.attr if isinstance(no.func, ast.Attribute) else getattr(no.func, "id", None))
        self.assertIn("VarianceThreshold", importados)
        self.assertFalse(importados & {".dados", "dados", "LogisticRegression", "GridSearchCV", "pickle",
                                       "StandardScaler", "RandomForestClassifier"})
        self.assertFalse(chamadas & {"var", "nanvar", "ptp", "predict", "predict_proba", "read_csv",
                                     "carregar"})

    def test_configuracao_inalterada(self):
        from test_guardas import HASH_CONFIG_INCREMENTO_1
        self.assertEqual(cfg.hash_configuracao(), HASH_CONFIG_INCREMENTO_1)


if __name__ == "__main__":
    unittest.main()
