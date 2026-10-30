"""Testes da seleção LASSO ajustada para a CV aninhada (``src/incremental/selecao.py``).

Somente dados sintéticos: pacientes fictícios (LIDC-IDRI-9xxx), de 1 a 3
nódulos por paciente, rótulos variando dentro do paciente. Pool sintético:
uma constante (sai no filtro de variância), dois sinais, uma cópia quase
exata do primeiro sinal (sai no filtro de correlação) e ruído. Nenhuma
leitura da base oficial; nenhuma métrica de validação ou teste.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/incremental -v
"""

from __future__ import annotations

import ast
import dataclasses
import json
import sys
import unittest
import unittest.mock
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
for caminho in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.incremental import config as cfg  # noqa: E402
from src.incremental import folds as fl  # noqa: E402
from src.incremental import rastreabilidade as rst  # noqa: E402
from src.incremental import selecao as sel  # noqa: E402
from src.incremental import transformadores as tf  # noqa: E402

ESPEC = cfg.ESPEC_SELECAO
GRADE = list(ESPEC.grade_C.valores())
FEATURES = ("original_sint_constante", "original_sint_sinal_a", "original_sint_ruido_1",
            "original_sint_copia_a", "original_sint_sinal_b", "original_sint_ruido_2",
            "original_sint_ruido_3", "original_sint_ruido_4")
CONSTANTE, SINAL_A, COPIA_A, SINAL_B = FEATURES[0], FEATURES[1], FEATURES[3], FEATURES[4]


def sintetico(n_pacientes=100, semente=11, sinal=1.0):
    """Amostra e X alinhados por posição; y depende de sinal_a − sinal_b."""
    rng = np.random.RandomState(semente)
    linhas = []
    for p in range(n_pacientes):
        paciente = f"LIDC-IDRI-{9000 + p:04d}"
        for k in range(1, rng.randint(1, 4) + 1):
            a, b = rng.normal(size=2)
            ruido = rng.normal(size=4)
            x = [3.0, a, ruido[0], 2 * a + 0.01 * rng.normal(), b, *ruido[1:]]
            linhas.append((f"{paciente}_N{k:02d}_scan{9000 + p}", int(sinal * (a - b) + rng.normal() > 0),
                           paciente, x))
    linhas.sort()
    ids, y, grupos, X = zip(*linhas)
    return fl.criar_amostra(list(ids), np.array(y, dtype=np.int64), list(grupos)), np.array(X)


AMOSTRA, X = sintetico()
PLANO = fl.plano_repeticao(AMOSTRA, 0)
#: Ruído puro com S_k vazio de ponta a ponta (semente fixada após inspeção).
AMOSTRA_RUIDO, X_RUIDO = sintetico(semente=4, sinal=0.0)

_CACHE = {}


def selecao_k0():
    if "k0" not in _CACHE:
        _CACHE["k0"] = sel.selecionar_fold_externo(AMOSTRA, X, FEATURES, PLANO, 0)
    return _CACHE["k0"]


def chave(linhas: np.ndarray) -> frozenset:
    """Identifica um conjunto de linhas pelo conteúdo (todas as linhas sintéticas são distintas)."""
    return frozenset(bytes(linha) for linha in np.ascontiguousarray(linhas, dtype=np.float64))


class Espiao:
    """Registra as linhas recebidas por ``fit`` dos quatro passos do pipeline."""

    def __init__(self):
        self.registros = {"variancia": [], "correlacao": [], "scaler": [], "l1": []}
        self._patches = []

    def _envolver(self, classe, nome):
        original = classe.fit
        registros = self.registros[nome]

        def fit(estimador, X_, y=None, *args, **kwargs):
            registros.append(np.array(X_, dtype=np.float64, copy=True))
            return original(estimador, X_, y, *args, **kwargs)

        return unittest.mock.patch.object(classe, "fit", fit)

    def __enter__(self):
        for classe, nome in ((VarianceThreshold, "variancia"), (tf.FiltroCorrelacao, "correlacao"),
                             (StandardScaler, "scaler"), (LogisticRegression, "l1")):
            p = self._envolver(classe, nome)
            p.start()
            self._patches.append(p)
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()


def problemas_de_aninhamento(registros, esperado_interno, esperado_refit):
    """Lista o que viola 'filtros e scaler ajustados DENTRO de cada treino interno'.

    ``esperado_interno``: chaves dos treinos internos (cada uma ajustada uma vez
    por valor de C); ``esperado_refit``: chaves dos treinos de estabilidade
    (um refit cada). Qualquer ajuste em outro conjunto de linhas é violação.
    """
    esperado = Counter()
    for k in esperado_interno:
        esperado[k] += len(GRADE)
    for k in esperado_refit:
        esperado[k] += 1
    tamanhos = Counter()
    for k, c in esperado.items():
        tamanhos[len(k)] += c
    problemas = []
    # VarianceThreshold recebe as linhas originais: conferidas pelo conteúdo.
    if Counter(chave(m) for m in registros["variancia"]) != esperado:
        problemas.append("variancia: ajustes fora dos treinos internos e do refit")
    # Os passos seguintes recebem colunas já filtradas: conferidos pelo número de linhas.
    for passo in ("correlacao", "scaler", "l1"):
        if Counter(len(m) for m in registros[passo]) != tamanhos:
            problemas.append(f"{passo}: número ou tamanho de ajustes diferente do esperado")
    return problemas


# --------------------------------------------------------------------------- #
# Pipeline, grade e índices
# --------------------------------------------------------------------------- #


class TestPipelineEGrade(unittest.TestCase):
    def test_pipeline_completo_com_parametros_da_configuracao(self):
        pipe = sel.pipeline_selecao()
        self.assertEqual([n for n, _ in pipe.steps],
                         [tf.PASSO_VARIANCIA, tf.PASSO_CORRELACAO, sel.PASSO_SCALER, sel.PASSO_L1])
        tipos = [type(e) for _, e in pipe.steps]
        self.assertEqual(tipos, [VarianceThreshold, tf.FiltroCorrelacao, StandardScaler, LogisticRegression])
        self.assertEqual(pipe.named_steps[tf.PASSO_VARIANCIA].threshold, ESPEC.limiar_variancia)
        self.assertEqual(pipe.named_steps[tf.PASSO_CORRELACAO].limiar, ESPEC.limiar_correlacao)
        lr = pipe.named_steps[sel.PASSO_L1]
        self.assertEqual((lr.penalty, lr.solver, lr.max_iter, lr.random_state),
                         (ESPEC.penalty, ESPEC.solver, ESPEC.max_iter, ESPEC.random_state))
        self.assertEqual(clone(pipe).get_params()[sel.PARAMETRO_C], 1.0)

    def test_busca_sobre_a_grade_original(self):
        busca = sel.busca_C([(np.arange(3), np.arange(3, 6))])
        self.assertIsInstance(busca, GridSearchCV)
        np.testing.assert_array_equal(busca.param_grid[sel.PARAMETRO_C], np.logspace(-4, 2, 13))
        self.assertEqual((busca.scoring, busca.refit, busca.n_jobs, busca.error_score),
                         (ESPEC.scoring, True, 1, "raise"))
        self.assertIsInstance(busca.estimator, Pipeline)

    def test_posicoes_locais(self):
        base = np.array([2, 5, 7, 11, 13], dtype=np.int64)
        np.testing.assert_array_equal(sel.posicoes_locais(base, np.array([5, 11, 13])), [1, 3, 4])
        for globais in (np.array([3]), np.array([14]), np.array([0, 2]), np.array([7, 5]), np.array([5, 5])):
            with self.subTest(globais=globais.tolist()):
                with self.assertRaises(sel.SelecaoInvalida):
                    sel.posicoes_locais(base, globais)


# --------------------------------------------------------------------------- #
# Resultado, mapeamento de coeficientes e S_k
# --------------------------------------------------------------------------- #


class TestResultado(unittest.TestCase):
    def test_estrutura_por_fold(self):
        r = selecao_k0()
        self.assertEqual((r.repeticao, r.fold_externo), (0, 0))
        self.assertEqual(len(r.folds), ESPEC.n_folds_estabilidade)
        self.assertEqual([f.seed_interna for f in r.folds],
                         [ESPEC.seed_interna(j) for j in range(1, ESPEC.n_folds_estabilidade + 1)])
        for f, est in zip(r.folds, PLANO.estabilidade[0]):
            self.assertIn(f.C, GRADE)
            self.assertEqual(f.n_nodulos, len(est.treino))
            self.assertEqual(f.n_pacientes, len(set(AMOSTRA.grupos[est.treino])))
            self.assertEqual((f.n_pos_variancia, f.n_pos_correlacao), (len(FEATURES) - 1, len(FEATURES) - 2))
            nomes = [n for n, _ in f.coeficientes]
            self.assertNotIn(CONSTANTE, nomes)
            self.assertNotIn(COPIA_A, nomes)
            self.assertEqual(nomes, [n for n in FEATURES if n in nomes])  # ordem do esquema
            self.assertEqual(f.selecionadas, tuple(n for n, c in f.coeficientes if abs(c) > ESPEC.limiar_coeficiente))
            self.assertTrue(f.convergiu and 0 < f.n_iter < ESPEC.max_iter)

    def test_frequencia_e_s_k(self):
        r = selecao_k0()
        self.assertEqual(set(r.frequencia), set(FEATURES))
        for nome in FEATURES:
            self.assertEqual(r.frequencia[nome], sum(nome in f.selecionadas for f in r.folds))
        self.assertEqual(r.s_k, tuple(n for n in FEATURES if r.frequencia[n] >= ESPEC.limiar_estabilidade))
        self.assertIn(SINAL_A, r.s_k)
        self.assertIn(SINAL_B, r.s_k)
        self.assertFalse(r.vazio)

    def test_coeficientes_mapeados_as_features_originais(self):
        # Refit independente, coluna a coluna por nome: um deslocamento de índice
        # (ignorar a constante removida antes) trocaria os nomes.
        r = selecao_k0()
        externa = PLANO.externos[0]
        for f, est in zip(r.folds, PLANO.estabilidade[0]):
            X_est = X[est.treino]
            filtros = tf.filtros_selecao().fit(X_est)
            nomes = [FEATURES[int(c[1:])] for c in filtros.get_feature_names_out()]
            self.assertEqual(nomes, [n for n, _ in f.coeficientes])
            modelo = Pipeline([("s", StandardScaler()),
                               ("l", LogisticRegression(penalty=ESPEC.penalty, solver=ESPEC.solver, C=f.C,
                                                        max_iter=ESPEC.max_iter, random_state=ESPEC.random_state))])
            modelo.fit(filtros.transform(X_est), AMOSTRA.y[est.treino])
            np.testing.assert_array_equal(modelo.named_steps["l"].coef_.ravel(), [c for _, c in f.coeficientes])
            coef = dict(f.coeficientes)
            self.assertGreater(coef[SINAL_A], 0)
            self.assertLess(coef[SINAL_B], 0)
            self.assertFalse(set(est.treino) & set(externa.avaliacao))

    def test_modelos_reutilizam_o_mesmo_s_k(self):
        r = selecao_k0()
        modelos = r.modelos()
        self.assertEqual(set(modelos), set(cfg.MODELOS_VARIAVEIS))
        self.assertEqual(modelos["M2v"].features, r.s_k)
        self.assertEqual(modelos["M3v"].features, cfg.uniao_ordenada((cfg.DIAMETRO,), r.s_k))
        self.assertEqual(modelos["M4v"].features, cfg.uniao_ordenada(cfg.FEATURES_M1, r.s_k))
        self.assertFalse(modelos["M2v"].usa_dummy)

    def test_serializacao_sem_dados_individuais(self):
        dados = selecao_k0().como_dict()
        texto = rst.json_canonico(dados)
        self.assertNotIn("LIDC", texto)
        self.assertEqual(json.loads(texto)["s_k"], list(selecao_k0().s_k))
        self.assertEqual(dados["hash_configuracao"], cfg.hash_configuracao())
        self.assertEqual(dados["fingerprint_folds"], fl.fingerprint_unidade(AMOSTRA, PLANO, 0))
        rst._exigir_sem_dados_individuais(dados)


class TestSkVazio(unittest.TestCase):
    def test_vazio_de_ponta_a_ponta(self):
        r = sel.selecionar_fold_externo(AMOSTRA_RUIDO, X_RUIDO, FEATURES, fl.plano_repeticao(AMOSTRA_RUIDO, 0), 0)
        self.assertEqual(r.s_k, ())
        self.assertTrue(r.vazio)
        self.assertTrue(all(v < ESPEC.limiar_estabilidade for v in r.frequencia.values()))
        modelos = r.modelos()
        self.assertEqual(modelos["M2v"].features, ())
        self.assertTrue(modelos["M2v"].usa_dummy)
        self.assertEqual(modelos["M3v"].features, (cfg.DIAMETRO,))
        self.assertEqual(modelos["M4v"].features, cfg.FEATURES_M1)
        self.assertEqual(r.como_dict()["s_k_vazio"], True)

    def test_consolidar_limiar_e_ordem(self):
        n = ESPEC.n_folds_estabilidade
        limiar = ESPEC.limiar_estabilidade
        cedo, tarde, abaixo = FEATURES[1], FEATURES[6], FEATURES[2]
        # tarde: todos os folds; cedo: exatamente o limiar; abaixo: limiar - 1. Entrada em ordem invertida.
        folds = [[tarde] + ([cedo] if j < limiar else []) + ([abaixo] if j < limiar - 1 else []) for j in range(n)]
        folds = [lista[::-1] for lista in folds]
        frequencia, s_k = sel.consolidar(FEATURES, folds)
        self.assertEqual((frequencia[tarde], frequencia[cedo], frequencia[abaixo]), (n, limiar, limiar - 1))
        self.assertEqual(s_k, (cedo, tarde))  # ordem do esquema, não da frequência nem da entrada
        self.assertEqual(sel.consolidar(FEATURES, [[] for _ in range(n)])[1], ())

    def test_consolidar_recusa(self):
        n = ESPEC.n_folds_estabilidade
        casos = {
            "folds a menos": [[]] * (n - 1),
            "nome desconhecido": [["original_sint_outra"]] + [[]] * (n - 1),
            "repetido": [[FEATURES[1], FEATURES[1]]] + [[]] * (n - 1),
        }
        for nome, folds in casos.items():
            with self.subTest(caso=nome):
                with self.assertRaises(sel.SelecaoInvalida):
                    sel.consolidar(FEATURES, folds)
        with self.assertRaises(TypeError):
            sel.consolidar(FEATURES, [FEATURES[1]] + [[]] * (n - 1))


# --------------------------------------------------------------------------- #
# Aninhamento, isolamento por paciente e ausência de vazamento
# --------------------------------------------------------------------------- #


class TestAninhamento(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = 1
        with Espiao() as espiao:
            cls.resultado = sel.selecionar_fold_externo(AMOSTRA, X, FEATURES, PLANO, cls.k)
        cls.registros = espiao.registros
        cls.internos, cls.refits = [], []
        for est in PLANO.estabilidade[cls.k]:
            cls.refits.append(chave(X[est.treino]))
            for d in PLANO.internos[(cls.k, est.fold_estabilidade)]:
                cls.internos.append(chave(X[d.treino]))

    def test_filtros_e_scaler_ajustados_dentro_de_cada_treino_interno(self):
        self.assertEqual(problemas_de_aninhamento(self.registros, self.internos, self.refits), [])
        n = ESPEC.n_folds_estabilidade * (ESPEC.n_folds_interna * len(GRADE) + 1)
        self.assertEqual({p: len(v) for p, v in self.registros.items()},
                         {"variancia": n, "correlacao": n, "scaler": n, "l1": n})

    def test_detector_acusa_filtros_ajustados_fora_da_cv_interna(self):
        # Variante do procedimento histórico (Seção 5.3): filtros ajustados no
        # treino de estabilidade inteiro ANTES da GridSearchCV. O detector deve acusá-la.
        est = PLANO.estabilidade[self.k][0]
        internas = PLANO.internos[(self.k, 0)]
        splits = [(sel.posicoes_locais(est.treino, d.treino), sel.posicoes_locais(est.treino, d.avaliacao))
                  for d in internas]
        with Espiao() as espiao:
            X_est = X[est.treino]
            filtros = tf.filtros_selecao().fit(X_est)
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(penalty=ESPEC.penalty, solver=ESPEC.solver,
                                                        max_iter=ESPEC.max_iter, random_state=ESPEC.random_state))])
            GridSearchCV(pipe, {"clf__C": GRADE}, cv=splits, scoring=ESPEC.scoring).fit(
                filtros.transform(X_est), AMOSTRA.y[est.treino])
        problemas = problemas_de_aninhamento(espiao.registros, [chave(X[d.treino]) for d in internas],
                                             [chave(X[est.treino])])
        self.assertTrue(any(p.startswith("variancia") for p in problemas), problemas)
        self.assertTrue(any(p.startswith("correlacao") for p in problemas), problemas)

    def test_nenhum_ajuste_ve_linhas_da_avaliacao_externa(self):
        externa = PLANO.externos[self.k]
        proibidas = chave(X[externa.avaliacao])
        for m in self.registros["variancia"]:
            self.assertFalse(chave(m) & proibidas)


class TestSemVazamentoExterno(unittest.TestCase):
    def test_avaliacao_externa_nunca_e_lida(self):
        # Linhas da avaliação externa trocadas por NaN, infinito e lixo: resultado idêntico.
        externa = PLANO.externos[0]
        for valor in (np.nan, 1e9):
            with self.subTest(valor=valor):
                X_alterado = X.copy()
                X_alterado[externa.avaliacao] = valor
                r = sel.selecionar_fold_externo(AMOSTRA, X_alterado, FEATURES, PLANO, 0)
                self.assertEqual(r.como_dict(), selecao_k0().como_dict())

    def test_grid_search_recebe_so_o_treino_de_estabilidade_em_posicoes_locais(self):
        chamadas = []
        original = GridSearchCV.fit

        def fit(busca, X_, y=None, **kwargs):
            chamadas.append((np.array(X_, copy=True), np.array(y, copy=True), list(busca.cv), kwargs))
            return original(busca, X_, y, **kwargs)

        k = 2
        with unittest.mock.patch.object(GridSearchCV, "fit", fit):
            sel.selecionar_fold_externo(AMOSTRA, X, FEATURES, PLANO, k)
        externa = PLANO.externos[k]
        pacientes_avaliacao = set(AMOSTRA.grupos[externa.avaliacao])
        self.assertEqual(len(chamadas), ESPEC.n_folds_estabilidade)
        for (X_, y_, cv, kwargs), est in zip(chamadas, PLANO.estabilidade[k]):
            np.testing.assert_array_equal(X_, X[est.treino])
            np.testing.assert_array_equal(y_, AMOSTRA.y[est.treino])
            self.assertEqual(kwargs, {})
            grupos = AMOSTRA.grupos[est.treino]
            self.assertFalse(pacientes_avaliacao & set(grupos))
            internas = PLANO.internos[(k, est.fold_estabilidade)]
            self.assertEqual(len(cv), len(internas))
            for (tr, te), d in zip(cv, internas):
                self.assertTrue(tr.max() < len(est.treino) and te.max() < len(est.treino))
                np.testing.assert_array_equal(est.treino[tr], d.treino)  # local → global
                np.testing.assert_array_equal(est.treino[te], d.avaliacao)
                self.assertFalse(set(grupos[tr]) & set(grupos[te]))  # isolamento por paciente

    def test_nao_finito_no_treino_externo_e_recusado(self):
        X_alterado = X.copy()
        X_alterado[PLANO.externos[0].treino[0], 2] = np.nan
        with self.assertRaisesRegex(sel.SelecaoInvalida, "não finitos"):
            sel.selecionar_fold_externo(AMOSTRA, X_alterado, FEATURES, PLANO, 0)


# --------------------------------------------------------------------------- #
# Determinismo, warnings e compatibilidade com os folds
# --------------------------------------------------------------------------- #


class TestDeterminismoEFolds(unittest.TestCase):
    def test_deterministico(self):
        amostra, X_ = sintetico()
        r = sel.selecionar_fold_externo(amostra, pd.DataFrame(X_, columns=FEATURES), list(FEATURES),
                                        fl.plano_repeticao(amostra, 0), 0)
        self.assertEqual(rst.json_canonico(r.como_dict()), rst.json_canonico(selecao_k0().como_dict()))

    def test_warnings_e_nao_convergencia_registrados(self):
        curta = dataclasses.replace(ESPEC, max_iter=1)
        with unittest.mock.patch.object(cfg, "ESPEC_SELECAO", curta):
            r = sel.selecionar_fold_externo(AMOSTRA, X, FEATURES, PLANO, 0)
        for f in r.folds:
            self.assertFalse(f.convergiu)
            categorias = {c: n for c, _, n in f.warnings}
            self.assertGreaterEqual(categorias.get("ConvergenceWarning", 0), 1)
        self.assertTrue(all(c.isidentifier() for f in r.folds for c, _, _ in f.warnings))
        self.assertTrue(all(f.warnings == () for f in selecao_k0().folds))

    def test_plano_de_outra_amostra_ou_forjado(self):
        outra, X_outra = sintetico(semente=12)
        with self.assertRaisesRegex(sel.SelecaoInvalida, "outra amostra"):
            sel.selecionar_fold_externo(AMOSTRA, X, FEATURES, fl.plano_repeticao(outra, 0), 0)
        # Mesma amostra, folds de estabilidade de outro fold externo: recusado.
        trocado = dict(PLANO.estabilidade)
        trocado[0], trocado[1] = PLANO.estabilidade[1], PLANO.estabilidade[0]
        forjado = fl.PlanoRepeticao(PLANO.repeticao, PLANO.sha256_amostra, PLANO.externos,
                                    trocado, PLANO.internos)
        with self.assertRaisesRegex(sel.SelecaoInvalida, "canônico"):
            sel.selecionar_fold_externo(AMOSTRA, X, FEATURES, forjado, 0)
        # Plano de outra repetição é aceito como tal e usa as divisões dela.
        self.assertEqual(sel.selecionar_fold_externo(AMOSTRA, X, FEATURES, fl.plano_repeticao(AMOSTRA, 1), 0)
                         .fingerprint_folds,
                         fl.fingerprint_unidade(AMOSTRA, fl.plano_repeticao(AMOSTRA, 1), 0))


class TestEntradasInvalidas(unittest.TestCase):
    def test_recusa(self):
        frame = pd.DataFrame(X, columns=FEATURES)
        casos = {
            "X com linha a menos": (X[:-1], FEATURES),
            "X com coluna a mais": (np.column_stack([X, X[:, 0]]), FEATURES),
            "X 1-D": (X[:, 0], FEATURES),
            "colunas reordenadas": (frame[list(FEATURES[::-1])], FEATURES),
            "colunas renomeadas": (frame.rename(columns={FEATURES[0]: "original_sint_x"}), FEATURES),
            "X texto": (X.astype(str), FEATURES),
            "feature sem prefixo": (X, ("sinal",) + FEATURES[1:]),
            "identificador como feature": (X, (cfg.COLUNA_PROIBIDA,) + FEATURES[1:]),
            "feature repetida": (X, (FEATURES[1],) + FEATURES[1:]),
            "sem features": (X[:, :0], ()),
        }
        for nome, (matriz, features) in casos.items():
            with self.subTest(caso=nome):
                with self.assertRaises(ValueError):
                    sel.selecionar_fold_externo(AMOSTRA, matriz, features, PLANO, 0)
        with self.assertRaises(TypeError):
            sel.selecionar_fold_externo(AMOSTRA, X, FEATURES[0], PLANO, 0)
        with self.assertRaises(TypeError):
            sel.selecionar_fold_externo((AMOSTRA.ids, AMOSTRA.y), X, FEATURES, PLANO, 0)
        for k in (-1, cfg.ESPEC_CV.n_folds_externos, True, "0", 0.0):
            with self.subTest(k=k):
                with self.assertRaises(sel.SelecaoInvalida):
                    sel.selecionar_fold_externo(AMOSTRA, X, FEATURES, PLANO, k)


# --------------------------------------------------------------------------- #
# Módulo
# --------------------------------------------------------------------------- #


class TestModulo(unittest.TestCase):
    def setUp(self):
        self.fonte = (REPO_ROOT / "src" / "incremental" / "selecao.py").read_text(encoding="utf-8")
        self.arvore = ast.parse(self.fonte)

    def test_parametros_nao_duplicados(self):
        for tipo, cientificos in ((int, {3, 4, 5, 13, 42, 100, 101, 5000}), (float, {1e-12, 0.9, 4.0, 2.0})):
            constantes = {n.value for n in ast.walk(self.arvore)
                          if isinstance(n, ast.Constant) and type(n.value) is tipo}
            self.assertFalse(constantes & cientificos, constantes)
        for nome in ("penalty", "solver", "max_iter", "random_state", "grade_C", "scoring",
                     "limiar_coeficiente", "limiar_estabilidade", "n_folds_estabilidade"):
            self.assertIn(f"espec.{nome}", self.fonte)

    def test_sem_leitura_de_dados_nem_procedimento_historico(self):
        importados = set()
        for no in ast.walk(self.arvore):
            if isinstance(no, ast.Import):
                importados.update(a.name for a in no.names)
            elif isinstance(no, ast.ImportFrom):
                importados.add(("." * no.level) + (no.module or ""))
        self.assertFalse({m for m in importados if "dados" in m or "scripts" in m or "reproduce" in m
                          or m in ("pandas", "pickle")}, importados)
        self.assertIn("NÃO reproduz seus resultados nem\nseus hashes", self.fonte)

    def test_configuracao_inalterada(self):
        from test_guardas import HASH_CONFIG_INCREMENTO_1
        self.assertEqual(cfg.hash_configuracao(), HASH_CONFIG_INCREMENTO_1)


if __name__ == "__main__":
    unittest.main()
