"""Testes do ajuste interno do benchmark S6 (``src/benchmark_s6/tuning.py``).

Só dados SINTÉTICOS em memória (``SINT-…``), sem arquivos e sem dados de
pacientes. A busca completa só roda para ``lr_tuning`` e ``svm`` (rápidas em
amostras pequenas); ``rf`` e ``xgb`` usam os dois primeiros pontos da grade,
substituídos só dentro do teste. Empates exatos e falhas usam um estimador
roteirizado. Nenhuma métrica externa é calculada, e nada é gravado.

O teste com o XGBoost real só roda se o XGBoost 2.1.4 estiver instalado.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/benchmark_s6 -v
"""

from __future__ import annotations

import ast
import dataclasses
import json
import sys
import unittest
import unittest.mock
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import threadpoolctl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[1]
for caminho in (REPO_ROOT, AQUI):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.benchmark_s6 import adaptadores_s4 as s4  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import dados  # noqa: E402
from src.benchmark_s6 import folds  # noqa: E402
from src.benchmark_s6 import modelos  # noqa: E402
from src.benchmark_s6 import tuning  # noqa: E402
from test_folds import amostra_sintetica, com  # noqa: E402

C = cfg.CONFIGURACAO
XGB_DISPONIVEL, XGB_MOTIVO = modelos.xgboost_disponivel()


def caracteristicas(amostra: s4.Amostra, semente: int = 0) -> np.ndarray:
    """Três características alinhadas à amostra, uma delas associada ao alvo."""
    rng = np.random.default_rng(semente)
    return np.column_stack([rng.lognormal(6, 1, amostra.n), rng.normal(15, 6, amostra.n) + 4 * amostra.y,
                            rng.uniform(0.3, 0.9, amostra.n)])


class Roteirizado(ClassifierMixin, BaseEstimator):
    """Estimador de pontuação controlada: ``qualidade`` escolhe a coluna pontuada.

    0 → ruído (coluna 2); 1 ou 2 → coluna 1 (associada ao alvo), com AUC
    idêntica entre si; 3 → coluna 1 ao contrário; ``nan``/``inf`` → saída
    inválida; ``falha`` → exceção no ajuste.
    """

    def __init__(self, qualidade: Any = 0, aviso: bool = False):
        self.qualidade = qualidade
        self.aviso = aviso

    def fit(self, X, y):
        if self.qualidade == "falha":
            raise RuntimeError("ajuste roteirizado falhou")
        if self.aviso:
            warnings.warn("roteirizado não convergiu", ConvergenceWarning)
        self.classes_ = np.array([0, 1])
        self.n_ajustado_ = len(y)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.qualidade == "nan":
            return np.full(len(X), np.nan)
        if self.qualidade == "inf":
            return np.where(np.arange(len(X)) % 2 == 0, np.inf, 0.0)
        if self.qualidade in (1, 2):
            return X[:, 1].copy()
        if self.qualidade == 3:
            return -X[:, 1]
        return X[:, 2].copy()


class BaseTuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amostra = amostra_sintetica(90)
        cls.X = caracteristicas(cls.amostra)
        cls.plano = folds.gerar_plano(cls.amostra, "implementacao")
        cls.dobra = cls.plano.dobra(0, 0)
        cls.treino = cls.dobra.externa.treino
        cls.avaliacao = cls.dobra.externa.avaliacao

    def ajustar(self, modelo: str, X=None, dobra=None):
        return tuning.ajustar(self.X if X is None else X, self.amostra, self.dobra if dobra is None else dobra,
                              modelo)

    def buscar(self, estimador, grade: List[Dict[str, List[Any]]]):
        configuracoes = tuple({k: v[0] for k, v in p.items()} for p in grade)
        Xt, yt = self.X[self.treino], self.amostra.y[self.treino]
        return tuning._executar(lambda: tuning._buscar(estimador, grade, configuracoes, Xt, yt,
                                                       self.dobra.cv_interno()), tuning.FASE_BUSCA)


def grade_reduzida(modelo: str, n: int = 2):
    """Substitui, só no teste, a grade de ``modelo`` pelos seus ``n`` primeiros pontos."""
    grade = modelos.grade_busca(modelo)[:n]
    configs = modelos.configuracoes(modelo)[:n]
    return unittest.mock.patch.multiple(tuning.modelos, grade_busca=lambda m: [dict(p) for p in grade],
                                        configuracoes=lambda m: configs)


class CapturaGrid(GridSearchCV):
    """``GridSearchCV`` que registra argumentos e matriz recebidos."""

    chamadas: List[Dict[str, Any]] = []

    def fit(self, X, y=None, **kwargs):
        CapturaGrid.chamadas.append({"X": np.array(X, copy=True), "y": np.array(y, copy=True),
                                     "params": self.get_params(deep=False)})
        return super().fit(X, y, **kwargs)


# --------------------------------------------------------------------------- #
# Regra de escolha
# --------------------------------------------------------------------------- #


class TestEscolha(unittest.TestCase):
    def test_maior_media(self):
        self.assertEqual(tuning.escolher_configuracao([0.8, 0.9, 0.85]), 1)
        self.assertEqual(tuning.escolher_configuracao([0.95]), 0)

    def test_empate_exato_vai_para_a_mais_simples(self):
        self.assertEqual(tuning.escolher_configuracao([0.8, 0.9, 0.9]), 1)
        self.assertEqual(tuning.escolher_configuracao([0.9, 0.9, 0.9]), 0)

    def test_empate_dentro_da_tolerancia(self):
        self.assertEqual(C.ajuste_interno.tolerancia_empate, 1e-9)
        self.assertEqual(tuning.escolher_configuracao([0.5, 0.5 + 2 ** -30, 0.4]), 0)  # ≈ 9,3e-10
        self.assertEqual(tuning.escolher_configuracao([0.7, 0.9, 0.9 + 5e-10]), 1)
        self.assertEqual(tuning.escolher_configuracao([0.9 - 9e-10, 0.9]), 0)

    def test_limite_da_tolerancia_e_inclusivo(self):
        self.assertEqual(tuning.escolher_configuracao([0.0, 1e-9]), 0)

    def test_fora_da_tolerancia(self):
        self.assertEqual(tuning.escolher_configuracao([0.5, 0.5 + 2 ** -29]), 1)  # ≈ 1,9e-9
        self.assertEqual(tuning.escolher_configuracao([0.9 - 1.5e-9, 0.9]), 1)
        self.assertEqual(tuning.escolher_configuracao([0.9, 0.9 + 2e-9, 0.9 + 2.5e-9]), 1)

    def test_medias_invalidas(self):
        for medias in ([0.8, np.nan], [np.inf, 0.5], [-np.inf], [], [[0.8, 0.9]]):
            with self.subTest(medias=medias), self.assertRaises(tuning.FalhaAjuste):
                tuning.escolher_configuracao(medias)

    def test_refit_confere_ordem_e_pontuacoes(self):
        configs = ({"c": 1}, {"c": 2})
        escolher = tuning._refit(configs, 2)
        base = {"params": [{"c": 1}, {"c": 2}], "split0_test_score": [0.7, 0.9], "split1_test_score": [0.9, 0.7],
                "mean_test_score": [0.0, 1.0]}
        # Média pelas dobras (0,8 e 0,8): empate → a primeira, ignorando um mean_test_score adulterado.
        self.assertEqual(escolher(base), 0)
        casos = {
            "ordem": {**base, "params": [{"c": 2}, {"c": 1}]},
            "NaN": {**base, "split1_test_score": [0.9, np.nan]},
            "inf": {**base, "split0_test_score": [np.inf, 0.9]},
            "dobra ausente": {k: v for k, v in base.items() if k != "split1_test_score"},
            "dobra a mais": {**base, "split2_test_score": [0.5, 0.5]},
            "forma": {**base, "split0_test_score": [0.7, 0.9, 0.1]},
        }
        for nome, resultados in casos.items():
            with self.subTest(nome), self.assertRaises(tuning.FalhaAjuste):
                escolher(resultados)


# --------------------------------------------------------------------------- #
# GridSearchCV: parâmetros, ordem, seleção e reajuste
# --------------------------------------------------------------------------- #


class TestBusca(BaseTuning):
    def test_parametros_exatos_da_p9_e_cinco_dobras_internas(self):
        CapturaGrid.chamadas = []
        with unittest.mock.patch.object(tuning, "GridSearchCV", CapturaGrid):
            self.ajustar("lr_tuning")
        self.assertEqual(len(CapturaGrid.chamadas), 1)
        params = CapturaGrid.chamadas[0]["params"]
        self.assertEqual((params["scoring"], params["n_jobs"], params["error_score"], params["return_train_score"]),
                         ("roc_auc", 1, "raise", False))
        self.assertTrue(callable(params["refit"]))
        self.assertEqual(params["param_grid"], modelos.grade_busca("lr_tuning"))
        cv = params["cv"]
        self.assertEqual(len(cv), 5)
        pacientes_aval = set(self.amostra.grupos[self.avaliacao].tolist())
        for (tr, te), interna in zip(cv, self.dobra.internas):
            np.testing.assert_array_equal(self.treino[tr], interna.treino)
            np.testing.assert_array_equal(self.treino[te], interna.avaliacao)
            globais = self.treino[np.concatenate([tr, te])]
            self.assertEqual(set(self.amostra.grupos[globais].tolist()) & pacientes_aval, set())
            self.assertEqual(set(self.amostra.grupos[self.treino[tr]]) & set(self.amostra.grupos[self.treino[te]]),
                             set())

    def test_so_o_treino_externo_entra_na_busca(self):
        CapturaGrid.chamadas = []
        with unittest.mock.patch.object(tuning, "GridSearchCV", CapturaGrid):
            self.ajustar("svm")
        chamada = CapturaGrid.chamadas[0]
        np.testing.assert_array_equal(chamada["X"], self.X[self.treino])
        np.testing.assert_array_equal(chamada["y"], self.amostra.y[self.treino])

    def test_avaliacao_externa_nunca_lida(self):
        """Avaliação externa com NaN e infinito: o ajuste não muda (e falharia se as linhas fossem usadas)."""
        X = self.X.copy()
        X[self.avaliacao] = np.nan
        X[self.avaliacao[::2], 1] = np.inf
        for modelo in ("lr_fixa", "lr_tuning"):
            normal, sujo = self.ajustar(modelo), self.ajustar(modelo, X=X)
            self.assertEqual(normal.registro(), sujo.registro())
            self.assertEqual(normal.medias_internas, sujo.medias_internas)
            np.testing.assert_array_equal(normal.estimador.decision_function(self.X[self.treino]),
                                          sujo.estimador.decision_function(self.X[self.treino]))

    def test_selecao_exata_conferida_por_recalculo_independente(self):
        r = self.ajustar("lr_tuning")
        Xt, yt = self.X[self.treino], self.amostra.y[self.treino]
        medias = []
        for config in modelos.configuracoes("lr_tuning"):
            aucs = []
            for tr, te in self.dobra.cv_interno():
                est = modelos.criar_estimador("lr_tuning").set_params(**config).fit(Xt[tr], yt[tr])
                aucs.append(roc_auc_score(yt[te], est.decision_function(Xt[te])))
            medias.append(np.mean(aucs))
        np.testing.assert_allclose(r.medias_internas, medias, rtol=0, atol=1e-15)
        esperado = int(np.flatnonzero(np.array(medias) >= max(medias) - 1e-9)[0])
        self.assertEqual(r.indice_configuracao, esperado)
        self.assertEqual(dict(r.configuracao), {"C": modelos.configuracoes("lr_tuning")[esperado]["classificador__C"]})

    def test_empate_exato_no_gridsearchcv_escolhe_a_mais_simples(self):
        busca, _ = self.buscar(Roteirizado(), [{"qualidade": [0]}, {"qualidade": [1]}, {"qualidade": [2]},
                                               {"qualidade": [3]}])
        medias = busca.cv_results_["mean_test_score"]
        self.assertEqual(medias[1], medias[2])
        self.assertGreater(medias[1], max(medias[0], medias[3]))
        self.assertEqual(busca.best_index_, 1)
        self.assertEqual(busca.best_estimator_.qualidade, 1)

    def test_reajuste_unico_no_treino_externo_inteiro(self):
        ajustes: List[int] = []
        original = Pipeline.fit

        def contar(self_, X, y=None, **kw):
            ajustes.append(len(X))
            return original(self_, X, y, **kw)

        with unittest.mock.patch.object(Pipeline, "fit", contar):
            r = self.ajustar("lr_tuning")
        self.assertEqual(len(ajustes), 13 * 5 + 1)
        self.assertEqual(r.n_ajustes, 13 * 5 + 1)
        self.assertEqual(ajustes[-1], len(self.treino))
        self.assertEqual(ajustes.count(len(self.treino)), 1)
        escala = r.estimador.named_steps["padronizacao"]
        np.testing.assert_allclose(escala.mean_, self.X[self.treino].mean(axis=0))
        novo = modelos.criar_estimador("lr_tuning").set_params(
            classificador__C=r.configuracao["C"]).fit(self.X[self.treino], self.amostra.y[self.treino])
        np.testing.assert_array_equal(novo.decision_function(self.X[self.treino]),
                                      r.estimador.decision_function(self.X[self.treino]))

    def test_rf_com_grade_reduzida(self):
        with grade_reduzida("rf"):
            r = self.ajustar("rf")
        self.assertIn(r.indice_configuracao, (0, 1))
        self.assertEqual(r.estimador.n_estimators, 500)
        self.assertEqual(dict(r.configuracao), dict(modelos.configuracoes("rf")[r.indice_configuracao]))
        self.assertEqual(r.borda["min_samples_leaf"].posicao, 0)
        self.assertTrue(r.borda["min_samples_leaf"].na_borda)


# --------------------------------------------------------------------------- #
# Registro, borda e avisos
# --------------------------------------------------------------------------- #


class TestRegistro(BaseTuning):
    def test_registro_sem_identificadores_nem_pontuacoes(self):
        r = self.ajustar("svm")
        registro = r.registro()
        texto = json.dumps(registro, ensure_ascii=False)
        self.assertNotIn("SINT-", texto)
        self.assertNotIn(self.amostra.sha256, texto)
        self.assertEqual(set(registro), {"modelo", "repeticao", "dobra_externa", "fingerprint_dobra", "configuracao",
                                         "indice_configuracao", "n_configuracoes", "borda", "na_borda_da_grade",
                                         "n_ajustes", "avisos", "avisos_convergencia"})
        for media in r.medias_internas:
            self.assertNotIn(repr(media), texto)
        self.assertEqual(registro["fingerprint_dobra"], self.dobra.fingerprint)
        self.assertEqual((registro["repeticao"], registro["dobra_externa"]), (0, 0))
        self.assertEqual(registro["n_configuracoes"], 30)
        self.assertEqual(registro["n_ajustes"], 30 * 5 + 1)
        self.assertEqual(len(r.medias_internas), 30)
        self.assertEqual(s4.json_canonico(registro), s4.json_canonico(json.loads(texto)))

    def test_posicao_e_borda_de_cada_hiperparametro(self):
        r = self.ajustar("svm")
        config = modelos.configuracoes("svm")[r.indice_configuracao]
        self.assertEqual(dict(r.configuracao),
                         {"C": config["classificador__C"], "gamma": config["classificador__gamma"]})
        espec = modelos.especificacao("svm")
        for nome, borda in r.borda.items():
            valores = espec.valores_de(nome)
            self.assertEqual(borda.posicao, valores.index(r.configuracao[nome]))
            self.assertEqual(borda.na_borda, borda.posicao in (0, len(valores) - 1))
        self.assertEqual(r.registro()["na_borda_da_grade"], any(b.na_borda for b in r.borda.values()))

    def test_borda_nos_extremos_e_no_interior(self):
        espec = modelos.especificacao("xgb")
        borda = tuning._borda(espec, {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.1,
                                      "min_child_weight": 5})
        self.assertEqual({n: (b.posicao, b.na_borda) for n, b in borda.items()},
                         {"max_depth": (3, True), "n_estimators": (0, True), "learning_rate": (1, True),
                          "min_child_weight": (0, True)})
        borda = tuning._borda(modelos.especificacao("rf"), {"min_samples_leaf": 5, "max_features": 2})
        self.assertEqual({n: b.na_borda for n, b in borda.items()}, {"min_samples_leaf": False, "max_features": False})
        borda = tuning._borda(modelos.especificacao("lr_tuning"), {"C": float(np.logspace(-3, 3, 13)[0])})
        self.assertTrue(borda["C"].na_borda)

    def test_avisos_de_convergencia_registrados_mesmo_com_filtro_ignore(self):
        original = Pipeline.fit

        def com_aviso(self_, X, y=None, **kw):
            warnings.warn("lbfgs failed to converge (teste)", ConvergenceWarning)
            return original(self_, X, y, **kw)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with unittest.mock.patch.object(Pipeline, "fit", com_aviso):
                busca = self.ajustar("lr_tuning")
                fixa = self.ajustar("lr_fixa")
        self.assertEqual(busca.avisos_convergencia, 66)
        self.assertEqual([(a.fase, a.categoria, a.ocorrencias) for a in busca.avisos if a.convergencia],
                         [("busca", "ConvergenceWarning", 66)])
        self.assertEqual([(a.fase, a.ocorrencias) for a in fixa.avisos], [("ajuste", 1)])
        self.assertEqual(fixa.registro()["avisos"][0]["mensagem"], "lbfgs failed to converge (teste)")

    def test_aviso_com_filtro_error_nao_vira_excecao_nem_some(self):
        busca_ok, avisos = None, None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            busca_ok, avisos = self.buscar(Roteirizado(aviso=True), [{"qualidade": [1]}, {"qualidade": [0]}])
        self.assertEqual(busca_ok.best_index_, 0)
        self.assertEqual(sum(a.ocorrencias for a in avisos if a.convergencia), 2 * 5 + 1)


# --------------------------------------------------------------------------- #
# Falhas técnicas e entradas
# --------------------------------------------------------------------------- #


class TestFalhas(BaseTuning):
    def test_pontuacao_interna_nan_ou_infinita(self):
        for qualidade in ("nan", "inf"):
            with self.subTest(qualidade), self.assertRaises(tuning.FalhaAjuste):
                self.buscar(Roteirizado(), [{"qualidade": [0]}, {"qualidade": [qualidade]}])

    def test_excecao_no_ajuste(self):
        with self.assertRaisesRegex(tuning.FalhaAjuste, "ajuste roteirizado falhou") as ctx:
            self.buscar(Roteirizado(), [{"qualidade": [0]}, {"qualidade": ["falha"]}])
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

        def falha(self_, X, y=None, **kw):
            raise FloatingPointError("explodiu")

        for modelo in ("lr_fixa", "lr_tuning"):
            with self.subTest(modelo), unittest.mock.patch.object(Pipeline, "fit", falha):
                with self.assertRaisesRegex(tuning.FalhaAjuste, "FloatingPointError"):
                    self.ajustar(modelo)

    def test_escolha_divergente_do_gridsearchcv(self):
        class Desobediente(GridSearchCV):
            def fit(self, X, y=None, **kw):
                super().fit(X, y, **kw)
                self.best_index_ = (self.best_index_ + 1) % len(self.cv_results_["params"])
                return self

        with unittest.mock.patch.object(tuning, "GridSearchCV", Desobediente):
            with self.assertRaisesRegex(tuning.FalhaAjuste, "difere da regra"):
                self.ajustar("lr_tuning")

    def test_reajuste_com_configuracao_diferente_da_escolhida(self):
        class TrocaReajuste(GridSearchCV):
            def fit(self, X, y=None, **kw):
                super().fit(X, y, **kw)
                outro = self.cv_results_["params"][(self.best_index_ + 1) % len(self.cv_results_["params"])]
                self.best_estimator_.set_params(**outro)
                return self

        with unittest.mock.patch.object(tuning, "GridSearchCV", TrocaReajuste):
            with self.assertRaisesRegex(tuning.FalhaAjuste, "configuração diferente da escolhida"):
                self.ajustar("lr_tuning")

    def test_estimador_final_sem_as_duas_classes_ou_nao_ajustado(self):
        class ClassesTrocadas(Roteirizado):
            def fit(self, X, y):
                super().fit(X, y)
                self.classes_ = np.array([1, 0])
                return self

        class NaoAjusta(Roteirizado):
            def fit(self, X, y):
                return self

        for estimador, mensagem in ((ClassesTrocadas(), "classes 0 e 1"), (NaoAjusta(), "não ajustado")):
            with self.subTest(type(estimador).__name__), \
                    unittest.mock.patch.object(tuning.modelos, "criar_estimador", lambda m, e=estimador: e):
                with self.assertRaisesRegex(tuning.FalhaAjuste, mensagem):
                    self.ajustar("lr_fixa")

    def test_entradas_desalinhadas_ou_invalidas(self):
        n, p = self.X.shape
        casos = {
            "linha a menos": self.X[:-1],
            "linha a mais": np.vstack([self.X, self.X[:1]]),
            "duas características": self.X[:, :2],
            "inteiro": self.X.astype(np.int64),
            "float32": self.X.astype(np.float32),
            "1D": self.X[:, 0],
        }
        for nome, X in casos.items():
            with self.subTest(nome), self.assertRaises(tuning.FalhaAjuste):
                self.ajustar("lr_fixa", X=X)
        X = self.X.copy()
        X[self.treino[3], 0] = np.nan
        with self.assertRaisesRegex(tuning.FalhaAjuste, "treino externo"):
            self.ajustar("lr_fixa", X=X)
        with self.assertRaises(TypeError):
            self.ajustar("lr_fixa", X=self.X.tolist())
        with self.assertRaises(TypeError):
            tuning.ajustar(self.X, {"n": n}, self.dobra, "lr_fixa")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.ajustar("lr_fixa", dobra=self.plano)
        with self.assertRaises(modelos.ModeloInvalido):
            self.ajustar("lr")

    def test_dobra_revalidada_pela_api_publica_de_folds(self):
        with unittest.mock.patch.object(tuning.folds, "validar_dobra", wraps=folds.validar_dobra) as validar:
            self.ajustar("lr_fixa")
        validar.assert_called_once_with(self.amostra, self.dobra)

    def test_dobra_de_outra_amostra_ou_adulterada(self):
        """A revalidação de ``folds`` recusa antes de qualquer ajuste; a exceção dela chega intacta."""
        outra = amostra_sintetica(90, semente=5)
        with self.assertRaisesRegex(folds.DobrasInvalidas, "outra amostra"):
            tuning.ajustar(caracteristicas(outra), outra, self.dobra, "lr_fixa")
        d = self.dobra
        intrusa = int(d.externa.avaliacao[0])
        vazada = com(d.internas[0], avaliacao=np.sort(np.append(d.internas[0].avaliacao, intrusa)))
        vazada.avaliacao.setflags(write=False)
        movida = com(d.externa, treino=d.externa.treino[1:], avaliacao=np.sort(np.append(d.externa.avaliacao,
                                                                                        d.externa.treino[0])))
        for arr in (movida.treino, movida.avaliacao):
            arr.setflags(write=False)
        casos = {
            "fingerprint": com(d, fingerprint="0" * 64),
            "semente externa": com(d, externa=com(d.externa, semente=43)),
            "semente interna": com(d, internas=(com(d.internas[0], semente=1001), *d.internas[1:])),
            "internas a menos": com(d, internas=d.internas[:4]),
            "vazamento da avaliação externa": com(d, internas=(vazada, *d.internas[1:])),
            "linha movida do treino para a avaliação": com(d, externa=movida),
            "repetição fora do plano": com(d, externa=com(d.externa, repeticao=10)),
        }
        for nome, dobra in casos.items():
            esperada = folds.PlanoDivergente if nome == "fingerprint" else folds.DobrasInvalidas
            with self.subTest(nome), self.assertRaises(esperada):
                self.ajustar("lr_fixa", dobra=dobra)

    def test_dobra_forjada_com_fingerprint_coerente(self):
        """Adulteração com o fingerprint recalculado: só a revalidação das divisões pega."""
        d = self.dobra

        def coerente(dobra):
            return com(dobra, fingerprint=folds._fingerprint_dobra(self.amostra, dobra.externa, dobra.internas))

        intrusa = int(d.externa.avaliacao[0])
        avaliacao = np.sort(np.append(d.internas[0].avaliacao, intrusa))
        avaliacao.setflags(write=False)
        vazada = coerente(com(d, internas=(com(d.internas[0], avaliacao=avaliacao), *d.internas[1:])))
        repetida = coerente(com(d, internas=(d.internas[0], com(d.internas[0], dobra_interna=1), *d.internas[2:])))
        externa_sobreposta = np.sort(np.append(d.externa.avaliacao, int(d.externa.treino[0])))
        externa_sobreposta.setflags(write=False)
        sobreposta = coerente(com(d, externa=com(d.externa, avaliacao=externa_sobreposta)))
        for nome, dobra in (("vazamento da avaliação externa", vazada), ("dobra interna repetida", repetida),
                            ("treino e avaliação externos sobrepostos", sobreposta)):
            with self.subTest(nome), self.assertRaises(folds.DobrasInvalidas):
                self.ajustar("lr_tuning", dobra=dobra)


# --------------------------------------------------------------------------- #
# LR fixa, reprodutibilidade, threads e XGBoost
# --------------------------------------------------------------------------- #


class TestLRFixaEReprodutibilidade(BaseTuning):
    def test_lr_fixa_sem_busca(self):
        def proibido(*a, **k):
            raise AssertionError("lr_fixa não usa GridSearchCV")

        with unittest.mock.patch.object(tuning, "GridSearchCV", proibido):
            r = self.ajustar("lr_fixa")
        self.assertEqual((dict(r.configuracao), r.indice_configuracao, r.n_configuracoes, r.n_ajustes, dict(r.borda),
                          r.medias_internas), ({}, 0, 1, 1, {}, ()))
        self.assertEqual(r.estimador.named_steps["classificador"].C, 1.0)
        novo = modelos.criar_estimador("lr_fixa").fit(self.X[self.treino], self.amostra.y[self.treino])
        np.testing.assert_array_equal(novo.named_steps["classificador"].coef_,
                                      r.estimador.named_steps["classificador"].coef_)
        self.assertFalse(r.registro()["na_borda_da_grade"])

    def test_reprodutivel_bit_a_bit(self):
        for modelo in ("lr_fixa", "lr_tuning", "svm"):
            a, b = self.ajustar(modelo), self.ajustar(modelo)
            self.assertEqual(a.registro(), b.registro(), modelo)
            self.assertEqual(a.medias_internas, b.medias_internas, modelo)
            np.testing.assert_array_equal(a.estimador.decision_function(self.X[self.treino]),
                                          b.estimador.decision_function(self.X[self.treino]))

    def test_dobras_diferentes_dao_resultados_rastreaveis(self):
        r1 = tuning.ajustar(self.X, self.amostra, self.plano.dobra(0, 1), "lr_tuning")
        self.assertEqual((r1.repeticao, r1.dobra, r1.fingerprint_dobra), (0, 1, self.plano.dobra(0, 1).fingerprint))

    def test_ajustar_dados_usa_x_e_amostra_da_mesma_carga(self):
        import pandas as pd

        a = self.amostra
        frame = pd.DataFrame({cfg.COLUNA_NODULO: a.ids, cfg.COLUNA_PACIENTE: a.grupos,
                              dados.COLUNA_PARTICAO: "train", cfg.COLUNA_ALVO: a.y})
        for i, nome in enumerate(C.caracteristicas):
            frame[nome] = self.X[:, i]
        carga = dados.DadosBenchmark("implementacao", ("train",), False, tuple(C.caracteristicas), {}, {}, None, frame)
        self.assertEqual(tuning.ajustar_dados(carga, self.dobra, "lr_tuning").registro(),
                         self.ajustar("lr_tuning").registro())
        with self.assertRaises(TypeError):
            tuning.ajustar_dados(self.X, self.dobra, "lr_tuning")  # type: ignore[arg-type]

    def test_blas_limitado_a_uma_thread_durante_os_ajustes(self):
        vistos: List[int] = []
        original = Pipeline.fit

        def observar(self_, X, y=None, **kw):
            vistos.extend(info["num_threads"] for info in threadpoolctl.threadpool_info())
            return original(self_, X, y, **kw)

        antes = [info["num_threads"] for info in threadpoolctl.threadpool_info()]
        with unittest.mock.patch.object(Pipeline, "fit", observar):
            self.ajustar("lr_tuning")
        self.assertTrue(vistos)
        self.assertEqual(set(vistos), {1})
        self.assertEqual([info["num_threads"] for info in threadpoolctl.threadpool_info()], antes)

    @unittest.skipUnless(XGB_DISPONIVEL, f"XGBoost real não testado no ajuste: {XGB_MOTIVO}")
    def test_xgboost_real_com_grade_reduzida(self):  # pragma: no cover - depende do ambiente
        with grade_reduzida("xgb"):
            a, b = self.ajustar("xgb"), self.ajustar("xgb")
        self.assertEqual(a.registro(), b.registro())
        self.assertEqual(a.medias_internas, b.medias_internas)
        self.assertEqual(type(a.estimador).__name__, "XGBClassifier")
        np.testing.assert_array_equal(a.estimador.predict_proba(self.X[self.treino]),
                                      b.estimador.predict_proba(self.X[self.treino]))


# --------------------------------------------------------------------------- #
# Isolamento do módulo
# --------------------------------------------------------------------------- #


class TestIsolamento(unittest.TestCase):
    def setUp(self):
        self.fonte = (REPO_ROOT / "src" / "benchmark_s6" / "tuning.py").read_text(encoding="utf-8")
        self.arvore = ast.parse(self.fonte)

    def test_importa_so_o_permitido(self):
        modulos_importados = set()
        for n in ast.walk(self.arvore):
            if isinstance(n, ast.Import):
                modulos_importados |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                modulos_importados.add(("." * n.level) + (n.module or ""))
        self.assertEqual(modulos_importados, {"__future__", "warnings", "collections", "dataclasses", "types",
                                              "typing", "numpy", "sklearn.model_selection",
                                              "sklearn.utils.validation", "threadpoolctl", ".", ".dados"})

    def test_nao_prediz_nao_mede_nao_grava(self):
        proibidos = {"predict", "predict_proba", "decision_function", "score", "roc_auc_score", "open", "write_text",
                     "write_bytes", "dump", "to_csv", "read_csv", "read_bytes", "read_text", "savez", "save"}
        nomes = {n.id for n in ast.walk(self.arvore) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(self.arvore) if isinstance(n, ast.Attribute)}
        self.assertEqual(nomes & proibidos, set())

    def test_avaliacao_externa_nunca_referenciada(self):
        """O ajuste nem menciona a avaliação externa: só o treino e ``cv_interno`` da dobra."""
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == "avaliacao" for n in ast.walk(self.arvore)))

    def test_so_a_api_publica_de_folds(self):
        privados = sorted({n.attr for n in ast.walk(self.arvore) if isinstance(n, ast.Attribute)
                           and isinstance(n.value, ast.Name) and n.value.id in ("folds", "modelos", "s4", "cfg")
                           and n.attr.startswith("_")})
        self.assertEqual(privados, [])
        usados = {n.attr for n in ast.walk(self.arvore) if isinstance(n, ast.Attribute)
                  and isinstance(n.value, ast.Name) and n.value.id == "folds"}
        self.assertEqual(usados, {"DobraExterna", "validar_dobra"})

    def test_parametros_do_gridsearchcv_vem_da_configuracao(self):
        busca = next(f for f in self.arvore.body if isinstance(f, ast.FunctionDef) and f.name == "_buscar")
        chamada = next(n for n in ast.walk(busca) if isinstance(n, ast.Call) and getattr(n.func, "id", "") ==
                       "GridSearchCV")
        self.assertEqual({k.arg for k in chamada.keywords},
                         {"param_grid", "scoring", "cv", "refit", "n_jobs", "error_score", "return_train_score"})
        self.assertFalse(any(isinstance(k.value, ast.Constant) for k in chamada.keywords))

    def test_resultado_congelado(self):
        r = dataclasses.fields(tuning.ResultadoAjuste)
        self.assertIn("medias_internas", {f.name for f in r})
        self.assertTrue(tuning.ResultadoAjuste.__dataclass_params__.frozen)


if __name__ == "__main__":
    unittest.main()
