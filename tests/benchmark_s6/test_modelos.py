"""Testes da fábrica de classificadores do benchmark S6 (``src/benchmark_s6/modelos.py``).

Só dados SINTÉTICOS em memória (``numpy``), sem nenhum arquivo e sem dados de
pacientes. Os ajustes aqui servem para conferir contratos dos estimadores
(forma das saídas, determinismo, compatibilidade de nomes com o
``GridSearchCV``); nenhum resultado é científico nem é gravado.

O XGBoost 2.1.4 é testado de duas formas:

* contratos da fábrica com um módulo ``xgboost`` FALSO injetado em
  ``sys.modules`` (versão, parâmetros, ausência de parada antecipada e de
  paralelismo), sem executar o XGBoost de verdade;
* o teste de fumaça REAL só roda se o XGBoost 2.1.4 estiver instalado; caso
  contrário fica marcado como pulado, e NÃO conta como aprovado.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/benchmark_s6 -v
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import sys
import types
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import modelos  # noqa: E402

C = cfg.CONFIGURACAO
SKLEARN = ("lr_fixa", "lr_tuning", "svm", "rf")
CLASSES = {"lr_fixa": LogisticRegression, "lr_tuning": LogisticRegression, "svm": SVC,
           "rf": RandomForestClassifier}
XGB_DISPONIVEL, XGB_MOTIVO = modelos.xgboost_disponivel()


def dados_sinteticos(n: int = 120, semente: int = 0):
    """Três características contínuas em escalas muito diferentes e alvo 0/1 dependente delas."""
    rng = np.random.default_rng(semente)
    X = np.column_stack([rng.lognormal(6, 1, n), rng.normal(15, 6, n), rng.uniform(0.3, 0.9, n)])
    logito = 0.002 * (X[:, 0] - 400) + 0.2 * (X[:, 1] - 15) + rng.normal(0, 1, n)
    y = (logito > 0).astype(np.int64)
    return X, y


# --------------------------------------------------------------------------- #
# XGBoost falso (só para os contratos da fábrica)
# --------------------------------------------------------------------------- #


def modulo_xgboost_falso(versao: Any = "2.1.4", early_stopping_padrao: Any = None, ignora_n_jobs: bool = False):
    class XGBClassifier(ClassifierMixin, BaseEstimator):
        def __init__(self, objective=None, tree_method=None, subsample=None, colsample_bytree=None,
                     reg_lambda=None, n_jobs=None, random_state=None, max_depth=None, n_estimators=None,
                     learning_rate=None, min_child_weight=None, early_stopping_rounds=early_stopping_padrao):
            self.objective = objective
            self.tree_method = tree_method
            self.subsample = subsample
            self.colsample_bytree = colsample_bytree
            self.reg_lambda = reg_lambda
            self.n_jobs = 8 if ignora_n_jobs else n_jobs
            self.random_state = random_state
            self.max_depth = max_depth
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.min_child_weight = min_child_weight
            self.early_stopping_rounds = early_stopping_rounds

        def fit(self, X, y):  # pragma: no cover - nunca chamado
            raise AssertionError("o XGBoost falso não deve ser ajustado")

    modulo = types.ModuleType("xgboost")
    if versao is not None:
        modulo.__version__ = versao
    modulo.XGBClassifier = XGBClassifier
    return modulo


class _FalhaAoCarregar:
    """Localizador que simula a DLL nativa bloqueada ao importar ``xgboost``."""

    def find_spec(self, nome, caminho=None, alvo=None):
        if nome == "xgboost":
            raise OSError("[WinError 4551] Uma política de Controle de Aplicativo bloqueou este arquivo")
        return None


class _Registro:
    """Localizador que só registra tentativas de importar ``xgboost``."""

    def __init__(self):
        self.tentativas = 0

    def find_spec(self, nome, caminho=None, alvo=None):
        if nome == "xgboost":
            self.tentativas += 1
        return None


@contextlib.contextmanager
def sem_xgboost_importado():
    """``xgboost`` fora de ``sys.modules`` durante o bloco, para o ``import`` passar pelos localizadores."""
    with unittest.mock.patch.dict(sys.modules):
        for nome in [n for n in sys.modules if n == "xgboost" or n.startswith("xgboost.")]:
            del sys.modules[nome]
        yield


# --------------------------------------------------------------------------- #
# Configuração e grades
# --------------------------------------------------------------------------- #


class TestConfiguracao(unittest.TestCase):
    def test_codigos_e_especificacoes(self):
        self.assertEqual(modelos.codigos(), ("lr_fixa", "lr_tuning", "svm", "rf", "xgb"))
        for codigo in modelos.codigos():
            self.assertIs(modelos.especificacao(codigo), next(m for m in C.modelos if m.codigo == codigo))
        for codigo in ("lr", "XGB", "", None, "lr_fixa "):
            with self.assertRaises(modelos.ModeloInvalido):
                modelos.especificacao(codigo)  # type: ignore[arg-type]

    def test_tamanho_e_ordem_das_grades(self):
        tamanhos = {c: len(modelos.configuracoes(c)) for c in modelos.codigos()}
        self.assertEqual(tamanhos, {"lr_fixa": 1, "lr_tuning": 13, "svm": 30, "rf": 15, "xgb": 32})
        cs = [c["classificador__C"] for c in modelos.configuracoes("lr_tuning")]
        np.testing.assert_array_equal(cs, np.logspace(-3, 3, 13))
        self.assertEqual(cs, sorted(cs))
        svm = modelos.configuracoes("svm")
        self.assertEqual([dict(x) for x in svm[:6]],
                         [{"classificador__C": 0.01, "classificador__gamma": g} for g in (0.01, 0.03, 0.1, 0.3, 1.0)]
                         + [{"classificador__C": 0.1, "classificador__gamma": 0.01}])
        self.assertEqual(dict(svm[-1]), {"classificador__C": 1000.0, "classificador__gamma": 1.0})
        rf = modelos.configuracoes("rf")
        self.assertEqual([dict(x) for x in rf[:4]],
                         [{"min_samples_leaf": 20, "max_features": f} for f in (1, 2, 3)]
                         + [{"min_samples_leaf": 10, "max_features": 1}])
        self.assertEqual([x["min_samples_leaf"] for x in rf[::3]], [20, 10, 5, 3, 1])
        xgb = modelos.configuracoes("xgb")
        self.assertEqual(dict(xgb[0]), {"max_depth": 1, "n_estimators": 100, "learning_rate": 0.03,
                                        "min_child_weight": 5})
        self.assertEqual(dict(xgb[1]), {"max_depth": 1, "n_estimators": 100, "learning_rate": 0.03,
                                        "min_child_weight": 1})
        self.assertEqual(dict(xgb[-1]), {"max_depth": 4, "n_estimators": 300, "learning_rate": 0.1,
                                         "min_child_weight": 1})
        self.assertEqual(modelos.configuracoes("lr_fixa"), ({},))

    def test_grade_busca_preserva_a_ordem_no_parametergrid(self):
        for codigo in ("lr_tuning", "svm", "rf", "xgb"):
            grade = modelos.grade_busca(codigo)
            self.assertTrue(all(len(v) == 1 for ponto in grade for v in ponto.values()), codigo)
            self.assertEqual([dict(p) for p in ParameterGrid(grade)],
                             [dict(c) for c in modelos.configuracoes(codigo)], codigo)
        self.assertEqual(modelos.grade_busca("lr_fixa"), [])

    def test_grade_busca_nova_a_cada_chamada(self):
        a = modelos.grade_busca("svm")
        a[0]["classificador__C"].append(999.0)
        a.pop()
        b = modelos.grade_busca("svm")
        self.assertEqual(len(b), 30)
        self.assertEqual(b[0]["classificador__C"], [0.01])
        with self.assertRaises(TypeError):
            modelos.configuracoes("svm")[0]["classificador__C"] = 1.0  # type: ignore[index]

    def test_nomes_de_parametro(self):
        self.assertEqual(modelos.nome_parametro("lr_tuning", "C"), "classificador__C")
        self.assertEqual(modelos.nome_parametro("svm", "gamma"), "classificador__gamma")
        self.assertEqual(modelos.nome_parametro("rf", "max_features"), "max_features")
        self.assertEqual(modelos.nome_parametro("xgb", "learning_rate"), "learning_rate")
        for codigo, nome in (("lr_fixa", "C"), ("rf", "C"), ("svm", "kernel"), ("xgb", "subsample")):
            with self.assertRaises(modelos.ModeloInvalido):
                modelos.nome_parametro(codigo, nome)

    def test_todas_as_configuracoes_sao_aceitas_pelo_estimador(self):
        """Os nomes da grade valem no estimador real: ``set_params`` de cada ponto, sem erro."""
        for codigo in ("lr_tuning", "svm", "rf"):
            for config in modelos.configuracoes(codigo):
                estimador = modelos.criar_estimador(codigo).set_params(**config)
                parametros = estimador.get_params()
                for nome, valor in config.items():
                    self.assertEqual(parametros[nome], valor)

    def test_valores_vem_da_configuracao(self):
        """Uma configuração diferente gera outro estimador: nada está escrito à mão na fábrica."""
        rf = modelos.especificacao("rf")
        outra = dataclasses.replace(rf, fixos={**rf.fixos, "n_estimators": 7, "random_state": 3})
        falsa = SimpleNamespace(modelos=(outra,), execucao=C.execucao)
        with unittest.mock.patch.object(cfg, "CONFIGURACAO", falsa):
            estimador = modelos.criar_estimador("rf")
            self.assertEqual(modelos.codigos(), ("rf",))
        self.assertEqual((estimador.n_estimators, estimador.random_state), (7, 3))
        self.assertEqual(modelos.criar_estimador("rf").n_estimators, 500)

    def test_estimador_nao_suportado(self):
        with self.assertRaisesRegex(modelos.ModeloInvalido, "não suportado"):
            modelos._classe("sklearn.tree.DecisionTreeClassifier")
        self.assertEqual(set(modelos.ESTIMADORES_SUPORTADOS), {m.estimador for m in C.modelos})


# --------------------------------------------------------------------------- #
# Pipelines e parâmetros fixos
# --------------------------------------------------------------------------- #


class TestPipelines(unittest.TestCase):
    def test_padronizacao_so_em_lr_e_svm(self):
        for codigo in ("lr_fixa", "lr_tuning", "svm"):
            estimador = modelos.criar_estimador(codigo)
            self.assertIsInstance(estimador, Pipeline)
            self.assertEqual([nome for nome, _ in estimador.steps], ["padronizacao", "classificador"])
            self.assertIsInstance(estimador.named_steps["padronizacao"], StandardScaler)
            self.assertEqual(estimador.named_steps["padronizacao"].get_params(), StandardScaler().get_params())
            self.assertIsInstance(estimador.named_steps["classificador"], CLASSES[codigo])
        rf = modelos.criar_estimador("rf")
        self.assertIsInstance(rf, RandomForestClassifier)
        self.assertNotIsInstance(rf, Pipeline)

    def test_parametros_fixos_exatos_e_nada_alem_deles(self):
        for codigo in SKLEARN:
            espec = modelos.especificacao(codigo)
            classificador = modelos.criar_classificador(codigo)
            parametros = classificador.get_params(deep=False)
            for nome, valor in espec.fixos.items():
                self.assertEqual((type(parametros[nome]), parametros[nome]), (type(valor), valor), (codigo, nome))
            padrao = CLASSES[codigo]().get_params(deep=False)
            alterados = {n for n, v in parametros.items() if not (type(v) is type(padrao[n]) and v == padrao[n])}
            self.assertLessEqual(alterados, set(espec.fixos), codigo)

    def test_p9_secao_5_1(self):
        lr = modelos.criar_classificador("lr_fixa").get_params()
        self.assertEqual({k: lr[k] for k in ("C", "penalty", "solver", "class_weight", "max_iter", "random_state")},
                         {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "class_weight": None, "max_iter": 1000,
                          "random_state": 42})
        lrt = modelos.criar_classificador("lr_tuning").get_params()
        self.assertEqual(lrt["C"], LogisticRegression().C)  # padrão; o valor vem da grade
        svm = modelos.criar_classificador("svm").get_params()
        self.assertEqual({k: svm[k] for k in ("kernel", "probability", "class_weight", "random_state")},
                         {"kernel": "rbf", "probability": False, "class_weight": None, "random_state": 42})
        rf = modelos.criar_classificador("rf").get_params()
        self.assertEqual({k: rf[k] for k in ("n_estimators", "criterion", "bootstrap", "class_weight", "n_jobs",
                                             "random_state")},
                         {"n_estimators": 500, "criterion": "gini", "bootstrap": True, "class_weight": None,
                          "n_jobs": 1, "random_state": 42})

    def test_svm_sem_probabilidade_nem_calibracao(self):
        svm = modelos.criar_estimador("svm")
        self.assertFalse(svm.named_steps["classificador"].probability)
        self.assertFalse(any(isinstance(p, CalibratedClassifierCV) for _, p in svm.steps))
        X, y = dados_sinteticos()
        svm.fit(X, y)
        self.assertFalse(hasattr(svm, "predict_proba"))
        self.assertEqual(svm.decision_function(X).shape, (len(y),))

    def test_paralelismo_e_sementes(self):
        self.assertEqual(modelos.criar_classificador("rf").n_jobs, 1)
        for codigo in ("lr_fixa", "lr_tuning"):
            # Não fixado pela configuração; padrão None = um processo, sem efeito no problema binário.
            self.assertIsNone(modelos.criar_classificador(codigo).n_jobs)
        self.assertNotIn("n_jobs", modelos.criar_classificador("svm").get_params())
        for codigo in SKLEARN:
            self.assertEqual(modelos.criar_classificador(codigo).random_state, 42)


class TestConferencias(unittest.TestCase):
    """A fábrica recusa um estimador construído diferente da configuração."""

    def recusa(self, codigo: str, **mudancas):
        classificador = modelos.criar_classificador(codigo).set_params(**mudancas)
        with self.assertRaises(modelos.ModeloInvalido):
            modelos._conferir(classificador, modelos.especificacao(codigo))

    def test_parametro_fixo_divergente(self):
        self.recusa("lr_fixa", max_iter=100)
        self.recusa("lr_fixa", C=1)  # int em vez de float
        self.recusa("rf", criterion="entropy")

    def test_svm_com_probabilidade(self):
        self.recusa("svm", probability=True)

    def test_paralelismo_e_semente(self):
        self.recusa("rf", n_jobs=-1)
        self.recusa("lr_tuning", n_jobs=4)
        self.recusa("svm", random_state=None)

    def com_fixos(self, codigo: str, fixos: Dict[str, Any]):
        espec = dataclasses.replace(modelos.especificacao(codigo), fixos=fixos)
        return unittest.mock.patch.object(cfg, "CONFIGURACAO", SimpleNamespace(modelos=(espec,), execucao=C.execucao))

    def test_svm_com_probabilidade_mesmo_se_a_configuracao_pedir(self):
        """A guarda do SVM não depende da configuração: probabilidade nunca é aceita (P9, D10)."""
        fixos = {**modelos.especificacao("svm").fixos, "probability": True}
        with self.com_fixos("svm", fixos), self.assertRaisesRegex(modelos.ModeloInvalido, "probability"):
            modelos.criar_estimador("svm")

    def test_semente_nao_fixada_mesmo_se_a_configuracao_omitir(self):
        for codigo in ("rf", "svm", "lr_tuning"):
            fixos = {k: v for k, v in modelos.especificacao(codigo).fixos.items() if k != "random_state"}
            with self.subTest(codigo), self.com_fixos(codigo, fixos):
                with self.assertRaisesRegex(modelos.ModeloInvalido, "semente não fixada"):
                    modelos.criar_estimador(codigo)

    def test_hiperparametro_da_grade_inexistente(self):
        rf = modelos.especificacao("rf")
        espec = dataclasses.replace(rf, grade=(cfg.ParametroGrade("inexistente", (1, 2)),))
        with self.assertRaisesRegex(modelos.ModeloInvalido, "inexistente"):
            modelos._conferir(modelos.criar_classificador("rf"), espec)


# --------------------------------------------------------------------------- #
# Instâncias independentes e clonagem
# --------------------------------------------------------------------------- #


class TestIndependencia(unittest.TestCase):
    def test_instancias_novas_sem_objetos_compartilhados(self):
        for codigo in SKLEARN:
            a, b = modelos.criar_estimador(codigo), modelos.criar_estimador(codigo)
            self.assertIsNot(a, b)
            if isinstance(a, Pipeline):
                for (_, pa), (_, pb) in zip(a.steps, b.steps):
                    self.assertIsNot(pa, pb)
                self.assertIsNot(a.steps, b.steps)
            # Mudar uma instância não muda a outra.
            chave = "classificador__random_state" if codigo != "rf" else "random_state"
            a.set_params(**{chave: 0})
            self.assertEqual(b.get_params()[chave], 42)

    def test_clone_preserva_parametros_e_nao_compartilha(self):
        for codigo in SKLEARN:
            original = modelos.criar_estimador(codigo)
            copia = clone(original)
            self.assertIsNot(copia, original)
            self.assertEqual(type(copia), type(original))
            for nome, valor in original.get_params(deep=True).items():
                if not hasattr(valor, "get_params") and not isinstance(valor, list):
                    self.assertEqual(copia.get_params(deep=True)[nome], valor, (codigo, nome))

    def test_ajustar_um_nao_ajusta_o_outro(self):
        X, y = dados_sinteticos()
        a, b = modelos.criar_estimador("lr_fixa"), modelos.criar_estimador("lr_fixa")
        a.fit(X, y)
        check_is_fitted(a)
        with self.assertRaises(NotFittedError):
            check_is_fitted(b)
        with self.assertRaises(NotFittedError):
            check_is_fitted(clone(a))


# --------------------------------------------------------------------------- #
# Comportamento dos estimadores disponíveis (dados sintéticos)
# --------------------------------------------------------------------------- #


class TestComportamento(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = dados_sinteticos()
        cls.X_novo, _ = dados_sinteticos(40, semente=1)

    def saida(self, codigo: str, config: Dict[str, Any]) -> np.ndarray:
        estimador = modelos.criar_estimador(codigo).set_params(**config).fit(self.X, self.y)
        pontuacao = modelos.especificacao(codigo).pontuacao
        saida = getattr(estimador, pontuacao)(self.X_novo)
        return saida[:, 1] if pontuacao == "predict_proba" else saida

    def test_saidas_finitas_no_formato_da_pontuacao(self):
        casos = {"lr_fixa": {}, "lr_tuning": dict(modelos.configuracoes("lr_tuning")[6]),
                 "svm": dict(modelos.configuracoes("svm")[13]), "rf": dict(modelos.configuracoes("rf")[7])}
        for codigo, config in casos.items():
            saida = self.saida(codigo, config)
            self.assertEqual(saida.shape, (len(self.X_novo),), codigo)
            self.assertTrue(np.all(np.isfinite(saida)), codigo)
            if modelos.especificacao(codigo).pontuacao == "predict_proba":
                self.assertTrue(np.all((saida >= 0) & (saida <= 1)), codigo)

    def test_ajuste_deterministico_bit_a_bit(self):
        casos = {"lr_tuning": dict(modelos.configuracoes("lr_tuning")[0]),
                 "svm": dict(modelos.configuracoes("svm")[-1]), "rf": dict(modelos.configuracoes("rf")[-1])}
        for codigo, config in casos.items():
            primeira, segunda = self.saida(codigo, config), self.saida(codigo, config)
            self.assertTrue(np.array_equal(primeira, segunda), codigo)

    def test_padronizacao_ajustada_dentro_do_pipeline(self):
        estimador = modelos.criar_estimador("lr_fixa").fit(self.X, self.y)
        escala = estimador.named_steps["padronizacao"]
        np.testing.assert_allclose(escala.mean_, self.X.mean(axis=0))
        np.testing.assert_allclose(escala.scale_, self.X.std(axis=0))
        # Sem padronização, a mesma regressão logística dá outra solução.
        crua = modelos.criar_classificador("lr_fixa").fit(self.X, self.y)
        self.assertFalse(np.allclose(crua.predict_proba(self.X_novo), estimador.predict_proba(self.X_novo)))

    def test_compatibilidade_com_gridsearchcv(self):
        """Nomes e ordem da grade no ``GridSearchCV`` (dados sintéticos; nada é selecionado nem gravado)."""
        rng = np.random.default_rng(3)
        cv = []
        posicoes = rng.permutation(len(self.y))
        for parte in np.array_split(posicoes, 3):
            cv.append((np.setdiff1d(posicoes, parte), np.sort(parte)))
        for codigo in ("lr_tuning", "svm"):
            busca = GridSearchCV(modelos.criar_estimador(codigo), param_grid=modelos.grade_busca(codigo),
                                 scoring="roc_auc", cv=cv, refit=False, n_jobs=1, error_score="raise",
                                 return_train_score=False)
            busca.fit(self.X, self.y)
            self.assertEqual(busca.cv_results_["params"], [dict(c) for c in modelos.configuracoes(codigo)])


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #


class TestXGBoostContratos(unittest.TestCase):
    """Contratos da fábrica com um ``xgboost`` falso: não executam o XGBoost."""

    def com_modulo(self, modulo):
        return unittest.mock.patch.dict(sys.modules, {"xgboost": modulo})

    def test_versao_fixada_vem_do_ambiente_da_configuracao(self):
        self.assertEqual(modelos.VERSAO_XGBOOST, "2.1.4")
        self.assertEqual(C.execucao.ambiente["xgboost"], "2.1.4")

    def test_construcao_com_a_versao_fixada(self):
        with self.com_modulo(modulo_xgboost_falso()):
            estimador = modelos.criar_estimador("xgb")
            self.assertEqual(modelos.xgboost_disponivel(), (True, "2.1.4"))
        self.assertNotIsInstance(estimador, Pipeline)
        parametros = estimador.get_params()
        for nome, valor in modelos.especificacao("xgb").fixos.items():
            self.assertEqual(parametros[nome], valor, nome)
        self.assertEqual((parametros["n_jobs"], parametros["random_state"], parametros["early_stopping_rounds"]),
                         (1, 42, None))
        self.assertEqual({k: parametros[k] for k in ("objective", "tree_method", "subsample", "colsample_bytree",
                                                      "reg_lambda")},
                         {"objective": "binary:logistic", "tree_method": "hist", "subsample": 0.8,
                          "colsample_bytree": 1.0, "reg_lambda": 1.0})
        for config in modelos.configuracoes("xgb"):
            clone(estimador).set_params(**config)

    def test_versao_diferente_recusada(self):
        for versao in ("3.0.0", "2.1.3", "2.1.4.post1", "", None):
            with self.subTest(versao=versao), self.com_modulo(modulo_xgboost_falso(versao)):
                with self.assertRaisesRegex(modelos.XGBoostIndisponivel, "exige 2.1.4"):
                    modelos.criar_estimador("xgb")
                self.assertFalse(modelos.xgboost_disponivel()[0])

    def test_ausente_recusado(self):
        with self.com_modulo(None):
            with self.assertRaisesRegex(modelos.XGBoostIndisponivel, "indisponível") as ctx:
                modelos.criar_estimador("xgb")
        self.assertIsInstance(ctx.exception.__cause__, ImportError)

    def test_biblioteca_nativa_bloqueada_recusada(self):
        with sem_xgboost_importado(), unittest.mock.patch.object(sys, "meta_path",
                                                                 [_FalhaAoCarregar(), *sys.meta_path]):
            with self.assertRaisesRegex(modelos.XGBoostIndisponivel, "OSError") as ctx:
                modelos.criar_estimador("xgb")
        self.assertIsInstance(ctx.exception.__cause__, OSError)

    def test_parada_antecipada_ou_paralelismo_recusados(self):
        with self.com_modulo(modulo_xgboost_falso(early_stopping_padrao=10)):
            with self.assertRaisesRegex(modelos.ModeloInvalido, "parada antecipada"):
                modelos.criar_estimador("xgb")
        with self.com_modulo(modulo_xgboost_falso(ignora_n_jobs=True)):
            with self.assertRaisesRegex(modelos.ModeloInvalido, "n_jobs"):
                modelos.criar_estimador("xgb")

    def test_importacao_so_sob_demanda(self):
        registro = _Registro()
        with sem_xgboost_importado(), unittest.mock.patch.object(sys, "meta_path", [registro, *sys.meta_path]):
            for codigo in SKLEARN:
                modelos.criar_estimador(codigo)
                modelos.grade_busca(codigo)
            modelos.configuracoes("xgb")
            modelos.grade_busca("xgb")
            self.assertEqual(registro.tentativas, 0)
            modelos.xgboost_disponivel()
            self.assertEqual(registro.tentativas, 1)

    def test_codigos_de_saida(self):
        codigos = C.execucao.codigos_saida
        self.assertEqual(codigos[modelos.XGBoostIndisponivel.chave_saida], 4)
        self.assertEqual(codigos[modelos.ModeloInvalido.chave_saida], 4)


class TestXGBoostAmbienteAtual(unittest.TestCase):
    @unittest.skipIf(XGB_DISPONIVEL, "XGBoost 2.1.4 instalado: a ausência não se aplica")
    def test_ausencia_no_ambiente_atual_e_explicita(self):
        with self.assertRaisesRegex(modelos.XGBoostIndisponivel, "2.1.4"):
            modelos.criar_estimador("xgb")
        self.assertIn("nada é instalado", XGB_MOTIVO)

    @unittest.skipUnless(XGB_DISPONIVEL, f"teste de fumaça do XGBoost NÃO executado: {XGB_MOTIVO}")
    def test_fumaca_xgboost_real(self):  # pragma: no cover - depende do ambiente
        """Teste de fumaça da P9, Seção 13: ajuste, clone, GridSearchCV e duas execuções idênticas."""
        X, y = dados_sinteticos()
        saidas = []
        for _ in range(2):
            estimador = modelos.criar_estimador("xgb").set_params(**modelos.configuracoes("xgb")[5])
            saidas.append(estimador.fit(X, y).predict_proba(X)[:, 1])
        self.assertTrue(np.array_equal(saidas[0], saidas[1]))
        clone(modelos.criar_estimador("xgb"))
        busca = GridSearchCV(modelos.criar_estimador("xgb"), param_grid=modelos.grade_busca("xgb")[:2],
                             scoring="roc_auc", cv=3, refit=False, n_jobs=1, error_score="raise")
        busca.fit(X, y)
        self.assertEqual(busca.cv_results_["params"], [dict(c) for c in modelos.configuracoes("xgb")[:2]])


# --------------------------------------------------------------------------- #
# Isolamento do módulo
# --------------------------------------------------------------------------- #


class TestIsolamento(unittest.TestCase):
    def setUp(self):
        self.arvore = ast.parse((REPO_ROOT / "src" / "benchmark_s6" / "modelos.py").read_text(encoding="utf-8"))

    def test_importa_so_o_permitido(self):
        modulos_importados = set()
        for n in ast.walk(self.arvore):
            if isinstance(n, ast.Import):
                modulos_importados |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                modulos_importados.add(("." * n.level) + (n.module or ""))
        self.assertEqual(modulos_importados, {"__future__", "types", "typing", "sklearn.pipeline",
                                              "sklearn.preprocessing", "sklearn.linear_model", "sklearn.svm",
                                              "sklearn.ensemble", "xgboost", "."})

    def test_nao_ajusta_nao_mede_nao_le(self):
        proibidos = {"fit", "predict", "predict_proba", "decision_function", "score", "GridSearchCV",
                     "roc_auc_score", "CalibratedClassifierCV", "open", "read_csv", "read_bytes", "read_text",
                     "folds", "dados"}
        nomes = {n.id for n in ast.walk(self.arvore) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(self.arvore) if isinstance(n, ast.Attribute)}
        self.assertEqual(nomes & proibidos, set())


if __name__ == "__main__":
    unittest.main()
