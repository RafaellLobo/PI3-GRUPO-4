"""Testes da execução de unidades do benchmark S6 (``src/benchmark_s6/unidades.py``).

Só dados SINTÉTICOS em memória (``SINT-…``), montados numa ``DadosBenchmark``
com ``oficial=False``; nenhum arquivo e nenhum dado de paciente. ``rf`` e
``xgb`` usam os dois primeiros pontos da grade, substituídos só no teste
(a grade completa da RF leva quase um minuto por unidade). O XGBoost real só
é exercitado se o XGBoost 2.1.4 estiver instalado. Nenhuma métrica externa é
calculada.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/benchmark_s6 -v
"""

from __future__ import annotations

import ast
import json
import pickle
import subprocess
import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[1]
for caminho in (REPO_ROOT, AQUI):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import dados  # noqa: E402
from src.benchmark_s6 import folds  # noqa: E402
from src.benchmark_s6 import modelos  # noqa: E402
from src.benchmark_s6 import tuning  # noqa: E402
from src.benchmark_s6 import unidades  # noqa: E402
from test_folds import amostra_sintetica  # noqa: E402
from test_tuning import caracteristicas, grade_reduzida  # noqa: E402

C = cfg.CONFIGURACAO
XGB_DISPONIVEL, XGB_MOTIVO = modelos.xgboost_disponivel()
PROBABILIDADE = ("lr_fixa", "lr_tuning", "rf", "xgb")


def carga_sintetica(n_pacientes: int = 90, semente: int = 0, modo: str = "implementacao",
                    X: np.ndarray = None) -> dados.DadosBenchmark:
    """``DadosBenchmark`` sintética (``oficial=False``), com X alinhado à amostra por construção."""
    a = amostra_sintetica(n_pacientes, semente=semente)
    X = caracteristicas(a) if X is None else X
    frame = pd.DataFrame({cfg.COLUNA_NODULO: a.ids, cfg.COLUNA_PACIENTE: a.grupos,
                          dados.COLUNA_PARTICAO: "train", cfg.COLUNA_ALVO: a.y})
    for i, nome in enumerate(C.caracteristicas):
        frame[nome] = X[:, i]
    return dados.DadosBenchmark(modo, dados.PARTICOES_POR_MODO[modo], False, tuple(C.caracteristicas), {}, {}, None,
                                frame)


class BaseUnidades(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.carga = carga_sintetica()
        cls.ctx = unidades.ContextoExecucao.de_dados(cls.carga)
        cls.dobra = cls.ctx.plano.dobra(0, 0)

    def executar(self, modelo: str, k: int = 0, ctx=None):
        return unidades.executar_unidade(ctx or self.ctx, unidades.Unidade(0, k, modelo))


# --------------------------------------------------------------------------- #
# Identificação, contexto e tarefas
# --------------------------------------------------------------------------- #


class TestIdentificacao(unittest.TestCase):
    def test_unidade_valida_e_rotulo(self):
        u = unidades.Unidade(3, 4, "svm")
        self.assertEqual(u.rotulo, "r03-k4-svm")
        for r, k, m in ((10, 0, "svm"), (-1, 0, "svm"), (True, 0, "svm"), (0, 5, "svm"), (0, 0.0, "svm")):
            with self.subTest((r, k, m)), self.assertRaises(ValueError):
                unidades.Unidade(r, k, m)
        with self.assertRaises(modelos.ModeloInvalido):
            unidades.Unidade(0, 0, "lr")

    def test_unidades_do_modo(self):
        impl = unidades.unidades_do_modo("implementacao")
        defin = unidades.unidades_do_modo("definitivo")
        self.assertEqual((len(impl), len(defin)), (25, 250))
        self.assertEqual(len(defin), C.n_unidades())
        self.assertEqual(impl[:6], tuple(unidades.Unidade(0, 0, m) for m in modelos.codigos()) +
                         (unidades.Unidade(0, 1, "lr_fixa"),))
        self.assertEqual(defin[-1], unidades.Unidade(9, 4, "xgb"))
        self.assertEqual(len(set(defin)), 250)
        with self.assertRaises(folds.DobrasInvalidas):
            unidades.unidades_do_modo("verificar")


class TestContexto(BaseUnidades):
    def test_contexto_da_mesma_carga(self):
        a = self.carga.amostra()
        self.assertEqual(self.ctx.amostra.sha256, a.sha256)
        self.assertEqual(self.ctx.plano.fingerprint, folds.gerar_plano(a, "implementacao").fingerprint)
        np.testing.assert_array_equal(self.ctx.X, self.carga.X())
        self.assertFalse(self.ctx.X.flags.writeable)
        self.assertEqual((self.ctx.modo, self.ctx.oficial), ("implementacao", False))
        self.assertEqual(self.ctx.sha256_configuracao, cfg.sha256_configuracao())
        with self.assertRaises(TypeError):
            unidades.ContextoExecucao.de_dados(a)  # type: ignore[arg-type]

    def test_tarefas_so_com_primitivos_e_mesma_dobra_para_os_modelos(self):
        tarefas = self.ctx.tarefas()
        self.assertEqual(len(tarefas), 25)
        for t in tarefas:
            for valor in vars(t).values():
                self.assertIn(type(valor), (str, int))
            self.assertEqual(t.fingerprint_dobra, self.ctx.plano.dobra(t.repeticao, t.dobra).fingerprint)
            self.assertEqual(t.fingerprint_plano, self.ctx.plano.fingerprint)
        por_dobra = {}
        for t in tarefas:
            por_dobra.setdefault((t.repeticao, t.dobra), set()).add(t.fingerprint_dobra)
        self.assertTrue(all(len(v) == 1 for v in por_dobra.values()))

    def test_unidade_fora_do_plano_do_modo(self):
        with self.assertRaisesRegex(unidades.FalhaUnidade, "fora do plano"):
            self.ctx.tarefa(unidades.Unidade(1, 0, "lr_fixa"))
        with self.assertRaises(TypeError):
            self.ctx.tarefa((0, 0, "lr_fixa"))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Execução dos cinco estimadores
# --------------------------------------------------------------------------- #


class TestExecucao(BaseUnidades):
    def conferir(self, resultado: unidades.ResultadoUnidade, modelo: str):
        a, d = self.ctx.amostra, self.dobra
        aval = d.externa.avaliacao
        p, reg = resultado.privado, resultado.registro
        self.assertEqual(resultado.estado, "concluida")
        self.assertEqual(reg.unidade, unidades.Unidade(0, 0, modelo))
        self.assertEqual(reg.metodo_pontuacao, "decision_function" if modelo == "svm" else "predict_proba")
        np.testing.assert_array_equal(p.posicoes, aval)
        self.assertEqual(p.nodulo_ids, tuple(a.ids[aval]))
        np.testing.assert_array_equal(p.y, a.y[aval])
        self.assertEqual((p.pontuacoes.shape, p.pontuacoes.dtype), ((len(aval),), np.float64))
        self.assertTrue(np.all(np.isfinite(p.pontuacoes)))
        for arr in (p.posicoes, p.y, p.pontuacoes):
            self.assertFalse(arr.flags.writeable)
        if modelo in PROBABILIDADE:
            self.assertTrue(np.all((p.pontuacoes >= 0) & (p.pontuacoes <= 1)))
        self.assertEqual((reg.fingerprint_dobra, p.fingerprint_dobra, reg.fingerprint_plano),
                         (d.fingerprint, d.fingerprint, self.ctx.plano.fingerprint))
        self.assertEqual(dict(reg.contagem_avaliacao)["nodulos"], len(aval))
        self.assertEqual(dict(reg.contagem_treino)["nodulos"], len(d.externa.treino))
        # Pontuação igual à de um ajuste independente, pelo método da P9, Seção 7.1.
        ajuste = tuning.ajustar(self.ctx.X, a, d, modelo)
        esperado = (ajuste.estimador.decision_function(self.ctx.X[aval]) if modelo == "svm"
                    else ajuste.estimador.predict_proba(self.ctx.X[aval])[:, 1])
        np.testing.assert_array_equal(p.pontuacoes, np.asarray(esperado, dtype=np.float64))
        self.assertEqual(dict(reg.configuracao), dict(ajuste.configuracao))
        self.assertEqual(p.medias_internas, ajuste.medias_internas)

    def test_lr_fixa_e_lr_tuning_por_probabilidade(self):
        for modelo in ("lr_fixa", "lr_tuning"):
            with self.subTest(modelo):
                self.conferir(self.executar(modelo), modelo)

    def test_svm_por_decision_function(self):
        r = self.executar("svm")
        self.conferir(r, "svm")
        self.assertTrue(np.any((r.privado.pontuacoes < 0) | (r.privado.pontuacoes > 1)))

    def test_rf_com_grade_reduzida(self):
        with grade_reduzida("rf"):
            r = self.executar("rf")
            self.conferir(r, "rf")

    @unittest.skipUnless(XGB_DISPONIVEL, f"XGBoost real não exercitado na unidade: {XGB_MOTIVO}")
    def test_xgboost_real_com_grade_reduzida(self):  # pragma: no cover - depende do ambiente
        with grade_reduzida("xgb"):
            a, b = self.executar("xgb"), self.executar("xgb")
            self.conferir(a, "xgb")
        np.testing.assert_array_equal(a.privado.pontuacoes, b.privado.pontuacoes)

    def test_registro_agregado_sem_identificadores_nem_pontuacoes(self):
        r = self.executar("lr_tuning")
        texto = json.dumps(r.registro.como_dict(), ensure_ascii=False)
        self.assertNotIn("SINT-", texto)
        self.assertNotIn(self.ctx.amostra.sha256, texto)
        for valor in (*r.privado.pontuacoes[:5], *r.privado.medias_internas[:5]):
            self.assertNotIn(repr(float(valor)), texto)
        registro = json.loads(texto)
        self.assertEqual(registro["estado"], "concluida")
        self.assertFalse(registro["oficial"])
        self.assertEqual(registro["n_ajustes"], 13 * 5 + 1)
        self.assertEqual(set(registro["contagem_avaliacao"]), {"nodulos", "pacientes", "classe_0", "classe_1"})
        privado = r.privado.como_dict()
        self.assertTrue(all(i.startswith("SINT-") for i in privado["nodulo_ids"]))
        self.assertEqual(len(privado["pontuacoes"]), len(privado["posicoes"]))
        json.dumps(privado)


# --------------------------------------------------------------------------- #
# Isolamento entre treino e avaliação
# --------------------------------------------------------------------------- #


class TestIsolamento(BaseUnidades):
    def test_avaliacao_externa_nao_influencia_o_ajuste(self):
        X = self.ctx.X.copy()
        aval = self.dobra.externa.avaliacao
        X[aval] = X[aval] * 3.0 + 7.0
        outro = unidades.ContextoExecucao.de_dados(carga_sintetica(X=X))
        self.assertEqual(outro.plano.fingerprint, self.ctx.plano.fingerprint)
        a, b = self.executar("lr_tuning"), self.executar("lr_tuning", ctx=outro)
        self.assertEqual(a.registro.como_dict(), b.registro.como_dict())
        self.assertEqual(a.privado.medias_internas, b.privado.medias_internas)
        self.assertFalse(np.array_equal(a.privado.pontuacoes, b.privado.pontuacoes))

    def test_caracteristica_invalida_na_avaliacao_e_falha_tecnica(self):
        X = self.ctx.X.copy()
        X[self.dobra.externa.avaliacao[0], 1] = np.nan
        ctx = unidades.ContextoExecucao.de_dados(carga_sintetica(X=X))
        with self.assertRaisesRegex(unidades.FalhaUnidade, "avaliação externa"):
            self.executar("lr_fixa", ctx=ctx)

    def test_avaliacao_e_treino_disjuntos_por_linha_e_paciente(self):
        r = self.executar("lr_fixa")
        a, d = self.ctx.amostra, self.dobra
        self.assertEqual(set(r.privado.posicoes.tolist()) & set(d.externa.treino.tolist()), set())
        self.assertEqual(set(a.grupos[r.privado.posicoes]) & set(a.grupos[d.externa.treino]), set())

    def test_modulo_nao_mede_nao_grava_e_so_usa_apis_publicas(self):
        arvore = ast.parse((REPO_ROOT / "src" / "benchmark_s6" / "unidades.py").read_text(encoding="utf-8"))
        nomes = {n.id for n in ast.walk(arvore) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(arvore) if isinstance(n, ast.Attribute)}
        proibidos = {"roc_auc_score", "brier_score_loss", "log_loss", "open", "write_text", "write_bytes", "dump",
                     "to_csv", "read_csv", "savez", "save", "Pool", "ProcessPoolExecutor", "Process"}
        self.assertEqual(nomes & proibidos, set())
        privados = sorted({n.attr for n in ast.walk(arvore) if isinstance(n, ast.Attribute)
                           and isinstance(n.value, ast.Name) and n.value.id in ("folds", "tuning", "modelos", "s4",
                                                                                "cfg", "dados")
                           and n.attr.startswith("_")})
        self.assertEqual(privados, [])
        importados = {(("." * n.level) + (n.module or "")) for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom)}
        importados |= {a.name for n in ast.walk(arvore) if isinstance(n, ast.Import) for a in n.names}
        self.assertFalse({m for m in importados if "metrics" in m or "multiprocessing" in m or "pickle" in m})


# --------------------------------------------------------------------------- #
# Falhas técnicas e de integridade
# --------------------------------------------------------------------------- #


class TestFalhas(BaseUnidades):
    def falha(self, modelo: str, **patch) -> unidades.FalhaRegistrada:
        tarefa = self.ctx.tarefa(unidades.Unidade(0, 0, modelo))
        alvo, nome, func = patch["alvo"], patch["nome"], patch["func"]
        with unittest.mock.patch.object(alvo, nome, func):
            resultado = unidades.executar_tarefa(self.ctx, tarefa)
        self.assertIsInstance(resultado, unidades.FalhaRegistrada)
        self.assertEqual(resultado.estado, "FT")
        return resultado

    def test_excecao_na_predicao(self):
        def explode(self_, X):
            raise FloatingPointError("predição explodiu")

        r = self.falha("lr_fixa", alvo=Pipeline, nome="predict_proba", func=explode)
        self.assertEqual((r.chave_saida, r.categoria), ("falha_tecnica", "FloatingPointError"))
        self.assertIn("predição", r.mensagem)

    def test_pontuacoes_invalidas(self):
        n = len(self.dobra.externa.avaliacao)
        casos = {
            "NaN": lambda s, X: np.column_stack([np.full(n, 0.5), np.full(n, np.nan)]),
            "fora de [0, 1]": lambda s, X: np.column_stack([np.full(n, -0.2), np.full(n, 1.2)]),
            "forma": lambda s, X: np.full((n, 3), 1 / 3),
            "linhas a menos": lambda s, X: np.full((n - 1, 2), 0.5),
            "inteiro": lambda s, X: np.ones((n, 2), dtype=np.int64),
        }
        for nome, func in casos.items():
            with self.subTest(nome):
                r = self.falha("lr_fixa", alvo=Pipeline, nome="predict_proba", func=func)
                self.assertEqual(r.chave_saida, "falha_tecnica")
                self.assertEqual(r.categoria, "FalhaUnidade")
        # SVM: estraga só o estimador já ajustado (o ajuste interno também usa decision_function).
        real = tuning.ajustar

        def ajustar_e_estragar(*args, **kwargs):
            resultado = real(*args, **kwargs)
            resultado.estimador.decision_function = lambda X: np.where(np.arange(len(X)) == 0, np.inf, 0.0)
            return resultado

        r = self.falha("svm", alvo=tuning, nome="ajustar", func=ajustar_e_estragar)
        self.assertEqual(r.chave_saida, "falha_tecnica")
        self.assertIn("NaN ou infinita", r.mensagem)

    def test_falha_no_ajuste(self):
        def falha(self_, X, y=None, **kw):
            raise ValueError("ajuste explodiu")

        r = self.falha("lr_tuning", alvo=Pipeline, nome="fit", func=falha)
        self.assertEqual((r.chave_saida, r.categoria), ("falha_tecnica", "FalhaAjuste"))

    def test_xgboost_indisponivel_e_integridade(self):
        def indisponivel():
            raise modelos.XGBoostIndisponivel("xgboost 2.1.4 indisponível (simulado)")

        r = self.falha("xgb", alvo=modelos, nome="_classe_xgboost", func=indisponivel)
        self.assertEqual((r.chave_saida, r.categoria), ("integridade", "XGBoostIndisponivel"))
        self.assertEqual(C.execucao.codigos_saida[r.chave_saida], 4)

    def test_executar_unidade_levanta_falhaunidade(self):
        def explode(self_, X):
            raise RuntimeError("x")

        with unittest.mock.patch.object(Pipeline, "predict_proba", explode):
            with self.assertRaises(unidades.FalhaUnidade) as ctx:
                self.executar("lr_fixa")
        self.assertEqual(ctx.exception.chave_saida, "falha_tecnica")
        self.assertEqual(ctx.exception.unidade, unidades.Unidade(0, 0, "lr_fixa"))

    def test_tarefa_divergente_do_contexto_e_integridade(self):
        t = self.ctx.tarefa(unidades.Unidade(0, 1, "lr_fixa"))
        casos = {
            "modo": unidades.TarefaUnidade("definitivo", *list(vars(t).values())[1:]),
            "configuração": unidades.TarefaUnidade(**{**vars(t), "sha256_configuracao": "0" * 64}),
            "plano": unidades.TarefaUnidade(**{**vars(t), "fingerprint_plano": "0" * 64}),
            "dobra": unidades.TarefaUnidade(**{**vars(t), "fingerprint_dobra": self.dobra.fingerprint}),
        }
        for nome, tarefa in casos.items():
            with self.subTest(nome):
                r = unidades.executar_tarefa(self.ctx, tarefa)
                self.assertIsInstance(r, unidades.FalhaRegistrada)
                self.assertEqual(r.chave_saida, "integridade")
        outro = unidades.ContextoExecucao.de_dados(carga_sintetica(semente=4))
        r = unidades.executar_tarefa(outro, t)
        self.assertEqual((r.estado, r.chave_saida), ("FT", "integridade"))

    def test_tipos_invalidos_nao_viram_falha_registrada(self):
        t = self.ctx.tarefa(unidades.Unidade(0, 0, "lr_fixa"))
        with self.assertRaises(TypeError):
            unidades.executar_tarefa(self.ctx, vars(t))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            unidades.executar_tarefa(None, t)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            unidades.executar_tarefa(self.ctx, unidades.TarefaUnidade(**{**vars(t), "repeticao": 99}))
        with self.assertRaises(TypeError):
            unidades.executar_unidade(self.ctx, (0, 0, "lr_fixa"))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Determinismo e desenho para workers
# --------------------------------------------------------------------------- #


class TestDeterminismoEWorkers(BaseUnidades):
    def test_reexecucao_identica_bit_a_bit(self):
        for modelo in ("lr_fixa", "svm"):
            a, b = self.executar(modelo, k=2), self.executar(modelo, k=2)
            self.assertEqual(a.registro, b.registro)
            self.assertEqual(a.privado.pontuacoes.tobytes(), b.privado.pontuacoes.tobytes())

    def test_contratos_serializam_com_pickle_e_json(self):
        t = self.ctx.tarefa(unidades.Unidade(0, 3, "lr_fixa"))
        self.assertEqual(pickle.loads(pickle.dumps(t)), t)
        r = unidades.executar_tarefa(self.ctx, t)
        copia = pickle.loads(pickle.dumps(r))
        self.assertEqual(copia.registro, r.registro)
        self.assertEqual(copia.privado.como_dict(), r.privado.como_dict())
        falha = unidades.FalhaRegistrada(t.unidade, "falha_tecnica", "FalhaAjuste", "teste")
        self.assertEqual(pickle.loads(pickle.dumps(falha)), falha)
        json.dumps(falha.como_dict())

    def test_tarefa_executada_em_processo_novo_reconstroi_o_contexto(self):
        """Como um worker ``spawn``: só a tarefa atravessa; o processo novo refaz carga, plano e dobra."""
        unidade = unidades.Unidade(0, 4, "svm")
        tarefa = self.ctx.tarefa(unidade)
        local = unidades.executar_tarefa(self.ctx, tarefa)
        codigo = (
            "import pickle, sys\n"
            f"sys.path.insert(0, {str(AQUI)!r})\n"
            "import test_unidades as t\n"
            "from src.benchmark_s6 import unidades\n"
            "tarefa = pickle.loads(bytes.fromhex(sys.argv[1]))\n"
            "ctx = unidades.ContextoExecucao.de_dados(t.carga_sintetica())\n"
            "sys.stdout.write(pickle.dumps(unidades.executar_tarefa(ctx, tarefa)).hex())\n"
        )
        saida = subprocess.run([sys.executable, "-B", "-c", codigo, pickle.dumps(tarefa).hex()], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True, timeout=300)
        remoto = pickle.loads(bytes.fromhex(saida.stdout.strip().splitlines()[-1]))
        self.assertIsInstance(remoto, unidades.ResultadoUnidade)
        self.assertEqual(remoto.registro, local.registro)
        self.assertEqual(remoto.privado.pontuacoes.tobytes(), local.privado.pontuacoes.tobytes())
        self.assertEqual(remoto.privado.nodulo_ids, local.privado.nodulo_ids)

    def test_contexto_definitivo_executa_ultima_repeticao(self):
        ctx = unidades.ContextoExecucao.de_dados(carga_sintetica(modo="definitivo"))
        self.assertEqual(len(ctx.tarefas()), 250)
        r = unidades.executar_tarefa(ctx, ctx.tarefa(unidades.Unidade(9, 4, "lr_fixa")))
        self.assertEqual(r.estado, "concluida")
        self.assertEqual(r.registro.fingerprint_dobra, ctx.plano.dobra(9, 4).fingerprint)
        self.assertEqual(r.registro.modo, "definitivo")


if __name__ == "__main__":
    unittest.main()
