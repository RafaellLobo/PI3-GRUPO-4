"""Testes de ``src/benchmark_s6/coordenador.py`` e ``invariante.py`` (P9, Seção 14.3). Só dados sintéticos.

O treinamento é substituído por resultados sintéticos (``unidades.executar_tarefa`` trocada), mas TUDO o mais roda de
verdade: executor, checkpoints atômicos, consolidação, bootstrap, seleção, figuras, relatórios, varredura e manifesto.
Os dados e o git são injetados por ``Dependencias``; a produção não tem esses parâmetros na linha de comando.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cli_sintetico  # noqa: E402
import sinteticos as sn  # noqa: E402
from src.benchmark_s6 import adaptadores_s4 as s4  # noqa: E402
from src.benchmark_s6 import aprovacoes as ap  # noqa: E402
from src.benchmark_s6 import bootstrap as bs  # noqa: E402
from src.benchmark_s6 import checkpoints as ck  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import consolidacao as co  # noqa: E402
from src.benchmark_s6 import coordenador as cd  # noqa: E402
from src.benchmark_s6 import dados as dd  # noqa: E402
from src.benchmark_s6 import executor as ex  # noqa: E402
from src.benchmark_s6 import figuras as fg  # noqa: E402
from src.benchmark_s6 import invariante as inv  # noqa: E402
from src.benchmark_s6 import manifesto as mf  # noqa: E402
from src.benchmark_s6 import metricas as mt  # noqa: E402
from src.benchmark_s6 import modelos  # noqa: E402
from src.benchmark_s6 import privacidade as pv  # noqa: E402
from src.benchmark_s6 import selecao as sl  # noqa: E402
from src.benchmark_s6 import unidades as un  # noqa: E402

C = cfg.CONFIGURACAO
CODIGOS = dict(C.execucao.codigos_saida)
GIT = shutil.which("git")
MODULO = Path(cd.__file__)
SCRIPT = Path(cli_sintetico.__file__)


def remover(caminho: Path) -> None:
    shutil.rmtree(caminho, ignore_errors=True)


@unittest.skipIf(GIT is None, "git indisponível")
class BaseCoordenador(unittest.TestCase):
    """Repositório git temporário (código simulado), pasta de checkpoints ignorada e dados sintéticos."""

    @classmethod
    def setUpClass(cls):
        # Patch com desfazimento garantido: um patch global vazaria para test_modelos, test_tuning etc.
        patcher = mock.patch.object(modelos, "especificacao", sn.especificacao_reduzida)
        patcher.start()
        cls.addClassCleanup(patcher.stop)
        cls.raiz = Path(tempfile.mkdtemp(prefix="coord-s6-"))
        cls.aux = Path(tempfile.mkdtemp(prefix="coord-aux-s6-"))  # arquivos de teste que NÃO podem sujar o git
        cls.git("init", "-q")
        (cls.raiz / ".gitignore").write_text("checkpoints/\n", encoding="utf-8")
        cls.config_versionada = cls.raiz / "docs" / "sprint6" / "config_benchmark_p9.json"
        cfg.gerar_arquivo(cls.config_versionada)
        cls.baseline = cls.raiz / "baseline.json"
        cli_sintetico.baseline_sintetico(cls.baseline)
        cls.git("add", "-A")
        cls.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
        cls.ctx_impl = un.ContextoExecucao.de_dados(sn.carga_definitiva(90, modo="implementacao"))
        cls.ctx_def = un.ContextoExecucao.de_dados(sn.carga_definitiva())
        cls.mapa_impl = {r.unidade: r for r in sn.resultados_sinteticos(cls.ctx_impl)}
        cls.mapa_def = {r.unidade: r for r in sn.resultados_sinteticos(cls.ctx_def)}

    @classmethod
    def tearDownClass(cls):
        remover(cls.raiz)
        remover(cls.aux)

    @classmethod
    def git(cls, *args):
        return subprocess.run([GIT, "-C", str(cls.raiz), *args], capture_output=True, check=True, text=True).stdout

    @classmethod
    def pasta_nova(cls, nome: str) -> Path:
        caminho = Path(tempfile.mkdtemp(prefix=f"{nome}-s6-"))
        return caminho

    @classmethod
    def checkpoints_novos(cls) -> Path:
        import uuid
        caminho = cls.raiz / "checkpoints" / uuid.uuid4().hex[:10]
        caminho.mkdir(parents=True)
        return caminho

    @classmethod
    def deps(cls, **kw) -> cd.Dependencias:
        base = dict(
            carregar=lambda modo, m: sn.carga_definitiva(90, modo=modo) if modo == "implementacao"
            else sn.carga_definitiva(),
            carregador_dos_workers=lambda modo, m: sn.carregar_implementacao_sintetica,
            estado_git=lambda: s4.estado_git(cls.raiz), executar=ex.executar, ambiente=cd.ambiente_instalado,
            exigir_oficial=False, n_replicas=300, config_versionada=cls.config_versionada, repo=cls.raiz,
            baseline=cls.baseline)
        base.update(kw)
        return cd.Dependencias(**base)

    @classmethod
    def rodar(cls, modo: str, destino: Path, checkpoints: Path, dependencias=None, extra=(), processos=1):
        argv = ["--modo", modo, "--checkpoints", str(checkpoints), "--destino", str(destino),
                "--processos", str(processos), *extra]
        if modo in ("definitivo", "verificar"):
            argv += ["--manifesto-s4", "reports/sprint4/x/manifesto.json"]
        saida, erro = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(saida), contextlib.redirect_stderr(erro):
            codigo = cd.main(argv, dependencias or cls.deps())
        return codigo, saida.getvalue(), erro.getvalue()

    @classmethod
    def sem_treinar(cls, mapa):
        """Troca o treinamento pelos resultados sintéticos: o executor, os checkpoints e o resto são reais."""
        return mock.patch.object(un, "executar_tarefa", side_effect=lambda ctx, tarefa: mapa[tarefa.unidade])

    @classmethod
    def proibir_treino(cls):
        return mock.patch.object(un, "executar_tarefa", side_effect=AssertionError("não deveria treinar"))


# --------------------------------------------------------------------------- #
# Uso, códigos de saída e pré-voo
# --------------------------------------------------------------------------- #


class TestUsoECodigos(unittest.TestCase):
    def executar(self, *argv):
        erro = io.StringIO()
        with contextlib.redirect_stderr(erro), contextlib.redirect_stdout(io.StringIO()):
            return cd.main(list(argv), mock.Mock(side_effect=AssertionError("não deveria carregar"))), erro.getvalue()

    def test_erros_de_uso_saem_com_64_sem_tocar_em_dados(self):
        invalidos = [
            ["--modo", "implementacao"], ["--modo", "definitivo"], ["--modo", "verificar"],
            ["--modo", "implementacao", "--checkpoints", "c", "--processos", "0"],
            ["--modo", "implementacao", "--checkpoints", "c", "--processos", "x"],
            ["--modo", "implementacao", "--checkpoints", "c", "--gerar", "a.json"],
            ["--modo", "implementacao", "--checkpoints", "c", "--manifesto-s4", "m.json"],
            ["--modo", "configuracao", "--checkpoints", "c"], ["--modo", "configuracao", "--processos", "2"],
            ["--modo", "inexistente"], [],
        ]
        for argv in invalidos:
            with self.subTest(argv=argv):
                codigo, erro = self.executar(*argv)
                self.assertEqual(codigo, cd.CODIGO_ERRO_USO)
                self.assertIn("erro:", erro)

    def test_codigos_de_saida_da_p9(self):
        self.assertEqual(CODIGOS, {"concluido": 0, "bloqueio_aprovacao": 2, "falha_tecnica": 3, "integridade": 4,
                                   "privacidade_ou_vocabulario": 5})
        casos = {
            ap.BloqueioAprovacao("x"): 2, ap.RegistroAusente("x"): 2, dd.AcessoNegado("x"): 2,
            dd.DadosInvalidos("x"): 4, cfg.ConfiguracaoDivergente("x"): 4, FileNotFoundError("x"): 4,
            s4.ArquivoS4Divergente("x"): 4, ck.CheckpointCorrompido("x"): 4, ck.CheckpointIncompativel("x"): 4,
            mf.ManifestoInvalido("x"): 4, cd.AmbienteDivergente("x"): 4, cd.PreVooRecusado("x"): 4,
            cd.DestinoOcupado("x"): 4, inv.InvarianteViolado("x"): 4, pv.PrivacidadeViolada("x"): 5,
            ck.DiretorioNaoLocal("x"): 5, un.FalhaUnidade(un.Unidade(0, 0, "svm"), "x"): 3,
            cd.ExecucaoIncompleta("x"): 3, cd.ExecucaoIncompleta("x", "integridade"): 4, ValueError("x"): 3,
            mt.MetricasInvalidas("x"): 4, co.ConsolidacaoInvalida("x"): 4, bs.BootstrapInvalido("x"): 4,
            sl.EntradaInvalida("x"): 4,
        }
        for exc, esperado in casos.items():
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(cd.codigo_de_saida(exc), esperado)
        self.assertNotIn(cd.CODIGO_ERRO_USO, CODIGOS.values())

    def test_producao_nao_tem_parametros_de_fonte_aprovacao_ou_configuracao(self):
        parser = cd._parser()
        opcoes = {o for a in parser._actions for o in a.option_strings}
        self.assertEqual(opcoes, {"-h", "--help", "--modo", "--gerar", "--conferir", "--checkpoints", "--destino",
                                  "--processos", "--manifesto-s4"})
        producao = cd.dependencias_de_producao()
        self.assertTrue(producao.exigir_oficial)
        self.assertIsNone(producao.n_replicas)  # B = 10.000 da configuração
        self.assertIs(producao.executar, ex.executar)
        self.assertIs(producao.carregador_dos_workers("implementacao", None), dd.carregar_implementacao)


class TestPreVoo(BaseCoordenador):
    def test_ambiente_divergente_aborta_antes_de_carregar_dados(self):
        for nome in ("xgboost", "scikit-learn", "python"):
            instalado = {**cd.ambiente_instalado(), nome: "0.0.0"}
            carregar = mock.Mock(side_effect=AssertionError("não deveria carregar"))
            for modo in ("implementacao", "definitivo", "verificar"):
                with self.subTest(pacote=nome, modo=modo):
                    codigo, _, erro = self.rodar(modo, self.pasta_nova("d"), self.checkpoints_novos(),
                                                 self.deps(ambiente=lambda i=instalado: i, carregar=carregar))
                    self.assertEqual(codigo, 4)
                    self.assertIn(nome, erro)
                    carregar.assert_not_called()
        self.assertEqual(set(cfg.CONFIGURACAO.execucao.ambiente), {"python", "scikit-learn", "numpy", "pandas",
                                                                  "scipy", "matplotlib", "xgboost"})
        self.assertEqual(cfg.CONFIGURACAO.execucao.ambiente["xgboost"], "2.1.4")

    def test_ambiente_instalado_confere_com_a_p9(self):
        cd.verificar_ambiente(self.deps())  # a .venv do projeto está nas versões fixadas

    def test_definitivo_sem_aprovacao_real_e_bloqueado_com_2_sem_treinar_nem_ler_dados(self):
        """A carga REAL de produção exige P9 e S4_E3_REGISTRADA antes de qualquer leitura; aqui não há registro."""
        dep = self.deps(carregar=cd._carregar_producao, exigir_oficial=True)
        destino = self.pasta_nova("d")
        with self.proibir_treino(), mock.patch.object(dd, "_ler_bytes", side_effect=AssertionError("leu dados")):
            codigo, _, erro = self.rodar("definitivo", destino, self.checkpoints_novos(), dep)
        self.assertEqual(codigo, 2, erro)
        self.assertEqual(list(destino.iterdir()), [])

    def test_definitivo_recusa_dados_nao_oficiais_git_sujo_e_config_ausente_antes_de_treinar(self):
        destino = self.pasta_nova("d")
        with self.proibir_treino():
            codigo, _, erro = self.rodar("definitivo", destino, self.checkpoints_novos(),
                                         self.deps(exigir_oficial=True))
            self.assertEqual((codigo, "oficial" in erro), (4, True))
            suja = SimpleNamespace(head="a" * 40, branch="x", limpo=False)
            codigo, _, erro = self.rodar("definitivo", destino, self.checkpoints_novos(),
                                         self.deps(estado_git=lambda: suja))
            self.assertEqual((codigo, "git limpa" in erro), (4, True))
            codigo, _, erro = self.rodar("definitivo", destino, self.checkpoints_novos(),
                                         self.deps(config_versionada=self.aux / "nao_existe.json"))
            self.assertEqual((codigo, "ausente" in erro), (4, True))
            divergente = self.aux / "divergente.json"
            divergente.write_text('{"a":1}\n', encoding="utf-8")
            codigo, _, _ = self.rodar("definitivo", destino, self.checkpoints_novos(),
                                      self.deps(config_versionada=divergente))
            self.assertEqual(codigo, 4)
        self.assertEqual(list(destino.iterdir()), [])

    def test_modo_errado_dos_dados_e_recusado(self):
        with self.proibir_treino():
            codigo, _, erro = self.rodar("definitivo", self.pasta_nova("d"), self.checkpoints_novos(),
                                         self.deps(carregar=lambda m, x: sn.carga_definitiva(90, modo="implementacao")))
            self.assertEqual((codigo, "definitivo" in erro), (4, True))
            definitivo = sn.carga_definitiva()
            codigo, _, erro = self.rodar("implementacao", self.pasta_nova("d"), self.checkpoints_novos(),
                                         self.deps(carregar=lambda m, x: definitivo))
            self.assertEqual((codigo, "train" in erro), (4, True))

    def test_destino_ocupado_nunca_e_sobrescrito(self):
        destino = self.pasta_nova("d")
        (destino / "resumo_benchmark.md").write_text("antigo", encoding="utf-8")
        with self.proibir_treino():
            codigo, _, erro = self.rodar("definitivo", destino, self.checkpoints_novos())
        self.assertEqual((codigo, "sobrescrito" in erro), (4, True))
        self.assertEqual((destino / "resumo_benchmark.md").read_text(encoding="utf-8"), "antigo")


# --------------------------------------------------------------------------- #
# Modo implementacao
# --------------------------------------------------------------------------- #


class TestImplementacao(BaseCoordenador):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.destino = cls.pasta_nova("impl")
        cls.ckpt = cls.raiz / "checkpoints" / "impl"
        cls.ckpt.mkdir(parents=True)
        with cls.sem_treinar(cls.mapa_impl):
            cls.codigo, cls.saida, cls.erro = cls.rodar("implementacao", cls.destino, cls.ckpt)

    @classmethod
    def tearDownClass(cls):
        remover(cls.destino)
        super().tearDownClass()

    def test_conclui_com_25_unidades_e_so_os_artefatos_exploratorios(self):
        self.assertEqual((self.codigo, self.erro), (0, ""))
        pasta = self.destino / "implementacao"
        self.assertEqual(sorted(p.name for p in pasta.iterdir()),
                         sorted([cd.ARQUIVO_EXECUCAO, cd.ARQUIVO_INVARIANTE, mf.ARQUIVO_MANIFESTO]))
        self.assertEqual([p.name for p in self.destino.iterdir()], ["implementacao"])
        execucao = json.loads((pasta / cd.ARQUIVO_EXECUCAO).read_text(encoding="utf-8"))
        self.assertEqual((execucao["modo"], execucao["unidades"]["executadas"], execucao["unidades"]["falhas"]),
                         ("implementacao", 25, 0))

    def test_manifesto_exploratorio_nunca_elegivel(self):
        m = mf.ler_manifesto(self.destino / "implementacao")
        self.assertEqual((m["modo"], m["elegivel"], m["exploratorio"]), ("implementacao", False, True))
        self.assertTrue(m["motivos_de_inelegibilidade"])
        self.assertEqual(m["configuracao"]["sha256"], cfg.sha256_configuracao())
        mf.verificar_saidas(self.destino / "implementacao", m)
        self.assertEqual(set(m["saidas"]), {cd.ARQUIVO_EXECUCAO, cd.ARQUIVO_INVARIANTE})

    def test_invariante_da_lr_fixa_passa_e_so_compara_parametros(self):
        inv_ = json.loads((self.destino / "implementacao" / cd.ARQUIVO_INVARIANTE).read_text(encoding="utf-8"))
        self.assertTrue(inv_["dentro_da_tolerancia"])
        self.assertEqual(set(inv_["grupos"]), {"coeficientes", "intercepto", "scaler_mean", "scaler_scale"})
        self.assertEqual(inv_["grupos"]["coeficientes"]["tolerancia"], {"rtol": 1e-6, "atol": 1e-8})
        self.assertEqual(inv_["grupos"]["scaler_mean"]["tolerancia"], {"rtol": 1e-10, "atol": 1e-12})
        self.assertEqual(inv_["n_linhas_de_ajuste"], self.ctx_impl.amostra.n)

    def test_nenhuma_metrica_externa_nos_artefatos(self):
        def chaves(v):
            if isinstance(v, dict):
                for k, x in v.items():
                    yield str(k).lower()
                    yield from chaves(x)
            elif isinstance(v, list):
                for x in v:
                    yield from chaves(x)

        manifesto = mf.ler_manifesto(self.destino / "implementacao")
        self.assertEqual(set(manifesto["saidas"]), {cd.ARQUIVO_EXECUCAO, cd.ARQUIVO_INVARIANTE})
        todas = set()
        for nome in (cd.ARQUIVO_EXECUCAO, cd.ARQUIVO_INVARIANTE):
            todas |= set(chaves(json.loads((self.destino / "implementacao" / nome).read_text(encoding="utf-8"))))
        for metrica in ("auc", "brier", "log_loss", "calibracao", "inclinacao", "sensibilidade", "especificidade",
                        "theta", "delta", "contraste", "bootstrap", "roc", "intervalo", "ic95", "medias_internas",
                        "pontuacoes", "predicoes"):
            achadas = [k for k in todas if metrica in k.split("_") or ("_" in metrica and metrica in k)]
            self.assertFalse(achadas, metrica)  # por palavra: "processos" não é "roc"
        pv.varrer_diretorio(self.destino, pv.identificadores_da_amostra(self.ctx_impl.amostra))

    def test_o_caminho_do_implementacao_nao_chama_nenhuma_metrica_externa(self):
        proibidos = [mock.patch.object(mod, nome, side_effect=AssertionError(f"{nome} chamada no implementacao"))
                     for mod, nome in ((co, "consolidar"), (co, "parear"), (bs, "bootstrap"), (fg, "gerar_todas"),
                                       (sl, "selecionar"), (mt.BaseAvaliacao, "de_contexto"),
                                       (mt, "auc_da_dobra"), (cd, "gerar_artefatos_definitivos"))]
        with contextlib.ExitStack() as pilha:
            for p in proibidos:
                pilha.enter_context(p)
            with self.sem_treinar(self.mapa_impl):
                codigo, _, erro = self.rodar("implementacao", self.pasta_nova("d2"), self.checkpoints_novos())
        self.assertEqual((codigo, erro), (0, ""))

    def test_codigo_do_implementacao_so_referencia_executor_invariante_e_manifesto(self):
        arvore = ast.parse(MODULO.read_text(encoding="utf-8"))
        funcao = next(n for n in ast.walk(arvore) if isinstance(n, ast.FunctionDef) and n.name == "_implementacao")
        usados = {n.id for n in ast.walk(funcao) if isinstance(n, ast.Name)} | \
                 {n.attr for n in ast.walk(funcao) if isinstance(n, ast.Attribute)}
        for proibido in ("metricas", "consolidacao", "bootstrap", "selecao", "figuras", "consolidar", "parear",
                         "gerar_todas", "gerar_artefatos_definitivos", "BaseAvaliacao", "relatorio_benchmark",
                         "resumo_md", "tabelas"):
            self.assertNotIn(proibido, usados)
        self.assertTrue({"inv", "mf", "pv"} <= usados)

    def test_retomada_nao_treina_de_novo_as_unidades_concluidas(self):
        with self.proibir_treino():
            codigo, _, erro = self.rodar("implementacao", self.pasta_nova("d3"), self.ckpt)
        self.assertEqual((codigo, erro), (0, ""))

    def test_destino_com_implementacao_anterior_e_recusado_sem_treinar(self):
        with self.proibir_treino():
            codigo, _, erro = self.rodar("implementacao", self.destino, self.ckpt)
        self.assertEqual((codigo, "sobrescrito" in erro), (4, True))

    def test_falha_de_unidade_deixa_a_execucao_incompleta_sem_artefatos(self):
        destino = self.pasta_nova("d4")
        alvo = un.Unidade(0, 2, "svm")

        def falha(ctx, tarefa):
            if tarefa.unidade == alvo:
                return un.FalhaRegistrada(alvo, "falha_tecnica", "FalhaAjuste", "mensagem local")
            return self.mapa_impl[tarefa.unidade]

        with mock.patch.object(un, "executar_tarefa", side_effect=falha):
            codigo, _, erro = self.rodar("implementacao", destino, self.checkpoints_novos())
        self.assertEqual((codigo, "incompleta" in erro), (3, True))
        self.assertEqual(list(destino.iterdir()), [])  # nenhum relatório parcial

    def test_falha_de_integridade_sai_com_4(self):
        destino = self.pasta_nova("d5")
        alvo = un.Unidade(0, 0, "rf")
        with mock.patch.object(un, "executar_tarefa", side_effect=lambda c, t: un.FalhaRegistrada(
                t.unidade, "integridade", "X", "m") if t.unidade == alvo else self.mapa_impl[t.unidade]):
            codigo, _, _ = self.rodar("implementacao", destino, self.checkpoints_novos())
        self.assertEqual(codigo, 4)

    def test_invariante_violado_sai_com_4_e_nada_e_gravado(self):
        ruim = self.aux / "baseline_ruim.json"
        dados_ = json.loads(self.baseline.read_text(encoding="utf-8"))
        dados_["coeficientes"]["intercepto"] += 0.01
        ruim.write_text(json.dumps(dados_), encoding="utf-8")
        destino = self.pasta_nova("d6")
        with self.sem_treinar(self.mapa_impl):
            codigo, _, erro = self.rodar("implementacao", destino, self.checkpoints_novos(), self.deps(baseline=ruim))
        self.assertEqual((codigo, "intercepto" in erro), (4, True))
        self.assertEqual(list(destino.iterdir()), [])

    def test_so_a_particao_train(self):
        self.assertEqual(self.ctx_impl.modo, "implementacao")
        self.assertEqual(dd.PARTICOES_POR_MODO["implementacao"], ("train",))


class TestInvariante(BaseCoordenador):
    def test_tolerancias_sao_as_da_s4(self):
        self.assertEqual(inv.TOLERANCIAS, {"scaler_mean": {"rtol": 1e-10, "atol": 1e-12},
                                           "scaler_scale": {"rtol": 1e-10, "atol": 1e-12},
                                           "coeficientes": {"rtol": 1e-6, "atol": 1e-8},
                                           "intercepto": {"rtol": 1e-6, "atol": 1e-8}})

    def test_fronteira_da_tolerancia_e_termos_fora(self):
        carga = sn.carga_definitiva(90, modo="implementacao")
        X, y = carga.X(), carga.y()
        self.assertTrue(inv.verificar(X, y, self.baseline)["dentro_da_tolerancia"])
        for grupo, chave, delta in (("scaler_mean", None, 1e-6), ("coeficientes", None, 1e-3)):
            dados_ = json.loads(self.baseline.read_text(encoding="utf-8"))
            alvo = dados_["coeficientes"]["scaler_mean" if grupo == "scaler_mean" else "por_feature"]
            nome = next(iter(alvo))
            alvo[nome] += delta
            ruim = self.aux / "b2.json"
            ruim.write_text(json.dumps(dados_), encoding="utf-8")
            with self.subTest(grupo=grupo), self.assertRaisesRegex(inv.InvarianteViolado, "fora da tolerância"):
                inv.verificar(X, y, ruim)

    def test_linhas_e_caracteristicas_diferentes_da_referencia(self):
        carga = sn.carga_definitiva(90, modo="implementacao")
        with self.assertRaises(inv.InvarianteViolado):
            inv.verificar(carga.X()[:-1], carga.y(), self.baseline)
        with self.assertRaisesRegex(inv.InvarianteViolado, "viu"):
            inv.verificar(carga.X()[:-1], carga.y()[:-1], self.baseline)
        dados_ = json.loads(self.baseline.read_text(encoding="utf-8"))
        dados_["features"] = list(reversed(dados_["features"]))
        ruim = self.aux / "b3.json"
        ruim.write_text(json.dumps(dados_), encoding="utf-8")
        with self.assertRaisesRegex(inv.InvarianteViolado, "características"):
            inv.verificar(carga.X(), carga.y(), ruim)

    def test_nao_calcula_predicoes_nem_metricas(self):
        arvore = ast.parse(Path(inv.__file__).read_text(encoding="utf-8"))
        nomes = {n.id for n in ast.walk(arvore) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(arvore) if isinstance(n, ast.Attribute)}
        for proibido in ("predict", "predict_proba", "decision_function", "roc_auc_score", "score"):
            self.assertNotIn(proibido, nomes)

    def test_baseline_real_da_s3_tem_o_formato_esperado(self):
        """O arquivo real é lido só como JSON versionado (nenhum dado de paciente)."""
        real = json.loads(inv.CAMINHO_BASELINE.read_text(encoding="utf-8"))
        self.assertEqual(real["features"], list(C.caracteristicas))
        self.assertEqual(real["prevencao_vazamento"]["scaler_n_samples_seen"], 417)
        self.assertEqual(set(real["coeficientes"]["por_feature"]), set(C.caracteristicas))


# --------------------------------------------------------------------------- #
# Modo definitivo e modo verificar
# --------------------------------------------------------------------------- #


class BaseDefinitivo(BaseCoordenador):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.destino = cls.pasta_nova("def")
        cls.ckpt = cls.raiz / "checkpoints" / "def"
        cls.ckpt.mkdir(parents=True)
        with cls.sem_treinar(cls.mapa_def):
            cls.codigo, cls.saida, cls.erro = cls.rodar("definitivo", cls.destino, cls.ckpt)

    @classmethod
    def tearDownClass(cls):
        remover(cls.destino)
        super().tearDownClass()

    def copia(self) -> Path:
        copia = self.pasta_nova("cp")
        remover(copia)
        shutil.copytree(self.destino, copia)
        self.addCleanup(remover, copia)
        return copia


class TestDefinitivoEVerificar(BaseDefinitivo):
    def test_conclui_e_grava_todos_os_artefatos_da_p9(self):
        self.assertEqual((self.codigo, self.erro), (0, ""))
        raiz = {p.name for p in self.destino.iterdir()}
        self.assertEqual(raiz, set(C.saidas.artefatos_versionados) | {"figuras"})
        for nome in fg.nomes_esperados():
            self.assertTrue((self.destino / nome).is_file(), nome)
        m = mf.ler_manifesto(self.destino)
        self.assertEqual(m["modo"], "definitivo")
        self.assertEqual(set(m["saidas"]) | {mf.ARQUIVO_MANIFESTO},
                         set(C.saidas.artefatos_versionados) | set(fg.nomes_esperados()))
        mf.verificar_saidas(self.destino, m)

    def test_manifesto_sintetico_nao_e_elegivel_e_registra_git_codigo_e_ambiente(self):
        m = mf.ler_manifesto(self.destino)
        self.assertEqual((m["elegivel"], m["oficial"]), (False, False))
        self.assertTrue(any("sintéticos" in x for x in m["motivos_de_inelegibilidade"]))
        self.assertEqual(m["git"]["head"], s4.estado_git(self.raiz).head)
        self.assertTrue(m["git"]["limpo"])
        self.assertIn("src/benchmark_s6/coordenador.py", m["codigo"])
        self.assertEqual(m["ambiente"]["versoes"]["xgboost"], "2.1.4")
        self.assertEqual(m["configuracao"]["sha256"], cfg.sha256_configuracao())

    def test_config_copiada_e_a_canonica_e_o_bootstrap_usa_a_configuracao(self):
        self.assertEqual((self.destino / "config_benchmark.json").read_bytes(), cfg.bytes_arquivo())
        resumo = json.loads((self.destino / "bootstrap_resumo.json").read_text(encoding="utf-8"))
        self.assertEqual((resumo["semente"], resumo["n_replicas"]), (42, 300))  # 300 só no teste

    def test_selecao_e_recalculavel_a_partir_do_relatorio_versionado(self):
        r = json.loads((self.destino / "relatorio_benchmark.json").read_text(encoding="utf-8"))
        refeita = sl.selecionar(r["selecao"]["entrada"])
        self.assertEqual(refeita.como_dict(), r["selecao"]["resultado"])
        s = json.loads((self.destino / "selecao_algoritmo.json").read_text(encoding="utf-8"))
        self.assertEqual(s, r["selecao"]["resultado"])
        self.assertEqual(r["selecao"]["entrada"]["unidades"],
                         {"esperadas": 250, "concluidas": 250, "falhas_tecnicas_pendentes": 0})

    def test_nada_individual_e_versionavel(self):
        ids = pv.identificadores_da_amostra(self.ctx_def.amostra)
        varridos = pv.varrer_diretorio(self.destino, ids)
        self.assertEqual(len(varridos), 15 + 24)  # os 15 artefatos da P9 (com o manifesto) e as 24 figuras e CSVs
        todo = " ".join((self.destino / n).read_text(encoding="utf-8")
                        for n in varridos if n.endswith((".json", ".csv", ".md")))
        self.assertNotIn("nodulo_ids", todo)
        resumo = (self.destino / "resumo_benchmark.md").read_text(encoding="utf-8")
        for vetada in C.saidas.vocabulario_vetado:
            self.assertNotIn(vetada.lower(), resumo.lower())

    def test_relatorio_de_execucao_sem_mensagens_e_com_250_unidades(self):
        r = json.loads((self.destino / cd.ARQUIVO_EXECUCAO).read_text(encoding="utf-8"))
        self.assertEqual((r["unidades"]["executadas"], r["unidades"]["falhas"], r["processos"]), (250, 0, 1))
        self.assertEqual(set(r["duracao_por_modelo"]), set(sn.CODIGOS))
        self.assertEqual(len(r["duracao_por_unidade"]), 250)

    def test_checkpoints_das_250_unidades_ficam_fora_dos_artefatos(self):
        sessao = ck.SessaoCheckpoints(self.ckpt, sn.carga_definitiva())
        self.assertEqual(len(sessao.verificar().concluidas), 250)
        self.assertFalse(any(p.is_relative_to(self.destino) for p in self.ckpt.rglob("*")))

    def test_retomada_total_nao_treina_e_gera_os_mesmos_bytes_deterministicos(self):
        destino = self.pasta_nova("def2")
        self.addCleanup(remover, destino)
        with self.proibir_treino():
            codigo, _, erro = self.rodar("definitivo", destino, self.ckpt)
        self.assertEqual((codigo, erro), (0, ""))
        for nome in set(C.saidas.artefatos_versionados) - {cd.ARQUIVO_EXECUCAO, mf.ARQUIVO_MANIFESTO}:
            self.assertEqual((destino / nome).read_bytes(), (self.destino / nome).read_bytes(), nome)

    # -- modo verificar (reaproveita a MESMA execução de 250 unidades) --------------------------------------
    def verificar(self, destino=None, ckpt=None, **kw):
        return self.rodar("verificar", destino or self.destino, ckpt or self.ckpt, self.deps(**kw))

    def test_reproduz_tudo_com_codigo_0_sem_treinar(self):
        with self.proibir_treino():
            codigo, saida, erro = self.verificar()
        self.assertEqual((codigo, erro), (0, ""))
        self.assertIn("recálculo idêntico", saida)

    def test_checkout_do_windows_com_crlf_em_texto_e_aceito(self):
        copia = self.copia()
        for nome in ("resumo_benchmark.md", "contrastes.csv", "relatorio_benchmark.json"):
            alvo = copia / nome
            alvo.write_bytes(alvo.read_bytes().replace(b"\n", b"\r\n"))
        self.assertEqual(self.verificar(copia)[0], 0)

    def test_artefato_alterado_ausente_ou_extra_sai_com_4(self):
        for editar in (lambda c: (c / "contrastes.csv").write_bytes(b"x\n"),
                       lambda c: (c / "figuras" / "F1.png").write_bytes(b"\x89PNG"),
                       lambda c: (c / "figuras" / "F2.svg").unlink(),
                       lambda c: (c / "extra.csv").write_text("x", encoding="utf-8"),
                       lambda c: (c / cd.ARQUIVO_EXECUCAO).write_text("{}", encoding="utf-8")):
            copia = self.copia()
            editar(copia)
            codigo, _, erro = self.verificar(copia)
            self.assertEqual(codigo, 4, erro)
            remover(copia)

    def test_manifesto_ausente_ou_adulterado(self):
        copia = self.copia()
        (copia / mf.ARQUIVO_MANIFESTO).write_text("{}", encoding="utf-8")
        self.assertEqual(self.verificar(copia)[0], 4)
        (copia / mf.ARQUIVO_MANIFESTO).unlink()
        self.assertEqual(self.verificar(copia)[0], 4)

    def test_artefato_consistente_com_o_manifesto_mas_diferente_do_recalculo_e_recusado(self):
        """Falsificação coerente: muda o CSV E o manifesto; só o recálculo a partir dos checkpoints pega."""
        copia = self.copia()
        alvo = copia / "auc_por_repeticao.csv"
        alvo.write_bytes(alvo.read_bytes().replace(b"0.9", b"0.8", 1))
        m = mf.ler_manifesto(copia)
        m["saidas"] = mf.hashes_de_saidas(copia, m["saidas"])
        (copia / mf.ARQUIVO_MANIFESTO).write_bytes(mf.bytes_json(m))
        codigo, _, erro = self.verificar(copia)
        self.assertEqual((codigo, "recálculo" in erro), (4, True))

    def test_codigo_ou_ambiente_ou_configuracao_alterados(self):
        codigo = dict(ck.hashes_do_codigo())
        codigo["src/benchmark_s6/selecao.py"] = "0" * 64
        self.assertEqual(self.verificar(codigo=lambda: codigo)[0], 4)
        amb = json.loads(json.dumps(ck.descricao_do_ambiente()))
        amb["versoes"]["scikit-learn"] = "9.9"
        self.assertEqual(self.verificar(descricao_do_ambiente=lambda: amb)[0], 4)

    def test_git_sujo_ou_codigo_novo_depois_da_execucao(self):
        head0 = self.git("rev-parse", "HEAD").strip()
        self.addCleanup(self.git, "reset", "-q", "--hard", head0)  # restaura o repositório temporário
        (self.raiz / "novo.txt").write_text("x", encoding="utf-8")
        try:
            codigo, _, erro = self.verificar()
            self.assertEqual((codigo, "suja" in erro), (4, True))
        finally:
            (self.raiz / "novo.txt").unlink()
        self.git("add", "-A")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "nada")
        self.assertEqual(self.verificar()[0], 0)  # o HEAD avançou, mas nada de código mudou
        (self.raiz / "codigo.py").write_text("x = 1\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "código novo")
        codigo, _, erro = self.verificar()
        self.assertEqual((codigo, "fora de reports/sprint6/benchmark" in erro), (4, True))

    def test_checkpoint_incompleto_ou_de_outra_identidade(self):
        ckpt = self.checkpoints_novos()
        codigo, _, erro = self.verificar(ckpt=ckpt)
        self.assertEqual(codigo, 3)  # nenhuma unidade concluída: nada a verificar
        self.assertIn("sem checkpoint", erro)

    def test_verificar_de_manifesto_exploratorio_confere_so_os_hashes(self):
        destino = self.pasta_nova("ex")
        self.addCleanup(remover, destino)
        with self.sem_treinar(self.mapa_impl):
            self.assertEqual(self.rodar("implementacao", destino, self.checkpoints_novos())[0], 0)
        with self.proibir_treino():
            codigo, saida, erro = self.rodar("verificar", destino, self.checkpoints_novos())
        self.assertEqual((codigo, erro), (0, ""))
        self.assertIn("exploratório", saida)
        alvo = destino / "implementacao" / cd.ARQUIVO_INVARIANTE
        alvo.write_text("{}", encoding="utf-8")
        self.assertEqual(self.rodar("verificar", destino, self.checkpoints_novos())[0], 4)


class TestDefinitivoFalhas(BaseCoordenador):
    def test_falha_tecnica_pendente_deixa_incompleta_sem_agregar_selecionar_ou_publicar(self):
        destino = self.pasta_nova("f1")
        self.addCleanup(remover, destino)
        ckpt = self.checkpoints_novos()
        alvo = un.Unidade(3, 1, "rf")

        def falha(ctx, tarefa):
            if tarefa.unidade == alvo:
                return un.FalhaRegistrada(alvo, "falha_tecnica", "FalhaAjuste", "x")
            return self.mapa_def[tarefa.unidade]

        with mock.patch.object(un, "executar_tarefa", side_effect=falha), \
                mock.patch.object(co, "consolidar", side_effect=AssertionError("agregou com FT pendente")), \
                mock.patch.object(sl, "selecionar", side_effect=AssertionError("selecionou com FT pendente")):
            codigo, _, erro = self.rodar("definitivo", destino, ckpt)
        self.assertEqual((codigo, "incompleta" in erro), (3, True))
        self.assertEqual(list(destino.iterdir()), [])
        sessao = ck.SessaoCheckpoints(ckpt, sn.carga_definitiva())
        relatorio = sessao.verificar()
        self.assertEqual((len(relatorio.concluidas), relatorio.falhas), (249, (alvo.rotulo,)))  # preservado

    def test_privacidade_e_vocabulario_barram_a_publicacao_inteira(self):
        original = cd.gerar_artefatos_definitivos
        for texto, esperado in (("LIDC-IDRI-0001", "identificador"), ("resultado significativo", "vocabulário")):
            destino = self.pasta_nova("f2")
            self.addCleanup(remover, destino)

            def contaminado(*a, _t=texto, **k):
                artefatos = original(*a, **k)
                artefatos["resumo_benchmark.md"] += f"\n{_t}\n".encode("utf-8")
                return artefatos

            with self.sem_treinar(self.mapa_def), mock.patch.object(cd, "gerar_artefatos_definitivos", contaminado):
                codigo, _, erro = self.rodar("definitivo", destino, self.checkpoints_novos())
            with self.subTest(texto=texto):
                self.assertEqual((codigo, esperado in erro), (5, True))
                self.assertEqual(list(destino.iterdir()), [])  # nem um arquivo

    def test_git_alterado_durante_a_execucao_impede_a_publicacao(self):
        destino = self.pasta_nova("f3")
        self.addCleanup(remover, destino)
        estados = iter([s4.estado_git(self.raiz), SimpleNamespace(head="b" * 40, branch="x", limpo=True)])
        with self.sem_treinar(self.mapa_def):
            codigo, _, erro = self.rodar("definitivo", destino, self.checkpoints_novos(),
                                         self.deps(estado_git=lambda: next(estados)))
        self.assertEqual((codigo, "mudou durante" in erro), (4, True))
        self.assertEqual(list(destino.iterdir()), [])

    def test_checkpoint_corrompido_aborta_com_4_sem_treinar(self):
        ckpt = self.checkpoints_novos()
        with self.sem_treinar(self.mapa_def):
            self.rodar("definitivo", self.pasta_nova("f4a"), ckpt)
        caminho = next(ckpt.rglob("predicoes.json"))
        caminho.write_bytes(caminho.read_bytes().replace(b"[", b"[ ", 1))
        destino = self.pasta_nova("f4")
        self.addCleanup(remover, destino)
        with self.proibir_treino():
            codigo, _, erro = self.rodar("definitivo", destino, ckpt)
        self.assertEqual((codigo, "retomada abortada" in erro), (4, True))
        self.assertEqual(list(destino.iterdir()), [])

    def test_hash_de_x_divergente_no_worker_vira_falha_de_integridade(self):
        """Com 2 workers ``spawn`` reais, um carregador cujo X difere do principal falha ANTES do treino."""
        destino = self.pasta_nova("f5")
        self.addCleanup(remover, destino)
        dep = self.deps(carregar=lambda m, x: sn.carga_definitiva(90, modo="implementacao"),
                        carregador_dos_workers=lambda m, x: sn.carregar_implementacao_sintetica_x_alterado)
        codigo, _, erro = self.rodar("implementacao", destino, self.checkpoints_novos(), dep, processos=2)
        self.assertEqual((codigo, "incompleta" in erro), (4, True))
        self.assertEqual(list(destino.iterdir()), [])


# --------------------------------------------------------------------------- #
# CLI real em subprocesso, com ``spawn``
# --------------------------------------------------------------------------- #


@unittest.skipIf(GIT is None, "git indisponível")
class TestCliSpawn(unittest.TestCase):
    """O script injeta dados sintéticos e roda o coordenador num PROCESSO NOVO; os workers são ``spawn`` reais."""

    @classmethod
    def setUpClass(cls):
        cls.raiz = Path(tempfile.mkdtemp(prefix="cli-s6-"))
        subprocess.run([GIT, "init", "-q", str(cls.raiz)], check=True, capture_output=True)
        (cls.raiz / ".gitignore").write_text("checkpoints/\n", encoding="utf-8")
        cls.resultados = {}
        for processos in (1, 2):
            ckpt = cls.raiz / "checkpoints" / f"p{processos}"
            ckpt.mkdir(parents=True)
            destino = cls.raiz / f"dest{processos}"
            saida = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), "--modo", "implementacao", "--checkpoints", str(ckpt),
                 "--destino", str(destino), "--processos", str(processos)],
                capture_output=True, text=True, timeout=900, cwd=str(cls.raiz))
            cls.resultados[processos] = (saida, ckpt, destino)

    @classmethod
    def tearDownClass(cls):
        remover(cls.raiz)

    def test_1_e_2_processos_concluem_com_codigo_0(self):
        for processos, (saida, _, destino) in self.resultados.items():
            with self.subTest(processos=processos):
                self.assertEqual(saida.returncode, 0, saida.stderr[-2000:])
                self.assertIn("implementacao concluida: 25 unidades", saida.stdout)
                self.assertTrue((destino / "implementacao" / mf.ARQUIVO_MANIFESTO).is_file())

    def test_o_spawn_nao_reinicia_o_coordenador_nos_workers(self):
        saida = self.resultados[2][0]
        self.assertEqual(saida.stdout.count("implementacao concluida"), 1)  # uma só mensagem: workers não a repetem
        self.assertNotIn("RuntimeError", saida.stderr)
        execucao = json.loads((self.resultados[2][2] / "implementacao" / cd.ARQUIVO_EXECUCAO).read_text("utf-8"))
        self.assertEqual((execucao["processos"], execucao["unidades"]["executadas"]), (2, 25))

    def test_checkpoints_identicos_bit_a_bit_com_1_e_2_processos(self):
        a, b = self.resultados[1][1], self.resultados[2][1]
        resultados = ("predicoes.json", "registro.json")
        arquivos_a = sorted(p.relative_to(a) for p in a.rglob("*.json") if p.name in resultados)
        arquivos_b = sorted(p.relative_to(b) for p in b.rglob("*.json") if p.name in resultados)
        self.assertEqual(arquivos_a, arquivos_b)
        self.assertEqual(len(arquivos_a), 50)
        for rel in arquivos_a:
            self.assertEqual((a / rel).read_bytes(), (b / rel).read_bytes(), str(rel))

    def test_so_o_processo_principal_grava_os_artefatos(self):
        for _, (_, _, destino) in self.resultados.items():
            self.assertEqual(sorted(p.name for p in (destino / "implementacao").iterdir()),
                             sorted([cd.ARQUIVO_EXECUCAO, cd.ARQUIVO_INVARIANTE, mf.ARQUIVO_MANIFESTO]))

    def test_erro_de_uso_do_processo_sai_com_64(self):
        saida = subprocess.run([sys.executable, "-B", "-m", "src.benchmark_s6", "--modo", "implementacao"],
                               capture_output=True, text=True, cwd=str(SCRIPT.parents[2]), timeout=120)
        self.assertEqual(saida.returncode, 64, saida.stderr)

    def test_modo_definitivo_de_producao_sem_aprovacao_ou_config_versionada_nao_executa(self):
        saida = subprocess.run([sys.executable, "-B", "-m", "src.benchmark_s6", "--modo", "definitivo",
                                "--checkpoints", str(self.raiz / "checkpoints")], capture_output=True, text=True,
                               cwd=str(SCRIPT.parents[2]), timeout=300)
        self.assertIn(saida.returncode, (2, 4), saida.stderr)  # bloqueio de aprovação ou integridade; nunca 0
        self.assertNotIn("definitivo concluido", saida.stdout)


if __name__ == "__main__":
    unittest.main()
