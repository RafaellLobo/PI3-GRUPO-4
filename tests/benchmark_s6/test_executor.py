"""Testes de ``src/benchmark_s6/executor.py`` (P9, Seções 6 e 14.3).

Só dados sintéticos (``oficial=False``). Os workers são processos ``spawn``
de verdade: cada um importa este módulo e roda ``carga_no_worker`` uma vez.

As grades de ``svm``, ``rf`` e ``xgb`` são reduzidas SÓ nos testes, trocando
``modelos.especificacao`` por uma especificação com menos valores por
hiperparâmetro (``especificacao_reduzida``): no processo principal com
``mock.patch``, nos workers dentro de ``carga_no_worker``. ``tuning``,
``unidades`` e ``checkpoints`` leem a grade só por ``modelos.especificacao``,
então ajuste, registro e conferência dos checkpoints ficam coerentes. A
configuração de produção não muda.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Optional
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.benchmark_s6 import checkpoints as ck  # noqa: E402
from src.benchmark_s6 import executor as ex  # noqa: E402
from src.benchmark_s6 import modelos  # noqa: E402
from src.benchmark_s6 import tuning  # noqa: E402
from src.benchmark_s6 import unidades as un  # noqa: E402
from test_checkpoints import GIT, _remover  # noqa: E402
from test_unidades import caracteristicas, carga_sintetica  # noqa: E402

MODULO = Path(ex.__file__)
_ESPECIFICACAO_ORIGINAL = modelos.especificacao
#: Valores mantidos por hiperparâmetro (os primeiros, na ordem de simplicidade); ausente = 1.
REDUCOES = {"svm": {"C": 2}, "rf": {"min_samples_leaf": 2}, "xgb": {"max_depth": 2}}
UNIDADES = tuple(un.Unidade(0, k, m) for k in (0, 1, 2) for m in modelos.codigos())


@functools.lru_cache(maxsize=None)
def especificacao_reduzida(codigo: str):
    """Especificação com grade reduzida (só testes): 2 configurações para ``svm``, ``rf`` e ``xgb``."""
    espec = _ESPECIFICACAO_ORIGINAL(codigo)
    if codigo not in REDUCOES:
        return espec
    grade = tuple(dataclasses.replace(p, valores=tuple(p.materializar())[:REDUCOES[codigo].get(p.nome, 1)])
                  for p in espec.grade)
    return dataclasses.replace(espec, grade=grade)


def grades_reduzidas():
    return mock.patch.object(modelos, "especificacao", especificacao_reduzida)


def carga_no_worker(evidencias: str, perturbar_x: bool = False, excecao_em: Optional[str] = None,
                    morrer_em: Optional[str] = None, falhar_carga: bool = False):
    """Carregador dos workers (roda UMA vez por worker, no inicializador).

    Deixa uma evidência ``<pid>-<id>.txt`` em ``evidencias`` (diretório de
    teste, não de checkpoints), reduz as grades, proíbe gravação de
    checkpoints neste processo e, se pedido, injeta falhas.
    """
    Path(evidencias, f"{os.getpid()}-{uuid.uuid4().hex}.txt").write_text("init", encoding="utf-8")
    modelos.especificacao = especificacao_reduzida

    def proibido(*args, **kwargs):
        raise AssertionError("worker tentou gravar checkpoint")

    ck._publicar = proibido
    ck.SessaoCheckpoints.salvar = proibido
    ck.SessaoCheckpoints.registrar_falha = proibido
    if falhar_carga:
        raise RuntimeError("carga indisponível (simulada)")
    if excecao_em or morrer_em:
        original = un.executar_unidade

        def alterada(contexto, unidade):
            if unidade.rotulo == morrer_em:
                os._exit(3)  # morte abrupta do worker
            if unidade.rotulo == excecao_em:
                raise RuntimeError("erro inesperado no worker (simulado)")
            return original(contexto, unidade)

        un.executar_unidade = alterada
    carga = carga_sintetica()
    if perturbar_x:
        X = caracteristicas(carga.amostra())
        X[0, 0] += 1e-9
        carga = carga_sintetica(X=X)
    return carga


@unittest.skipIf(GIT is None, "git indisponível")
class BaseExecutor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raiz = Path(tempfile.mkdtemp(prefix="exec-s6-"))
        subprocess.run([GIT, "init", "-q", str(cls.raiz)], check=True, capture_output=True)
        (cls.raiz / ".gitignore").write_text("checkpoints/\n", encoding="utf-8")
        (cls.raiz / "checkpoints").mkdir()
        cls.carga = carga_sintetica()

    @classmethod
    def tearDownClass(cls):
        _remover(cls.raiz)

    def setUp(self):
        patcher = grades_reduzidas()
        patcher.start()
        self.addCleanup(patcher.stop)

    @classmethod
    def novo(cls, nome: str) -> Path:
        caminho = cls.raiz / "checkpoints" / f"{nome}-{uuid.uuid4().hex[:8]}"
        caminho.mkdir()
        return caminho

    @classmethod
    def sessao(cls, diretorio: Path) -> ck.SessaoCheckpoints:
        return ck.SessaoCheckpoints(diretorio, cls.carga)

    @classmethod
    def carregador(cls, **opcoes):
        evidencias = cls.novo("evidencias")
        return functools.partial(carga_no_worker, str(evidencias), **opcoes), evidencias

    @staticmethod
    def arquivos(sessao: ck.SessaoCheckpoints, unidade: un.Unidade) -> dict:
        pasta = sessao.diretorio / unidade.rotulo
        return {p.name: p.read_bytes() for p in pasta.iterdir() if p.name != ck.ARQUIVO_CONCLUIDA}

    @staticmethod
    def causas(sessao: ck.SessaoCheckpoints, unidade: un.Unidade) -> list:
        pasta = sessao.diretorio / unidade.rotulo
        return [json.loads(p.read_bytes())["falha"] for p in sorted(pasta.glob("falha-*.json"))]


# --------------------------------------------------------------------------- #
# 1 e 4 processos
# --------------------------------------------------------------------------- #


class TestDeterminismo(BaseExecutor):
    """Uma execução com 1 processo e outra com 4 workers ``spawn``, das mesmas 15 unidades."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with grades_reduzidas():
            cls.um = cls.sessao(cls.novo("um"))
            cls.rel_um = ex.executar(cls.um, processos=1, unidades=UNIDADES)
            carregador, cls.evidencias = cls.carregador()
            cls.quatro = cls.sessao(cls.novo("quatro"))
            cls.rel_quatro = ex.executar(cls.quatro, carregador, processos=4, unidades=UNIDADES)

    def test_resultados_identicos_bit_a_bit_com_1_e_4_processos(self):
        self.assertEqual(self.rel_um.executadas, self.rel_quatro.executadas)
        self.assertEqual(self.rel_um.falhas, self.rel_quatro.falhas)
        self.assertGreaterEqual(len(self.rel_um.executadas), 12)  # lr_fixa, lr_tuning, svm, rf (+ xgb se houver)
        for a, b in zip(self.rel_um.resultados, self.rel_quatro.resultados):
            self.assertEqual(a.registro, b.registro)
            for nome in ("posicoes", "y", "pontuacoes"):
                self.assertEqual(getattr(a.privado, nome).tobytes(), getattr(b.privado, nome).tobytes())
            self.assertEqual(a.privado.medias_internas, b.privado.medias_internas)
            self.assertEqual(self.arquivos(self.um, a.unidade), self.arquivos(self.quatro, b.unidade))
        for unidade in UNIDADES:
            self.assertEqual(self.causas(self.um, unidade), self.causas(self.quatro, unidade))

    def test_grade_reduzida_so_nos_testes(self):
        rf = next(r for r in self.rel_quatro.resultados if r.unidade.modelo == "rf")
        self.assertEqual(rf.registro.n_configuracoes, 2)  # o worker também usou a grade reduzida
        self.assertEqual(_ESPECIFICACAO_ORIGINAL("rf").n_configuracoes, 15)

    def test_ordem_canonica_e_estado(self):
        rotulos = tuple(u.rotulo for u in UNIDADES)
        for rel in (self.rel_um, self.rel_quatro):
            self.assertEqual(rel.unidades, rotulos)
            self.assertEqual(tuple(r.unidade.rotulo for r in rel.resultados), rel.executadas)
            self.assertEqual(list(rel.executadas), [r for r in rotulos if r in set(rel.executadas)])
            self.assertEqual(rel.reaproveitadas, ())
            self.assertEqual(rel.concluida, not rel.falhas)
            self.assertEqual(rel.estado, "concluida" if rel.concluida else "incompleta")
        if not modelos.xgboost_disponivel()[0]:  # sem XGBoost: integridade, e a execução não está concluída
            self.assertEqual({f.unidade.modelo for f in self.rel_quatro.falhas}, {"xgb"})
            self.assertTrue(all(f.chave_saida == "integridade" for f in self.rel_quatro.falhas))
            self.assertEqual(self.rel_quatro.estado, "incompleta")

    def test_inicializacao_uma_vez_por_worker(self):
        workers = self.rel_quatro.workers
        self.assertTrue(1 <= len(workers) <= 4)
        self.assertTrue(all(w.inicializacoes == 1 for w in workers))
        self.assertEqual(sum(w.tarefas for w in workers), len(UNIDADES))
        pids = [int(p.name.split("-")[0]) for p in self.evidencias.iterdir()]
        self.assertEqual(len(pids), len(set(pids)))  # uma carga por worker, nunca por tarefa
        self.assertTrue(len(pids) <= 4)
        self.assertLessEqual({w.pid for w in workers}, set(pids))
        self.assertNotIn(os.getpid(), pids)
        self.assertEqual(self.rel_um.workers, ())

    def test_ordem_de_conclusao_nao_altera_nada(self):
        original = ex.as_completed

        def invertido(futuros):
            return list(reversed(list(original(futuros))))

        unidades = tuple(u for u in UNIDADES[:10] if u.modelo in ("lr_fixa", "lr_tuning", "svm"))
        sessao = self.sessao(self.novo("invertido"))
        carregador, _ = self.carregador()
        with mock.patch.object(ex, "as_completed", side_effect=invertido):
            rel = ex.executar(sessao, carregador, processos=3, unidades=unidades)
        self.assertEqual(rel.executadas, tuple(u.rotulo for u in unidades))
        for resultado in rel.resultados:
            self.assertEqual(self.arquivos(sessao, resultado.unidade), self.arquivos(self.um, resultado.unidade))

    def test_retomada_sem_novo_treinamento(self):
        concluidas = [u for u in UNIDADES if u.rotulo in set(self.rel_quatro.executadas)]
        sessao = self.sessao(self.quatro.diretorio.parent.parent)  # processo novo, mesmo diretório
        carregador, evidencias = self.carregador()
        with mock.patch.object(un, "executar_tarefa", side_effect=AssertionError("não deveria treinar")), \
                mock.patch.object(ex, "ProcessPoolExecutor", side_effect=AssertionError("não deveria criar pool")):
            rel = ex.executar(sessao, carregador, processos=4, unidades=concluidas)
        self.assertEqual(rel.reaproveitadas, tuple(u.rotulo for u in concluidas))
        self.assertEqual(rel.executadas, ())
        self.assertTrue(rel.concluida)
        self.assertEqual(list(evidencias.iterdir()), [])
        for a, b in zip(rel.resultados, self.rel_quatro.resultados):
            self.assertEqual(a.privado.pontuacoes.tobytes(), b.privado.pontuacoes.tobytes())

    def test_falhas_anteriores_sao_reexecutadas_e_concluidas_nao(self):
        d = self.novo("refazer")
        shutil.copytree(self.um.diretorio.parent, d / ck.SUBDIRETORIO)
        sessao = self.sessao(d)
        with mock.patch.object(un, "executar_tarefa", wraps=un.executar_tarefa) as executar:
            rel = ex.executar(sessao, processos=1, unidades=UNIDADES)
        refeitas = [c.args[1].unidade.rotulo for c in executar.call_args_list]
        self.assertEqual(refeitas, [f.unidade.rotulo for f in self.rel_um.falhas])
        self.assertEqual(rel.reaproveitadas, self.rel_um.executadas)


# --------------------------------------------------------------------------- #
# Retomada, integridade e falhas
# --------------------------------------------------------------------------- #


U_FIXA = tuple(un.Unidade(0, k, "lr_fixa") for k in range(3))


class TestRetomada(BaseExecutor):
    def test_interrupcao_reexecuta_somente_a_incompleta(self):
        d = self.novo("interrompida")
        sessao = self.sessao(d)
        alvo = sessao.diretorio / U_FIXA[1].rotulo / ck.ARQUIVO_CONCLUIDA
        original = os.replace

        def replace(origem, destino):
            if Path(destino) == alvo:
                raise KeyboardInterrupt  # interrupção ao gravar a segunda unidade
            return original(origem, destino)

        with mock.patch.object(ck.os, "replace", side_effect=replace), self.assertRaises(KeyboardInterrupt):
            ex.executar(sessao, processos=1, unidades=U_FIXA)
        primeira = self.arquivos(sessao, U_FIXA[0])
        retomada = self.sessao(d)
        self.assertEqual(retomada.verificar(U_FIXA).incompletas, (U_FIXA[1].rotulo,))
        with mock.patch.object(un, "executar_tarefa", wraps=un.executar_tarefa) as executar:
            rel = ex.executar(retomada, processos=1, unidades=U_FIXA)
        self.assertEqual([c.args[1].unidade for c in executar.call_args_list], list(U_FIXA[1:]))
        self.assertEqual(rel.reaproveitadas, (U_FIXA[0].rotulo,))
        self.assertEqual(rel.executadas, (U_FIXA[1].rotulo, U_FIXA[2].rotulo))
        self.assertTrue(rel.concluida)
        self.assertEqual(self.arquivos(retomada, U_FIXA[0]), primeira)
        self.assertTrue((retomada.diretorio.parent / ck.QUARENTENA).is_dir())

    def test_checkpoint_corrompido_aborta_antes_de_treinar(self):
        sessao = self.sessao(self.novo("corrompido"))
        ex.executar(sessao, processos=1, unidades=U_FIXA[:1])
        caminho = sessao.diretorio / U_FIXA[0].rotulo / ck.ARQUIVO_PREDICOES
        caminho.write_bytes(caminho.read_bytes().replace(b"[", b"[ ", 1))
        carregador, evidencias = self.carregador()
        for processos in (1, 3):
            with self.subTest(processos=processos), \
                    mock.patch.object(un, "executar_tarefa", side_effect=AssertionError("não deveria treinar")), \
                    mock.patch.object(ex, "ProcessPoolExecutor", side_effect=AssertionError("não deveria criar pool")):
                with self.assertRaisesRegex(ck.CheckpointCorrompido, "retomada abortada"):
                    ex.executar(self.sessao(sessao.diretorio.parent.parent), carregador, processos=processos,
                                unidades=U_FIXA)
        self.assertEqual(list(evidencias.iterdir()), [])
        self.assertEqual(sorted(p.name for p in sessao.diretorio.iterdir()), [U_FIXA[0].rotulo, ck.ARQUIVO_SESSAO])


class TestIntegridade(BaseExecutor):
    def test_tarefa_leva_os_hashes_de_x_e_da_amostra_da_identidade(self):
        sessao = self.sessao(self.novo("hashes"))
        tarefa = sessao.contexto.tarefa(U_FIXA[0])
        entradas = sessao.identidade_sessao["entradas"]
        self.assertEqual(tarefa.sha256_caracteristicas, entradas["sha256_caracteristicas"])
        self.assertEqual(tarefa.sha256_amostra, entradas["sha256_amostra"])
        self.assertEqual(tarefa.sha256_caracteristicas, un.sha256_caracteristicas(self.carga.X()))

    def test_x_ou_amostra_divergente_falha_antes_do_treinamento(self):
        X = caracteristicas(self.carga.amostra())
        X[5, 2] = np.nextafter(X[5, 2], np.inf)  # um único ULP
        outro = un.ContextoExecucao.de_dados(carga_sintetica(X=X))
        tarefa = self.sessao(self.novo("x")).contexto.tarefa(U_FIXA[0])
        self.assertEqual(outro.plano.fingerprint, tarefa.fingerprint_plano)  # o plano não percebe a diferença
        casos = {"X": (outro, tarefa, "características"),
                 "amostra": (un.ContextoExecucao.de_dados(self.carga),
                             dataclasses.replace(tarefa, sha256_amostra="0" * 64), "amostra")}
        for nome, (contexto, t, padrao) in casos.items():
            with self.subTest(caso=nome), \
                    mock.patch.object(un.tuning, "ajustar", side_effect=AssertionError("não deveria treinar")):
                falha = un.executar_tarefa(contexto, t)
                self.assertIsInstance(falha, un.FalhaRegistrada)
                self.assertEqual(falha.chave_saida, "integridade")
                self.assertIn(padrao, falha.mensagem)

    def test_worker_com_x_divergente_registra_falha_e_nao_conclui(self):
        sessao = self.sessao(self.novo("worker-x"))
        carregador, _ = self.carregador(perturbar_x=True)
        rel = ex.executar(sessao, carregador, processos=2, unidades=U_FIXA[:2])
        self.assertEqual(rel.executadas, ())
        self.assertEqual(rel.estado, "incompleta")
        self.assertEqual([f.chave_saida for f in rel.falhas], ["integridade", "integridade"])
        self.assertTrue(all("características (X)" in f.mensagem for f in rel.falhas))
        self.assertEqual(sessao.verificar(U_FIXA[:2]).falhas, tuple(u.rotulo for u in U_FIXA[:2]))

    def test_retorno_que_nao_corresponde_a_tarefa_vira_falha_de_integridade(self):
        sessao = self.sessao(self.novo("retorno"))
        de_outra = un.executar_unidade(sessao.contexto, U_FIXA[2])
        retornos = {"outra unidade": de_outra, "tipo": "resultado",
                    "falha de outra unidade": un.FalhaRegistrada(U_FIXA[2], "falha_tecnica", "X", "m"),
                    "chave inválida": un.FalhaRegistrada(U_FIXA[0], "concluido", "X", "m")}
        for n, (nome, retorno) in enumerate(retornos.items(), start=1):
            with self.subTest(caso=nome), mock.patch.object(un, "executar_tarefa", return_value=retorno):
                rel = ex.executar(sessao, processos=1, unidades=U_FIXA[:1])
                self.assertEqual(rel.executadas, ())
                falha, = rel.falhas
                self.assertEqual((falha.unidade, falha.chave_saida, falha.categoria),
                                 (U_FIXA[0], "integridade", "RetornoDivergente"))
                self.assertEqual(len(self.causas(sessao, U_FIXA[0])), n)
        self.assertEqual(sessao.estado(U_FIXA[2]), ck.ESTADO_AUSENTE)  # nada gravado no slot alheio


class TestFalhasDoWorker(BaseExecutor):
    def test_excecao_no_worker_vira_falha_tecnica_e_as_demais_concluem(self):
        sessao = self.sessao(self.novo("excecao"))
        carregador, _ = self.carregador(excecao_em=U_FIXA[1].rotulo)
        rel = ex.executar(sessao, carregador, processos=2, unidades=U_FIXA)
        falha, = rel.falhas
        self.assertEqual((falha.unidade, falha.chave_saida, falha.categoria),
                         (U_FIXA[1], "falha_tecnica", "RuntimeError"))
        self.assertEqual(rel.executadas, (U_FIXA[0].rotulo, U_FIXA[2].rotulo))
        self.assertFalse(rel.concluida)
        self.assertEqual(sessao.estado(U_FIXA[1]), ck.ESTADO_FALHOU)

    def test_morte_do_worker_e_registrada_e_a_retomada_refaz_so_as_afetadas(self):
        d = self.novo("morte")
        sessao = self.sessao(d)
        carregador, _ = self.carregador(morrer_em=U_FIXA[1].rotulo)
        rel = ex.executar(sessao, carregador, processos=2, unidades=U_FIXA)
        falhas = {f.unidade: f for f in rel.falhas}
        self.assertIn(U_FIXA[1], falhas)
        self.assertEqual(falhas[U_FIXA[1]].categoria, "BrokenProcessPool")
        self.assertTrue(all(f.chave_saida == "falha_tecnica" for f in falhas.values()))
        self.assertEqual(set(rel.executadas) | {u.rotulo for u in falhas}, {u.rotulo for u in U_FIXA})
        self.assertEqual(rel.estado, "incompleta")

        carregador, _ = self.carregador()
        retomada = ex.executar(self.sessao(d), carregador, processos=2, unidades=U_FIXA)
        self.assertEqual(set(retomada.executadas), {u.rotulo for u in falhas})
        self.assertEqual(retomada.reaproveitadas, rel.executadas)
        self.assertTrue(retomada.concluida)

    def test_falha_na_inicializacao_do_worker(self):
        sessao = self.sessao(self.novo("init"))
        carregador, evidencias = self.carregador(falhar_carga=True)
        rel = ex.executar(sessao, carregador, processos=2, unidades=U_FIXA[:2])
        self.assertEqual(rel.executadas, ())
        self.assertTrue(all(f.categoria == "RuntimeError" and "inicialização do worker" in f.mensagem
                            for f in rel.falhas))
        self.assertEqual(len(rel.falhas), 2)
        self.assertTrue(1 <= len(list(evidencias.iterdir())) <= 2)


class TestEntradas(BaseExecutor):
    def test_argumentos_invalidos(self):
        sessao = self.sessao(self.novo("args"))
        carregador, _ = self.carregador()
        for processos in (0, -1, 1.0, True, "2"):
            with self.subTest(processos=processos), self.assertRaises(ValueError):
                ex.executar(sessao, carregador, processos=processos)
        with self.assertRaisesRegex(TypeError, "carregador"):
            ex.executar(sessao, None, processos=2)
        with self.assertRaisesRegex(TypeError, "serializável"):
            ex.executar(sessao, lambda: carga_sintetica(), processos=2)
        with self.assertRaisesRegex(ValueError, "fora do plano"):
            ex.executar(sessao, unidades=[un.Unidade(1, 0, "lr_fixa")])
        with self.assertRaisesRegex(ValueError, "repetidas"):
            ex.executar(sessao, unidades=[U_FIXA[0], U_FIXA[0]])
        with self.assertRaises(TypeError):
            ex.executar(sessao.contexto)
        self.assertEqual(list(sessao.diretorio.iterdir()), [sessao.diretorio / ck.ARQUIVO_SESSAO])

    def test_ponto_de_entrada_tem_a_guarda_main(self):
        """Sem a guarda, o ``spawn`` reiniciaria o coordenador em cada worker (P9, Seção 14.3)."""
        fonte = (MODULO.parent / "__main__.py").read_text(encoding="utf-8")
        arvore = ast.parse(fonte)
        chamadas = [n for n in arvore.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
        self.assertEqual(chamadas, [])  # nenhuma chamada solta no nível do módulo
        guardas = [n for n in arvore.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) in ("__name__ == '__main__'", "__name__ == \"__main__\"")]
        self.assertEqual(len(guardas), 1)
        self.assertIn("sys.exit(main())", ast.unparse(guardas[0]))
        saida = subprocess.run([sys.executable, "-B", "-c", "import src.benchmark_s6.__main__"],
                               cwd=MODULO.parents[2], capture_output=True, timeout=120)
        self.assertEqual(saida.returncode, 0, saida.stderr)  # importar não executa o coordenador

    def test_workers_nao_gravam_nem_medem(self):
        arvore = ast.parse(MODULO.read_text(encoding="utf-8"))
        funcoes = {n.name: n for n in ast.walk(arvore) if isinstance(n, ast.FunctionDef)}
        for nome in ("_inicializar_worker", "_executar_no_worker"):
            usados = {n.id for n in ast.walk(funcoes[nome]) if isinstance(n, ast.Name)} | \
                     {n.attr for n in ast.walk(funcoes[nome]) if isinstance(n, ast.Attribute)}
            for proibido in ("salvar", "registrar_falha", "SessaoCheckpoints", "sessao", "open", "write_bytes"):
                self.assertNotIn(proibido, usados, nome)
        modulos = {a.name for n in ast.walk(arvore) if isinstance(n, ast.Import) for a in n.names} | \
                  {n.module or "" for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom)}
        self.assertNotIn("sklearn.metrics", modulos)
        self.assertEqual(ex.CONTEXTO_MULTIPROCESSAMENTO, "spawn")


if __name__ == "__main__":
    unittest.main()
