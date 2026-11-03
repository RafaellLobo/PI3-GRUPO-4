"""Testes de ``src/benchmark_s6/checkpoints.py`` (P9, Seções 6 e 14.5).

Só dados sintéticos (``oficial=False``) em memória; nenhum dado real é lido.
Os checkpoints vão para uma pasta ignorada de um repositório git temporário
(nesta máquina o home é versionado, então um temporário comum seria recusado,
corretamente). Cada teste usa um diretório novo.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import pickle
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.benchmark_s6 import adaptadores_s4 as s4  # noqa: E402
from src.benchmark_s6 import checkpoints as ck  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import dados  # noqa: E402
from src.benchmark_s6 import unidades as un  # noqa: E402
from test_unidades import caracteristicas, carga_sintetica  # noqa: E402

REPO_ROOT = s4.REPO_ROOT
MODULO = REPO_ROOT / "src" / "benchmark_s6" / "checkpoints.py"
GIT = shutil.which("git")
U0 = un.Unidade(0, 0, "lr_fixa")
U1 = un.Unidade(0, 1, "svm")
U2 = un.Unidade(0, 2, "lr_fixa")


def _remover(raiz: Path) -> None:
    def liberar(funcao, caminho, _):
        os.chmod(caminho, stat.S_IWRITE)
        funcao(caminho)
    shutil.rmtree(raiz, onerror=liberar)


def ler(caminho: Path) -> dict:
    return json.loads(caminho.read_bytes())


def gravar(caminho: Path, valor) -> None:
    caminho.write_bytes(ck._bytes_json(valor))


def reassinar(pasta: Path) -> None:
    """Recalcula hashes e ``sha256_conteudo`` do ``concluida.json``: falsificação COERENTE."""
    controle = ler(pasta / ck.ARQUIVO_CONCLUIDA)
    for nome in (ck.ARQUIVO_PREDICOES, ck.ARQUIVO_REGISTRO):
        controle["arquivos"][nome] = ck._hash_arquivo((pasta / nome).read_bytes())
    controle["sha256_identidade"] = s4.sha256_canonico(controle["identidade"])
    controle["sha256_conteudo"] = s4.sha256_canonico({k: v for k, v in controle.items() if k != "sha256_conteudo"})
    gravar(pasta / ck.ARQUIVO_CONCLUIDA, controle)


def carga_com(X: np.ndarray = None, oficial: bool = False, semente: int = 0) -> dados.DadosBenchmark:
    base = carga_sintetica(semente=semente, X=X)
    return dados.DadosBenchmark(base.modo, base.particoes, oficial, base.caracteristicas, base.sha256_entradas,
                                base.contagens, None, base.frame())


@unittest.skipIf(GIT is None, "git indisponível")
class BaseCheckpoints(unittest.TestCase):
    """Carga sintética, resultados reais de três unidades (calculados uma vez) e um repositório temporário."""

    @classmethod
    def setUpClass(cls):
        cls.raiz = Path(tempfile.mkdtemp(prefix="ckpt-s6-"))
        subprocess.run([GIT, "init", "-q", str(cls.raiz)], check=True, capture_output=True)
        (cls.raiz / ".gitignore").write_text("checkpoints/\n", encoding="utf-8")
        (cls.raiz / "checkpoints").mkdir()
        cls.carga = carga_sintetica()
        ctx = un.ContextoExecucao.de_dados(cls.carga)
        cls.resultados = {u: un.executar_unidade(ctx, u) for u in (U0, U1, U2)}

    @classmethod
    def tearDownClass(cls):
        _remover(cls.raiz)

    def diretorio(self) -> Path:
        caminho = self.raiz / "checkpoints" / uuid.uuid4().hex[:12]
        caminho.mkdir()
        return caminho

    def sessao(self, diretorio: Path = None, carga=None) -> ck.SessaoCheckpoints:
        return ck.SessaoCheckpoints(diretorio or self.diretorio(), carga or self.carga)

    def sessao_com(self, *unidades_salvas) -> ck.SessaoCheckpoints:
        sessao = self.sessao()
        for u in unidades_salvas:
            sessao.salvar(self.resultados[u])
        return sessao


# --------------------------------------------------------------------------- #
# Diretório local
# --------------------------------------------------------------------------- #


class TestDiretorio(BaseCheckpoints):
    def test_repositorio_do_projeto_e_data_sempre_recusados(self):
        for caminho in (REPO_ROOT, REPO_ROOT / "data", REPO_ROOT / "src", REPO_ROOT / ".venv"):
            if caminho.is_dir():
                with self.subTest(caminho=caminho.name), self.assertRaises(ck.DiretorioNaoLocal):
                    ck.exigir_diretorio_local(caminho)
        with self.assertRaises(ck.DiretorioInvalido):
            ck.exigir_diretorio_local(self.raiz / "nao_existe")

    def test_outro_repositorio_so_se_ignorado(self):
        livre = self.raiz / "versionavel"
        livre.mkdir(exist_ok=True)
        with self.assertRaisesRegex(ck.DiretorioNaoLocal, "não é ignorado"):
            ck.SessaoCheckpoints(livre, self.carga)
        self.assertEqual(list(livre.iterdir()), [])  # nada gravado no diretório recusado
        ignorado = self.diretorio()
        self.assertEqual(ck.exigir_diretorio_local(ignorado), ignorado.resolve())

    def test_falha_do_git_nunca_libera_e_fora_de_repositorio_e_aceito(self):
        ignorado = self.diretorio()
        with mock.patch.object(ck.subprocess, "run", return_value=mock.Mock(returncode=128)):
            with self.assertRaisesRegex(ck.DiretorioNaoLocal, "falha do git"):
                ck.exigir_diretorio_local(ignorado)
        with mock.patch.object(ck.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ck.DiretorioNaoLocal, "git indisponível"):
                ck.exigir_diretorio_local(ignorado)
        # Nesta máquina o home é um repositório; "fora de qualquer repositório" é simulado.
        with mock.patch.object(ck, "_dentro_de_repositorio", return_value=False), \
                mock.patch.object(ck.subprocess, "run", side_effect=AssertionError("git não deveria ser chamado")):
            self.assertEqual(ck.exigir_diretorio_local(ignorado), ignorado.resolve())

    def test_chaves_de_saida_existem_na_configuracao(self):
        codigos = cfg.CONFIGURACAO.execucao.codigos_saida
        for classe in (ck.ErroCheckpoint, ck.CheckpointCorrompido, ck.CheckpointIncompativel, ck.SobrescritaRecusada,
                       ck.CheckpointAusente, ck.CheckpointNaoConcluido, ck.DiretorioInvalido, ck.DiretorioNaoLocal,
                       ck.DadosIndividuaisNoRegistro):
            self.assertIn(classe.chave_saida, codigos, classe.__name__)
        self.assertEqual(ck.CheckpointIncompativel.chave_saida, "integridade")
        self.assertEqual(ck.DiretorioNaoLocal.chave_saida, "privacidade_ou_vocabulario")


# --------------------------------------------------------------------------- #
# Gravação e retomada
# --------------------------------------------------------------------------- #


class TestGravacaoERetomada(BaseCheckpoints):
    def test_salvar_e_carregar_reproduz_o_resultado_bit_a_bit(self):
        for u in (U0, U1):  # predict_proba e decision_function
            with self.subTest(unidade=u.rotulo):
                sessao = self.sessao_com(u)
                original, carregado = self.resultados[u], sessao.carregar(u)
                self.assertEqual(sessao.estado(u), ck.ESTADO_CONCLUIDA)
                self.assertEqual(carregado.registro, original.registro)
                for nome in ("posicoes", "y", "pontuacoes"):
                    a, b = getattr(carregado.privado, nome), getattr(original.privado, nome)
                    self.assertEqual((a.dtype, a.tobytes()), (b.dtype, b.tobytes()), nome)
                    self.assertFalse(a.flags.writeable, nome)
                self.assertEqual(carregado.privado.nodulo_ids, original.privado.nodulo_ids)
                self.assertEqual(carregado.privado.medias_internas, original.privado.medias_internas)

    def test_estrutura_em_disco(self):
        sessao = self.sessao_com(U0)
        arquivos = sorted(p.relative_to(sessao.diretorio).as_posix() for p in sessao.diretorio.rglob("*"))
        self.assertEqual(arquivos, ["r00-k0-lr_fixa", "r00-k0-lr_fixa/concluida.json", "r00-k0-lr_fixa/predicoes.json",
                                    "r00-k0-lr_fixa/registro.json", "sessao.json"])
        self.assertEqual(sessao.diretorio.parent.name, ck.SUBDIRETORIO)

    def test_retomada_recupera_sem_treinar_de_novo(self):
        d = self.diretorio()
        primeiro, reaproveitado = ck.obter_unidade(self.sessao(d), U0)
        self.assertFalse(reaproveitado)
        with mock.patch.object(un, "executar_tarefa", side_effect=AssertionError("não deveria treinar")), \
                mock.patch.object(un, "executar_unidade", side_effect=AssertionError("não deveria treinar")):
            retomado, reaproveitado = ck.obter_unidade(self.sessao(d), U0)
        self.assertTrue(reaproveitado)
        self.assertEqual(retomado.registro, primeiro.registro)
        self.assertEqual(retomado.privado.pontuacoes.tobytes(), primeiro.privado.pontuacoes.tobytes())

    def test_bytes_dos_resultados_sao_deterministicos(self):
        a, b = self.sessao_com(U0), self.sessao_com(U0)
        for nome in (ck.ARQUIVO_PREDICOES, ck.ARQUIVO_REGISTRO):
            self.assertEqual((a.diretorio / U0.rotulo / nome).read_bytes(),
                             (b.diretorio / U0.rotulo / nome).read_bytes())
        self.assertEqual((a.diretorio / ck.ARQUIVO_SESSAO).read_bytes(), (b.diretorio / ck.ARQUIVO_SESSAO).read_bytes())

    def test_resultado_vindo_de_worker_por_pickle(self):
        copia = pickle.loads(pickle.dumps(self.resultados[U0]))
        sessao = self.sessao()
        sessao.salvar(copia)
        self.assertEqual(sessao.carregar(U0).registro, self.resultados[U0].registro)

    def test_sem_sobrescrita(self):
        sessao = self.sessao_com(U0)
        antes = {p.name: p.read_bytes() for p in (sessao.diretorio / U0.rotulo).iterdir()}
        with self.assertRaises(ck.SobrescritaRecusada):
            sessao.salvar(self.resultados[U0])
        with self.assertRaises(ck.SobrescritaRecusada):
            sessao.registrar_falha(un.FalhaRegistrada(U0, "falha_tecnica", "X", "m"))
        self.assertEqual({p.name: p.read_bytes() for p in (sessao.diretorio / U0.rotulo).iterdir()}, antes)

    def test_verificar_classifica_as_unidades_do_modo(self):
        sessao = self.sessao_com(U0, U1)
        sessao.registrar_falha(un.FalhaRegistrada(U2, "falha_tecnica", "FalhaAjuste", "m"))
        relatorio = self.sessao(sessao.diretorio.parent.parent).verificar()
        self.assertEqual(relatorio.concluidas, (U0.rotulo, U1.rotulo))
        self.assertEqual(relatorio.falhas, (U2.rotulo,))
        self.assertEqual(relatorio.n_esperadas, 25)
        self.assertEqual(len(relatorio.a_executar), 23)
        self.assertIn(U2.rotulo, relatorio.a_executar)
        self.assertNotIn(U0.rotulo, relatorio.a_executar)


# --------------------------------------------------------------------------- #
# Falhas
# --------------------------------------------------------------------------- #


class TestFalhas(BaseCheckpoints):
    def test_falha_nunca_e_resultado_concluido(self):
        sessao = self.sessao()
        with self.assertRaises(ck.CheckpointAusente):
            sessao.carregar(U0)
        sessao.registrar_falha(un.FalhaRegistrada(U0, "falha_tecnica", "FalhaAjuste", "ajuste: erro"))
        self.assertEqual(sessao.estado(U0), ck.ESTADO_FALHOU)
        with self.assertRaises(ck.CheckpointNaoConcluido):
            sessao.carregar(U0)
        controle = sessao.salvar(self.resultados[U0])  # repetir depois de FT é permitido (P9, Seção 6)
        self.assertEqual(controle["metadados"]["tentativa"], 2)
        self.assertEqual(sessao.estado(U0), ck.ESTADO_CONCLUIDA)
        self.assertEqual(ler(sessao.diretorio / U0.rotulo / "falha-001.json")["falha"]["categoria"], "FalhaAjuste")

    def test_obter_unidade_grava_a_falha_e_reexecuta_na_proxima(self):
        d = self.diretorio()
        falha = un.FalhaRegistrada(U0, "falha_tecnica", "FalhaUnidade", "predição: erro")
        with mock.patch.object(un, "executar_tarefa", return_value=falha):
            resultado, reaproveitado = ck.obter_unidade(self.sessao(d), U0)
        self.assertIs(resultado, falha)
        self.assertFalse(reaproveitado)
        sessao = self.sessao(d)
        self.assertEqual(sessao.estado(U0), ck.ESTADO_FALHOU)
        resultado, reaproveitado = ck.obter_unidade(sessao, U0)
        self.assertIsInstance(resultado, un.ResultadoUnidade)
        self.assertFalse(reaproveitado)
        self.assertEqual(sessao.verificar([U0]).concluidas, (U0.rotulo,))

    def test_tipos_trocados_sao_recusados(self):
        sessao = self.sessao()
        with self.assertRaises(TypeError):
            sessao.salvar(un.FalhaRegistrada(U0, "falha_tecnica", "X", "m"))
        with self.assertRaises(TypeError):
            sessao.registrar_falha(self.resultados[U0])
        with self.assertRaises(TypeError):
            ck.SessaoCheckpoints(self.diretorio(), un.ContextoExecucao.de_dados(self.carga))

    def test_falha_fora_de_sequencia_e_corrompida(self):
        sessao = self.sessao()
        sessao.registrar_falha(un.FalhaRegistrada(U0, "falha_tecnica", "X", "m"))
        pasta = sessao.diretorio / U0.rotulo
        (pasta / "falha-001.json").rename(pasta / "falha-002.json")
        with self.assertRaisesRegex(ck.CheckpointCorrompido, "sequência"):
            sessao.estado(U0)


# --------------------------------------------------------------------------- #
# Corrupção
# --------------------------------------------------------------------------- #


class TestCorrupcao(BaseCheckpoints):
    def exigir_corrompido(self, sessao, padrao: str = ""):
        with self.assertRaisesRegex(ck.CheckpointCorrompido, padrao):
            sessao.estado(U0)
        with self.assertRaises(ck.CheckpointCorrompido):
            sessao.carregar(U0)
        with self.assertRaisesRegex(ck.CheckpointCorrompido, "retomada abortada"):
            sessao.verificar()
        with self.assertRaises(ck.CheckpointCorrompido):
            ck.obter_unidade(sessao, U0)  # não treina por cima de um slot corrompido

    def test_byte_alterado_em_qualquer_arquivo(self):
        for nome in (ck.ARQUIVO_PREDICOES, ck.ARQUIVO_REGISTRO, ck.ARQUIVO_CONCLUIDA):
            with self.subTest(arquivo=nome):
                sessao = self.sessao_com(U0)
                caminho = sessao.diretorio / U0.rotulo / nome
                conteudo = bytearray(caminho.read_bytes())
                conteudo[len(conteudo) // 2] ^= 0x01
                caminho.write_bytes(bytes(conteudo))
                self.exigir_corrompido(sessao)

    def test_json_invalido_ou_fora_da_forma_canonica(self):
        variantes = {
            "truncado": lambda b: b[: len(b) // 2],
            "vazio": lambda b: b"",
            "indentado": lambda b: json.dumps(json.loads(b), indent=1, sort_keys=True).encode("utf-8"),
            "crlf": lambda b: b.replace(b"\n", b"\r\n"),
            "chave_duplicada": lambda b: b[:-2] + b',"formato":"x"}\n',
            "nan": lambda b: b.replace(b'"tentativa":1', b'"tentativa":NaN'),
        }
        for nome, alterar in variantes.items():
            with self.subTest(variante=nome):
                sessao = self.sessao_com(U0)
                caminho = sessao.diretorio / U0.rotulo / ck.ARQUIVO_CONCLUIDA
                caminho.write_bytes(alterar(caminho.read_bytes()))
                self.exigir_corrompido(sessao)

    def test_falsificacao_coerente_e_detectada_pela_conferencia_com_a_sessao(self):
        def trocar_id(p):
            p["nodulo_ids"][0] = p["nodulo_ids"][1]

        def probabilidade_invalida(p):
            p["pontuacoes"][0] = 1.5

        def posicao_trocada(p):
            p["posicoes"][0] = p["posicoes"][0] + 1

        def rotulo_trocado(p):
            p["y"][0] = 1 - p["y"][0]

        for alterar in (trocar_id, probabilidade_invalida, posicao_trocada, rotulo_trocado):
            with self.subTest(falsificacao=alterar.__name__):
                sessao = self.sessao_com(U0)
                pasta = sessao.diretorio / U0.rotulo
                predicoes = ler(pasta / ck.ARQUIVO_PREDICOES)
                alterar(predicoes)
                gravar(pasta / ck.ARQUIVO_PREDICOES, predicoes)
                reassinar(pasta)  # hashes coerentes: só a conferência semântica pega
                self.exigir_corrompido(sessao)

    def test_registro_falsificado_de_forma_coerente(self):
        sessao = self.sessao_com(U0)
        pasta = sessao.diretorio / U0.rotulo
        registro = ler(pasta / ck.ARQUIVO_REGISTRO)
        registro["contagem_avaliacao"]["nodulos"] += 1
        gravar(pasta / ck.ARQUIVO_REGISTRO, registro)
        reassinar(pasta)
        with self.assertRaisesRegex(ck.CheckpointCorrompido, "contagens"):
            sessao.carregar(U0)

    def test_arquivo_desconhecido_e_slot_que_nao_e_diretorio(self):
        sessao = self.sessao_com(U0)
        (sessao.diretorio / U0.rotulo / "notas.txt").write_text("x", encoding="utf-8")
        self.exigir_corrompido(sessao, "desconhecidos")
        sessao = self.sessao()
        (sessao.diretorio / U0.rotulo).write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(ck.CheckpointCorrompido, "não é diretório"):
            sessao.estado(U0)

    def test_arquivo_ausente_com_conclusao_presente(self):
        sessao = self.sessao_com(U0)
        (sessao.diretorio / U0.rotulo / ck.ARQUIVO_REGISTRO).unlink()
        self.exigir_corrompido(sessao, "ausente")

    def test_entrada_inesperada_aborta_a_verificacao(self):
        sessao = self.sessao_com(U0)
        (sessao.diretorio / "r09-k4-xgb").mkdir()  # unidade do definitivo num diretório do implementacao
        with self.assertRaisesRegex(ck.CheckpointCorrompido, "não esperadas"):
            sessao.verificar()
        self.assertEqual(sessao.estado(U0), ck.ESTADO_CONCLUIDA)


# --------------------------------------------------------------------------- #
# Gravação interrompida
# --------------------------------------------------------------------------- #


class TestInterrupcao(BaseCheckpoints):
    def interromper(self, sessao, nome: str, resultado=None):
        """Falha ao publicar ``nome`` no slot (``salvar``: predições, registro e conclusão, nessa ordem)."""
        resultado = resultado or self.resultados[U0]
        pasta, original = sessao.diretorio / resultado.unidade.rotulo, os.replace

        def replace(origem, destino):
            if Path(destino).parent == pasta and Path(destino).name == nome:
                raise OSError("disco cheio (simulado)")
            return original(origem, destino)

        with mock.patch.object(ck.os, "replace", side_effect=replace):
            with self.assertRaisesRegex(OSError, "simulado"):
                sessao.salvar(resultado)

    @staticmethod
    def conteudo(pasta: Path) -> dict:
        return {p.name: p.read_bytes() for p in pasta.iterdir()} if pasta.is_dir() else {}

    def quarentenas(self, sessao, unidade=U0) -> list:
        base = sessao.diretorio.parent / ck.QUARENTENA / sessao.contexto.modo / unidade.rotulo
        return sorted(base.iterdir()) if base.is_dir() else []

    def cenarios_recuperaveis(self):
        """Cada estado que uma interrupção de ``salvar`` ou de ``registrar_falha`` pode deixar no slot."""
        rotulo = U0.rotulo

        def falha_ao_publicar(nome):
            def preparar(sessao):
                self.interromper(sessao, nome)
            return preparar

        def morto_ao_escrever(nome_antes, residuo):
            def preparar(sessao):
                if nome_antes:
                    self.interromper(sessao, nome_antes)
                (sessao.diretorio / rotulo).mkdir(exist_ok=True)
                (sessao.diretorio / rotulo / residuo).write_bytes(b'{"parcial":')
            return preparar

        return {
            "falha_ao_publicar_registro": falha_ao_publicar(ck.ARQUIVO_REGISTRO),
            "falha_ao_publicar_conclusao": falha_ao_publicar(ck.ARQUIVO_CONCLUIDA),
            "morto_escrevendo_predicoes": morto_ao_escrever(None, ".predicoes.json.k3j9x_2a.tmp"),
            "morto_escrevendo_registro": morto_ao_escrever(ck.ARQUIVO_REGISTRO, ".registro.json.a1b2c3d4.tmp"),
            "morto_escrevendo_conclusao": morto_ao_escrever(ck.ARQUIVO_CONCLUIDA, ".concluida.json.zz09_abc.tmp"),
            "morto_escrevendo_falha": morto_ao_escrever(None, ".falha-001.json.q1w2e3r4.tmp"),
        }

    def test_interrupcao_reexecuta_somente_a_unidade_incompleta(self):
        d = self.diretorio()
        sessao = self.sessao(d)
        for u in (U0, U1):
            sessao.salvar(self.resultados[u])
        concluidas = {u: self.conteudo(sessao.diretorio / u.rotulo) for u in (U0, U1)}
        self.interromper(sessao, ck.ARQUIVO_CONCLUIDA, self.resultados[U2])

        retomada = self.sessao(d)  # processo novo depois da interrupção
        relatorio = retomada.verificar()
        self.assertEqual(relatorio.concluidas, (U0.rotulo, U1.rotulo))
        self.assertEqual(relatorio.incompletas, (U2.rotulo,))
        self.assertIn(U2.rotulo, relatorio.a_executar)
        self.assertEqual(relatorio.n_esperadas, 25)
        with mock.patch.object(un, "executar_tarefa", wraps=un.executar_tarefa) as executar:
            obtidos = {u: ck.obter_unidade(retomada, u) for u in (U0, U1, U2)}
        self.assertEqual([c.args[1].unidade for c in executar.call_args_list], [U2])  # só a incompleta treina
        self.assertEqual({u: reap for u, (_, reap) in obtidos.items()}, {U0: True, U1: True, U2: False})
        self.assertEqual(obtidos[U2][0].registro, self.resultados[U2].registro)
        self.assertEqual({u: self.conteudo(retomada.diretorio / u.rotulo) for u in (U0, U1)}, concluidas)
        self.assertEqual(retomada.verificar().concluidas, (U0.rotulo, U1.rotulo, U2.rotulo))
        self.assertEqual(len(self.quarentenas(retomada, U2)), 1)
        self.assertEqual(self.quarentenas(retomada, U0), [])

    def test_todo_ponto_de_interrupcao_e_recuperado_e_preservado_em_quarentena(self):
        for nome, preparar in self.cenarios_recuperaveis().items():
            with self.subTest(cenario=nome):
                sessao = self.sessao()
                preparar(sessao)
                pasta = sessao.diretorio / U0.rotulo
                residuos = self.conteudo(pasta)
                self.assertNotIn(ck.ARQUIVO_CONCLUIDA, residuos)
                self.assertEqual(sessao.estado(U0), ck.ESTADO_INCOMPLETA)
                with self.assertRaisesRegex(ck.CheckpointNaoConcluido, "interrompida"):
                    sessao.carregar(U0)
                self.assertEqual(sessao.verificar().incompletas, (U0.rotulo,))
                self.assertEqual(self.conteudo(pasta), residuos)  # verificar só lê
                if nome == "morto_escrevendo_falha":
                    sessao.registrar_falha(un.FalhaRegistrada(U0, "falha_tecnica", "X", "m"))
                    self.assertEqual(sessao.estado(U0), ck.ESTADO_FALHOU)
                else:
                    sessao.salvar(self.resultados[U0])
                    self.assertEqual(sessao.carregar(U0).registro, self.resultados[U0].registro)
                quarentena, = self.quarentenas(sessao)
                preservados = self.conteudo(quarentena)
                recuperacao = json.loads(preservados.pop(ck.ARQUIVO_RECUPERACAO))
                self.assertEqual(preservados, residuos)  # os mesmos bytes, nada apagado
                self.assertEqual(recuperacao["arquivos"], {n: ck._hash_arquivo(b) for n, b in residuos.items()})
                self.assertEqual(recuperacao["sha256_identidade"], s4.sha256_canonico(sessao.identidade(U0)))
                self.assertFalse(any(n.endswith(".tmp") for n in self.conteudo(pasta)))

    def test_recuperacao_preserva_as_falhas_anteriores(self):
        sessao = self.sessao()
        sessao.registrar_falha(un.FalhaRegistrada(U0, "falha_tecnica", "FalhaAjuste", "m"))
        falha = (sessao.diretorio / U0.rotulo / "falha-001.json").read_bytes()
        self.interromper(sessao, ck.ARQUIVO_CONCLUIDA)
        self.assertEqual(sessao.estado(U0), ck.ESTADO_INCOMPLETA)
        controle = sessao.salvar(self.resultados[U0])
        self.assertEqual(controle["metadados"]["tentativa"], 2)
        self.assertEqual((sessao.diretorio / U0.rotulo / "falha-001.json").read_bytes(), falha)
        self.assertNotIn("falha-001.json", self.conteudo(self.quarentenas(sessao)[0]))

    def test_quarentena_nunca_sobrescreve(self):
        sessao = self.sessao()
        self.interromper(sessao, ck.ARQUIVO_CONCLUIDA)
        self.interromper(sessao, ck.ARQUIVO_CONCLUIDA)  # recupera a primeira e é interrompida de novo
        primeira, = self.quarentenas(sessao)
        antes = self.conteudo(primeira)
        sessao.salvar(self.resultados[U0])
        quarentenas = self.quarentenas(sessao)
        self.assertEqual(len(quarentenas), 2)
        self.assertIn(primeira, quarentenas)
        self.assertEqual(self.conteudo(primeira), antes)
        self.assertEqual(sessao.estado(U0), ck.ESTADO_CONCLUIDA)

    def test_interrupcao_na_primeira_publicacao_deixa_o_slot_vazio_e_retomavel(self):
        sessao = self.sessao()
        self.interromper(sessao, ck.ARQUIVO_PREDICOES)
        self.assertEqual(list((sessao.diretorio / U0.rotulo).iterdir()), [])
        self.assertEqual(sessao.estado(U0), ck.ESTADO_AUSENTE)
        sessao.salvar(self.resultados[U0])
        self.assertEqual(sessao.estado(U0), ck.ESTADO_CONCLUIDA)
        self.assertEqual(self.quarentenas(sessao), [])

    def test_estados_que_nao_sao_transacao_incompleta_continuam_erro(self):
        def com_predicoes(sessao):
            self.interromper(sessao, ck.ARQUIVO_REGISTRO)
            return sessao.diretorio / U0.rotulo

        def residuo(nome):
            def preparar(sessao):
                pasta = com_predicoes(sessao)
                (pasta / nome).write_bytes(b"{")
            return preparar

        def registro_sem_predicoes(sessao):
            pasta = sessao.diretorio / U0.rotulo
            self.interromper(sessao, ck.ARQUIVO_CONCLUIDA)
            (pasta / ck.ARQUIVO_PREDICOES).unlink()

        def dois_residuos(sessao):
            pasta = com_predicoes(sessao)
            (pasta / ".registro.json.a1b2c3d4.tmp").write_bytes(b"{")
            (pasta / ".registro.json.e5f6g7h8.tmp").write_bytes(b"{")

        def predicoes_alteradas(alterar):
            def preparar(sessao):
                caminho = com_predicoes(sessao) / ck.ARQUIVO_PREDICOES
                caminho.write_bytes(alterar(caminho.read_bytes()))
            return preparar

        def outra_dobra(bruto):
            valor = json.loads(bruto)
            valor["fingerprint_dobra"] = "0" * 64
            return ck._bytes_json(valor)

        def predicoes_de_outra_unidade(sessao):
            pasta = com_predicoes(sessao)
            outra = self.sessao_com(U2)
            shutil.copyfile(outra.diretorio / U2.rotulo / ck.ARQUIVO_PREDICOES, pasta / ck.ARQUIVO_PREDICOES)

        def arquivo_desconhecido(sessao):
            (com_predicoes(sessao) / "notas.txt").write_text("x", encoding="utf-8")

        def residuo_junto_da_conclusao(sessao):
            sessao.salvar(self.resultados[U0])
            (sessao.diretorio / U0.rotulo / ".concluida.json.k3j9x_2a.tmp").write_bytes(b"{")

        cenarios = {
            "registro_sem_predicoes": (registro_sem_predicoes, ck.CheckpointCorrompido),
            "dois_residuos": (dois_residuos, ck.CheckpointCorrompido),
            "residuo_fora_da_sequencia": (residuo(".concluida.json.k3j9x_2a.tmp"), ck.CheckpointCorrompido),
            "residuo_de_falha_com_resultado": (residuo(".falha-001.json.k3j9x_2a.tmp"), ck.CheckpointCorrompido),
            "residuo_de_formato_desconhecido": (residuo(".registro.json.abc.tmp"), ck.CheckpointCorrompido),
            "predicoes_truncadas": (predicoes_alteradas(lambda b: b[:-10]), ck.CheckpointCorrompido),
            "predicoes_fora_da_forma_canonica": (predicoes_alteradas(lambda b: b.replace(b",", b", ", 1)),
                                                 ck.CheckpointCorrompido),
            "predicoes_de_outra_dobra": (predicoes_alteradas(outra_dobra), ck.CheckpointIncompativel),
            "predicoes_de_outra_unidade": (predicoes_de_outra_unidade, ck.CheckpointIncompativel),
            "arquivo_desconhecido": (arquivo_desconhecido, ck.CheckpointCorrompido),
            "residuo_junto_da_conclusao": (residuo_junto_da_conclusao, ck.CheckpointCorrompido),
        }
        for nome, (preparar, erro) in cenarios.items():
            with self.subTest(cenario=nome):
                sessao = self.sessao()
                preparar(sessao)
                pasta = sessao.diretorio / U0.rotulo
                antes = self.conteudo(pasta)
                with self.assertRaises(erro):
                    sessao.estado(U0)
                with self.assertRaises(erro):
                    sessao.salvar(self.resultados[U0])
                with self.assertRaises(erro):
                    sessao.registrar_falha(un.FalhaRegistrada(U0, "falha_tecnica", "X", "m"))
                with mock.patch.object(un, "executar_tarefa", side_effect=AssertionError("não deveria treinar")):
                    with self.assertRaises(erro):
                        ck.obter_unidade(sessao, U0)
                with self.assertRaisesRegex(ck.CheckpointCorrompido, "retomada abortada"):
                    sessao.verificar([U0])
                self.assertEqual(self.conteudo(pasta), antes)  # nada movido nem apagado
                self.assertEqual(self.quarentenas(sessao), [])

    def test_conteudo_sem_marca_de_sessao(self):
        sessao = self.sessao_com(U0)
        (sessao.diretorio / ck.ARQUIVO_SESSAO).unlink()
        with self.assertRaisesRegex(ck.CheckpointCorrompido, "sem sessao.json"):
            self.sessao(sessao.diretorio.parent.parent)

    def test_publicacao_nao_sobrescreve_arquivo_existente(self):
        destino = self.diretorio() / "a.json"
        ck._publicar(b"1\n", destino)
        with self.assertRaises(ck.SobrescritaRecusada):
            ck._publicar(b"2\n", destino)
        self.assertEqual(destino.read_bytes(), b"1\n")
        self.assertEqual([p.name for p in destino.parent.iterdir()], ["a.json"])


# --------------------------------------------------------------------------- #
# Identidade
# --------------------------------------------------------------------------- #


class TestIdentidade(BaseCheckpoints):
    def reabrir_incompativel(self, d: Path, carga=None, padrao: str = ""):
        with self.assertRaisesRegex(ck.CheckpointIncompativel, padrao):
            self.sessao(d, carga)

    def test_identidade_reune_unidade_modo_fingerprints_configuracao_entradas_codigo_e_ambiente(self):
        sessao = self.sessao()
        identidade = sessao.identidade(U1)
        self.assertEqual(set(identidade), {
            "modo", "oficial", "sha256_configuracao", "fingerprint_plano", "entradas", "codigo", "ambiente",
            "unidade", "repeticao", "dobra_externa", "modelo", "fingerprint_dobra", "metodo_pontuacao"})
        self.assertEqual(identidade["sha256_configuracao"], cfg.sha256_configuracao())
        self.assertEqual(identidade["fingerprint_dobra"], sessao.contexto.plano.dobra(0, 1).fingerprint)
        self.assertEqual(identidade["metodo_pontuacao"], "decision_function")
        codigo = identidade["codigo"]
        self.assertIn("src/benchmark_s6/checkpoints.py", codigo)
        self.assertIn("src/benchmark_s6/tuning.py", codigo)
        self.assertTrue(set(s4.ARQUIVOS_S4) <= set(codigo))
        self.assertEqual({k: codigo[k] for k in s4.ARQUIVOS_S4}, dict(s4.ARQUIVOS_S4))
        self.assertEqual(set(identidade["ambiente"]["versoes"]),
                         {"numpy", "scipy", "scikit-learn", "pandas", "joblib", "threadpoolctl", "xgboost"})
        self.assertEqual(set(identidade["entradas"]), {"arquivos", "particoes", "caracteristicas", "n_linhas",
                                                       "sha256_amostra", "sha256_caracteristicas"})

    def test_codigo_alterado_impede_reutilizacao(self):
        d = self.diretorio()
        self.sessao(d).salvar(self.resultados[U0])
        real = ck._codigo()
        alterado = {**real, "src/benchmark_s6/tuning.py": "0" * 64}
        with mock.patch.object(ck, "_codigo", return_value=alterado):
            self.reabrir_incompativel(d, padrao="codigo")
        sessao = self.sessao(d)  # código voltou: a sessão abre e a unidade está lá
        self.assertEqual(sessao.estado(U0), ck.ESTADO_CONCLUIDA)
        with mock.patch.object(ck, "_codigo", return_value=alterado):  # mudança DURANTE a sessão
            for operacao in (lambda: sessao.estado(U0), lambda: sessao.carregar(U0), lambda: sessao.verificar(),
                             lambda: sessao.salvar(self.resultados[U1]), lambda: ck.obter_unidade(sessao, U0)):
                with self.assertRaisesRegex(ck.CheckpointIncompativel, "codigo alterado"):
                    operacao()
        self.assertFalse((sessao.diretorio / U1.rotulo).exists())

    def test_ambiente_alterado_impede_reutilizacao(self):
        d = self.diretorio()
        self.sessao(d).salvar(self.resultados[U0])
        real = ck._ambiente()
        outro = {**real, "versoes": {**real["versoes"], "scikit-learn": "1.4.0"}}
        with mock.patch.object(ck, "_ambiente", return_value=outro):
            self.reabrir_incompativel(d, padrao="ambiente")
        sessao = self.sessao(d)
        with mock.patch.object(ck, "_ambiente", return_value=outro):
            with self.assertRaisesRegex(ck.CheckpointIncompativel, "ambiente alterado"):
                sessao.carregar(U0)

    def test_entradas_diferentes_com_o_mesmo_plano(self):
        """Mudar só X (mesma amostra, mesmo plano) já muda a identidade."""
        d = self.diretorio()
        self.sessao(d).salvar(self.resultados[U0])
        base = carga_sintetica()
        X = caracteristicas(base.amostra())
        X[0, 0] += 1.0
        outra = carga_com(X=X)
        self.assertEqual(un.ContextoExecucao.de_dados(outra).plano.fingerprint,
                         un.ContextoExecucao.de_dados(self.carga).plano.fingerprint)
        self.reabrir_incompativel(d, outra, "entradas")

    def test_outra_amostra_ou_outra_origem(self):
        d = self.diretorio()
        self.sessao(d).salvar(self.resultados[U0])
        self.reabrir_incompativel(d, carga_com(semente=1), "fingerprint_plano")
        self.reabrir_incompativel(d, carga_com(oficial=True), "oficial")

    def test_slot_copiado_de_outra_identidade(self):
        origem = self.sessao_com(U0)
        X = caracteristicas(self.carga.amostra())
        X[:, 1] *= 2.0
        destino = self.sessao(carga=carga_com(X=X))
        shutil.copytree(origem.diretorio / U0.rotulo, destino.diretorio / U0.rotulo)
        with self.assertRaisesRegex(ck.CheckpointIncompativel, "entradas"):
            destino.estado(U0)
        with self.assertRaisesRegex(ck.CheckpointCorrompido, "retomada abortada"):
            destino.verificar()

    def test_identidade_falsificada_de_forma_coerente(self):
        sessao = self.sessao_com(U0)
        pasta = sessao.diretorio / U0.rotulo
        controle = ler(pasta / ck.ARQUIVO_CONCLUIDA)
        controle["identidade"]["fingerprint_dobra"] = "0" * 64
        gravar(pasta / ck.ARQUIVO_CONCLUIDA, controle)
        reassinar(pasta)
        with self.assertRaisesRegex(ck.CheckpointIncompativel, "fingerprint_dobra"):
            sessao.estado(U0)

    def test_marca_de_sessao_adulterada(self):
        sessao = self.sessao()
        caminho = sessao.diretorio / ck.ARQUIVO_SESSAO
        marca = ler(caminho)
        marca["identidade_sessao"]["oficial"] = True
        gravar(caminho, marca)
        with self.assertRaisesRegex(ck.CheckpointCorrompido, "hash"):
            self.sessao(sessao.diretorio.parent.parent)

    def test_unidade_fora_do_plano_e_resultado_de_outra_sessao(self):
        sessao = self.sessao()
        with self.assertRaisesRegex(ck.CheckpointIncompativel, "fora do plano"):
            sessao.estado(un.Unidade(1, 0, "lr_fixa"))
        outra = self.sessao(carga=carga_com(semente=1))
        with self.assertRaisesRegex(ck.CheckpointIncompativel, "outra sessão"):
            outra.salvar(self.resultados[U0])
        self.assertFalse((outra.diretorio / U0.rotulo).exists())

    def test_modos_ficam_em_diretorios_separados(self):
        d = self.diretorio()
        implementacao = self.sessao(d)
        definitivo = self.sessao(d, carga_sintetica(modo="definitivo"))
        self.assertNotEqual(implementacao.diretorio, definitivo.diretorio)
        self.assertEqual(definitivo.verificar().n_esperadas, 250)


# --------------------------------------------------------------------------- #
# Isolamento e privacidade
# --------------------------------------------------------------------------- #


def _estado_do_repositorio() -> bytes:
    return subprocess.run([GIT, "-C", str(REPO_ROOT), "status", "--porcelain=v1", "--untracked-files=all",
                           "--ignored"], capture_output=True, check=True).stdout


class TestIsolamento(BaseCheckpoints):
    def test_nada_vaza_para_o_repositorio(self):
        antes = _estado_do_repositorio()
        sessao = self.sessao_com(U0, U1)
        sessao.registrar_falha(un.FalhaRegistrada(U2, "falha_tecnica", "X", "m"))
        self.sessao(sessao.diretorio.parent.parent).verificar()
        self.assertEqual(_estado_do_repositorio(), antes)

    def test_identificadores_e_pontuacoes_so_nas_predicoes_privadas(self):
        sessao = self.sessao_com(U0)
        privado = self.resultados[U0].privado
        ids = set(sessao.contexto.amostra.ids.tolist()) | set(sessao.contexto.amostra.grupos.tolist())
        for caminho in sessao.diretorio.rglob("*.json"):
            texto = caminho.read_text(encoding="utf-8")
            com_ids = any(i in texto for i in ids)
            with self.subTest(arquivo=caminho.name):
                self.assertEqual(com_ids, caminho.name == ck.ARQUIVO_PREDICOES)
                if caminho.name != ck.ARQUIVO_PREDICOES:
                    self.assertNotIn("pontuacoes", texto)
                    self.assertNotIn("nodulo_ids", texto)
                    self.assertNotIn(repr(float(privado.pontuacoes[0])), texto)

    def test_registro_agregado_com_identificador_e_recusado_antes_de_gravar(self):
        original = self.resultados[U0]
        nodulo = str(original.privado.nodulo_ids[0])
        for mensagem in (f"aviso sobre {nodulo}", "paciente LIDC-IDRI-0001"):
            with self.subTest(mensagem=mensagem):
                registro = dataclasses.replace(original.registro,
                                               avisos=(("predicao", "UserWarning", mensagem, 1),))
                sessao = self.sessao()
                with self.assertRaises(ck.DadosIndividuaisNoRegistro):
                    sessao.salvar(un.ResultadoUnidade(registro, original.privado))
                self.assertFalse(any((sessao.diretorio / U0.rotulo).glob("*")))
                self.assertEqual(sessao.estado(U0), ck.ESTADO_AUSENTE)

    def test_modulo_nao_mede_nao_usa_pickle_nem_paralelismo(self):
        arvore = ast.parse(MODULO.read_text(encoding="utf-8"))
        modulos = {a.name for n in ast.walk(arvore) if isinstance(n, ast.Import) for a in n.names} | \
                  {n.module or "" for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom)}
        for proibido in ("pickle", "multiprocessing", "concurrent.futures", "sklearn.metrics"):
            self.assertNotIn(proibido, modulos)
        nomes = {n.id for n in ast.walk(arvore) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(arvore) if isinstance(n, ast.Attribute)}
        for proibido in ("roc_auc_score", "brier_score_loss", "log_loss", "fit", "predict_proba"):
            self.assertNotIn(proibido, nomes)

    def test_exportacoes(self):
        for nome in ck.__all__:
            self.assertTrue(hasattr(ck, nome), nome)


if __name__ == "__main__":
    unittest.main()
