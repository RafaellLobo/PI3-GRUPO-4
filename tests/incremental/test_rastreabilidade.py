"""Testes do Incremento 2C-A — manifestos e rastreabilidade (``src/incremental/rastreabilidade.py``).

Usam somente arquivos e repositórios git temporários. Não leem dados reais,
não treinam modelos e não calculam métricas. Aprovações de decisões, quando
usadas, são fictícias e ficam num repositório temporário.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/incremental -v
"""

from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import hashlib
import itertools
import json
import os
import platform
import random
import re
import sys
import tempfile
import unittest
import warnings
from importlib import metadata
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
for caminho in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.incremental import config as cfg  # noqa: E402
from src.incremental import guardas as g  # noqa: E402
from src.incremental import rastreabilidade as r  # noqa: E402
from test_guardas import (  # noqa: E402
    HASH_CONFIG_INCREMENTO_1, VALOR_P2, RepoTemporario, aprovar, protocolo, registro,
)

E0, E1 = cfg.Etapa.E0.value, cfg.Etapa.E1.value
PILOTO = cfg.ATIVIDADE_PILOTO_EXPLORATORIO
CV_REPETIDA = cfg.ATIVIDADE_CV_REPETIDA_EXPLORATORIA
DEF, EXP = r.Finalidade.DEFINITIVA, r.Finalidade.EXPLORATORIA
UTC = dt.timezone.utc
AMBIENTE_OK = {
    "versoes": {**dict(cfg.AMBIENTE_ESPERADO)},
    "implementacao": "CPython", "plataforma": sys.platform,
    "confere_com_esperado": True, "divergencias": {},
}
AMBIENTE_DIVERGENTE = {**AMBIENTE_OK, "confere_com_esperado": False,
                       "divergencias": {"scikit-learn": {"esperado": "1.3.2", "obtido": "9.9"}}}


def relogios(inicio=dt.datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC), duracao=2.5):
    instantes = iter([inicio, inicio + dt.timedelta(seconds=duracao)])
    monotonicos = iter([100.0, 100.0 + duracao])
    return {"relogio_utc": lambda: next(instantes), "relogio_monotonico": lambda: next(monotonicos)}


class ComRepo(unittest.TestCase):
    """Repositório temporário com uma entrada versionada e ``.gitignore``."""

    def setUp(self):
        self.repo = RepoTemporario()
        self.addCleanup(self.repo.remover)
        self.raiz = self.repo.raiz
        self.head = self.repo.commit({
            "entradas/tabela.csv": "a,b\n1,2\n",
            ".gitignore": "ignorado/\n",
            "saidas/.gitkeep": "",
        }, "2026-10-01")
        self.entrada = self.raiz / "entradas" / "tabela.csv"

    def execucao(self, finalidade=EXP, atividade=E0, decisoes=cfg.DECISOES, entradas=None,
                 seeds=None, parametros=None, **kwargs):
        autorizacao = kwargs.pop("autorizacao", None) or g.exigir_autorizacao(atividade, decisoes, self.raiz)
        return r.Execucao(
            atividade, finalidade, autorizacao,
            entradas={"tabela": self.entrada} if entradas is None else entradas,
            seeds={"cv_externa": 42} if seeds is None else seeds,
            parametros={"n_folds": 5} if parametros is None else parametros,
            decisoes=decisoes, repo=self.raiz, **{**relogios(), **kwargs})

    def rodar(self, finalidade=EXP, corpo=None, ambiente=AMBIENTE_OK, **kwargs):
        with mock.patch.object(r, "ambiente", return_value=ambiente):
            ex = self.execucao(finalidade, **kwargs)
            with ex:
                if corpo:
                    corpo(ex)
        return ex.manifesto


# --------------------------------------------------------------------------- #
# Hashes
# --------------------------------------------------------------------------- #


class TestHashes(ComRepo):
    def escrever(self, nome: str, conteudo: bytes) -> Path:
        caminho = self.raiz / nome
        caminho.write_bytes(conteudo)
        return caminho

    def test_vetor_conhecido_e_lf(self):
        caminho = self.escrever("abc.txt", b"abc")
        sha, sha_lf, n = r.calcular_sha256(caminho)
        self.assertEqual(sha, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual((sha_lf, n), (sha, 3))

    def test_crlf_distingue_bruto_e_normalizado(self):
        crlf = self.escrever("crlf.csv", b"a,b\r\n1,2\r\n")
        lf = self.escrever("lf.csv", b"a,b\n1,2\n")
        h_crlf, h_lf = r.hash_arquivo(crlf, self.raiz), r.hash_arquivo(lf, self.raiz)
        self.assertNotEqual(h_crlf.sha256, h_lf.sha256)
        self.assertEqual(h_crlf.sha256_lf, h_lf.sha256)
        self.assertEqual(h_crlf.sha256_lf, h_lf.sha256_lf)
        self.assertEqual((h_crlf.caminho, h_crlf.bytes), ("crlf.csv", 10))

    def test_normalizacao_em_blocos_igual_a_da_reproducao_auditada(self):
        gerador = random.Random(3)
        for _ in range(200):
            conteudo = b"".join(gerador.choice([b"a", b"\r", b"\n", b"\r\n"])
                                for _ in range(gerador.randint(0, 60)))
            caminho = self.escrever("bloco.bin", conteudo)
            esperado = hashlib.sha256(conteudo.replace(b"\r\n", b"\n")).hexdigest()
            for bloco in (1, 2, 3, 7):
                with self.subTest(bloco=bloco):
                    self.assertEqual(r.calcular_sha256(caminho, bloco)[1], esperado)

    def test_arquivo_ausente_diretorio_e_fora_do_repositorio(self):
        with self.assertRaises(r.ArquivoAusente):
            r.hash_arquivo(self.raiz / "nao_existe.csv", self.raiz)
        with self.assertRaises(r.ArquivoAusente):
            r.hash_arquivo(self.raiz / "entradas", self.raiz)
        with tempfile.TemporaryDirectory() as fora:
            externo = Path(fora) / "x.csv"
            externo.write_bytes(b"x")
            with self.assertRaisesRegex(r.ArquivoAusente, "fora do repositório"):
                r.hash_arquivo(externo, self.raiz)

    def test_exigir_hash(self):
        crlf = self.escrever("crlf.csv", b"a\r\n")
        sha, sha_lf, _ = r.calcular_sha256(crlf)
        self.assertEqual(r.exigir_hash(crlf, sha), "bruto")
        self.assertEqual(r.exigir_hash(crlf, "0" * 64, sha_lf), "lf")
        with self.assertRaises(r.HashIncompativel):
            r.exigir_hash(crlf, "0" * 64)
        with self.assertRaises(r.HashIncompativel):
            r.exigir_hash(crlf, "0" * 64, "1" * 64)

    def test_conferir_arquivos(self):
        registro_entrada = r.hash_arquivo(self.entrada, self.raiz).como_dict()
        r.conferir_arquivos([registro_entrada], self.raiz)
        self.entrada.write_bytes(b"a,b\n1,3\n")
        with self.assertRaises(r.HashIncompativel):
            r.conferir_arquivos([registro_entrada], self.raiz)

    def test_codigo_do_pacote(self):
        codigo = r.codigo_do_pacote()
        for modulo in ("config", "guardas", "dados", "rastreabilidade"):
            self.assertIn(f"src.incremental.{modulo}", codigo)


# --------------------------------------------------------------------------- #
# Git
# --------------------------------------------------------------------------- #


class TestGit(ComRepo):
    def test_repositorio_limpo(self):
        estado = r.estado_git(self.raiz)
        self.assertEqual((estado.head, estado.branch, estado.limpo), (self.head, "main", True))
        self.assertEqual(estado.como_dict()["alteracoes"], [])

    def test_alteracoes_locais_sao_registradas(self):
        self.entrada.write_bytes(b"a,b\n9,9\n")
        (self.raiz / "novo arquivo ç.txt").write_text("x", encoding="utf-8")
        (self.raiz / "preparado.txt").write_text("y", encoding="utf-8")
        self.repo.git("add", "preparado.txt")
        (self.raiz / "saidas" / ".gitkeep").unlink()
        (self.raiz / "ignorado").mkdir()
        (self.raiz / "ignorado" / "x.txt").write_text("z", encoding="utf-8")
        estado = r.estado_git(self.raiz)
        self.assertFalse(estado.limpo)
        self.assertEqual(set(estado.alteracoes), {
            (" M", "entradas/tabela.csv"), ("??", "novo arquivo ç.txt"),
            ("A ", "preparado.txt"), (" D", "saidas/.gitkeep")})

    def test_renomeacao_preparada(self):
        self.repo.git("mv", "entradas/tabela.csv", "entradas/renomeada.csv")
        (self.raiz / "depois.txt").write_text("x", encoding="utf-8")
        estado = r.estado_git(self.raiz)
        self.assertEqual(estado.alteracoes, (("??", "depois.txt"), ("R ", "entradas/renomeada.csv")))

    def test_head_destacado(self):
        self.repo.git("checkout", "-q", "--detach")
        estado = r.estado_git(self.raiz)
        self.assertIsNone(estado.branch)
        self.assertEqual(estado.head, self.head)

    def test_falhas_explicitas(self):
        with self.assertRaises(r.ErroGit):
            r.estado_git(self.raiz / "nao_existe")
        with self.assertRaisesRegex(r.ErroGit, "não é a raiz"):
            r.estado_git(self.raiz / "entradas")
        with mock.patch.object(r.shutil, "which", return_value=None):
            with self.assertRaisesRegex(r.ErroGit, "indisponível"):
                r.estado_git(self.raiz)
        vazio = RepoTemporario()
        self.addCleanup(vazio.remover)
        with self.assertRaises(r.ErroGit):
            r.estado_git(vazio.raiz)


# --------------------------------------------------------------------------- #
# Ambiente
# --------------------------------------------------------------------------- #


class TestAmbiente(unittest.TestCase):
    def test_versoes_registradas(self):
        amb = r.ambiente()
        self.assertEqual(set(amb["versoes"]), {"python", "numpy", "pandas", "scikit-learn", "scipy"})
        self.assertEqual(amb["versoes"]["python"], platform.python_version())
        for pacote in r.PACOTES:
            self.assertEqual(amb["versoes"][pacote], metadata.version(pacote))
        confere = all(amb["versoes"].get(k) == v for k, v in cfg.AMBIENTE_ESPERADO.items())
        self.assertEqual(amb["confere_com_esperado"], confere)

    def test_pacote_ausente_vira_divergencia(self):
        original = metadata.version

        def versao(nome):
            if nome == "scipy":
                raise metadata.PackageNotFoundError(nome)
            return original(nome)

        with mock.patch.object(r.metadata, "version", versao):
            amb = r.ambiente()
        self.assertIsNone(amb["versoes"]["scipy"])
        self.assertFalse(amb["confere_com_esperado"])
        self.assertEqual(amb["divergencias"]["scipy"]["obtido"], None)


# --------------------------------------------------------------------------- #
# Execução rastreada
# --------------------------------------------------------------------------- #


class TestExecucao(ComRepo):
    def test_execucao_exploratoria(self):
        m = self.rodar()
        ex, res, tempo = m["execucao"], m["resultado"], m["tempo"]
        self.assertEqual((ex["atividade"], ex["finalidade"], ex["elegivel_definitiva"]), (E0, "exploratoria", False))
        self.assertEqual(ex["hash_configuracao"], HASH_CONFIG_INCREMENTO_1)
        self.assertEqual(ex["seeds"], {"cv_externa": 42})
        self.assertEqual(ex["parametros"], {"n_folds": 5})
        self.assertEqual(ex["git"], {"head": self.head, "branch": "main", "limpo": True, "alteracoes": []})
        self.assertEqual(ex["entradas"]["tabela"]["caminho"], "entradas/tabela.csv")
        self.assertEqual(ex["codigo"]["src.incremental.config"]["caminho"], "src/incremental/config.py")
        self.assertEqual(ex["autorizacao"]["atividade"], E0)
        self.assertEqual((res["status"], res["erro"], res["warnings"], res["saidas"]), ("concluida", None, [], {}))
        self.assertEqual(tempo, {"inicio": "2026-10-12T12:00:00.000000Z",
                                 "termino": "2026-10-12T12:00:02.500000Z", "duracao_s": 2.5})

    def test_exploratoria_registra_alteracoes_locais(self):
        self.entrada.write_bytes(b"a,b\n7,7\n")
        m = self.rodar()
        self.assertFalse(m["execucao"]["git"]["limpo"])
        self.assertEqual(m["execucao"]["git"]["alteracoes"], [{"codigo": " M", "caminho": "entradas/tabela.csv"}])
        self.assertFalse(m["execucao"]["elegivel_definitiva"])

    def test_definitiva_exige_arvore_limpa_e_ambiente(self):
        executado = []
        (self.raiz / "rascunho.txt").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(r.ExecucaoNaoDefinitiva, "alterações locais"):
            self.rodar(DEF, corpo=lambda ex: executado.append(1))
        (self.raiz / "rascunho.txt").unlink()
        with self.assertRaisesRegex(r.ExecucaoNaoDefinitiva, "ambiente divergente"):
            self.rodar(DEF, corpo=lambda ex: executado.append(1), ambiente=AMBIENTE_DIVERGENTE)
        self.assertEqual(executado, [])

    def test_definitiva_com_saida_registrada(self):
        def corpo(ex):
            saida = self.raiz / "saidas" / "resultado.json"
            saida.write_text('{"ok": true}\n', encoding="utf-8")
            ex.registrar_saida("resultado", saida)

        m = self.rodar(DEF, corpo=corpo)
        self.assertTrue(m["execucao"]["elegivel_definitiva"])
        self.assertEqual(m["resultado"]["saidas"]["resultado"]["caminho"], "saidas/resultado.json")
        self.assertEqual(m["resultado"]["git_final"]["alteracoes"],
                         [{"codigo": "??", "caminho": "saidas/resultado.json"}])
        self.assertEqual(m["resultado"]["git_final"]["alteracoes_alem_das_saidas"], [])
        destino = self.raiz / "saidas" / "manifesto.json"
        r.escrever_manifesto(m, destino, self.raiz)
        self.assertEqual(r.ler_manifesto(destino), m)

    def test_definitiva_recusa_alteracao_nao_registrada_e_commit(self):
        nao_registrada = lambda ex: (self.raiz / "saidas" / "extra.txt").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(r.ExecucaoNaoDefinitiva, "além das saídas"):
            self.rodar(DEF, corpo=nao_registrada)
        (self.raiz / "saidas" / "extra.txt").unlink()
        with self.assertRaisesRegex(r.ExecucaoNaoDefinitiva, "HEAD"):
            self.rodar(DEF, corpo=lambda ex: self.repo.commit({"outro.txt": "x"}, "2026-10-02"))

    def test_exploratoria_registra_alteracao_nao_registrada(self):
        m = self.rodar(corpo=lambda ex: (self.raiz / "solto.txt").write_text("x", encoding="utf-8"))
        self.assertEqual(m["resultado"]["git_final"]["alteracoes_alem_das_saidas"],
                         [{"codigo": "??", "caminho": "solto.txt"}])

    def test_entrada_alterada_durante_a_execucao(self):
        with self.assertRaisesRegex(r.HashIncompativel, "entrada alterada"):
            self.rodar(corpo=lambda ex: self.entrada.write_bytes(b"a,b\n0,0\n"))

    def test_entradas_e_saidas_invalidas(self):
        with self.assertRaises(r.ArquivoAusente):
            self.rodar(entradas={"x": self.raiz / "nao_existe.csv"})
        with self.assertRaises(r.ArquivoAusente):
            self.rodar(corpo=lambda ex: ex.registrar_saida("x", self.raiz / "saidas" / "nunca.json"))
        with tempfile.TemporaryDirectory() as fora:
            with self.assertRaisesRegex(r.ArquivoAusente, "fora do repositório"):
                self.rodar(corpo=lambda ex: ex.registrar_saida("x", Path(fora) / "x.json"))
        with self.assertRaises(ValueError):
            self.rodar(corpo=lambda ex: [ex.registrar_saida("x", self.entrada),
                                         ex.registrar_saida("x", self.entrada)])
        ex = self.execucao()
        with self.assertRaises(r.ErroRastreabilidade):
            ex.registrar_saida("x", self.entrada)

    def test_falha_e_interrupcao_no_corpo(self):
        def falhar(_ex):
            raise ValueError("detalhe que não entra no manifesto")

        with mock.patch.object(r, "ambiente", return_value=AMBIENTE_OK):
            ex = self.execucao()
            with self.assertRaises(ValueError):
                with ex:
                    falhar(ex)
            self.assertEqual((ex.manifesto["resultado"]["status"], ex.manifesto["resultado"]["erro"]),
                             ("falhou", "ValueError"))
            self.assertNotIn("detalhe", json.dumps(ex.manifesto))
            ex = self.execucao()
            with self.assertRaises(KeyboardInterrupt):
                with ex:
                    raise KeyboardInterrupt
            self.assertEqual(ex.manifesto["resultado"]["status"], "interrompida")

    def test_execucao_e_de_uso_unico(self):
        with mock.patch.object(r, "ambiente", return_value=AMBIENTE_OK):
            ex = self.execucao()
            with ex:
                pass
            with self.assertRaises(r.ErroRastreabilidade):
                with ex:
                    pass

    def test_warnings_agregados_e_saneados(self):
        def corpo(_ex):
            for _ in range(3):
                warnings.warn("não convergiu", RuntimeWarning)
            warnings.warn(r"arquivo em C:\Programas\lib\x.py e /home/fulano/y.py", UserWarning)

        aviso = self.rodar(corpo=corpo)["resultado"]["warnings"]
        self.assertEqual(len(aviso), 2)
        por_categoria = {a["categoria"]: a for a in aviso}
        self.assertEqual(por_categoria["RuntimeWarning"]["ocorrencias"], 3)
        self.assertEqual(por_categoria["RuntimeWarning"]["arquivo"], "test_rastreabilidade.py")
        self.assertEqual(por_categoria["UserWarning"]["mensagem"], "arquivo em <caminho> e <caminho>")
        self.assertEqual(aviso, sorted(aviso, key=lambda a: (a["categoria"], a["mensagem"])))

    def test_warning_com_identificador_falha_explicitamente(self):
        with self.assertRaises(r.DadosIndividuais):
            self.rodar(corpo=lambda ex: warnings.warn("problema no LIDC-IDRI-9001", UserWarning))

    def test_seeds_invalidas(self):
        for seeds in ({"cv": True}, {"cv": "42"}, {"cv": -1}, {"cv": 2 ** 32}, {"CV": 1}, {"": 1}, [42]):
            with self.subTest(seeds=seeds):
                with self.assertRaises(ValueError):
                    self.execucao(seeds=seeds)

    def test_parametros_com_dado_individual_ou_nao_finito(self):
        casos = [({"limiar": float("nan")}, r.ManifestoInvalido),
                 ({"predicoes": [0.1, 0.9]}, r.DadosIndividuais),
                 ({"fold": {"y_pred": [1]}}, r.DadosIndividuais),
                 ({"exemplo": "LIDC-IDRI-9001_N01_scan9001"}, r.DadosIndividuais),
                 ({"saida": "C:/Users/fulano/x.csv"}, r.DadosIndividuais),
                 ({"saida": "/home/fulano/x.csv"}, r.DadosIndividuais),
                 ({"saida": str(REPO_ROOT / "x.csv")}, r.DadosIndividuais)]
        for parametros, erro in casos:
            with self.subTest(parametros=parametros):
                with self.assertRaises(erro):
                    self.execucao(parametros=parametros)


class TestAutorizacaoNaExecucao(ComRepo):
    def test_autorizacao_de_outra_atividade_ou_configuracao(self):
        aut_e0 = g.exigir_autorizacao(E0)
        with self.assertRaisesRegex(r.ErroRastreabilidade, "outra atividade"):
            self.execucao(atividade=cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS, autorizacao=aut_e0)
        outra = dataclasses.replace(aut_e0, hash_configuracao="0" * 64)
        with self.assertRaisesRegex(r.ErroRastreabilidade, "outra configuração"):
            self.execucao(autorizacao=outra)
        with self.assertRaises(TypeError):
            self.execucao(autorizacao=aut_e0.como_dict())

    def test_autorizacao_forjada_nao_entra_no_manifesto(self):
        forjada = dataclasses.replace(g.exigir_autorizacao(E0), head=self.head)
        with mock.patch.object(r, "ambiente", return_value=AMBIENTE_OK):
            with self.assertRaisesRegex(r.ErroRastreabilidade, "não confere"):
                with self.execucao(autorizacao=forjada):
                    self.fail("o corpo não deveria executar")

    def test_etapa1_com_aprovacoes_ficticias_e_head_alterado(self):
        r12 = [registro("P1"), registro("P2", VALOR_P2)]
        c1 = self.repo.commit(protocolo(*r12), "2026-10-02")
        decisoes = aprovar(P1=(c1, None), P2=(c1, VALOR_P2))
        m = self.rodar(DEF, atividade=E1, decisoes=decisoes)
        self.assertEqual(m["execucao"]["autorizacao"]["head"], c1)
        self.assertEqual([v["codigo"] for v in m["execucao"]["autorizacao"]["verificacoes"]], ["P1", "P2"])
        self.assertEqual(m["execucao"]["hash_configuracao"], cfg.hash_configuracao(decisoes))
        antiga = g.exigir_autorizacao(E1, decisoes, self.raiz)
        self.repo.commit({"outro.txt": "x"}, "2026-10-03")
        with self.assertRaisesRegex(r.ErroRastreabilidade, "não confere"):
            self.rodar(DEF, atividade=E1, decisoes=decisoes, autorizacao=antiga)


# --------------------------------------------------------------------------- #
# Registro auditável versus resultado definitivo utilizável
# --------------------------------------------------------------------------- #


class TestElegibilidade(ComRepo):
    """Só execução definitiva, elegível e concluída fornece resultado utilizável."""

    def executar(self, finalidade, excecao=None, com_saida=True):
        def corpo(ex):
            if com_saida:
                saida = self.raiz / "saidas" / "resultado.json"
                saida.write_text('{"auc": "agregado"}\n', encoding="utf-8")
                ex.registrar_saida("resultado", saida)
            if excecao is not None:
                raise excecao

        with mock.patch.object(r, "ambiente", return_value=AMBIENTE_OK):
            ex = self.execucao(finalidade)
            try:
                with ex:
                    corpo(ex)
            except BaseException as erro:  # noqa: B902 — inclui KeyboardInterrupt de propósito
                if excecao is None or not isinstance(erro, type(excecao)):
                    raise
        return ex.manifesto

    def gravar(self, manifesto, nome="manifesto.json") -> Path:
        # Diretório ignorado: os manifestos gravados não sujam a árvore das execuções seguintes.
        destino = self.raiz / "ignorado" / nome
        destino.parent.mkdir(exist_ok=True)
        r.escrever_manifesto(manifesto, destino, self.raiz)
        return destino

    def test_definitiva_concluida_e_utilizavel(self):
        m = self.executar(DEF)
        self.assertTrue(m["execucao"]["elegivel_definitiva"])
        self.assertTrue(r.resultado_definitivo_utilizavel(m))
        self.assertEqual(r.exigir_resultado_definitivo(self.gravar(m), self.raiz), m)

    def test_falha_e_interrupcao_sao_auditaveis_mas_nunca_utilizaveis(self):
        for nome, excecao in (("falhou", ValueError()), ("interrompida", KeyboardInterrupt())):
            with self.subTest(status=nome):
                m = self.executar(DEF, excecao)
                self.assertEqual(m["resultado"]["status"], nome)
                # Íntegro e auditável: valida, grava, relê, saídas parciais registradas.
                self.assertTrue(m["execucao"]["elegivel_definitiva"])
                self.assertIn("resultado", m["resultado"]["saidas"])
                caminho = self.gravar(m, f"{nome}.json")
                self.assertEqual(r.ler_manifesto(caminho), m)
                # Nunca utilizável como resultado científico definitivo.
                self.assertFalse(r.resultado_definitivo_utilizavel(m))
                with self.assertRaisesRegex(r.ResultadoNaoUtilizavel, nome):
                    r.exigir_resultado_definitivo(caminho, self.raiz)
                (self.raiz / "saidas" / "resultado.json").unlink()

    def test_exploratoria_concluida_nao_e_utilizavel(self):
        m = self.executar(EXP)
        self.assertEqual(m["resultado"]["status"], "concluida")
        self.assertFalse(r.resultado_definitivo_utilizavel(m))
        with self.assertRaisesRegex(r.ResultadoNaoUtilizavel, "exploratoria"):
            r.exigir_resultado_definitivo(self.gravar(m), self.raiz)

    def test_saida_alterada_depois_da_execucao_invalida_o_resultado(self):
        caminho = self.gravar(self.executar(DEF))
        (self.raiz / "saidas" / "resultado.json").write_text('{"auc": "outro"}\n', encoding="utf-8")
        with self.assertRaises(r.HashIncompativel):
            r.exigir_resultado_definitivo(caminho, self.raiz)
        (self.raiz / "saidas" / "resultado.json").unlink()
        with self.assertRaises(r.ArquivoAusente):
            r.exigir_resultado_definitivo(caminho, self.raiz)

    def test_adulteracoes_que_tentariam_tornar_utilizavel(self):
        falhou = self.executar(DEF, ValueError())
        (self.raiz / "saidas" / "resultado.json").unlink()
        concluida = self.executar(DEF)
        exploratoria_suja = None
        (self.raiz / "rascunho.txt").write_text("x", encoding="utf-8")
        exploratoria_suja = self.executar(EXP, com_saida=False)

        def corpo_de(m):
            return {k: json.loads(json.dumps(m[k]))
                    for k in ("formato", "versao_formato", "execucao", "resultado", "tempo")}

        casos = {
            "falhou → concluida, erro mantido": (falhou, lambda c: c["resultado"].update(status="concluida")),
            "falhou → erro removido, status mantido": (falhou, lambda c: c["resultado"].update(erro=None)),
            "interrompida inventada sem erro": (concluida, lambda c: c["resultado"].update(status="interrompida")),
            "definitiva concluída marcada não elegível": (
                concluida, lambda c: c["execucao"].update(elegivel_definitiva=False)),
            "exploratória suja promovida a definitiva": (exploratoria_suja, lambda c: c["execucao"].update(
                finalidade="definitiva", elegivel_definitiva=True)),
            "definitiva com alteração além das saídas": (concluida, lambda c: c["resultado"]["git_final"].update(
                alteracoes_alem_das_saidas=[{"codigo": "??", "caminho": "x.txt"}])),
            "definitiva com HEAD final diferente": (concluida, lambda c: c["resultado"]["git_final"].update(
                head="0" * 40)),
        }
        for nome, (base, alterar) in casos.items():
            with self.subTest(caso=nome):
                alterado = corpo_de(base)
                alterar(alterado)
                forjado = r._com_hashes(alterado)  # hashes recalculados de propósito
                with self.assertRaises(r.ManifestoInvalido):
                    r.validar_manifesto(forjado)
                with self.assertRaises(r.ManifestoInvalido):
                    r.resultado_definitivo_utilizavel(forjado)
        # Sem recalcular os hashes, qualquer edição é recusada.
        sem_recalculo = json.loads(json.dumps(falhou))
        sem_recalculo["resultado"].update(status="concluida", erro=None)
        with self.assertRaisesRegex(r.ManifestoInvalido, "não confere"):
            r.resultado_definitivo_utilizavel(sem_recalculo)

    def test_saida_forjada_no_manifesto_nao_passa_na_conferencia(self):
        m = self.executar(DEF)
        corpo = {k: json.loads(json.dumps(m[k])) for k in ("formato", "versao_formato", "execucao",
                                                           "resultado", "tempo")}
        corpo["resultado"]["saidas"]["resultado"]["sha256"] = "0" * 64
        caminho = self.gravar(r._com_hashes(corpo))
        with self.assertRaises(r.HashIncompativel):
            r.exigir_resultado_definitivo(caminho, self.raiz)

    def test_falha_com_identificador_na_excecao_nao_vaza(self):
        m = self.executar(DEF, ValueError("erro no LIDC-IDRI-9001_N01_scan9001"))
        self.assertEqual((m["resultado"]["status"], m["resultado"]["erro"]), ("falhou", "ValueError"))
        self.assertNotRegex(r.serializar(m).decode("utf-8"), r"LIDC-IDRI-\d{4}")
        self.assertFalse(r.resultado_definitivo_utilizavel(m))

    def test_predicado_valida_o_manifesto_inteiro(self):
        with self.assertRaises(r.ManifestoInvalido):
            r.resultado_definitivo_utilizavel({"formato": r.FORMATO})


class TestPilotoExploratorio(ComRepo):
    """O piloto exploratório nunca é definitivo: nem na execução, nem no manifesto."""

    ATIVIDADE = PILOTO
    OUTRA_EXPLORATORIA = CV_REPETIDA

    @staticmethod
    def corpo_de(m):
        return {k: json.loads(json.dumps(m[k])) for k in ("formato", "versao_formato", "execucao",
                                                          "resultado", "tempo")}

    def test_execucao_definitiva_recusada_na_construcao(self):
        autorizacao = g.exigir_autorizacao(self.ATIVIDADE, cfg.DECISOES, self.raiz)
        for finalidade in (DEF, "definitiva"):
            with self.subTest(finalidade=finalidade), \
                    mock.patch.object(g, "exigir_autorizacao", side_effect=AssertionError("não deveria chegar aqui")), \
                    mock.patch.object(r, "estado_git", side_effect=AssertionError("não deveria chegar aqui")):
                with self.assertRaisesRegex(r.ExecucaoNaoDefinitiva, "somente exploratória"):
                    r.Execucao(self.ATIVIDADE, finalidade, autorizacao, {"tabela": self.entrada},
                               {"cv_externa": 42}, repo=self.raiz)
        # Mesmo com árvore limpa e ambiente esperado (todo o resto de uma definitiva atendido).
        with self.assertRaisesRegex(r.ExecucaoNaoDefinitiva, "somente exploratória"):
            self.rodar(DEF, atividade=self.ATIVIDADE)

    def test_autorizacao_de_outra_atividade_e_recusada(self):
        # A autorização de uma exploratória não abre execução de outra, nem de uma etapa oficial.
        autorizacao = g.exigir_autorizacao(self.ATIVIDADE, cfg.DECISOES, self.raiz)
        for outra in (self.OUTRA_EXPLORATORIA, E0, cfg.Etapa.E2.value):
            for finalidade in (EXP, DEF):
                with self.subTest(outra=outra, finalidade=finalidade):
                    with self.assertRaises((r.ErroRastreabilidade, r.ExecucaoNaoDefinitiva)) as ctx:
                        r.Execucao(outra, finalidade, autorizacao, {"tabela": self.entrada}, {"cv_externa": 42},
                                   repo=self.raiz)
                    if outra != self.OUTRA_EXPLORATORIA or finalidade is EXP:
                        self.assertRegex(str(ctx.exception), "outra atividade")

    def test_execucao_exploratoria_do_piloto_nunca_utilizavel(self):
        m = self.rodar(atividade=self.ATIVIDADE)
        ex = m["execucao"]
        self.assertEqual((ex["atividade"], ex["finalidade"], ex["elegivel_definitiva"]),
                         (self.ATIVIDADE, "exploratoria", False))
        self.assertEqual((ex["autorizacao"]["atividade"], ex["autorizacao"]["decisoes_exigidas"]),
                         (self.ATIVIDADE, []))
        self.assertEqual(ex["hash_configuracao"], HASH_CONFIG_INCREMENTO_1)
        self.assertFalse(r.resultado_definitivo_utilizavel(m))
        destino = self.raiz / "ignorado" / f"{self.ATIVIDADE}.json"
        destino.parent.mkdir(exist_ok=True)
        r.escrever_manifesto(m, destino, self.raiz)
        with self.assertRaisesRegex(r.ResultadoNaoUtilizavel, "exploratoria"):
            r.exigir_resultado_definitivo(destino, self.raiz)

    def test_manifesto_do_piloto_promovido_a_definitivo_e_recusado(self):
        piloto = self.rodar(atividade=self.ATIVIDADE)  # árvore limpa, ambiente esperado, HEAD estável
        casos = {
            "finalidade e elegibilidade": dict(finalidade="definitiva", elegivel_definitiva=True),
            "só elegibilidade": dict(elegivel_definitiva=True),
            "só finalidade": dict(finalidade="definitiva"),
        }
        pasta = self.raiz / "ignorado"
        pasta.mkdir(exist_ok=True)
        for nome, campos in casos.items():
            with self.subTest(caso=nome):
                corpo = self.corpo_de(piloto)
                corpo["execucao"].update(campos)
                forjado = r._com_hashes(corpo)  # hashes recalculados de propósito
                for operacao in (r.validar_manifesto, r.resultado_definitivo_utilizavel):
                    with self.assertRaisesRegex(r.ManifestoInvalido, "somente exploratória"):
                        operacao(forjado)
                with self.assertRaisesRegex(r.ManifestoInvalido, "somente exploratória"):
                    r.escrever_manifesto(forjado, pasta / f"forjado-{len(nome)}.json", self.raiz)
                bruto = pasta / f"bruto-{len(nome)}.json"
                bruto.write_bytes(r.serializar(forjado))
                with self.assertRaisesRegex(r.ManifestoInvalido, "somente exploratória"):
                    r.exigir_resultado_definitivo(bruto, self.raiz)
        # Reetiquetar só a execução como Etapa 0 (a autorização continua do piloto) não passa.
        reetiquetado = self.corpo_de(piloto)
        reetiquetado["execucao"].update(atividade=E0, finalidade="definitiva", elegivel_definitiva=True)
        with self.assertRaisesRegex(r.ManifestoInvalido, "atividade diferente"):
            r.resultado_definitivo_utilizavel(r._com_hashes(reetiquetado))
        # Controle: a mesma promoção num manifesto da Etapa 0 é coerente — a recusa vem da atividade.
        e0 = self.corpo_de(self.rodar(atividade=E0))
        e0["execucao"].update(finalidade="definitiva", elegivel_definitiva=True)
        self.assertTrue(r.resultado_definitivo_utilizavel(r._com_hashes(e0)))


class TestCvRepetidaExploratoria(TestPilotoExploratorio):
    """A CV repetida exploratória herda as mesmas recusas: nunca definitiva nem elegível."""

    ATIVIDADE = CV_REPETIDA
    OUTRA_EXPLORATORIA = PILOTO


# --------------------------------------------------------------------------- #
# Determinismo
# --------------------------------------------------------------------------- #


class TestDeterminismo(ComRepo):
    def test_relogio_nao_altera_identidade_nem_resultado(self):
        a = self.rodar()
        b = self.rodar(**relogios(dt.datetime(2027, 1, 1, tzinfo=UTC), 99.0))
        self.assertEqual(a["execucao"], b["execucao"])
        self.assertEqual(a["sha256_execucao"], b["sha256_execucao"])
        self.assertEqual(a["sha256_reprodutivel"], b["sha256_reprodutivel"])
        self.assertNotEqual(a["sha256_manifesto"], b["sha256_manifesto"])

    def test_ordem_de_insercao_nao_importa_e_seed_importa(self):
        a = self.rodar(parametros={"x": 1, "y_lim": {"b": 2, "a": 1}}, seeds={"a": 1, "b": 2})
        b = self.rodar(parametros={"y_lim": {"a": 1, "b": 2}, "x": 1}, seeds={"b": 2, "a": 1})
        self.assertEqual(a["sha256_execucao"], b["sha256_execucao"])
        c = self.rodar(parametros={"x": 1, "y_lim": {"b": 2, "a": 1}}, seeds={"a": 1, "b": 3})
        self.assertNotEqual(a["sha256_execucao"], c["sha256_execucao"])

    def test_resultado_diferente_muda_hash_reprodutivel(self):
        a = self.rodar()
        b = self.rodar(corpo=lambda ex: warnings.warn("aviso", UserWarning))
        self.assertEqual(a["sha256_execucao"], b["sha256_execucao"])
        self.assertNotEqual(a["sha256_reprodutivel"], b["sha256_reprodutivel"])

    def test_serializacao_canonica(self):
        m = self.rodar()
        conteudo = r.serializar(m)
        self.assertEqual(conteudo, r.serializar(json.loads(conteudo)))
        self.assertNotIn(b"\r", conteudo)
        self.assertTrue(conteudo.endswith(b"}\n"))
        self.assertEqual(list(json.loads(conteudo)), sorted(m))
        self.assertEqual(r.json_canonico({"b": 1, "a": [1, 2]}), '{"a":[1,2],"b":1}')
        with self.assertRaises(r.ManifestoInvalido):
            r.json_canonico({"x": float("inf")})


# --------------------------------------------------------------------------- #
# Escrita atômica e leitura com validação
# --------------------------------------------------------------------------- #


class TestEscritaLeitura(ComRepo):
    def setUp(self):
        super().setUp()
        self.m = self.rodar()
        self.destino = self.raiz / "saidas" / "manifesto.json"

    def residuos(self) -> list:
        return [p.name for p in self.destino.parent.iterdir() if p.name.endswith(".tmp")]

    def regravar(self, manifesto: dict) -> Path:
        caminho = self.raiz / "saidas" / "alterado.json"
        caminho.write_bytes(r.serializar(manifesto))
        return caminho

    def test_escrita_e_leitura(self):
        registro_escrito = r.escrever_manifesto(self.m, self.destino, self.raiz)
        self.assertEqual(registro_escrito.caminho, "saidas/manifesto.json")
        self.assertEqual(self.destino.read_bytes(), r.serializar(self.m))
        self.assertEqual(r.ler_manifesto(self.destino), self.m)
        self.assertEqual(self.residuos(), [])

    def test_sem_sobrescrita(self):
        r.escrever_manifesto(self.m, self.destino, self.raiz)
        original = self.destino.read_bytes()
        outro = self.rodar(**relogios(duracao=7.0))
        with self.assertRaises(r.SobrescritaRecusada):
            r.escrever_manifesto(outro, self.destino, self.raiz)
        # Destino criado entre a checagem e a publicação: os.link recusa.
        with mock.patch.object(r.os.path, "lexists", return_value=False):
            with self.assertRaises(r.SobrescritaRecusada):
                r.escrever_manifesto(outro, self.destino, self.raiz)
        self.assertEqual(self.destino.read_bytes(), original)
        self.assertEqual(self.residuos(), [])

    def test_falha_na_publicacao_nao_deixa_arquivo(self):
        with mock.patch.object(r.os, "link", side_effect=OSError("sem suporte")):
            with self.assertRaisesRegex(r.ErroRastreabilidade, "atômica"):
                r.escrever_manifesto(self.m, self.destino, self.raiz)
        self.assertFalse(self.destino.exists())
        self.assertEqual(self.residuos(), [])

    def test_destinos_recusados(self):
        with self.assertRaises(r.ArquivoAusente):
            r.escrever_manifesto(self.m, self.raiz / "nao_existe" / "m.json", self.raiz)
        (self.raiz / "data").mkdir()
        with self.assertRaisesRegex(r.SobrescritaRecusada, "data/"):
            r.escrever_manifesto(self.m, self.raiz / "data" / "m.json", self.raiz)
        with self.assertRaisesRegex(r.SobrescritaRecusada, "registrado"):
            r.escrever_manifesto(self.m, self.entrada, self.raiz)
        with tempfile.TemporaryDirectory() as fora:
            with self.assertRaises(r.ArquivoAusente):
                r.escrever_manifesto(self.m, Path(fora) / "m.json", self.raiz)

    def test_manifesto_adulterado_nao_e_gravado(self):
        adulterado = json.loads(json.dumps(self.m))
        adulterado["resultado"]["status"] = "falhou"
        with self.assertRaises(r.ManifestoInvalido):
            r.escrever_manifesto(adulterado, self.destino, self.raiz)
        self.assertFalse(self.destino.exists())

    def test_leitura_recusa_corrupcao(self):
        r.escrever_manifesto(self.m, self.destino, self.raiz)
        conteudo = self.destino.read_bytes()
        texto = conteudo.decode("utf-8")
        casos = {
            "truncado": conteudo[: len(conteudo) // 2],
            "valor alterado": texto.replace('"concluida"', '"falhou"').encode("utf-8"),
            "hash alterado": re.sub(r'"sha256_execucao": "[0-9a-f]{64}"',
                                    '"sha256_execucao": "' + "0" * 64 + '"', texto).encode("utf-8"),
            "chave duplicada": texto.replace('{\n  "execucao"', '{\n  "formato": "x",\n  "execucao"', 1).encode("utf-8"),
            "NaN": texto.replace('"duracao_s": 2.5', '"duracao_s": NaN').encode("utf-8"),
            "não UTF-8": conteudo + b"\xff",
            "fora da forma canônica": json.dumps(self.m, sort_keys=True).encode("utf-8"),
            "vazio": b"",
        }
        for nome, bytes_corrompidos in casos.items():
            with self.subTest(caso=nome):
                caminho = self.raiz / "saidas" / "corrompido.json"
                caminho.write_bytes(bytes_corrompidos)
                with self.assertRaises(r.ManifestoInvalido):
                    r.ler_manifesto(caminho)
        with self.assertRaises(r.ArquivoAusente):
            r.ler_manifesto(self.raiz / "saidas" / "nao_existe.json")

    def test_leitura_aceita_crlf(self):
        caminho = self.raiz / "saidas" / "crlf.json"
        caminho.write_bytes(r.serializar(self.m).replace(b"\n", b"\r\n"))
        self.assertEqual(r.ler_manifesto(caminho), self.m)

    def test_hashes_recalculados_nao_legitimam_conteudo_invalido(self):
        corpo = {k: json.loads(json.dumps(self.m[k]))
                 for k in ("formato", "versao_formato", "execucao", "resultado", "tempo")}
        casos = {
            "chave extra": lambda c: c["execucao"].update(extra=1),
            "sem chave": lambda c: c["resultado"].pop("warnings"),
            "formato": lambda c: c.update(formato="outro"),
            "versão": lambda c: c.update(versao_formato=2),
            "status": lambda c: c["resultado"].update(status="ok"),
            "erro incoerente": lambda c: c["resultado"].update(erro="ValueError"),
            "definitiva não elegível": lambda c: c["execucao"].update(finalidade="definitiva"),
            "exploratória elegível": lambda c: c["execucao"].update(elegivel_definitiva=True),
            "definitiva com árvore suja": lambda c: (
                c["execucao"].update(finalidade="definitiva", elegivel_definitiva=True),
                c["execucao"]["git"].update(limpo=False)),
            "definitiva com ambiente divergente": lambda c: (
                c["execucao"].update(finalidade="definitiva", elegivel_definitiva=True),
                c["execucao"]["ambiente"].update(confere_com_esperado=False)),
            "seed inválida": lambda c: c["execucao"]["seeds"].update(cv_externa="42"),
            "git_final sem head": lambda c: c["resultado"]["git_final"].update(head="x"),
            "caminho absoluto": lambda c: c["execucao"]["entradas"]["tabela"].update(caminho="/tmp/x.csv"),
            "identificador": lambda c: c["resultado"]["warnings"].append(
                {"categoria": "UserWarning", "mensagem": "LIDC-IDRI-9001", "arquivo": "x.py", "linha": 1,
                 "ocorrencias": 1}),
            "chave individual": lambda c: c["execucao"]["parametros"].update(y_pred=[1, 0]),
            "instante local": lambda c: c["tempo"].update(inicio="2026-10-12 12:00:00"),
            "atividade": lambda c: c["execucao"].update(atividade="etapa9"),
        }
        for nome, alterar in casos.items():
            with self.subTest(caso=nome):
                alterado = json.loads(json.dumps(corpo))
                alterar(alterado)
                with self.assertRaises(r.ManifestoInvalido):
                    r.ler_manifesto(self.regravar(r._com_hashes(alterado)))
                (self.raiz / "saidas" / "alterado.json").unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Ausência de dados individuais e isolamento do módulo
# --------------------------------------------------------------------------- #


class TestSemDadosIndividuais(ComRepo):
    def test_manifesto_nao_contem_dados_individuais_nem_caminhos_locais(self):
        def corpo(ex):
            saida = self.raiz / "saidas" / "r.json"
            saida.write_text("{}", encoding="utf-8")
            ex.registrar_saida("r", saida)
            warnings.warn(f"aviso com {self.raiz}", UserWarning)

        m = self.rodar(corpo=corpo)
        texto = r.serializar(m).decode("utf-8")
        self.assertNotRegex(texto, r"LIDC-IDRI-\d{4}")
        for local in (Path.home(), self.raiz, REPO_ROOT, Path(tempfile.gettempdir())):
            for forma in {str(local.resolve()), str(local.resolve()).replace("\\", "/")}:
                self.assertNotIn(forma.lower(), texto.lower())
        self.assertNotRegex(texto, r'"[A-Za-z]:[\\/]')

        def chaves(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield k
                    yield from chaves(v)
            elif isinstance(obj, list):
                for v in obj:
                    yield from chaves(v)

        self.assertFalse({k.lower() for k in chaves(m)} & r.CHAVES_PROIBIDAS)

    def test_detector(self):
        for texto in ("LIDC-IDRI-9002", "x LIDC-IDRI-9999_N01_scan9999", "C:\\dados", "D:/x", "/home/u/x",
                      "\\\\servidor\\x\\y", str(Path.home() / "x")):
            with self.subTest(texto=texto):
                with self.assertRaises(r.DadosIndividuais):
                    r._exigir_sem_dados_individuais({"mensagem": texto})
        for texto in ("src/incremental/config.py", "2026-10-12T12:00:00.000000Z", "LIDC", "razão 1:2",
                      "etapa0_prerequisitos", "<caminho>"):
            with self.subTest(texto=texto):
                r._exigir_sem_dados_individuais({"mensagem": texto})
        for chave in ("nodulo_id", "Patient_ID", "y_pred", "probabilidades"):
            with self.subTest(chave=chave):
                with self.assertRaises(r.DadosIndividuais):
                    r._exigir_sem_dados_individuais({chave: 1})


class TestModulo(unittest.TestCase):
    def test_modulo_nao_le_dados_nem_treina_nem_importa_historicos(self):
        arvore = ast.parse((REPO_ROOT / "src" / "incremental" / "rastreabilidade.py").read_text(encoding="utf-8"))
        importados = set()
        for no in ast.walk(arvore):
            if isinstance(no, ast.Import):
                importados.update(a.name.split(".")[0] for a in no.names)
            elif isinstance(no, ast.ImportFrom):
                importados.add(("." * no.level) + (no.module or ""))
        self.assertFalse(importados & {"pandas", "numpy", "sklearn", "scipy", "scripts", ".dados"})
        chamadas = {no.func.attr if isinstance(no.func, ast.Attribute) else getattr(no.func, "id", None)
                    for no in ast.walk(arvore) if isinstance(no, ast.Call)}
        self.assertFalse(chamadas & {"fit", "predict", "predict_proba", "read_csv", "roc_auc_score"})

    def test_configuracao_e_decisoes_inalteradas(self):
        self.assertEqual(cfg.hash_configuracao(), HASH_CONFIG_INCREMENTO_1)
        self.assertTrue(all(not d.aprovada for d in cfg.DECISOES.values()))


if __name__ == "__main__":
    unittest.main()
