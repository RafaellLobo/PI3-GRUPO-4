"""Testes de ``src/benchmark_s6/manifesto.py`` (P9, Seção 14.1). Só dados sintéticos."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.benchmark_s6 import checkpoints as ck  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import manifesto as mf  # noqa: E402

GIT = shutil.which("git")
SHA = cfg.sha256_configuracao()


def estado(limpo=True, head="a" * 40):
    return SimpleNamespace(head=head, branch="main", limpo=limpo)


class BaseManifesto(unittest.TestCase):
    def setUp(self):
        self.pasta = Path(tempfile.mkdtemp(prefix="manif-s6-"))
        self.addCleanup(shutil.rmtree, self.pasta, ignore_errors=True)

    def arquivos(self):
        (self.pasta / "a.csv").write_bytes(b"x,y\n1,2\n")
        (self.pasta / "figuras").mkdir(exist_ok=True)
        (self.pasta / "figuras" / "F1.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01")
        return ["a.csv", "figuras/F1.png"]

    def manifesto(self, **kw):
        saidas = mf.hashes_de_saidas(self.pasta, self.arquivos())
        base = dict(modo="definitivo", oficial=True, elegivel=True, motivos=[], estado_git=estado(),
                    sha256_configuracao=SHA, entradas={"tabela": "1" * 64}, codigo={"src/x.py": "2" * 64},
                    ambiente={"python": "3.9.13"}, saidas=saidas)
        base.update(kw)
        return mf.montar_manifesto(**base)


class TestGravacao(BaseManifesto):
    def test_gravacao_atomica_sem_sobrescrita(self):
        alvo = self.pasta / "x" / "a.json"
        self.assertTrue(mf.gravar_atomico(alvo, b"1\n"))
        self.assertFalse(mf.gravar_atomico(alvo, b"1\n"))  # mesmos bytes: não faz nada
        with self.assertRaises(mf.SobrescritaRecusada):
            mf.gravar_atomico(alvo, b"2\n")
        self.assertEqual(alvo.read_bytes(), b"1\n")
        self.assertEqual([p.name for p in alvo.parent.iterdir()], ["a.json"])  # sem resíduo

    def test_falha_na_gravacao_nao_deixa_residuo(self):
        alvo = self.pasta / "b.json"
        original = os.replace

        def falha(origem, destino):
            raise OSError("disco cheio (simulado)")

        mf.os.replace = falha
        try:
            with self.assertRaises(OSError):
                mf.gravar_atomico(alvo, b"1\n")
        finally:
            mf.os.replace = original
        self.assertEqual(list(self.pasta.iterdir()), [])

    def test_json_canonico_com_lf_final(self):
        b = mf.bytes_json({"b": 1, "a": [1.5, None]})
        self.assertEqual(b, b'{"a":[1.5,null],"b":1}\n')
        with self.assertRaises(Exception):
            mf.bytes_json({"x": float("nan")})


class TestMontagem(BaseManifesto):
    def test_manifesto_elegivel_exige_definitivo_oficial_e_git_limpo(self):
        self.assertTrue(self.manifesto()["elegivel"])
        for ruim in ({"modo": "implementacao"}, {"oficial": False}, {"estado_git": estado(limpo=False)}):
            with self.subTest(ruim=list(ruim)), self.assertRaises(mf.ManifestoInvalido):
                self.manifesto(**ruim)

    def test_nao_elegivel_exige_motivo_e_implementacao_e_exploratorio(self):
        with self.assertRaises(mf.ManifestoInvalido):
            self.manifesto(elegivel=False, motivos=[])
        m = self.manifesto(modo="implementacao", oficial=True, elegivel=False, motivos=["modo exploratório"])
        self.assertEqual((m["elegivel"], m["exploratorio"]), (False, True))
        self.assertFalse(self.manifesto()["exploratorio"])

    def test_o_manifesto_nao_lista_a_si_mesmo(self):
        self.arquivos()
        (self.pasta / mf.ARQUIVO_MANIFESTO).write_text("{}", encoding="utf-8")
        with self.assertRaises(mf.ManifestoInvalido):
            mf.hashes_de_saidas(self.pasta, [mf.ARQUIVO_MANIFESTO])
        with self.assertRaises(mf.ManifestoInvalido):
            mf.hashes_de_saidas(self.pasta, ["nao_existe.csv"])

    def test_hashes_bruto_e_lf(self):
        self.arquivos()
        (self.pasta / "t.csv").write_bytes(b"a\r\nb\r\n")
        h = mf.hashes_de_saidas(self.pasta, ["t.csv"])["t.csv"]
        self.assertNotEqual(h["sha256"], h["sha256_lf"])
        self.assertEqual(h["bytes"], 6)


class TestVerificacao(BaseManifesto):
    def publicar(self):
        m = self.manifesto()
        mf.escrever_manifesto(self.pasta, m)
        return m

    def test_ida_e_volta_e_verificacao_das_saidas(self):
        m = self.publicar()
        lido = mf.ler_manifesto(self.pasta)
        self.assertEqual(lido, m)
        mf.verificar_saidas(self.pasta, lido)

    def test_detecta_arquivo_alterado_ausente_ou_extra(self):
        m = self.publicar()
        (self.pasta / "a.csv").write_bytes(b"x,y\n1,3\n")
        with self.assertRaisesRegex(mf.ManifestoInvalido, "SHA-256"):
            mf.verificar_saidas(self.pasta, m)
        (self.pasta / "a.csv").write_bytes(b"x,y\n1,2\n")
        mf.verificar_saidas(self.pasta, m)
        (self.pasta / "extra.csv").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(mf.ManifestoInvalido, "sobram"):
            mf.verificar_saidas(self.pasta, m)
        (self.pasta / "extra.csv").unlink()
        (self.pasta / "figuras" / "F1.png").unlink()
        with self.assertRaisesRegex(mf.ManifestoInvalido, "faltam"):
            mf.verificar_saidas(self.pasta, m)

    def test_crlf_do_checkout_do_windows_e_tolerado_em_texto(self):
        m = self.publicar()
        (self.pasta / "a.csv").write_bytes(b"x,y\r\n1,2\r\n")
        mf.verificar_saidas(self.pasta, m)

    def test_manifesto_malformado(self):
        self.publicar()
        caminho = self.pasta / mf.ARQUIVO_MANIFESTO
        original = caminho.read_bytes()
        for nome, bruto in {"truncado": original[:20], "chave_extra": original[:-2] + b',"x":1}\n',
                            "indentado": json.dumps(json.loads(original), indent=2).encode(),
                            "versao": original.replace(b'"versao_esquema":1', b'"versao_esquema":9')}.items():
            with self.subTest(caso=nome):
                caminho.write_bytes(bruto)
                with self.assertRaises(mf.ManifestoInvalido):
                    mf.ler_manifesto(self.pasta)
        caminho.unlink()
        with self.assertRaises(mf.ManifestoInvalido):
            mf.ler_manifesto(self.pasta)

    def test_codigo_ambiente_e_configuracao(self):
        m = self.publicar()
        mf.verificar_ambiente_e_codigo(m, {"src/x.py": "2" * 64}, {"python": "3.9.13"})
        with self.assertRaisesRegex(mf.ManifestoInvalido, "código"):
            mf.verificar_ambiente_e_codigo(m, {"src/x.py": "3" * 64}, {"python": "3.9.13"})
        with self.assertRaisesRegex(mf.ManifestoInvalido, "código"):
            mf.verificar_ambiente_e_codigo(m, {"src/x.py": "2" * 64, "src/y.py": "4" * 64}, {"python": "3.9.13"})
        with self.assertRaisesRegex(mf.ManifestoInvalido, "ambiente"):
            mf.verificar_ambiente_e_codigo(m, {"src/x.py": "2" * 64}, {"python": "3.10"})
        with self.assertRaisesRegex(mf.ManifestoInvalido, "configuração"):
            mf.verificar_ambiente_e_codigo(m, {"src/x.py": "2" * 64}, {"python": "3.9.13"}, "0" * 64)

    def test_codigo_e_ambiente_reais_sao_json_serializaveis(self):
        json.dumps(ck.hashes_do_codigo())
        self.assertIn("xgboost", ck.descricao_do_ambiente()["versoes"])


@unittest.skipIf(GIT is None, "git indisponível")
class TestGit(BaseManifesto):
    def git(self, *args):
        return subprocess.run([GIT, "-C", str(self.pasta), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                              capture_output=True, check=True, text=True).stdout.strip()

    def commit(self, arquivos: dict, mensagem: str) -> str:
        for nome, conteudo in arquivos.items():
            caminho = self.pasta / nome
            caminho.parent.mkdir(parents=True, exist_ok=True)
            caminho.write_text(conteudo, encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", mensagem)
        return self.git("rev-parse", "HEAD")

    def setUp(self):
        super().setUp()
        self.git("init", "-q")
        self.head = self.commit({"src/codigo.py": "x = 1\n"}, "código")
        self.manifesto_ = {"git": {"head": self.head, "branch": "main", "limpo": True}}

    def test_mesmo_head_e_arvore_limpa(self):
        self.assertEqual(mf.verificar_git(self.manifesto_, self.pasta), self.head)

    def test_artefatos_versionados_depois_da_execucao_sao_aceitos(self):
        novo = self.commit({f"{mf.PREFIXO_ARTEFATOS}/resumo_benchmark.md": "# r\n"}, "artefatos")
        self.assertEqual(mf.verificar_git(self.manifesto_, self.pasta), novo)

    def test_codigo_alterado_depois_da_execucao_e_recusado(self):
        self.commit({f"{mf.PREFIXO_ARTEFATOS}/a.csv": "x\n", "src/codigo.py": "x = 2\n"}, "mudou o código")
        with self.assertRaisesRegex(mf.ManifestoInvalido, "fora de reports/sprint6/benchmark"):
            mf.verificar_git(self.manifesto_, self.pasta)

    def test_arvore_suja_e_recusada(self):
        (self.pasta / "src" / "codigo.py").write_text("x = 3\n", encoding="utf-8")
        with self.assertRaisesRegex(mf.ManifestoInvalido, "suja"):
            mf.verificar_git(self.manifesto_, self.pasta)

    def test_head_que_nao_e_ancestral_e_recusado(self):
        self.git("checkout", "-q", "--orphan", "outra")
        self.commit({"z.txt": "z\n"}, "outra história")
        with self.assertRaisesRegex(mf.ManifestoInvalido, "ancestral"):
            mf.verificar_git(self.manifesto_, self.pasta)


if __name__ == "__main__":
    unittest.main()
