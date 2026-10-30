"""Testes do Incremento 2C-B — checkpoints e retomada (``src/incremental/checkpoints.py``).

Exclusivamente sintéticos: repositórios git temporários, entradas fictícias,
identificadores fictícios (LIDC-IDRI-9xxx) e payloads inventados. Não leem
dados reais, não treinam e não executam CV. Aprovações, quando usadas, são
fictícias e ficam num repositório temporário.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/incremental -v
"""

from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
for caminho in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.incremental import checkpoints as ck  # noqa: E402
from src.incremental import config as cfg  # noqa: E402
from src.incremental import guardas as g  # noqa: E402
from src.incremental import rastreabilidade as rst  # noqa: E402
from test_guardas import (  # noqa: E402
    HASH_CONFIG_INCREMENTO_1, VALOR_P2, VALOR_P3, RepoTemporario, aprovar, protocolo, registro,
)

E0, E1, E3 = cfg.Etapa.E0.value, cfg.Etapa.E1.value, cfg.Etapa.E3.value
UTC = dt.timezone.utc


def ids(inicio: int, n: int) -> list:
    return [f"LIDC-IDRI-{9000 + i:04d}_N01_scan{9000 + i}" for i in range(inicio, inicio + n)]


FP = ck.fingerprint_folds({"treino_externo": ids(0, 8), "teste_externo": ids(8, 2)})
FP_OUTRO = ck.fingerprint_folds({"treino_externo": ids(0, 7), "teste_externo": ids(7, 3)})
PAYLOAD = {"predicoes": [{"nodulo_id": ids(8, 1)[0], "proba": 0.73}], "coeficientes": [0.1, -0.2]}
RESUMO = {"auc_fold": 0.81, "n_teste": 2}


class ComSessao(unittest.TestCase):
    def setUp(self):
        self.repo = RepoTemporario()
        self.addCleanup(self.repo.remover)
        self.raiz = self.repo.raiz
        self.repo.commit({".gitignore": "checkpoints/\nlivre/\n", "entradas/tabela.csv": "a,b\n1,2\n"},
                         "2026-10-01")
        self.entrada = self.raiz / "entradas" / "tabela.csv"
        self.dir = self.raiz / "checkpoints"
        self.dir.mkdir()

    def sessao(self, atividade=E0, decisoes=cfg.DECISOES, diretorio=None, **kwargs) -> ck.SessaoCheckpoints:
        autorizacao = kwargs.pop("autorizacao", None) or g.exigir_autorizacao(atividade, decisoes, self.raiz)
        return ck.SessaoCheckpoints(
            diretorio or self.dir, atividade, autorizacao, kwargs.pop("entradas", {"tabela": self.entrada}),
            decisoes=decisoes, repo=self.raiz,
            relogio_utc=kwargs.pop("relogio_utc", lambda: dt.datetime(2026, 10, 12, 12, tzinfo=UTC)), **kwargs)

    @staticmethod
    def ident(sessao, r=0, f=0, modelo="M0", seeds=None, fp=FP) -> ck.IdentidadeUnidade:
        return sessao.identidade(r, f, modelo, seeds or {"cv_externa": 42 + r}, fp)

    def pasta(self, sessao, identidade) -> Path:
        return self.dir / sessao.atividade / identidade.slot

    def assertNuncaConcluida(self, sessao, identidade, erro=ck.CheckpointCorrompido):
        with self.assertRaises(erro):
            sessao.estado(identidade)
        with self.assertRaises(erro):
            sessao.carregar(identidade)
        with self.assertRaises(rst.ErroRastreabilidade):
            sessao.salvar(identidade, PAYLOAD)


# --------------------------------------------------------------------------- #
# Identidade e fingerprint
# --------------------------------------------------------------------------- #


class TestIdentidade(ComSessao):
    def test_fingerprint_deterministico_sem_gravar_ids(self):
        invertido = ck.fingerprint_folds({"teste_externo": ids(8, 2)[::-1], "treino_externo": ids(0, 8)[::-1]})
        self.assertEqual(FP, invertido)
        self.assertNotEqual(FP, FP_OUTRO)
        self.assertNotEqual(FP, ck.fingerprint_folds({"treino": ids(0, 8), "teste_externo": ids(8, 2)}))
        for invalida in ({}, {"t": ids(0, 1) * 2}, {"": ids(0, 1)}, {"t": "abc"}, [("t", ids(0, 1))]):
            with self.subTest(composicao=invalida):
                with self.assertRaises((ValueError, TypeError)):
                    ck.fingerprint_folds(invalida)

    def test_identidade_completa_e_deterministica(self):
        a = self.ident(self.sessao())
        b = self.ident(self.sessao(relogio_utc=lambda: dt.datetime(2030, 1, 1, tzinfo=UTC)))
        self.assertEqual(a.sha256, b.sha256)
        d = a.como_dict()
        self.assertEqual(set(d), set(ck._CHAVES_IDENTIDADE))
        self.assertEqual((d["atividade"], d["repeticao"], d["fold"], d["modelo"]), (E0, 0, 0, "M0"))
        self.assertEqual(d["hash_configuracao"], HASH_CONFIG_INCREMENTO_1)
        self.assertEqual(d["dados"]["tabela"]["caminho"], "entradas/tabela.csv")
        self.assertEqual(d["codigo"]["src.incremental.checkpoints"]["caminho"], "src/incremental/checkpoints.py")
        self.assertEqual(set(d["ambiente"]), {"versoes", "implementacao", "plataforma"})
        self.assertEqual(d["fingerprint_folds"], FP)
        self.assertNotIn("LIDC-IDRI", json.dumps(d))

    def test_cada_componente_muda_a_identidade(self):
        s = self.sessao()
        base = self.ident(s).sha256
        for variacao in (dict(r=1), dict(f=1), dict(modelo="M1"), dict(seeds={"cv_externa": 7}),
                         dict(fp=FP_OUTRO)):
            with self.subTest(**{k: str(v) for k, v in variacao.items()}):
                self.assertNotEqual(self.ident(s, **variacao).sha256, base)

    def test_unidade_invalida(self):
        s = self.sessao()
        for args in ((-1, 0, "M0"), (10, 0, "M0"), (0, 5, "M0"), (True, 0, "M0"), (0, 0, "M9")):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    s.identidade(*args, {"cv_externa": 1}, FP)
        with self.assertRaises(ValueError):
            s.identidade(0, 0, "M0", {"cv_externa": "1"}, FP)
        with self.assertRaises(ValueError):
            s.identidade(0, 0, "M0", {"cv_externa": 1}, "abc")


# --------------------------------------------------------------------------- #
# Gravação e retomada
# --------------------------------------------------------------------------- #


class TestGravacaoERetomada(ComSessao):
    def test_salvar_e_carregar(self):
        s = self.sessao()
        unidade = self.ident(s)
        self.assertEqual(s.estado(unidade), ck.ESTADO_AUSENTE)
        registro_gravado = s.salvar(unidade, PAYLOAD, RESUMO)
        self.assertEqual(s.estado(unidade), ck.ESTADO_CONCLUIDA)
        self.assertEqual(s.carregar(unidade), PAYLOAD)
        self.assertEqual(sorted(p.name for p in self.pasta(s, unidade).iterdir()),
                         ["concluida.json", "payload.json"])
        self.assertEqual(registro_gravado["resumo"], RESUMO)
        self.assertEqual(registro_gravado["metadados"]["tentativa"], 1)
        # Retomada numa sessão nova (outro processo) reconhece a unidade concluída.
        nova = self.sessao()
        self.assertEqual(nova.carregar(self.ident(nova)), PAYLOAD)

    def test_registro_sem_dados_individuais_payload_so_no_diretorio_local(self):
        s = self.sessao()
        unidade = self.ident(s)
        s.salvar(unidade, PAYLOAD, RESUMO)
        texto_registro = (self.pasta(s, unidade) / "concluida.json").read_text(encoding="utf-8")
        self.assertNotRegex(texto_registro, r"LIDC-IDRI-\d{4}")
        self.assertNotIn("proba", texto_registro)
        self.assertIn("LIDC-IDRI-9008", (self.pasta(s, unidade) / "payload.json").read_text(encoding="utf-8"))
        self.assertEqual(self.repo.git("status", "--porcelain", "--untracked-files=all"), "")
        with self.assertRaises(rst.DadosIndividuais):
            s.salvar(self.ident(s, r=1), PAYLOAD, {"predicoes": [0.7]})
        self.assertFalse(self.pasta(s, self.ident(s, r=1)).exists())

    def test_sem_sobrescrita(self):
        s = self.sessao()
        unidade = self.ident(s)
        s.salvar(unidade, PAYLOAD)
        antes = {p.name: p.read_bytes() for p in self.pasta(s, unidade).iterdir()}
        with self.assertRaises(rst.SobrescritaRecusada):
            s.salvar(unidade, {"outro": 1})
        with self.assertRaises(rst.SobrescritaRecusada):
            s.registrar_falha(unidade, ValueError())
        self.assertEqual({p.name: p.read_bytes() for p in self.pasta(s, unidade).iterdir()}, antes)

    def test_conflito_de_escrita_entre_sessoes(self):
        a, b = self.sessao(), self.sessao()
        unidade_b = self.ident(b)
        a.salvar(self.ident(a), PAYLOAD)
        antes = {p.name: p.read_bytes() for p in self.pasta(a, unidade_b).iterdir()}
        # B decidiu executar antes de A terminar (estado obsoleto): a publicação recusa.
        with mock.patch.object(b, "estado", return_value=ck.ESTADO_AUSENTE):
            with self.assertRaises(rst.SobrescritaRecusada):
                b.salvar(unidade_b, {"outro": 1})
            with mock.patch.object(ck.os.path, "lexists", return_value=False):
                with self.assertRaises(rst.SobrescritaRecusada):
                    b.salvar(unidade_b, {"outro": 1})
        self.assertEqual({p.name: p.read_bytes() for p in self.pasta(a, unidade_b).iterdir()}, antes)
        self.assertEqual(b.carregar(unidade_b), PAYLOAD)

    def test_falhas_preservadas_e_nova_tentativa(self):
        s = self.sessao()
        unidade = self.ident(s)
        s.registrar_falha(unidade, ValueError("mensagem com LIDC-IDRI-9001 que não é gravada"))
        s.registrar_falha(unidade, KeyboardInterrupt)
        self.assertEqual(s.estado(unidade), ck.ESTADO_FALHOU)
        with self.assertRaises(ck.CheckpointNaoConcluido):
            s.carregar(unidade)
        registro_final = s.salvar(unidade, PAYLOAD)
        self.assertEqual(registro_final["metadados"]["tentativa"], 3)
        nomes = sorted(p.name for p in self.pasta(s, unidade).iterdir())
        self.assertEqual(nomes, ["concluida.json", "falha-001.json", "falha-002.json", "payload.json"])
        falha = json.loads((self.pasta(s, unidade) / "falha-001.json").read_text(encoding="utf-8"))
        self.assertEqual((falha["status"], falha["erro"], falha["payload"]), ("falhou", "ValueError", None))
        self.assertNotIn("LIDC", json.dumps(falha))
        self.assertEqual(s.estado(unidade), ck.ESTADO_CONCLUIDA)
        for erro in ("tem espaço", "", 3, None):
            with self.subTest(erro=erro):
                with self.assertRaises(ValueError):
                    s.registrar_falha(self.ident(s, r=1), erro)

    def test_carregar_ausente(self):
        s = self.sessao()
        with self.assertRaises(ck.CheckpointAusente):
            s.carregar(self.ident(s))

    def test_payload_invalido_nao_grava_nada(self):
        s = self.sessao()
        unidade = self.ident(s)
        for payload in ({"x": float("nan")}, ["lista"], {"x": object()}):
            with self.subTest(payload=str(payload)):
                with self.assertRaises((ValueError, TypeError)):
                    s.salvar(unidade, payload)
        self.assertEqual(s.estado(unidade), ck.ESTADO_AUSENTE)


# --------------------------------------------------------------------------- #
# Retomada indevida: interrupção, resíduos e corrupção
# --------------------------------------------------------------------------- #


class TestRetomadaIndevida(ComSessao):
    def setUp(self):
        super().setUp()
        self.s = self.sessao()
        self.unidade = self.ident(self.s)

    def concluir(self):
        self.s.salvar(self.unidade, PAYLOAD, RESUMO)
        return self.pasta(self.s, self.unidade)

    def test_interrupcao_entre_payload_e_registro(self):
        original = ck._publicar

        def publicar(conteudo, destino):
            if destino.name == ck.ARQUIVO_CONCLUIDA:
                raise KeyboardInterrupt  # processo morto antes do registro de conclusão
            original(conteudo, destino)

        with mock.patch.object(ck, "_publicar", publicar):
            with self.assertRaises(KeyboardInterrupt):
                self.s.salvar(self.unidade, PAYLOAD)
        pasta = self.pasta(self.s, self.unidade)
        self.assertEqual(sorted(p.name for p in pasta.iterdir()), ["payload.json"])
        self.assertNuncaConcluida(self.s, self.unidade)
        with self.assertRaisesRegex(ck.CheckpointIncompativel, "payload sem registro"):
            self.s.verificar([self.unidade])
        self.assertEqual(sorted(p.name for p in pasta.iterdir()), ["payload.json"])  # nada apagado

    def test_residuo_de_escrita_interrompida(self):
        pasta = self.concluir()
        (pasta / ".concluida.json.x1y2.tmp").write_bytes(b"{")
        with self.assertRaisesRegex(ck.CheckpointCorrompido, "resíduo de escrita interrompida"):
            self.s.estado(self.unidade)
        self.assertNuncaConcluida(self.s, self.unidade)

    def test_arquivo_desconhecido_ou_slot_que_nao_e_diretorio(self):
        pasta = self.concluir()
        (pasta / "notas.txt").write_text("x", encoding="utf-8")
        self.assertNuncaConcluida(self.s, self.unidade)
        outro = self.ident(self.s, r=1)
        (self.dir / E0).mkdir(exist_ok=True)
        (self.dir / E0 / outro.slot).write_text("x", encoding="utf-8")
        self.assertNuncaConcluida(self.s, outro)

    def test_corrupcao_do_registro_ou_do_payload(self):
        def truncar(pasta):
            (pasta / "concluida.json").write_bytes((pasta / "concluida.json").read_bytes()[:50])

        def editar_sem_hash(pasta):
            caminho = pasta / "concluida.json"
            caminho.write_bytes(caminho.read_bytes().replace(b'"auc_fold": 0.81', b'"auc_fold": 0.99'))

        def payload_alterado(pasta):
            (pasta / "payload.json").write_bytes((pasta / "payload.json").read_bytes().replace(b"0.73", b"0.74"))

        def payload_apagado(pasta):
            (pasta / "payload.json").unlink()

        def forma_nao_canonica(pasta):
            caminho = pasta / "concluida.json"
            caminho.write_bytes(json.dumps(json.loads(caminho.read_bytes()), sort_keys=True).encode("utf-8"))

        def chave_duplicada(pasta):
            caminho = pasta / "concluida.json"
            caminho.write_bytes(caminho.read_bytes().replace(b'{\n  "erro"', b'{\n  "erro": null,\n  "erro"', 1))

        def falha_renomeada(pasta):
            # Registro de falha válido colocado no lugar do de conclusão.
            (pasta / "concluida.json").unlink()
            (pasta / "payload.json").unlink()
            self.s.registrar_falha(self.unidade, ValueError())
            (pasta / "falha-001.json").rename(pasta / "concluida.json")

        for corromper in (truncar, editar_sem_hash, payload_alterado, payload_apagado, forma_nao_canonica,
                          chave_duplicada, falha_renomeada):
            with self.subTest(caso=corromper.__name__):
                shutil.rmtree(self.dir / E0, ignore_errors=True)
                corromper(self.concluir())
                self.assertNuncaConcluida(self.s, self.unidade)

    def test_adulteracao_com_hashes_recalculados(self):
        def regravar(pasta, alterar):
            caminho = pasta / "concluida.json"
            registro_lido = json.loads(caminho.read_text(encoding="utf-8"))
            alterar(registro_lido)
            registro_lido["sha256_identidade"] = rst.sha256_canonico(registro_lido["identidade"])
            corpo = {k: v for k, v in registro_lido.items() if k != "sha256_registro"}
            caminho.unlink()
            caminho.write_bytes(rst.serializar(ck._com_hash(corpo)))

        casos = {
            "status falhou com payload": lambda r: r.update(status="falhou", erro="ValueError"),
            "concluída com erro": lambda r: r.update(erro="ValueError"),
            "hash de payload trocado": lambda r: r["payload"].update(sha256="0" * 64),
            "identificador no resumo": lambda r: r["resumo"].update(nota="LIDC-IDRI-9001"),
            "caminho absoluto": lambda r: r["resumo"].update(nota="C:\\dados\\x.csv"),
            "formato": lambda r: r.update(versao_formato=2),
            "fingerprint de outro fold": lambda r: r["identidade"].update(fingerprint_folds=FP_OUTRO),
            "outra configuração": lambda r: r["identidade"].update(hash_configuracao="0" * 64),
            "outro modelo": lambda r: r["identidade"].update(modelo="M1"),
        }
        for nome, alterar in casos.items():
            with self.subTest(caso=nome):
                shutil.rmtree(self.dir / E0, ignore_errors=True)
                regravar(self.concluir(), alterar)
                with self.assertRaises((ck.CheckpointCorrompido, ck.CheckpointIncompativel)):
                    self.s.estado(self.unidade)
                with self.assertRaises((ck.CheckpointCorrompido, ck.CheckpointIncompativel)):
                    self.s.carregar(self.unidade)

    def test_slot_copiado_para_outra_unidade(self):
        pasta = self.concluir()
        outra = self.ident(self.s, r=1, seeds={"cv_externa": 42})
        shutil.copytree(pasta, self.dir / E0 / outra.slot)
        self.assertNuncaConcluida(self.s, outra, ck.CheckpointIncompativel)


# --------------------------------------------------------------------------- #
# Incompatibilidade de configuração, código, dados, folds, seeds e ambiente
# --------------------------------------------------------------------------- #


class TestIncompatibilidade(ComSessao):
    def setUp(self):
        super().setUp()
        s = self.sessao()
        s.salvar(self.ident(s), PAYLOAD)
        self.original = s

    def assertIncompativel(self, sessao, **ident):
        unidade = self.ident(sessao, **ident)
        with self.assertRaises(ck.CheckpointIncompativel):
            sessao.estado(unidade)
        with self.assertRaises(ck.CheckpointIncompativel):
            sessao.carregar(unidade)
        with self.assertRaises(ck.CheckpointIncompativel):
            sessao.verificar([unidade])
        with self.assertRaises(ck.CheckpointIncompativel):
            sessao.salvar(unidade, PAYLOAD)

    def test_configuracao_diferente(self):
        decisoes = aprovar(P1=("1" * 40, None))  # outra configuração; E0 não consulta o git
        self.assertIncompativel(self.sessao(decisoes=decisoes))

    def test_codigo_diferente(self):
        self.assertIncompativel(self.sessao(codigo={"extra": REPO_ROOT / "src" / "incremental" / "__init__.py"}))

    def test_dados_diferentes(self):
        self.entrada.write_bytes(b"a,b\n1,3\n")
        self.assertIncompativel(self.sessao())

    def test_folds_e_seeds_diferentes(self):
        s = self.sessao()
        self.assertIncompativel(s, fp=FP_OUTRO)
        self.assertIncompativel(s, seeds={"cv_externa": 7})

    def test_ambiente_diferente(self):
        amb = rst.ambiente()
        outro = {**amb, "versoes": {**amb["versoes"], "scikit-learn": "9.9.9"}}
        with mock.patch.object(rst, "ambiente", return_value=outro):
            s = self.sessao()
        self.assertIncompativel(s)

    def test_identidade_de_outra_sessao(self):
        outra = self.sessao(decisoes=aprovar(P1=("1" * 40, None)))
        unidade_original = self.ident(self.original)
        for operacao in (outra.estado, outra.carregar, lambda u: outra.salvar(u, PAYLOAD),
                         lambda u: outra.verificar([u]), lambda u: outra.registrar_falha(u, ValueError())):
            with self.subTest(operacao=getattr(operacao, "__name__", "lambda")):
                with self.assertRaisesRegex(ck.CheckpointIncompativel, "outra sessão"):
                    operacao(unidade_original)

    def operacoes_de_retomada(self, sessao, unidade):
        return {"estado": lambda: sessao.estado(unidade), "carregar": lambda: sessao.carregar(unidade),
                "verificar": lambda: sessao.verificar([unidade]),
                "registrar_falha": lambda: sessao.registrar_falha(self.ident(sessao, r=1), ValueError())}

    def test_entrada_alterada_depois_da_abertura_bloqueia_retomada(self):
        # Regressão: a sessão aberta antes da alteração não pode reaproveitar o checkpoint.
        s = self.sessao()
        unidade = self.ident(s)
        self.assertEqual(s.carregar(unidade), PAYLOAD)
        slot = self.dir / E0 / unidade.slot
        antes = {p.name: p.read_bytes() for p in slot.iterdir()}
        original = self.entrada.read_bytes()
        self.entrada.write_bytes(b"a,b\n1,9\n")
        for nome, operacao in self.operacoes_de_retomada(s, unidade).items():
            with self.subTest(operacao=nome):
                with self.assertRaisesRegex(rst.HashIncompativel, "dados alterado"):
                    operacao()
        self.assertEqual({p.name: p.read_bytes() for p in slot.iterdir()}, antes)  # nada apagado
        self.assertFalse((self.dir / E0 / self.ident(s, r=1).slot).exists())
        self.entrada.write_bytes(original)
        self.assertEqual(s.carregar(unidade), PAYLOAD)

    def test_codigo_alterado_depois_da_abertura_bloqueia_retomada(self):
        s = self.sessao()
        unidade = self.ident(s)
        original = rst.hash_arquivo

        def hash_com_codigo_alterado(caminho, raiz=rst.REPO_ROOT):
            h = original(caminho, raiz)
            if h.caminho == "src/incremental/config.py":
                return dataclasses.replace(h, sha256="0" * 64, sha256_lf="0" * 64)
            return h

        with mock.patch.object(rst, "hash_arquivo", hash_com_codigo_alterado):
            for nome, operacao in self.operacoes_de_retomada(s, unidade).items():
                with self.subTest(operacao=nome):
                    with self.assertRaisesRegex(rst.HashIncompativel, "codigo alterado"):
                        operacao()
        self.assertEqual(s.carregar(unidade), PAYLOAD)

    def test_dados_alterados_durante_a_sessao(self):
        s = self.sessao()
        self.entrada.write_bytes(b"a,b\n5,5\n")
        with self.assertRaises(rst.HashIncompativel):
            s.salvar(self.ident(s, r=1), PAYLOAD)
        self.assertFalse((self.dir / E0 / self.ident(s, r=1).slot).exists())


# --------------------------------------------------------------------------- #
# Verificação do conjunto esperado
# --------------------------------------------------------------------------- #


class TestVerificar(ComSessao):
    def test_classifica_sem_declarar_etapa_concluida(self):
        s = self.sessao()
        esperadas = [self.ident(s, r=r, f=f) for r in range(2) for f in range(2)]
        s.salvar(esperadas[0], PAYLOAD)
        s.salvar(esperadas[3], PAYLOAD)
        s.registrar_falha(esperadas[1], ValueError())
        relatorio = s.verificar(esperadas)
        self.assertEqual(relatorio.concluidas, ("r000-f00-M0", "r001-f01-M0"))
        self.assertEqual(relatorio.falhas, ("r000-f01-M0",))
        self.assertEqual(relatorio.ausentes, ("r001-f00-M0",))
        self.assertEqual(relatorio.a_executar, ("r000-f01-M0", "r001-f00-M0"))
        self.assertEqual(relatorio.n_esperadas, 4)
        nomes = {f.name for f in dataclasses.fields(relatorio)} | {n for n in dir(relatorio) if not n.startswith("_")}
        self.assertFalse({n for n in nomes if "etapa" in n or n in ("concluida", "completa", "finalizada")})
        # Tudo concluído continua sendo só um relatório de unidades.
        for unidade in (esperadas[1], esperadas[2]):
            s.salvar(unidade, PAYLOAD)
        self.assertEqual(s.verificar(esperadas).a_executar, ())

    def test_slot_nao_esperado_aborta(self):
        s = self.sessao()
        s.salvar(self.ident(s, r=1), PAYLOAD)
        with self.assertRaisesRegex(ck.CheckpointIncompativel, "não esperados"):
            s.verificar([self.ident(s, r=0)])

    def test_problemas_sao_listados_juntos(self):
        s = self.sessao()
        a, b = self.ident(s, r=0), self.ident(s, r=1)
        s.salvar(a, PAYLOAD)
        s.salvar(b, PAYLOAD)
        (self.pasta(s, a) / "payload.json").write_bytes(b"{}\n")
        (self.pasta(s, b) / "extra.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(ck.CheckpointIncompativel) as ctx:
            s.verificar([a, b])
        self.assertIn(a.slot, str(ctx.exception))
        self.assertIn(b.slot, str(ctx.exception))
        self.assertNotIn("LIDC", str(ctx.exception))

    def test_esperadas_repetidas(self):
        s = self.sessao()
        with self.assertRaises(ValueError):
            s.verificar([self.ident(s), self.ident(s)])


# --------------------------------------------------------------------------- #
# Guardas e diretório local
# --------------------------------------------------------------------------- #


class TestGuardasEDiretorio(ComSessao):
    def test_autorizacao_forjada_ou_de_outra_atividade(self):
        aut = g.exigir_autorizacao(E0)
        with self.assertRaisesRegex(rst.ErroRastreabilidade, "não confere"):
            self.sessao(autorizacao=dataclasses.replace(aut, head="0" * 40))
        with self.assertRaisesRegex(rst.ErroRastreabilidade, "outra atividade"):
            self.sessao(atividade=cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS, autorizacao=aut)
        with self.assertRaises(TypeError):
            self.sessao(autorizacao=aut.como_dict())

    def test_etapas_bloqueadas_continuam_bloqueadas(self):
        # P1 e P2 pendentes: uma autorização forjada da Etapa 1 não abre sessão.
        forjada = g.Autorizacao(E1, ("P1", "P2"), (), None, HASH_CONFIG_INCREMENTO_1)
        with self.assertRaises(cfg.BloqueioDecisao):
            ck.SessaoCheckpoints(self.dir, E1, forjada, {"tabela": self.entrada}, repo=self.raiz)
        self.assertFalse((self.dir / E1).exists())

    def test_modelos_com_painel_exigem_p3_na_autorizacao(self):
        r12 = [registro("P1"), registro("P2", VALOR_P2)]
        c1 = self.repo.commit(protocolo(*r12), "2026-10-02")
        c2 = self.repo.commit(protocolo(*r12, registro("P3", VALOR_P3, "2026-10-03")), "2026-10-03")
        e1 = aprovar(P1=(c1, None), P2=(c1, VALOR_P2))
        s1 = self.sessao(atividade=E1, decisoes=e1)
        self.ident(s1, modelo="M2v")
        for modelo in cfg.MODELOS_COM_PAINEL:
            with self.subTest(modelo=modelo):
                with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
                    self.ident(s1, modelo=modelo)
        # Identidade de M2 montada à mão não contorna a exigência.
        forjada = dataclasses.replace(self.ident(s1), modelo="M2")
        for operacao in (s1.estado, s1.carregar, lambda u: s1.salvar(u, PAYLOAD)):
            with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
                operacao(forjada)
        s3 = self.sessao(atividade=E3, decisoes=aprovar(e1, P3=(c2, VALOR_P3)))
        unidade = self.ident(s3, modelo="M2")
        s3.salvar(unidade, PAYLOAD)
        self.assertEqual(s3.carregar(unidade), PAYLOAD)
        # O checkpoint da Etapa 3 não aparece numa sessão da Etapa 1.
        self.assertEqual(s1.verificar([self.ident(s1, modelo="M2v")]).ausentes, ("r000-f00-M2v",))

    def test_modulo_nao_le_dados_nem_da_acesso_a_particoes(self):
        fonte = (REPO_ROOT / "src" / "incremental" / "checkpoints.py").read_text(encoding="utf-8")
        arvore = ast.parse(fonte)
        importados = set()
        for no in ast.walk(arvore):
            if isinstance(no, ast.Import):
                importados.update(a.name.split(".")[0] for a in no.names)
            elif isinstance(no, ast.ImportFrom):
                base = ("." * no.level) + (no.module or "")
                importados.add(base)
                importados.update(f"{base}{a.name}" if base == "." else a.name for a in no.names)
        self.assertFalse(importados & {"pickle", "marshal", "shelve", "dill", "joblib", "numpy", "pandas",
                                       "sklearn", ".dados", "scripts"})
        chamadas = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
                    for n in ast.walk(arvore) if isinstance(n, ast.Call)}
        self.assertFalse(chamadas & {"eval", "exec", "loads_pickle", "load", "carregar_dados", "read_csv", "fit"})
        self.assertNotIn("dados.carregar", fonte)

    def test_reutiliza_rastreabilidade_sem_duplicar(self):
        arvore = ast.parse((REPO_ROOT / "src" / "incremental" / "checkpoints.py").read_text(encoding="utf-8"))
        definidas = {n.name for n in ast.walk(arvore) if isinstance(n, ast.FunctionDef)}
        self.assertFalse(definidas & {"calcular_sha256", "hash_arquivo", "ambiente", "estado_git", "json_canonico",
                                      "sha256_canonico", "_exigir_sem_dados_individuais", "_validar_seeds", "_git"})
        usadas = {n.attr for n in ast.walk(arvore)
                  if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "rst"}
        self.assertLessEqual({"hash_arquivo", "calcular_sha256", "ambiente", "codigo_do_pacote", "estado_git",
                              "json_canonico", "sha256_canonico", "serializar", "_exigir_sem_dados_individuais",
                              "_validar_seeds"}, usadas)

    def test_diretorio_local_nao_versionado(self):
        versionavel = self.raiz / "versionavel"
        versionavel.mkdir()
        with self.assertRaisesRegex(ck.DiretorioNaoLocal, "não é ignorado"):
            self.sessao(diretorio=versionavel)
        (self.raiz / "data").mkdir()
        with self.assertRaisesRegex(ck.DiretorioNaoLocal, "data/"):
            self.sessao(diretorio=self.raiz / "data")
        with self.assertRaisesRegex(ck.DiretorioNaoLocal, "inexistente"):
            self.sessao(diretorio=self.raiz / "nao_existe")
        livre = self.raiz / "livre"
        livre.mkdir()
        self.sessao(diretorio=livre)

    def test_falha_do_git_nunca_libera_e_fora_de_repositorio_e_aceito(self):
        verificar_diretorio = ck.SessaoCheckpoints._exigir_diretorio_local
        sessao_falsa = mock.Mock(repo=self.raiz)
        with mock.patch.object(rst, "_git", return_value=mock.Mock(returncode=128)):
            with self.assertRaises(rst.ErroGit):
                verificar_diretorio(sessao_falsa, self.dir)
        # Nesta máquina o home é um repositório git; "fora de qualquer repositório" é simulado.
        with mock.patch.object(ck, "_dentro_de_repositorio", return_value=False), \
                mock.patch.object(rst, "_git", side_effect=AssertionError("git não deveria ser chamado")):
            self.assertEqual(verificar_diretorio(sessao_falsa, self.dir), self.dir.resolve())

    def test_configuracao_e_decisoes_inalteradas(self):
        self.assertEqual(cfg.hash_configuracao(), HASH_CONFIG_INCREMENTO_1)
        self.assertTrue(all(not d.aprovada for d in cfg.DECISOES.values()))


if __name__ == "__main__":
    unittest.main()
