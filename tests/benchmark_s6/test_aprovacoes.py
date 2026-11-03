"""Testes das guardas de aprovação do benchmark S6 (``src/benchmark_s6/aprovacoes.py``).

Todos os cenários usam repositórios git TEMPORÁRIOS e SINTÉTICOS, criados
pelo próprio teste, com documento, configuração e manifesto sintéticos e
datas de commit controladas. Nenhum teste lê o repositório real, documentos
oficiais, aprovações reais ou dados de pacientes, e nenhum grava registro em
documento versionado do projeto.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/benchmark_s6 -v
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark_s6 import aprovacoes as ap  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.incremental import config as cfg_s4  # noqa: E402
from src.incremental import guardas, rastreabilidade  # noqa: E402

GIT = shutil.which("git")
SHA_CFG = cfg.sha256_configuracao()
DOC = ap.DOCUMENTO_APROVACAO
MANIFESTO = "reports/sprint4/etapa3_sintetica/manifesto.json"
SAIDA = "reports/sprint4/etapa3_sintetica/resumo.json"
CONTEUDO_SAIDA = b'{"sintetico":true}\n'


# --------------------------------------------------------------------------- #
# Repositório, documento e manifesto sintéticos
# --------------------------------------------------------------------------- #


class RepoSintetico:
    """Repositório git descartável. Assinatura de commits desativada só nele."""

    def __init__(self, raiz: Path) -> None:
        self.raiz = raiz
        self.git("init", "-q")
        self.git("symbolic-ref", "HEAD", "refs/heads/main")
        for chave, valor in (("user.name", "Teste Sintetico"), ("user.email", "teste@example.invalid"),
                             ("commit.gpgsign", "false"), ("core.autocrlf", "false")):
            self.git("config", chave, valor)

    def git(self, *args: str, data: Optional[str] = None) -> str:
        ambiente = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
        if data is not None:
            ambiente["GIT_AUTHOR_DATE"] = ambiente["GIT_COMMITTER_DATE"] = f"{data}T12:00:00+0000"
        processo = subprocess.run([GIT, "-C", str(self.raiz), *args], capture_output=True, text=True,
                                  env=ambiente, timeout=60)
        if processo.returncode != 0:
            raise AssertionError(f"git {args}: {processo.stderr}")
        return processo.stdout.strip()

    def escrever(self, caminho: str, conteudo) -> None:
        destino = self.raiz / caminho
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(conteudo if isinstance(conteudo, bytes) else conteudo.encode("utf-8"))

    def remover(self, caminho: str) -> None:
        (self.raiz / caminho).unlink()

    def commit(self, data: str, mensagem: str = "sintetico") -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", mensagem, data=data)
        return self.git("rev-parse", "HEAD")


def registro(codigo: str, data: str, sha: str, status: str = "aprovada") -> str:
    return f"APROVACAO-DECISAO codigo={codigo} status={status} data={data} valor_sha256={sha}"


def documento(registros: Sequence[str] = (), anotacao: Optional[str] = SHA_CFG, extra: str = "") -> str:
    linhas = ["# Pré-registro P9 — documento SINTÉTICO de teste", "", "## 19. Registro de aprovação", ""]
    if anotacao is not None:
        linhas.append(f"**SHA-256 da configuração canônica:** `{anotacao}` (Seção 14.4).")
    linhas += ["", "Formato (exemplo, que não conta como registro):", "", "```",
               "APROVACAO-DECISAO codigo=P9 status=aprovada data=AAAA-MM-DD valor_sha256=<64 hex>",
               registro("P9", "2026-01-01", SHA_CFG), "```", "", "**Registros de aprovação:**", ""]
    return "\n".join([*linhas, *registros, extra]) + "\n"


def com_hashes(corpo: dict) -> dict:
    """Mesma regra de ``rastreabilidade._com_hashes``, com a função pública de hash."""
    manifesto = dict(corpo)
    manifesto["sha256_execucao"] = rastreabilidade.sha256_canonico(corpo["execucao"])
    manifesto["sha256_reprodutivel"] = rastreabilidade.sha256_canonico(
        {"execucao": corpo["execucao"], "resultado": corpo["resultado"]})
    manifesto["sha256_manifesto"] = rastreabilidade.sha256_canonico(manifesto)
    return manifesto


def manifesto_s4(head: str, mudar: Optional[Callable[[dict], None]] = None) -> dict:
    """Manifesto SINTÉTICO da Etapa 3, definitivo, elegível e concluído."""
    sha = hashlib.sha256(CONTEUDO_SAIDA).hexdigest()
    git = {"head": head, "branch": "main", "limpo": True, "alteracoes": []}
    corpo = {
        "formato": rastreabilidade.FORMATO,
        "versao_formato": rastreabilidade.VERSAO_FORMATO,
        "execucao": {
            "atividade": cfg_s4.Etapa.E3.value, "finalidade": "definitiva", "elegivel_definitiva": True,
            "autorizacao": {"atividade": cfg_s4.Etapa.E3.value, "sintetico": True},
            "hash_configuracao": "a" * 64, "seeds": {"cv": 42}, "parametros": {}, "git": dict(git),
            "ambiente": {"confere_com_esperado": True}, "entradas": {}, "codigo": {},
        },
        "resultado": {
            "status": "concluida", "erro": None, "warnings": [],
            "saidas": {"resumo": {"caminho": SAIDA, "sha256": sha, "sha256_lf": sha,
                                  "bytes": len(CONTEUDO_SAIDA)}},
            "git_final": {**git, "alteracoes_alem_das_saidas": []},
        },
        "tempo": {"inicio": "2026-11-06T10:00:00.000000Z", "termino": "2026-11-06T11:00:00.000000Z",
                  "duracao_s": 3600.0},
    }
    if mudar is not None:
        mudar(corpo)
    return com_hashes(corpo)


class Cenario:
    """Linha do tempo válida, que cada teste interrompe ou altera:

    c_cfg (03/11) configuração versionada e anotada → c_p9 (05/11) registro P9 →
    c_exec (06/11) HEAD da execução da Etapa 3 → c_man (07/11) saída e manifesto →
    c_s4 (08/11) registro S4_E3_REGISTRADA.
    """

    def __init__(self, repo: RepoSintetico) -> None:
        self.repo = repo
        self.registro_p9 = registro("P9", "2026-11-05", SHA_CFG)
        self.registro_s4: Optional[str] = None
        self.manifesto: Optional[dict] = None

    def configuracao(self, data: str = "2026-11-03") -> str:
        self.repo.escrever(cfg.ARQUIVO_CONFIGURACAO, cfg.bytes_arquivo())
        self.repo.escrever(DOC, documento())
        self.c_cfg = self.repo.commit(data, "configuração canônica")
        return self.c_cfg

    def p9(self, data: str = "2026-11-05") -> str:
        self.repo.escrever(DOC, documento([self.registro_p9]))
        self.c_p9 = self.repo.commit(data, "aprovação P9")
        return self.c_p9

    def manifesto_e3(self, mudar: Optional[Callable[[dict], None]] = None) -> str:
        self.repo.escrever("src/etapa3_sintetica.txt", "execução sintética\n")
        self.c_exec = self.repo.commit("2026-11-06", "execução da Etapa 3")
        self.manifesto = manifesto_s4(self.c_exec, mudar)
        self.repo.escrever(SAIDA, CONTEUDO_SAIDA)
        self.repo.escrever(MANIFESTO, rastreabilidade.serializar(self.manifesto))
        self.c_man = self.repo.commit("2026-11-07", "manifesto da Etapa 3")
        return self.c_man

    def s4(self, data: str = "2026-11-08", sha: Optional[str] = None) -> str:
        sha = sha or rastreabilidade.sha256_canonico(self.manifesto)
        self.registro_s4 = registro("S4_E3_REGISTRADA", data, sha)
        self.repo.escrever(DOC, documento([self.registro_p9, self.registro_s4]))
        self.c_s4 = self.repo.commit(data, "registro S4_E3_REGISTRADA")
        return self.c_s4

    def completo(self) -> "Cenario":
        self.configuracao()
        self.p9()
        self.manifesto_e3()
        self.s4()
        return self


@unittest.skipIf(GIT is None, "git indisponível")
class BaseRepo(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.raiz = Path(self._tmp.name) / "repo"
        self.raiz.mkdir()
        self.repo = RepoSintetico(self.raiz)
        self.cenario = Cenario(self.repo)

    def autorizar(self, **kwargs) -> ap.AutorizacaoDefinitiva:
        kwargs.setdefault("manifesto_s4", MANIFESTO)
        return ap.autorizar_definitivo(repo=self.raiz, **kwargs)

    def assertBloqueia(self, excecao, *, mensagem: Optional[str] = None, **kwargs):
        with self.assertRaises(excecao) as contexto:
            self.autorizar(**kwargs)
        if mensagem is not None:
            self.assertIn(mensagem, str(contexto.exception))
        return contexto.exception


# --------------------------------------------------------------------------- #
# Autorização válida
# --------------------------------------------------------------------------- #


class TestAutorizacaoValida(BaseRepo):

    def test_autoriza_cenario_completo(self):
        c = self.cenario.completo()
        autorizacao = self.autorizar()
        self.assertEqual(autorizacao.head, c.c_s4)
        self.assertEqual(autorizacao.sha256_configuracao, SHA_CFG)
        self.assertEqual((autorizacao.p9.commit, autorizacao.p9.data, autorizacao.p9.valor_sha256),
                         (c.c_p9, "2026-11-05", SHA_CFG))
        self.assertEqual(autorizacao.s4_e3.commit, c.c_s4)
        self.assertEqual(autorizacao.s4_e3.valor_sha256, rastreabilidade.sha256_canonico(c.manifesto))
        self.assertEqual(json.loads(json.dumps(autorizacao.como_dict()))["p9"]["commit"], c.c_p9)

    def test_commits_informados_corretos(self):
        c = self.cenario.completo()
        self.autorizar(commit_p9=c.c_p9, commit_s4_e3=c.c_s4)
        self.assertEqual(ap.verificar_p9(self.raiz, commit_aprovacao=c.c_p9).commit, c.c_p9)
        self.assertEqual(ap.verificar_s4_e3(MANIFESTO, self.raiz, commit_registro=c.c_s4).commit, c.c_s4)

    def test_aprovacao_vinda_de_branch_mesclado(self):
        c = self.cenario
        c.configuracao()
        self.repo.git("checkout", "-q", "-b", "aprovacao")
        c.p9()
        self.repo.git("checkout", "-q", "main")
        self.repo.escrever("outro.txt", "x\n")
        self.repo.commit("2026-11-05")
        self.repo.git("merge", "-q", "--no-ff", "--no-edit", "aprovacao", data="2026-11-05")
        c.manifesto_e3()
        c.s4()
        self.assertEqual(self.autorizar().p9.commit, c.c_p9)

    def test_reaprovacao_explicita_depois_de_revogacao(self):
        c = self.cenario.completo()
        revogacao = registro("P9", "2026-11-09", SHA_CFG, status="revogada")
        self.repo.escrever(DOC, documento([c.registro_p9, c.registro_s4, revogacao]))
        self.repo.commit("2026-11-09")
        novo = registro("P9", "2026-11-10", SHA_CFG)
        self.repo.escrever(DOC, documento([novo, c.registro_s4]))
        c_novo = self.repo.commit("2026-11-10")
        self.assertEqual(self.autorizar().p9.commit, c_novo)

    def test_nao_escreve_no_repositorio(self):
        self.cenario.completo()
        self.autorizar()
        self.assertEqual(self.repo.git("status", "--porcelain", "--untracked-files=all"), "")

    def test_registro_igual_ao_formatado_pela_s4(self):
        linha = guardas.formatar_registro("P9", "2026-11-05", cfg.conteudo_canonico())
        self.assertEqual(linha, registro("P9", "2026-11-05", SHA_CFG))
        self.assertEqual(ap.ler_registros(linha), (ap.Registro("P9", "aprovada", "2026-11-05", SHA_CFG),))


# --------------------------------------------------------------------------- #
# Execução prematura e registros ausentes
# --------------------------------------------------------------------------- #


class TestExecucaoPrematura(BaseRepo):

    def test_sem_documento(self):
        self.repo.escrever("LEIA.md", "vazio\n")
        self.repo.commit("2026-11-03")
        self.assertBloqueia(ap.RegistroAusente, mensagem="não existe no HEAD")

    def test_sem_nenhum_registro(self):
        self.cenario.configuracao()
        self.assertBloqueia(ap.RegistroAusente, mensagem="P9")

    def test_p9_aprovado_sem_s4_e3(self):
        c = self.cenario
        c.configuracao()
        c.p9()
        c.manifesto_e3()
        self.assertBloqueia(ap.RegistroAusente, mensagem="S4_E3_REGISTRADA")

    def test_s4_e3_sem_p9(self):
        c = self.cenario
        c.configuracao()
        c.manifesto_e3()
        sha = rastreabilidade.sha256_canonico(c.manifesto)
        self.repo.escrever(DOC, documento([registro("S4_E3_REGISTRADA", "2026-11-08", sha)]))
        self.repo.commit("2026-11-08")
        self.assertBloqueia(ap.RegistroAusente, mensagem="P9")

    def test_sem_manifesto_informado(self):
        self.cenario.completo()
        self.assertBloqueia(ap.RegistroAusente, manifesto_s4=None, mensagem="manifesto")

    def test_exemplos_e_registros_escondidos_nao_contam(self):
        self.cenario.configuracao()
        escondidos = "\n".join([
            "~~~", registro("P9", "2026-11-05", SHA_CFG), "~~~",
            "<!--", registro("P9", "2026-11-05", SHA_CFG), "-->",
            "    " + registro("P9", "2026-11-05", SHA_CFG),
            "- " + registro("P9", "2026-11-05", SHA_CFG),
            "> " + registro("P9", "2026-11-05", SHA_CFG),
        ])
        self.repo.escrever(DOC, documento(extra=escondidos))
        self.repo.commit("2026-11-05")
        self.assertBloqueia(ap.RegistroAusente, mensagem="P9")


# --------------------------------------------------------------------------- #
# Hash alterado
# --------------------------------------------------------------------------- #


class TestHashAlterado(BaseRepo):

    def test_registro_p9_com_outro_hash(self):
        c = self.cenario
        c.configuracao()
        c.registro_p9 = registro("P9", "2026-11-05", "0" * 64)
        c.p9()
        excecao = self.assertBloqueia(ap.DivergenciaAprovacao)
        self.assertEqual(excecao.chave_saida, "integridade")

    def test_configuracao_do_codigo_diferente_da_aprovada(self):
        self.cenario.completo()
        outra = dataclasses.replace(cfg.CONFIGURACAO, bootstrap=dataclasses.replace(
            cfg.CONFIGURACAO.bootstrap, n_replicas=2000))
        self.assertBloqueia(ap.DivergenciaAprovacao, configuracao=outra)

    def test_arquivo_de_configuracao_alterado_depois_da_aprovacao(self):
        self.cenario.completo()
        conteudo = cfg.conteudo_canonico()
        conteudo["versao_esquema"] = 2
        self.repo.escrever(cfg.ARQUIVO_CONFIGURACAO, (rastreabilidade.json_canonico(conteudo) + "\n").encode())
        self.repo.commit("2026-11-09")
        self.assertBloqueia(ap.DivergenciaAprovacao, mensagem="no HEAD")

    def test_arquivo_de_configuracao_fora_da_forma_canonica(self):
        self.cenario.completo()
        self.repo.escrever(cfg.ARQUIVO_CONFIGURACAO,
                           json.dumps(cfg.conteudo_canonico(), indent=2, sort_keys=True).encode() + b"\n")
        self.repo.commit("2026-11-09")
        self.assertBloqueia(ap.DivergenciaAprovacao, mensagem="forma canônica")

    def test_arquivo_de_configuracao_removido(self):
        self.cenario.completo()
        self.repo.remover(cfg.ARQUIVO_CONFIGURACAO)
        self.repo.commit("2026-11-09")
        self.assertBloqueia(ap.BloqueioAprovacao, mensagem="não está versionado no HEAD")

    def test_registro_s4_e3_com_outro_hash(self):
        c = self.cenario
        c.configuracao()
        c.p9()
        c.manifesto_e3()
        c.s4(sha="1" * 64)
        self.assertBloqueia(ap.DivergenciaAprovacao, mensagem="S4_E3_REGISTRADA")

    def test_manifesto_alterado_depois_do_registro(self):
        c = self.cenario.completo()
        alterado = manifesto_s4(c.c_exec, lambda m: m["resultado"]["warnings"].append("tarde"))
        self.repo.escrever(MANIFESTO, rastreabilidade.serializar(alterado))
        self.repo.commit("2026-11-09")
        self.assertBloqueia(ap.ManifestoS4Invalido, mensagem="mudou depois do registro")

    def test_saida_alterada_depois_do_registro(self):
        self.cenario.completo()
        self.repo.escrever(SAIDA, b'{"sintetico":false}\n')
        self.repo.commit("2026-11-09")
        self.assertBloqueia(ap.ManifestoS4Invalido, mensagem="difere do registrado")

    def test_saida_removida(self):
        self.cenario.completo()
        self.repo.remover(SAIDA)
        self.repo.commit("2026-11-09")
        self.assertBloqueia(ap.ManifestoS4Invalido, mensagem="não versionada")


# --------------------------------------------------------------------------- #
# Revogação
# --------------------------------------------------------------------------- #


class TestRevogacao(BaseRepo):

    def _documento(self, *registros: str) -> None:
        self.repo.escrever(DOC, documento(list(registros)))
        self.repo.commit("2026-11-09")

    def test_revogacao_no_head(self):
        c = self.cenario.completo()
        self._documento(c.registro_p9, c.registro_s4, registro("P9", "2026-11-09", SHA_CFG, "revogada"))
        self.assertBloqueia(ap.RegistroRevogado, mensagem="P9")

    def test_revogacao_de_s4_e3(self):
        c = self.cenario.completo()
        sha = rastreabilidade.sha256_canonico(c.manifesto)
        self._documento(c.registro_p9, registro("S4_E3_REGISTRADA", "2026-11-09", sha, "revogada"))
        self.assertBloqueia(ap.RegistroRevogado, mensagem="S4_E3_REGISTRADA")

    def test_registro_apagado(self):
        c = self.cenario.completo()
        self._documento(c.registro_s4)
        self.assertBloqueia(ap.RegistroAusente, mensagem="P9")

    def test_revogacao_apagada_nao_reabilita(self):
        c = self.cenario.completo()
        self._documento(c.registro_p9, c.registro_s4, registro("P9", "2026-11-09", SHA_CFG, "revogada"))
        self.repo.escrever(DOC, documento([c.registro_p9, c.registro_s4]))
        self.repo.commit("2026-11-10")
        self.assertBloqueia(ap.RegistroRevogado, mensagem="não é anterior à aprovação")

    def test_revogacao_em_branch_paralelo_descartada_no_merge(self):
        c = self.cenario
        c.configuracao()
        self.repo.git("checkout", "-q", "-b", "paralelo")
        self.repo.escrever(DOC, documento([registro("P9", "2026-11-06", SHA_CFG, "revogada")]))
        self.repo.commit("2026-11-06")
        self.repo.git("checkout", "-q", "main")
        c.p9()
        self.repo.git("merge", "-q", "--no-commit", "-s", "ours", "paralelo", data="2026-11-07")
        self.repo.escrever(DOC, documento([c.registro_p9]))
        self.repo.commit("2026-11-07", "merge que descarta a revogação")
        with self.assertRaises(ap.RegistroRevogado):
            ap.verificar_p9(self.raiz)

    def test_mesmo_registro_reintroduzido(self):
        c = self.cenario.completo()
        self._documento(c.registro_s4)
        self.repo.escrever(DOC, documento([c.registro_p9, c.registro_s4]))
        self.repo.commit("2026-11-10")
        self.assertBloqueia(ap.RegistroInvalido, mensagem="introduzido 2 vezes")


# --------------------------------------------------------------------------- #
# Ancestralidade, ordem e datas
# --------------------------------------------------------------------------- #


class TestAncestralidadeEOrdem(BaseRepo):

    def test_aprovacao_so_em_branch_nao_mesclado(self):
        c = self.cenario
        c.configuracao()
        self.repo.git("checkout", "-q", "-b", "lateral")
        c.p9()
        self.repo.git("checkout", "-q", "main")
        self.assertBloqueia(ap.RegistroAusente, mensagem="P9")

    def test_commit_informado_nao_ancestral(self):
        c = self.cenario.completo()
        self.repo.git("checkout", "-q", "-b", "lateral", c.c_cfg)
        self.repo.escrever(DOC, documento([c.registro_p9]))
        lateral = self.repo.commit("2026-11-05")
        self.repo.git("checkout", "-q", "main")
        self.assertBloqueia(ap.RegistroInvalido, commit_p9=lateral, mensagem="não é ancestral")

    def test_commit_informado_nao_introdutor(self):
        c = self.cenario.completo()
        self.assertBloqueia(ap.RegistroInvalido, commit_p9=c.c_s4, mensagem="não foi introduzido")
        self.assertBloqueia(ap.RegistroInvalido, commit_s4_e3=c.c_p9, mensagem="não foi introduzido")

    def test_commit_informado_invalido(self):
        self.cenario.completo()
        for commit in ("abc", "f" * 40, "F" * 40):
            with self.subTest(commit=commit):
                self.assertBloqueia(ap.RegistroInvalido, commit_p9=commit)

    def test_aprovacao_no_mesmo_commit_da_configuracao(self):
        c = self.cenario
        self.repo.escrever("LEIA.md", "início\n")
        self.repo.commit("2026-11-01")
        self.repo.escrever(cfg.ARQUIVO_CONFIGURACAO, cfg.bytes_arquivo())
        self.repo.escrever(DOC, documento([c.registro_p9]))
        self.repo.commit("2026-11-05")
        self.assertBloqueia(ap.RegistroInvalido, mensagem="antes da aprovação")

    def test_aprovacao_com_anotacao_mas_sem_configuracao_previa(self):
        c = self.cenario
        self.repo.escrever(DOC, documento())
        self.repo.commit("2026-11-03")
        self.repo.escrever(cfg.ARQUIVO_CONFIGURACAO, cfg.bytes_arquivo())
        self.repo.escrever(DOC, documento([c.registro_p9]))
        self.repo.commit("2026-11-05")
        self.assertBloqueia(ap.RegistroInvalido, mensagem="não está versionado antes da aprovação")

    def test_aprovacao_sem_anotacao_previa_do_sha(self):
        c = self.cenario
        self.repo.escrever(cfg.ARQUIVO_CONFIGURACAO, cfg.bytes_arquivo())
        self.repo.escrever(DOC, documento(anotacao=None))
        self.repo.commit("2026-11-03")
        self.repo.escrever(DOC, documento([c.registro_p9]))
        self.repo.commit("2026-11-05")
        self.assertBloqueia(ap.RegistroInvalido, mensagem="Seção 19")

    def test_aprovacao_no_primeiro_commit(self):
        self.repo.escrever(cfg.ARQUIVO_CONFIGURACAO, cfg.bytes_arquivo())
        self.repo.escrever(DOC, documento([self.cenario.registro_p9]))
        self.repo.commit("2026-11-05")
        self.assertBloqueia(ap.RegistroInvalido, mensagem="primeiro do histórico")

    def test_registro_datado_depois_do_commit(self):
        c = self.cenario
        c.configuracao()
        c.registro_p9 = registro("P9", "2026-11-20", SHA_CFG)
        c.p9(data="2026-11-05")
        self.assertBloqueia(ap.RegistroInvalido, mensagem="depois do commit")

    def test_execucao_da_etapa3_nao_ancestral_do_registro(self):
        c = self.cenario
        c.configuracao()
        c.p9()
        self.repo.git("checkout", "-q", "-b", "lateral")
        self.repo.escrever("lateral.txt", "x\n")
        lateral = self.repo.commit("2026-11-06")
        self.repo.git("checkout", "-q", "main")
        c.manifesto_e3(lambda m: (m["execucao"]["git"].update(head=lateral),
                                  m["resultado"]["git_final"].update(head=lateral)))
        c.s4()
        self.assertBloqueia(ap.ManifestoS4Invalido, mensagem="não é ancestral")

    def test_execucao_da_etapa3_inexistente(self):
        c = self.cenario
        c.configuracao()
        c.p9()
        c.manifesto_e3(lambda m: (m["execucao"]["git"].update(head="e" * 40),
                                  m["resultado"]["git_final"].update(head="e" * 40)))
        c.s4()
        self.assertBloqueia(ap.ManifestoS4Invalido, mensagem="não existe")

    def test_manifesto_versionado_depois_do_registro(self):
        c = self.cenario
        c.configuracao()
        c.p9()
        self.repo.escrever("src/etapa3_sintetica.txt", "execução\n")
        c.c_exec = self.repo.commit("2026-11-06")
        c.manifesto = manifesto_s4(c.c_exec)
        c.s4(data="2026-11-07")
        self.repo.escrever(SAIDA, CONTEUDO_SAIDA)
        self.repo.escrever(MANIFESTO, rastreabilidade.serializar(c.manifesto))
        self.repo.commit("2026-11-08")
        self.assertBloqueia(ap.ManifestoS4Invalido, mensagem="não está versionado no commit do registro")


# --------------------------------------------------------------------------- #
# Registros malformados
# --------------------------------------------------------------------------- #


class TestRegistrosMalformados(BaseRepo):

    def test_linhas_malformadas_bloqueiam(self):
        casos = {
            "status desconhecido": registro("P9", "2026-11-05", SHA_CFG, status="aprovado"),
            "código desconhecido": registro("P10", "2026-11-05", SHA_CFG),
            "espaço extra": registro("P9", "2026-11-05", SHA_CFG).replace(" status", "  status"),
            "data inválida": registro("P9", "2026-02-30", SHA_CFG),
            "hash curto": registro("P9", "2026-11-05", SHA_CFG[:63]),
            "texto depois": registro("P9", "2026-11-05", SHA_CFG) + " ok",
        }
        for rotulo, linha in casos.items():
            with self.subTest(rotulo):
                with self.assertRaises(ap.RegistroInvalido):
                    ap.ler_registros(documento([linha]))
                self.assertEqual(ap.ler_registros(documento([linha]), estrito=False), ())

    def test_malformado_no_documento_bloqueia_autorizacao(self):
        c = self.cenario.completo()
        self.repo.escrever(DOC, documento([c.registro_p9, c.registro_s4, "APROVACAO-DECISAO codigo=P9"]))
        self.repo.commit("2026-11-09")
        self.assertBloqueia(ap.RegistroInvalido, mensagem="malformado")

    def test_registro_duplicado(self):
        c = self.cenario.completo()
        self.repo.escrever(DOC, documento([c.registro_p9, c.registro_p9, c.registro_s4]))
        self.repo.commit("2026-11-09")
        self.assertBloqueia(ap.RegistroInvalido, mensagem="2 registros")


# --------------------------------------------------------------------------- #
# Manifesto da S4
# --------------------------------------------------------------------------- #


def _mudar(caminho: Sequence[str], valor) -> Callable[[dict], None]:
    def aplicar(m: dict) -> None:
        alvo = m
        for chave in caminho[:-1]:
            alvo = alvo[chave]
        alvo[caminho[-1]] = valor
    return aplicar


MUTACOES_MANIFESTO: Dict[str, Callable[[dict], None]] = {
    "atividade da Etapa 2": lambda m: (m["execucao"].update(atividade=cfg_s4.Etapa.E2.value),
                                       m["execucao"]["autorizacao"].update(atividade=cfg_s4.Etapa.E2.value)),
    "exploratória": lambda m: m["execucao"].update(finalidade="exploratoria", elegivel_definitiva=False),
    "falhou": lambda m: m["resultado"].update(status="falhou", erro="falha sintética"),
    "interrompida": lambda m: m["resultado"].update(status="interrompida", erro="interrupção sintética"),
    "árvore suja": _mudar(("execucao", "git", "limpo"), False),
    "ambiente divergente": _mudar(("execucao", "ambiente", "confere_com_esperado"), False),
    "HEAD final diferente": _mudar(("resultado", "git_final", "head"), "d" * 40),
    "alterações além das saídas": _mudar(("resultado", "git_final", "alteracoes_alem_das_saidas"),
                                         [{"codigo": "M", "caminho": "x.py"}]),
    "autorização de outra atividade": _mudar(("execucao", "autorizacao", "atividade"), cfg_s4.Etapa.E2.value),
    "formato desconhecido": _mudar(("formato",), "outro-formato"),
    "saída em data/": _mudar(("resultado", "saidas", "resumo", "caminho"), "data/tabela.csv"),
}


class TestManifestoS4(BaseRepo):

    def test_manifestos_nao_definitivos_bloqueiam(self):
        for rotulo, mudar in MUTACOES_MANIFESTO.items():
            with self.subTest(rotulo):
                bruto = rastreabilidade.serializar(manifesto_s4("c" * 40, mudar))
                with self.assertRaises(ap.ManifestoS4Invalido):
                    ap.validar_manifesto_s4(bruto)

    def test_manifesto_nao_definitivo_bloqueia_autorizacao(self):
        c = self.cenario
        c.configuracao()
        c.p9()
        c.manifesto_e3(MUTACOES_MANIFESTO["falhou"])
        c.s4()
        self.assertBloqueia(ap.ManifestoS4Invalido, mensagem="concluida")

    def test_hash_interno_adulterado(self):
        m = manifesto_s4("c" * 40)
        m["resultado"]["warnings"].append("adulterado sem recalcular")
        with self.assertRaises(ap.ManifestoS4Invalido):
            ap.validar_manifesto_s4(rastreabilidade.serializar(m))

    def test_bytes_fora_da_forma_canonica(self):
        m = manifesto_s4("c" * 40)
        for bruto in (json.dumps(m).encode(), rastreabilidade.serializar(m)[:-1],
                      rastreabilidade.serializar(m).replace(b"{", b'{"x":1,"x":2,', 1)):
            with self.assertRaises(ap.ManifestoS4Invalido):
                ap.validar_manifesto_s4(bruto)

    def test_entradas_em_data_sao_validadas_mas_nao_lidas(self):
        """O manifesto real da Etapa 3 registra a tabela de data/ como entrada."""
        entrada = {"caminho": "data/modeling_table_sprint3.csv", "sha256": "b" * 64, "sha256_lf": "b" * 64,
                   "bytes": 1}
        m = manifesto_s4("c" * 40, lambda m: m["execucao"]["entradas"].update(tabela=entrada))
        self.assertEqual(ap.validar_manifesto_s4(rastreabilidade.serializar(m)), m)

    def test_crlf_tolerado(self):
        bruto = rastreabilidade.serializar(manifesto_s4("c" * 40)).replace(b"\n", b"\r\n")
        self.assertEqual(ap.validar_manifesto_s4(bruto)["execucao"]["atividade"], cfg_s4.Etapa.E3.value)

    def test_caminhos_recusados_sem_leitura(self):
        self.cenario.completo()
        for caminho in ("data/manifesto.json", "../fora.json", "/abs/manifesto.json", "reports\\m.json",
                        "C:/m.json", "reports/./m.json", ""):
            with self.subTest(caminho=caminho):
                self.assertBloqueia(ap.ManifestoS4Invalido, manifesto_s4=caminho)

    def test_espelho_da_validacao_da_s4(self):
        """O validador reimplementado concorda com o da S4 nos casos sintéticos."""
        valido = manifesto_s4("c" * 40)
        self.assertTrue(rastreabilidade.resultado_definitivo_utilizavel(valido))
        self.assertEqual(ap.validar_manifesto_s4(rastreabilidade.serializar(valido)), valido)
        self.assertEqual(ap.sha256_manifesto_s4(valido), guardas.hash_valor_decisao(valido))
        # Regras só da S6: a S4 considera utilizável o resultado definitivo de
        # qualquer etapa e aceita saídas em qualquer caminho relativo.
        so_s6 = {"atividade da Etapa 2", "saída em data/"}
        for rotulo in so_s6:
            with self.subTest(rotulo):
                self.assertTrue(rastreabilidade.resultado_definitivo_utilizavel(
                    manifesto_s4("c" * 40, MUTACOES_MANIFESTO[rotulo])))
        for rotulo, mudar in MUTACOES_MANIFESTO.items():
            if rotulo in so_s6:
                continue
            with self.subTest(rotulo):
                mutado = manifesto_s4("c" * 40, mudar)
                try:
                    aceito_s4 = rastreabilidade.resultado_definitivo_utilizavel(mutado)
                except rastreabilidade.ManifestoInvalido:
                    aceito_s4 = False
                self.assertFalse(aceito_s4)

    def test_constantes_espelhadas_da_s4(self):
        self.assertEqual(ap.FORMATO_MANIFESTO_S4, rastreabilidade.FORMATO)
        self.assertEqual(ap.VERSAO_MANIFESTO_S4, rastreabilidade.VERSAO_FORMATO)
        self.assertEqual(ap.STATUS_S4_CONCLUIDA, rastreabilidade.STATUS_CONCLUIDA)
        self.assertEqual(ap.FINALIDADE_S4_DEFINITIVA, rastreabilidade.Finalidade.DEFINITIVA.value)
        self.assertEqual(ap.ATIVIDADE_S4_E3, cfg_s4.Etapa.E3.value)
        self.assertEqual(ap.MARCADOR_REGISTRO, guardas.MARCADOR_REGISTRO)
        self.assertEqual(ap.STATUS_APROVADA, guardas.STATUS_REGISTRO_APROVADA)
        self.assertEqual(ap.CODIGOS_REGISTRO, tuple(cfg.CONFIGURACAO.execucao.registros_exigidos))
        self.assertEqual(ap.DOCUMENTO_APROVACAO, cfg.PREREGISTRO)


# --------------------------------------------------------------------------- #
# Paridade com rastreabilidade.validar_manifesto
# --------------------------------------------------------------------------- #


def _rejeitado_pela_s4(manifesto: dict) -> Optional[type]:
    """Classe da exceção da S4, ou ``None`` se a S4 aceitar como utilizável."""
    try:
        return None if rastreabilidade.resultado_definitivo_utilizavel(manifesto) else type(None)
    except rastreabilidade.ManifestoInvalido as exc:
        return type(exc)


def _parametros(valor) -> Callable[[dict], None]:
    return lambda m: m["execucao"]["parametros"].update(valor)


_RAIZ_POSIX = str(REPO_ROOT.resolve()).replace("\\", "/")
_HOME_POSIX = str(Path.home().resolve()).replace("\\", "/")

DADOS_INDIVIDUAIS: Dict[str, Callable[[dict], None]] = {
    "chave patient_id": _parametros({"patient_id": "x"}),
    "chave nodulo_id na autorização": lambda m: m["execucao"]["autorizacao"].update(nodulo_id=1),
    "chave original_nodule_idx": _parametros({"original_nodule_idx": 3}),
    "chave target_binary": _parametros({"target_binary": [0, 1]}),
    "chave Y_PRED em maiúsculas": _parametros({"Y_PRED": [0.1]}),
    "chave aninhada em lista": _parametros({"lotes": [{"ok": 1}, {"predicoes": [0.2]}]}),
    "chave em resultado.git_final": lambda m: m["resultado"]["git_final"].update(scan_id="s"),
    "identificador em valor": lambda m: m["resultado"]["warnings"].append("falha no LIDC-IDRI-0001"),
    "identificador em chave": _parametros({"LIDC-IDRI-0042": 1}),
    "caminho absoluto do Windows": lambda m: m["resultado"]["warnings"].append("lido de C:\\dados\\t.csv"),
    "caminho de rede": lambda m: m["resultado"]["warnings"].append("em \\\\servidor\\pasta\\t.csv"),
    "caminho /home": _parametros({"origem": "/home/usuario/t.csv"}),
    "caminho /Users": _parametros({"origem": "'/Users/usuario/t.csv'"}),
    "caminho da raiz do repositório": _parametros({"origem": f"ref{_RAIZ_POSIX}/reports"}),
    "caminho da pasta pessoal": _parametros({"origem": f"ref{_HOME_POSIX}/x"}),
}

SEMENTES_INVALIDAS = {
    "nome com maiúscula": {"Semente": 1},
    "nome começando por dígito": {"1cv": 1},
    "booleana": {"cv": True},
    "negativa": {"cv": -1},
    "2^32": {"cv": 2 ** 32},
    "real": {"cv": 1.0},
}

ESTRUTURAIS = {
    "concluída com erro": lambda m: m["resultado"].update(erro="erro sintético"),
    "warnings não lista": lambda m: m["resultado"].update(warnings="nenhum"),
    "instante fora do UTC": lambda m: m["tempo"].update(inicio="2026-11-06 10:00:00"),
    "duração negativa": lambda m: m["tempo"].update(duracao_s=-1.0),
    "hash_configuracao inválido": lambda m: m["execucao"].update(hash_configuracao="xyz"),
    "entrada com caminho absoluto": lambda m: m["execucao"]["entradas"].update(
        t={"caminho": "/abs/t.csv", "sha256": "b" * 64, "sha256_lf": "b" * 64, "bytes": 1}),
    "entrada com dois-pontos": lambda m: m["execucao"]["entradas"].update(
        t={"caminho": "C:t.csv", "sha256": "b" * 64, "sha256_lf": "b" * 64, "bytes": 1}),
    "saída com bytes negativos": _mudar(("resultado", "saidas", "resumo", "bytes"), -1),
    "saída com sha256 inválido": _mudar(("resultado", "saidas", "resumo", "sha256"), "Z" * 64),
    "parametros não objeto": lambda m: m["execucao"].update(parametros=[]),
    "head inicial inválido": _mudar(("execucao", "git", "head"), "abc"),
}


class TestParidadeManifestoS4(BaseRepo):

    def _ambos_recusam(self, casos: Dict[str, Callable[[dict], None]], classe_s4=rastreabilidade.ManifestoInvalido):
        for rotulo, mudar in casos.items():
            with self.subTest(rotulo):
                mutado = manifesto_s4("c" * 40, mudar)
                recusa_s4 = _rejeitado_pela_s4(mutado)
                self.assertIsNotNone(recusa_s4, "a S4 aceitou como utilizável")
                if classe_s4 is not None:
                    self.assertTrue(issubclass(recusa_s4, classe_s4), recusa_s4)
                with self.assertRaises(ap.ManifestoS4Invalido):
                    ap.validar_manifesto_s4(rastreabilidade.serializar(mutado))

    def test_chaves_proibidas_iguais_as_da_s4(self):
        self.assertEqual(ap.CHAVES_PROIBIDAS_S4, rastreabilidade.CHAVES_PROIBIDAS)

    def test_dados_individuais_e_caminhos_pessoais(self):
        self._ambos_recusam(DADOS_INDIVIDUAIS, rastreabilidade.DadosIndividuais)

    def test_sementes(self):
        casos = {r: (lambda s: lambda m: m["execucao"].update(seeds=s))(s) for r, s in SEMENTES_INVALIDAS.items()}
        self._ambos_recusam(casos)
        valido = manifesto_s4("c" * 40, lambda m: m["execucao"].update(seeds={"cv": 0, "boot": 2 ** 32 - 1}))
        self.assertTrue(rastreabilidade.resultado_definitivo_utilizavel(valido))
        self.assertEqual(ap.validar_manifesto_s4(rastreabilidade.serializar(valido)), valido)

    def test_regras_estruturais(self):
        self._ambos_recusam(ESTRUTURAIS)

    def test_dado_individual_bloqueia_autorizacao(self):
        c = self.cenario
        c.configuracao()
        c.p9()
        c.manifesto_e3(DADOS_INDIVIDUAIS["identificador em valor"])
        c.s4()
        self.assertBloqueia(ap.ManifestoS4Invalido, mensagem="identificador de paciente")


# --------------------------------------------------------------------------- #
# S4_E3_REGISTRADA: só a Etapa 3, definitiva, elegível e concluída
# --------------------------------------------------------------------------- #


class TestSomenteEtapa3Definitiva(unittest.TestCase):

    def _recusa(self, mudar: Callable[[dict], None], mensagem: str) -> None:
        with self.assertRaises(ap.ManifestoS4Invalido) as contexto:
            ap.validar_manifesto_s4(rastreabilidade.serializar(manifesto_s4("c" * 40, mudar)))
        self.assertIn(mensagem, str(contexto.exception))

    def test_todas_as_outras_atividades_da_s4(self):
        outras = [a for a in cfg_s4.DECISOES_POR_ETAPA if a != cfg_s4.Etapa.E3.value]
        self.assertGreaterEqual(len(outras), 8)
        for atividade in outras:
            with self.subTest(atividade=atividade):
                def mudar(m, atividade=atividade):
                    m["execucao"].update(atividade=atividade)
                    m["execucao"]["autorizacao"].update(atividade=atividade)
                    if atividade in cfg_s4.ATIVIDADES_SOMENTE_EXPLORATORIAS:
                        m["execucao"].update(finalidade="exploratoria", elegivel_definitiva=False)
                self._recusa(mudar, "não da Etapa 3")

    def test_finalidade_elegibilidade_e_status(self):
        casos = {
            "exploratória coerente": (lambda m: m["execucao"].update(finalidade="exploratoria",
                                                                     elegivel_definitiva=False),
                                      "não é de execução definitiva"),
            "definitiva não elegível": (lambda m: m["execucao"].update(elegivel_definitiva=False),
                                        "não é elegível"),
            "elegível como texto": (lambda m: m["execucao"].update(elegivel_definitiva="true"),
                                    "não é elegível"),
            "finalidade desconhecida": (lambda m: m["execucao"].update(finalidade="DEFINITIVA"),
                                        "não é de execução definitiva"),
            "falhou": (lambda m: m["resultado"].update(status="falhou", erro="x"), "exige 'concluida'"),
            "interrompida": (lambda m: m["resultado"].update(status="interrompida", erro="x"),
                             "exige 'concluida'"),
            "concluída com erro": (lambda m: m["resultado"].update(erro="x"), "exige 'concluida'"),
        }
        for rotulo, (mudar, mensagem) in casos.items():
            with self.subTest(rotulo):
                self._recusa(mudar, mensagem)
                self.assertIsNotNone(_rejeitado_pela_s4(manifesto_s4("c" * 40, mudar)))

    def test_somente_o_caso_completo_e_aceito(self):
        m = manifesto_s4("c" * 40)
        self.assertEqual((m["execucao"]["atividade"], m["execucao"]["finalidade"],
                          m["execucao"]["elegivel_definitiva"], m["resultado"]["status"]),
                         ("etapa3_validacao", "definitiva", True, "concluida"))
        self.assertEqual(ap.validar_manifesto_s4(rastreabilidade.serializar(m)), m)


# --------------------------------------------------------------------------- #
# Caminho do manifesto: só objetos versionados em reports/sprint4/
# --------------------------------------------------------------------------- #


CAMINHOS_RECUSADOS = (
    "", ".", "..", "../m.json", "reports/sprint4/../../m.json", "reports/sprint4/../sprint6/m.json",
    "/etc/m.json", "/reports/sprint4/m.json", "C:/m.json", "C:\\m.json", "reports\\sprint4\\m.json",
    "reports/sprint4/./m.json", "reports//sprint4/m.json", "reports/sprint4/m.json/", "./reports/sprint4/m.json",
    "data/m.json", "Data/m.json", "reports/sprint4/m.txt", "reports/sprint4", "docs/sprint6/config_benchmark_p9.json",
    "reports/sprint3/m.json", "Reports/sprint4/m.json", "HEAD:reports/sprint4/m.json", ":/m.json",
    "reports/sprint4/m.json\n", "reports/sprint4/m\x00.json", "~/reports/sprint4/m.json",
)


class TestCaminhoDoManifesto(BaseRepo):

    def test_caminhos_recusados_antes_de_consultar_o_git(self):
        with unittest.mock.patch.object(ap, "_git", side_effect=AssertionError("git consultado")) as git:
            for caminho in (*CAMINHOS_RECUSADOS, 123, Path(MANIFESTO), MANIFESTO.encode()):
                with self.subTest(caminho=caminho):
                    with self.assertRaises(ap.ManifestoS4Invalido):
                        ap.verificar_s4_e3(caminho, repo=self.raiz)
                    with self.assertRaises(ap.ManifestoS4Invalido):
                        ap.autorizar_definitivo(caminho, repo=self.raiz)
            with self.assertRaises(ap.RegistroAusente):
                ap.autorizar_definitivo(None, repo=self.raiz)
            git.assert_not_called()

    def test_copia_de_trabalho_nunca_e_lida(self):
        c = self.cenario.completo()
        self.repo.escrever(MANIFESTO, b"{ alterado na copia de trabalho, nao versionado")
        self.assertEqual(self.autorizar().s4_e3.commit, c.c_s4)
        local = "reports/sprint4/etapa3_sintetica/so_local.json"
        self.repo.escrever(local, rastreabilidade.serializar(c.manifesto))
        with self.assertRaises(ap.ManifestoS4Invalido) as contexto:
            ap.verificar_s4_e3(local, repo=self.raiz)
        self.assertIn("não está versionado", str(contexto.exception))

    def test_saidas_com_caminho_fora_do_repositorio(self):
        for caminho in ("../fora.json", "reports/../data/t.csv", "/etc/t.json", "Data/t.csv", "reports/./t.json",
                        "reports\\t.json", "C:/t.json", "", "."):
            with self.subTest(caminho=caminho):
                mutado = manifesto_s4("c" * 40, _mudar(("resultado", "saidas", "resumo", "caminho"), caminho))
                with self.assertRaises(ap.ManifestoS4Invalido):
                    ap.validar_manifesto_s4(rastreabilidade.serializar(mutado))


# --------------------------------------------------------------------------- #
# Verificação impossível e isolamento
# --------------------------------------------------------------------------- #


class TestVerificacaoIndisponivel(BaseRepo):

    def test_git_indisponivel(self):
        self.cenario.completo()
        with unittest.mock.patch.object(ap.shutil, "which", return_value=None):
            self.assertBloqueia(ap.VerificacaoIndisponivel, mensagem="git indisponível")

    def test_diretorio_que_nao_e_a_raiz(self):
        self.cenario.completo()
        with self.assertRaises(ap.VerificacaoIndisponivel):
            ap.autorizar_definitivo(MANIFESTO, repo=self.raiz / "docs")

    def test_diretorio_sem_git(self):
        with tempfile.TemporaryDirectory() as vazio, self.assertRaises(ap.VerificacaoIndisponivel):
            ap.autorizar_definitivo(MANIFESTO, repo=vazio)

    def test_head_muda_durante_a_verificacao(self):
        c = self.cenario.completo()
        original = ap._LeitorGit.head
        chamadas = []

        def head_instavel(leitor):
            chamadas.append(1)
            return original(leitor) if len(chamadas) == 1 else c.c_p9

        with unittest.mock.patch.object(ap._LeitorGit, "head", head_instavel):
            self.assertBloqueia(ap.VerificacaoIndisponivel, mensagem="HEAD mudou")


class TestContrato(unittest.TestCase):

    def test_toda_recusa_e_bloqueio_com_codigo_da_p9(self):
        codigos = cfg.CONFIGURACAO.execucao.codigos_saida
        classes = [ap.RegistroAusente, ap.RegistroInvalido, ap.RegistroRevogado, ap.ManifestoS4Invalido,
                   ap.VerificacaoIndisponivel, ap.DivergenciaAprovacao]
        for classe in classes:
            with self.subTest(classe=classe.__name__):
                self.assertTrue(issubclass(classe, ap.BloqueioAprovacao))
                esperado = "integridade" if classe is ap.DivergenciaAprovacao else "bloqueio_aprovacao"
                self.assertEqual(classe.chave_saida, esperado)
        self.assertEqual(codigos[ap.BloqueioAprovacao.chave_saida], 2)
        self.assertEqual(codigos[ap.DivergenciaAprovacao.chave_saida], 4)

    def test_configuracao_invalida_bloqueia(self):
        with self.assertRaises(ap.DivergenciaAprovacao):
            ap.autorizar_definitivo(MANIFESTO, repo=REPO_ROOT, configuracao=object())  # type: ignore[arg-type]

    def test_modulo_so_le_objetos_do_git(self):
        """Nenhuma leitura direta de arquivo: só ``git cat-file`` e afins."""
        arvore = ast.parse((REPO_ROOT / "src/benchmark_s6/aprovacoes.py").read_text(encoding="utf-8"))
        proibidas = {"open", "read_bytes", "read_text", "write_bytes", "write_text", "unlink", "mkdir",
                     "rename", "replace_file", "remove"}
        chamadas = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
                    for n in ast.walk(arvore) if isinstance(n, ast.Call)}
        self.assertEqual(chamadas & proibidas, set())
        argumentos_git = [ast.literal_eval(a) for n in ast.walk(arvore) if isinstance(n, ast.Call)
                          and getattr(n.func, "id", None) == "_git" for a in n.args[1:2]]
        self.assertTrue(set(argumentos_git) <= {"rev-parse", "merge-base", "rev-list", "log", "cat-file"},
                        argumentos_git)

    def test_simbolos_da_s4_inalterados(self):
        """Usar outros nomes da S4 mudaria a configuração canônica já publicada."""
        self.assertEqual(cfg.sha256_configuracao(),
                         "a0d096a5415244f17f40360a2a5a8901bc8af1da5cf8687b891f920e1ee122b3")


if __name__ == "__main__":
    unittest.main()
