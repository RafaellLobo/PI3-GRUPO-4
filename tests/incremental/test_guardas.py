"""Testes do Incremento 2A — guardas científicas (``src/incremental/guardas.py``).

Não leem dados de pacientes nem treinam modelos. Os cenários de aprovação usam
repositórios git TEMPORÁRIOS, com documentos e decisões fictícios criados só
para o teste. O repositório real é apenas consultado (HEAD e protocolo
versionado), sem escrita.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/incremental -v
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.incremental import config as cfg  # noqa: E402
from src.incremental import guardas as g  # noqa: E402

HASH_CONFIG_INCREMENTO_1 = "810c70f751a6252acc69b52b8ca7cdd380988f85d73831a2ab3122ead0c4b8b2"
PROTOCOLO = cfg.PROTOCOLO
TEXTO_BASE = "# Protocolo fictício de teste\n\nTexto sem registros.\n"
#: Conteúdos fictícios aprovados nos repositórios temporários.
VALOR_P2 = cfg.SPLINE_PROPOSTO
VALOR_P3 = cfg.PAINEL_DECLARADO
OUTRO_PAINEL = cfg.PAINEL_REPRODUZIDO


# --------------------------------------------------------------------------- #
# Infraestrutura de teste
# --------------------------------------------------------------------------- #


def _ambiente_git(data: str = None) -> dict:
    ambiente = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    if data:
        ambiente["GIT_AUTHOR_DATE"] = ambiente["GIT_COMMITTER_DATE"] = f"{data}T12:00:00+00:00"
    return ambiente


def _remover(caminho: Path) -> None:
    def tornar_gravavel(funcao, alvo, _info):
        os.chmod(alvo, stat.S_IWRITE)
        funcao(alvo)

    shutil.rmtree(caminho, onerror=tornar_gravavel)


class RepoTemporario:
    """Repositório git descartável, fora do repositório do projeto."""

    def __init__(self) -> None:
        self.raiz = Path(tempfile.mkdtemp(prefix="guardas_"))
        self.git("init", "-q", "-b", "main")
        for chave, valor in (("user.name", "teste"), ("user.email", "teste@example.invalid"),
                             ("core.autocrlf", "false"), ("commit.gpgsign", "false")):
            self.git("config", chave, valor)

    def git(self, *args: str, data: str = None) -> str:
        return subprocess.run(["git", "-C", str(self.raiz), *args], check=True,
                              capture_output=True, text=True, env=_ambiente_git(data)).stdout.strip()

    def commit(self, arquivos: dict, data: str, mensagem: str = "teste", eol: str = "\n") -> str:
        for caminho, texto in arquivos.items():
            destino = self.raiz / caminho
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(texto.replace("\n", eol).encode("utf-8"))
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", mensagem, data=data)
        return self.git("rev-parse", "HEAD")

    def remover(self) -> None:
        _remover(self.raiz)


def registro(codigo: str, valor=None, data: str = "2026-10-02") -> str:
    return g.formatar_registro(codigo, data, valor)


def protocolo(*linhas: str) -> dict:
    return {PROTOCOLO: TEXTO_BASE + "".join(f"\n{linha}\n" for linha in linhas)}


def aprovar(decisoes: dict = None, **aprovadas) -> dict:
    """Cópia das decisões com as indicadas aprovadas: codigo=(commit, valor)."""
    novas = dict(decisoes if decisoes is not None else cfg.DECISOES)
    for codigo, (commit, valor) in aprovadas.items():
        novas[codigo] = dataclasses.replace(
            cfg.DECISOES[codigo], status=cfg.StatusDecisao.APROVADA,
            commit_revisao_protocolo=commit, valor=valor)
    return novas


def adulterar(decisoes: dict, chave: str, **atributos) -> dict:
    """Altera uma Decisao congelada depois da construção, como faria código descuidado."""
    novas = dict(decisoes)
    alvo = dataclasses.replace(novas[chave])
    for nome, valor in atributos.items():
        object.__setattr__(alvo, nome, valor)
    novas[chave] = alvo
    return novas


def git_proibido(*_args, **_kwargs):
    raise AssertionError("a guarda não deveria consultar o git")


class ComRepoCompartilhado(unittest.TestCase):
    """Histórico (datas fictícias):

    c0 base sem registros → c1 registra P1 e P2 → c2 registra P3 →
    c3 registra P4 fora do protocolo → c4 mudança neutra (HEAD de main).
    O ramo ``lado`` sai de c0 e registra P1 em s1, que não é ancestral do HEAD.
    """

    @classmethod
    def setUpClass(cls):
        repo = cls.repo = RepoTemporario()
        r_p1, r_p2 = registro("P1"), registro("P2", VALOR_P2)
        r_p3 = registro("P3", VALOR_P3, "2026-10-03")
        cls.c0 = repo.commit(protocolo(), "2026-10-01")
        cls.c1 = repo.commit(protocolo(r_p1, r_p2), "2026-10-02")
        cls.c2 = repo.commit(protocolo(r_p1, r_p2, r_p3), "2026-10-03")
        cls.c3 = repo.commit({"docs/outro.md": registro("P4", data="2026-10-04") + "\n"}, "2026-10-04")
        cls.c4 = repo.commit({"LEIAME.md": "mudança neutra\n"}, "2026-10-05")
        repo.git("checkout", "-q", "-b", "lado", cls.c0)
        cls.s1 = repo.commit(protocolo(r_p1), "2026-10-02")
        repo.git("checkout", "-q", "main")
        cls.blob = repo.git("rev-parse", f"{cls.c1}:{PROTOCOLO}")

    @classmethod
    def tearDownClass(cls):
        cls.repo.remover()

    def decisoes_e1(self, **extras) -> dict:
        return aprovar(**{"P1": (self.c1, None), "P2": (self.c1, VALOR_P2), **extras})

    def decisoes_e3(self) -> dict:
        return self.decisoes_e1(P3=(self.c2, VALOR_P3))


# --------------------------------------------------------------------------- #
# Estado real do projeto
# --------------------------------------------------------------------------- #


class TestEstadoReal(unittest.TestCase):
    def test_todas_as_decisoes_continuam_pendentes(self):
        self.assertEqual(set(cfg.DECISOES), {"P1", "P2", "P3", "P4", "P5", "REVISAO_PROTOCOLO"})
        for codigo, decisao in cfg.DECISOES.items():
            with self.subTest(decisao=codigo):
                self.assertFalse(decisao.aprovada)
                self.assertIsNone(decisao.commit_revisao_protocolo)
                self.assertIsNone(decisao.valor)

    def test_configuracao_inalterada(self):
        self.assertEqual(cfg.hash_configuracao(), HASH_CONFIG_INCREMENTO_1)

    def test_atividades_com_decisoes_bloqueadas_no_repositorio_real(self):
        for atividade, exigidas in cfg.DECISOES_POR_ETAPA.items():
            if not exigidas:
                continue
            with self.subTest(atividade=atividade):
                with self.assertRaises(cfg.BloqueioDecisao):
                    g.exigir_autorizacao(atividade)

    def test_commit_existente_e_ancestral_sem_registro_nao_autoriza(self):
        # O próprio HEAD do projeto existe e é ancestral de si mesmo, mas o
        # protocolo versionado não tem registro de aprovação: falha conservadora.
        head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, env=_ambiente_git()).stdout.strip()
        decisoes = aprovar(P1=(head, None), P2=(head, VALOR_P2))
        with self.assertRaisesRegex(g.AprovacaoNaoVerificada, "não contém registro explícito"):
            g.exigir_autorizacao(cfg.Etapa.E1, decisoes)
        self.assertEqual(cfg.hash_configuracao(), HASH_CONFIG_INCREMENTO_1)


# --------------------------------------------------------------------------- #
# Atividades sem decisões e decisões pendentes (sem git)
# --------------------------------------------------------------------------- #


class TestSemGit(unittest.TestCase):
    def test_estruturais_e_etapa0_liberadas_sem_consultar_git(self):
        for atividade in (cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS, cfg.Etapa.E0):
            with self.subTest(atividade=atividade), mock.patch.object(g, "_git", git_proibido):
                autorizacao = g.exigir_autorizacao(atividade)
                self.assertEqual(autorizacao.decisoes_exigidas, ())
                self.assertEqual(autorizacao.verificacoes, ())
                self.assertIsNone(autorizacao.head)
                self.assertEqual(autorizacao.hash_configuracao, HASH_CONFIG_INCREMENTO_1)

    def test_estruturais_liberadas_fora_de_repositorio(self):
        with tempfile.TemporaryDirectory() as vazio:
            g.exigir_autorizacao(cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS, repo=vazio)

    def test_pendentes_bloqueiam_antes_do_git_e_listam_todas(self):
        for atividade, exigidas in cfg.DECISOES_POR_ETAPA.items():
            if not exigidas:
                continue
            with self.subTest(atividade=atividade), mock.patch.object(g, "_git", git_proibido):
                with self.assertRaises(cfg.BloqueioDecisao) as ctx:
                    g.exigir_autorizacao(atividade)
                self.assertNotIsInstance(ctx.exception, g.AprovacaoNaoVerificada)
                for codigo in exigidas:
                    self.assertIn(codigo, str(ctx.exception))

    def test_aprovacao_parcial_bloqueia_pela_pendente(self):
        decisoes = aprovar(P1=("1" * 40, None))
        with mock.patch.object(g, "_git", git_proibido):
            with self.assertRaisesRegex(cfg.BloqueioDecisao, r"P2") as ctx:
                g.exigir_autorizacao(cfg.Etapa.E1, decisoes)
        self.assertNotIn("P1 (", str(ctx.exception))

    def test_cv_lista_fixa_exige_p3(self):
        decisoes = aprovar(P1=("1" * 40, None), P2=("1" * 40, VALOR_P2))
        with mock.patch.object(g, "_git", git_proibido):
            with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
                g.exigir_autorizacao(cfg.ATIVIDADE_CV_LISTA_FIXA, decisoes)

    def test_atividade_desconhecida(self):
        for atividade in ("etapa9", None, 3, "E1"):
            with self.subTest(atividade=atividade):
                with self.assertRaises(g.AtividadeDesconhecida):
                    g.exigir_autorizacao(atividade)

    def test_variantes_de_nome_de_atividade_sao_desconhecidas(self):
        for atividade in (" etapa1_piloto", "etapa1_piloto ", "ETAPA1_PILOTO", cfg.Etapa.E1.name,
                          "etapa0", "verificacoes estruturais", "", b"etapa1_piloto",
                          ("etapa1_piloto",), cfg.Etapa):
            with self.subTest(atividade=atividade), mock.patch.object(g, "_git", git_proibido):
                with self.assertRaises(g.AtividadeDesconhecida):
                    g.exigir_autorizacao(atividade)
                with self.assertRaises(g.AtividadeDesconhecida):
                    g.decisoes_exigidas(atividade)

    def test_somente_estruturais_e_etapa0_dispensam_decisoes(self):
        sem_decisoes = {a for a, exigidas in cfg.DECISOES_POR_ETAPA.items() if not exigidas}
        self.assertEqual(sem_decisoes, {cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS, cfg.Etapa.E0.value})
        self.assertEqual(set(cfg.DECISOES_POR_ETAPA),
                         {cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS, cfg.ATIVIDADE_CV_LISTA_FIXA,
                          *(e.value for e in cfg.Etapa)})

    def test_modelos_incorretos_nao_dispensam_p3(self):
        # Regressão: "M2" como string era iterado como "M" e "2", e P3 sumia.
        estrutural = cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS
        with mock.patch.object(g, "_git", git_proibido):
            for modelos in ("M2", b"M2", "M2v"):
                with self.subTest(modelos=modelos):
                    with self.assertRaises(TypeError):
                        g.exigir_autorizacao(estrutural, modelos=modelos)
            for modelos in (("M5",), ("m2",), ("M2 ",), ("M2", None), [cfg.modelo_fixo("M0")]):
                with self.subTest(modelos=modelos):
                    with self.assertRaises(g.ModeloDesconhecido):
                        g.exigir_autorizacao(estrutural, modelos=modelos)
            with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
                g.exigir_autorizacao(estrutural, modelos=iter(["M0", "M3"]))
            self.assertEqual(g.exigir_autorizacao(estrutural, modelos=["M0", "M1", "M4v"]).decisoes_exigidas, ())

    def test_decisoes_exigidas_seguem_config_e_modelos_com_painel_somam_p3(self):
        for atividade, exigidas in cfg.DECISOES_POR_ETAPA.items():
            self.assertEqual(g.decisoes_exigidas(atividade), exigidas)
        self.assertEqual(g.decisoes_exigidas(cfg.Etapa.E1, ("M2v",)), ("P1", "P2"))
        for nome in cfg.MODELOS_COM_PAINEL:
            self.assertEqual(g.decisoes_exigidas(cfg.Etapa.E1, (nome,)), ("P1", "P2", "P3"))
            self.assertEqual(g.decisoes_exigidas(cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS, (nome,)), ("P3",))


class TestConsistencia(unittest.TestCase):
    """Decisões inconsistentes são recusadas antes de qualquer consulta ao git."""

    def assertInconsistente(self, decisoes, padrao=None, atividade=cfg.Etapa.E1):
        with mock.patch.object(g, "_git", git_proibido):
            with self.assertRaises(g.DecisaoInconsistente) as ctx:
                g.exigir_autorizacao(atividade, decisoes)
        if padrao:
            self.assertRegex(str(ctx.exception), padrao)

    def test_hashes_invalidos(self):
        base = aprovar(P1=("1" * 40, None), P2=("1" * 40, VALOR_P2))
        for invalido in ("1" * 39, "1" * 41, "A" * 40, "g" * 40, "HEAD", "-" + "1" * 39, 123, None):
            with self.subTest(commit=invalido):
                self.assertInconsistente(adulterar(base, "P1", commit_revisao_protocolo=invalido),
                                         "hash de commit inválido")

    def test_pendente_com_commit_ou_valor(self):
        self.assertInconsistente(adulterar(dict(cfg.DECISOES), "P4", commit_revisao_protocolo="1" * 40))
        self.assertInconsistente(adulterar(dict(cfg.DECISOES), "P5", valor=("x",)))

    def test_mapa_com_decisao_faltando_ou_sobrando(self):
        faltando = dict(cfg.DECISOES)
        del faltando["P4"]
        self.assertInconsistente(faltando, "faltam")
        sobrando = {**cfg.DECISOES, "P6": cfg.DECISOES["P5"]}
        self.assertInconsistente(sobrando, "sobram")
        self.assertInconsistente(list(cfg.DECISOES.values()))

    def test_definicao_divergente(self):
        for atributo, valor in (("codigo", "P9"), ("descricao", "outra"), ("prazo", "nunca"),
                                ("etapa_limite", cfg.Etapa.E5), ("status", "aprovada")):
            with self.subTest(atributo=atributo):
                self.assertInconsistente(adulterar(dict(cfg.DECISOES), "P1", **{atributo: valor}))
        self.assertInconsistente({**cfg.DECISOES, "P1": object()})

    def test_conteudo_aprovado_invalido(self):
        # P2 exige EspecSpline; P3 exige tupla de 8 nomes válidos.
        commit = "1" * 40
        self.assertInconsistente(aprovar(P1=(commit, None), P2=(commit, None)), "P2")
        base = aprovar(P1=(commit, None), P2=(commit, VALOR_P2))
        for painel in (list(VALOR_P3), VALOR_P3[:7], VALOR_P3[:7] + ("original_nodule_idx",)):
            with self.subTest(painel=painel):
                self.assertInconsistente(aprovar(base, P3=(commit, painel)), "P3",
                                         atividade=cfg.ATIVIDADE_CV_LISTA_FIXA)

    def test_valor_nao_serializavel(self):
        commit = "1" * 40
        self.assertInconsistente(aprovar(P1=(commit, object()), P2=(commit, VALOR_P2)))


# --------------------------------------------------------------------------- #
# Evidência no repositório
# --------------------------------------------------------------------------- #


class TestAutorizado(ComRepoCompartilhado):
    def test_etapa1_autorizada_com_registros_no_commit_de_aprovacao(self):
        autorizacao = g.exigir_autorizacao(cfg.Etapa.E1, self.decisoes_e1(), self.repo.raiz)
        self.assertEqual(autorizacao.atividade, cfg.Etapa.E1.value)
        self.assertEqual(autorizacao.decisoes_exigidas, ("P1", "P2"))
        self.assertEqual(autorizacao.head, self.c4)
        self.assertEqual([(v.codigo, v.commit, v.documento, v.data) for v in autorizacao.verificacoes],
                         [("P1", self.c1, PROTOCOLO, "2026-10-02"), ("P2", self.c1, PROTOCOLO, "2026-10-02")])
        self.assertEqual(autorizacao.verificacoes[1].valor_sha256, g.hash_valor_decisao(VALOR_P2))
        self.assertEqual(autorizacao.hash_configuracao, cfg.hash_configuracao(self.decisoes_e1()))
        self.assertEqual(autorizacao.como_dict()["verificacoes"][0]["commit"], self.c1)

    def test_etapa2_e_cv_lista_fixa(self):
        g.exigir_autorizacao(cfg.Etapa.E2, self.decisoes_e1(), self.repo.raiz)
        with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
            g.exigir_autorizacao(cfg.ATIVIDADE_CV_LISTA_FIXA, self.decisoes_e1(), self.repo.raiz)
        autorizacao = g.exigir_autorizacao(cfg.ATIVIDADE_CV_LISTA_FIXA, self.decisoes_e3(), self.repo.raiz)
        self.assertEqual(autorizacao.decisoes_exigidas, ("P1", "P2", "P3"))

    def test_etapa3_autorizada_e_etapa4_bloqueada(self):
        g.exigir_autorizacao(cfg.Etapa.E3, self.decisoes_e3(), self.repo.raiz)
        with self.assertRaisesRegex(cfg.BloqueioDecisao, "P5"):
            g.exigir_autorizacao(cfg.Etapa.E4, self.decisoes_e3(), self.repo.raiz)

    def test_registro_no_head_tambem_e_aceito(self):
        repo = RepoTemporario()
        try:
            repo.commit(protocolo(), "2026-10-01")
            c1 = repo.commit(protocolo(registro("P1"), registro("P2", VALOR_P2)), "2026-10-02")
            decisoes = aprovar(P1=(c1, None), P2=(c1, VALOR_P2))
            self.assertEqual(g.exigir_autorizacao(cfg.Etapa.E1, decisoes, repo.raiz).head, c1)
        finally:
            repo.remover()


class TestBloqueadoPorEvidencia(ComRepoCompartilhado):
    def assertNaoVerificada(self, decisoes, padrao, atividade=cfg.Etapa.E1):
        with self.assertRaisesRegex(g.AprovacaoNaoVerificada, padrao):
            g.exigir_autorizacao(atividade, decisoes, self.repo.raiz)

    def test_commit_inexistente(self):
        inexistente = "1234567890abcdef" * 2 + "12345678"
        self.assertNaoVerificada(self.decisoes_e1(P1=(inexistente, None)), "inexistente")

    def test_objeto_que_nao_e_commit(self):
        self.assertNaoVerificada(self.decisoes_e1(P1=(self.blob, None)), "blob, não commit")

    def test_commit_nao_ancestral_do_head(self):
        # s1 contém o registro de P1, mas está fora do histórico de main.
        self.assertNaoVerificada(self.decisoes_e1(P1=(self.s1, None)), "não é ancestral")

    def test_commit_existente_sem_registro(self):
        self.assertNaoVerificada(self.decisoes_e1(P1=(self.c0, None)), "não contém registro explícito")

    def test_commit_que_nao_introduziu_o_registro(self):
        # c2 e c4 contêm o registro de P1, mas ele foi introduzido em c1.
        for commit in (self.c2, self.c4):
            with self.subTest(commit=commit):
                self.assertNaoVerificada(self.decisoes_e1(P1=(commit, None)), "já existia")

    def test_conteudo_diferente_do_registrado(self):
        outro_spline = dataclasses.replace(VALOR_P2, n_knots=5)
        self.assertNaoVerificada(self.decisoes_e1(P2=(self.c1, outro_spline)), "não corresponde")
        self.assertNaoVerificada(self.decisoes_e1(P1=(self.c1, ("valor",))), "não corresponde")
        self.assertNaoVerificada(aprovar(self.decisoes_e1(), P3=(self.c2, OUTRO_PAINEL)),
                                 "não corresponde", atividade=cfg.Etapa.E3)

    def test_registro_fora_dos_documentos_de_aprovacao(self):
        decisoes = aprovar(P4=(self.c3, None))
        with self.assertRaisesRegex(g.AprovacaoNaoVerificada, "não contém registro explícito"):
            g.verificar_aprovacao("P4", decisoes, self.repo.raiz)

    def test_repositorio_inexistente_ou_git_indisponivel(self):
        with tempfile.TemporaryDirectory() as vazio:
            self.assertNaoVerificadaEm(Path(vazio) / "nao_existe", "HEAD")

    def test_subdiretorio_nao_e_raiz_do_repositorio(self):
        # Um diretório dentro de outro repositório não herda as aprovações dele.
        subdiretorio = self.repo.raiz / "docs"
        self.assertNaoVerificadaEm(subdiretorio, "não é a raiz")

    def test_git_indisponivel(self):
        with mock.patch.object(g.shutil, "which", return_value=None):
            self.assertNaoVerificadaEm(self.repo.raiz, "git indisponível")

    def assertNaoVerificadaEm(self, repo, padrao):
        with self.assertRaisesRegex(g.AprovacaoNaoVerificada, padrao):
            g.exigir_autorizacao(cfg.Etapa.E1, self.decisoes_e1(), repo)

    def test_head_alterado_durante_a_verificacao(self):
        with mock.patch.object(g, "_head", side_effect=[self.c4, self.c3]):
            self.assertNaoVerificada(self.decisoes_e1(), "HEAD mudou")

    def test_verificar_aprovacao_publica_usa_o_head_atual(self):
        verificacao = g.verificar_aprovacao("P3", self.decisoes_e3(), self.repo.raiz)
        self.assertEqual((verificacao.codigo, verificacao.commit), ("P3", self.c2))
        # Não há parâmetro para informar outro HEAD (atalho para a ancestralidade).
        self.assertEqual(list(inspect.signature(g.verificar_aprovacao).parameters),
                         ["codigo", "decisoes", "repo"])

    def test_verificar_aprovacao_recusa_pendente_e_desconhecida(self):
        with self.assertRaises(cfg.BloqueioDecisao):
            g.verificar_aprovacao("P1", cfg.DECISOES, self.repo.raiz)
        with self.assertRaises(g.DecisaoInconsistente):
            g.verificar_aprovacao("P9", self.decisoes_e1(), self.repo.raiz)


class TestRegistrosInconsistentes(unittest.TestCase):
    """Cada cenário usa um repositório próprio: c0 base → c1 aprovação → c2 HEAD."""

    def cenario(self, linhas_c1, linhas_c2=None, data_c1="2026-10-02", base=()):
        repo = RepoTemporario()
        self.addCleanup(repo.remover)
        repo.commit(protocolo(*base), "2026-10-01")
        c1 = repo.commit(protocolo(*linhas_c1), data_c1)
        repo.commit(protocolo(*(linhas_c1 if linhas_c2 is None else linhas_c2)), "2026-10-03")
        return repo, aprovar(P1=(c1, None), P2=(c1, VALOR_P2))

    def assertNaoVerificada(self, padrao, *args, **kwargs):
        repo, decisoes = self.cenario(*args, **kwargs)
        with self.assertRaisesRegex(g.AprovacaoNaoVerificada, padrao):
            g.exigir_autorizacao(cfg.Etapa.E1, decisoes, repo.raiz)

    def test_registro_duplicado_no_commit(self):
        self.assertNaoVerificada("2 registros", [registro("P1"), registro("P1"), registro("P2", VALOR_P2)])

    def test_registros_conflitantes_no_commit(self):
        self.assertNaoVerificada("2 registros", [registro("P1"), registro("P1", ("outro",)),
                                                 registro("P2", VALOR_P2)])

    def test_registro_malformado(self):
        p2 = registro("P2", VALOR_P2)
        for malformado in ("APROVACAO-DECISAO codigo=P1 status=aprovada",
                           registro("P1").replace("data=2026-10-02", "data=2026-13-40"),
                           registro("P1").upper(),
                           registro("P1") + " extra"):
            with self.subTest(linha=malformado):
                self.assertNaoVerificada("malformado|data inválida", [malformado, p2])

    def test_status_diferente_de_aprovada(self):
        linha = registro("P1").replace("status=aprovada", "status=proposta")
        self.assertNaoVerificada("status 'proposta'", [linha, registro("P2", VALOR_P2)])

    def test_registro_datado_depois_do_commit(self):
        self.assertNaoVerificada("datado depois", [registro("P1", data="2026-10-09"),
                                                   registro("P2", VALOR_P2)])

    def test_registro_revogado_ate_o_head(self):
        self.assertNaoVerificada("revogado", [registro("P1"), registro("P2", VALOR_P2)],
                                 [registro("P2", VALOR_P2)])

    def test_registro_alterado_ate_o_head(self):
        alterado = registro("P1").replace("status=aprovada", "status=revogada")
        self.assertNaoVerificada("revogado", [registro("P1"), registro("P2", VALOR_P2)],
                                 [alterado, registro("P2", VALOR_P2)])

    def test_registro_duplicado_ate_o_head(self):
        self.assertNaoVerificada("revogado", [registro("P1"), registro("P2", VALOR_P2)],
                                 [registro("P1"), registro("P1"), registro("P2", VALOR_P2)])

    def test_malformado_no_pai_nao_impede_a_correcao(self):
        repo, decisoes = self.cenario([registro("P1"), registro("P2", VALOR_P2)],
                                      base=["APROVACAO-DECISAO codigo=P1 rascunho"])
        g.exigir_autorizacao(cfg.Etapa.E1, decisoes, repo.raiz)

    def test_marcador_no_meio_da_linha_e_ignorado(self):
        repo, decisoes = self.cenario([registro("P1"), registro("P2", VALOR_P2),
                                       "O formato `APROVACAO-DECISAO codigo=…` é descrito aqui."])
        g.exigir_autorizacao(cfg.Etapa.E1, decisoes, repo.raiz)

    def test_documento_com_crlf(self):
        repo = RepoTemporario()
        self.addCleanup(repo.remover)
        repo.commit(protocolo(), "2026-10-01", eol="\r\n")
        c1 = repo.commit(protocolo(registro("P1"), registro("P2", VALOR_P2)), "2026-10-02", eol="\r\n")
        g.exigir_autorizacao(cfg.Etapa.E1, aprovar(P1=(c1, None), P2=(c1, VALOR_P2)), repo.raiz)


class TestExemplosEmMarkdown(unittest.TestCase):
    """Regressão: marcadores em exemplos Markdown não são aprovações reais.

    Cada documento tem registros VÁLIDOS para P1 e P2, mas só em contextos de
    exemplo; se algum fosse lido como aprovação, a Etapa 1 seria autorizada.
    """

    def documento(self, corpo: str) -> dict:
        return {PROTOCOLO: TEXTO_BASE + "\n" + corpo + "\n"}

    def autorizar(self, corpo: str, eol: str = "\n"):
        repo = RepoTemporario()
        self.addCleanup(repo.remover)
        repo.commit(protocolo(), "2026-10-01", eol=eol)
        c1 = repo.commit(self.documento(corpo), "2026-10-02", eol=eol)
        return g.exigir_autorizacao(cfg.Etapa.E1, aprovar(P1=(c1, None), P2=(c1, VALOR_P2)), repo.raiz)

    def assertNaoAutoriza(self, corpo: str, eol: str = "\n"):
        with self.assertRaisesRegex(g.AprovacaoNaoVerificada, "não contém registro explícito"):
            self.autorizar(corpo, eol)

    @property
    def validos(self) -> str:
        return f"{registro('P1')}\n{registro('P2', VALOR_P2)}"

    def test_bloco_cercado_por_crases(self):
        self.assertNaoAutoriza(f"```\n{self.validos}\n```")
        self.assertNaoAutoriza(f"```text\n{self.validos}\n```")
        self.assertNaoAutoriza(f"```\n{self.validos}\n```", eol="\r\n")

    def test_bloco_cercado_por_tils(self):
        self.assertNaoAutoriza(f"~~~\n{self.validos}\n~~~")

    def test_bloco_cercado_recuado(self):
        self.assertNaoAutoriza(f"   ```\n{self.validos}\n   ```")
        # Cerca dentro de item de lista: a guarda abre o bloco com qualquer
        # recuo — leitura conservadora, que só pode ignorar registros.
        self.assertNaoAutoriza(f"- item\n\n    ```\n{self.validos}\n    ```")

    def test_cerca_interna_menor_nao_fecha_o_bloco(self):
        self.assertNaoAutoriza(f"````\n```\n{self.validos}\n```\n````")
        self.assertNaoAutoriza(f"~~~\n```\n{self.validos}\n~~~")

    def test_cerca_de_fechamento_com_texto_ou_recuo_de_4_nao_fecha(self):
        self.assertNaoAutoriza(f"```\n``` texto\n{self.validos}\n```")
        self.assertNaoAutoriza(f"```\n    ```\n{self.validos}\n```")
        self.assertNaoAutoriza(f"```\n\t```\n{self.validos}\n```")

    def test_bloco_cercado_sem_fechamento_vai_ate_o_fim(self):
        self.assertNaoAutoriza(f"```\n{self.validos}")

    def test_bloco_indentado(self):
        recuado = "\n".join("    " + linha for linha in self.validos.split("\n"))
        self.assertNaoAutoriza(f"Exemplo:\n\n{recuado}")
        self.assertNaoAutoriza("\n".join("\t" + linha for linha in self.validos.split("\n")))

    def test_recuo_de_um_a_tres_espacos_nao_e_registro(self):
        for recuo in (" ", "  ", "   "):
            with self.subTest(recuo=len(recuo)):
                self.assertNaoAutoriza("\n".join(recuo + linha for linha in self.validos.split("\n")))

    def test_comentario_html(self):
        self.assertNaoAutoriza(f"<!--\n{self.validos}\n-->")
        self.assertNaoAutoriza(f"Texto <!-- início\n{self.validos}\nfim -->")
        self.assertNaoAutoriza(f"<!-- a -->\n<!-- b\nfecha --> e reabre <!--\n{self.validos}\n-->")
        self.assertNaoAutoriza(f"<!-- a --> texto <!-- b\n{self.validos}\n-->")

    def test_citacao_lista_e_tabela(self):
        for prefixo in ("> ", "- ", "1. ", "| "):
            with self.subTest(prefixo=prefixo):
                self.assertNaoAutoriza("\n".join(prefixo + linha for linha in self.validos.split("\n")))

    def test_registro_real_depois_de_exemplo_fechado_e_aceito(self):
        exemplo = ("APROVACAO-DECISAO codigo=P3 status=aprovada data=AAAA-MM-DD "
                   "valor_sha256=<64 hex>")
        corpo = (f"Formato:\n\n```\n{exemplo}\n{self.validos}\n```\n\n<!-- {exemplo} -->\n\n"
                 f"    {exemplo}\n\n{self.validos}")
        autorizacao = self.autorizar(corpo)
        self.assertEqual([v.codigo for v in autorizacao.verificacoes], ["P1", "P2"])

    def test_exemplo_em_bloco_nao_conta_como_duplicata(self):
        autorizacao = self.autorizar(f"```\n{self.validos}\n```\n\n{self.validos}")
        self.assertEqual(len(autorizacao.verificacoes), 2)

    def test_malformado_na_coluna_zero_fora_de_exemplo_continua_bloqueando(self):
        with self.assertRaisesRegex(g.AprovacaoNaoVerificada, "malformado"):
            self.autorizar(f"{self.validos}\nAPROVACAO-DECISAO codigo=P5 rascunho")


class TestObjetosSubstituidos(unittest.TestCase):
    def test_git_replace_nao_forja_registro(self):
        repo = RepoTemporario()
        self.addCleanup(repo.remover)
        repo.commit(protocolo(), "2026-10-01")
        c1 = repo.commit({PROTOCOLO: TEXTO_BASE + "\nRevisão sem registros.\n"}, "2026-10-02")
        forjado = repo.raiz / "forjado.md"
        forjado.write_bytes(protocolo(registro("P1"), registro("P2", VALOR_P2))[PROTOCOLO].encode("utf-8"))
        blob_forjado = repo.git("hash-object", "-w", str(forjado))
        forjado.unlink()
        repo.git("replace", repo.git("rev-parse", f"{c1}:{PROTOCOLO}"), blob_forjado)
        self.assertIn("APROVACAO-DECISAO", repo.git("cat-file", "blob", f"{c1}:{PROTOCOLO}"))
        with self.assertRaisesRegex(g.AprovacaoNaoVerificada, "não contém registro explícito"):
            g.exigir_autorizacao(cfg.Etapa.E1, aprovar(P1=(c1, None), P2=(c1, VALOR_P2)), repo.raiz)


class TestRevalidacao(unittest.TestCase):
    def test_cada_chamada_revalida_no_momento_da_execucao(self):
        repo = RepoTemporario()
        self.addCleanup(repo.remover)
        repo.commit(protocolo(), "2026-10-01")
        c1 = repo.commit(protocolo(registro("P1"), registro("P2", VALOR_P2)), "2026-10-02")
        decisoes = aprovar(P1=(c1, None), P2=(c1, VALOR_P2))
        g.exigir_autorizacao(cfg.Etapa.E1, decisoes, repo.raiz)
        repo.commit(protocolo(registro("P2", VALOR_P2)), "2026-10-03")  # revoga P1
        with self.assertRaisesRegex(g.AprovacaoNaoVerificada, "P1: .*revogado"):
            g.exigir_autorizacao(cfg.Etapa.E1, decisoes, repo.raiz)


# --------------------------------------------------------------------------- #
# Modelos — revalidação independente da marca de EspecModelo
# --------------------------------------------------------------------------- #


class TestAutorizarModelo(ComRepoCompartilhado):
    def test_m0_m1_e_variaveis_sem_decisoes_em_atividade_estrutural(self):
        especs = [cfg.modelo_fixo("M0"), cfg.modelo_fixo("M1"), cfg.modelo_variavel("M2v", ()),
                  cfg.modelo_variavel("M3v", (cfg.DIAMETRO, "original_glcm_Imc1")),
                  cfg.modelo_variavel("M4v", VALOR_P3)]
        for espec in especs:
            with self.subTest(modelo=espec.nome), mock.patch.object(g, "_git", git_proibido):
                g.autorizar_modelo(espec, cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS)

    def test_variavel_na_etapa1_com_evidencia(self):
        espec = cfg.modelo_variavel("M4v", ("original_glcm_Imc1",))
        autorizacao = g.autorizar_modelo(espec, cfg.Etapa.E1, self.decisoes_e1(), self.repo.raiz)
        self.assertEqual(autorizacao.decisoes_exigidas, ("P1", "P2"))

    def test_modelos_com_painel_autorizados_com_p3_verificada(self):
        decisoes = self.decisoes_e3()
        for nome in cfg.MODELOS_COM_PAINEL:
            with self.subTest(modelo=nome):
                espec = cfg.modelo_fixo(nome, decisoes)
                g.autorizar_modelo(espec, cfg.Etapa.E3, decisoes, self.repo.raiz)

    def test_modelo_com_painel_exige_p3_mesmo_em_etapa_que_nao_a_exige(self):
        espec = cfg.modelo_fixo("M2", self.decisoes_e3())
        with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
            g.autorizar_modelo(espec, cfg.Etapa.E1, self.decisoes_e1(), self.repo.raiz)
        with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
            g.autorizar_modelo(espec, cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS, cfg.DECISOES,
                               self.repo.raiz)

    def test_modelo_montado_com_p3_sem_evidencia(self):
        # A marca de EspecModelo existe, mas P3 aponta para commit sem registro.
        decisoes = self.decisoes_e1(P3=(self.c1, VALOR_P3))
        espec = cfg.modelo_fixo("M3", decisoes)
        with self.assertRaisesRegex(g.AprovacaoNaoVerificada, "P3"):
            g.autorizar_modelo(espec, cfg.Etapa.E3, decisoes, self.repo.raiz)

    def test_marca_forjada_com_outro_painel_e_recusada(self):
        decisoes = self.decisoes_e3()
        forjado = cfg.EspecModelo("M2", cfg.TipoModelo.FIXO, OUTRO_PAINEL,
                                  _autorizacao=(cfg._AUTORIZACAO_PAINEL_P3, OUTRO_PAINEL))
        with self.assertRaisesRegex(g.DecisaoInconsistente, "composição diverge"):
            g.autorizar_modelo(forjado, cfg.Etapa.E3, decisoes, self.repo.raiz)

    def test_especs_adulteradas_depois_da_construcao(self):
        decisoes = self.decisoes_e3()
        casos = [
            (cfg.modelo_fixo("M4", decisoes), "features", cfg.FEATURES_M1 + OUTRO_PAINEL),
            (cfg.modelo_fixo("M1"), "features", (cfg.DIAMETRO,)),
            (cfg.modelo_fixo("M0"), "tipo", cfg.TipoModelo.VARIAVEL),
            (cfg.modelo_variavel("M3v", ("original_glcm_Imc1",)), "features", ("original_glcm_Imc1",)),
            (cfg.modelo_variavel("M2v", ()), "usa_dummy", False),
            (cfg.modelo_variavel("M2v", ("original_glcm_Imc1",)), "usa_dummy", True),
            (cfg.modelo_variavel("M4v", ("original_glcm_Imc1",)), "features",
             cfg.FEATURES_M1 + ("original_glcm_Imc1", "original_glcm_Imc1")),
        ]
        # Atividade estrutural: só M4 consulta o git (por exigir P3).
        for espec, atributo, valor in casos:
            with self.subTest(modelo=espec.nome, atributo=atributo):
                object.__setattr__(espec, atributo, valor)
                with self.assertRaises(g.DecisaoInconsistente):
                    g.autorizar_modelo(espec, cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS, decisoes,
                                       self.repo.raiz)

    def test_nome_adulterado_e_tipo_invalido(self):
        espec = cfg.modelo_fixo("M0")
        object.__setattr__(espec, "nome", "M9")
        with self.assertRaises(g.DecisaoInconsistente):
            g.autorizar_modelo(espec, cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS)
        with self.assertRaises(TypeError):
            g.autorizar_modelo(("M0",), cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS)


# --------------------------------------------------------------------------- #
# Formato do registro e isolamento do módulo
# --------------------------------------------------------------------------- #


class TestFormatoEIsolamento(unittest.TestCase):
    def test_hash_do_valor(self):
        self.assertEqual(g.hash_valor_decisao(None), hashlib.sha256(b"null").hexdigest())
        self.assertEqual(g.hash_valor_decisao(VALOR_P3), g.hash_valor_decisao(list(VALOR_P3)))
        self.assertNotEqual(g.hash_valor_decisao(VALOR_P3), g.hash_valor_decisao(OUTRO_PAINEL))

    def test_formatar_registro(self):
        linha = g.formatar_registro("P4", "2026-11-03")
        self.assertEqual(linha, "APROVACAO-DECISAO codigo=P4 status=aprovada data=2026-11-03 "
                                f"valor_sha256={hashlib.sha256(b'null').hexdigest()}")
        with self.assertRaises(g.AprovacaoNaoVerificada):
            g.formatar_registro("P4", "03/11/2026")

    def test_documentos_de_aprovacao_sao_apenas_o_protocolo(self):
        self.assertEqual(g.DOCUMENTOS_APROVACAO, (cfg.PROTOCOLO,))

    def test_modulo_nao_le_dados_nem_escreve_arquivos(self):
        arvore = ast.parse((REPO_ROOT / "src" / "incremental" / "guardas.py").read_text(encoding="utf-8"))
        importados = set()
        for no in ast.walk(arvore):
            if isinstance(no, ast.Import):
                importados.update(a.name.split(".")[0] for a in no.names)
            elif isinstance(no, ast.ImportFrom) and no.level == 0:
                importados.add(no.module.split(".")[0])
        self.assertLessEqual(importados, {"__future__", "datetime", "hashlib", "json", "os", "re",
                                          "shutil", "subprocess", "dataclasses", "pathlib", "typing"})
        chamadas = {no.func.attr if isinstance(no.func, ast.Attribute) else getattr(no.func, "id", None)
                    for no in ast.walk(arvore) if isinstance(no, ast.Call)}
        proibidas = {"open", "read_csv", "read_parquet", "read_text", "read_bytes", "write_text",
                     "write_bytes", "load", "fit", "predict_proba", "remove", "unlink", "rmtree"}
        self.assertFalse(chamadas & proibidas, chamadas & proibidas)

    def test_comandos_git_sao_somente_leitura(self):
        fonte = (REPO_ROOT / "src" / "incremental" / "guardas.py").read_text(encoding="utf-8")
        arvore = ast.parse(fonte)
        comandos = set()
        for no in ast.walk(arvore):
            if (isinstance(no, ast.Call) and getattr(no.func, "id", None) == "_git" and len(no.args) > 1
                    and isinstance(no.args[1], ast.Constant)):
                comandos.add(no.args[1].value)
        self.assertEqual(comandos, {"rev-parse", "cat-file", "merge-base", "rev-list", "log"})

    def test_guardas_nao_alteram_a_configuracao(self):
        antes = cfg.hash_configuracao()
        with mock.patch.object(g, "_git", git_proibido):
            g.exigir_autorizacao(cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS)
            with self.assertRaises(cfg.BloqueioDecisao):
                g.exigir_autorizacao(cfg.Etapa.E5)
        self.assertEqual(cfg.hash_configuracao(), antes)
        self.assertEqual(antes, HASH_CONFIG_INCREMENTO_1)


if __name__ == "__main__":
    unittest.main()
