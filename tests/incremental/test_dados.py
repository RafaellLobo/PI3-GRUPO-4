"""Testes do Incremento 2B — acesso controlado aos dados (``src/incremental/dados.py``).

Usam EXCLUSIVAMENTE CSVs sintéticos em diretórios temporários. Os únicos
arquivos reais consultados são versionados e não têm dados de pacientes: o CSV
com os NOMES das 107 candidatas e o JSON da reprodução (para conferir os pins).
``data/`` nunca é lido. Aprovações de decisões existem só em repositórios git
temporários; nenhuma aprovação real é presumida.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/incremental -v
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import hashlib
import inspect
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for caminho in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.incremental import config as cfg  # noqa: E402
from src.incremental import dados  # noqa: E402
from src.incremental import guardas as g  # noqa: E402
from test_guardas import (  # noqa: E402
    HASH_CONFIG_INCREMENTO_1, VALOR_P2, VALOR_P3, RepoTemporario, aprovar, protocolo, registro,
)

E0, E1, E2, E3, E4, E5 = (e.value for e in cfg.Etapa)
PILOTO = cfg.ATIVIDADE_PILOTO_EXPLORATORIO
SENTINELA = "987654.321"
SENTINELA_ALVO_DERIVADO = "[9, 9, 9, 9]"


def _nomes_pool_versionado() -> tuple:
    with open(dados.POOL_VERSIONADO.caminho, encoding="utf-8", newline="") as arquivo:
        return tuple(linha["feature"] for linha in csv.DictReader(arquivo))


POOL = _nomes_pool_versionado()
#: Ordem de cabeçalho sintética, diferente da ordem do CSV versionado.
ORDEM_CABECALHO = tuple(random.Random(11).sample(POOL, len(POOL)))
COLUNAS_INICIAIS = [
    "nodulo_id", "patient_id", "scan_id", "pos_selecionada", "original_nodule_idx",
    "n_annotations_total", "n_annotations_with_characteristics", "malignancy_values",
    "malignancy_median", "target_binary", "exclusion_reason", "teto_adaptativo_HU",
    "n_voxels_base", "n_voxels_resegmentado",
]
COLUNAS_DERIVADAS_DO_ROTULO = {"malignancy_values", "malignancy_median", "exclusion_reason",
                               "n_annotations_total", "n_annotations_with_characteristics"}


def valor_feature(n: int, k: int, j: int) -> float:
    """Valor determinístico que identifica paciente n, nódulo k e feature j."""
    return n + k / 10 + j / 10000


class Sintetico:
    """Tabela e split sintéticos no formato da Sprint 3 (pacientes LIDC-IDRI-9xxx, fictícios)."""

    def __init__(self, pasta: Path, por_particao=(("train", 6), ("validation", 3), ("test", 3))):
        self.pasta = pasta
        self.linhas, self.particao_do_paciente, self.n_k = [], {}, {}
        n = 0
        for particao, quantidade in por_particao:
            for _ in range(quantidade):
                n += 1
                paciente = f"LIDC-IDRI-{9000 + n:04d}"
                self.particao_do_paciente[paciente] = particao
                for k in range(1, n % 3 + 2):
                    alvo = (n + k) % 2
                    nodulo = f"{paciente}_N{k:02d}_scan{9000 + n}"
                    self.n_k[nodulo] = (n, k)
                    linha = {
                        "nodulo_id": nodulo, "patient_id": paciente, "scan_id": str(9000 + n),
                        "pos_selecionada": str(k), "original_nodule_idx": str(k - 1),
                        "n_annotations_total": "4", "n_annotations_with_characteristics": "4",
                        "malignancy_values": json.dumps([4, 5] if alvo else [1, 2]),
                        "malignancy_median": "4.5" if alvo else "1.5",
                        "target_binary": str(alvo), "exclusion_reason": "included",
                        "teto_adaptativo_HU": "200", "n_voxels_base": "100",
                        "n_voxels_resegmentado": "90",
                    }
                    for j, nome in enumerate(ORDEM_CABECALHO):
                        linha[nome] = repr(valor_feature(n, k, j))
                    self.linhas.append(linha)
        random.Random(5).shuffle(self.linhas)  # arquivo fora da ordem canônica
        self.cabecalho = COLUNAS_INICIAIS + list(ORDEM_CABECALHO)

    def particao(self, linha: dict) -> str:
        return self.particao_do_paciente[linha["patient_id"]]

    def esperado(self) -> tuple:
        coorte = dict.fromkeys(("nodulos", "pacientes", "classe_0", "classe_1"), 0)
        particoes = {p: dict(coorte) for p in dados.PARTICOES}
        for linha in self.linhas:
            alvo = f"classe_{linha['target_binary']}"
            for destino in (coorte, particoes[self.particao(linha)]):
                destino["nodulos"] += 1
                destino[alvo] += 1
        for paciente, particao in self.particao_do_paciente.items():
            coorte["pacientes"] += 1
            particoes[particao]["pacientes"] += 1
        return coorte, particoes

    def escrever(self, eol="\n", proibidas="normal", mutar=None, mutar_split=None,
                 mutar_texto=None, coorte=None, particoes=None, nomes_pool=None,
                 hashes_de=None) -> dados.FonteDados:
        """Grava os CSVs e devolve a fonte sintética correspondente.

        ``proibidas``: ``normal``, ``sentinela`` (valores marcados) ou ``lixo``
        (valores inválidos) nas linhas fora do treino.
        """
        linhas = [dict(linha) for linha in self.linhas]
        for linha in linhas:
            if self.particao(linha) == "train" or proibidas == "normal":
                continue
            if proibidas == "sentinela":
                linha.update({nome: SENTINELA for nome in ORDEM_CABECALHO})
                linha["malignancy_values"] = SENTINELA_ALVO_DERIVADO
            elif proibidas == "lixo":
                linha.update({nome: "lixo" for nome in ORDEM_CABECALHO})
                linha["target_binary"] = "7"
        cabecalho = list(self.cabecalho)
        if mutar:
            mutar(linhas, cabecalho)
        texto = io.StringIO()
        escritor = csv.writer(texto, lineterminator=eol)
        escritor.writerow(cabecalho)
        for linha in linhas:
            escritor.writerow([linha.get(c, "") for c in cabecalho])
        conteudo = texto.getvalue()
        if mutar_texto:
            conteudo = mutar_texto(conteudo)

        split = [["patient_id", "split"]] + [[p, s] for p, s in sorted(self.particao_do_paciente.items())]
        if mutar_split:
            mutar_split(split)
        texto_split = io.StringIO()
        csv.writer(texto_split, lineterminator=eol).writerows(split)

        tabela_csv, split_csv = self.pasta / "tabela.csv", self.pasta / "split.csv"
        tabela_csv.write_bytes(conteudo.encode("utf-8"))
        split_csv.write_bytes(texto_split.getvalue().encode("utf-8"))
        coorte_esperada, particoes_esperadas = self.esperado()
        return dados.FonteDados(
            tabela=self._verificado(tabela_csv, hashes_de),
            split=self._verificado(split_csv, None),
            nomes_pool=frozenset(POOL if nomes_pool is None else nomes_pool),
            coorte=coorte or coorte_esperada,
            particoes=particoes or particoes_esperadas,
        )

    @staticmethod
    def _verificado(caminho: Path, conteudo_para_hash) -> dados.ArquivoVerificado:
        conteudo = caminho.read_bytes() if conteudo_para_hash is None else conteudo_para_hash
        return dados.ArquivoVerificado(
            caminho, hashlib.sha256(conteudo).hexdigest(),
            hashlib.sha256(conteudo.replace(b"\r\n", b"\n")).hexdigest())


class ComSintetico(unittest.TestCase):
    def setUp(self):
        temporario = tempfile.TemporaryDirectory()
        self.addCleanup(temporario.cleanup)
        self.pasta = Path(temporario.name)
        self.sint = Sintetico(self.pasta)

    def carregar_e0(self, fonte, features=POOL, particoes=("train",)):
        return dados.carregar(E0, particoes, features, fonte=fonte)

    def assertInvalido(self, padrao, fonte, **kwargs):
        with self.assertRaisesRegex(dados.DadosInvalidos, padrao):
            self.carregar_e0(fonte, **kwargs)

    def ids_da_particao(self, *particoes) -> set:
        return {l["nodulo_id"] for l in self.sint.linhas if self.sint.particao(l) in particoes}


def leituras_proibidas(*_args, **_kwargs):
    raise AssertionError("nenhum arquivo deveria ser lido")


class CapturaReadCsv:
    """Guarda todo DataFrame que ``pd.read_csv`` materializa durante a carga."""

    def __init__(self):
        self.frames = []
        self._original = pd.read_csv

    def __call__(self, *args, **kwargs):
        frame = self._original(*args, **kwargs)
        self.frames.append(frame)
        return frame

    def __enter__(self):
        self._patch = mock.patch.object(dados.pd, "read_csv", self)
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        return self._patch.__exit__(*exc)


# --------------------------------------------------------------------------- #
# Fonte oficial e pins
# --------------------------------------------------------------------------- #


class TestFonteOficial(unittest.TestCase):
    def test_pins_iguais_aos_da_reproducao_versionada(self):
        relatorio = json.loads((REPO_ROOT / "reports" / "sprint4" / "feature_selection" / "reproducao_juan"
                                / "relatorio_reproducao.json").read_text(encoding="utf-8"))
        entradas = relatorio["sha256_entradas"]
        for arquivo, chave in ((dados.TABELA_OFICIAL, "data/modeling_table_sprint3.csv"),
                               (dados.SPLIT_OFICIAL, "reports/sprint3/baseline/patient_split.csv")):
            with self.subTest(arquivo=chave):
                self.assertEqual(arquivo.caminho.relative_to(REPO_ROOT).as_posix(), chave)
                self.assertEqual(arquivo.sha256, entradas[chave]["sha256"])
                self.assertEqual(arquivo.sha256_lf, entradas[chave]["sha256_lf_normalizado"])
        self.assertEqual(relatorio["checagens_previas"]["n_features_candidatas"], cfg.N_POOL_CANDIDATAS)

    def test_fonte_oficial_le_apenas_o_pool_versionado(self):
        lidos = []
        original = dados._ler_bytes
        with mock.patch.object(dados, "_ler_bytes", lambda c: lidos.append(c) or original(c)):
            fonte = dados.fonte_oficial()
        self.assertEqual(lidos, [dados.POOL_VERSIONADO.caminho])
        self.assertEqual(fonte.tabela, dados.TABELA_OFICIAL)
        self.assertEqual(dict(fonte.coorte), dict(cfg.COORTE_ESPERADA))
        self.assertEqual({p: dict(v) for p, v in fonte.particoes.items()},
                         {p: dict(v) for p, v in cfg.PARTICOES_ESPERADAS.items()})
        self.assertEqual(fonte, dados.fonte_oficial())

    def test_pool_versionado_tem_107_nomes_e_contem_modelos_e_paineis(self):
        self.assertEqual(len(POOL), cfg.N_POOL_CANDIDATAS)
        self.assertEqual(len(set(POOL)), cfg.N_POOL_CANDIDATAS)
        self.assertNotIn(cfg.COLUNA_PROIBIDA, POOL)
        for grupo in (cfg.FEATURES_M1, cfg.PAINEL_DECLARADO, cfg.PAINEL_REPRODUZIDO):
            self.assertLessEqual(set(grupo), set(POOL))

    def test_pool_versionado_alterado_e_recusado(self):
        with tempfile.TemporaryDirectory() as pasta:
            copia = Path(pasta) / "frequencia.csv"
            copia.write_bytes(dados.POOL_VERSIONADO.caminho.read_bytes() + b"original_x,0\n")
            alterado = dataclasses.replace(dados.POOL_VERSIONADO, caminho=copia)
            with mock.patch.object(dados, "POOL_VERSIONADO", alterado):
                with self.assertRaisesRegex(dados.DadosInvalidos, "SHA-256"):
                    dados.fonte_oficial()

    def test_fonte_que_toca_arquivos_oficiais_tem_de_ser_a_oficial(self):
        with tempfile.TemporaryDirectory() as pasta:
            fonte = Sintetico(Path(pasta)).escrever()
            trocas = {
                "tabela em data/": dataclasses.replace(
                    fonte, tabela=dataclasses.replace(fonte.tabela, caminho=dados.TABELA_OFICIAL.caminho)),
                "qualquer arquivo em data/": dataclasses.replace(
                    fonte, tabela=dataclasses.replace(fonte.tabela, caminho=dados.DIRETORIO_DADOS / "x.csv")),
                "split oficial": dataclasses.replace(
                    fonte, split=dataclasses.replace(fonte.split, caminho=dados.SPLIT_OFICIAL.caminho)),
                "oficial com expectativa trocada": dataclasses.replace(
                    dados.fonte_oficial(), particoes=fonte.particoes, coorte=fonte.coorte),
            }
            original = dados._ler_bytes
            for nome, trocada in trocas.items():
                with self.subTest(caso=nome):
                    def ler(caminho):
                        if caminho != dados.POOL_VERSIONADO.caminho:
                            raise AssertionError(f"leitura indevida de {caminho.name}")
                        return original(caminho)

                    with mock.patch.object(dados, "_ler_bytes", ler):
                        with self.assertRaises(dados.AcessoNegado):
                            dados.carregar(E0, ["train"], POOL, fonte=trocada)

    def test_validacoes_da_fonte(self):
        with tempfile.TemporaryDirectory() as pasta:
            fonte = Sintetico(Path(pasta)).escrever()
        with self.assertRaises(ValueError):
            dados.ArquivoVerificado(Path("x"), "abc", "abc")
        with self.assertRaises(TypeError):
            dados.ArquivoVerificado("x", "0" * 64, "0" * 64)
        for campo, valor in (("nomes_pool", frozenset(POOL[:106])),
                             ("nomes_pool", frozenset(POOL[:106] + (cfg.COLUNA_PROIBIDA,))),
                             ("coorte", {**fonte.coorte, "nodulos": fonte.coorte["nodulos"] + 1}),
                             ("particoes", {"train": fonte.particoes["train"]})):
            with self.subTest(campo=campo):
                with self.assertRaises((ValueError, TypeError)):
                    dataclasses.replace(fonte, **{campo: valor})
        with self.assertRaises(TypeError):
            dataclasses.replace(fonte, nomes_pool=set(POOL))


# --------------------------------------------------------------------------- #
# Acesso por atividade e integração com as guardas
# --------------------------------------------------------------------------- #


class TestAcessoSemAprovacoes(ComSintetico):
    def test_etapa0_le_somente_treino(self):
        carga = self.carregar_e0(self.sint.escrever())
        self.assertEqual(carga.particoes, ("train",))
        self.assertEqual(set(carga.frame("train")["nodulo_id"]), self.ids_da_particao("train"))
        self.assertIsNone(carga.autorizacao.head)
        for particao in ("validation", "test"):
            with self.assertRaises(dados.AcessoNegado):
                carga.frame(particao)

    def test_partições_fora_da_etapa_negadas_sem_ler_arquivos(self):
        fonte = self.sint.escrever()
        casos = [(E0, ["validation"]), (E0, ["train", "test"]), (E1, ["validation"]),
                 (E2, ["test"]), (cfg.ATIVIDADE_CV_LISTA_FIXA, ["validation"]),
                 (E3, ["test"]), (E3, ["train", "validation", "test"]), (E5, ["validation"]),
                 (E4, ["train"]), (cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS, ["train"]),
                 (PILOTO, ["validation"]), (PILOTO, ["test"]), (PILOTO, ["train", "validation"]),
                 (PILOTO, ["train", "test"]), (PILOTO, ["train", "validation", "test"])]
        for atividade, particoes in casos:
            with self.subTest(atividade=atividade, particoes=particoes), \
                    mock.patch.object(dados, "_ler_bytes", leituras_proibidas), \
                    mock.patch.object(g, "_git", leituras_proibidas):
                with self.assertRaises(dados.AcessoNegado):
                    dados.carregar(atividade, particoes, POOL, fonte=fonte)

    def test_piloto_exploratorio_le_somente_treino_sem_decisoes(self):
        # Atividade própria: sem decisões (nem git), só a política de leitura da Etapa 1.
        fonte = self.sint.escrever()
        with mock.patch.object(g, "_git", leituras_proibidas):
            carga = dados.carregar(PILOTO, ["train"], POOL, fonte=fonte)
        self.assertEqual((carga.atividade, carga.particoes), (PILOTO, ("train",)))
        self.assertEqual((carga.autorizacao.atividade, carga.autorizacao.decisoes_exigidas), (PILOTO, ()))
        self.assertIsNone(carga.autorizacao.head)
        self.assertEqual(set(carga.frame("train")["nodulo_id"]), self.ids_da_particao("train"))
        for particao in ("validation", "test"):
            with self.assertRaises(dados.AcessoNegado):
                carga.frame(particao)
        # A Etapa 1 oficial continua exigindo P1 e P2 para a mesma leitura.
        with mock.patch.object(dados, "_ler_bytes", leituras_proibidas):
            with self.assertRaisesRegex(cfg.BloqueioDecisao, "P1"):
                dados.carregar(E1, ["train"], POOL, fonte=fonte)

    def test_etapas_com_decisoes_bloqueadas_sem_ler_arquivos(self):
        # Estado real: P1–P5 e REVISAO_PROTOCOLO pendentes. Nomear a etapa não basta.
        fonte = self.sint.escrever()
        casos = [(E1, ["train"]), (E2, ["train"]), (cfg.ATIVIDADE_CV_LISTA_FIXA, ["train"]),
                 (E3, ["train", "validation"]), (E3, ["validation"]), (E5, ["train", "test"]),
                 (E5, ["test"]), (cfg.Etapa.E3, ["validation"])]
        for atividade, particoes in casos:
            with self.subTest(atividade=atividade, particoes=particoes), \
                    mock.patch.object(dados, "_ler_bytes", leituras_proibidas):
                with self.assertRaises(cfg.BloqueioDecisao):
                    dados.carregar(atividade, particoes, POOL, fonte=fonte)

    def test_atividade_e_particoes_invalidas(self):
        fonte = self.sint.escrever()
        with mock.patch.object(dados, "_ler_bytes", leituras_proibidas):
            for atividade in ("etapa0", "E0", " etapa0_prerequisitos", None):
                with self.subTest(atividade=atividade):
                    with self.assertRaises(g.AtividadeDesconhecida):
                        dados.carregar(atividade, ["train"], POOL, fonte=fonte)
            with self.assertRaises(TypeError):
                dados.carregar(E0, "train", POOL, fonte=fonte)
            for particoes in ([], ["treino"], ["train", "train"], [None], ["TRAIN"]):
                with self.subTest(particoes=particoes):
                    with self.assertRaises(dados.AcessoNegado):
                        dados.carregar(E0, particoes, POOL, fonte=fonte)
            with self.assertRaises(TypeError):
                dados.carregar(E0, ["train"], cfg.DIAMETRO, fonte=fonte)

    def test_api_nao_aceita_autorizacao_pronta(self):
        parametros = set(inspect.signature(dados.carregar).parameters)
        self.assertEqual(parametros, {"atividade", "particoes", "features", "decisoes", "repo", "fonte"})

    def test_autorizacao_forjada_nao_tem_efeito(self):
        # Um objeto Autorizacao montado à mão não é aceito por nenhuma via.
        forjada = g.Autorizacao(E5, ("P1",), (), "0" * 40, "0" * 64)
        with mock.patch.object(dados, "_ler_bytes", leituras_proibidas):
            with self.assertRaises(TypeError):
                dados.carregar(E5, ["test"], POOL, fonte=self.sint.escrever(), autorizacao=forjada)


class TestAcessoComAprovacoesFicticias(ComSintetico):
    """Aprovações fictícias num repositório git temporário (nenhuma é real)."""

    @classmethod
    def setUpClass(cls):
        repo = cls.repo = RepoTemporario()
        r12 = [registro("P1"), registro("P2", VALOR_P2)]
        r3 = r12 + [registro("P3", VALOR_P3, "2026-10-03")]
        r5 = r3 + [registro(c, data="2026-10-04") for c in ("P4", "P5", "REVISAO_PROTOCOLO")]
        repo.commit(protocolo(), "2026-10-01")
        c1 = repo.commit(protocolo(*r12), "2026-10-02")
        c2 = repo.commit(protocolo(*r3), "2026-10-03")
        c3 = repo.commit(protocolo(*r5), "2026-10-04")
        cls.c0_sem_registro = repo.git("rev-parse", "HEAD~3")
        cls.dec_e1 = aprovar(P1=(c1, None), P2=(c1, VALOR_P2))
        cls.dec_e3 = aprovar(cls.dec_e1, P3=(c2, VALOR_P3))
        cls.dec_e5 = aprovar(cls.dec_e3, P4=(c3, None), P5=(c3, None), REVISAO_PROTOCOLO=(c3, None))

    @classmethod
    def tearDownClass(cls):
        cls.repo.remover()

    def carregar(self, atividade, particoes, decisoes, fonte=None):
        return dados.carregar(atividade, particoes, POOL, decisoes=decisoes, repo=self.repo.raiz,
                              fonte=fonte or self.sint.escrever())

    def test_etapas_1_e_2_somente_treino(self):
        for atividade in (E1, E2):
            with self.subTest(atividade=atividade):
                carga = self.carregar(atividade, ["train"], self.dec_e1)
                self.assertEqual(carga.autorizacao.decisoes_exigidas, ("P1", "P2"))
                with self.assertRaises(dados.AcessoNegado):
                    self.carregar(atividade, ["train", "validation"], self.dec_e1)

    def test_cv_lista_fixa_exige_p3(self):
        with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
            self.carregar(cfg.ATIVIDADE_CV_LISTA_FIXA, ["train"], self.dec_e1)
        self.carregar(cfg.ATIVIDADE_CV_LISTA_FIXA, ["train"], self.dec_e3)

    def test_etapa3_treino_e_validacao_mas_nunca_teste(self):
        carga = self.carregar(E3, ["validation", "train"], self.dec_e3)
        self.assertEqual(carga.particoes, ("train", "validation"))
        self.assertEqual(set(carga.frame("validation")["nodulo_id"]), self.ids_da_particao("validation"))
        self.assertTrue(set(carga.frame("train")["nodulo_id"]).isdisjoint(self.ids_da_particao("validation")))
        with self.assertRaises(dados.AcessoNegado):
            carga.frame("test")
        with self.assertRaises(dados.AcessoNegado):
            self.carregar(E3, ["train", "test"], self.dec_e5)
        with self.assertRaises(cfg.BloqueioDecisao):
            self.carregar(E3, ["train", "validation"], self.dec_e1)

    def test_etapa5_teste_so_com_todas_as_decisoes(self):
        with self.assertRaises(cfg.BloqueioDecisao):
            self.carregar(E5, ["train", "test"], self.dec_e3)
        carga = self.carregar(E5, ["train", "test"], self.dec_e5)
        self.assertEqual(set(carga.autorizacao.decisoes_exigidas), set(cfg.DECISOES))
        self.assertEqual(set(carga.frame("test")["nodulo_id"]), self.ids_da_particao("test"))
        with self.assertRaises(dados.AcessoNegado):
            self.carregar(E5, ["validation"], self.dec_e5)

    def test_decisao_sem_evidencia_bloqueia_antes_de_ler(self):
        sem_registro = aprovar(self.dec_e1, P3=(self.c0_sem_registro, VALOR_P3))
        fonte = self.sint.escrever()
        with mock.patch.object(dados, "_ler_bytes", leituras_proibidas):
            with self.assertRaises(g.AprovacaoNaoVerificada):
                self.carregar(E3, ["train", "validation"], sem_registro, fonte=fonte)

    def test_aprovacoes_ficticias_nao_liberam_dados_oficiais(self):
        # Regressão: o repositório temporário destes testes liberaria o teste real.
        oficial = dados.fonte_oficial()
        for fonte in (None, oficial):
            for atividade, particoes, decisoes in ((E5, ["train", "test"], self.dec_e5),
                                                   (E3, ["validation"], self.dec_e3),
                                                   (E0, ["train"], cfg.DECISOES),
                                                   (PILOTO, ["train"], cfg.DECISOES)):
                with self.subTest(fonte=fonte is None, atividade=atividade), \
                        mock.patch.object(dados, "_ler_bytes", leituras_proibidas), \
                        mock.patch.object(g, "_git", leituras_proibidas):
                    with self.assertRaisesRegex(dados.AcessoNegado, "repositório do projeto"):
                        dados.carregar(atividade, particoes, POOL, decisoes=decisoes,
                                       repo=self.repo.raiz, fonte=fonte)

    def test_copia_do_conteudo_oficial_em_outro_caminho_e_recusada(self):
        # Regressão: simula a cópia da tabela (ou do split) oficial para fora de
        # data/, fixando como "oficial" o hash do arquivo sintético.
        fonte = self.sint.escrever()
        for nome, alvo, arquivo in (("tabela", "TABELA_OFICIAL", fonte.tabela),
                                    ("split", "SPLIT_OFICIAL", fonte.split)):
            pin = dados.ArquivoVerificado(getattr(dados, alvo).caminho, arquivo.sha256, arquivo.sha256_lf)
            with self.subTest(copia=nome), mock.patch.object(dados, alvo, pin), CapturaReadCsv() as captura:
                with self.assertRaisesRegex(dados.AcessoNegado, "conteúdo oficial"):
                    self.carregar(E5, ["train", "test"], self.dec_e5, fonte=fonte)
                with self.assertRaisesRegex(dados.AcessoNegado, "conteúdo oficial"):
                    dados.verificar_estrutura(fonte)
                self.assertEqual(captura.frames, [])

    def test_carregar_dados_nao_autoriza_modelo(self):
        # E1 com as 107 candidatas carregadas: a autorização é só da leitura (P1, P2).
        carga = self.carregar(E1, ["train"], self.dec_e1)
        self.assertEqual(carga.features, ORDEM_CABECALHO)
        self.assertNotIn("P3", carga.autorizacao.decisoes_exigidas)
        espec_m2 = cfg.modelo_fixo("M2", self.dec_e3)
        with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
            g.autorizar_modelo(espec_m2, E1, self.dec_e1, self.repo.raiz)
        fonte = (REPO_ROOT / "src" / "incremental" / "dados.py").read_text(encoding="utf-8")
        self.assertNotIn("autorizar_modelo(", fonte.replace("``guardas.autorizar_modelo``", ""))
        self.assertNotIn("EspecModelo", fonte)

    def test_etapa4_nao_le_nem_com_aprovacoes(self):
        with self.assertRaises(dados.AcessoNegado):
            self.carregar(E4, ["train"], self.dec_e5)


# --------------------------------------------------------------------------- #
# Isolamento: dados proibidos não chegam a DataFrames
# --------------------------------------------------------------------------- #


class TestIsolamento(ComSintetico):
    def test_valores_de_particoes_proibidas_nunca_sao_materializados(self):
        fonte = self.sint.escrever(proibidas="sentinela")
        proibidos = self.ids_da_particao("validation", "test")
        with CapturaReadCsv() as captura:
            carga = self.carregar_e0(fonte)
        self.assertEqual(len(captura.frames), 3)  # split, identificadores, linhas autorizadas
        colunas_permitidas_sem_filtro = {"nodulo_id", "patient_id", "split"}
        for frame in captura.frames:
            texto = frame.astype(str).to_numpy()
            self.assertFalse((texto == SENTINELA).any() | (texto == str(float(SENTINELA))).any())
            self.assertFalse((texto == SENTINELA_ALVO_DERIVADO).any())
            if not set(frame.columns) <= colunas_permitidas_sem_filtro:
                self.assertTrue(set(frame["nodulo_id"]).isdisjoint(proibidos))
        dados_lidos = captura.frames[-1]
        self.assertEqual(set(dados_lidos["nodulo_id"]), self.ids_da_particao("train"))
        self.assertTrue(set(carga.frame("train")["nodulo_id"]).isdisjoint(proibidos))

    def test_piloto_nunca_materializa_validacao_ou_teste(self):
        fonte = self.sint.escrever(proibidas="sentinela")
        proibidos = self.ids_da_particao("validation", "test")
        with CapturaReadCsv() as captura:
            dados.carregar(PILOTO, ["train"], POOL, fonte=fonte)
        for frame in captura.frames:
            texto = frame.astype(str).to_numpy()
            self.assertFalse((texto == SENTINELA).any() | (texto == str(float(SENTINELA))).any())
            if not set(frame.columns) <= {"nodulo_id", "patient_id", "split"}:
                self.assertTrue(set(frame["nodulo_id"]).isdisjoint(proibidos))
        self.assertEqual(set(captura.frames[-1]["nodulo_id"]), self.ids_da_particao("train"))

    def test_leitura_inicial_restrita_a_identificadores(self):
        with CapturaReadCsv() as captura:
            self.carregar_e0(self.sint.escrever())
        self.assertEqual(list(captura.frames[0].columns), ["patient_id", "split"])
        self.assertEqual(list(captura.frames[1].columns), ["nodulo_id", "patient_id"])

    def test_lixo_em_particoes_proibidas_nao_e_interpretado(self):
        # Alvo "7" e features "lixo" fora do treino: se fossem lidos, a carga falharia.
        carga = self.carregar_e0(self.sint.escrever(proibidas="lixo"))
        self.assertEqual(set(carga.frame("train")["nodulo_id"]), self.ids_da_particao("train"))

    def test_colunas_derivadas_do_rotulo_nunca_sao_lidas(self):
        with CapturaReadCsv() as captura:
            carga = self.carregar_e0(self.sint.escrever())
        for frame in [*captura.frames, carga.frame("train")]:
            self.assertFalse(set(frame.columns) & COLUNAS_DERIVADAS_DO_ROTULO)
            self.assertNotIn(cfg.COLUNA_PROIBIDA, frame.columns)
            self.assertNotIn("scan_id", frame.columns)

    def test_verificacao_estrutural_nao_le_alvo_nem_features(self):
        with CapturaReadCsv() as captura:
            relatorio = dados.verificar_estrutura(self.sint.escrever())
        for frame in captura.frames:
            self.assertLessEqual(set(frame.columns), {"nodulo_id", "patient_id", "split"})
        _, esperado = self.sint.esperado()
        for p in dados.PARTICOES:
            self.assertEqual(dict(relatorio.contagens[p]),
                             {k: esperado[p][k] for k in ("nodulos", "pacientes")})

    def test_registro_em_varias_linhas_aborta_antes_da_leitura_seletiva(self):
        def quebra(linhas, _cabecalho):
            for linha in linhas:
                if self.sint.particao(linha) == "test":
                    linha["exclusion_reason"] = "linha\nquebrada"
                    return

        fonte = self.sint.escrever(mutar=quebra)
        with CapturaReadCsv() as captura:
            with self.assertRaisesRegex(dados.DadosInvalidos, "linha física"):
                self.carregar_e0(fonte)
        self.assertEqual(len(captura.frames), 2)  # nenhuma leitura de alvo ou feature

    def test_linha_em_branco_e_cr_isolado_abortam(self):
        for nome, mutar_texto, padrao in (
                ("branco", lambda t: t.replace("\n", "\n\n", 3), "linha em branco|linha física"),
                ("cr", lambda t: t.replace("\n", "\r", 1), r"\\r isolado")):
            with self.subTest(caso=nome):
                self.assertInvalido(padrao, self.sint.escrever(mutar_texto=mutar_texto))


# --------------------------------------------------------------------------- #
# Alinhamento, ordem canônica e acesso aos dados carregados
# --------------------------------------------------------------------------- #


class TestAlinhamentoEOrdem(ComSintetico):
    def test_ordem_canonica_por_nodulo_e_valores_alinhados(self):
        carga = dados.carregar(E0, ["train"], POOL, fonte=self.sint.escrever(eol="\r\n"))
        frame = carga.frame("train")
        self.assertEqual(list(frame["nodulo_id"]), sorted(self.ids_da_particao("train")))
        self.assertEqual(list(frame.columns), ["nodulo_id", "patient_id", "target_binary", *ORDEM_CABECALHO])
        for _, linha in frame.iterrows():
            n, k = self.sint.n_k[linha["nodulo_id"]]
            self.assertEqual(linha["patient_id"], f"LIDC-IDRI-{9000 + n:04d}")
            self.assertEqual(linha["target_binary"], (n + k) % 2)
            # Tolerância: o parser padrão do pandas (o do baseline) pode diferir
            # em 1 ulp; valores de linhas ou features distintas diferem em >= 1e-4.
            for j, nome in enumerate(ORDEM_CABECALHO):
                self.assertAlmostEqual(linha[nome], valor_feature(n, k, j), delta=1e-9)

    def test_carga_deterministica(self):
        fonte = self.sint.escrever()
        a, b = self.carregar_e0(fonte), self.carregar_e0(fonte)
        pd.testing.assert_frame_equal(a.frame("train"), b.frame("train"))
        self.assertEqual(a.sha256_entradas, b.sha256_entradas)

    def test_acessores(self):
        carga = self.carregar_e0(self.sint.escrever())
        modelo = (cfg.FEATURES_M1[1], cfg.FEATURES_M1[0], cfg.FEATURES_M1[2])
        self.assertEqual(list(carga.X("train", modelo).columns), list(modelo))
        self.assertEqual(carga.y("train").dtype, np.int64)
        self.assertTrue(set(carga.y("train")) <= {0, 1})
        self.assertEqual(list(carga.grupos("train")), list(carga.frame("train")["patient_id"]))
        copia = carga.frame("train")
        copia.iloc[0, 3] = -1.0
        self.assertNotEqual(carga.frame("train").iloc[0, 3], -1.0)
        for colunas in ((cfg.DIAMETRO, cfg.DIAMETRO), ("nodulo_id",), (cfg.COLUNA_ALVO,), ()):
            with self.subTest(colunas=colunas):
                with self.assertRaises(dados.FeatureInvalida):
                    carga.X("train", colunas)
        with self.assertRaises(TypeError):
            carga.X("train", cfg.DIAMETRO)
        parcial = self.carregar_e0(self.sint.escrever(), features=cfg.FEATURES_M1)
        self.assertEqual(parcial.features, tuple(n for n in ORDEM_CABECALHO if n in cfg.FEATURES_M1))
        with self.assertRaisesRegex(dados.FeatureInvalida, "não carregadas"):
            parcial.X("train", (cfg.PAINEL_DECLARADO[0],))
        self.assertEqual(list(parcial.frame("train").columns),
                         ["nodulo_id", "patient_id", "target_binary", *parcial.features])

    def test_entradas_nao_sao_alteradas(self):
        fonte = self.sint.escrever()
        antes = {c: c.read_bytes() for c in (fonte.tabela.caminho, fonte.split.caminho)}
        self.carregar_e0(fonte)
        dados.verificar_estrutura(fonte)
        self.assertEqual({c: c.read_bytes() for c in antes}, antes)


# --------------------------------------------------------------------------- #
# Invariantes estruturais e de valores
# --------------------------------------------------------------------------- #


class TestInvariantes(ComSintetico):
    def primeira(self, linhas, particao="train"):
        return next(l for l in linhas if self.sint.particao(l) == particao)

    def test_integridade_por_hash(self):
        fonte = self.sint.escrever()
        conteudo = fonte.tabela.caminho.read_bytes()
        posicao = conteudo.rindex(b"1")
        fonte.tabela.caminho.write_bytes(conteudo[:posicao] + b"2" + conteudo[posicao + 1:])
        self.assertInvalido("SHA-256", fonte)
        fonte = self.sint.escrever()
        fonte.split.caminho.write_bytes(fonte.split.caminho.read_bytes() + b"\n")
        self.assertInvalido("SHA-256", fonte)

    def test_hash_normalizado_em_lf_aceita_crlf(self):
        lf = self.sint.escrever(eol="\n").tabela.caminho.read_bytes()
        fonte = self.sint.escrever(eol="\r\n", hashes_de=lf)
        self.assertNotEqual(fonte.tabela.sha256, hashlib.sha256(fonte.tabela.caminho.read_bytes()).hexdigest())
        self.carregar_e0(fonte)

    def test_identificadores(self):
        def duplicar(linhas, _c):
            linhas.append(dict(self.primeira(linhas)))

        def vazio(linhas, _c):
            self.primeira(linhas, "test")["nodulo_id"] = ""

        def malformado(linhas, _c):
            self.primeira(linhas, "test")["nodulo_id"] = "nodulo-sem-formato"

        def incoerente(linhas, _c):
            linha = self.primeira(linhas, "validation")
            linha["nodulo_id"] = linha["nodulo_id"].replace("LIDC-IDRI-9", "LIDC-IDRI-8")

        def piloto(linhas, _c):
            linha = self.primeira(linhas)
            linha["nodulo_id"] = linha["nodulo_id"].split("_scan")[0]

        for mutar, padrao in ((duplicar, "repetido"), (vazio, "vazio"), (malformado, "formato"),
                              (incoerente, "incoerentes"), (piloto, "formato")):
            with self.subTest(caso=mutar.__name__):
                self.assertInvalido(padrao, self.sint.escrever(mutar=mutar))

    def test_associacao_com_split(self):
        def sem_paciente(split):
            del split[1]

        def paciente_extra(split):
            split.append(["LIDC-IDRI-9999", "train"])

        def paciente_em_duas(split):
            split.append([split[1][0], "test"])

        def particao_desconhecida(split):
            split[2][1] = "holdout"

        def coluna_extra(split):
            for linha in split:
                linha.append("x")

        def vazio(split):
            split[3][0] = ""

        for mutar, padrao in ((sem_paciente, "diferem"), (paciente_extra, "diferem"),
                              (paciente_em_duas, "repetido"), (particao_desconhecida, "desconhecidas"),
                              (coluna_extra, "exatamente as colunas"), (vazio, "vazio")):
            with self.subTest(caso=mutar.__name__):
                self.assertInvalido(padrao, self.sint.escrever(mutar_split=mutar))

    def test_paciente_movido_de_particao_altera_contagens(self):
        def mover(split):
            linha = next(l for l in split[1:] if l[1] == "test")
            linha[1] = "train"

        self.assertInvalido("partição", self.sint.escrever(mutar_split=mover))

    def test_contagens_esperadas(self):
        coorte, particoes = self.sint.esperado()
        trocado = {**particoes,
                   "train": {**particoes["train"], "nodulos": particoes["train"]["nodulos"] + 1},
                   "test": {**particoes["test"], "nodulos": particoes["test"]["nodulos"] - 1}}
        self.assertInvalido("partição", self.sint.escrever(particoes=trocado))
        classes = {**particoes,
                   "train": {**particoes["train"], "classe_0": particoes["train"]["classe_0"] + 1,
                             "classe_1": particoes["train"]["classe_1"] - 1},
                   "test": {**particoes["test"], "classe_0": particoes["test"]["classe_0"] - 1,
                            "classe_1": particoes["test"]["classe_1"] + 1}}
        self.assertInvalido("classes", self.sint.escrever(particoes=classes))

    def test_alvo_nao_binario_na_particao_autorizada(self):
        for valor in ("2", "", "0.5", "1.0", "-1", "sim", "nan"):
            with self.subTest(valor=valor):
                def mutar(linhas, _c, valor=valor):
                    self.primeira(linhas)["target_binary"] = valor

                self.assertInvalido("target_binary", self.sint.escrever(mutar=mutar))

    def test_features_nao_finitas_ou_nao_numericas(self):
        for valor in ("", "nan", "inf", "-inf", "NA", "abc", "1,5"):
            with self.subTest(valor=valor):
                def mutar(linhas, _c, valor=valor):
                    self.primeira(linhas)[cfg.DIAMETRO] = valor

                self.assertInvalido("NaN ou infinito|não numérico", self.sint.escrever(mutar=mutar))

    def test_feature_invalida_fora_das_colunas_pedidas_nao_afeta(self):
        def mutar(linhas, _c):
            self.primeira(linhas)[cfg.PAINEL_DECLARADO[0]] = "inf"

        carga = self.carregar_e0(self.sint.escrever(mutar=mutar), features=cfg.FEATURES_M1)
        self.assertEqual(len(carga.features), 3)

    def test_esquema_do_cabecalho(self):
        def duplicada(_l, cabecalho):
            cabecalho.append(cfg.DIAMETRO)

        def sobra(linhas, cabecalho):
            cabecalho.append("original_shape_Extra")

        def falta(_l, cabecalho):
            cabecalho.remove(cfg.PAINEL_DECLARADO[0])

        def renomeada(linhas, cabecalho):
            cabecalho[cabecalho.index(cfg.PAINEL_DECLARADO[0])] = "original_glcm_Outra"

        def sem_alvo(_l, cabecalho):
            cabecalho.remove(cfg.COLUNA_ALVO)

        def sem_idx(_l, cabecalho):
            cabecalho.remove(cfg.COLUNA_PROIBIDA)

        for mutar, padrao in ((duplicada, "repetidas"), (sobra, "pool"), (falta, "pool"),
                              (renomeada, "pool"), (sem_alvo, "ausentes"), (sem_idx, "ausentes")):
            with self.subTest(caso=mutar.__name__):
                self.assertInvalido(padrao, self.sint.escrever(mutar=mutar))

    def test_pool_esperado_diferente(self):
        outro = POOL[:106] + ("original_glcm_Inexistente",)
        self.assertInvalido("pool", self.sint.escrever(nomes_pool=outro))


# --------------------------------------------------------------------------- #
# Esquema e validação de S_k
# --------------------------------------------------------------------------- #


class TestEsquemaESk(ComSintetico):
    def setUp(self):
        super().setUp()
        self.esquema = dados.esquema_features(self.sint.escrever())

    def test_esquema_segue_a_ordem_do_cabecalho(self):
        self.assertEqual(self.esquema.nomes, ORDEM_CABECALHO)
        self.assertNotEqual(self.esquema.nomes, POOL)
        self.assertNotIn(cfg.COLUNA_PROIBIDA, self.esquema.nomes)

    def test_s_k_valido_volta_na_ordem_do_esquema(self):
        s_k = (cfg.PAINEL_DECLARADO[3], cfg.DIAMETRO, cfg.PAINEL_DECLARADO[0])
        esperado = tuple(n for n in ORDEM_CABECALHO if n in s_k)
        self.assertEqual(self.esquema.validar(s_k), esperado)
        self.assertEqual(self.esquema.validar(reversed(s_k)), esperado)
        self.assertEqual(self.esquema.validar(set(s_k)), esperado)
        self.assertEqual(self.esquema.validar(()), ())
        self.assertEqual(self.esquema.validar(POOL), ORDEM_CABECALHO)

    def test_s_k_invalido(self):
        casos = [
            (("original_glcm_Inexistente",), "não está"),
            ((cfg.DIAMETRO, cfg.DIAMETRO), "duplicados"),
            ((cfg.COLUNA_PROIBIDA,), "identificador"),
            ((cfg.COLUNA_NODULO,), "identificador"),
            ((cfg.COLUNA_PACIENTE,), "identificador"),
            ((cfg.COLUNA_ALVO,), "identificador"),
            (("split",), "identificador"),
            (("malignancy_median",), "não está"),
            ((cfg.DIAMETRO.lower(),), "não está"),
            ((None,), "inválido"),
        ]
        for s_k, padrao in casos:
            with self.subTest(s_k=s_k):
                with self.assertRaisesRegex(dados.FeatureInvalida, padrao):
                    self.esquema.validar(s_k)
        with self.assertRaises(TypeError):
            self.esquema.validar(cfg.DIAMETRO)
        with self.assertRaises(dados.FeatureInvalida):
            self.esquema.validar((), vazio_permitido=False)

    def test_features_da_carga_validadas_pelo_esquema(self):
        fonte = self.sint.escrever()
        for features in ((), ("original_glcm_Inexistente",), (cfg.DIAMETRO, cfg.DIAMETRO),
                         (cfg.COLUNA_PROIBIDA,), (cfg.COLUNA_ALVO,)):
            with self.subTest(features=features):
                with self.assertRaises(dados.FeatureInvalida):
                    dados.carregar(E0, ["train"], features, fonte=fonte)

    def test_esquema_construido_a_mao_exige_107_nomes_validos(self):
        for nomes in (POOL[:106], POOL[:106] + (cfg.COLUNA_PROIBIDA,), POOL[:106] + (POOL[0],),
                      POOL[:106] + ("nodulo_id",)):
            with self.subTest(n=len(nomes)):
                with self.assertRaises(ValueError):
                    dados.EsquemaFeatures(nomes)
        with self.assertRaises(TypeError):
            dados.EsquemaFeatures(list(POOL))


# --------------------------------------------------------------------------- #
# Isolamento do módulo
# --------------------------------------------------------------------------- #


class TestModulo(unittest.TestCase):
    def test_modulo_nao_escreve_nem_treina(self):
        arvore = ast.parse((REPO_ROOT / "src" / "incremental" / "dados.py").read_text(encoding="utf-8"))
        chamadas = {no.func.attr if isinstance(no.func, ast.Attribute) else getattr(no.func, "id", None)
                    for no in ast.walk(arvore) if isinstance(no, ast.Call)}
        proibidas = {"open", "to_csv", "to_parquet", "to_pickle", "write_text", "write_bytes", "mkdir",
                     "unlink", "remove", "rmtree", "fit", "predict", "predict_proba", "roc_auc_score"}
        self.assertFalse(chamadas & proibidas, chamadas & proibidas)
        leituras = [no for no in ast.walk(arvore) if isinstance(no, ast.Call)
                    and isinstance(no.func, ast.Attribute) and no.func.attr == "read_bytes"]
        self.assertEqual(len(leituras), 1)  # só em _ler_bytes

    def test_parser_de_float_e_o_padrao_do_pandas(self):
        # Paridade com o baseline da Sprint 3 e a reprodução auditada.
        self.assertIsNone(dados.PRECISAO_FLOAT_CSV)
        baseline = (REPO_ROOT / "scripts" / "run_baseline.py").read_text(encoding="utf-8")
        self.assertNotIn("float_precision", baseline)

    def test_configuracao_e_decisoes_inalteradas(self):
        self.assertEqual(cfg.hash_configuracao(), HASH_CONFIG_INCREMENTO_1)
        self.assertTrue(all(not d.aprovada for d in cfg.DECISOES.values()))

    def test_mapa_de_atividades_para_etapas(self):
        self.assertEqual(set(dados.ETAPA_DE_LEITURA) | {cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS},
                         set(cfg.DECISOES_POR_ETAPA))
        self.assertNotIn(cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS, dados.ETAPA_DE_LEITURA)
        self.assertIs(dados.ETAPA_DE_LEITURA[cfg.ATIVIDADE_CV_LISTA_FIXA], cfg.Etapa.E2)
        # O piloto reutiliza só a política de leitura da Etapa 1: treino.
        self.assertIs(dados.ETAPA_DE_LEITURA[PILOTO], cfg.Etapa.E1)
        self.assertEqual(cfg.PARTICOES_POR_ETAPA[dados.ETAPA_DE_LEITURA[PILOTO]], ("train",))
        for etapa in cfg.Etapa:
            self.assertIs(dados.ETAPA_DE_LEITURA[etapa.value], etapa)  # etapas oficiais inalteradas
        for etapa in (cfg.Etapa.E0, cfg.Etapa.E1, cfg.Etapa.E2):
            self.assertEqual(cfg.PARTICOES_POR_ETAPA[etapa], ("train",))


if __name__ == "__main__":
    unittest.main()
