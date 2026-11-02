"""Testes do carregamento seguro do benchmark S6 (``src/benchmark_s6/dados.py``).

Todas as tabelas e splits são SINTÉTICOS, gravados em diretórios temporários
pelo próprio teste, com identificadores ``SINT-…``. Nenhum teste lê ``data/``,
o split oficial ou qualquer dado de paciente: os caminhos oficiais só
aparecem em testes que provam a recusa ANTES de qualquer leitura, com a
leitura substituída por uma função que falha.

Nas linhas de ``test`` (e, quando indicado, de ``validation``) o alvo e as
características são o texto ``VAZOU``: se o carregador as interpretasse como
número, a carga falharia. É a prova de que essas linhas nunca são lidas.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/benchmark_s6 -v
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[1]
for caminho in (REPO_ROOT, AQUI):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.benchmark_s6 import adaptadores_s4 as s4  # noqa: E402
from src.benchmark_s6 import aprovacoes as ap  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import dados  # noqa: E402
import test_aprovacoes as ta  # noqa: E402  (auxiliares de repositório sintético)

C = cfg.CONFIGURACAO
FEATURES = tuple(C.caracteristicas)
VOLUME, DIAMETRO, ESFERICIDADE = FEATURES
LIXO = "VAZOU"
SHA_CFG = cfg.sha256_configuracao()


# --------------------------------------------------------------------------- #
# Fonte sintética
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Linha:
    nodulo: str
    paciente: str
    particao: str
    alvo: int
    diametro: float


def linhas_padrao() -> List[Linha]:
    """6 pacientes de treino, 3 de validação e 3 de teste, 2 nódulos cada.

    Diâmetros nas bordas da faixa: 12.0 entra, 30.0 não entra.
    """
    linhas = []
    diametros = [8.0, 12.0, 15.5, 29.9, 30.0, 45.0]
    for i in range(12):
        particao = "train" if i < 6 else ("validation" if i < 9 else "test")
        paciente = f"SINT-{i + 1:03d}"
        for k in range(2):
            linhas.append(Linha(f"{paciente}_n{k + 1}", paciente, particao, (i + k) % 2,
                                diametros[(2 * i + k) % len(diametros)]))
    return linhas


def _contar(linhas: List[Linha]) -> Dict[str, int]:
    return {"pacientes": len({l.paciente for l in linhas}), "nodulos": len(linhas),
            "classe_0": sum(l.alvo == 0 for l in linhas), "classe_1": sum(l.alvo == 1 for l in linhas)}


def _arquivo(caminho: Path, conteudo: bytes) -> s4.ArquivoVerificado:
    caminho.write_bytes(conteudo)
    return s4.ArquivoVerificado(caminho, hashlib.sha256(conteudo).hexdigest(),
                                hashlib.sha256(conteudo.replace(b"\r\n", b"\n")).hexdigest())


def montar(diretorio: Path, linhas: Optional[List[Linha]] = None, *, lixo_em=("test",),
           mudar_tabela: Optional[Callable[[str], str]] = None,
           mudar_split: Optional[Callable[[str], str]] = None,
           contagens: Optional[dict] = None, faixa: Optional[dict] = None,
           ordem: Optional[Callable[[List[Linha]], List[Linha]]] = None) -> dados.FonteBenchmark:
    """Grava tabela e split sintéticos e devolve a fonte com SHA-256 e contagens.

    As linhas vão para o arquivo em ordem embaralhada (ou ``ordem``), nunca
    pela ordem canônica.
    """
    linhas = linhas if linhas is not None else linhas_padrao()
    no_arquivo = ordem(linhas) if ordem else linhas[1::2][::-1] + linhas[0::2]
    cabecalho = ["nodulo_id", "patient_id", "malignancy_mean", "target_binary", VOLUME, DIAMETRO,
                 ESFERICIDADE, "original_firstorder_Mean", "exclusion_reason"]
    corpo = [",".join(cabecalho)]
    for l in no_arquivo:
        if l.particao in lixo_em:
            valores = [LIXO, LIXO, LIXO, LIXO]
        else:
            valores = [str(l.alvo), f"{100.0 + l.diametro * 3:.6f}", f"{l.diametro:.6f}", f"{0.5 + l.alvo / 4:.6f}"]
        corpo.append(",".join([l.nodulo, l.paciente, "3.5", valores[0], *valores[1:], "1.25", ""]))
    texto_tabela = "\n".join(corpo) + "\n"
    pacientes = {}
    for l in linhas:
        pacientes.setdefault(l.paciente, l.particao)
    texto_split = "patient_id,split\n" + "".join(f"{p},{s}\n" for p, s in sorted(pacientes.items()))
    if mudar_tabela:
        texto_tabela = mudar_tabela(texto_tabela)
    if mudar_split:
        texto_split = mudar_split(texto_split)
    lidas = [l for l in linhas if l.particao in ("train", "validation")]
    return dados.FonteBenchmark(
        tabela=_arquivo(diretorio / "tabela_sintetica.csv", texto_tabela.encode("utf-8")),
        split=_arquivo(diretorio / "split_sintetico.csv", texto_split.encode("utf-8")),
        contagens=contagens or {p: _contar([l for l in linhas if l.particao == p]) for p in ("train", "validation")},
        faixa=faixa or _contar([l for l in lidas if 12.0 <= l.diametro < 30.0]),
    )


def autorizacao_sintetica() -> ap.AutorizacaoDefinitiva:
    evidencia = ap.Evidencia("P9", "b" * 40, ap.DOCUMENTO_APROVACAO, "2026-11-05", SHA_CFG)
    s4_e3 = ap.Evidencia("S4_E3_REGISTRADA", "c" * 40, ap.DOCUMENTO_APROVACAO, "2026-11-08", "d" * 64)
    return ap.AutorizacaoDefinitiva("a" * 40, SHA_CFG, evidencia, s4_e3, ta.MANIFESTO)


def _falhar(*args, **kwargs):
    raise AssertionError("leitura de arquivo não permitida neste ponto")


class BaseDados(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def fonte(self, **kwargs) -> dados.FonteBenchmark:
        return montar(self.dir, **kwargs)

    def definitivo(self, fonte: dados.FonteBenchmark, **kwargs) -> dados.DadosBenchmark:
        with unittest.mock.patch.object(dados.ap, "autorizar_definitivo", return_value=autorizacao_sintetica()):
            return dados._carregar_definitivo_sintetico(ta.MANIFESTO, self.dir, fonte, **kwargs)

    def assertInvalido(self, mensagem: str, fonte_kwargs: dict, definitivo: bool = False):
        fonte = self.fonte(**fonte_kwargs)
        with self.assertRaises(dados.DadosInvalidos) as contexto:
            self.definitivo(fonte) if definitivo else dados._carregar_implementacao_sintetico(fonte)
        self.assertIn(mensagem, str(contexto.exception))


# --------------------------------------------------------------------------- #
# Modo implementacao: só train
# --------------------------------------------------------------------------- #


class TestImplementacao(BaseDados):

    def test_le_so_train_em_ordem_canonica(self):
        linhas = linhas_padrao()
        base = dados._carregar_implementacao_sintetico(self.fonte())
        treino = sorted(l.nodulo for l in linhas if l.particao == "train")
        self.assertEqual(base.modo, "implementacao")
        self.assertEqual(base.particoes, ("train",))
        self.assertIsNone(base.autorizacao)
        self.assertEqual(list(base.nodulos()), treino)
        self.assertEqual(set(base.frame()["particao"]), {"train"})
        self.assertEqual(dict(base.contagens["train"]), {"pacientes": 6, "nodulos": 12, "classe_0": 6, "classe_1": 6})
        self.assertEqual(base.amostra().n, 12)

    def test_validacao_e_teste_nunca_interpretados(self):
        base = dados._carregar_implementacao_sintetico(self.fonte(lixo_em=("test", "validation")))
        self.assertEqual(base.contagens["base"]["nodulos"], 12)
        self.assertFalse(any(n.startswith(("SINT-007", "SINT-010")) for n in base.nodulos()))

    def test_colunas_e_tipos(self):
        base = dados._carregar_implementacao_sintetico(self.fonte())
        self.assertEqual(list(base.frame().columns),
                         ["nodulo_id", "patient_id", "particao", "target_binary", *FEATURES])
        self.assertEqual(base.caracteristicas, tuple(s4.FEATURES_M1))
        self.assertEqual(base.X().dtype, np.float64)
        self.assertEqual(base.X().shape, (12, 3))
        self.assertEqual(base.y().dtype, np.int64)

    def test_alinhamento_depois_da_ordenacao(self):
        linhas = {l.nodulo: l for l in linhas_padrao()}
        base = dados._carregar_implementacao_sintetico(self.fonte())
        for nodulo, x, y, g in zip(base.nodulos(), base.X(), base.y(), base.grupos()):
            self.assertEqual((y, g, x[1]), (linhas[nodulo].alvo, linhas[nodulo].paciente, linhas[nodulo].diametro))

    def test_acessos_devolvem_copias(self):
        base = dados._carregar_implementacao_sintetico(self.fonte())
        base.frame().iloc[0, 0] = "alterado"
        base.X()[0, 0] = -1.0
        base.y()[0] = 9
        self.assertNotEqual(base.nodulos()[0], "alterado")
        self.assertGreater(base.X()[0, 0], 0)
        self.assertIn(base.y()[0], (0, 1))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            base.modo = "definitivo"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            base.contagens["train"]["nodulos"] = 1  # type: ignore[index]

    def test_segunda_fase_pula_as_outras_particoes(self):
        """A leitura de alvo e características pula, por linha física, tudo que não é train."""
        fonte = self.fonte()
        with unittest.mock.patch.object(dados.pd, "read_csv", wraps=pd.read_csv) as espiao:
            dados._carregar_implementacao_sintetico(fonte)
        segunda = [c for c in espiao.call_args_list if "target_binary" in (c.kwargs.get("usecols") or [])]
        self.assertEqual(len(segunda), 1)
        self.assertEqual(segunda[0].kwargs["usecols"], ["nodulo_id", "patient_id", "target_binary", *FEATURES])
        texto = fonte.tabela.caminho.read_text(encoding="utf-8").splitlines()
        nao_treino = [i for i, linha in enumerate(texto) if i > 0 and not linha.startswith(
            tuple(f"SINT-{n:03d}_" for n in range(1, 7)))]
        self.assertEqual(sorted(segunda[0].kwargs["skiprows"]), nao_treino)

    def test_crlf_e_sem_quebra_final_tolerados_e_alinhados(self):
        linhas = {l.nodulo: l for l in linhas_padrao()}
        casos = {"CRLF": lambda t: t.replace("\n", "\r\n"), "sem quebra final": lambda t: t.rstrip("\n")}
        for rotulo, mudar in casos.items():
            with self.subTest(rotulo):
                base = dados._carregar_implementacao_sintetico(self.fonte(mudar_tabela=mudar))
                self.assertEqual(base.contagens["train"]["nodulos"], 12)
                for nodulo, x, y in zip(base.nodulos(), base.X(), base.y()):
                    self.assertEqual((y, x[1]), (linhas[nodulo].alvo, linhas[nodulo].diametro))


# --------------------------------------------------------------------------- #
# Modo definitivo: train + validation, só com autorização
# --------------------------------------------------------------------------- #


class TestDefinitivo(BaseDados):

    def test_le_train_e_validation_sem_teste(self):
        linhas = linhas_padrao()
        base = self.definitivo(self.fonte())
        lidas = sorted(l.nodulo for l in linhas if l.particao != "test")
        self.assertEqual(base.particoes, ("train", "validation"))
        self.assertEqual(list(base.nodulos()), lidas)
        self.assertEqual(set(base.frame()["particao"]), {"train", "validation"})
        self.assertEqual(dict(base.contagens["base"]), {"pacientes": 9, "nodulos": 18, "classe_0": 9, "classe_1": 9})
        # Conferido à mão: 12.0 entra e 30.0 não entra na faixa.
        self.assertEqual(dict(base.contagens["faixa"]),
                         {"pacientes": 6, "nodulos": 9, "classe_0": 4, "classe_1": 5})
        self.assertIsInstance(base.autorizacao, ap.AutorizacaoDefinitiva)

    def test_autorizacao_chamada_com_os_parametros(self):
        fonte = self.fonte()
        with unittest.mock.patch.object(dados.ap, "autorizar_definitivo",
                                        return_value=autorizacao_sintetica()) as autorizar:
            dados._carregar_definitivo_sintetico(ta.MANIFESTO, self.dir, fonte, commit_p9="1" * 40,
                                      commit_s4_e3="2" * 40)
        autorizar.assert_called_once_with(ta.MANIFESTO, repo=self.dir, configuracao=C,
                                          commit_p9="1" * 40, commit_s4_e3="2" * 40)

    def test_faixa_divergente_aborta(self):
        self.assertInvalido("faixa", {"faixa": {"pacientes": 187, "nodulos": 226, "classe_0": 91,
                                                "classe_1": 135}}, definitivo=True)

    def test_bloqueio_da_guarda_impede_qualquer_leitura(self):
        fonte = self.fonte()
        for excecao in (ap.RegistroAusente("P9 ausente"), ap.RegistroRevogado("revogado"),
                        ap.DivergenciaAprovacao("hash")):
            with self.subTest(excecao=type(excecao).__name__), \
                    unittest.mock.patch.object(dados.ap, "autorizar_definitivo", side_effect=excecao), \
                    unittest.mock.patch.object(dados, "_ler_bytes", side_effect=_falhar) as ler:
                with self.assertRaises(type(excecao)):
                    dados._carregar_definitivo_sintetico(ta.MANIFESTO, self.dir, fonte)
                ler.assert_not_called()

    def test_autorizacao_invalida_recusada(self):
        fonte = self.fonte()
        outra = dataclasses.replace(autorizacao_sintetica(), sha256_configuracao="0" * 64)
        for retorno in (None, object(), outra):
            with self.subTest(retorno=type(retorno).__name__), \
                    unittest.mock.patch.object(dados.ap, "autorizar_definitivo", return_value=retorno), \
                    unittest.mock.patch.object(dados, "_ler_bytes", side_effect=_falhar) as ler:
                with self.assertRaises(dados.AcessoNegado):
                    dados._carregar_definitivo_sintetico(ta.MANIFESTO, self.dir, fonte)
                ler.assert_not_called()


@unittest.skipIf(ta.GIT is None, "git indisponível")
class TestDefinitivoComGuardasReais(BaseDados):
    """Guardas reais de ``aprovacoes`` num repositório git sintético."""

    def setUp(self) -> None:
        super().setUp()
        self.raiz = self.dir / "repo"
        self.raiz.mkdir()
        self.repo = ta.RepoSintetico(self.raiz)
        self.cenario = ta.Cenario(self.repo)
        self.dir_dados = self.dir / "dados"
        self.dir_dados.mkdir()

    def test_autorizado(self):
        c = self.cenario.completo()
        base = dados._carregar_definitivo_sintetico(ta.MANIFESTO, self.raiz, montar(self.dir_dados))
        self.assertEqual(base.autorizacao.head, c.c_s4)
        self.assertEqual(base.autorizacao.p9.commit, c.c_p9)
        self.assertEqual(base.particoes, ("train", "validation"))

    def test_execucao_prematura_sem_s4_e3(self):
        c = self.cenario
        c.configuracao()
        c.p9()
        fonte = montar(self.dir_dados)
        with unittest.mock.patch.object(dados, "_ler_bytes", side_effect=_falhar) as ler:
            with self.assertRaises(ap.RegistroAusente):
                dados._carregar_definitivo_sintetico(ta.MANIFESTO, self.raiz, fonte)
            ler.assert_not_called()

    def test_p9_revogado(self):
        c = self.cenario.completo()
        revogacao = ta.registro("P9", "2026-11-09", SHA_CFG, status="revogada")
        self.repo.escrever(ta.DOC, ta.documento([c.registro_p9, c.registro_s4, revogacao]))
        self.repo.commit("2026-11-09")
        with self.assertRaises(ap.RegistroRevogado):
            dados._carregar_definitivo_sintetico(ta.MANIFESTO, self.raiz, montar(self.dir_dados))


# --------------------------------------------------------------------------- #
# Corrupção e invariantes
# --------------------------------------------------------------------------- #


def _trocar_valor(nodulo: str, coluna: int, valor: str) -> Callable[[str], str]:
    def aplicar(texto: str) -> str:
        linhas = texto.split("\n")
        for i, linha in enumerate(linhas):
            if linha.startswith(nodulo + ","):
                campos = linha.split(",")
                campos[coluna] = valor
                linhas[i] = ",".join(campos)
        return "\n".join(linhas)
    return aplicar


class TestCorrupcao(BaseDados):

    def test_arquivo_alterado_depois_da_fixacao(self):
        for nome in ("tabela", "split"):
            with self.subTest(nome):
                fonte = self.fonte()
                arquivo = getattr(fonte, nome).caminho
                arquivo.write_bytes(arquivo.read_bytes() + b"\n")
                with self.assertRaises(dados.DadosInvalidos) as contexto:
                    dados._carregar_implementacao_sintetico(fonte)
                self.assertIn("SHA-256 diferente", str(contexto.exception))

    def test_arquivo_ausente(self):
        fonte = self.fonte()
        fonte.tabela.caminho.unlink()
        with self.assertRaises(dados.DadosInvalidos):
            dados._carregar_implementacao_sintetico(fonte)

    def test_leitura_seletiva_insegura(self):
        """Casos em que o skiprows do pandas 1.5.3 pularia a linha errada (linha
        vazia) ou que tornam a correspondência linha↔registro ambígua: todos
        recusados antes da segunda fase, em qualquer partição."""
        def em_linha_de_teste(trocar):
            return lambda t: "\n".join(trocar(l) if l.startswith("SINT-011_n1,") else l for l in t.split("\n"))

        casos = {
            "linha em branco": (lambda t: t.replace("\n", "\n\n", 3), "linha em branco"),
            "linha só com espaços": (lambda t: t.replace("\n", "\n   \n", 2), "linha em branco"),
            "linha em branco com CRLF": (lambda t: t.replace("\n", "\r\n").replace("\r\n", "\r\n\r\n", 2),
                                         "linha em branco"),
            "\\r isolado": (lambda t: t.replace("\n", "\r", 1), "\\r isolado"),
            "registro em duas linhas (LF entre aspas)": (lambda t: t.replace(",3.5,", ',"3\n5",', 1),
                                                         "mais de uma linha física"),
            "registro em duas linhas (CRLF entre aspas)": (
                lambda t: t.replace("\n", "\r\n").replace(",3.5,", ',"3\r\n5",', 1), "mais de uma linha física"),
            "registro multilinha numa linha de teste": (em_linha_de_teste(lambda l: l.replace(",3.5,", ',"3\n5",')),
                                                        "mais de uma linha física"),
        }
        for rotulo, (mudar, mensagem) in casos.items():
            with self.subTest(rotulo):
                fonte = self.fonte(mudar_tabela=mudar)
                with self.assertRaises(dados.DadosInvalidos) as contexto:
                    dados._carregar_implementacao_sintetico(fonte)
                self.assertIn("leitura seletiva insegura", str(contexto.exception))
                self.assertIn(mensagem, str(contexto.exception))

    def test_aspas_sem_fechamento(self):
        fonte = self.fonte(mudar_tabela=lambda t: t.replace(",3.5,", ',"3.5,', 1))
        with self.assertRaises(dados.DadosInvalidos):
            dados._carregar_implementacao_sintetico(fonte)

    def test_linha_vazia_desalinharia_o_skiprows(self):
        """Prova do risco no pandas 1.5.3 e da segunda trava: com a checagem de
        linhas físicas desligada, uma linha vazia faz o skiprows pular a linha
        errada, e a comparação de identificadores aborta."""
        fonte = self.fonte(lixo_em=(), mudar_tabela=lambda t: t.replace("\n", "\n\n", 1))
        with unittest.mock.patch.object(dados, "_linhas_fisicas", side_effect=lambda conteudo: (
                len([l for l in conteudo.decode().split("\n")[1:] if l.strip()]) + 1)):
            with self.assertRaises(dados.DadosInvalidos) as contexto:
                dados._carregar_implementacao_sintetico(fonte)
        self.assertIn("desalinhadas", str(contexto.exception))

    def test_segunda_fase_desalinhada_aborta(self):
        """Defesa em profundidade: se a leitura do alvo e das características
        devolvesse as linhas em outra ordem ou deslocadas, a comparação com os
        identificadores da primeira fase aborta."""
        fonte = self.fonte()
        ler = pd.read_csv
        casos = {
            "ordem invertida": lambda quadro: quadro.iloc[::-1].reset_index(drop=True),
            "uma linha a menos": lambda quadro: quadro.iloc[1:].reset_index(drop=True),
        }
        for rotulo, mexer in casos.items():
            with self.subTest(rotulo):
                def leitura(*args, mexer=mexer, **kwargs):
                    quadro = ler(*args, **kwargs)
                    return mexer(quadro) if "target_binary" in (kwargs.get("usecols") or []) else quadro

                with unittest.mock.patch.object(dados.pd, "read_csv", side_effect=leitura):
                    with self.assertRaises(dados.DadosInvalidos) as contexto:
                        dados._carregar_implementacao_sintetico(fonte)
                self.assertIn("desalinhadas", str(contexto.exception))

    def test_linha_de_teste_na_matriz_aborta(self):
        """Defesa em profundidade: se a máscara de leitura incluísse um paciente do
        teste, a partição rederivada do split aborta a carga."""
        fonte = self.fonte(lixo_em=())
        original = dados._ler_estrutura

        def estrutura_com_mascara_errada(*args, **kwargs):
            estrutura = original(*args, **kwargs)
            ids = estrutura.ids.copy()
            ids.loc[ids["patient_id"] == "SINT-010", "particao"] = "train"
            return dataclasses.replace(estrutura, ids=ids)

        with unittest.mock.patch.object(dados, "_ler_estrutura", side_effect=estrutura_com_mascara_errada), \
                unittest.mock.patch.object(dados, "_exigir_contagens_estruturais"):
            with self.assertRaises(dados.DadosInvalidos) as contexto:
                dados._carregar_implementacao_sintetico(fonte)
        self.assertIn("partição não autorizada", str(contexto.exception))

    def test_identificadores_e_particoes(self):
        casos = {
            "nodulo_id repetido": ({"mudar_tabela": lambda t: t.replace("SINT-002_n1,", "SINT-001_n1,", 1)},
                                   "nodulo_id repetido"),
            "identificador vazio": ({"mudar_tabela": lambda t: t.replace("SINT-002_n1,", ",", 1)},
                                    "identificador vazio"),
            "paciente fora do split": ({"mudar_split": lambda t: t.replace("SINT-003,train\n", "")},
                                       "pacientes da tabela e do split diferem"),
            "paciente a mais no split": ({"mudar_split": lambda t: t + "SINT-099,train\n"},
                                         "pacientes da tabela e do split diferem"),
            "paciente repetido no split": ({"mudar_split": lambda t: t + "SINT-003,validation\n"},
                                           "repetido"),
            "partição desconhecida": ({"mudar_split": lambda t: t.replace("SINT-011,test", "SINT-011,teste")},
                                      "partições desconhecidas"),
            "split com outras colunas": ({"mudar_split": lambda t: t.replace("patient_id,split", "paciente,split")},
                                         "exatamente as colunas"),
            "característica ausente": ({"mudar_tabela": lambda t: t.replace(ESFERICIDADE, "original_shape_Outra", 1)},
                                       "colunas ausentes"),
            "cabeçalho repetido": ({"mudar_tabela": lambda t: t.replace("original_firstorder_Mean", VOLUME, 1)},
                                   "colunas repetidas"),
            "paciente trocado de partição": ({"mudar_split": lambda t: t.replace("SINT-007,validation", "SINT-007,train")},
                                             "partição train"),
        }
        for rotulo, (kwargs, mensagem) in casos.items():
            with self.subTest(rotulo):
                self.assertInvalido(mensagem, kwargs)

    def test_valores_das_particoes_lidas(self):
        casos = {
            "alvo 2": ((_trocar_valor("SINT-001_n1", 3, "2")), "0 ou 1"),
            "alvo vazio": ((_trocar_valor("SINT-001_n1", 3, "")), "0 ou 1"),
            "característica vazia": ((_trocar_valor("SINT-002_n2", 5, "")), "NaN ou infinito"),
            "característica infinita": ((_trocar_valor("SINT-002_n2", 4, "inf")), "NaN ou infinito"),
            "característica texto": ((_trocar_valor("SINT-002_n2", 6, "abc")), "não numérico"),
        }
        for rotulo, (mudar, mensagem) in casos.items():
            with self.subTest(rotulo):
                self.assertInvalido(mensagem, {"mudar_tabela": mudar})

    def test_contagens_esperadas(self):
        padrao = {p: _contar([l for l in linhas_padrao() if l.particao == p]) for p in ("train", "validation")}
        casos = {
            "nódulos de train": {**padrao, "train": {**padrao["train"], "nodulos": 13, "classe_0": 7}},
            "classes de train": {**padrao, "train": {**padrao["train"], "classe_0": 7, "classe_1": 5}},
            "pacientes de validation (não lida)": {**padrao, "validation": {**padrao["validation"], "pacientes": 4}},
        }
        for rotulo, contagens in casos.items():
            with self.subTest(rotulo):
                self.assertInvalido("esperado", {"contagens": contagens})

    def test_classes_da_validacao_no_definitivo(self):
        padrao = {p: _contar([l for l in linhas_padrao() if l.particao == p]) for p in ("train", "validation")}
        contagens = {**padrao, "validation": {**padrao["validation"], "classe_0": 4, "classe_1": 2}}
        self.assertInvalido("partição validation", {"contagens": contagens}, definitivo=True)
        self.assertEqual(dados._carregar_implementacao_sintetico(self.fonte(contagens=contagens)).particoes, ("train",))

    def test_fonte_invalida(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(self.fonte(), contagens={"train": _contar(linhas_padrao())})
        with self.assertRaises(ValueError):
            dataclasses.replace(self.fonte(), faixa={"pacientes": 1, "nodulos": 2, "classe_0": 0, "classe_1": 1})
        with self.assertRaises(TypeError):
            dataclasses.replace(self.fonte(), tabela="tabela.csv")
        with self.assertRaises(TypeError):
            dados._carregar_implementacao_sintetico("tabela.csv")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Isolamento: fonte oficial, teste histórico, leitura de arquivos
# --------------------------------------------------------------------------- #


class TestIsolamento(BaseDados):

    def test_fonte_que_toca_caminho_oficial_e_recusada_sem_ler(self):
        sintetica = self.fonte()
        falsas = {
            "tabela em data/": dataclasses.replace(sintetica, tabela=s4.ArquivoVerificado(
                REPO_ROOT / "data" / "outra.csv", sintetica.tabela.sha256, sintetica.tabela.sha256_lf)),
            "split oficial com pin trocado": dataclasses.replace(sintetica, split=s4.ArquivoVerificado(
                s4.SPLIT_OFICIAL.caminho, sintetica.split.sha256, sintetica.split.sha256_lf)),
            "fonte oficial com contagens trocadas": dataclasses.replace(
                dados.fonte_oficial(), contagens={"train": _contar(linhas_padrao()[:4]),
                                                  "validation": _contar(linhas_padrao()[4:8])}),
            "a própria fonte oficial": dados.fonte_oficial(),
        }
        with unittest.mock.patch.object(dados, "_ler_bytes", side_effect=_falhar) as ler, \
                unittest.mock.patch.object(dados, "_verificacao_estrutural_s4", side_effect=_falhar):
            for rotulo, fonte in falsas.items():
                with self.subTest(rotulo):
                    with self.assertRaises(dados.AcessoNegado):
                        dados._carregar_implementacao_sintetico(fonte)
                    with self.assertRaises(dados.AcessoNegado):
                        dados._carregar_definitivo_sintetico(ta.MANIFESTO, self.dir, fonte)
            ler.assert_not_called()

    def test_conteudo_oficial_em_outro_caminho_e_recusado(self):
        sintetica = self.fonte()
        falsa_oficial = s4.ArquivoVerificado(REPO_ROOT / "data" / "nunca_lido.csv",
                                             sintetica.tabela.sha256, sintetica.tabela.sha256_lf)
        with unittest.mock.patch.object(dados.s4, "TABELA_OFICIAL", falsa_oficial):
            with self.assertRaises(dados.AcessoNegado) as contexto:
                dados._carregar_implementacao_sintetico(sintetica)
        self.assertIn("conteúdo oficial fora da fonte oficial", str(contexto.exception))

    def test_producao_sem_fonte_repositorio_ou_configuracao_substituiveis(self):
        """As portas públicas não aceitam fonte, repositório, configuração nem partição."""
        proibidos = {"fonte", "repo", "configuracao", "particoes"}
        self.assertEqual(list(inspect.signature(dados.carregar_implementacao).parameters), [])
        self.assertEqual(list(inspect.signature(dados.carregar_definitivo).parameters),
                         ["manifesto_s4", "commit_p9", "commit_s4_e3"])
        with unittest.mock.patch.object(dados, "_ler_bytes", side_effect=_falhar) as ler, \
                unittest.mock.patch.object(dados.ap, "autorizar_definitivo", side_effect=_falhar) as autorizar:
            for nome in proibidos:
                with self.subTest(parametro=nome):
                    with self.assertRaises(TypeError):
                        dados.carregar_implementacao(**{nome: self.fonte()})
                    with self.assertRaises(TypeError):
                        dados.carregar_definitivo(ta.MANIFESTO, **{nome: self.dir})
            ler.assert_not_called()
            autorizar.assert_not_called()

    def test_producao_usa_o_repositorio_e_a_configuracao_do_projeto(self):
        with unittest.mock.patch.object(dados, "_ler_bytes", side_effect=_falhar) as ler, \
                unittest.mock.patch.object(dados.ap, "autorizar_definitivo",
                                           side_effect=ap.RegistroAusente("sem P9")) as autorizar:
            with self.assertRaises(ap.RegistroAusente):
                dados.carregar_definitivo(ta.MANIFESTO, commit_p9="1" * 40)
            ler.assert_not_called()
        autorizar.assert_called_once_with(ta.MANIFESTO, repo=dados.REPO_ROOT, configuracao=C,
                                          commit_p9="1" * 40, commit_s4_e3=None)

    def test_resultado_sintetico_marcado_como_nao_oficial(self):
        self.assertFalse(dados._carregar_implementacao_sintetico(self.fonte()).oficial)
        self.assertFalse(self.definitivo(self.fonte()).oficial)

    def test_validation_nunca_sem_autorizacao_nem_pela_porta_interna(self):
        with unittest.mock.patch.object(dados, "_ler_bytes", side_effect=_falhar) as ler:
            for autorizacao in (None, object()):
                with self.assertRaises(dados.AcessoNegado):
                    dados._carregar(dados.MODO_DEFINITIVO, self.fonte(), False, C, autorizacao)
            ler.assert_not_called()

    def test_variantes_sinteticas_nao_usadas_por_outros_modulos(self):
        """Só testes chamam as variantes sintéticas; nenhum módulo do pacote as usa."""
        nomes = {"_carregar_implementacao_sintetico", "_carregar_definitivo_sintetico", "_carregar",
                 "FonteBenchmark", "_exigir_fonte_sintetica"}
        for caminho in (REPO_ROOT / "src" / "benchmark_s6").glob("*.py"):
            if caminho.name == "dados.py":
                continue
            arvore = ast.parse(caminho.read_text(encoding="utf-8"))
            usados = {n.attr for n in ast.walk(arvore) if isinstance(n, ast.Attribute)} | \
                     {n.id for n in ast.walk(arvore) if isinstance(n, ast.Name)} | \
                     {a.name for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom) for a in n.names}
            self.assertEqual(usados & nomes, set(), caminho.name)

    def test_definitivo_oficial_autoriza_antes_de_qualquer_leitura(self):
        ordem = []

        def registrar(nome: str, excecao: Exception):
            def efeito(*args, **kwargs):
                ordem.append(nome)
                raise excecao
            return efeito

        with unittest.mock.patch.object(dados, "_ler_bytes", side_effect=registrar("ler", AssertionError())), \
                unittest.mock.patch.object(dados, "_verificacao_estrutural_s4",
                                           side_effect=registrar("estrutura_s4", AssertionError())), \
                unittest.mock.patch.object(dados.ap, "autorizar_definitivo",
                                           side_effect=registrar("autorizar", ap.RegistroAusente("sem P9"))):
            with self.assertRaises(ap.RegistroAusente):
                dados.carregar_definitivo(ta.MANIFESTO)
        self.assertEqual(ordem, ["autorizar"])

    def test_implementacao_oficial_passa_pela_estrutura_da_s4_antes_de_ler(self):
        ordem = []

        class Parar(Exception):
            pass

        def estrutura():
            ordem.append("estrutura_s4")
            raise Parar

        with unittest.mock.patch.object(dados, "_ler_bytes", side_effect=lambda *a: ordem.append("ler") or _falhar()), \
                unittest.mock.patch.object(dados, "_verificacao_estrutural_s4", side_effect=estrutura):
            with self.assertRaises(Parar):
                dados.carregar_implementacao()
        self.assertEqual(ordem, ["estrutura_s4"])

    def test_fonte_oficial_igual_a_configuracao_e_a_s4(self):
        oficial = dados.fonte_oficial()
        self.assertEqual(oficial.tabela.caminho.resolve(), s4.TABELA_OFICIAL.caminho.resolve())
        self.assertEqual(oficial.split.sha256, C.dados.split.sha256)
        self.assertEqual(dict(oficial.contagens["train"]),
                         {"pacientes": 291, "nodulos": 417, "classe_0": 209, "classe_1": 208})
        self.assertEqual(dict(oficial.contagens["validation"]),
                         {"pacientes": 62, "nodulos": 92, "classe_0": 49, "classe_1": 43})
        self.assertEqual(dict(oficial.faixa), {"pacientes": 187, "nodulos": 226, "classe_0": 91, "classe_1": 135})

    def test_nenhuma_porta_para_o_teste(self):
        self.assertNotIn("test", {p for ps in dados.PARTICOES_POR_MODO.values() for p in ps})
        for funcao in (dados.carregar_implementacao, dados.carregar_definitivo,
                       dados._carregar_implementacao_sintetico, dados._carregar_definitivo_sintetico):
            self.assertNotIn("particoes", inspect.signature(funcao).parameters)

    def test_modulo_le_arquivos_so_por_ler_bytes(self):
        arvore = ast.parse((REPO_ROOT / "src/benchmark_s6/dados.py").read_text(encoding="utf-8"))
        chamadas = [n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
                    for n in ast.walk(arvore) if isinstance(n, ast.Call)]
        self.assertEqual(chamadas.count("read_bytes"), 1)
        for proibida in ("open", "read_text", "write_bytes", "write_text", "to_csv", "unlink"):
            self.assertNotIn(proibida, chamadas)
        # read_csv só sobre bytes já verificados, em memória (io.BytesIO).
        leituras = [n for n in ast.walk(arvore) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "read_csv"]
        self.assertEqual(len(leituras), 3)
        for n in leituras:
            self.assertTrue(isinstance(n.args[0], ast.Call) and getattr(n.args[0].func, "attr", "") == "BytesIO")

    def test_processo_nao_abre_dados_reais(self):
        """Gancho de auditoria num processo novo: importar e carregar uma fonte sintética
        não abre nada em ``data/`` nem o split oficial."""
        fonte = self.fonte()
        codigo = (
            "import json, os, sys\n"
            "abertos = []\n"
            "def gancho(evento, args):\n"
            "    if evento == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):\n"
            "        abertos.append(os.path.abspath(os.fsdecode(args[0])))\n"
            "sys.addaudithook(gancho)\n"
            "from pathlib import Path\n"
            "from src.benchmark_s6 import dados, adaptadores_s4 as s4\n"
            f"tabela = s4.ArquivoVerificado(Path({str(fonte.tabela.caminho)!r}), {fonte.tabela.sha256!r}, "
            f"{fonte.tabela.sha256_lf!r})\n"
            f"split = s4.ArquivoVerificado(Path({str(fonte.split.caminho)!r}), {fonte.split.sha256!r}, "
            f"{fonte.split.sha256_lf!r})\n"
            f"contagens = {json.dumps({p: dict(c) for p, c in fonte.contagens.items()})}\n"
            f"faixa = {json.dumps(dict(fonte.faixa))}\n"
            "base = dados._carregar_implementacao_sintetico(dados.FonteBenchmark(tabela, split, contagens, faixa))\n"
            "print(json.dumps({'n': len(base.nodulos()), 'abertos': abertos}))\n"
        )
        resultado = subprocess.run([sys.executable, "-B", "-c", codigo], cwd=REPO_ROOT, capture_output=True,
                                   text=True, check=True, timeout=120)
        relatorio = json.loads(resultado.stdout.strip().splitlines()[-1])
        self.assertEqual(relatorio["n"], 12)
        abertos = [p.lower().replace("\\", "/") for p in relatorio["abertos"]]
        self.assertTrue(any(p.endswith("tabela_sintetica.csv") for p in abertos))
        proibidos = [str(REPO_ROOT / "data").lower().replace("\\", "/") + "/",
                     str(s4.SPLIT_OFICIAL.caminho).lower().replace("\\", "/")]
        for caminho in abertos:
            self.assertFalse(any(caminho.startswith(p) for p in proibidos), caminho)


if __name__ == "__main__":
    unittest.main()
