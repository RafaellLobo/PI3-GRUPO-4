"""Testes do plano de dobras do benchmark S6 (``src/benchmark_s6/folds.py``).

Todas as amostras são SINTÉTICAS, montadas em memória com identificadores
``SINT-…``. Nenhum teste lê arquivos, ``data/`` ou dados de pacientes, e
nenhum teste treina modelos ou calcula métricas.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/benchmark_s6 -v
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import os
import subprocess
import sys
import unittest
import unittest.mock
from pathlib import Path
from types import MappingProxyType
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark_s6 import adaptadores_s4 as s4  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import dados  # noqa: E402
from src.benchmark_s6 import folds  # noqa: E402

C = cfg.CONFIGURACAO
D = C.dobras
MODELOS = tuple(m.codigo for m in C.modelos)
PACOTE = REPO_ROOT / "src" / "benchmark_s6"


def amostra_sintetica(n_pacientes: int = 150, semente: int = 0, max_nodulos: int = 3,
                      prevalencia: float = 0.5) -> s4.Amostra:
    """Pacientes ``SINT-pppp`` com 1 a ``max_nodulos`` nódulos e rótulo por nódulo, na ordem canônica."""
    rng = np.random.default_rng(semente)
    ids: List[str] = []
    y: List[int] = []
    grupos: List[str] = []
    for p in range(n_pacientes):
        for n in range(int(rng.integers(1, max_nodulos + 1))):
            ids.append(f"SINT-{p:04d}-N{n}")
            grupos.append(f"SINT-{p:04d}")
            y.append(int(rng.random() < prevalencia))
    ordem = np.argsort(ids)
    return s4.criar_amostra(np.array(ids, dtype=object)[ordem], np.array(y, dtype=np.int64)[ordem],
                            np.array(grupos, dtype=object)[ordem])


def amostra_de(ids: List[str], y: List[int], grupos: List[str]) -> s4.Amostra:
    return s4.criar_amostra(np.array(ids, dtype=object), np.array(y, dtype=np.int64), np.array(grupos, dtype=object))


def ro(valores) -> np.ndarray:
    arr = np.array(sorted(valores), dtype=np.int64)
    arr.setflags(write=False)
    return arr


def com(obj, **mudancas):
    """Cópia de um dataclass congelado com campos trocados (simula adulteração)."""
    return dataclasses.replace(obj, **mudancas)


def trocar_dobra(plano: folds.PlanoDobras, chave: Tuple[int, int], nova: folds.DobraExterna) -> folds.PlanoDobras:
    dobras = dict(plano.dobras)
    dobras[chave] = nova
    return com(plano, dobras=MappingProxyType(dobras))


class BaseFolds(unittest.TestCase):
    amostra: s4.Amostra
    impl: folds.PlanoDobras
    defin: folds.PlanoDobras

    @classmethod
    def setUpClass(cls):
        cls.amostra = amostra_sintetica()
        cls.impl = folds.gerar_plano(cls.amostra, dados.MODO_IMPLEMENTACAO)
        cls.defin = folds.gerar_plano(cls.amostra, dados.MODO_DEFINITIVO)


# --------------------------------------------------------------------------- #
# Estrutura, sementes e contrato
# --------------------------------------------------------------------------- #


class TestPlano(BaseFolds):
    def test_implementacao_so_r0(self):
        p = self.impl
        self.assertEqual((p.modo, p.repeticoes), ("implementacao", (0,)))
        self.assertEqual(list(p.dobras), [(0, k) for k in range(5)])
        for k in range(5):
            d = p.dobra(0, k)
            self.assertEqual((d.externa.semente, d.externa.dobra_externa, d.externa.dobra_interna), (42, k, None))
            self.assertEqual([i.semente for i in d.internas], [1000 + k] * 5)
            self.assertEqual([i.dobra_interna for i in d.internas], list(range(5)))
        with self.assertRaises(folds.DobrasInvalidas):
            p.dobra(1, 0)

    def test_definitivo_r0_a_r9_com_50_externas_e_250_internas(self):
        p = self.defin
        self.assertEqual(p.repeticoes, tuple(range(10)))
        self.assertEqual(len(p.dobras), 50)
        self.assertEqual(sum(len(d.internas) for d in p.dobras.values()), 250)
        for (r, k), d in p.dobras.items():
            self.assertEqual(d.externa.semente, 42 + r)
            self.assertEqual({i.semente for i in d.internas}, {1000 + 10 * r + k})
            self.assertEqual(d.externa.rotulo, f"r{r:02d}-k{k}")
        self.assertEqual(len({d.externa.semente for d in p.dobras.values()}), 10)
        self.assertEqual(len({d.internas[0].semente for d in p.dobras.values()}), 50)

    def test_sementes_e_numeros_vem_da_configuracao(self):
        self.assertEqual((D.n_dobras_externas, D.n_dobras_internas, D.n_repeticoes), (5, 5, 10))
        self.assertEqual(D.sementes_externas(), tuple(42 + r for r in range(10)))
        self.assertEqual([D.semente_interna(r, k) for r in range(10) for k in range(5)],
                         [1000 + 10 * r + k for r in range(10) for k in range(5)])
        self.assertEqual(folds.repeticoes_do_modo("implementacao"), (0,))
        self.assertEqual(folds.repeticoes_do_modo("definitivo"), tuple(range(10)))
        for modo in ("configuracao", "verificar", "DEFINITIVO", None):
            with self.assertRaises(folds.DobrasInvalidas):
                folds.repeticoes_do_modo(modo)

    def test_igual_ao_stratifiedgroupkfold_direto(self):
        """Externas e internas são exatamente as do divisor com as sementes da P9."""
        a = self.amostra
        x = np.zeros((a.n, 1))
        for r in (0, 9):
            esperadas = list(StratifiedGroupKFold(5, shuffle=True, random_state=42 + r).split(x, a.y, a.grupos))
            for k, (tr, te) in enumerate(esperadas):
                d = self.defin.dobra(r, k)
                np.testing.assert_array_equal(d.externa.treino, np.sort(tr))
                np.testing.assert_array_equal(d.externa.avaliacao, np.sort(te))
                base = d.externa.treino
                internas = StratifiedGroupKFold(5, shuffle=True, random_state=1000 + 10 * r + k).split(
                    np.zeros((len(base), 1)), a.y[base], a.grupos[base])
                for i, (itr, ite) in zip(d.internas, internas):
                    np.testing.assert_array_equal(i.treino, np.sort(base[itr]))
                    np.testing.assert_array_equal(i.avaliacao, np.sort(base[ite]))

    def test_indices_somente_leitura_e_estruturas_congeladas(self):
        d = self.defin.dobra(2, 3)
        for div in (d.externa, *d.internas):
            for lado in (div.treino, div.avaliacao):
                self.assertEqual(lado.dtype, np.int64)
                self.assertFalse(lado.flags.writeable)
                self.assertTrue(np.all(np.diff(lado) > 0))
                with self.assertRaises(ValueError):
                    lado[0] = 0
        for obj, campo in ((d.externa, "semente"), (d, "fingerprint"), (self.defin, "modo")):
            with self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(obj, campo, 0)
        with self.assertRaises(TypeError):
            self.defin.dobras[(0, 0)] = d  # type: ignore[index]
        self.assertIsInstance(d.internas, tuple)

    def test_cv_interno_em_posicoes_do_treino_externo(self):
        a = self.amostra
        d = self.defin.dobra(4, 1)
        cv = d.cv_interno()
        self.assertEqual(len(cv), 5)
        sub_ids = a.ids[d.externa.treino]
        for (tr, te), interna in zip(cv, d.internas):
            self.assertFalse(tr.flags.writeable or te.flags.writeable)
            np.testing.assert_array_equal(d.externa.treino[tr], interna.treino)
            np.testing.assert_array_equal(d.externa.treino[te], interna.avaliacao)
            self.assertEqual(set(sub_ids[tr].tolist()), set(a.ids[interna.treino].tolist()))
            self.assertTrue(np.all(te < len(d.externa.treino)))

    def test_cv_interno_recusa_interna_fora_do_treino_externo(self):
        d = self.impl.dobra(0, 0)
        fora = com(d.internas[0], avaliacao=ro([*d.internas[0].avaliacao.tolist(), int(d.externa.avaliacao[0])]))
        with self.assertRaisesRegex(folds.DobrasInvalidas, "fora do treino externo"):
            com(d, internas=(fora, *d.internas[1:])).cv_interno()

    def test_externas_da_repeticao(self):
        externas = self.defin.externas(7)
        self.assertEqual([e.dobra_externa for e in externas], list(range(5)))
        self.assertTrue(all(e.repeticao == 7 for e in externas))
        for r in (10, -1, True, "0"):
            with self.assertRaises(folds.DobrasInvalidas):
                self.defin.externas(r)

    def test_dobra_recusa_chaves_invalidas(self):
        for r, k in ((True, 0), (0, False), (0, 5), (10, 0), ("0", 0), (0.0, 0)):
            with self.assertRaises(folds.DobrasInvalidas):
                self.defin.dobra(r, k)

    def test_plano_dos_dados_segue_o_modo_da_carga(self):
        a = self.amostra
        frame = pd.DataFrame({cfg.COLUNA_NODULO: a.ids, cfg.COLUNA_PACIENTE: a.grupos,
                              dados.COLUNA_PARTICAO: "train", cfg.COLUNA_ALVO: a.y})

        def carga(modo):
            return dados.DadosBenchmark(modo, dados.PARTICOES_POR_MODO[modo], False, tuple(C.caracteristicas),
                                        MappingProxyType({}), MappingProxyType({}), None, frame)

        self.assertEqual(folds.plano_dos_dados(carga("implementacao")).fingerprint, self.impl.fingerprint)
        self.assertEqual(folds.plano_dos_dados(carga("definitivo")).fingerprint, self.defin.fingerprint)
        with self.assertRaises(TypeError):
            folds.plano_dos_dados(a)  # type: ignore[arg-type]

    def test_entradas_invalidas(self):
        with self.assertRaises(TypeError):
            folds.gerar_plano({"ids": self.amostra.ids}, "implementacao")  # type: ignore[arg-type]
        with self.assertRaises(folds.DobrasInvalidas):
            folds.gerar_plano(self.amostra, "verificar")
        with self.assertRaises(TypeError):
            folds.validar_plano(self.amostra, {"modo": "implementacao"})  # type: ignore[arg-type]

    def test_codigos_de_saida(self):
        codigos = C.execucao.codigos_saida
        self.assertEqual(codigos[folds.DobrasInvalidas.chave_saida], 3)
        self.assertEqual(codigos[folds.PlanoDivergente.chave_saida], 4)
        self.assertTrue(issubclass(folds.PlanoDivergente, folds.DobrasInvalidas))
        self.assertTrue(issubclass(folds.DobrasInvalidas, ValueError))


# --------------------------------------------------------------------------- #
# Integridade e ausência de vazamento, conferidas de forma independente
# --------------------------------------------------------------------------- #


class TestIntegridade(BaseFolds):
    def test_externas_disjuntas_por_paciente_com_duas_classes_e_cobertura_integral(self):
        a = self.amostra
        for r in self.defin.repeticoes:
            avaliadas = []
            for e in self.defin.externas(r):
                tr, te = set(e.treino.tolist()), set(e.avaliacao.tolist())
                self.assertTrue(tr and te)
                self.assertEqual(tr & te, set())
                self.assertEqual(tr | te, set(range(a.n)))
                self.assertEqual(set(a.grupos[e.treino]) & set(a.grupos[e.avaliacao]), set())
                self.assertEqual(set(a.y[e.treino]), {0, 1})
                self.assertEqual(set(a.y[e.avaliacao]), {0, 1})
                avaliadas.extend(te)
            self.assertEqual(sorted(avaliadas), list(range(a.n)), f"r{r}: cada linha avaliada uma vez")

    def test_internas_cobrem_o_treino_externo_sem_vazamento(self):
        a = self.amostra
        for (r, k), d in self.defin.dobras.items():
            aval_ext = set(d.externa.avaliacao.tolist())
            pac_ext = set(a.grupos[d.externa.avaliacao].tolist())
            avaliadas = []
            for i in d.internas:
                usadas = set(i.treino.tolist()) | set(i.avaliacao.tolist())
                self.assertEqual(usadas, set(d.externa.treino.tolist()))
                self.assertEqual(usadas & aval_ext, set(), f"r{r} k{k}: linha da avaliação externa")
                self.assertEqual(set(a.grupos[sorted(usadas)].tolist()) & pac_ext, set(),
                                 f"r{r} k{k}: paciente da avaliação externa")
                self.assertEqual(set(a.grupos[i.treino]) & set(a.grupos[i.avaliacao]), set())
                self.assertEqual(set(a.y[i.treino]), {0, 1})
                self.assertEqual(set(a.y[i.avaliacao]), {0, 1})
                avaliadas.extend(i.avaliacao.tolist())
            self.assertEqual(sorted(avaliadas), d.externa.treino.tolist())

    def test_fingerprints_compativeis_com_a_api_da_s4(self):
        a = self.amostra
        d = self.defin.dobra(5, 4)
        composicao = {}
        for div in (d.externa, *d.internas):
            composicao[f"{div.rotulo}/treino"] = list(a.ids[div.treino])
            composicao[f"{div.rotulo}/avaliacao"] = list(a.ids[div.avaliacao])
        self.assertEqual(d.fingerprint, s4.fingerprint_folds(composicao))
        self.assertRegex(d.fingerprint, r"^[0-9a-f]{64}$")
        resumo = self.defin.resumo(a)
        item = next(x for x in resumo["dobras"] if (x["repeticao"], x["dobra_externa"]) == (5, 4))
        self.assertEqual(item["externa"]["fingerprint"], s4.fingerprint_folds(
            {"treino": a.ids[d.externa.treino].tolist(), "avaliacao": a.ids[d.externa.avaliacao].tolist()}))
        self.assertEqual(item["fingerprint_dobra"], d.fingerprint)
        # A ordem das linhas não importa para a S4: o fingerprint depende só da composição.
        invertida = {p: list(reversed(v)) for p, v in composicao.items()}
        self.assertEqual(s4.fingerprint_folds(invertida), d.fingerprint)

    def test_resumo_sem_identificadores_e_com_contagens_agregadas(self):
        a = self.amostra
        resumo = self.defin.resumo(a)
        texto = s4.json_canonico(resumo)
        self.assertNotIn("SINT-", texto)
        self.assertNotIn(a.sha256, texto)
        self.assertNotIn("sha256_amostra", texto)
        self.assertEqual(resumo["fingerprint_plano"], self.defin.fingerprint)
        self.assertEqual(resumo["sha256_configuracao"], cfg.sha256_configuracao())
        self.assertEqual(len(resumo["dobras"]), 50)
        n0, n1 = int(np.sum(a.y == 0)), int(np.sum(a.y == 1))
        for item in resumo["dobras"]:
            ext = item["externa"]
            self.assertEqual(ext["treino"]["nodulos"] + ext["avaliacao"]["nodulos"], a.n)
            self.assertEqual(ext["treino"]["classe_0"] + ext["avaliacao"]["classe_0"], n0)
            self.assertEqual(ext["treino"]["classe_1"] + ext["avaliacao"]["classe_1"], n1)
            self.assertEqual(ext["treino"]["pacientes"] + ext["avaliacao"]["pacientes"], len(set(a.grupos)))
            self.assertEqual(len(item["internas"]), 5)
            for i in item["internas"]:
                self.assertEqual(i["treino"]["nodulos"] + i["avaliacao"]["nodulos"], ext["treino"]["nodulos"])
        json.loads(texto)

    def test_resumo_exige_a_amostra_do_plano(self):
        with self.assertRaisesRegex(folds.DobrasInvalidas, "outra amostra"):
            self.defin.resumo(amostra_sintetica(semente=1))


# --------------------------------------------------------------------------- #
# Determinismo
# --------------------------------------------------------------------------- #


class TestDeterminismo(BaseFolds):
    def test_mesma_amostra_mesmo_plano(self):
        outro = folds.gerar_plano(self.amostra, "definitivo")
        self.assertEqual(outro.fingerprint, self.defin.fingerprint)
        for chave, d in self.defin.dobras.items():
            o = outro.dobras[chave]
            self.assertEqual(o.fingerprint, d.fingerprint)
            np.testing.assert_array_equal(o.externa.avaliacao, d.externa.avaliacao)
            for i, j in zip(o.internas, d.internas):
                np.testing.assert_array_equal(i.avaliacao, j.avaliacao)

    def test_mesmo_plano_em_outro_processo_e_outra_semente_de_hash(self):
        codigo = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(AQUI)!r})\n"
            "import test_folds as t\n"
            "from src.benchmark_s6 import folds\n"
            "print(json.dumps(folds.gerar_plano(t.amostra_sintetica(), 'definitivo').fingerprint))\n"
        )
        for semente_hash in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=semente_hash)
            saida = subprocess.run([sys.executable, "-B", "-c", codigo], cwd=REPO_ROOT, env=env,
                                   capture_output=True, text=True, check=True, timeout=300)
            self.assertEqual(json.loads(saida.stdout.strip().splitlines()[-1]), self.defin.fingerprint)

    def test_implementacao_e_o_r0_do_definitivo_na_mesma_amostra(self):
        for k in range(5):
            self.assertEqual(self.impl.dobra(0, k).fingerprint, self.defin.dobra(0, k).fingerprint)
        self.assertNotEqual(self.impl.fingerprint, self.defin.fingerprint)

    def test_amostra_diferente_muda_o_plano(self):
        a = self.amostra
        y = a.y.copy()
        y[0] = 1 - y[0]
        alterada = s4.criar_amostra(a.ids, y, a.grupos)
        self.assertNotEqual(alterada.sha256, a.sha256)
        with self.assertRaisesRegex(folds.DobrasInvalidas, "outra amostra"):
            folds.validar_plano(alterada, self.defin)
        outra = folds.gerar_plano(amostra_sintetica(semente=7), "implementacao")
        self.assertNotEqual(outra.fingerprint, self.impl.fingerprint)


# --------------------------------------------------------------------------- #
# Mesmas dobras para todos os modelos
# --------------------------------------------------------------------------- #


class TestReusoPorModelos(BaseFolds):
    def test_mesmo_objeto_e_fingerprint_para_os_cinco_modelos(self):
        self.assertEqual(MODELOS, ("lr_fixa", "lr_tuning", "svm", "rf", "xgb"))
        for (r, k) in self.defin.dobras:
            vistas = [folds.dobra_para_modelo(self.defin, r, k, m) for m in MODELOS]
            self.assertTrue(all(v is vistas[0] for v in vistas))
            self.assertEqual({v.fingerprint for v in vistas}, {self.defin.dobra(r, k).fingerprint})

    def test_modelo_fora_da_configuracao_recusado(self):
        for modelo in ("lr", "XGB", "", None):
            with self.assertRaises(folds.DobrasInvalidas):
                folds.dobra_para_modelo(self.impl, 0, 0, modelo)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            folds.dobra_para_modelo(self.impl.dobras, 0, 0, "svm")  # type: ignore[arg-type]

    def test_geracao_nao_depende_de_modelo(self):
        for funcao in (folds.gerar_plano, folds.plano_dos_dados, folds.validar_plano, folds.verificar_plano):
            parametros = set(inspect.signature(funcao).parameters)
            self.assertFalse(parametros & {"modelo", "modelos", "configuracao", "semente", "random_state"},
                             funcao.__name__)

    def test_so_folds_py_divide_dados_no_pacote(self):
        """Nenhum outro módulo cria divisores nem chama ``split``: não há divisão por modelo."""
        divisores = {"StratifiedGroupKFold", "GroupKFold", "StratifiedKFold", "KFold", "ShuffleSplit",
                     "GroupShuffleSplit", "StratifiedShuffleSplit", "train_test_split", "LeaveOneGroupOut"}
        for caminho in PACOTE.glob("*.py"):
            arvore = ast.parse(caminho.read_text(encoding="utf-8"))
            nomes = {n.id for n in ast.walk(arvore) if isinstance(n, ast.Name)} | \
                    {n.attr for n in ast.walk(arvore) if isinstance(n, ast.Attribute)} | \
                    {a.name for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom) for a in n.names}
            esperado = {"StratifiedGroupKFold"} if caminho.name == "folds.py" else set()
            self.assertEqual(nomes & divisores, esperado, caminho.name)


# --------------------------------------------------------------------------- #
# Falhas
# --------------------------------------------------------------------------- #


def _particoes(*pares) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [(ro(tr), ro(te)) for tr, te in pares]


class TestFalhas(unittest.TestCase):
    #: 6 pacientes, 2 nódulos cada: P0–P2 só classe 0, P3–P5 só classe 1.
    pequena = amostra_de([f"SINT-{i:02d}" for i in range(12)], [0] * 6 + [1] * 6,
                         [f"P{i // 2}" for i in range(12)])
    base = ro(range(12))

    def validar(self, partes, n=2):
        folds._validar_nivel(self.pequena, self.base, partes, n, "teste")

    def test_divisao_valida_passa(self):
        self.validar(_particoes(([0, 1, 2, 3, 6, 7, 8, 9], [4, 5, 10, 11]),
                                ([4, 5, 10, 11, 2, 3, 8, 9], [0, 1, 6, 7]),
                                ([0, 1, 4, 5, 6, 7, 10, 11], [2, 3, 8, 9])), n=3)

    def test_numero_de_dobras(self):
        with self.assertRaisesRegex(folds.DobrasInvalidas, "esperadas 3"):
            self.validar(_particoes(([0, 1, 2, 3, 6, 7, 8, 9], [4, 5, 10, 11])), n=3)

    def test_lado_vazio(self):
        vazio = np.array([], dtype=np.int64)
        vazio.setflags(write=False)
        with self.assertRaisesRegex(folds.DobrasInvalidas, "vazio"):
            self.validar([(ro(range(12)), vazio)], n=1)

    def test_linha_nos_dois_lados(self):
        with self.assertRaisesRegex(folds.DobrasInvalidas, "linha em treino e avaliação"):
            self.validar(_particoes((range(12), [4, 5, 10, 11])), n=1)

    def test_cobertura_do_nivel(self):
        with self.assertRaisesRegex(folds.DobrasInvalidas, "difere das linhas do nível"):
            self.validar(_particoes(([0, 1, 2, 3, 6, 7, 8], [4, 5, 10, 11])), n=1)

    def test_paciente_nos_dois_lados(self):
        with self.assertRaisesRegex(folds.DobrasInvalidas, "paciente em treino e avaliação"):
            self.validar(_particoes(([0, 2, 3, 4, 5, 6, 7, 8, 9], [1, 10, 11])), n=1)

    def test_lado_sem_as_duas_classes(self):
        with self.assertRaisesRegex(folds.DobrasInvalidas, "avaliação sem as duas classes"):
            self.validar(_particoes(([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11])), n=1)

    def test_linha_avaliada_mais_de_uma_vez(self):
        with self.assertRaisesRegex(folds.DobrasInvalidas, "exatamente uma vez"):
            self.validar(_particoes(([0, 1, 2, 3, 6, 7, 8, 9], [4, 5, 10, 11]),
                                    ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5, 10, 11])))

    def test_indices_graváveis_ou_fora_de_ordem_recusados(self):
        gravavel = np.array([4, 5, 10, 11], dtype=np.int64)
        with self.assertRaisesRegex(folds.DobrasInvalidas, "índices inválidos"):
            self.validar([(ro([0, 1, 2, 3, 6, 7, 8, 9]), gravavel)], n=1)
        fora = np.array([5, 4, 10, 11], dtype=np.int64)
        fora.setflags(write=False)
        with self.assertRaisesRegex(folds.DobrasInvalidas, "índices inválidos"):
            self.validar([(ro([0, 1, 2, 3, 6, 7, 8, 9]), fora)], n=1)

    def test_vazamento_de_linha_e_de_paciente_da_avaliacao_externa(self):
        a = self.pequena
        externa = folds.Divisao("externo", 0, 0, None, 42, ro([0, 1, 2, 3, 6, 7, 8, 9]), ro([4, 5, 10, 11]))
        limpa = folds.Divisao("interno", 0, 0, 0, 1000, ro([0, 1, 6, 7]), ro([2, 3, 8, 9]))
        folds._exigir_sem_vazamento(a, externa, (limpa,))
        linha = folds.Divisao("interno", 0, 0, 0, 1000, ro([0, 1, 6, 7]), ro([2, 3, 4, 8, 9]))
        with self.assertRaisesRegex(folds.DobrasInvalidas, "avaliação externa"):
            folds._exigir_sem_vazamento(a, externa, (linha,))
        # P2 tem as linhas 4, 5 e 10. Com a avaliação externa em 4 e 5, uma interna com a linha 10
        # vaza pelo PACIENTE, mesmo sem repetir nenhuma linha da avaliação externa.
        b = amostra_de([f"SINT-{i:02d}" for i in range(12)], [0] * 6 + [1] * 6,
                       ["P0", "P0", "P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4", "P2", "P5"])
        externa_b = folds.Divisao("externo", 0, 0, None, 42, ro([0, 1, 2, 3, 6, 7, 8, 9, 11]), ro([4, 5, 10]))
        paciente = folds.Divisao("interno", 0, 0, 0, 1000, ro([0, 1, 6, 7]), ro([2, 3, 8, 9, 11]))
        folds._exigir_sem_vazamento(b, externa_b, (paciente,))
        with self.assertRaisesRegex(folds.DobrasInvalidas, "avaliação externa"):
            folds._exigir_sem_vazamento(b, com(externa_b, avaliacao=ro([4, 5])),
                                        (com(paciente, avaliacao=ro([2, 3, 8, 9, 10, 11])),))

    def test_amostra_menor_que_o_numero_de_dobras(self):
        a = amostra_de(["SINT-0", "SINT-1", "SINT-2", "SINT-3"], [0, 1, 0, 1], ["P0", "P1", "P2", "P3"])
        with self.assertRaisesRegex(folds.DobrasInvalidas, "divisor recusou"):
            folds.gerar_plano(a, "implementacao")

    def test_poucos_pacientes_deixam_dobra_sem_classe(self):
        with self.assertRaisesRegex(folds.DobrasInvalidas, "sem as duas classes"):
            folds.gerar_plano(self.pequena, "implementacao")

    def test_aviso_do_divisor_e_erro(self):
        """Classe com menos membros que dobras: o aviso do divisor vira falha técnica."""
        ids = [f"SINT-{i:03d}" for i in range(60)]
        y = [1 if i in (5, 30, 55) else 0 for i in range(60)]
        a = amostra_de(ids, y, [f"P{i:03d}" for i in range(60)])
        with self.assertRaisesRegex(folds.DobrasInvalidas, "divisor recusou"):
            folds.gerar_plano(a, "implementacao")

    def test_paciente_concentrando_a_classe_deixa_lado_sem_classe(self):
        """Todos os positivos em 3 pacientes com muitos nódulos: alguma dobra fica sem positivo."""
        ids, y, g = [], [], []
        for p in range(40):
            nodulos = 6 if p < 3 else 1
            for n in range(nodulos):
                ids.append(f"SINT-{p:03d}-{n}")
                g.append(f"P{p:03d}")
                y.append(1 if p < 3 else 0)
        with self.assertRaises(folds.DobrasInvalidas):
            folds.gerar_plano(amostra_de(ids, y, g), "implementacao")


class TestPlanoAdulterado(BaseFolds):
    def test_plano_gerado_valida_e_verifica(self):
        self.assertIs(folds.validar_plano(self.amostra, self.defin), self.defin)
        self.assertIs(folds.verificar_plano(self.amostra, self.impl), self.impl)

    def test_linha_trocada_entre_treino_e_avaliacao_externa(self):
        d = self.impl.dobra(0, 2)
        movida = int(d.externa.avaliacao[0])
        externa = com(d.externa, treino=ro([*d.externa.treino.tolist(), movida]),
                      avaliacao=ro(d.externa.avaliacao.tolist()[1:]))
        with self.assertRaises(folds.DobrasInvalidas):
            folds.validar_plano(self.amostra, trocar_dobra(self.impl, (0, 2), com(d, externa=externa)))

    def test_interna_com_linha_da_avaliacao_externa(self):
        d = self.impl.dobra(0, 1)
        intrusa = int(d.externa.avaliacao[0])
        interna = com(d.internas[0], avaliacao=ro([*d.internas[0].avaliacao.tolist(), intrusa]))
        with self.assertRaises(folds.DobrasInvalidas):
            folds.validar_plano(self.amostra, trocar_dobra(self.impl, (0, 1),
                                                           com(d, internas=(interna, *d.internas[1:]))))

    def test_semente_ou_metadados_adulterados(self):
        d = self.defin.dobra(3, 0)
        casos = {
            "semente externa": com(d, externa=com(d.externa, semente=43)),
            "semente interna": com(d, internas=(com(d.internas[0], semente=1031), *d.internas[1:])),
            "nível": com(d, externa=com(d.externa, nivel="interno")),
            "dobra interna fora de ordem": com(d, internas=(d.internas[1], d.internas[0], *d.internas[2:])),
            "internas a menos": com(d, internas=d.internas[:4]),
            "amostra": com(d, sha256_amostra="0" * 64),
        }
        for nome, adulterada in casos.items():
            with self.subTest(nome), self.assertRaises(folds.DobrasInvalidas):
                folds.validar_plano(self.amostra, trocar_dobra(self.defin, (3, 0), adulterada))

    def test_fingerprints_adulterados(self):
        d = self.defin.dobra(1, 1)
        with self.assertRaisesRegex(folds.PlanoDivergente, "fingerprint da dobra"):
            folds.validar_plano(self.amostra, trocar_dobra(self.defin, (1, 1), com(d, fingerprint="0" * 64)))
        with self.assertRaisesRegex(folds.PlanoDivergente, "fingerprint do plano"):
            folds.validar_plano(self.amostra, com(self.defin, fingerprint="0" * 64))

    def test_dobras_trocadas_de_lugar_ou_faltando(self):
        dobras = dict(self.impl.dobras)
        dobras[(0, 0)], dobras[(0, 1)] = dobras[(0, 1)], dobras[(0, 0)]
        with self.assertRaises(folds.DobrasInvalidas):
            folds.validar_plano(self.amostra, com(self.impl, dobras=MappingProxyType(dobras)))
        incompleto = {c: d for c, d in self.impl.dobras.items() if c != (0, 4)}
        with self.assertRaisesRegex(folds.DobrasInvalidas, "incompletas"):
            folds.validar_plano(self.amostra, com(self.impl, dobras=MappingProxyType(incompleto)))

    def test_modo_e_repeticoes_incoerentes(self):
        with self.assertRaisesRegex(folds.DobrasInvalidas, "repetições do plano"):
            folds.validar_plano(self.amostra, com(self.impl, modo="definitivo"))
        with self.assertRaisesRegex(folds.DobrasInvalidas, "repetições do plano"):
            folds.validar_plano(self.amostra, com(self.defin, repeticoes=(0,)))

    def test_outra_configuracao(self):
        with self.assertRaisesRegex(folds.PlanoDivergente, "outra configuração"):
            folds.validar_plano(self.amostra, com(self.impl, sha256_configuracao="0" * 64))

    def test_plano_coerente_mas_gerado_com_outras_divisoes_so_o_verificar_pega(self):
        """Divisões de outro divisor, com metadados e fingerprints recalculados: só regerar detecta."""

        class DivisorDeslocado(StratifiedGroupKFold):
            def __init__(self, n_splits, shuffle, random_state):
                super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state + 1)

        with unittest.mock.patch.object(folds, "StratifiedGroupKFold", DivisorDeslocado):
            forjado = folds.gerar_plano(self.amostra, "implementacao")
        self.assertNotEqual(forjado.fingerprint, self.impl.fingerprint)
        folds.validar_plano(self.amostra, forjado)
        with self.assertRaisesRegex(folds.PlanoDivergente, "regerado"):
            folds.verificar_plano(self.amostra, forjado)

    def test_dobra_valida_registrada_na_posicao_de_outra(self):
        """Cada dobra é válida isoladamente; só o plano detecta a troca de posição."""
        dobras = dict(self.impl.dobras)
        dobras[(0, 0)], dobras[(0, 1)] = dobras[(0, 1)], dobras[(0, 0)]
        for d in dobras.values():
            folds.validar_dobra(self.amostra, d)
        with self.assertRaisesRegex(folds.DobrasInvalidas, "posição de outra"):
            folds.validar_plano(self.amostra, com(self.impl, dobras=MappingProxyType(dobras)))

    def test_repeticao_sem_cobertura_com_dobras_validas_isoladamente(self):
        """A dobra 1 vira cópia coerente da 0 (metadados, sementes e fingerprint de k=1): só o plano pega."""
        d0 = self.impl.dobra(0, 0)
        internas = tuple(com(i, dobra_externa=1, semente=D.semente_interna(0, 1)) for i in d0.internas)
        copia = com(d0, externa=com(d0.externa, dobra_externa=1), internas=internas)
        copia = com(copia, fingerprint=folds._fingerprint_dobra(self.amostra, copia.externa, copia.internas))
        folds.validar_dobra(self.amostra, copia)
        with self.assertRaisesRegex(folds.DobrasInvalidas, "exatamente uma vez"):
            folds.validar_plano(self.amostra, trocar_dobra(self.impl, (0, 1), copia))


# --------------------------------------------------------------------------- #
# Validação pública de uma dobra isolada
# --------------------------------------------------------------------------- #


def ro_array(valores) -> np.ndarray:
    arr = np.array(valores, dtype=np.int64)
    arr.setflags(write=False)
    return arr


class TestValidarDobra(BaseFolds):
    def coerente(self, dobra: folds.DobraExterna) -> folds.DobraExterna:
        """Recalcula o fingerprint: a adulteração só pode ser pega pelas regras estruturais."""
        return com(dobra, fingerprint=folds._fingerprint_dobra(self.amostra, dobra.externa, dobra.internas))

    def recusa(self, dobra, mensagem: str = "", tipo=folds.DobrasInvalidas):
        with self.assertRaisesRegex(tipo, mensagem):
            folds.validar_dobra(self.amostra, dobra)

    def test_todas_as_dobras_geradas_passam_e_sao_devolvidas(self):
        for plano in (self.impl, self.defin):
            for d in plano.dobras.values():
                self.assertIs(folds.validar_dobra(self.amostra, d), d)

    def test_copia_por_pickle_perde_o_somente_leitura_e_e_recusada(self):
        """Limitação registrada para os futuros workers: nenhuma dobra copiada entra sem validação."""
        import pickle

        d = self.defin.dobra(7, 3)
        copia = pickle.loads(pickle.dumps(d))
        self.assertEqual(copia.fingerprint, d.fingerprint)
        self.assertTrue(copia.externa.treino.flags.writeable)
        self.recusa(copia, "índices inválidos")

    def test_tipos_invalidos(self):
        d = self.impl.dobra(0, 0)
        with self.assertRaises(TypeError):
            folds.validar_dobra(self.amostra, self.impl)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            folds.validar_dobra({"ids": self.amostra.ids}, d)  # type: ignore[arg-type]
        self.recusa(com(d, externa=d.externa.treino), "malformadas")
        self.recusa(com(d, internas=list(d.internas)), "malformadas")
        self.recusa(com(d, internas=(*d.internas[:4], d.externa.treino)), "malformadas")

    def test_outra_amostra(self):
        outra = amostra_sintetica(semente=3)
        with self.assertRaisesRegex(folds.DobrasInvalidas, "outra amostra"):
            folds.validar_dobra(outra, self.impl.dobra(0, 0))

    def test_metadados_e_sementes(self):
        d = self.defin.dobra(4, 2)
        casos = {
            "r fora do plano": com(d, externa=com(d.externa, repeticao=10)),
            "k fora do plano": com(d, externa=com(d.externa, dobra_externa=5)),
            "r booleano": com(d, externa=com(d.externa, repeticao=True)),
            "nível externo": com(d, externa=com(d.externa, nivel="interno")),
            "externa com índice interno": com(d, externa=com(d.externa, dobra_interna=0)),
            "semente externa": com(d, externa=com(d.externa, semente=42)),
            "semente interna": com(d, internas=(com(d.internas[0], semente=1000 + 40 + 3), *d.internas[1:])),
            "interna de outra dobra": com(d, internas=(com(d.internas[0], dobra_externa=1), *d.internas[1:])),
            "internas fora de ordem": com(d, internas=(d.internas[1], d.internas[0], *d.internas[2:])),
            "internas a menos": com(d, internas=d.internas[:4]),
        }
        for nome, adulterada in casos.items():
            with self.subTest(nome):
                self.recusa(self.coerente(adulterada) if nome != "r booleano" else adulterada)

    def test_fingerprint_divergente(self):
        d = self.impl.dobra(0, 3)
        self.recusa(com(d, fingerprint="0" * 64), "fingerprint da dobra", folds.PlanoDivergente)

    def test_externa_com_linha_nos_dois_lados(self):
        d = self.impl.dobra(0, 1)
        linha = int(d.externa.treino[0])
        avaliacao = ro_array(sorted([*d.externa.avaliacao.tolist(), linha]))
        self.recusa(self.coerente(com(d, externa=com(d.externa, avaliacao=avaliacao))), "linha em treino e avaliação")

    def test_externa_com_paciente_nos_dois_lados(self):
        a = self.amostra
        d = self.impl.dobra(0, 2)
        aval = d.externa.avaliacao.tolist()
        paciente = next(g for g in a.grupos[d.externa.avaliacao] if (a.grupos[d.externa.avaliacao] == g).sum() > 1)
        linha = next(i for i in aval if a.grupos[i] == paciente)
        externa = com(d.externa, treino=ro_array(sorted([*d.externa.treino.tolist(), linha])),
                      avaliacao=ro_array([i for i in aval if i != linha]))
        self.recusa(self.coerente(com(d, externa=externa)), "paciente em treino e avaliação")

    def test_externa_sem_as_duas_classes(self):
        a = self.amostra
        d = self.impl.dobra(0, 4)
        so_positivos = ro_array([i for i in d.externa.avaliacao.tolist() if a.y[i] == 1])
        self.recusa(self.coerente(com(d, externa=com(d.externa, avaliacao=so_positivos))),
                    "avaliação sem as duas classes")

    def test_indices_externos_gravaveis_ou_fora_de_ordem(self):
        """Só a divisão externa adulterada (internas íntegras, fingerprint coerente)."""
        d = self.impl.dobra(0, 3)
        gravavel = com(d.externa, treino=np.array(d.externa.treino))
        self.recusa(self.coerente(com(d, externa=gravavel)), "índices inválidos")
        fora_de_ordem = com(d.externa, avaliacao=ro_array(d.externa.avaliacao.tolist()[::-1]))
        self.recusa(self.coerente(com(d, externa=fora_de_ordem)), "índices inválidos")

    def test_internas_com_vazamento_ou_sem_cobertura(self):
        d = self.impl.dobra(0, 0)
        intrusa = int(d.externa.avaliacao[0])
        vazada = com(d.internas[0], avaliacao=ro_array(sorted([*d.internas[0].avaliacao.tolist(), intrusa])))
        self.recusa(self.coerente(com(d, internas=(vazada, *d.internas[1:]))))
        repetida = com(d.internas[0], dobra_interna=1)
        self.recusa(self.coerente(com(d, internas=(d.internas[0], repetida, *d.internas[2:]))), "exatamente uma vez")
        gravavel = com(d.internas[2], treino=np.array(d.internas[2].treino))
        self.recusa(self.coerente(com(d, internas=(*d.internas[:2], gravavel, *d.internas[3:]))), "índices inválidos")


# --------------------------------------------------------------------------- #
# Isolamento do módulo
# --------------------------------------------------------------------------- #


class TestIsolamento(unittest.TestCase):
    def setUp(self):
        self.arvore = ast.parse((PACOTE / "folds.py").read_text(encoding="utf-8"))

    def test_importa_so_o_permitido(self):
        modulos = set()
        for n in ast.walk(self.arvore):
            if isinstance(n, ast.Import):
                modulos |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                modulos.add(("." * n.level) + (n.module or ""))
        self.assertEqual(modulos, {"__future__", "warnings", "dataclasses", "types", "typing", "numpy",
                                   "sklearn.model_selection", ".", ".dados"})
        do_pacote = {a.name for n in ast.walk(self.arvore) if isinstance(n, ast.ImportFrom) and n.level
                     for a in n.names}
        self.assertEqual(do_pacote, {"adaptadores_s4", "config", "MODO_DEFINITIVO", "MODO_IMPLEMENTACAO",
                                     "DadosBenchmark"})

    def test_nao_le_arquivos_nem_treina_nem_mede(self):
        proibidos = {"open", "read_csv", "read_bytes", "read_text", "fit", "predict", "predict_proba",
                     "decision_function", "roc_auc_score", "GridSearchCV", "_carregar", "carregar_definitivo",
                     "carregar_implementacao", "_ler_bytes", "verificar_estrutura"}
        nomes = {n.id for n in ast.walk(self.arvore) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(self.arvore) if isinstance(n, ast.Attribute)}
        self.assertEqual(nomes & proibidos, set())

    def test_processo_nao_abre_arquivos_de_dados(self):
        """Gancho de auditoria: gerar e verificar um plano não abre nenhum arquivo de dados."""
        codigo = (
            "import json, os, sys\n"
            f"sys.path.insert(0, {str(AQUI)!r})\n"
            "import test_folds as t\n"
            "from src.benchmark_s6 import folds\n"
            "a = t.amostra_sintetica(60)\n"
            "abertos = []\n"
            "def gancho(evento, args):\n"
            "    if evento == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):\n"
            "        abertos.append(os.fsdecode(args[0]))\n"
            "sys.addaudithook(gancho)\n"
            "p = folds.gerar_plano(a, 'definitivo')\n"
            "folds.verificar_plano(a, p)\n"
            "p.resumo(a)\n"
            "print(json.dumps(abertos))\n"
        )
        saida = subprocess.run([sys.executable, "-B", "-c", codigo], cwd=REPO_ROOT, capture_output=True,
                               text=True, check=True, timeout=300)
        abertos = [str(Path(p).resolve()).lower() for p in json.loads(saida.stdout.strip().splitlines()[-1])]
        proibidos = [str(REPO_ROOT / pasta).lower() for pasta in ("data", "reports")]
        for caminho in abertos:
            self.assertFalse(caminho.startswith(tuple(proibidos)), caminho)
            self.assertFalse(caminho.endswith((".csv", ".json", ".parquet", ".npy")), caminho)


if __name__ == "__main__":
    unittest.main()
