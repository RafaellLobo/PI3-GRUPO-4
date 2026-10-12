"""Testes dos folds agrupados por paciente (``src/incremental/folds.py``).

Somente dados sintéticos: pacientes fictícios (LIDC-IDRI-9xxx), de 1 a 4
nódulos por paciente e rótulos que variam dentro do mesmo paciente. Nenhuma
leitura da base oficial, nenhum treino.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/incremental -v
"""

from __future__ import annotations

import ast
import os
import random
import re
import subprocess
import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
for caminho in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.incremental import checkpoints as ck  # noqa: E402
from src.incremental import config as cfg  # noqa: E402
from src.incremental import folds as fl  # noqa: E402

N_EXT = cfg.ESPEC_CV.n_folds_externos
N_EST = cfg.ESPEC_SELECAO.n_folds_estabilidade
N_INT = cfg.ESPEC_SELECAO.n_folds_interna


def sintetico(n_pacientes=120, semente=7):
    """ids na ordem canônica, y com classes mistas por paciente e grupos = paciente."""
    gerador = random.Random(semente)
    linhas = []
    for p in range(n_pacientes):
        paciente = f"LIDC-IDRI-{9000 + p:04d}"
        propensao = gerador.random()
        for k in range(1, gerador.randint(1, 4) + 1):
            linhas.append((f"{paciente}_N{k:02d}_scan{9000 + p}", int(gerador.random() < propensao), paciente))
    linhas.sort()
    ids, y, grupos = zip(*linhas)
    return list(ids), np.array(y, dtype=np.int64), list(grupos)


IDS, Y, GRUPOS = sintetico()
AMOSTRA = fl.criar_amostra(IDS, Y, GRUPOS)


def particao_valida(teste, amostra, base, divisoes, n):
    """Invariantes de um nível: disjunção, cobertura, pacientes, classes, avaliação única."""
    teste.assertEqual(len(divisoes), n)
    avaliadas = []
    for d in divisoes:
        tr, te = d.treino, d.avaliacao
        teste.assertFalse(set(tr) & set(te))
        teste.assertEqual(set(tr) | set(te), set(base))
        teste.assertFalse(set(amostra.grupos[tr]) & set(amostra.grupos[te]), d.rotulo)
        teste.assertEqual(set(amostra.y[tr]), {0, 1})
        teste.assertEqual(set(amostra.y[te]), {0, 1})
        teste.assertTrue(np.all(np.diff(tr) > 0) and np.all(np.diff(te) > 0))
        teste.assertEqual((tr.dtype, te.dtype), (np.int64, np.int64))
        avaliadas.extend(te.tolist())
    teste.assertEqual(sorted(avaliadas), sorted(np.asarray(base).tolist()))


# --------------------------------------------------------------------------- #
# Amostra e alinhamento
# --------------------------------------------------------------------------- #


class TestAmostra(unittest.TestCase):
    def test_amostra_valida_e_somente_leitura(self):
        a = AMOSTRA
        self.assertEqual(a.n, len(IDS))
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", a.sha256))
        for arr in (a.ids, a.y, a.grupos):
            with self.assertRaises(ValueError):
                arr[0] = arr[1]
        self.assertEqual(a.sha256, fl.criar_amostra(IDS, Y, GRUPOS).sha256)

    def test_entradas_copiadas_e_nao_modificadas(self):
        ids, y, grupos = np.array(IDS, dtype=object), Y.astype(np.uint8), np.array(GRUPOS, dtype=object)
        originais = (ids.copy(), y.copy(), grupos.copy())
        a = fl.criar_amostra(ids, y, grupos)
        for entrada, original in zip((ids, y, grupos), originais):
            np.testing.assert_array_equal(entrada, original)
            self.assertTrue(entrada.flags.writeable)
        self.assertEqual((a.y.dtype, a.sha256), (np.int64, AMOSTRA.sha256))
        ids[0], y[0], grupos[0] = "LIDC-IDRI-9999_N09_scan9999", 1 - y[0], "LIDC-IDRI-9999"
        np.testing.assert_array_equal(a.ids, IDS)
        np.testing.assert_array_equal(a.y, Y)
        np.testing.assert_array_equal(a.grupos, GRUPOS)

    def test_entradas_incompativeis_sao_recusadas(self):
        y_lista = Y.tolist()
        casos = {
            "comprimentos": (IDS, Y[:-1], GRUPOS),
            "vazio": ([], np.array([], dtype=np.int64), []),
            "id repetido": ([IDS[0]] + IDS[:-1], Y, GRUPOS),
            "fora da ordem canônica": (IDS[::-1], Y[::-1], GRUPOS[::-1]),
            "id vazio": ([""] + IDS[1:], Y, GRUPOS),
            "grupo não textual": (IDS, Y, [None] + GRUPOS[1:]),
            "y float": (IDS, Y.astype(float), GRUPOS),
            "y booleano": (IDS, Y.astype(bool), GRUPOS),
            "y com 2": (IDS, np.array([2] + y_lista[1:]), GRUPOS),
            "uma classe": (IDS, np.zeros(len(IDS), dtype=np.int64), GRUPOS),
            "y 2-D": (IDS, Y.reshape(-1, 1), GRUPOS),
            "y texto": (IDS, np.array([str(v) for v in y_lista]), GRUPOS),
        }
        for nome, (ids, y, grupos) in casos.items():
            with self.subTest(caso=nome):
                with self.assertRaises(fl.FoldsInvalidos):
                    fl.criar_amostra(ids, y, grupos)

    def test_alinhamento_pandas(self):
        frame = pd.DataFrame({"nodulo_id": IDS, "target_binary": Y, "patient_id": GRUPOS})
        a = fl.criar_amostra(frame["nodulo_id"], frame["target_binary"], frame["patient_id"], X=frame)
        self.assertEqual(a.sha256, AMOSTRA.sha256)
        deslocado = frame["target_binary"].copy()
        deslocado.index = deslocado.index + 1
        with self.assertRaisesRegex(fl.FoldsInvalidos, "desalinhados"):
            fl.criar_amostra(frame["nodulo_id"], deslocado, frame["patient_id"])
        filtrado = frame[frame["target_binary"] == 1]
        with self.assertRaisesRegex(fl.FoldsInvalidos, "desalinhados"):
            fl.criar_amostra(frame["nodulo_id"], frame["target_binary"], frame["patient_id"], X=filtrado)
        with self.assertRaisesRegex(fl.FoldsInvalidos, "comprimentos"):
            fl.criar_amostra(IDS, Y, GRUPOS, X=np.zeros((len(IDS) - 1, 3)))


# --------------------------------------------------------------------------- #
# Três níveis: paridade, cobertura e ausência de vazamento
# --------------------------------------------------------------------------- #


class TestExternos(unittest.TestCase):
    def test_todas_as_repeticoes(self):
        base = np.arange(AMOSTRA.n)
        for r in range(cfg.ESPEC_CV.n_repeticoes):
            with self.subTest(repeticao=r):
                externos = fl.folds_externos(AMOSTRA, r)
                particao_valida(self, AMOSTRA, base, externos, N_EXT)
                self.assertEqual({d.seed for d in externos}, {cfg.ESPEC_CV.seeds_repeticoes()[r]})
                self.assertEqual([d.fold_externo for d in externos], list(range(N_EXT)))
                self.assertEqual({d.nivel for d in externos}, {fl.NIVEL_EXTERNO})

    def test_paridade_com_o_protocolo(self):
        # Seção 7.2: StratifiedGroupKFold(5, shuffle=True, random_state=42 + r), grupos = patient_id.
        for r in (0, 9):
            direto = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + r).split(
                np.zeros((AMOSTRA.n, 1)), Y, GRUPOS)
            for d, (tr, te) in zip(fl.folds_externos(AMOSTRA, r), direto):
                np.testing.assert_array_equal(d.treino, tr)
                np.testing.assert_array_equal(d.avaliacao, te)

    def test_repeticao_invalida(self):
        for r in (-1, cfg.ESPEC_CV.n_repeticoes, True, 1.0, "0"):
            with self.subTest(repeticao=r):
                with self.assertRaises(fl.FoldsInvalidos):
                    fl.folds_externos(AMOSTRA, r)
        with self.assertRaises(TypeError):
            fl.folds_externos((IDS, Y, GRUPOS), 0)


class TestNiveisAninhados(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plano = fl.plano_repeticao(AMOSTRA, 0)

    def test_estabilidade_dentro_do_treino_externo(self):
        for externa in self.plano.externos:
            est = self.plano.estabilidade[externa.fold_externo]
            particao_valida(self, AMOSTRA, externa.treino, est, N_EST)
            self.assertEqual({d.seed for d in est}, {cfg.ESPEC_SELECAO.seed_estabilidade})
            pacientes_avaliacao = set(AMOSTRA.grupos[externa.avaliacao])
            for d in est:
                usadas = np.concatenate([d.treino, d.avaliacao])
                self.assertFalse(np.isin(usadas, externa.avaliacao).any())
                self.assertFalse(pacientes_avaliacao & set(AMOSTRA.grupos[usadas]))

    def test_internos_dentro_do_treino_de_estabilidade(self):
        for (k, j), internos in self.plano.internos.items():
            est = self.plano.estabilidade[k][j]
            particao_valida(self, AMOSTRA, est.treino, internos, N_INT)
            self.assertEqual({d.seed for d in internos}, {cfg.ESPEC_SELECAO.seed_interna(j + 1)})
            externa = self.plano.externos[k]
            for d in internos:
                usadas = np.concatenate([d.treino, d.avaliacao])
                self.assertFalse(np.isin(usadas, est.avaliacao).any())
                self.assertFalse(np.isin(usadas, externa.avaliacao).any())
                self.assertFalse(set(AMOSTRA.grupos[externa.avaliacao]) & set(AMOSTRA.grupos[usadas]))

    def test_seeds_internas_sao_as_da_configuracao(self):
        seeds = sorted({d.seed for ints in self.plano.internos.values() for d in ints})
        self.assertEqual(seeds, [cfg.ESPEC_SELECAO.seed_interna(j) for j in range(1, N_EST + 1)])
        self.assertEqual(seeds, [101, 102, 103, 104, 105])

    def test_contagens_e_rotulos(self):
        self.assertEqual(len(self.plano.externos), N_EXT)
        self.assertEqual(sum(len(v) for v in self.plano.estabilidade.values()), N_EXT * N_EST)
        self.assertEqual(sum(len(v) for v in self.plano.internos.values()), N_EXT * N_EST * N_INT)
        self.assertEqual(self.plano.internos[(2, 3)][1].rotulo, "r00-k2-e3-i1")
        self.assertEqual(len(self.plano.divisoes_do_fold(0)), 1 + N_EST * (1 + N_INT))

    def test_paridade_com_o_procedimento_original(self):
        # Como no script da reprodução: estabilidade SGKF(5, True, 42) e, no fold
        # enumerate(start=1), CV interna SGKF(4, True, 100 + fold), no treino externo.
        externa = self.plano.externos[1]
        base = externa.treino
        y_b, g_b = Y[base], np.asarray(GRUPOS, dtype=object)[base]
        estab = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (tr, te) in enumerate(estab.split(np.zeros((len(base), 1)), y_b, g_b), start=1):
            d = self.plano.estabilidade[1][fold - 1]
            np.testing.assert_array_equal(d.treino, base[tr])
            np.testing.assert_array_equal(d.avaliacao, base[te])
            interna = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=100 + fold)
            for di, (itr, ite) in zip(self.plano.internos[(1, fold - 1)],
                                      interna.split(np.zeros((len(tr), 1)), y_b[tr], g_b[tr])):
                np.testing.assert_array_equal(di.treino, base[tr][itr])
                np.testing.assert_array_equal(di.avaliacao, base[tr][ite])

    def test_divisao_pai_forjada_e_recusada(self):
        externa = self.plano.externos[0]
        tr, te = externa.treino, externa.avaliacao
        # Um nódulo de paciente avaliado com 2+ nódulos passa ao treino: sem linha
        # repetida, mas com o mesmo paciente nos dois lados.
        mover = next(i for i in te if (AMOSTRA.grupos == AMOSTRA.grupos[i]).sum() > 1)
        casos = {
            "linha nos dois lados": (np.union1d(tr, [te[0]]), te),
            "paciente nos dois lados": (np.union1d(tr, [mover]), np.setdiff1d(te, [mover])),
            "lista em vez de array": (list(tr), te),
            "fora do intervalo": (np.append(tr, AMOSTRA.n).astype(np.int64), te),
            "desordenado": (tr[::-1].copy(), te),
            "dtype": (tr.astype(np.int32), te),
        }
        for nome, (treino, avaliacao) in casos.items():
            with self.subTest(caso=nome):
                forjada = fl.Divisao(fl.NIVEL_EXTERNO, 0, 0, None, None, externa.seed, treino,
                                     avaliacao, AMOSTRA.sha256)
                with self.assertRaises(fl.FoldsInvalidos):
                    fl.folds_estabilidade(AMOSTRA, forjada)

    def test_nivel_errado_ou_outra_amostra(self):
        externa = self.plano.externos[0]
        with self.assertRaises(fl.FoldsInvalidos):
            fl.folds_estabilidade(AMOSTRA, self.plano.estabilidade[0][0])
        with self.assertRaises(fl.FoldsInvalidos):
            fl.folds_internos(AMOSTRA, externa)
        outra = fl.criar_amostra(*sintetico(semente=8))
        with self.assertRaisesRegex(fl.FoldsInvalidos, "outra amostra"):
            fl.folds_estabilidade(outra, externa)
        with self.assertRaisesRegex(fl.FoldsInvalidos, "outra amostra"):
            fl.fingerprint_unidade(outra, self.plano, 0)


# --------------------------------------------------------------------------- #
# Entradas degeneradas: recusa explícita
# --------------------------------------------------------------------------- #


class TestDegenerados(unittest.TestCase):
    def test_poucos_pacientes(self):
        ids, y, grupos = sintetico(n_pacientes=4)
        with self.assertRaises(fl.FoldsInvalidos):
            fl.folds_externos(fl.criar_amostra(ids, y, grupos), 0)

    def test_classe_rara_deixa_fold_sem_positivos(self):
        ids, y, grupos = sintetico(n_pacientes=60)
        y = np.zeros(len(ids), dtype=np.int64)
        y[[0, 1]] = 1  # positivos em no máximo 2 pacientes: algum fold fica sem classe 1
        with self.assertRaises(fl.FoldsInvalidos):
            fl.folds_externos(fl.criar_amostra(ids, y, grupos), 0)

    def test_paciente_com_um_unico_grupo_por_linha_nao_e_reagrupado(self):
        # Grupos são usados como vieram: um "paciente" por nódulo não é corrigido em silêncio.
        ids, y, _ = sintetico()
        por_nodulo = fl.criar_amostra(ids, y, ids)
        externos = fl.folds_externos(por_nodulo, 0)
        self.assertEqual(sum(len(d.avaliacao) for d in externos), len(ids))
        self.assertNotEqual(fl.fingerprint_divisao(por_nodulo, externos[0]),
                            fl.fingerprint_divisao(AMOSTRA, fl.folds_externos(AMOSTRA, 0)[0]))


class TestValidadorContraSplitterDefeituoso(unittest.TestCase):
    """O validador do módulo, não o sklearn, é que garante as invariantes."""

    def splitter(self, gerar):
        class Falso:
            def __init__(self, n_splits, shuffle, random_state):
                self.n_splits = n_splits

            def split(self, X, y, groups):
                return gerar(len(X), np.asarray(groups), np.asarray(y), self.n_splits)

        return Falso

    def bons(self, n, grupos, y, k):
        return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0).split(
            np.zeros((n, 1)), y, grupos))

    def test_cada_violacao_e_recusada(self):
        def vazando_paciente(n, grupos, y, k):
            partes = self.bons(n, grupos, y, k)
            tr, te = partes[0]
            mover = next(i for i in te if (grupos == grupos[i]).sum() > 1)  # paciente com 2+ nódulos
            partes[0] = (np.union1d(tr, [mover]), np.setdiff1d(te, [mover]))
            return partes

        def sem_cobertura(n, grupos, y, k):
            partes = self.bons(n, grupos, y, k)
            tr, te = partes[0]
            partes[0] = (tr, te[1:])
            return partes

        def avaliada_duas_vezes(n, grupos, y, k):
            partes = self.bons(n, grupos, y, k)
            partes[1] = partes[0]
            return partes

        def fold_sem_positivos(n, grupos, y, k):
            partes = self.bons(n, grupos, y, k)
            tr, te = partes[0]
            positivos = te[y[te] == 1]
            pacientes = set(grupos[positivos])
            mover = np.array([i for i in te if grupos[i] in pacientes])
            partes[0] = (np.union1d(tr, mover), np.setdiff1d(te, mover))
            return partes

        def folds_a_menos(n, grupos, y, k):
            return self.bons(n, grupos, y, k)[:-1]

        for gerar, mensagem in ((vazando_paciente, "paciente em treino e avaliação"),
                                (sem_cobertura, "difere das linhas do nível"),
                                (avaliada_duas_vezes, "exatamente uma vez"),
                                (fold_sem_positivos, "sem as duas classes"),
                                (folds_a_menos, "esperados")):
            with self.subTest(defeito=gerar.__name__):
                with unittest.mock.patch.object(fl, "StratifiedGroupKFold", self.splitter(gerar)):
                    with self.assertRaisesRegex(fl.FoldsInvalidos, mensagem):
                        fl.folds_externos(AMOSTRA, 0)

    def test_warning_do_splitter_vira_erro(self):
        class ComAviso:
            def __init__(self, **_):
                pass

            def split(self, X, y, groups):
                import warnings
                warnings.warn("menos membros que n_splits", UserWarning)
                return iter([])

        with unittest.mock.patch.object(fl, "StratifiedGroupKFold", ComAviso):
            with self.assertRaisesRegex(fl.FoldsInvalidos, "splitter recusou"):
                fl.folds_externos(AMOSTRA, 0)


# --------------------------------------------------------------------------- #
# Determinismo e fingerprints
# --------------------------------------------------------------------------- #


class TestDeterminismoEFingerprints(unittest.TestCase):
    def test_plano_deterministico(self):
        a, b = fl.plano_repeticao(AMOSTRA, 3), fl.plano_repeticao(fl.criar_amostra(IDS, Y, GRUPOS), 3)
        for k in range(N_EXT):
            for da, db in zip(a.divisoes_do_fold(k), b.divisoes_do_fold(k)):
                self.assertEqual(da.rotulo, db.rotulo)
                np.testing.assert_array_equal(da.treino, db.treino)
                np.testing.assert_array_equal(da.avaliacao, db.avaliacao)
            self.assertEqual(fl.fingerprint_unidade(AMOSTRA, a, k), fl.fingerprint_unidade(AMOSTRA, b, k))

    def test_deterministico_entre_processos(self):
        # Outro interpretador, com outra semente de hash: mesmos fingerprints.
        codigo = ("import sys; sys.path[:0] = sys.argv[1:3]; import test_folds as t; "
                  "p = t.fl.plano_repeticao(t.AMOSTRA, 4); "
                  "print(*[t.fl.fingerprint_unidade(t.AMOSTRA, p, k) for k in range(t.N_EXT)])")
        plano = fl.plano_repeticao(AMOSTRA, 4)
        esperado = " ".join(fl.fingerprint_unidade(AMOSTRA, plano, k) for k in range(N_EXT))
        for semente in ("1", "2"):
            with self.subTest(PYTHONHASHSEED=semente):
                saida = subprocess.run(
                    [sys.executable, "-B", "-c", codigo, str(REPO_ROOT), str(Path(__file__).resolve().parent)],
                    capture_output=True, text=True, check=True, cwd=str(REPO_ROOT),
                    env={**os.environ, "PYTHONHASHSEED": semente})
                self.assertEqual(saida.stdout.strip(), esperado)

    def test_fingerprints_distinguem_repeticoes_folds_e_aninhamento(self):
        p0, p1 = fl.plano_repeticao(AMOSTRA, 0), fl.plano_repeticao(AMOSTRA, 1)
        unidades = {fl.fingerprint_unidade(AMOSTRA, p, k) for p in (p0, p1) for k in range(N_EXT)}
        self.assertEqual(len(unidades), 2 * N_EXT)
        externo = fl.fingerprint_divisao(AMOSTRA, p0.externos[0])
        self.assertNotEqual(externo, fl.fingerprint_unidade(AMOSTRA, p0, 0))

    def test_compativel_com_checkpoints(self):
        plano = fl.plano_repeticao(AMOSTRA, 0)
        d = plano.externos[2]
        esperado = ck.fingerprint_folds({"treino": [IDS[i] for i in d.treino],
                                         "avaliacao": [IDS[i] for i in d.avaliacao]})
        self.assertEqual(fl.fingerprint_divisao(AMOSTRA, d), esperado)
        fp = fl.fingerprint_unidade(AMOSTRA, plano, 2)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", fp))
        self.assertNotIn("LIDC", fp)

    def test_fingerprint_muda_se_a_composicao_mudar(self):
        # Sem o último paciente, a composição de toda unidade muda.
        ultimo = GRUPOS[-1]
        manter = [i for i, g in enumerate(GRUPOS) if g != ultimo]
        menor = fl.criar_amostra([IDS[i] for i in manter], Y[manter], [GRUPOS[i] for i in manter])
        p_completo, p_menor = fl.plano_repeticao(AMOSTRA, 0), fl.plano_repeticao(menor, 0)
        for k in range(N_EXT):
            self.assertNotEqual(fl.fingerprint_unidade(AMOSTRA, p_completo, k),
                                fl.fingerprint_unidade(menor, p_menor, k))

    def test_fold_inexistente(self):
        plano = fl.plano_repeticao(AMOSTRA, 0)
        for k in (-1, N_EXT, "0", True):
            with self.subTest(k=k):
                with self.assertRaises(fl.FoldsInvalidos):
                    fl.fingerprint_unidade(AMOSTRA, plano, k)


# --------------------------------------------------------------------------- #
# Módulo
# --------------------------------------------------------------------------- #


class TestModulo(unittest.TestCase):
    def setUp(self):
        self.fonte = (REPO_ROOT / "src" / "incremental" / "folds.py").read_text(encoding="utf-8")
        self.arvore = ast.parse(self.fonte)

    def test_parametros_vem_da_configuracao(self):
        numeros = {n.value for n in ast.walk(self.arvore)
                   if isinstance(n, ast.Constant) and type(n.value) is int}
        self.assertFalse(numeros & {4, 5, 42, 100, 101}, numeros)
        for nome in ("n_folds_externos", "seeds_repeticoes", "n_folds_estabilidade", "seed_estabilidade",
                     "n_folds_interna", "seed_interna"):
            self.assertIn(nome, self.fonte)

    def test_sem_treino_leitura_de_dados_ou_modelos(self):
        importados = set()
        for no in ast.walk(self.arvore):
            if isinstance(no, ast.Import):
                importados.update(a.name for a in no.names)
            elif isinstance(no, ast.ImportFrom):
                importados.add(("." * no.level) + (no.module or ""))
                importados.update(a.name for a in no.names)
        self.assertFalse(importados & {"pandas", ".dados", "dados", "LogisticRegression", "GridSearchCV",
                                       "pickle"})
        chamadas = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
                    for n in ast.walk(self.arvore) if isinstance(n, ast.Call)}
        self.assertFalse(chamadas & {"fit", "predict", "predict_proba", "read_csv", "carregar"})

    def test_configuracao_inalterada(self):
        from test_guardas import HASH_CONFIG_INCREMENTO_1
        self.assertEqual(cfg.hash_configuracao(), HASH_CONFIG_INCREMENTO_1)


if __name__ == "__main__":
    unittest.main()
