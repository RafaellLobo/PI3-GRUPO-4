"""Testes da Etapa 0, item 2 (``src/incremental/etapa0.py`` e a seleção do treino completo).

Somente CSVs sintéticos em repositórios git temporários (ambiente de ``test_piloto``). As linhas de
validação e teste têm valores inválidos: se fossem lidas, a carga falharia. A referência da seleção é
um cálculo independente com ``StratifiedGroupKFold`` e ``GridSearchCV`` diretos.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/incremental -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
for caminho in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.incremental import config as cfg  # noqa: E402
from src.incremental import etapa0  # noqa: E402
from src.incremental import folds as fl  # noqa: E402
from src.incremental import modelos as md  # noqa: E402
from src.incremental import rastreabilidade as rst  # noqa: E402
from src.incremental import selecao as sel  # noqa: E402
from test_piloto import Ambiente, LINHAS, ORDEM_CABECALHO, PARTICAO_DO_PACIENTE, amostra_de_treino  # noqa: E402


def matriz_do_treino():
    treino = sorted((l for l in LINHAS if PARTICAO_DO_PACIENTE[l["patient_id"]] == "train"),
                    key=lambda l: l["nodulo_id"])
    return np.array([[float(l[n]) for n in ORDEM_CABECALHO] for l in treino])


def referencia_independente(amostra, X, features):
    """Seleção da Seção 5.4 sobre o treino inteiro, sem usar ``folds`` nem ``selecionar_*``."""
    espec = cfg.ESPEC_SELECAO
    estabilidade = StratifiedGroupKFold(n_splits=espec.n_folds_estabilidade, shuffle=True,
                                        random_state=espec.seed_estabilidade)
    por_fold = []
    for j, (tr, _te) in enumerate(estabilidade.split(X, amostra.y, amostra.grupos)):
        interno = StratifiedGroupKFold(n_splits=espec.n_folds_interna, shuffle=True,
                                       random_state=espec.seed_interna_base + j + 1)
        splits = list(interno.split(X[tr], amostra.y[tr], amostra.grupos[tr]))
        busca = sel.busca_C(splits)
        busca.fit(X[tr], amostra.y[tr])
        melhor = busca.best_estimator_
        variancia, correlacao = melhor.steps[0][1], melhor.steps[1][1]  # filtros de variância e correlação
        nomes = correlacao.get_feature_names_out(variancia.get_feature_names_out(list(features)))
        coef = melhor.named_steps[sel.PASSO_L1].coef_.ravel()
        por_fold.append((float(busca.best_params_[sel.PARAMETRO_C]),
                         tuple(n for n, c in zip(nomes, coef) if abs(c) > espec.limiar_coeficiente)))
    contagem = {f: sum(f in s for _, s in por_fold) for f in features}
    return por_fold, tuple(f for f in features if contagem[f] >= espec.limiar_estabilidade), contagem


class TestFoldsDoTreinoCompleto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amostra, _, _ = amostra_de_treino()
        cls.folds = sel.folds_estabilidade_treino_completo(cls.amostra)

    def test_cinco_folds_cobrindo_cada_linha_uma_vez_sem_paciente_nos_dois_lados(self):
        n = self.amostra.n
        self.assertEqual(len(self.folds), cfg.ESPEC_SELECAO.n_folds_estabilidade)
        avaliadas = np.concatenate([d.avaliacao for d in self.folds])
        self.assertEqual(sorted(avaliadas.tolist()), list(range(n)))
        for d in self.folds:
            self.assertEqual(len(d.treino) + len(d.avaliacao), n)
            self.assertFalse(set(self.amostra.grupos[d.treino].tolist()) & set(self.amostra.grupos[d.avaliacao].tolist()))
            self.assertEqual((d.repeticao, d.fold_externo, d.nivel, d.seed),
                             (sel.TREINO_COMPLETO, sel.TREINO_COMPLETO, fl.NIVEL_ESTABILIDADE,
                              cfg.ESPEC_SELECAO.seed_estabilidade))

    def test_mesmas_particoes_que_o_splitter_direto(self):
        espec = cfg.ESPEC_SELECAO
        direto = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=espec.seed_estabilidade)
        for d, (tr, te) in zip(self.folds, direto.split(np.zeros((self.amostra.n, 1)), self.amostra.y,
                                                       self.amostra.grupos)):
            self.assertEqual(d.treino.tolist(), sorted(tr.tolist()))
            self.assertEqual(d.avaliacao.tolist(), sorted(te.tolist()))

    def test_internos_usam_a_funcao_existente_com_seed_100_mais_j_mais_1(self):
        for d in self.folds:
            internos = fl.folds_internos(self.amostra, d)
            self.assertEqual({i.seed for i in internos}, {100 + d.fold_estabilidade + 1})
            self.assertTrue(all(set(i.treino.tolist()) | set(i.avaliacao.tolist()) == set(d.treino.tolist())
                                for i in internos))

    def test_recusa_entrada_que_nao_e_amostra(self):
        with self.assertRaises(TypeError):
            sel.folds_estabilidade_treino_completo(object())


class TestSelecaoDoTreinoCompleto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amostra, _, _ = amostra_de_treino()
        cls.X = matriz_do_treino()
        cls.features = tuple(ORDEM_CABECALHO)
        cls.res = sel.selecionar_treino_completo(cls.amostra, cls.X, cls.features)
        cls.ref_por_fold, cls.ref_lista, cls.ref_contagem = referencia_independente(cls.amostra, cls.X,
                                                                                    cls.features)

    def test_igual_a_referencia_independente(self):
        self.assertEqual(self.res.s_k, self.ref_lista)
        self.assertEqual(dict(self.res.frequencia), self.ref_contagem)
        self.assertEqual([(f.C, f.selecionadas) for f in self.res.folds], self.ref_por_fold)

    def test_marcas_do_treino_completo_e_sem_identificadores(self):
        self.assertEqual((self.res.repeticao, self.res.fold_externo), (sel.TREINO_COMPLETO, sel.TREINO_COMPLETO))
        self.assertEqual(self.res.sha256_amostra, self.amostra.sha256)
        self.assertEqual(self.res.hash_configuracao, cfg.hash_configuracao())
        self.assertNotIn("LIDC", json.dumps(self.res.como_dict()))
        self.assertEqual([f.n_nodulos for f in self.res.folds], [len(sel.folds_estabilidade_treino_completo(self.amostra)[j].treino)
                                                                  for j in range(5)])

    def test_determinismo(self):
        outra = sel.selecionar_treino_completo(self.amostra, self.X, self.features)
        self.assertEqual(json.dumps(outra.como_dict(), sort_keys=True), json.dumps(self.res.como_dict(), sort_keys=True))

    def test_entradas_invalidas(self):
        com_nan = self.X.copy()
        com_nan[0, 0] = np.nan
        with self.assertRaises(sel.SelecaoInvalida):
            sel.selecionar_treino_completo(self.amostra, com_nan, self.features)
        with self.assertRaises(sel.SelecaoInvalida):
            sel.selecionar_treino_completo(self.amostra, self.X[:, :-1], self.features)
        with self.assertRaises(TypeError):
            sel.selecionar_treino_completo(object(), self.X, self.features)


class TestCompararListas(unittest.TestCase):
    def test_diferencas(self):
        c = etapa0.comparar_listas(("a", "b", "x"), ("a", "b", "c", "d"))
        self.assertEqual((c["em_comum"], c["so_na_ajustada"], c["so_na_declarada"], c["n_em_comum"]),
                         (["a", "b"], ["x"], ["c", "d"], 2))
        self.assertAlmostEqual(c["jaccard"], 2 / 5)

    def test_listas_vazias(self):
        c = etapa0.comparar_listas((), cfg.PAINEL_DECLARADO)
        self.assertEqual((c["n_em_comum"], c["jaccard"]), (0, 0.0))


class TestOrquestradorEtapa0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amb = Ambiente(cls.addClassCleanup)
        cls.res = etapa0.executar_etapa0_item2(cls.amb.destino("e0"), repo=cls.amb.raiz, fonte=cls.amb.fonte)
        cls.rel = json.loads(cls.res.caminho_relatorio.read_text(encoding="utf-8"))

    def test_so_treino_e_sem_modelos_de_avaliacao(self):
        r = self.rel
        self.assertEqual((r["atividade"], r["particoes_lidas"], r["rotulo"]),
                         ("etapa0_prerequisitos", ["train"], "descritivo_etapa0_item2"))
        self.assertEqual(r["fora_desta_modalidade"], {"modelos_ajustados": False, "metricas_de_desempenho": False,
                                                      "particoes_nao_lidas": ["validation", "test"]})
        self.assertEqual(r["amostra"]["n_nodulos"], self.amb.fonte.particoes["train"]["nodulos"])

    def test_lista_e_comparacao_batem_com_a_referencia(self):
        amostra, _, _ = amostra_de_treino()
        _, lista, _ = referencia_independente(amostra, matriz_do_treino(), tuple(ORDEM_CABECALHO))
        self.assertEqual(tuple(self.rel["selecao"]["s_k"]), lista)
        comparacao = self.rel["comparacao_com_lista_declarada"]
        self.assertEqual(comparacao["lista_declarada"], list(cfg.PAINEL_DECLARADO))
        self.assertEqual(comparacao["lista_ajustada"], list(lista))
        self.assertIn("não é a aprovação de P3", comparacao["referencia"])

    def test_manifesto_definitivo_e_valido_sem_decisoes(self):
        m = rst.validar_manifesto(json.loads(self.res.caminho_manifesto.read_text(encoding="utf-8")))
        self.assertEqual((m["execucao"]["atividade"], m["execucao"]["finalidade"], m["resultado"]["status"]),
                         ("etapa0_prerequisitos", "definitiva", "concluida"))
        self.assertEqual(m["execucao"]["autorizacao"]["decisoes_exigidas"], [])
        self.assertEqual(set(m["resultado"]["saidas"]), {"relatorio"})

    def test_relatorio_sem_dados_individuais(self):
        texto = self.res.caminho_relatorio.read_text(encoding="utf-8")
        self.assertNotIn("LIDC", texto)
        rst._exigir_sem_dados_individuais(self.rel, "relatorio")

    def test_nao_ajusta_modelo_de_avaliacao(self):
        with mock.patch.object(md, "ajustar", side_effect=AssertionError("modelo de avaliação ajustado")):
            etapa0.executar_etapa0_item2(self.amb.destino("e0b"), repo=self.amb.raiz, fonte=self.amb.fonte,
                                         finalidade=rst.Finalidade.EXPLORATORIA)

    def test_determinismo_do_relatorio(self):
        outra = etapa0.executar_etapa0_item2(self.amb.destino("e0c"), repo=self.amb.raiz, fonte=self.amb.fonte,
                                             finalidade=rst.Finalidade.EXPLORATORIA)
        self.assertEqual(outra.caminho_relatorio.read_bytes(), self.res.caminho_relatorio.read_bytes())
        self.assertFalse(outra.manifesto["execucao"]["elegivel_definitiva"])

    def test_definitiva_recusa_arvore_suja(self):
        # Depois da primeira execução há saídas não versionadas: uma nova execução definitiva é recusada.
        with self.assertRaisesRegex(rst.ExecucaoNaoDefinitiva, "alterações locais"):
            etapa0.executar_etapa0_item2(self.amb.destino("e0d"), repo=self.amb.raiz, fonte=self.amb.fonte)

    def test_destino_ocupado_e_recusado(self):
        with self.assertRaises(rst.SobrescritaRecusada):
            etapa0.executar_etapa0_item2(self.res.caminho_relatorio.parent, repo=self.amb.raiz, fonte=self.amb.fonte)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
