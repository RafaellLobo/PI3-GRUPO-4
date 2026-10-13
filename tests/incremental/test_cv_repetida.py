"""Testes da CV repetida exploratória (``src/incremental/cv_repetida.py``).

Somente CSVs sintéticos (os de ``test_piloto``) em repositórios git
temporários; nenhuma leitura da base oficial. Seleção e modelos rodam de
verdade. As linhas de validação e teste têm valores inválidos: se fossem
lidas, a carga falharia. Testes de retomada e isolamento reduzem as
repetições (``REPETICOES``) para caber no tempo; a execução completa 10 × 5
roda uma vez, compartilhada.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/incremental -v
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import math
import shutil
import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
for caminho in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.incremental import checkpoints as ck  # noqa: E402
from src.incremental import config as cfg  # noqa: E402
from src.incremental import cv_repetida as cv  # noqa: E402
from src.incremental import dados  # noqa: E402
from src.incremental import folds as fl  # noqa: E402
from src.incremental import guardas as g  # noqa: E402
from src.incremental import modelos as md  # noqa: E402
from src.incremental import piloto  # noqa: E402
from src.incremental import rastreabilidade as rst  # noqa: E402
from src.incremental import selecao as sel  # noqa: E402
from test_piloto import POOL, Ambiente, InterromperNoAjuste, amostra_de_treino, leitura_proibida  # noqa: E402

N_FOLDS = cfg.ESPEC_CV.n_folds_externos


def executar(amb: Ambiente, nome: str, **kwargs) -> cv.ResultadoCvRepetida:
    return cv.executar_cv_repetida(amb.checkpoints, amb.destino(nome), repo=amb.raiz, fonte=amb.fonte, **kwargs)


def slot(amb: Ambiente, r: int, k: int, modelo: str) -> Path:
    return amb.checkpoints / cv.ATIVIDADE / f"r{r:03d}-f{k:02d}-{modelo}"


def payload(amb: Ambiente, r: int, k: int, modelo: str) -> dict:
    return json.loads((slot(amb, r, k, modelo) / ck.ARQUIVO_PAYLOAD).read_text(encoding="utf-8"))


def chaves_de(valor, acumulado=None) -> set:
    acumulado = set() if acumulado is None else acumulado
    if isinstance(valor, dict):
        acumulado.update(valor)
        for v in valor.values():
            chaves_de(v, acumulado)
    elif isinstance(valor, list):
        for v in valor:
            chaves_de(v, acumulado)
    return acumulado


def poucas_repeticoes(*repeticoes):
    return unittest.mock.patch.object(cv, "REPETICOES", tuple(repeticoes))


# --------------------------------------------------------------------------- #
# Fluxo ponta a ponta 10 × 5 (uma execução completa compartilhada)
# --------------------------------------------------------------------------- #


class TestPontaAPonta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amb = Ambiente(cls.addClassCleanup)
        cls.resultado = executar(cls.amb, "completa")
        cls.relatorio = json.loads(cls.resultado.caminho_relatorio.read_text(encoding="utf-8"))
        cls.amostra, _, cls.matriz = amostra_de_treino()
        cls.planos = [fl.plano_repeticao(cls.amostra, r) for r in cv.REPETICOES]

    def test_relatorio_agregado_preliminar_e_somente_descritivo(self):
        r = self.relatorio
        self.assertEqual((r["rotulo"], r["atividade"], r["finalidade"]),
                         ("preliminar_exploratorio", cfg.ATIVIDADE_CV_REPETIDA_EXPLORATORIA, "exploratoria"))
        self.assertEqual(r["particoes_lidas"], ["train"])
        self.assertEqual(r["repeticoes"], list(range(10)))
        self.assertEqual(r["modelos"], ["M0", "M1", "M2v", "M3v", "M4v"])
        self.assertEqual(r["contrastes"], ["M3v-M0", "M4v-M1"])
        self.assertEqual(r["decisoes_pendentes"], list(cfg.DECISOES))
        self.assertEqual(r["hash_configuracao"], cfg.hash_configuracao())
        self.assertEqual(r["amostra"]["n_nodulos"], self.amb.fonte.particoes["train"]["nodulos"])
        self.assertEqual(r["amostra"]["n_pacientes"], self.amb.fonte.particoes["train"]["pacientes"])
        self.assertEqual(len(r["por_repeticao"]), 10)
        self.assertTrue(all(len(rep["folds"]) == N_FOLDS for rep in r["por_repeticao"]))
        self.assertEqual(r["entre_repeticoes"]["unidade_de_resumo"], "repeticao")
        self.assertFalse(any(r["fora_desta_modalidade"][k] for k in ("permutacoes", "spline", "cv_com_lista_fixa")))
        texto = self.resultado.caminho_relatorio.read_text(encoding="utf-8")
        self.assertNotIn("LIDC", texto)
        rst._exigir_sem_dados_individuais(r, "relatorio")
        # Nomes de features (ex.: original_glcm_MaximumProbability) são chaves legítimas da frequência.
        chaves = chaves_de(r) - set(POOL)
        self.assertTrue(set(r["selecao"]["frequencia_nos_treinos_externos"]) <= set(POOL))
        proibidas = {c for c in chaves if c.lower().startswith(("p_valor", "pvalor", "ic_", "ic95"))
                     or any(p in c.lower() for p in ("signific", "intervalo", "desvio", "erro_padrao", "padrao",
                                                     "std", "proba", "predic", "nodulos_aval", "tempo", "duracao",
                                                     "teste_t"))}
        self.assertFalse(proibidas, proibidas)

    def test_oof_completa_e_aucs_por_repeticao_recalculadas_dos_checkpoints(self):
        posicao = {n: i for i, n in enumerate(self.amostra.ids.tolist())}
        for r, plano in zip(cv.REPETICOES, self.planos):
            rep = self.relatorio["por_repeticao"][r]
            self.assertEqual((rep["repeticao"], rep["seed_cv_externa"]), (r, 42 + r))
            for modelo in cv.MODELOS:
                oof, vistos = np.full(self.amostra.n, np.nan), np.zeros(self.amostra.n, dtype=int)
                for k in range(N_FOLDS):
                    p = payload(self.amb, r, k, modelo)
                    self.assertEqual((p["atividade"], p["repeticao"], p["fold"], p["formato"]),
                                     (cv.ATIVIDADE, r, k, cv.FORMATO_PAYLOAD))
                    idx = [posicao[n] for n in p["nodulos_avaliacao"]]
                    np.testing.assert_array_equal(idx, plano.externos[k].avaliacao)
                    oof[idx] = p["proba"]
                    vistos[idx] += 1
                    self.assertEqual(rep["folds"][k]["auc"][modelo],
                                     roc_auc_score(self.amostra.y[idx], p["proba"]))
                self.assertTrue(np.all(vistos == 1), (r, modelo))  # uma predição por nódulo em CADA repetição
                self.assertEqual(rep["auc_oof"][modelo], roc_auc_score(self.amostra.y, oof))
            auc = rep["auc_oof"]
            self.assertEqual(rep["contrastes_descritivos"],
                             {"M3v-M0": auc["M3v"] - auc["M0"], "M4v-M1": auc["M4v"] - auc["M1"]})

    def test_entre_repeticoes_so_valores_media_minimo_e_maximo(self):
        entre = self.relatorio["entre_repeticoes"]
        reps = self.relatorio["por_repeticao"]
        for nome, grupo in (("auc_oof", cv.MODELOS), ("contrastes_descritivos", ("M3v-M0", "M4v-M1"))):
            for chave in grupo:
                valores = [rep[nome][chave] for rep in reps]
                with self.subTest(grupo=nome, chave=chave):
                    self.assertEqual(entre[nome][chave], {"valores_por_repeticao": valores,
                                                          "media": math.fsum(valores) / 10,
                                                          "minimo": min(valores), "maximo": max(valores)})

    def test_isolamento_por_paciente_em_cada_fold_e_repeticao(self):
        for r, plano in zip(cv.REPETICOES, self.planos):
            avaliados = []
            for k, externa in enumerate(plano.externos):
                treino = set(self.amostra.grupos[externa.treino].tolist())
                avaliacao = set(self.amostra.grupos[externa.avaliacao].tolist())
                self.assertFalse(treino & avaliacao, (r, k))
                avaliados.extend(externa.avaliacao.tolist())
                self.assertEqual(payload(self.amb, r, k, "M2v")["selecao"]["registro"]["fingerprint_folds"],
                                 fl.fingerprint_unidade(self.amostra, plano, k))
            self.assertEqual(sorted(avaliados), list(range(self.amostra.n)))
        # Repetições com partições externas diferentes (seed 42 + r), não cópias de r = 0.
        fingerprints = {f["fingerprint_folds"] for rep in self.relatorio["por_repeticao"] for f in rep["folds"]}
        self.assertEqual(len(fingerprints), 10 * N_FOLDS)

    def test_s_k_unico_por_fold_frequencias_e_jaccard(self):
        frequencia = {}
        for r in cv.REPETICOES:
            for k, fold in enumerate(self.relatorio["por_repeticao"][r]["folds"]):
                gravados = [payload(self.amb, r, k, m)["selecao"] for m in cfg.MODELOS_VARIAVEIS]
                self.assertEqual(len({rst.sha256_canonico(s) for s in gravados}), 1)
                self.assertEqual(gravados[0]["s_k"], fold["s_k"])
                self.assertEqual(gravados[0]["registro"]["repeticao"], r)
                self.assertEqual(fold["s_k_tamanho"], len(fold["s_k"]))
                s_k, painel = set(fold["s_k"]), set(cfg.PAINEL_DECLARADO)
                self.assertEqual(fold["jaccard_painel_declarado"], len(s_k & painel) / len(s_k | painel))
                for m in cfg.MODELOS_VARIAVEIS:
                    self.assertEqual(tuple(payload(self.amb, r, k, m)["features"]),
                                     cfg.modelo_variavel(m, tuple(fold["s_k"])).features)
                for nome in fold["s_k"]:
                    frequencia[nome] = frequencia.get(nome, 0) + 1
        selecao = self.relatorio["selecao"]
        self.assertEqual(selecao["n_treinos_externos"], 50)
        self.assertEqual(set(selecao["frequencia_nos_treinos_externos"]), set(POOL))
        self.assertEqual({f: n for f, n in selecao["frequencia_nos_treinos_externos"].items() if n}, frequencia)
        self.assertEqual(selecao["jaccard_painel_declarado"]["painel"], list(cfg.PAINEL_DECLARADO))
        tamanhos = [f["s_k_tamanho"] for rep in self.relatorio["por_repeticao"] for f in rep["folds"]]
        self.assertEqual(sum(selecao["tamanho_s_k"]["contagem_por_tamanho"].values()), 50)
        self.assertEqual(selecao["treinos_externos_com_s_k_vazio"], tamanhos.count(0))

    def test_modelos_ajustados_so_no_treino_externo(self):
        X = np.column_stack([self.matriz[n] for n in cfg.FEATURES_M1])
        for r in (3, 9):
            for k, externa in enumerate(self.planos[r].externos):
                ref = Pipeline([("s", StandardScaler()),
                                ("l", LogisticRegression(**cfg.ESPEC_AVALIACAO.parametros()))])
                ref.fit(X[externa.treino], self.amostra.y[externa.treino])
                np.testing.assert_allclose(payload(self.amb, r, k, "M1")["proba"],
                                           ref.predict_proba(X[externa.avaliacao])[:, 1], rtol=0, atol=1e-12)

    def test_repeticao_zero_reproduz_o_piloto(self):
        # Mesmos dados, seeds e configuração: r = 0 tem de ser numericamente idêntica ao piloto.
        piloto_res = self.amb.executar("piloto")
        rel_piloto = json.loads(piloto_res.caminho_relatorio.read_text(encoding="utf-8"))
        rep0 = self.relatorio["por_repeticao"][0]
        self.assertEqual(rep0["auc_oof"], rel_piloto["auc_oof"])
        self.assertEqual(rep0["contrastes_descritivos"], rel_piloto["contrastes_descritivos"])
        novos = {"s_k_tamanho", "jaccard_painel_declarado"}
        self.assertEqual([{k: v for k, v in f.items() if k not in novos} for f in rep0["folds"]], rel_piloto["folds"])
        for k in range(N_FOLDS):
            for m in cv.MODELOS:
                self.assertEqual(payload(self.amb, 0, k, m)["proba"], self.amb.payload(k, m)["proba"])

    def test_manifesto_exploratorio_nunca_definitivo(self):
        m = rst.ler_manifesto(self.resultado.caminho_manifesto)
        ex = m["execucao"]
        self.assertEqual((ex["atividade"], ex["finalidade"], ex["elegivel_definitiva"], m["resultado"]["status"]),
                         (cfg.ATIVIDADE_CV_REPETIDA_EXPLORATORIA, "exploratoria", False, "concluida"))
        self.assertEqual(ex["autorizacao"]["decisoes_exigidas"], [])
        self.assertEqual(ex["seeds"], {**{f"cv_externa_r{r:02d}": 42 + r for r in range(10)},
                                       "estabilidade": 42, "interna_base": 100, "logistica_selecao": 42,
                                       "logistica_avaliacao": 42})
        self.assertEqual(ex["parametros"]["repeticoes"], list(range(10)))
        self.assertEqual(set(m["resultado"]["saidas"]), {"relatorio"})
        rst.conferir_arquivos(m["resultado"]["saidas"].values(), self.amb.raiz)
        self.assertFalse(rst.resultado_definitivo_utilizavel(m))
        with self.assertRaises(rst.ResultadoNaoUtilizavel):
            rst.exigir_resultado_definitivo(self.resultado.caminho_manifesto, self.amb.raiz)
        self.assertIn("src.incremental.cv_repetida", ex["codigo"])

    def test_checkpoints_tempos_e_predicoes_so_no_diretorio_local(self):
        pasta = self.amb.checkpoints / cv.ATIVIDADE
        self.assertEqual(len([p for p in pasta.iterdir() if (p / ck.ARQUIVO_CONCLUIDA).is_file()]), 250)
        tempos = json.loads(self.resultado.caminho_tempos.read_text(encoding="utf-8"))
        self.assertEqual(self.resultado.caminho_tempos.parent, self.amb.checkpoints.resolve())
        self.assertEqual(len(tempos["unidades_s"]), 250)
        self.assertEqual(sorted(tempos["selecao_s"]), [f"r{r:03d}" for r in range(10)])
        self.assertTrue(all(len(v) == N_FOLDS for v in tempos["selecao_s"].values()))
        ignorado = g._git(self.amb.raiz, "check-ignore", "-q", str(self.amb.checkpoints / "x"))
        self.assertEqual(ignorado.returncode, 0)
        status = rst.estado_git(self.amb.raiz)
        self.assertFalse([c for _, c in status.alteracoes if c.startswith("checkpoints")])

    def test_convergencia_contabilizada(self):
        conv = self.relatorio["convergencia"]
        self.assertEqual(conv["refits_selecao"], 50 * cfg.ESPEC_SELECAO.n_folds_estabilidade)
        nao_convergidos = sum(not e["convergiu"] for r in cv.REPETICOES for k in range(N_FOLDS)
                              for e in payload(self.amb, r, k, "M2v")["selecao"]["registro"]["folds_estabilidade"])
        self.assertEqual(conv["refits_selecao_sem_convergencia"], nao_convergidos)
        self.assertLessEqual(conv["ajustes_avaliacao_logistica"], 250)


# --------------------------------------------------------------------------- #
# Retomada, determinismo e isolamento entre atividades
# --------------------------------------------------------------------------- #


class TestRetomada(unittest.TestCase):
    def test_interrupcao_e_retomada_dao_o_mesmo_relatorio(self):
        amb = Ambiente(self.addCleanup)
        with poucas_repeticoes(0, 1, 2):
            # r = 0 completa (25 ajustes); em r = 1, fold 0 completo e fold 1 com M0, M1 e interrupção em M2v.
            with unittest.mock.patch.object(md, "ajustar", InterromperNoAjuste(33)):
                with self.assertRaises(KeyboardInterrupt):
                    executar(amb, "interrompida")
            manifesto = rst.ler_manifesto(amb.raiz / "saidas" / "interrompida" / cv.ARQUIVO_MANIFESTO)
            self.assertEqual((manifesto["resultado"]["status"], manifesto["resultado"]["saidas"]),
                             ("interrompida", {}))
            self.assertTrue((slot(amb, 1, 1, "M2v") / "falha-001.json").is_file())

            chamadas = []
            original = sel.selecionar_fold_externo

            def espiao(amostra, X, features, plano, k):
                chamadas.append((plano.repeticao, k))
                return original(amostra, X, features, plano, k)

            with unittest.mock.patch.object(sel, "selecionar_fold_externo", espiao):
                retomado = executar(amb, "retomada")
            self.assertEqual(chamadas, [(1, k) for k in range(1, 5)] + [(2, k) for k in range(5)])
            tempos = json.loads(retomado.caminho_tempos.read_text(encoding="utf-8"))
            self.assertEqual(len(tempos["unidades_reaproveitadas"]), 25 + 5 + 2)
            # Mesmos dados noutro repositório, sem interrupção: relatório idêntico byte a byte (determinismo).
            completo = executar(Ambiente(self.addCleanup), "completa")
        self.assertEqual(retomado.caminho_relatorio.read_bytes(), completo.caminho_relatorio.read_bytes())

    def test_checkpoints_do_piloto_nao_sao_reaproveitados_nem_alterados(self):
        amb = Ambiente(self.addCleanup)
        amb.executar("piloto")
        pasta_piloto = amb.checkpoints / piloto.ATIVIDADE
        antes = {p.relative_to(pasta_piloto).as_posix(): p.read_bytes() for p in pasta_piloto.rglob("*") if p.is_file()}
        chamadas = []
        original = sel.selecionar_fold_externo

        def espiao(amostra, X, features, plano, k):
            chamadas.append((plano.repeticao, k))
            return original(amostra, X, features, plano, k)

        with poucas_repeticoes(0), unittest.mock.patch.object(sel, "selecionar_fold_externo", espiao):
            resultado = executar(amb, "cv")
        self.assertEqual(chamadas, [(0, k) for k in range(5)])  # r = 0 recalculada na própria atividade
        depois = {p.relative_to(pasta_piloto).as_posix(): p.read_bytes() for p in pasta_piloto.rglob("*") if p.is_file()}
        self.assertEqual(antes, depois)
        tempos = json.loads(resultado.caminho_tempos.read_text(encoding="utf-8"))
        self.assertEqual(tempos["unidades_reaproveitadas"], [])

    def test_slot_de_outra_atividade_plantado_e_recusado(self):
        amb = Ambiente(self.addCleanup)
        amb.executar("piloto")
        destino = slot(amb, 0, 0, "M0")
        destino.parent.mkdir(parents=True)
        shutil.copytree(amb.slot(0, "M0"), destino)
        with poucas_repeticoes(0), unittest.mock.patch.object(md, "ajustar", leitura_proibida):
            with self.assertRaisesRegex(ck.CheckpointIncompativel, "identidade gravada difere"):
                executar(amb, "plantado")

    def test_payload_corrompido_aborta_a_retomada(self):
        amb = Ambiente(self.addCleanup)
        with poucas_repeticoes(0, 1):
            with unittest.mock.patch.object(md, "ajustar", InterromperNoAjuste(28)):
                with self.assertRaises(KeyboardInterrupt):
                    executar(amb, "parcial")
            caminho = slot(amb, 1, 0, "M1") / ck.ARQUIVO_PAYLOAD
            caminho.write_bytes(caminho.read_bytes().replace(b'"repeticao":1', b'"repeticao":0', 1))
            with self.assertRaisesRegex(ck.CheckpointIncompativel, "payload difere"):
                executar(amb, "retomada")


# --------------------------------------------------------------------------- #
# Recusas: partições, modelos, repetições, destino e superfície da API
# --------------------------------------------------------------------------- #


class TestRecusas(unittest.TestCase):
    def test_particoes_de_validacao_ou_teste_negadas_sem_ler(self):
        amb = Ambiente(self.addCleanup)
        for particoes in (("train", "validation"), ("train", "test"), ("validation",), ("test",)):
            nome = "p-" + "-".join(particoes)
            with self.subTest(particoes=particoes), \
                    unittest.mock.patch.object(cv, "PARTICOES", particoes), \
                    unittest.mock.patch.object(dados, "_ler_bytes", leitura_proibida):
                with self.assertRaises(dados.AcessoNegado):
                    executar(amb, nome)
                m = rst.ler_manifesto(amb.raiz / "saidas" / nome / cv.ARQUIVO_MANIFESTO)
                self.assertEqual((m["resultado"]["status"], m["resultado"]["erro"]), ("falhou", "AcessoNegado"))

    def test_modelos_com_painel_bloqueados_antes_de_ler_dados(self):
        amb = Ambiente(self.addCleanup)
        for nome in cfg.MODELOS_COM_PAINEL:
            with self.subTest(modelo=nome), \
                    unittest.mock.patch.object(cv, "MODELOS", cv.MODELOS + (nome,)), \
                    unittest.mock.patch.object(dados, "_ler_bytes", leitura_proibida):
                with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
                    executar(amb, f"m-{nome}")

    def test_repeticoes_invalidas_recusadas_antes_de_ler(self):
        amb = Ambiente(self.addCleanup)
        for i, repeticoes in enumerate(((), (10,), (-1,), (1, 1), (2, 1), (0.0,), [0, 1], (True,))):
            with self.subTest(repeticoes=repeticoes), \
                    unittest.mock.patch.object(cv, "REPETICOES", repeticoes), \
                    unittest.mock.patch.object(dados, "_ler_bytes", leitura_proibida), \
                    unittest.mock.patch.object(g, "exigir_autorizacao", leitura_proibida):
                with self.assertRaises(cv.CvRepetidaInvalida):
                    executar(amb, f"repeticoes-{i}")

    def test_destino_existente_ou_em_data_recusado(self):
        amb = Ambiente(self.addCleanup)
        destino = amb.destino("ocupado")
        (destino / cv.ARQUIVO_MANIFESTO).write_text("{}", encoding="utf-8")
        with unittest.mock.patch.object(dados, "_ler_bytes", leitura_proibida):
            with self.assertRaises(rst.SobrescritaRecusada):
                cv.executar_cv_repetida(amb.checkpoints, destino, repo=amb.raiz, fonte=amb.fonte)
            dentro_de_data = amb.raiz / "data" / "cv"
            dentro_de_data.mkdir(parents=True)
            with self.assertRaises(piloto.PilotoInvalido):
                cv.executar_cv_repetida(amb.checkpoints, dentro_de_data, repo=amb.raiz, fonte=amb.fonte)

    def test_superficie_sem_parametros_de_modelo_particao_permutacao_ou_spline(self):
        parametros = set(inspect.signature(cv.executar_cv_repetida).parameters)
        self.assertEqual(parametros, {"diretorio_checkpoints", "destino", "decisoes", "repo", "fonte"})
        self.assertEqual(cv.PARTICOES, ("train",))
        self.assertEqual(cv.MODELOS, ("M0", "M1", "M2v", "M3v", "M4v"))
        self.assertEqual(cv.REPETICOES, tuple(range(10)))
        self.assertEqual(cv.ATIVIDADE, "cv_repetida_exploratoria")
        self.assertIn(cv.ATIVIDADE, cfg.ATIVIDADES_SOMENTE_EXPLORATORIAS)
        self.assertNotEqual(cv.FORMATO_PAYLOAD, piloto.FORMATO_PAYLOAD)
        fonte = (REPO_ROOT / "src" / "incremental" / "cv_repetida.py").read_text(encoding="utf-8")
        arvore = ast.parse(fonte)
        nomes = {n.id for n in ast.walk(arvore) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(arvore) if isinstance(n, ast.Attribute)}
        self.assertFalse({n for n in nomes if any(p in n.lower() for p in ("permut", "spline", "shuffle"))}, nomes)
        # Sem ajuste próprio nem modelos de lista fixa: tudo passa pela execução por fold do piloto.
        self.assertFalse(nomes & {"modelo_fixo", "construir_fixo", "ajustar", "selecionar_fold_externo",
                                  "LogisticRegression", "GridSearchCV", "fit"}, nomes)
        importados = {no.module for no in ast.walk(arvore) if isinstance(no, ast.ImportFrom)}
        self.assertFalse({m for m in importados if m and ("scripts" in m or "sklearn" in m)})

    def test_main_exige_confirmacao_explicita(self):
        with unittest.mock.patch.object(cv, "executar_cv_repetida", leitura_proibida), \
                unittest.mock.patch("sys.stderr", io.StringIO()):
            for argv in ([], ["--checkpoints", "c", "--destino", "d"]):
                with self.subTest(argv=argv):
                    with self.assertRaises(SystemExit):
                        cv.main(argv)
        entre = {"auc_oof": {"M0": {"media": 0.5}},
                 "contrastes_descritivos": {"M3v-M0": {"media": 0.0, "minimo": 0.0, "maximo": 0.0}}}
        falso = unittest.mock.Mock(return_value=unittest.mock.Mock(
            relatorio={"entre_repeticoes": entre}, caminho_relatorio=Path("r.json"),
            caminho_manifesto=Path("m.json")))
        with unittest.mock.patch.object(cv, "executar_cv_repetida", falso), \
                unittest.mock.patch("sys.stdout", io.StringIO()):
            self.assertEqual(cv.main(["--checkpoints", "c", "--destino", "d",
                                      "--confirmo-preliminar-exploratorio"]), 0)
        falso.assert_called_once_with(Path("c"), Path("d"))


class TestModulo(unittest.TestCase):
    def test_descrever_e_jaccard(self):
        self.assertEqual(cv._descrever([0.1, 0.3, 0.2]),
                         {"valores_por_repeticao": [0.1, 0.3, 0.2], "media": math.fsum([0.1, 0.3, 0.2]) / 3,
                          "minimo": 0.1, "maximo": 0.3})
        self.assertEqual(cv._jaccard([], ("a", "b")), 0.0)
        self.assertEqual(cv._jaccard(("a", "c"), ("a", "b")), 1 / 3)

    def test_configuracao_inalterada(self):
        from test_guardas import HASH_CONFIG_INCREMENTO_1
        self.assertEqual(cfg.hash_configuracao(), HASH_CONFIG_INCREMENTO_1)
        self.assertTrue(all(not d.aprovada for d in cfg.DECISOES.values()))


if __name__ == "__main__":
    unittest.main()
