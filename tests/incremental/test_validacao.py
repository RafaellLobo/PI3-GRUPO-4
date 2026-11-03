"""Testes do orquestrador da Etapa 3 (``src/incremental/validacao.py``).

Somente CSVs sintéticos em repositórios git temporários, com aprovações FICTÍCIAS de P1, P2 e P3
registradas no protocolo do repositório temporário (nada é aprovado no projeto real). As linhas de
``test`` têm valores inválidos: se fossem lidas, a carga falharia. As linhas de ``train`` e
``validation`` são válidas. Os modelos rodam de verdade (scikit-learn).

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/incremental -v
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
for caminho in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from src.benchmark_s6 import aprovacoes as ap_s6  # noqa: E402  (contrato cruzado: só leitura)
from src.incremental import bootstrap as bs  # noqa: E402
from src.incremental import checkpoints as ck  # noqa: E402
from src.incremental import config as cfg  # noqa: E402
from src.incremental import dados  # noqa: E402
from src.incremental import guardas as g  # noqa: E402
from src.incremental import modelos as md  # noqa: E402
from src.incremental import rastreabilidade as rst  # noqa: E402
from src.incremental import validacao  # noqa: E402
from test_guardas import PROTOCOLO, VALOR_P2, VALOR_P3, RepoTemporario, aprovar, protocolo, registro  # noqa: E402
from test_piloto import POOL  # noqa: E402

ORDEM_CABECALHO = tuple(random.Random(3).sample(POOL, len(POOL)))
INFORMATIVAS = ("original_glcm_Imc1", "original_firstorder_Skewness", "original_glcm_Imc2")
ESPEC_PEQUENA = dataclasses.replace(cfg.ESPEC_BOOTSTRAP, n_replicas=200)
EXPLORATORIA = rst.Finalidade.EXPLORATORIA


def gerar_linhas(semente=21, por_particao=(("train", 80), ("validation", 30), ("test", 8))):
    """Linhas no formato da Sprint 3. ``test`` tem lixo; ``train`` e ``validation`` são válidos."""
    rng = random.Random(semente)
    linhas, particao_do_paciente, n = [], {}, 0
    for particao, quantidade in por_particao:
        for _ in range(quantidade):
            n += 1
            paciente = f"LIDC-IDRI-{9000 + n:04d}"
            particao_do_paciente[paciente] = particao
            for k in range(1, rng.randint(1, 3) + 1):
                d = rng.gauss(10.0, 3.0)
                imc1 = -0.4 * d + rng.gauss(0.0, 1.0)
                logito = 0.9 * (d - 10.0) - 0.6 * (imc1 + 4.0) + rng.gauss(0.0, 1.0)
                alvo = int(logito > 0)
                linha = {"nodulo_id": f"{paciente}_N{k:02d}_scan{9000 + n}", "patient_id": paciente,
                         "original_nodule_idx": str(k - 1), "target_binary": str(alvo)}
                valores = {cfg.DIAMETRO: d, "original_shape_MeshVolume": d ** 3 * (1 + rng.gauss(0, 0.05)),
                           "original_shape_Sphericity": rng.gauss(0.8, 0.05), "original_glcm_Imc1": imc1,
                           "original_firstorder_Skewness": rng.gauss(0.0, 1.0),
                           "original_glcm_Imc2": rng.gauss(0.5, 0.1)}
                for j, nome in enumerate(ORDEM_CABECALHO):
                    linha[nome] = repr(valores.get(nome, float(j)))
                if particao == "test":
                    linha.update({nome: "lixo" for nome in ORDEM_CABECALHO})
                    linha["target_binary"] = "7"
                linhas.append(linha)
    rng.shuffle(linhas)
    return linhas, particao_do_paciente


LINHAS, PARTICAO_DO_PACIENTE = gerar_linhas()


def linhas_de(particao):
    return sorted((l for l in LINHAS if PARTICAO_DO_PACIENTE[l["patient_id"]] == particao),
                  key=lambda l: l["nodulo_id"])


def pipeline_independente():
    return Pipeline([("s", StandardScaler()), ("l", LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", class_weight=None, max_iter=1000, random_state=42))])


def ajuste_independente(features):
    """Ajusta no treino e prevê na validação com sklearn direto (sem ``modelos`` nem ``dados``)."""
    treino, validacao_ = linhas_de("train"), linhas_de("validation")
    X = np.array([[float(l[f]) for f in features] for l in treino])
    y = np.array([int(l["target_binary"]) for l in treino])
    Xv = np.array([[float(l[f]) for f in features] for l in validacao_])
    yv = np.array([int(l["target_binary"]) for l in validacao_])
    pipe = pipeline_independente().fit(X, y)
    return pipe, pipe.predict_proba(Xv)[:, 1], yv


def features_dos_modelos():
    painel = cfg.PAINEL_DECLARADO if VALOR_P3 == cfg.PAINEL_DECLARADO else VALOR_P3
    return {"M0": cfg.REFERENCIA["M0"], "M1": cfg.REFERENCIA["M1"], "M2": painel,
            "M3": cfg.uniao_ordenada(cfg.REFERENCIA["M3"], painel),
            "M4": cfg.uniao_ordenada(cfg.REFERENCIA["M4"], painel)}


def baseline_json(alterar_intercepto=0.0):
    """``baseline_metrics.json`` sintético: parâmetros de M1 ajustados de forma independente."""
    nomes = list(cfg.FEATURES_M1)
    pipe, _, _ = ajuste_independente(nomes)
    scaler, logreg = pipe.named_steps["s"], pipe.named_steps["l"]
    return json.dumps({
        "features": nomes,
        "prevencao_vazamento": {"scaler_n_samples_seen": int(scaler.n_samples_seen_)},
        "coeficientes": {
            "por_feature": {n: float(c) for n, c in zip(nomes, logreg.coef_.ravel())},
            "intercepto": float(logreg.intercept_[0]) + alterar_intercepto,
            "scaler_mean": {n: float(v) for n, v in zip(nomes, scaler.mean_)},
            "scaler_scale": {n: float(v) for n, v in zip(nomes, scaler.scale_)},
        }}, indent=2) + "\n"


class AmbienteE3:
    """Repositório temporário com tabela, split, baseline e protocolo com P1, P2 e P3 registradas."""

    def __init__(self, registrar_limpeza, com_aprovacoes=True):
        self.repo = RepoTemporario()
        registrar_limpeza(self.repo.remover)
        self.raiz = self.repo.raiz
        cabecalho = ["nodulo_id", "patient_id", "original_nodule_idx", "target_binary", *ORDEM_CABECALHO]
        tabela = io.StringIO()
        escritor = csv.writer(tabela, lineterminator="\n")
        escritor.writerow(cabecalho)
        escritor.writerows([[linha[c] for c in cabecalho] for linha in LINHAS])
        split = "patient_id,split\n" + "".join(f"{p},{s}\n" for p, s in sorted(PARTICAO_DO_PACIENTE.items()))
        arquivos = {"entradas/tabela.csv": tabela.getvalue(), "entradas/split.csv": split,
                    "reports/sprint3/baseline/baseline_metrics.json": baseline_json(),
                    ".gitignore": "checkpoints/\n", "saidas/.gitkeep": ""}
        registros = (registro("P1", data="2026-10-12"), registro("P2", VALOR_P2, "2026-10-12"),
                     registro("P3", VALOR_P3, "2026-10-12"))
        arquivos.update(protocolo(*registros) if com_aprovacoes else protocolo())
        self.commit = self.repo.commit(arquivos, "2026-10-12")
        self.decisoes = (aprovar(P1=(self.commit, None), P2=(self.commit, VALOR_P2),
                                 P3=(self.commit, VALOR_P3)) if com_aprovacoes else dict(cfg.DECISOES))
        (self.raiz / "checkpoints").mkdir()
        self.fonte = dados.FonteDados(
            tabela=self._verificado(self.raiz / "entradas" / "tabela.csv"),
            split=self._verificado(self.raiz / "entradas" / "split.csv"),
            nomes_pool=frozenset(POOL), coorte=self._contagens(None),
            particoes={p: self._contagens(p) for p in dados.PARTICOES})
        self._n = 0

    @staticmethod
    def _verificado(caminho):
        conteudo = caminho.read_bytes()
        return dados.ArquivoVerificado(caminho, hashlib.sha256(conteudo).hexdigest(),
                                       hashlib.sha256(conteudo.replace(b"\r\n", b"\n")).hexdigest())

    @staticmethod
    def _contagens(particao):
        linhas = [l for l in LINHAS if particao is None or PARTICAO_DO_PACIENTE[l["patient_id"]] == particao]
        positivos = sum(l["target_binary"] == "1" for l in linhas)
        return {"nodulos": len(linhas), "pacientes": len({l["patient_id"] for l in linhas}),
                "classe_0": len(linhas) - positivos, "classe_1": positivos}

    def novo(self, rotulo):
        self._n += 1
        destino = self.raiz / "saidas" / f"{rotulo}{self._n}"
        destino.mkdir()
        checkpoints = self.raiz / "checkpoints" / f"{rotulo}{self._n}"
        checkpoints.mkdir()
        return checkpoints, destino

    def executar(self, rotulo="exec", checkpoints=None, **kwargs):
        ck_dir, destino = self.novo(rotulo)
        kwargs.setdefault("decisoes", self.decisoes)
        kwargs.setdefault("repo", self.raiz)
        kwargs.setdefault("fonte", self.fonte)
        return validacao.executar_etapa3(checkpoints or ck_dir, destino, **kwargs), ck_dir, destino


class TestContrato(unittest.TestCase):
    def test_constantes_seguem_o_protocolo(self):
        self.assertEqual(validacao.ATIVIDADE, "etapa3_validacao")
        self.assertEqual(validacao.PARTICOES, cfg.PARTICOES_POR_ETAPA[cfg.Etapa.E3])
        self.assertNotIn("test", validacao.PARTICOES)
        self.assertEqual(validacao.MODELOS, ("M0", "M1", "M2", "M3", "M4"))
        self.assertEqual(validacao.CONTRASTES[:2], (("M3", "M0"), ("M4", "M1")))
        self.assertEqual(validacao.PAPEL_DO_CONTRASTE["M3-M0"], "primario")
        self.assertEqual(validacao.PAPEL_DO_CONTRASTE["M4-M1"], "secundario")
        self.assertEqual(guardas_exigidas(), ("P1", "P2", "P3"))

    def test_leitura_da_secao_9(self):
        casos = {(0.01, 0.05): "L_maior_que_0", (-0.01, 0.05): "L_menor_ou_igual_a_0_menor_ou_igual_a_U",
                 (0.0, 0.05): "L_menor_ou_igual_a_0_menor_ou_igual_a_U",
                 (-0.05, 0.0): "L_menor_ou_igual_a_0_menor_ou_igual_a_U", (-0.05, -0.01): "U_menor_que_0"}
        for (baixo, alto), esperado in casos.items():
            self.assertEqual(validacao.leitura_secao_9(baixo, alto)["codigo"], esperado, (baixo, alto))
        self.assertIn("não equivale a ausência de ganho", validacao.leitura_secao_9(-1, 1)["texto_do_protocolo"])


def guardas_exigidas():
    return g.decisoes_exigidas(validacao.ATIVIDADE, validacao.MODELOS)


class TestPontaAPonta(unittest.TestCase):
    """Uma execução DEFINITIVA completa (B = 2000) compartilhada."""

    @classmethod
    def setUpClass(cls):
        cls.amb = AmbienteE3(cls.addClassCleanup)
        cls.chamadas = []
        original = dados.carregar

        def espiao(atividade, particoes, features, *a, **k):
            cls.chamadas.append((atividade, tuple(particoes)))
            return original(atividade, particoes, features, *a, **k)

        with mock.patch.object(dados, "carregar", espiao):
            cls.res, cls.ck_dir, cls.destino = cls.amb.executar("def")
        cls.rel = json.loads(cls.res.caminho_relatorio.read_text(encoding="utf-8"))
        cls.feats = features_dos_modelos()

    def test_le_somente_train_e_validation(self):
        self.assertEqual(self.chamadas, [("etapa3_validacao", ("train", "validation"))])
        self.assertEqual(self.rel["particoes_lidas"], ["train", "validation"])
        self.assertEqual(self.rel["fora_desta_modalidade"]["particoes_nao_lidas"], ["test"])

    def test_amostra_e_composicao_dos_modelos(self):
        treino, val = linhas_de("train"), linhas_de("validation")
        self.assertEqual(self.rel["amostra"]["treino"]["n_nodulos"], len(treino))
        self.assertEqual(self.rel["amostra"]["validacao"]["n_nodulos"], len(val))
        self.assertEqual(self.rel["amostra"]["validacao"]["n_pacientes"], len({l["patient_id"] for l in val}))
        for m, feats in self.feats.items():
            self.assertEqual(self.rel["modelos"][m]["features"], list(feats))
        self.assertEqual({m: self.rel["modelos"][m]["n_features"] for m in self.feats},
                         {"M0": 1, "M1": 3, "M2": 8, "M3": 9, "M4": 10})

    def test_auc_igual_ao_ajuste_independente(self):
        for m, feats in self.feats.items():
            _, proba, yv = ajuste_independente(feats)
            self.assertAlmostEqual(self.rel["modelos"][m]["metricas"]["auc"]["ponto"], roc_auc_score(yv, proba),
                                   places=10, msg=m)

    def test_contrastes_sao_a_diferenca_dos_pontos_e_a_leitura_vem_do_ic(self):
        for par, c in self.rel["contrastes"].items():
            a, b = par.split("-")
            pontos = {m: self.rel["modelos"][m]["metricas"]["auc"]["ponto"] for m in (a, b)}
            self.assertAlmostEqual(c["auc"]["ponto"], pontos[a] - pontos[b], places=12)
            self.assertEqual(c["leitura_secao_9"], validacao.leitura_secao_9(c["auc"]["limite_inferior"],
                                                                             c["auc"]["limite_superior"]))
            self.assertEqual(c["papel"], validacao.PAPEL_DO_CONTRASTE[par])
        self.assertEqual(set(self.rel["contrastes"]), {"M3-M0", "M4-M1", "M1-M0", "M2-M0"})  # JSON canônico ordena as chaves

    def test_bootstrap_com_os_parametros_da_secao_8_2_e_mesmo_resultado_de_uma_chamada_direta(self):
        b = self.rel["bootstrap"]
        self.assertEqual((b["n_replicas"], b["seed"], b["percentis"], b["unidade"], b["pareado"], b["retreino"]),
                         (2000, 42, [2.5, 97.5], "patient_id", True, False))
        self.assertEqual(b["n_validas"] + b["n_descartadas"], 2000)
        val = linhas_de("validation")
        probas = {m: ajuste_independente(f)[1] for m, f in self.feats.items()}
        direto = bs.bootstrap_pareado([l["patient_id"] for l in val], [int(l["target_binary"]) for l in val],
                                      probas, validacao.CONTRASTES)
        for par in self.rel["contrastes"]:
            for metrica, chave in (("auc", "auc"), ("brier", "brier"), ("log_loss", "log_loss")):
                esperado = direto.contrastes[metrica][par]
                obtido = self.rel["contrastes"][par][chave]
                self.assertAlmostEqual(obtido["limite_inferior"], esperado.limite_inferior, places=9)
                self.assertAlmostEqual(obtido["limite_superior"], esperado.limite_superior, places=9)

    def test_metricas_descritivas_batem_com_calculo_independente(self):
        _, proba, yv = ajuste_independente(self.feats["M3"])
        classe = (proba >= 0.5).astype(int)
        d = self.rel["modelos"]["M3"]["descritivas_limiar_0_5"]
        self.assertAlmostEqual(d["recall_classe_1"], ((classe == 1) & (yv == 1)).sum() / (yv == 1).sum())
        self.assertAlmostEqual(d["especificidade_classe_0"], ((classe == 0) & (yv == 0)).sum() / (yv == 0).sum())
        q = np.clip(proba, 1e-15, 1 - 1e-15)
        ajuste = LogisticRegression(penalty=None, max_iter=1000).fit(np.log(q / (1 - q)).reshape(-1, 1), yv)
        calibracao = self.rel["modelos"]["M3"]["calibracao"]
        self.assertAlmostEqual(calibracao["inclinacao"], float(ajuste.coef_[0][0]), places=6)
        self.assertAlmostEqual(calibracao["intercepto"], float(ajuste.intercept_[0]), places=6)

    def test_invariante_m1_dentro_da_tolerancia(self):
        inv = self.rel["invariante_m1"]
        self.assertTrue(inv["dentro_da_tolerancia"])
        self.assertEqual(set(inv["grupos"]), {"coeficientes", "intercepto", "scaler_mean", "scaler_scale"})

    def test_relatorio_sem_dados_individuais_p_valor_ou_significancia(self):
        texto = self.res.caminho_relatorio.read_text(encoding="utf-8")
        self.assertNotIn("LIDC", texto)
        rst._exigir_sem_dados_individuais(self.rel, "relatorio")
        for proibido in ("p-valor", "p_valor", "significativ", "significânc"):
            self.assertNotIn(proibido, texto.lower())
        self.assertFalse(self.rel["fora_desta_modalidade"]["teste_de_hipotese"])
        self.assertTrue(self.rel["limitacoes_declaradas"])

    def test_manifesto_definitivo_elegivel_e_aceito_pelas_guardas_da_s6(self):
        bruto = self.res.caminho_manifesto.read_bytes()
        m = rst.validar_manifesto(json.loads(bruto.decode("utf-8")))
        self.assertEqual((m["execucao"]["atividade"], m["execucao"]["finalidade"],
                          m["execucao"]["elegivel_definitiva"], m["resultado"]["status"]),
                         ("etapa3_validacao", "definitiva", True, "concluida"))
        self.assertTrue(rst.resultado_definitivo_utilizavel(m))
        self.assertEqual(m["execucao"]["autorizacao"]["decisoes_exigidas"], ["P1", "P2", "P3"])
        self.assertEqual(set(m["resultado"]["saidas"]), {"relatorio"})
        # Contrato cruzado: o validador estrito do benchmark S6 aceita o manifesto produzido aqui.
        validado = ap_s6.validar_manifesto_s4(bruto)
        self.assertEqual(validado["execucao"]["atividade"], "etapa3_validacao")

    def test_checkpoints_locais_um_por_modelo(self):
        slots = sorted(p.name for p in (self.ck_dir / validacao.ATIVIDADE).iterdir())
        self.assertEqual(slots, [f"r000-f00-{m}" for m in validacao.MODELOS])
        self.assertFalse(str(self.ck_dir).startswith(str(self.destino)))

    def test_definitiva_recusa_arvore_suja(self):
        with self.assertRaisesRegex(rst.ExecucaoNaoDefinitiva, "alterações locais"):
            self.amb.executar("def_suja")


class TestBloqueios(unittest.TestCase):
    def test_sem_aprovacoes_nada_e_lido_nem_gravado(self):
        amb = AmbienteE3(self.addCleanup, com_aprovacoes=False)
        with mock.patch.object(dados, "carregar", side_effect=AssertionError("dados lidos sem aprovação")):
            with self.assertRaises(cfg.BloqueioDecisao):
                amb.executar("bloq")
        self.assertEqual(list((amb.raiz / "checkpoints").rglob("*.json")), [])
        self.assertEqual(list((amb.raiz / "saidas").rglob("*.json")), [])

    def test_p3_ausente_bloqueia(self):
        amb = AmbienteE3(self.addCleanup)
        parcial = aprovar(P1=(amb.commit, None), P2=(amb.commit, VALOR_P2))
        with mock.patch.object(dados, "carregar", side_effect=AssertionError("dados lidos sem P3")):
            with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
                amb.executar("bloq", decisoes=parcial)

    def test_aprovacao_sem_evidencia_no_git_bloqueia(self):
        amb = AmbienteE3(self.addCleanup)
        falso = "ab" * 20
        sem_evidencia = aprovar(P1=(falso, None), P2=(falso, VALOR_P2), P3=(falso, VALOR_P3))
        with mock.patch.object(dados, "carregar", side_effect=AssertionError("dados lidos sem evidência")):
            with self.assertRaises(cfg.BloqueioDecisao):
                amb.executar("bloq", decisoes=sem_evidencia)

    def test_destino_ocupado_ou_em_data_e_recusado(self):
        amb = AmbienteE3(self.addCleanup)
        _, destino = amb.novo("ocupado")
        (destino / validacao.ARQUIVO_RELATORIO).write_text("{}", encoding="utf-8")
        with self.assertRaises(rst.SobrescritaRecusada):
            validacao.executar_etapa3(amb.raiz / "checkpoints" / "ocupado1", destino, decisoes=amb.decisoes,
                                      repo=amb.raiz, fonte=amb.fonte, finalidade=EXPLORATORIA)


class TestInvarianteM1(unittest.TestCase):
    def test_baseline_divergente_aborta_antes_de_qualquer_predicao(self):
        amb = AmbienteE3(self.addCleanup)
        alterado = amb.raiz / "entradas" / "baseline_alterado.json"
        alterado.write_text(baseline_json(alterar_intercepto=0.01), encoding="utf-8")
        with mock.patch.object(md.ModeloAjustado, "prever_proba", side_effect=AssertionError("predição feita")):
            with self.assertRaises(validacao.InvarianteM1Violado):
                amb.executar("inv", finalidade=EXPLORATORIA, caminho_baseline=alterado)
        self.assertEqual(list((amb.raiz / "checkpoints").rglob("concluida.json")), [])

    def test_features_do_baseline_diferentes_abortam(self):
        amb = AmbienteE3(self.addCleanup)
        alterado = amb.raiz / "entradas" / "baseline_alterado.json"
        dados_json = json.loads(baseline_json())
        dados_json["features"] = list(reversed(dados_json["features"]))
        alterado.write_text(json.dumps(dados_json), encoding="utf-8")
        with self.assertRaises(validacao.InvarianteM1Violado):
            amb.executar("inv", finalidade=EXPLORATORIA, caminho_baseline=alterado)


class TestRetomadaEDeterminismo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amb = AmbienteE3(cls.addClassCleanup)
        cls.ref, cls.ck_ref, _ = cls.amb.executar("ref", finalidade=EXPLORATORIA, espec_bootstrap=ESPEC_PEQUENA)
        cls.bytes_ref = cls.ref.caminho_relatorio.read_bytes()

    def test_determinismo_entre_execucoes_independentes(self):
        outra, _, _ = self.amb.executar("det", finalidade=EXPLORATORIA, espec_bootstrap=ESPEC_PEQUENA)
        self.assertEqual(outra.caminho_relatorio.read_bytes(), self.bytes_ref)

    def test_retomada_total_nao_retreina_os_modelos(self):
        chamadas = []
        original = md.ajustar

        def contar(*a, **k):
            chamadas.append(a[0].nome)
            return original(*a, **k)

        with mock.patch.object(md, "ajustar", contar):
            _, destino = self.amb.novo("ret")
            retomada = validacao.executar_etapa3(self.ck_ref, destino, decisoes=self.amb.decisoes,
                                                 repo=self.amb.raiz, fonte=self.amb.fonte, finalidade=EXPLORATORIA,
                                                 espec_bootstrap=ESPEC_PEQUENA)
        self.assertEqual(chamadas, ["M1"])  # só o reajuste do invariante; nenhuma unidade retreinada
        self.assertEqual(retomada.caminho_relatorio.read_bytes(), self.bytes_ref)

    def test_interrupcao_reexecuta_so_o_que_faltava(self):
        chamadas = []
        original = md.ajustar

        def interromper(*a, **k):
            chamadas.append(a[0].nome)
            if len(chamadas) == 4:  # invariante (M1), M0, M1, M2 -> interrompe no M2
                raise KeyboardInterrupt("interrupção simulada")
            return original(*a, **k)

        ck_dir, destino = self.amb.novo("int")
        with mock.patch.object(md, "ajustar", interromper):
            with self.assertRaises(KeyboardInterrupt):
                validacao.executar_etapa3(ck_dir, destino, decisoes=self.amb.decisoes, repo=self.amb.raiz,
                                          fonte=self.amb.fonte, finalidade=EXPLORATORIA,
                                          espec_bootstrap=ESPEC_PEQUENA)
        pasta = ck_dir / validacao.ATIVIDADE
        self.assertTrue((pasta / "r000-f00-M0" / ck.ARQUIVO_CONCLUIDA).is_file())
        self.assertTrue((pasta / "r000-f00-M1" / ck.ARQUIVO_CONCLUIDA).is_file())
        self.assertFalse((pasta / "r000-f00-M2" / ck.ARQUIVO_CONCLUIDA).exists())
        self.assertTrue(list((pasta / "r000-f00-M2").glob("falha-*.json")))  # falha registrada, não concluída
        manifesto = json.loads((destino / validacao.ARQUIVO_MANIFESTO).read_text(encoding="utf-8"))
        self.assertEqual(manifesto["resultado"]["status"], "interrompida")
        self.assertNotIn("relatorio", manifesto["resultado"]["saidas"])

        chamadas.clear()
        with mock.patch.object(md, "ajustar", lambda *a, **k: (chamadas.append(a[0].nome), original(*a, **k))[1]):
            _, novo_destino = self.amb.novo("int_retomada")
            final = validacao.executar_etapa3(ck_dir, novo_destino, decisoes=self.amb.decisoes,
                                              repo=self.amb.raiz, fonte=self.amb.fonte, finalidade=EXPLORATORIA,
                                              espec_bootstrap=ESPEC_PEQUENA)
        self.assertEqual(chamadas, ["M1", "M2", "M3", "M4"])  # invariante + as 3 unidades que faltavam
        self.assertEqual(final.caminho_relatorio.read_bytes(), self.bytes_ref)

    def test_checkpoint_de_outro_conjunto_de_dados_nao_e_reaproveitado(self):
        # Alterar o código/entradas muda a identidade; aqui basta um slot inesperado para abortar a retomada.
        ck_dir, destino = self.amb.novo("estranho")
        (ck_dir / validacao.ATIVIDADE / "r000-f00-INVALIDO").mkdir(parents=True)
        with self.assertRaises(ck.CheckpointIncompativel):
            validacao.executar_etapa3(ck_dir, destino, decisoes=self.amb.decisoes, repo=self.amb.raiz,
                                      fonte=self.amb.fonte, finalidade=EXPLORATORIA, espec_bootstrap=ESPEC_PEQUENA)

    def test_payload_gravado_e_validado_contra_o_esperado(self):
        slot = self.ck_ref / validacao.ATIVIDADE / "r000-f00-M3" / ck.ARQUIVO_PAYLOAD
        payload = json.loads(slot.read_text(encoding="utf-8"))
        ids = linhas_de("validation")
        ids = [l["nodulo_id"] for l in ids]
        feats = tuple(features_dos_modelos()["M3"])
        validacao._validar_payload(payload, "M3", feats, ids, len(linhas_de("train")))  # íntegro: aceito
        for mudanca in ({"modelo": "M4"}, {"nodulos_avaliacao": ids[::-1]}, {"proba": payload["proba"][:-1]},
                        {"proba": [1.5] + payload["proba"][1:]}, {"n_treino": 1}, {"features": feats[:-1]}):
            with self.assertRaises(validacao.ValidacaoInvalida, msg=str(list(mudanca))):
                validacao._validar_payload({**payload, **mudanca}, "M3", feats, ids, len(linhas_de("train")))
        with self.assertRaises(validacao.ValidacaoInvalida):
            validacao._validar_payload({**payload, "extra": 1}, "M3", feats, ids, len(linhas_de("train")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
