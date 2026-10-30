"""Testes de integração do piloto exploratório (``src/incremental/piloto.py``).

Somente CSVs sintéticos em repositórios git temporários; nenhuma leitura da
base oficial. Seleção e modelos rodam de verdade (scikit-learn) sobre os
dados sintéticos. As linhas de validação e teste têm valores inválidos
(``"lixo"`` e alvo 7): se fossem lidas, a carga falharia.

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
from src.incremental import dados  # noqa: E402
from src.incremental import folds as fl  # noqa: E402
from src.incremental import guardas as g  # noqa: E402
from src.incremental import modelos as md  # noqa: E402
from src.incremental import piloto  # noqa: E402
from src.incremental import rastreabilidade as rst  # noqa: E402
from src.incremental import selecao as sel  # noqa: E402
from test_guardas import RepoTemporario  # noqa: E402

with open(dados.POOL_VERSIONADO.caminho, encoding="utf-8", newline="") as _arquivo:
    POOL = tuple(linha["feature"] for linha in csv.DictReader(_arquivo))
ORDEM_CABECALHO = tuple(random.Random(3).sample(POOL, len(POOL)))
INFORMATIVAS = ("original_glcm_Imc1", "original_firstorder_Skewness", "original_glcm_Imc2")
NAO_CONSTANTES = (*cfg.FEATURES_M1, *INFORMATIVAS)
SLOTS = piloto.ATIVIDADE
#: SHA-256 do relatório do piloto sobre os dados sintéticos abaixo, registrado
#: ANTES da refatoração que tirou repetição e atividade das globais (Fase B,
#: configuração c36335c1…). Depende só de dados sintéticos, seeds e
#: configuração: muda com ``hash_configuracao``, nunca com código equivalente.
SHA256_RELATORIO_SINTETICO = "8e10dafa6f081959b79825eca7c232b4168ef71ca94cebb9623fbdd34d654827"
HASH_CONFIG_DA_REFERENCIA = "c36335c1696b99f5d2344e80f14884721007da5810f6be6c8b892de60f4c5885"


# --------------------------------------------------------------------------- #
# Dados sintéticos e ambiente
# --------------------------------------------------------------------------- #


def gerar_linhas(semente=21, por_particao=(("train", 80), ("validation", 8), ("test", 8))):
    """Linhas no formato da Sprint 3; pacientes fictícios LIDC-IDRI-9xxx."""
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
                    linha[nome] = repr(valores.get(nome, float(j)))  # demais: constantes
                if particao != "train":
                    linha.update({nome: "lixo" for nome in ORDEM_CABECALHO})
                    linha["target_binary"] = "7"
                linhas.append(linha)
    rng.shuffle(linhas)
    return linhas, particao_do_paciente


LINHAS, PARTICAO_DO_PACIENTE = gerar_linhas()


class Ambiente:
    """Repositório temporário com tabela e split sintéticos, checkpoints ignorados e destino."""

    def __init__(self, registrar_limpeza):
        self.repo = RepoTemporario()
        registrar_limpeza(self.repo.remover)
        self.raiz = self.repo.raiz
        cabecalho = ["nodulo_id", "patient_id", "original_nodule_idx", "target_binary", *ORDEM_CABECALHO]
        tabela = io.StringIO()
        escritor = csv.writer(tabela, lineterminator="\n")
        escritor.writerow(cabecalho)
        escritor.writerows([[linha[c] for c in cabecalho] for linha in LINHAS])
        split = "patient_id,split\n" + "".join(f"{p},{s}\n" for p, s in sorted(PARTICAO_DO_PACIENTE.items()))
        self.repo.commit({"entradas/tabela.csv": tabela.getvalue(), "entradas/split.csv": split,
                          ".gitignore": "checkpoints/\n", "saidas/.gitkeep": ""}, "2026-10-12")
        self.checkpoints = self.raiz / "checkpoints"
        self.checkpoints.mkdir()
        self.fonte = dados.FonteDados(
            tabela=self._verificado(self.raiz / "entradas" / "tabela.csv"),
            split=self._verificado(self.raiz / "entradas" / "split.csv"),
            nomes_pool=frozenset(POOL), coorte=self._contagens(None), particoes={
                p: self._contagens(p) for p in dados.PARTICOES})

    @staticmethod
    def _verificado(caminho: Path) -> dados.ArquivoVerificado:
        conteudo = caminho.read_bytes()
        return dados.ArquivoVerificado(caminho, hashlib.sha256(conteudo).hexdigest(),
                                       hashlib.sha256(conteudo.replace(b"\r\n", b"\n")).hexdigest())

    @staticmethod
    def _contagens(particao):
        linhas = [l for l in LINHAS if particao is None or PARTICAO_DO_PACIENTE[l["patient_id"]] == particao]
        pacientes = {l["patient_id"] for l in linhas}
        positivos = sum(l["target_binary"] == "1" for l in linhas)
        return {"nodulos": len(linhas), "pacientes": len(pacientes), "classe_0": len(linhas) - positivos,
                "classe_1": positivos}

    def destino(self, nome: str) -> Path:
        caminho = self.raiz / "saidas" / nome
        caminho.mkdir()
        return caminho

    def executar(self, nome: str = "execucao", **kwargs) -> piloto.ResultadoPiloto:
        return piloto.executar_piloto(self.checkpoints, self.destino(nome), repo=self.raiz, fonte=self.fonte,
                                      **kwargs)

    def slot(self, k: int, modelo: str) -> Path:
        return self.checkpoints / SLOTS / f"r{piloto.REPETICAO:03d}-f{k:02d}-{modelo}"

    def payload(self, k: int, modelo: str) -> dict:
        return json.loads((self.slot(k, modelo) / ck.ARQUIVO_PAYLOAD).read_text(encoding="utf-8"))


def amostra_de_treino():
    """Amostra e plano calculados direto das linhas sintéticas (caminho independente do piloto)."""
    treino = sorted((l for l in LINHAS if PARTICAO_DO_PACIENTE[l["patient_id"]] == "train"),
                    key=lambda l: l["nodulo_id"])
    amostra = fl.criar_amostra([l["nodulo_id"] for l in treino],
                               np.array([int(l["target_binary"]) for l in treino], dtype=np.int64),
                               [l["patient_id"] for l in treino])
    matriz = {nome: np.array([float(l[nome]) for l in treino]) for nome in NAO_CONSTANTES}
    return amostra, fl.plano_repeticao(amostra, piloto.REPETICAO), matriz


class InterromperNoAjuste:
    """Substitui ``modelos.ajustar``: interrompe na chamada ``n`` (1-based)."""

    def __init__(self, n: int):
        self.n, self.chamadas, self.original = n, 0, md.ajustar

    def __call__(self, *args, **kwargs):
        self.chamadas += 1
        if self.chamadas == self.n:
            raise KeyboardInterrupt("interrupção simulada")
        return self.original(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Fluxo ponta a ponta (uma execução completa compartilhada)
# --------------------------------------------------------------------------- #


class TestPontaAPonta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amb = Ambiente(cls.addClassCleanup)
        cls.resultado = cls.amb.executar()
        cls.relatorio = json.loads(cls.resultado.caminho_relatorio.read_text(encoding="utf-8"))

    def test_relatorio_agregado_preliminar(self):
        r = self.relatorio
        self.assertEqual((r["rotulo"], r["atividade"], r["finalidade"]),
                         ("preliminar_exploratorio", cfg.ATIVIDADE_PILOTO_EXPLORATORIO, "exploratoria"))
        self.assertEqual(r["particoes_lidas"], ["train"])
        self.assertEqual(r["modelos"], ["M0", "M1", "M2v", "M3v", "M4v"])
        self.assertEqual(set(r["contrastes_descritivos"]), {"M3v-M0", "M4v-M1"})
        self.assertEqual(r["decisoes_pendentes"], list(cfg.DECISOES))
        self.assertEqual(r["fora_desta_modalidade"]["modelos_com_painel"], ["M2", "M3", "M4"])
        self.assertEqual(r["amostra"]["n_nodulos"], self.amb.fonte.particoes["train"]["nodulos"])
        self.assertEqual(r["hash_configuracao"], cfg.hash_configuracao())
        self.assertEqual(len(r["folds"]), cfg.ESPEC_CV.n_folds_externos)
        texto = self.resultado.caminho_relatorio.read_text(encoding="utf-8")
        self.assertNotIn("LIDC", texto)
        rst._exigir_sem_dados_individuais(r, "relatorio")
        chaves = set()

        def coletar(valor):
            if isinstance(valor, dict):
                chaves.update(valor)
                for v in valor.values():
                    coletar(v)
            elif isinstance(valor, list):
                for v in valor:
                    coletar(v)

        coletar(r)
        proibidas = {c for c in chaves if c.lower().startswith(("p_valor", "pvalor", "ic_", "ic95"))
                     or any(p in c.lower() for p in ("signific", "intervalo", "proba", "predic", "nodulos_aval",
                                                     "tempo", "duracao"))}
        self.assertFalse(proibidas, chaves)

    def test_cobertura_oof_e_aucs_recalculadas_dos_checkpoints(self):
        amostra, plano, _ = amostra_de_treino()
        posicao = {n: i for i, n in enumerate(amostra.ids.tolist())}
        for modelo in piloto.MODELOS:
            oof, vistos = np.full(amostra.n, np.nan), np.zeros(amostra.n, dtype=int)
            for k in range(cfg.ESPEC_CV.n_folds_externos):
                p = self.amb.payload(k, modelo)
                idx = [posicao[n] for n in p["nodulos_avaliacao"]]
                np.testing.assert_array_equal(idx, plano.externos[k].avaliacao)
                oof[idx] = p["proba"]
                vistos[idx] += 1
                self.assertAlmostEqual(self.relatorio["folds"][k]["auc"][modelo],
                                       roc_auc_score(amostra.y[idx], p["proba"]), places=15)
            self.assertTrue(np.all(vistos == 1), modelo)  # uma predição por nódulo, sem duplicação
            self.assertEqual(self.relatorio["auc_oof"][modelo], roc_auc_score(amostra.y, oof))
        auc = self.relatorio["auc_oof"]
        self.assertEqual(self.relatorio["contrastes_descritivos"],
                         {"M3v-M0": auc["M3v"] - auc["M0"], "M4v-M1": auc["M4v"] - auc["M1"]})

    def test_modelos_ajustados_so_no_treino_externo(self):
        # Refit independente de M1 em cada fold: só as linhas de treino externo.
        amostra, plano, matriz = amostra_de_treino()
        X = np.column_stack([matriz[n] for n in cfg.FEATURES_M1])
        for k, externa in enumerate(plano.externos):
            ref = Pipeline([("s", StandardScaler()), ("l", LogisticRegression(**cfg.ESPEC_AVALIACAO.parametros()))])
            ref.fit(X[externa.treino], amostra.y[externa.treino])
            np.testing.assert_allclose(self.amb.payload(k, "M1")["proba"],
                                       ref.predict_proba(X[externa.avaliacao])[:, 1], rtol=0, atol=1e-12)

    def test_s_k_unico_por_fold_e_compartilhado_por_m2v_m4v(self):
        for k, fold in enumerate(self.relatorio["folds"]):
            gravados = [self.amb.payload(k, m)["selecao"] for m in cfg.MODELOS_VARIAVEIS]
            self.assertEqual(len({rst.sha256_canonico(s) for s in gravados}), 1)
            self.assertEqual(gravados[0]["s_k"], fold["s_k"])
            self.assertEqual(gravados[0]["sha256"], fold["sha256_selecao"])
            s_k = tuple(fold["s_k"])
            for m in cfg.MODELOS_VARIAVEIS:
                self.assertEqual(tuple(self.amb.payload(k, m)["features"]), cfg.modelo_variavel(m, s_k).features)
            for m in piloto.MODELOS_FIXOS_PILOTO:
                self.assertIsNone(self.amb.payload(k, m)["selecao"])
        self.assertTrue(any(f["s_k"] for f in self.relatorio["folds"]))

    def test_manifesto_exploratorio_nunca_definitivo(self):
        m = rst.ler_manifesto(self.resultado.caminho_manifesto)
        self.assertEqual((m["execucao"]["atividade"], m["execucao"]["finalidade"],
                          m["execucao"]["elegivel_definitiva"], m["resultado"]["status"]),
                         (cfg.ATIVIDADE_PILOTO_EXPLORATORIO, "exploratoria", False, "concluida"))
        self.assertEqual(m["execucao"]["autorizacao"]["decisoes_exigidas"], [])
        self.assertEqual(set(m["resultado"]["saidas"]), {"relatorio"})
        rst.conferir_arquivos(m["resultado"]["saidas"].values(), self.amb.raiz)
        self.assertFalse(rst.resultado_definitivo_utilizavel(m))
        with self.assertRaises(rst.ResultadoNaoUtilizavel):
            rst.exigir_resultado_definitivo(self.resultado.caminho_manifesto, self.amb.raiz)
        self.assertIn("src.incremental.piloto", m["execucao"]["codigo"])

    def test_relatorio_sintetico_identico_a_referencia_anterior_a_refatoracao(self):
        if cfg.hash_configuracao() != HASH_CONFIG_DA_REFERENCIA:  # pragma: no cover — referência de outra config
            self.skipTest("configuração mudou: registre uma nova referência antes de refatorar")
        bruto = self.resultado.caminho_relatorio.read_bytes()
        self.assertEqual(hashlib.sha256(bruto).hexdigest(), SHA256_RELATORIO_SINTETICO)

    def test_tempos_separados_e_predicoes_so_nos_checkpoints(self):
        tempos = json.loads(self.resultado.caminho_tempos.read_text(encoding="utf-8"))
        self.assertEqual(self.resultado.caminho_tempos.parent, self.amb.checkpoints.resolve())
        self.assertEqual(len(tempos["unidades_s"]), 25)
        self.assertEqual(len(tempos["selecao_s"]), cfg.ESPEC_CV.n_folds_externos)
        manifesto = json.loads(self.resultado.caminho_manifesto.read_text(encoding="utf-8"))
        self.assertNotIn("tempos", json.dumps(manifesto["resultado"]["saidas"]))
        # Predições individuais: só nos payloads locais, dentro do diretório ignorado.
        ignorado = g._git(self.amb.raiz, "check-ignore", "-q", str(self.amb.checkpoints / "x"))
        self.assertEqual(ignorado.returncode, 0)
        status = rst.estado_git(self.amb.raiz)
        self.assertFalse([c for _, c in status.alteracoes if c.startswith("checkpoints")])


# --------------------------------------------------------------------------- #
# Retomada, divergência e inconsistências
# --------------------------------------------------------------------------- #


class TestRetomada(unittest.TestCase):
    def test_interrupcao_e_retomada_dao_o_mesmo_relatorio(self):
        amb = Ambiente(self.addCleanup)
        # Folds 0 e 1 completos (10 unidades); no fold 2, M0 e M1 completos e M2v interrompido.
        interromper = InterromperNoAjuste(13)
        with unittest.mock.patch.object(md, "ajustar", interromper):
            with self.assertRaises(KeyboardInterrupt):
                amb.executar("interrompida")
        manifesto = rst.ler_manifesto(amb.raiz / "saidas" / "interrompida" / piloto.ARQUIVO_MANIFESTO)
        self.assertEqual((manifesto["resultado"]["status"], manifesto["resultado"]["saidas"]), ("interrompida", {}))
        self.assertTrue((amb.slot(2, "M2v") / "falha-001.json").is_file())
        self.assertFalse((amb.raiz / "saidas" / "interrompida" / piloto.ARQUIVO_RELATORIO).exists())

        chamadas = []
        original = sel.selecionar_fold_externo

        def espiao(amostra, X, features, plano, k):
            chamadas.append(k)
            return original(amostra, X, features, plano, k)

        with unittest.mock.patch.object(sel, "selecionar_fold_externo", espiao):
            retomado = amb.executar("retomada")
        self.assertEqual(chamadas, [2, 3, 4])  # folds 0 e 1 inteiros e M0/M1 do fold 2 reaproveitados
        tempos = json.loads(retomado.caminho_tempos.read_text(encoding="utf-8"))
        self.assertEqual(len(tempos["unidades_reaproveitadas"]), 12)
        # Mesmos dados em outro repositório, execução sem interrupção: relatório idêntico byte a byte.
        completo = Ambiente(self.addCleanup).executar("completa")
        self.assertEqual(retomado.caminho_relatorio.read_bytes(), completo.caminho_relatorio.read_bytes())
        self.assertEqual(retomado.manifesto["resultado"]["saidas"]["relatorio"]["sha256"],
                         completo.manifesto["resultado"]["saidas"]["relatorio"]["sha256"])


class TestDivergencias(unittest.TestCase):
    def preparar(self, alterar_selecao):
        """Fold 0 gravado com uma seleção alterada; depois M2v do fold 0 some (retomada parcial)."""
        amb = Ambiente(self.addCleanup)
        original = sel.selecionar_fold_externo

        def adulterada(amostra, X, features, plano, k):
            resultado = original(amostra, X, features, plano, k)
            return alterar_selecao(resultado) if k == 0 else resultado

        with unittest.mock.patch.object(sel, "selecionar_fold_externo", adulterada), \
                unittest.mock.patch.object(md, "ajustar", InterromperNoAjuste(6)):
            with self.assertRaises(KeyboardInterrupt):
                amb.executar("adulterada")
        shutil.rmtree(amb.slot(0, "M2v"))
        return amb

    def test_s_k_divergente(self):
        def outro_s_k(s):
            novo = s.s_k[:-1] if s.s_k else (INFORMATIVAS[0],)
            return dataclasses.replace(s, s_k=novo)

        amb = self.preparar(outro_s_k)
        with self.assertRaisesRegex(piloto.SelecaoDivergente, "S_k recalculado difere"):
            amb.executar("retomada")

    def test_registro_divergente_com_o_mesmo_s_k(self):
        def outro_registro(s):
            ultimo = s.folds[-1]
            outro_c = next(c for c in cfg.ESPEC_SELECAO.grade_C.valores() if c != ultimo.C)
            return dataclasses.replace(s, folds=s.folds[:-1] + (dataclasses.replace(ultimo, C=outro_c),))

        amb = self.preparar(outro_registro)
        with self.assertRaisesRegex(piloto.SelecaoDivergente, "registro de seleção recalculado difere"):
            amb.executar("retomada")

    def test_payload_corrompido_aborta_a_retomada(self):
        amb = Ambiente(self.addCleanup)
        with unittest.mock.patch.object(md, "ajustar", InterromperNoAjuste(3)):
            with self.assertRaises(KeyboardInterrupt):
                amb.executar("parcial")
        caminho = amb.slot(0, "M0") / ck.ARQUIVO_PAYLOAD
        caminho.write_bytes(caminho.read_bytes().replace(b'"fold":0', b'"fold":1', 1))
        with self.assertRaisesRegex(ck.CheckpointIncompativel, "payload difere"):
            amb.executar("retomada")

    def test_payload_coerente_de_outra_unidade_e_recusado(self):
        # Checkpoint íntegro, mas com as predições de outro fold: a validação do piloto recusa.
        amb = Ambiente(self.addCleanup)
        with unittest.mock.patch.object(md, "ajustar", InterromperNoAjuste(7)):
            with self.assertRaises(KeyboardInterrupt):
                amb.executar("parcial")
        sessao = ck.SessaoCheckpoints(amb.checkpoints, piloto.ATIVIDADE, g.exigir_autorizacao(piloto.ATIVIDADE),
                                      {"tabela": amb.fonte.tabela.caminho, "split": amb.fonte.split.caminho},
                                      repo=amb.raiz)
        amostra, plano, _ = amostra_de_treino()
        ident = sessao.identidade(piloto.REPETICAO, 1, "M1", piloto.seeds_piloto(),
                                  fl.fingerprint_unidade(amostra, plano, 1))
        trocado = {**amb.payload(0, "M1"), "fold": 1}
        shutil.rmtree(amb.slot(1, "M1"), ignore_errors=True)
        sessao.salvar(ident, trocado)
        with self.assertRaisesRegex(piloto.PilotoInvalido, "nódulos de avaliação diferem"):
            amb.executar("retomada")


# --------------------------------------------------------------------------- #
# Recusas: partições, modelos, destino e superfície da API
# --------------------------------------------------------------------------- #


def leitura_proibida(*_a, **_k):
    raise AssertionError("nenhum dado deveria ser lido")


class TestRecusas(unittest.TestCase):
    def test_modelos_com_painel_bloqueados_antes_de_ler_dados(self):
        amb = Ambiente(self.addCleanup)
        for nome in cfg.MODELOS_COM_PAINEL:
            with self.subTest(modelo=nome), \
                    unittest.mock.patch.object(piloto, "MODELOS", piloto.MODELOS + (nome,)), \
                    unittest.mock.patch.object(dados, "_ler_bytes", leitura_proibida):
                with self.assertRaisesRegex(cfg.BloqueioDecisao, "P3"):
                    amb.executar(f"m-{nome}")

    def test_particoes_de_validacao_ou_teste_negadas_sem_ler(self):
        amb = Ambiente(self.addCleanup)
        for particoes in (("train", "validation"), ("train", "test"), ("validation",), ("test",)):
            with self.subTest(particoes=particoes), \
                    unittest.mock.patch.object(piloto, "PARTICOES", particoes), \
                    unittest.mock.patch.object(dados, "_ler_bytes", leitura_proibida):
                nome = "p-" + "-".join(particoes)
                with self.assertRaises(dados.AcessoNegado):
                    amb.executar(nome)
                m = rst.ler_manifesto(amb.raiz / "saidas" / nome / piloto.ARQUIVO_MANIFESTO)
                self.assertEqual((m["resultado"]["status"], m["resultado"]["erro"]), ("falhou", "AcessoNegado"))

    def test_destino_e_checkpoints_invalidos(self):
        amb = Ambiente(self.addCleanup)
        destino = amb.destino("ocupado")
        (destino / piloto.ARQUIVO_RELATORIO).write_text("{}", encoding="utf-8")
        with unittest.mock.patch.object(dados, "_ler_bytes", leitura_proibida):
            with self.assertRaises(rst.SobrescritaRecusada):
                piloto.executar_piloto(amb.checkpoints, destino, repo=amb.raiz, fonte=amb.fonte)
            with self.assertRaises(piloto.PilotoInvalido):
                piloto.executar_piloto(amb.checkpoints, amb.raiz / "inexistente", repo=amb.raiz, fonte=amb.fonte)
            with self.assertRaises(rst.ArquivoAusente):
                piloto.executar_piloto(amb.checkpoints, amb.checkpoints.parent.parent, repo=amb.raiz,
                                       fonte=amb.fonte)
            versionado = amb.raiz / "versionado"
            versionado.mkdir()
            with self.assertRaises(ck.DiretorioNaoLocal):
                piloto.executar_piloto(versionado, amb.destino("ok"), repo=amb.raiz, fonte=amb.fonte)

    def test_superficie_sem_parametros_de_modelo_particao_permutacao_ou_spline(self):
        parametros = set(inspect.signature(piloto.executar_piloto).parameters)
        self.assertEqual(parametros, {"diretorio_checkpoints", "destino", "decisoes", "repo", "fonte"})
        self.assertEqual(piloto.PARTICOES, ("train",))
        self.assertEqual(piloto.MODELOS, ("M0", "M1", "M2v", "M3v", "M4v"))
        self.assertEqual(piloto.REPETICAO, 0)
        fonte = (REPO_ROOT / "src" / "incremental" / "piloto.py").read_text(encoding="utf-8")
        arvore = ast.parse(fonte)
        nomes = {n.id for n in ast.walk(arvore) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(arvore) if isinstance(n, ast.Attribute)}
        self.assertFalse({n for n in nomes if any(p in n.lower() for p in ("permut", "spline", "painel_fixo",
                                                                            "shuffle"))}, nomes)
        importados = {no.module for no in ast.walk(arvore) if isinstance(no, ast.ImportFrom)}
        self.assertFalse({m for m in importados if m and "scripts" in m})

    def test_main_exige_confirmacao_explicita(self):
        with unittest.mock.patch.object(piloto, "executar_piloto", leitura_proibida), \
                unittest.mock.patch("sys.stderr", io.StringIO()):
            for argv in ([], ["--checkpoints", "c", "--destino", "d"]):
                with self.subTest(argv=argv):
                    with self.assertRaises(SystemExit):
                        piloto.main(argv)
        falso = unittest.mock.Mock(return_value=unittest.mock.Mock(
            relatorio={"auc_oof": {}, "contrastes_descritivos": {}},
            caminho_relatorio=Path("r.json"), caminho_manifesto=Path("m.json")))
        with unittest.mock.patch.object(piloto, "executar_piloto", falso), \
                unittest.mock.patch("sys.stdout", io.StringIO()):
            self.assertEqual(piloto.main(["--checkpoints", "c", "--destino", "d",
                                          "--confirmo-preliminar-exploratorio"]), 0)
        falso.assert_called_once_with(Path("c"), Path("d"))


class TestModulo(unittest.TestCase):
    #: Funções que podem ler as constantes de unidade do piloto; as demais as recebem pelo contexto.
    PODEM_LER_GLOBAIS = {"seeds_piloto", "_parametros", "_consolidar", "executar_piloto", "main"}

    def test_funcoes_compartilhadas_nao_leem_globais_de_unidade(self):
        arvore = ast.parse((REPO_ROOT / "src" / "incremental" / "piloto.py").read_text(encoding="utf-8"))
        globais = {"REPETICAO", "ATIVIDADE", "FORMATO_PAYLOAD", "FORMATO_RELATORIO", "ARQUIVO_RELATORIO"}
        for no in arvore.body:
            if isinstance(no, ast.FunctionDef) and no.name not in self.PODEM_LER_GLOBAIS:
                usados = {n.id for n in ast.walk(no) if isinstance(n, ast.Name)} & globais
                # Defaults de parâmetro (nomes de arquivo) não leem unidade; o corpo, sim.
                defaults = {n.id for d in no.args.defaults for n in ast.walk(d) if isinstance(n, ast.Name)}
                with self.subTest(funcao=no.name):
                    self.assertFalse(usados - defaults, usados)

    def test_contexto_incoerente_e_recusado(self):
        amb = Ambiente(self.addCleanup)
        autorizacao = g.exigir_autorizacao(piloto.ATIVIDADE)
        entradas = {"tabela": amb.fonte.tabela.caminho, "split": amb.fonte.split.caminho}
        sessao = ck.SessaoCheckpoints(amb.checkpoints, piloto.ATIVIDADE, autorizacao, entradas, repo=amb.raiz)
        carga, frame, amostra = piloto._carregar_treino(piloto.ATIVIDADE, piloto.PARTICOES, autorizacao,
                                                        cfg.DECISOES, amb.raiz, amb.fonte)
        plano = fl.plano_repeticao(amostra, piloto.REPETICAO)
        base = dict(carga=carga, frame=frame, amostra=amostra, plano=plano, sessao=sessao,
                    seeds=piloto.seeds_piloto(), decisoes=cfg.DECISOES, repo=amb.raiz, tempos={},
                    atividade=piloto.ATIVIDADE, repeticao=piloto.REPETICAO, formato_payload=piloto.FORMATO_PAYLOAD)
        piloto._Contexto(**base)  # coerente
        for campo, valor in (("repeticao", 1), ("atividade", cfg.ATIVIDADE_CV_REPETIDA_EXPLORATORIA),
                             ("plano", fl.plano_repeticao(amostra, 1))):
            with self.subTest(campo=campo):
                with self.assertRaises(piloto.PilotoInvalido):
                    piloto._Contexto(**{**base, campo: valor})

    def test_seeds_por_repeticao(self):
        self.assertEqual(piloto.seeds_piloto(), piloto.seeds_repeticao(0))
        for r, seed in enumerate(cfg.ESPEC_CV.seeds_repeticoes()):
            self.assertEqual(piloto.seeds_repeticao(r)["cv_externa"], seed)
            self.assertEqual({k: v for k, v in piloto.seeds_repeticao(r).items() if k != "cv_externa"},
                             {k: v for k, v in piloto.seeds_piloto().items() if k != "cv_externa"})

    def test_configuracao_inalterada(self):
        from test_guardas import HASH_CONFIG_INCREMENTO_1
        self.assertEqual(cfg.hash_configuracao(), HASH_CONFIG_INCREMENTO_1)
        self.assertTrue(all(not d.aprovada for d in cfg.DECISOES.values()))


if __name__ == "__main__":
    unittest.main()
