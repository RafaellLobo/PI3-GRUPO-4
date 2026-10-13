"""Testes do Incremento 1 — configuração centralizada (``src/incremental/config.py``).

Não leem dados de pacientes nem treinam modelos. Os testes de paridade leem
apenas arquivos de texto versionados: a tabela da Seção 5.1 do protocolo, o CSV
com os NOMES das features da reprodução versionada e o código de
``scripts/run_baseline.py``, analisado por AST, sem ser executado.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/incremental -v
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.incremental import config as cfg  # noqa: E402

COMMIT_FICTICIO = "0" * 40  # usado só para montar decisões aprovadas de teste
SHA_CSV_REPRODUZIDO = "d45c68824548443be72443c8e2acdf4605cd3952e60906b861bacf82f073db34"


def decisoes_com(**aprovadas) -> dict:
    """Cópia das decisões com as indicadas aprovadas — somente para teste."""
    novas = dict(cfg.DECISOES)
    for codigo, valor in aprovadas.items():
        base = cfg.DECISOES[codigo]
        novas[codigo] = dataclasses.replace(
            base, status=cfg.StatusDecisao.APROVADA,
            commit_revisao_protocolo=COMMIT_FICTICIO, valor=valor,
        )
    return novas


class TestPaineis(unittest.TestCase):
    def test_paineis_tem_oito_nomes_unicos_e_validos(self):
        for nome, painel in cfg.PAINEIS_CANDIDATOS.items():
            with self.subTest(painel=nome):
                self.assertIsInstance(painel, tuple)
                self.assertEqual(len(painel), cfg.N_PAINEL)
                self.assertEqual(len(set(painel)), cfg.N_PAINEL)
                self.assertNotIn(cfg.COLUNA_PROIBIDA, painel)
                self.assertEqual(cfg.validar_nomes(painel, nome), painel)

    def test_paineis_diferem_como_registrado_na_auditoria(self):
        dec, rep = set(cfg.PAINEL_DECLARADO), set(cfg.PAINEL_REPRODUZIDO)
        self.assertEqual(len(dec & rep), 6)
        self.assertEqual(dec - rep, {"original_firstorder_Skewness", "original_ngtdm_Coarseness"})
        self.assertEqual(rep - dec, {"original_glcm_Imc1", "original_glcm_ClusterShade"})

    def test_nenhum_painel_contem_diametro_ou_volume_e_ambos_contem_sphericity(self):
        for painel in cfg.PAINEIS_CANDIDATOS.values():
            self.assertNotIn(cfg.DIAMETRO, painel)
            self.assertNotIn("original_shape_MeshVolume", painel)
            self.assertIn("original_shape_Sphericity", painel)

    def test_nao_existe_painel_padrao(self):
        # Os únicos atributos do módulo que são painéis de 8 radiômicas são as
        # duas candidatas nomeadas; nenhum outro nome aponta para um painel.
        paineis = {
            nome for nome in dir(cfg)
            if isinstance(getattr(cfg, nome), tuple) and len(getattr(cfg, nome)) == cfg.N_PAINEL
            and all(isinstance(f, str) and f.startswith(cfg.PREFIXO_RADIOMICA) for f in getattr(cfg, nome))
        }
        self.assertEqual(paineis, {"PAINEL_DECLARADO", "PAINEL_REPRODUZIDO"})
        self.assertEqual(set(cfg.PAINEIS_CANDIDATOS), {"declarado", "reproduzido"})
        with self.assertRaises(cfg.BloqueioDecisao):
            cfg.painel_fixo()

    def test_painel_declarado_igual_a_secao_5_1_do_protocolo(self):
        texto = (REPO_ROOT / cfg.PROTOCOLO).read_text(encoding="utf-8")
        secao = texto.split("### 5.1", 1)[1].split("### 5.2", 1)[0]
        linhas = re.findall(r"^\| (\d) \| `([^`]+)` \|", secao, flags=re.M)
        self.assertEqual([int(i) for i, _ in linhas], list(range(1, 9)))
        self.assertEqual(tuple(n for _, n in linhas), cfg.PAINEL_DECLARADO)

    def test_painel_reproduzido_igual_ao_csv_versionado(self):
        caminho = (REPO_ROOT / "reports/sprint4/feature_selection/reproducao_juan"
                   / "features_selecionadas_lasso.csv")
        self.assertEqual(hashlib.sha256(caminho.read_bytes()).hexdigest(), SHA_CSV_REPRODUZIDO)
        with caminho.open(encoding="utf-8", newline="") as f:
            nomes = tuple(linha["feature"] for linha in csv.DictReader(f))
        self.assertEqual(nomes, cfg.PAINEL_REPRODUZIDO)


class TestParidadeSprint3(unittest.TestCase):
    """M1 e a especificação de avaliação conferidos com o baseline, sem executá-lo."""

    @classmethod
    def setUpClass(cls):
        arvore = ast.parse((REPO_ROOT / "scripts/run_baseline.py").read_text(encoding="utf-8"))
        cls.atribuicoes = {}
        for no in arvore.body:
            if isinstance(no, ast.AnnAssign) and isinstance(no.target, ast.Name) and no.value is not None:
                cls.atribuicoes[no.target.id] = no.value
            elif isinstance(no, ast.Assign) and len(no.targets) == 1 and isinstance(no.targets[0], ast.Name):
                cls.atribuicoes[no.targets[0].id] = no.value

    def _valor(self, no):
        if isinstance(no, ast.Name):
            return self._valor(self.atribuicoes[no.id])
        if isinstance(no, ast.Dict):
            return {self._valor(k): self._valor(v) for k, v in zip(no.keys, no.values)}
        if isinstance(no, ast.Tuple):
            return tuple(self._valor(e) for e in no.elts)
        return ast.literal_eval(no)

    def test_features_m1_iguais_ao_baseline(self):
        self.assertEqual(self._valor(self.atribuicoes["BASELINE_FEATURES"]), cfg.FEATURES_M1)

    def test_hiperparametros_iguais_ao_baseline(self):
        sprint3 = self._valor(self.atribuicoes["LOGREG_PARAMS"])
        atual = cfg.ESPEC_AVALIACAO.parametros()
        self.assertEqual(atual.pop("solver"), "lbfgs")  # explícito aqui; padrão no baseline
        self.assertEqual(atual, sprint3)
        self.assertEqual(self._valor(self.atribuicoes["THRESHOLD"]), cfg.THRESHOLD)
        self.assertEqual(self._valor(self.atribuicoes["RANDOM_STATE"]), cfg.ESPEC_AVALIACAO.random_state)


class TestModelosFixos(unittest.TestCase):
    def test_m0_e_m1(self):
        m0, m1 = cfg.modelo_fixo("M0"), cfg.modelo_fixo("M1")
        self.assertEqual(m0.features, (cfg.DIAMETRO,))
        self.assertEqual(m1.features, cfg.FEATURES_M1)
        for m in (m0, m1):
            self.assertIs(m.tipo, cfg.TipoModelo.FIXO)
            self.assertFalse(m.usa_dummy)

    def test_m2_m3_m4_bloqueados_com_p3_pendente(self):
        for nome in cfg.MODELOS_COM_PAINEL:
            with self.subTest(modelo=nome), self.assertRaises(cfg.BloqueioDecisao):
                cfg.modelo_fixo(nome)

    def test_composicao_com_p3_aprovada_para_cada_candidato(self):
        for rotulo, painel in cfg.PAINEIS_CANDIDATOS.items():
            dec = decisoes_com(P3=painel)
            m2, m3, m4 = (cfg.modelo_fixo(n, dec) for n in ("M2", "M3", "M4"))
            with self.subTest(painel=rotulo):
                self.assertEqual(m2.features, painel)
                self.assertEqual(m3.features, (cfg.DIAMETRO, *painel))
                self.assertEqual((m2.n_features, m3.n_features, m4.n_features), (8, 9, 10))
                self.assertEqual(m4.features[:3], cfg.FEATURES_M1)
                self.assertEqual(m4.features.count("original_shape_Sphericity"), 1)
                self.assertEqual(set(m4.features), set(cfg.FEATURES_M1) | set(painel))
                for m in (m2, m3, m4):
                    self.assertNotIn(cfg.COLUNA_PROIBIDA, m.features)
                    self.assertEqual(len(set(m.features)), m.n_features)

    def test_p3_aprovada_com_painel_invalido_e_rejeitada(self):
        invalidos = {
            "sete": cfg.PAINEL_DECLARADO[:7],
            "duplicado": cfg.PAINEL_DECLARADO[:7] + (cfg.PAINEL_DECLARADO[0],),
            "nodule_idx": cfg.PAINEL_DECLARADO[:7] + (cfg.COLUNA_PROIBIDA,),
            "lista": list(cfg.PAINEL_DECLARADO),
        }
        for rotulo, painel in invalidos.items():
            with self.subTest(caso=rotulo), self.assertRaises(ValueError):
                cfg.painel_fixo(decisoes_com(P3=painel))

    def test_modelo_desconhecido(self):
        with self.assertRaises(ValueError):
            cfg.modelo_fixo("M2v")
        with self.assertRaises(ValueError):
            cfg.modelo_variavel("M2", ())


class TestModelosVariaveis(unittest.TestCase):
    S = ("original_glcm_Imc1", "original_firstorder_Skewness")

    def test_composicao(self):
        self.assertEqual(cfg.modelo_variavel("M2v", self.S).features, self.S)
        self.assertEqual(cfg.modelo_variavel("M3v", self.S).features, (cfg.DIAMETRO, *self.S))
        self.assertEqual(cfg.modelo_variavel("M4v", self.S).features, (*cfg.FEATURES_M1, *self.S))
        for nome in cfg.MODELOS_VARIAVEIS:
            m = cfg.modelo_variavel(nome, self.S)
            self.assertIs(m.tipo, cfg.TipoModelo.VARIAVEL)
            self.assertFalse(m.usa_dummy)

    def test_s_k_com_features_de_referencia_entram_uma_vez(self):
        s = (cfg.DIAMETRO, "original_shape_Sphericity", "original_glcm_Imc2")
        m2v, m3v, m4v = (cfg.modelo_variavel(n, s) for n in cfg.MODELOS_VARIAVEIS)
        self.assertEqual(m2v.features, s)
        self.assertEqual(m3v.features, (cfg.DIAMETRO, "original_shape_Sphericity", "original_glcm_Imc2"))
        self.assertEqual(m4v.features, (*cfg.FEATURES_M1, "original_glcm_Imc2"))
        for m in (m2v, m3v, m4v):
            self.assertEqual(len(set(m.features)), m.n_features)

    def test_s_k_vazio_dummy_somente_em_m2v(self):
        m2v, m3v, m4v = (cfg.modelo_variavel(n, ()) for n in cfg.MODELOS_VARIAVEIS)
        self.assertTrue(m2v.usa_dummy)
        self.assertEqual(m2v.features, ())
        self.assertFalse(m3v.usa_dummy)
        self.assertEqual(m3v.features, cfg.modelo_fixo("M0").features)
        self.assertFalse(m4v.usa_dummy)
        self.assertEqual(m4v.features, cfg.modelo_fixo("M1").features)

    def test_s_k_invalido_e_rejeitado(self):
        casos = {
            "nodule_idx": ("original_glcm_Imc1", cfg.COLUNA_PROIBIDA),
            "duplicado": ("original_glcm_Imc1", "original_glcm_Imc1"),
            "sem_prefixo": ("target_binary",),
        }
        for rotulo, s in casos.items():
            for nome in cfg.MODELOS_VARIAVEIS:
                with self.subTest(caso=rotulo, modelo=nome), self.assertRaises(ValueError):
                    cfg.modelo_variavel(nome, s)


class TestDecisoes(unittest.TestCase):
    def test_todas_pendentes(self):
        self.assertEqual(set(cfg.DECISOES), {"P1", "P2", "P3", "P4", "P5", "REVISAO_PROTOCOLO"})
        for codigo, d in cfg.DECISOES.items():
            with self.subTest(decisao=codigo):
                self.assertIs(d.status, cfg.StatusDecisao.PENDENTE)
                self.assertIsNone(d.commit_revisao_protocolo)
                self.assertIsNone(d.valor)
                with self.assertRaises(cfg.BloqueioDecisao):
                    cfg.exigir_aprovada(codigo)

    def test_aprovacao_exige_hash_completo_do_commit(self):
        for commit in (None, "abc1234", "Z" * 40):
            with self.subTest(commit=commit), self.assertRaises(ValueError):
                cfg.Decisao("PX", "teste", "—", cfg.StatusDecisao.APROVADA, commit)

    def test_pendente_nao_aceita_commit_nem_valor(self):
        with self.assertRaises(ValueError):
            cfg.Decisao("PX", "teste", "—", commit_revisao_protocolo=COMMIT_FICTICIO)
        with self.assertRaises(ValueError):
            cfg.Decisao("PX", "teste", "—", valor=cfg.PAINEL_DECLARADO)

    def test_spline_bloqueado_com_p2_pendente(self):
        with self.assertRaises(cfg.BloqueioDecisao):
            cfg.spline_confirmado()
        self.assertEqual(cfg.spline_confirmado(decisoes_com(P2=cfg.SPLINE_PROPOSTO)), cfg.SPLINE_PROPOSTO)

    def test_cv_com_lista_fixa_exige_p3(self):
        self.assertIn("P3", cfg.DECISOES_POR_ETAPA[cfg.ATIVIDADE_CV_LISTA_FIXA])
        self.assertNotIn("P3", cfg.DECISOES_POR_ETAPA[cfg.Etapa.E2.value])
        for etapa in (cfg.Etapa.E3, cfg.Etapa.E4, cfg.Etapa.E5):
            self.assertIn("P3", cfg.DECISOES_POR_ETAPA[etapa.value])


class TestProtecaoModelos(unittest.TestCase):
    """EspecModelo valida coerência e impede M2–M4 fora de ``modelo_fixo`` com P3 aprovada."""

    def test_construcao_direta_de_m2_m3_m4_bloqueada_com_p3_pendente(self):
        for nome in cfg.MODELOS_COM_PAINEL:
            features = cfg.uniao_ordenada(cfg.REFERENCIA[nome], cfg.PAINEL_DECLARADO)
            with self.subTest(modelo=nome), self.assertRaises(cfg.BloqueioDecisao):
                cfg.EspecModelo(nome, cfg.TipoModelo.FIXO, features)

    def test_replace_a_partir_de_m0_ou_m1_bloqueado(self):
        m0, m1 = cfg.modelo_fixo("M0"), cfg.modelo_fixo("M1")
        casos = {
            "M1->M4": (m1, "M4", cfg.uniao_ordenada(cfg.FEATURES_M1, cfg.PAINEL_DECLARADO)),
            "M0->M3": (m0, "M3", cfg.uniao_ordenada((cfg.DIAMETRO,), cfg.PAINEL_REPRODUZIDO)),
            "M1->M2": (m1, "M2", cfg.PAINEL_DECLARADO),
        }
        for rotulo, (origem, nome, features) in casos.items():
            with self.subTest(caso=rotulo), self.assertRaises(cfg.BloqueioDecisao):
                dataclasses.replace(origem, nome=nome, features=features)

    def test_marca_forjada_ou_incompleta_bloqueada(self):
        features = cfg.uniao_ordenada((cfg.DIAMETRO,), cfg.PAINEL_DECLARADO)
        for marca in (object(), True, (object(), cfg.PAINEL_DECLARADO), (None,)):
            with self.subTest(marca=type(marca).__name__), self.assertRaises(cfg.BloqueioDecisao):
                cfg.EspecModelo("M3", cfg.TipoModelo.FIXO, features, _autorizacao=marca)

    def test_replace_de_modelo_autorizado_nao_troca_painel_nem_nome(self):
        dec = decisoes_com(P3=cfg.PAINEL_DECLARADO)
        m2, m3 = cfg.modelo_fixo("M2", dec), cfg.modelo_fixo("M3", dec)
        outra = cfg.uniao_ordenada((cfg.DIAMETRO,), cfg.PAINEL_REPRODUZIDO)
        with self.assertRaises(ValueError):
            dataclasses.replace(m3, features=outra)
        with self.assertRaises(ValueError):
            dataclasses.replace(m2, nome="M3")
        with self.assertRaises(ValueError):
            dataclasses.replace(m2, nome="M4")
        self.assertEqual(dataclasses.replace(m3), m3)

    def test_combinacoes_invalidas(self):
        F, V = cfg.TipoModelo.FIXO, cfg.TipoModelo.VARIAVEL
        s = ("original_glcm_Imc1",)
        casos = {
            "nome_desconhecido": ("M5", F, (cfg.DIAMETRO,), False),
            "M3v_como_fixo": ("M3v", F, (cfg.DIAMETRO,), False),
            "M0_como_variavel": ("M0", V, (cfg.DIAMETRO,), False),
            "M0_com_radiomica": ("M0", F, (cfg.DIAMETRO, *s), False),
            "M0_vazio": ("M0", F, (), False),
            "M1_fora_de_ordem": ("M1", F, tuple(reversed(cfg.FEATURES_M1)), False),
            "M1_incompleto": ("M1", F, cfg.FEATURES_M1[:2], False),
            "M2v_vazio_sem_dummy": ("M2v", V, (), False),
            "M2v_com_features_e_dummy": ("M2v", V, s, True),
            "M3v_com_dummy": ("M3v", V, (cfg.DIAMETRO,), True),
            "M4v_com_dummy": ("M4v", V, cfg.FEATURES_M1, True),
            "M3v_sem_diametro_primeiro": ("M3v", V, (*s, cfg.DIAMETRO), False),
            "M4v_sem_referencia": ("M4v", V, s, False),
            "nodule_idx": ("M2v", V, (cfg.COLUNA_PROIBIDA,), False),
            "duplicada": ("M3v", V, (cfg.DIAMETRO, cfg.DIAMETRO), False),
        }
        for rotulo, (nome, tipo, features, dummy) in casos.items():
            with self.subTest(caso=rotulo), self.assertRaises(ValueError):
                cfg.EspecModelo(nome, tipo, features, usa_dummy=dummy)

    def test_tipos_invalidos(self):
        with self.assertRaises(TypeError):
            cfg.EspecModelo("M0", cfg.TipoModelo.FIXO, [cfg.DIAMETRO])
        with self.assertRaises(TypeError):
            cfg.EspecModelo("M2v", cfg.TipoModelo.VARIAVEL, (), usa_dummy=1)
        with self.assertRaises(ValueError):
            cfg.EspecModelo("M0", "fixo", (cfg.DIAMETRO,))

    def test_construcoes_validas_continuam_aceitas(self):
        self.assertEqual(cfg.EspecModelo("M0", cfg.TipoModelo.FIXO, (cfg.DIAMETRO,)), cfg.modelo_fixo("M0"))
        self.assertEqual(cfg.EspecModelo("M2v", cfg.TipoModelo.VARIAVEL, (), usa_dummy=True),
                         cfg.modelo_variavel("M2v", ()))
        dec = decisoes_com(P3=cfg.PAINEL_REPRODUZIDO)
        self.assertEqual(cfg.modelo_fixo("M4", dec), cfg.modelo_fixo("M4", dec))

    def test_marca_privada_fora_do_snapshot(self):
        m4 = cfg.modelo_fixo("M4", decisoes_com(P3=cfg.PAINEL_DECLARADO))
        serializado = cfg._jsonavel(m4)
        self.assertNotIn("_autorizacao", serializado)
        self.assertEqual(serializado["features"], list(m4.features))
        self.assertNotIn("_autorizacao", repr(m4))


class TestMapaDecisoes(unittest.TestCase):
    ESPERADO = {
        "verificacoes_estruturais": (),
        "etapa0_prerequisitos": (),
        "etapa1_piloto": ("P1", "P2"),
        "etapa2_cv_aninhada": ("P1", "P2"),
        "etapa2_cv_lista_fixa": ("P1", "P2", "P3"),
        "etapa3_validacao": ("P1", "P2", "P3"),
        "etapa4_congelamento": ("P1", "P2", "P3", "P5", "REVISAO_PROTOCOLO"),
        "etapa5_teste": ("P1", "P2", "P3", "P4", "P5", "REVISAO_PROTOCOLO"),
        "piloto_exploratorio": (),
    }

    def test_mapa_explicito(self):
        self.assertEqual(dict(cfg.DECISOES_POR_ETAPA), self.ESPERADO)

    def test_piloto_exploratorio_e_atividade_propria_e_somente_exploratoria(self):
        piloto = cfg.ATIVIDADE_PILOTO_EXPLORATORIO
        self.assertEqual(piloto, "piloto_exploratorio")
        self.assertNotIn(piloto, {e.value for e in cfg.Etapa} | {e.name for e in cfg.Etapa})
        self.assertNotEqual(piloto, cfg.Etapa.E1.value)
        self.assertEqual(cfg.DECISOES_POR_ETAPA[piloto], ())
        self.assertEqual(cfg.ATIVIDADES_SOMENTE_EXPLORATORIAS, (piloto,))
        self.assertEqual(cfg.snapshot()["atividades_somente_exploratorias"], [piloto])
        # Sem partições nem etapa próprias; nenhuma decisão aprovada.
        self.assertEqual(set(cfg.PARTICOES_POR_ETAPA), set(cfg.Etapa))
        self.assertTrue(all(not d.aprovada for d in cfg.DECISOES.values()))

    def test_cumulativo_ao_longo_das_etapas(self):
        etapas = list(cfg.Etapa)
        for anterior, seguinte in zip(etapas, etapas[1:]):
            with self.subTest(de=anterior.value, para=seguinte.value):
                self.assertLessEqual(set(cfg.DECISOES_POR_ETAPA[anterior.value]),
                                     set(cfg.DECISOES_POR_ETAPA[seguinte.value]))

    def test_cada_decisao_vale_a_partir_do_seu_prazo(self):
        for codigo, decisao in cfg.DECISOES.items():
            self.assertIsInstance(decisao.etapa_limite, cfg.Etapa)
            numero = int(re.search(r"Etapa (\d)", decisao.prazo).group(1))
            self.assertEqual(decisao.etapa_limite.ordem, numero, codigo)
            for etapa in cfg.Etapa:
                with self.subTest(decisao=codigo, etapa=etapa.value):
                    exigida = codigo in cfg.DECISOES_POR_ETAPA[etapa.value]
                    self.assertEqual(exigida, etapa.ordem >= numero)

    def test_prazos_iguais_a_secao_13_do_protocolo(self):
        texto = (REPO_ROOT / cfg.PROTOCOLO).read_text(encoding="utf-8")
        prazos = dict(re.findall(r"^\| (P[1-5]) \|.*\| antes da Etapa (\d) \|$", texto, flags=re.M))
        self.assertEqual(set(prazos), {"P1", "P2", "P3", "P4", "P5"})
        for codigo, numero in prazos.items():
            self.assertEqual(cfg.DECISOES[codigo].etapa_limite.ordem, int(numero), codigo)

    def test_verificacoes_estruturais_e_etapa0_sem_decisao(self):
        self.assertEqual(cfg.DECISOES_POR_ETAPA[cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS], ())
        self.assertEqual(cfg.DECISOES_POR_ETAPA[cfg.Etapa.E0.value], ())
        # Composição de M0, M1 e dos modelos variáveis não depende de decisão.
        cfg.modelo_fixo("M0"), cfg.modelo_fixo("M1")
        for nome in cfg.MODELOS_VARIAVEIS:
            cfg.modelo_variavel(nome, ("original_glcm_Imc1",))

    def test_spline_e_painel_exigem_valor_do_tipo_certo(self):
        with self.assertRaises(ValueError):
            cfg.spline_confirmado(decisoes_com(P2={"n_knots": 4}))
        with self.assertRaises(ValueError):
            cfg.painel_fixo(decisoes_com(P3=None))
        self.assertEqual(cfg.painel_fixo(decisoes_com(P3=cfg.PAINEL_REPRODUZIDO)), cfg.PAINEL_REPRODUZIDO)


class TestParametrosEIsolamento(unittest.TestCase):
    def test_seeds_e_limiares_do_protocolo(self):
        s, cv, b = cfg.ESPEC_SELECAO, cfg.ESPEC_CV, cfg.ESPEC_BOOTSTRAP
        self.assertEqual(cv.seeds_repeticoes(), tuple(range(42, 52)))
        self.assertEqual(cv.seed_piloto, cv.seeds_repeticoes()[0])
        self.assertEqual([s.seed_interna(j) for j in range(1, 6)], [101, 102, 103, 104, 105])
        with self.assertRaises(ValueError):
            s.seed_interna(0)
        self.assertEqual((s.seed_estabilidade, s.n_folds_estabilidade, s.n_folds_interna), (42, 5, 4))
        self.assertEqual((s.limiar_correlacao, s.limiar_coeficiente, s.limiar_estabilidade), (0.90, 1e-12, 3))
        self.assertEqual((b.n_replicas, b.seed, b.percentis, b.unidade), (2000, 42, (2.5, 97.5), "patient_id"))
        self.assertEqual(cv.workers_a_medir_no_piloto, (1, 2, 4))
        self.assertEqual(cfg.ORDEM_LINHAS, ("nodulo_id", "mergesort"))
        self.assertFalse(cfg.CALCULA_P_VALOR)

    def test_grade_c_igual_ao_logspace(self):
        import numpy as np

        valores = cfg.ESPEC_SELECAO.grade_C.valores()
        self.assertEqual(len(valores), 13)
        self.assertEqual(valores, tuple(float(v) for v in np.logspace(-4, 2, 13)))

    def test_isolamento_por_etapa(self):
        for etapa in (cfg.Etapa.E0, cfg.Etapa.E1, cfg.Etapa.E2):
            self.assertEqual(cfg.PARTICOES_POR_ETAPA[etapa], ("train",))
        self.assertNotIn("test", cfg.PARTICOES_POR_ETAPA[cfg.Etapa.E3])
        self.assertNotIn("validation", cfg.PARTICOES_POR_ETAPA[cfg.Etapa.E5])
        self.assertEqual([e for e, p in cfg.PARTICOES_POR_ETAPA.items() if "test" in p], [cfg.Etapa.E5])


class TestImutabilidade(unittest.TestCase):
    def test_dataclasses_congeladas(self):
        for obj, campo in ((cfg.ESPEC_AVALIACAO, "C"), (cfg.ESPEC_SELECAO, "max_iter"),
                           (cfg.ESPEC_CV, "n_repeticoes"), (cfg.ESPEC_BOOTSTRAP, "seed"),
                           (cfg.DECISOES["P3"], "status"), (cfg.modelo_fixo("M0"), "features")):
            with self.subTest(obj=type(obj).__name__), self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(obj, campo, None)

    def test_mapeamentos_somente_leitura(self):
        for mapa in (cfg.DECISOES, cfg.PAINEIS_CANDIDATOS, cfg.REFERENCIA, cfg.PARTICOES_POR_ETAPA,
                     cfg.PARTICOES_ESPERADAS, cfg.PARTICOES_ESPERADAS["train"]):
            with self.assertRaises(TypeError):
                mapa["x"] = None


class TestHash(unittest.TestCase):
    def test_deterministico_e_serializavel(self):
        h = cfg.hash_configuracao()
        self.assertRegex(h, r"^[0-9a-f]{64}$")
        self.assertEqual(h, cfg.hash_configuracao())
        json.dumps(cfg.snapshot(), allow_nan=False)

    def test_estavel_entre_processos(self):
        codigo = ("import sys; sys.path.insert(0, sys.argv[1]); "
                  "from src.incremental import config; print(config.hash_configuracao())")
        hashes = set()
        for semente in ("0", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": semente}
            saida = subprocess.run([sys.executable, "-B", "-c", codigo, str(REPO_ROOT)],
                                   capture_output=True, text=True, env=env, check=True)
            hashes.add(saida.stdout.strip())
        self.assertEqual(hashes, {cfg.hash_configuracao()})

    def test_muda_quando_uma_decisao_muda(self):
        base = cfg.hash_configuracao()
        self.assertNotEqual(base, cfg.hash_configuracao(decisoes_com(P3=cfg.PAINEL_DECLARADO)))
        self.assertNotEqual(cfg.hash_configuracao(decisoes_com(P3=cfg.PAINEL_DECLARADO)),
                            cfg.hash_configuracao(decisoes_com(P3=cfg.PAINEL_REPRODUZIDO)))


if __name__ == "__main__":
    unittest.main()
