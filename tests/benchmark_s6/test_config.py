"""Testes da configuração canônica do benchmark S6 (``src/benchmark_s6/config.py``).

Conferem a configuração contra os valores LITERAIS do pré-registro P9
(``docs/sprint6/preregistro_p9_benchmark.md``, versão 1.0, PR #11), a
estabilidade da serialização e do SHA-256, a recusa de configurações
inválidas, a geração do arquivo canônico sem sobrescrita, o isolamento em
relação à S4 e a ausência de leitura de dados. São sintéticos: não leem dados
de pacientes nem ajustam modelos.

Execução, a partir da raiz do repositório:

    python -B -m unittest discover -s tests/benchmark_s6 -v
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path
from types import MappingProxyType

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark_s6 import adaptadores_s4 as s4  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import coordenador  # noqa: E402
from src.incremental import guardas, rastreabilidade  # noqa: E402

#: SHA-256 da configuração canônica. Qualquer mudança de parâmetro muda este
#: valor e exige revisão do pré-registro (P9, Seções 14.4 e 15).
SHA256_ESPERADO = "a0d096a5415244f17f40360a2a5a8901bc8af1da5cf8687b891f920e1ee122b3"

C = cfg.CONFIGURACAO
PACOTE = REPO_ROOT / "src" / "benchmark_s6"


def substituir(obj, **mudancas):
    return dataclasses.replace(obj, **mudancas)


def com_modelo(codigo, **mudancas):
    modelos = tuple(substituir(m, **mudancas) if m.codigo == codigo else m for m in C.modelos)
    return substituir(C, modelos=modelos)


# --------------------------------------------------------------------------- #
# Conferência literal com a P9
# --------------------------------------------------------------------------- #


class TestEspecificacaoP9(unittest.TestCase):

    def test_dados_e_isolamento_secao_3(self):
        d = C.dados
        self.assertEqual(d.particoes, ("train", "validation"))
        self.assertEqual(d.particao_excluida, "test")
        self.assertNotIn("test", d.particoes)
        self.assertEqual(dict(d.contagem_base), {"pacientes": 353, "nodulos": 509, "classe_0": 258, "classe_1": 251})
        self.assertEqual(dict(d.contagens_particoes["train"]),
                         {"pacientes": 291, "nodulos": 417, "classe_0": 209, "classe_1": 208})
        self.assertEqual(dict(d.contagens_particoes["validation"]),
                         {"pacientes": 62, "nodulos": 92, "classe_0": 49, "classe_1": 43})
        self.assertEqual(d.tabela.caminho, "data/modeling_table_sprint3.csv")
        self.assertEqual(d.tabela.sha256, s4.TABELA_OFICIAL.sha256)
        self.assertEqual(d.split.caminho, "reports/sprint3/baseline/patient_split.csv")
        self.assertEqual(d.split.sha256_lf, s4.SPLIT_OFICIAL.sha256_lf)
        self.assertEqual(C.execucao.registros_exigidos, ("P9", "S4_E3_REGISTRADA"))

    def test_caracteristicas_e_faixa_secoes_3_e_7(self):
        self.assertEqual(C.caracteristicas, ("original_shape_MeshVolume", "original_shape_Maximum3DDiameter",
                                             "original_shape_Sphericity"))
        f = C.faixa
        self.assertEqual((f.variavel, f.inferior, f.superior, f.inclui_inferior, f.inclui_superior, f.unidade),
                         ("original_shape_Maximum3DDiameter", 12.0, 30.0, True, False, "mm"))
        self.assertEqual(dict(f.contagem_esperada), {"pacientes": 187, "nodulos": 226, "classe_0": 91, "classe_1": 135})
        self.assertTrue(f.contem(12.0))
        self.assertFalse(f.contem(30.0))
        self.assertFalse(f.contem(11.999))

    def test_dobras_e_sementes_secao_4_e_d12(self):
        d = C.dobras
        self.assertEqual((d.n_dobras_externas, d.n_repeticoes, d.n_dobras_internas), (5, 10, 5))
        self.assertEqual(d.sementes_externas(), tuple(range(42, 52)))
        self.assertEqual(d.semente_interna(0, 0), 1000)
        self.assertEqual(d.semente_interna(3, 4), 1034)
        self.assertEqual(d.semente_interna(9, 4), 1094)
        self.assertEqual(len(set(d.sementes_internas())), 50)
        self.assertEqual((d.estratificacao, d.grupo, d.embaralhar), ("target_binary", "patient_id", True))
        with self.assertRaises(cfg.ConfiguracaoInvalida):
            d.semente_interna(10, 0)

    def test_modelos_e_grades_secao_5(self):
        self.assertEqual(tuple(m.codigo for m in C.modelos), ("lr_fixa", "lr_tuning", "svm", "rf", "xgb"))
        self.assertEqual({m.codigo: m.n_configuracoes for m in C.modelos},
                         {"lr_fixa": 1, "lr_tuning": 13, "svm": 30, "rf": 15, "xgb": 32})
        lr_fixa, lr_tuning = C.modelo("lr_fixa"), C.modelo("lr_tuning")
        self.assertEqual(dict(lr_fixa.fixos), {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "class_weight": None,
                                               "max_iter": 1000, "random_state": 42})
        self.assertEqual(lr_tuning.valores_de("C"), tuple(float(v) for v in np.logspace(-3, 3, 13)))
        self.assertEqual(lr_tuning.valores_de("C")[6], 1.0)
        svm = C.modelo("svm")
        self.assertEqual(svm.valores_de("C"), (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0))
        self.assertEqual(svm.valores_de("gamma"), (0.01, 0.03, 0.1, 0.3, 1.0))
        self.assertEqual((svm.pontuacao, svm.metricas_probabilisticas, svm.calibracao),
                         ("decision_function", False, False))
        self.assertEqual(dict(svm.fixos), {"kernel": "rbf", "probability": False, "class_weight": None,
                                           "random_state": 42})
        rf = C.modelo("rf")
        self.assertEqual(rf.valores_de("min_samples_leaf"), (20, 10, 5, 3, 1))
        self.assertEqual(rf.valores_de("max_features"), (1, 2, 3))
        self.assertEqual(rf.fixos["n_estimators"], 500)
        xgb = C.modelo("xgb")
        self.assertEqual(xgb.estimador, "xgboost.XGBClassifier")
        self.assertEqual([p.nome for p in xgb.grade], ["max_depth", "n_estimators", "learning_rate", "min_child_weight"])
        self.assertEqual(xgb.valores_de("min_child_weight"), (5, 1))
        self.assertEqual(dict(xgb.fixos), {"objective": "binary:logistic", "tree_method": "hist", "subsample": 0.8,
                                           "colsample_bytree": 1.0, "reg_lambda": 1.0, "n_jobs": 1, "random_state": 42})
        self.assertFalse(xgb.parada_antecipada)

    def test_ordem_de_simplicidade_e_volume_secoes_5_2_e_15(self):
        self.assertEqual(dict(C.modelo("svm").configuracoes()[0]), {"C": 0.01, "gamma": 0.01})
        self.assertEqual(dict(C.modelo("svm").configuracoes()[1]), {"C": 0.01, "gamma": 0.03})
        self.assertEqual(dict(C.modelo("rf").configuracoes()[0]), {"min_samples_leaf": 20, "max_features": 1})
        self.assertEqual(dict(C.modelo("xgb").configuracoes()[0]),
                         {"max_depth": 1, "n_estimators": 100, "learning_rate": 0.03, "min_child_weight": 5})
        self.assertEqual(C.ajustes_por_dobra_externa(), 455)
        self.assertEqual(C.ajustes_totais(), 22_750)
        self.assertEqual(C.n_unidades(), 250)
        a = C.ajuste_interno
        self.assertEqual((a.scoring, a.tolerancia_empate, a.desempate, a.error_score, a.n_jobs),
                         ("roc_auc", 1e-9, "primeira_na_ordem_de_simplicidade", "raise", 1))

    def test_validade_metricas_e_estados_secoes_6_e_7_d3(self):
        v = C.validade
        self.assertEqual((v.min_positivos, v.min_negativos, v.min_dobras_validas_por_repeticao,
                          v.min_repeticoes_validas, v.max_fracao_descarte_bootstrap),
                         (5, 5, 4, 8, 0.01))
        self.assertEqual(v.estados, ("FT", "AVISO", "DD", "RI", "NE", "RD", "IQ"))
        m = C.metricas
        self.assertEqual(tuple(d.codigo for d in m.desfechos), ("G", "F"))
        self.assertEqual((m.unidade, m.corte_log_loss, m.limiar_probabilidade, m.limiar_decisao_nativa),
                         ("dobra_externa", 1e-15, 0.5, 0.0))

    def test_bootstrap_contrastes_e_multiplicidade_secoes_8_e_9_d4(self):
        b = C.bootstrap
        self.assertEqual((b.n_replicas, b.gerador, b.semente, b.reajuste, b.metodo),
                         (10_000, "numpy.random.default_rng", 42, False, "percentil"))
        self.assertEqual(b.percentis_descritivos(), (2.5, 97.5))
        k = C.contrastes
        self.assertEqual((k.comparador, k.desafiantes, k.desfechos, k.alfa, k.correcao),
                         ("lr_tuning", ("rf", "xgb", "svm"), ("G", "F"), 0.05, "bonferroni"))
        self.assertEqual(k.n_familia, 6)
        inferior, superior = k.percentis_familia()
        self.assertEqual((round(inferior, 5), round(superior, 5)), (0.41667, 99.58333))
        self.assertIn(("lr_tuning", "lr_fixa"), k.descritivos)

    def test_selecao_secao_10_d5(self):
        s = C.selecao
        self.assertEqual((s.padrao, s.delta, s.natureza_delta, s.fracao_min_repeticoes_positivas,
                          s.delta_minimo_faixa, s.margem_desempate),
                         ("lr_tuning", 0.02, "operacional_sem_interpretacao_clinica", 0.8, 0.0, 0.005))
        self.assertEqual(s.ordem_simplicidade, ("lr_tuning", "svm", "rf", "xgb"))
        self.assertFalse(s.ic_e_criterio)
        self.assertFalse(s.descarte_bootstrap_e_criterio)

    def test_saidas_e_execucao_secoes_11_a_15(self):
        figuras = {f.codigo: f.intervalo for f in C.saidas.figuras}
        self.assertEqual(figuras["F1"], "ic95_descritivo_sem_correcao")
        self.assertEqual(figuras["F5"], "dispersao_entre_dobras_nao_e_intervalo_de_confianca")
        self.assertEqual(C.saidas.vocabulario_vetado[:2], ("significativo", "significância"))
        e = C.execucao
        self.assertEqual(e.modos, ("configuracao", "implementacao", "definitivo", "verificar"))
        self.assertEqual(dict(e.codigos_saida), {"concluido": 0, "bloqueio_aprovacao": 2, "falha_tecnica": 3,
                                                 "integridade": 4, "privacidade_ou_vocabulario": 5})
        self.assertFalse(e.altera_parametros_por_tempo)
        self.assertFalse(e.implementacao_metricas_externas)
        self.assertEqual(e.ambiente["xgboost"], "2.1.4")
        self.assertEqual(e.ambiente["python"], "3.9.13")

    def test_referencias(self):
        r = C.referencias
        self.assertEqual(r.preregistro, "docs/sprint6/preregistro_p9_benchmark.md")
        self.assertEqual(r.arquivo_configuracao, "docs/sprint6/config_benchmark_p9.json")
        self.assertEqual(r.commit_emenda, "573bc1729f7dec33c73675ccc9c48c97d5e50c5a")
        self.assertEqual(dict(r.arquivos_s4), dict(s4.ARQUIVOS_S4))


# --------------------------------------------------------------------------- #
# Serialização canônica e hash
# --------------------------------------------------------------------------- #


class TestSerializacaoEHash(unittest.TestCase):

    def test_sha256_fixado(self):
        self.assertEqual(cfg.sha256_configuracao(), SHA256_ESPERADO)

    def test_reconstrucao_identica(self):
        outra = cfg.construir_configuracao()
        self.assertEqual(outra, C)
        self.assertEqual(cfg.bytes_arquivo(outra), cfg.bytes_arquivo(C))

    def test_hash_coincide_com_guardas_e_rastreabilidade(self):
        conteudo = cfg.conteudo_canonico()
        self.assertEqual(guardas.hash_valor_decisao(conteudo), SHA256_ESPERADO)
        self.assertEqual(rastreabilidade.sha256_canonico(conteudo), SHA256_ESPERADO)
        # Contrato documentado da S4 (chaves ordenadas, UTF-8, sem espaços,
        # sem NaN), recalculado sem o código da S4.
        texto = json.dumps(conteudo, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                           allow_nan=False)
        self.assertEqual(hashlib.sha256(texto.encode("utf-8")).hexdigest(), SHA256_ESPERADO)
        self.assertRegex(SHA256_ESPERADO, r"\A[0-9a-f]{64}\Z")

    def test_forma_canonica_e_ida_e_volta(self):
        dados = cfg.bytes_arquivo()
        self.assertTrue(dados.endswith(b"\n") and dados.count(b"\n") == 1)
        conteudo = json.loads(dados.decode("utf-8"))
        self.assertEqual(conteudo, cfg.conteudo_canonico())
        self.assertEqual(dados, (rastreabilidade.json_canonico(conteudo) + "\n").encode("utf-8"))

    def test_derivados_fora_do_json(self):
        conteudo = cfg.conteudo_canonico()
        grade_c = conteudo["modelos"][1]["grade"][0]["valores"]
        self.assertEqual(grade_c, {"tipo": "logspace", "inicio": -3, "fim": 3, "n": 13})
        texto = cfg.texto_canonico()
        self.assertNotIn("0.0031622776601683794", texto)
        self.assertNotIn("0.4166", texto)

    def test_estavel_entre_processos(self):
        codigo = "from src.benchmark_s6 import config as c; print(c.sha256_configuracao())"
        obtidos = set()
        for semente in ("0", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": semente}
            saida = subprocess.run([sys.executable, "-B", "-c", codigo], cwd=REPO_ROOT, env=env,
                                   capture_output=True, text=True, check=True, timeout=120)
            obtidos.add(saida.stdout.strip())
        self.assertEqual(obtidos, {SHA256_ESPERADO})

    def test_imutavel(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            C.bootstrap.n_replicas = 2000  # type: ignore[misc]
        with self.assertRaises(TypeError):
            C.modelo("rf").fixos["n_estimators"] = 300  # type: ignore[index]
        with self.assertRaises(TypeError):
            C.dados.contagem_base["nodulos"] = 1  # type: ignore[index]

    def test_nenhum_conteiner_mutavel_alcancavel(self):
        """Só dataclasses congeladas, tuplas, MappingProxyType e escalares."""
        mutaveis = []

        def visitar(obj, rota):
            if dataclasses.is_dataclass(obj):
                if not type(obj).__dataclass_params__.frozen:
                    mutaveis.append(rota)
                for campo in dataclasses.fields(obj):
                    visitar(getattr(obj, campo.name), f"{rota}.{campo.name}")
            elif isinstance(obj, MappingProxyType):
                for chave, valor in obj.items():
                    visitar(valor, f"{rota}[{chave!r}]")
            elif type(obj) is tuple:
                for i, valor in enumerate(obj):
                    visitar(valor, f"{rota}[{i}]")
            elif not (obj is None or type(obj) in (str, int, float, bool)):
                mutaveis.append(f"{rota}: {type(obj).__name__}")

        visitar(C, "C")
        for modelo in C.modelos:
            for i, configuracao in enumerate(modelo.configuracoes()):
                visitar(configuracao, f"{modelo.codigo}.configuracoes()[{i}]")
        self.assertEqual(mutaveis, [])

    def test_entrada_mutavel_nao_vaza(self):
        fixos = dict(C.modelo("rf").fixos)
        modelo = substituir(C.modelo("rf"), fixos=fixos)
        fixos["n_estimators"] = 1
        self.assertEqual(modelo.fixos["n_estimators"], 500)
        contagens = {p: dict(c) for p, c in C.dados.contagens_particoes.items()}
        dados = substituir(C.dados, contagens_particoes=contagens)
        contagens["train"]["nodulos"] = 1
        self.assertEqual(dados.contagens_particoes["train"]["nodulos"], 417)
        ambiente = dict(C.execucao.ambiente)
        execucao = substituir(C.execucao, ambiente=ambiente)
        ambiente["xgboost"] = "3.0.0"
        self.assertEqual(execucao.ambiente["xgboost"], "2.1.4")

    def test_lista_ou_tipo_errado_recusados(self):
        def campos(obj):
            return {campo.name: getattr(obj, campo.name) for campo in dataclasses.fields(obj)}

        casos = [
            (C, {"modelos": list(C.modelos)}),
            (C, {"dados": campos(C.dados)}),
            (C, {"bootstrap": types.SimpleNamespace(**campos(C.bootstrap))}),
            (C.metricas, {"desfechos": list(C.metricas.desfechos)}),
            (C.saidas, {"figuras": list(C.saidas.figuras)}),
            (C.modelo("svm"), {"grade": list(C.modelo("svm").grade)}),
            (C.dados, {"tabela": types.SimpleNamespace(**campos(C.dados.tabela))}),
            (C.referencias, {"arquivos_s4": list(C.referencias.arquivos_s4.items())}),
        ]
        for objeto, mudanca in casos:
            with self.subTest(campo=next(iter(mudanca))), self.assertRaises(cfg.ConfiguracaoInvalida):
                substituir(objeto, **mudanca)

    def test_sem_identificador_de_paciente(self):
        self.assertNotRegex(cfg.texto_canonico(), r"LIDC-IDRI-\d{4}")


# --------------------------------------------------------------------------- #
# Recusa de configurações inválidas
# --------------------------------------------------------------------------- #


class TestRecusa(unittest.TestCase):

    def assertInvalida(self, funcao, *args, **kwargs):
        with self.assertRaises(cfg.ConfiguracaoInvalida):
            funcao(*args, **kwargs)

    def test_bootstrap(self):
        self.assertInvalida(substituir, C.bootstrap, n_replicas=10_000.0)
        self.assertInvalida(substituir, C.bootstrap, n_replicas=500)
        self.assertInvalida(substituir, C.bootstrap, n_replicas=True)
        self.assertInvalida(substituir, C.bootstrap, reajuste=True)
        self.assertInvalida(substituir, C.bootstrap, alfa_descritivo=float("nan"))

    def test_teste_entre_particoes(self):
        self.assertInvalida(substituir, C.dados, particoes=("train", "validation", "test"))
        self.assertInvalida(substituir, C.dados, particao_excluida="validation")

    def test_contagens_incoerentes(self):
        self.assertInvalida(substituir, C.dados, contagem_base={"pacientes": 353, "nodulos": 510,
                                                                "classe_0": 259, "classe_1": 251})

    def test_alteracao_por_tempo(self):
        self.assertInvalida(substituir, C.execucao, altera_parametros_por_tempo=True)
        self.assertInvalida(substituir, C.execucao, implementacao_metricas_externas=True)

    def test_ambiente_incompleto(self):
        ambiente = {k: v for k, v in C.execucao.ambiente.items() if k != "xgboost"}
        self.assertInvalida(substituir, C.execucao, ambiente=ambiente)

    def test_svm_com_probabilidade_ou_calibracao(self):
        self.assertInvalida(com_modelo, "svm", metricas_probabilisticas=True)
        self.assertInvalida(com_modelo, "svm", calibracao=True)
        self.assertInvalida(com_modelo, "xgb", parada_antecipada=True)

    def test_grades_invalidas(self):
        self.assertInvalida(cfg.ParametroGrade, "C", (1.0, 1.0))
        self.assertInvalida(cfg.ParametroGrade, "C", ())
        self.assertInvalida(cfg.ParametroGrade, "C", (True, False))
        self.assertInvalida(cfg.ParametroGrade, "C", (float("nan"),))
        self.assertInvalida(cfg.GradeLog, 3, -3, 13)
        self.assertInvalida(cfg.GradeLog, -3.0, 3, 13)
        grade = (cfg.ParametroGrade("n_estimators", (100, 300)),)
        self.assertInvalida(com_modelo, "rf", grade=grade)  # fixo e ajustado ao mesmo tempo

    def test_parametro_fixo_nao_finito(self):
        self.assertInvalida(com_modelo, "xgb", fixos={**C.modelo("xgb").fixos, "subsample": float("inf")})

    def test_comparador_sem_c_igual_a_1(self):
        self.assertInvalida(com_modelo, "lr_tuning", grade=(cfg.ParametroGrade("C", cfg.GradeLog(-3, 3, 12)),))

    def test_faixa_invalida(self):
        self.assertInvalida(substituir, C.faixa, inferior=30.0, superior=12.0)
        self.assertInvalida(substituir, C.faixa, inclui_superior=True)
        self.assertInvalida(substituir, C, faixa=substituir(C.faixa, variavel="original_shape_MeshVolume_x"))

    def test_sementes_que_colidem(self):
        self.assertInvalida(substituir, C.dobras, semente_interna_passo_repeticao=4)
        self.assertInvalida(substituir, C.dobras, semente_interna_base=45, semente_interna_passo_repeticao=10)
        self.assertInvalida(substituir, C.dobras, estratificacao="faixa")

    def test_caracteristicas(self):
        self.assertInvalida(substituir, C, caracteristicas=C.caracteristicas + ("original_glcm_Imc2",))
        self.assertInvalida(substituir, C, caracteristicas=tuple(reversed(C.caracteristicas)))

    def test_modelos_e_contrastes(self):
        self.assertInvalida(substituir, C, modelos=C.modelos + (C.modelo("rf"),))
        self.assertInvalida(substituir, C, contrastes=substituir(C.contrastes, desafiantes=("rf", "xgb")))
        self.assertInvalida(substituir, C, contrastes=substituir(C.contrastes, comparador="lr_fixa",
                                                                 descritivos=(("rf", "xgb"),)))
        self.assertInvalida(substituir, C.contrastes, correcao="holm")

    def test_selecao(self):
        self.assertInvalida(substituir, C.selecao, ic_e_criterio=True)
        self.assertInvalida(substituir, C.selecao, descarte_bootstrap_e_criterio=True)
        self.assertInvalida(substituir, C.selecao, natureza_delta="clinica")
        self.assertInvalida(substituir, C.selecao, delta=0.0)
        self.assertInvalida(substituir, C, selecao=substituir(C.selecao, padrao="rf",
                                                              ordem_simplicidade=("rf", "lr_tuning", "svm", "xgb")))
        self.assertInvalida(substituir, C, selecao=substituir(C.selecao,
                                                              ordem_simplicidade=("lr_tuning", "xgb", "rf", "svm")))

    def test_estados_e_imputacao(self):
        self.assertInvalida(substituir, C.validade, estados=("FT", "DD"))
        self.assertInvalida(substituir, C.validade, imputacao_de_auc=True)

    def test_figuras(self):
        f1 = next(f for f in C.saidas.figuras if f.codigo == "F1")
        self.assertInvalida(substituir, f1, intervalo="bonferroni")
        self.assertInvalida(cfg.EspecFigura, "F5", "roc", "ic95")
        self.assertInvalida(substituir, C.saidas, figuras=C.saidas.figuras[:-1])

    def test_referencias_s4(self):
        arquivos = {**s4.ARQUIVOS_S4, "src/incremental/config.py": "0" * 64}
        self.assertInvalida(substituir, C.referencias, arquivos_s4=arquivos)
        self.assertInvalida(substituir, C.referencias, commit_emenda="573bc17")

    def test_tipo_nao_serializavel(self):
        with self.assertRaises(TypeError):
            cfg.conteudo_canonico({"nao": "configuracao"})


# --------------------------------------------------------------------------- #
# Arquivo canônico
# --------------------------------------------------------------------------- #


class TestArquivo(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.alvo = self.dir / "docs" / "sprint6" / "config_benchmark_p9.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_gera_e_nao_sobrescreve(self):
        self.assertEqual(cfg.gerar_arquivo(self.alvo), (SHA256_ESPERADO, True))
        self.assertEqual(self.alvo.read_bytes(), cfg.bytes_arquivo())
        self.assertEqual(cfg.gerar_arquivo(self.alvo), (SHA256_ESPERADO, False))
        self.assertEqual(sorted(p.name for p in self.alvo.parent.iterdir()), [self.alvo.name])

    def test_crlf_tolerado(self):
        self.alvo.parent.mkdir(parents=True)
        self.alvo.write_bytes(cfg.bytes_arquivo().replace(b"\n", b"\r\n"))
        self.assertEqual(cfg.conferir_arquivo(self.alvo), SHA256_ESPERADO)

    def test_recusa_forma_nao_canonica(self):
        self.alvo.parent.mkdir(parents=True)
        bonito = json.dumps(cfg.conteudo_canonico(), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        self.alvo.write_text(bonito, encoding="utf-8")
        with self.assertRaises(cfg.ConfiguracaoDivergente):
            cfg.conferir_arquivo(self.alvo)

    def test_recusa_conteudo_diferente_e_nao_sobrescreve(self):
        self.alvo.parent.mkdir(parents=True)
        conteudo = cfg.conteudo_canonico()
        conteudo["bootstrap"]["n_replicas"] = 2000
        original = (rastreabilidade.json_canonico(conteudo) + "\n").encode("utf-8")
        self.alvo.write_bytes(original)
        with self.assertRaises(cfg.ConfiguracaoDivergente):
            cfg.conferir_arquivo(self.alvo)
        with self.assertRaises(cfg.ConfiguracaoDivergente):
            cfg.gerar_arquivo(self.alvo)
        self.assertEqual(self.alvo.read_bytes(), original)

    def test_recusa_chave_repetida_e_nan(self):
        self.alvo.parent.mkdir(parents=True)
        for texto in ('{"a":1,"a":2}\n', '{"a":NaN}\n', "não é json\n"):
            self.alvo.write_text(texto, encoding="utf-8")
            with self.assertRaises(cfg.ConfiguracaoDivergente):
                cfg.ler_arquivo(self.alvo)


# --------------------------------------------------------------------------- #
# Isolamento: S4 e dados
# --------------------------------------------------------------------------- #


def _importacoes(caminho: Path):
    arvore = ast.parse(caminho.read_text(encoding="utf-8"))
    return [n for n in ast.walk(arvore) if isinstance(n, (ast.Import, ast.ImportFrom))]


class TestIsolamento(unittest.TestCase):

    def test_adaptador_so_importa_nomes_publicos(self):
        importados = {}
        for no in _importacoes(PACOTE / "adaptadores_s4.py"):
            if isinstance(no, ast.ImportFrom) and no.level == 2 and (no.module or "").startswith("incremental"):
                arquivo = f"src/{no.module.replace('.', '/')}.py"
                nomes = tuple(a.name for a in no.names)
                self.assertTrue(all(not n.startswith("_") for n in nomes), nomes)
                importados[arquivo] = nomes
        self.assertEqual({k: set(v) for k, v in importados.items()},
                         {k: set(v) for k, v in s4.SIMBOLOS_S4.items()})

    def test_so_o_adaptador_importa_a_s4(self):
        for caminho in PACOTE.glob("*.py"):
            if caminho.name == "adaptadores_s4.py":
                continue
            for no in _importacoes(caminho):
                modulos = [no.module or ""] if isinstance(no, ast.ImportFrom) else [a.name for a in no.names]
                nivel = getattr(no, "level", 0)
                for modulo in modulos:
                    self.assertFalse("incremental" in modulo or nivel >= 2,
                                     f"{caminho.name} importa a S4 diretamente: {modulo}")

    def test_nenhum_acesso_privado_ou_dinamico_a_s4(self):
        for caminho in PACOTE.glob("*.py"):
            arvore = ast.parse(caminho.read_text(encoding="utf-8"))
            for no in ast.walk(arvore):
                if isinstance(no, ast.Attribute) and isinstance(no.value, ast.Name) and no.value.id == "s4":
                    self.assertFalse(no.attr.startswith("_"), f"{caminho.name}: s4.{no.attr}")
                self.assertFalse(isinstance(no, ast.Name) and no.id == "__import__", caminho.name)
                if isinstance(no, (ast.Import, ast.ImportFrom)):
                    modulos = [no.module or ""] if isinstance(no, ast.ImportFrom) else [a.name for a in no.names]
                    self.assertNotIn("importlib", modulos, caminho.name)

    def test_arquivos_s4_fixados(self):
        self.assertEqual(s4.verificar_arquivos_s4(), s4.ARQUIVOS_S4)

    def test_arquivo_s4_alterado_e_recusado(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            for relativo in s4.ARQUIVOS_S4:
                destino = raiz / relativo
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_bytes((REPO_ROOT / relativo).read_bytes())
            self.assertEqual(s4.verificar_arquivos_s4(raiz), s4.ARQUIVOS_S4)
            (raiz / "src/incremental/folds.py").write_bytes(b"# alterado\n")
            with self.assertRaises(s4.ArquivoS4Divergente):
                s4.verificar_arquivos_s4(raiz)
            (raiz / "src/incremental/folds.py").unlink()
            with self.assertRaises(s4.ArquivoS4Divergente):
                s4.verificar_arquivos_s4(raiz)

    def test_nenhum_dado_lido(self):
        proibidos = (REPO_ROOT / "data").resolve(), s4.SPLIT_OFICIAL.caminho.resolve()
        abrir = io.open

        def vigiado(arquivo, *args, **kwargs):
            if isinstance(arquivo, (str, bytes, os.PathLike)):
                caminho = Path(os.fsdecode(arquivo)).resolve()
                if caminho == proibidos[1] or proibidos[0] in caminho.parents:
                    raise AssertionError(f"leitura de dado: {caminho}")
            return abrir(arquivo, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch("io.open", vigiado):
            cfg.construir_configuracao()
            cfg.sha256_configuracao()
            s4.verificar_arquivos_s4()
            cfg.gerar_arquivo(Path(tmp) / "c.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(coordenador.main(["--modo", "configuracao"]), 0)

    def test_nenhum_dado_aberto_no_processo(self):
        """Gancho de auditoria (evento ``open``) num processo novo: pega também
        aberturas fora de ``io.open``, inclusive as feitas na importação."""
        codigo = (
            "import json, os, sys, tempfile\n"
            "abertos = []\n"
            "def gancho(evento, args):\n"
            "    if evento == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):\n"
            "        abertos.append(os.path.abspath(os.fsdecode(args[0])))\n"
            "sys.addaudithook(gancho)\n"
            "from src.benchmark_s6 import coordenador\n"
            "with tempfile.TemporaryDirectory() as tmp:\n"
            "    alvo = os.path.join(tmp, 'c.json')\n"
            "    codigos = [coordenador.main(['--modo', 'configuracao', *extra])\n"
            "               for extra in ([], ['--gerar', alvo], ['--conferir', alvo])]\n"
            "print(json.dumps({'codigos': codigos, 'abertos': abertos}))\n"
        )
        resultado = subprocess.run([sys.executable, "-B", "-c", codigo], cwd=REPO_ROOT, capture_output=True,
                                   text=True, check=True, timeout=120)
        relatorio = json.loads(resultado.stdout.strip().splitlines()[-1])
        self.assertEqual(relatorio["codigos"], [0, 0, 0])
        # Comparação textual: o processo registra caminhos absolutos, e
        # ``resolve`` falha em dispositivos como ``nul``.
        abertos = [os.path.normcase(p) for p in relatorio["abertos"]]
        self.assertTrue(any(p.endswith("c.json") for p in abertos), "o gancho não registrou aberturas")
        proibidos = [os.path.normcase(str(REPO_ROOT / pasta)) for pasta in ("data", "reports")]
        for caminho in abertos:
            with self.subTest(caminho=os.path.basename(caminho)):
                self.assertFalse(any(caminho == p or caminho.startswith(p + os.sep) for p in proibidos), caminho)
                self.assertNotIn(os.path.splitext(caminho)[1], {".csv", ".parquet", ".xlsx", ".npy", ".npz", ".pkl"})


# --------------------------------------------------------------------------- #
# Coordenador
# --------------------------------------------------------------------------- #


class TestCoordenador(unittest.TestCase):

    def _executar(self, *argumentos):
        saida, erro = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(saida), contextlib.redirect_stderr(erro):
            codigo = coordenador.main(list(argumentos))
        return codigo, saida.getvalue(), erro.getvalue()

    def test_imprime_sha(self):
        codigo, saida, _ = self._executar("--modo", "configuracao")
        self.assertEqual(codigo, 0)
        self.assertIn(f"sha256_configuracao={SHA256_ESPERADO}", saida)

    def test_gerar_e_conferir(self):
        with tempfile.TemporaryDirectory() as tmp:
            alvo = Path(tmp) / "config_benchmark_p9.json"
            self.assertEqual(self._executar("--modo", "configuracao", "--gerar", str(alvo))[0], 0)
            self.assertEqual(self._executar("--modo", "configuracao", "--conferir", str(alvo))[0], 0)
            alvo.write_text('{"a":1}\n', encoding="utf-8")
            codigo, _, erro = self._executar("--modo", "configuracao", "--conferir", str(alvo))
            self.assertEqual(codigo, 4)
            self.assertIn("erro de integridade", erro)
            self.assertEqual(self._executar("--modo", "configuracao", "--conferir", str(Path(tmp) / "x.json"))[0], 4)

    def test_erro_de_argumentos_sai_com_64(self):
        invalidos = [
            *(["--modo", modo] for modo in ("implementacao", "definitivo", "verificar", "x")),
            [],
            ["--modo", "configuracao", "--gerar", "a.json", "--conferir", "a.json"],
            ["--modo", "configuracao", "--desconhecido"],
            ["--modo", "configuracao", "--gerar"],
        ]
        for argumentos in invalidos:
            with self.subTest(argumentos=argumentos):
                codigo, saida, erro = self._executar(*argumentos)
                self.assertEqual(codigo, coordenador.CODIGO_ERRO_USO)
                self.assertEqual(saida, "")
                self.assertIn("erro:", erro)

    def test_codigo_2_reservado_ao_bloqueio_de_aprovacao(self):
        codigos = dict(C.execucao.codigos_saida)
        self.assertEqual(coordenador.CODIGO_ERRO_USO, 64)
        self.assertNotIn(coordenador.CODIGO_ERRO_USO, codigos.values())
        self.assertEqual(codigos["bloqueio_aprovacao"], 2)

    def test_ajuda_sai_com_0(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as saida:
            coordenador.main(["--help"])
        self.assertEqual(saida.exception.code, 0)

    def test_codigo_de_saida_do_processo(self):
        executar = [sys.executable, "-B", "-m", "src.benchmark_s6", "--modo"]
        for modo, esperado in (("configuracao", 0), ("definitivo", coordenador.CODIGO_ERRO_USO)):
            with self.subTest(modo=modo):
                resultado = subprocess.run([*executar, modo], cwd=REPO_ROOT, capture_output=True,
                                           text=True, timeout=120)
                self.assertEqual(resultado.returncode, esperado, resultado.stderr)


if __name__ == "__main__":
    unittest.main()
