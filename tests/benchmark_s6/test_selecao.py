"""Testes de ``src/benchmark_s6/selecao.py`` (P9, Seção 10). Entradas escritas à mão, em JSON; nada é treinado."""

from __future__ import annotations

import ast
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sinteticos as sn  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import consolidacao as co  # noqa: E402
from src.benchmark_s6 import metricas as mt  # noqa: E402
from src.benchmark_s6 import selecao as sl  # noqa: E402
from src.benchmark_s6 import unidades as un  # noqa: E402

C = cfg.CONFIGURACAO
MODULO = Path(sl.__file__)
DESAFIANTES = ("rf", "xgb", "svm")


def contraste(delta, positivas=10, validas=10, estimavel=True):
    return {"estimavel": estimavel, "delta": delta if estimavel else None, "n_repeticoes_validas": validas,
            "n_repeticoes_positivas": positivas,
            "fracao_repeticoes_positivas": positivas / validas if estimavel else None}


def montar(desafiantes, ft=0, esperadas=250, concluidas=250, g_estimavel=True, f_estimavel=True):
    return {"versao_esquema": sl.VERSAO_ESQUEMA, "parametros": sl.parametros(),
            "unidades": {"esperadas": esperadas, "concluidas": concluidas, "falhas_tecnicas_pendentes": ft},
            "desfechos_estimaveis": {"G": g_estimavel, "F": f_estimavel},
            "desafiantes": {m: {"avisos_convergencia": d["avisos_convergencia"],
                                "contrastes": {"G": d["contraste_G"], "F": d["contraste_F"]}}
                            for m, d in desafiantes.items()}}


def entrada(**por_modelo):
    """Entrada em que todo desafiante cumpre C1 a C4, com ``dG``, ``dF``, ``rep`` e ``avisos`` ajustáveis por modelo."""
    desafiantes = {}
    for m in DESAFIANTES:
        a = por_modelo.get(m, {})
        desafiantes[m] = {
            "avisos_convergencia": a.get("avisos", 0),
            "contraste_G": a.get("G", contraste(a.get("dG", 0.03), *a.get("rep", (10, 10)))),
            "contraste_F": a.get("F", contraste(a.get("dF", 0.01))),
        }
    return montar(desafiantes)


def sem_os_outros(*mantidos, **ajustes):
    """``entrada`` em que só os modelos de ``mantidos`` podem ser elegíveis (os demais têm ΔG negativo)."""
    base = {m: {"dG": -1.0} for m in DESAFIANTES if m not in mantidos}
    base.update(ajustes)
    return entrada(**base)


class TestCriterios(unittest.TestCase):
    def eleg(self, e):
        return sl.selecionar(e).elegiveis

    def test_parametros_da_configuracao(self):
        s = C.selecao
        self.assertEqual((s.delta, s.fracao_min_repeticoes_positivas, s.delta_minimo_faixa, s.margem_desempate),
                         (0.02, 0.8, 0.0, 0.005))
        self.assertEqual(sl.parametros()["ordem_simplicidade"], ["lr_tuning", "svm", "rf", "xgb"])

    def test_sem_elegivel_fica_a_lr_com_ajuste(self):
        r = sl.selecionar(entrada(rf={"dG": 0.001}, xgb={"dG": 0.0}, svm={"dG": -0.01}))
        self.assertEqual((r.algoritmo, r.padrao, r.elegiveis, r.desafiante_escolhido),
                         ("lr_tuning", "lr_tuning", (), False))
        self.assertIn("Nenhum desafiante", " ".join(r.justificativa))

    def test_c1_fronteira_em_delta(self):
        casos = ((0.02, True), (0.0200001, True), (0.0199999, False), (0.019999999999, False), (0.0, False))
        for dG, ok in casos:
            with self.subTest(dG=dG):
                self.assertEqual(self.eleg(sem_os_outros("rf", rf={"dG": dG})), ("rf",) if ok else ())

    def test_c2_fronteira_em_80_por_cento_das_repeticoes_validas(self):
        casos = ((8, 10, True), (7, 10, False), (10, 10, True), (7, 9, False), (8, 9, True), (7, 8, True),
                 (6, 8, False))
        for positivas, validas, ok in casos:
            with self.subTest(positivas=positivas, validas=validas):
                e = sem_os_outros("rf", rf={"G": contraste(0.05, positivas, validas)})
                self.assertEqual(self.eleg(e), ("rf",) if ok else ())

    def test_c3_fronteira_e_estimabilidade_da_faixa(self):
        for dF, ok in ((0.0, True), (1e-12, True), (-1e-12, False), (-0.05, False)):
            with self.subTest(dF=dF):
                self.assertEqual(self.eleg(sem_os_outros("rf", rf={"dF": dF})), ("rf",) if ok else ())
        nao_estimavel = montar({m: {"avisos_convergencia": 0, "contraste_G": contraste(0.05),
                                    "contraste_F": contraste(0.0, estimavel=False)} for m in DESAFIANTES},
                               f_estimavel=False)
        r = sl.selecionar(nao_estimavel)
        self.assertEqual((r.algoritmo, r.elegiveis), ("lr_tuning", ()))
        self.assertTrue(all(not next(c for c in a.criterios if c.codigo == "C3").cumprido for a in r.avaliacoes))

    def test_c4_falha_tecnica_aviso_de_convergencia_e_g_estimavel(self):
        def todos(**kw):
            return montar({m: {"avisos_convergencia": 0, "contraste_G": contraste(0.03),
                               "contraste_F": contraste(0.01)} for m in DESAFIANTES}, **kw)

        self.assertEqual(len(self.eleg(todos())), 3)
        self.assertEqual(self.eleg(todos(ft=1)), ())
        self.assertEqual(self.eleg(todos(concluidas=249)), ())
        self.assertEqual(self.eleg(sem_os_outros("rf", rf={"avisos": 1})), ())
        self.assertEqual(set(self.eleg(entrada(rf={"avisos": 1}))), {"svm", "xgb"})  # só desqualifica o dono do aviso
        g_ne = montar({m: {"avisos_convergencia": 0, "contraste_G": contraste(0.0, estimavel=False),
                           "contraste_F": contraste(0.01)} for m in DESAFIANTES}, g_estimavel=False)
        self.assertEqual(self.eleg(g_ne), ())

    def test_um_criterio_a_menos_basta_para_nao_ser_elegivel(self):
        r = sl.selecionar(entrada(rf={"dF": -0.001}, xgb={"dG": 0.0199}, svm={"rep": (7, 10)}))
        faltou = {a.modelo: [c.codigo for c in a.criterios if not c.cumprido] for a in r.avaliacoes}
        self.assertEqual(faltou, {"rf": ["C3"], "svm": ["C2"], "xgb": ["C1"]})
        self.assertEqual(r.elegiveis, ())

    def test_ic_e_taxa_de_descarte_nao_sao_criterio(self):
        e = entrada()
        e["desafiantes"]["rf"]["bootstrap_iq"] = True
        with self.assertRaises(sl.EntradaInvalida):  # a entrada nem aceita esse campo
            sl.selecionar(e)
        texto = " ".join(c.descricao.lower() for a in sl.selecionar(entrada()).avaliacoes for c in a.criterios)
        for ausente in ("intervalo", "descarte", "bootstrap", "iq "):
            self.assertNotIn(ausente, texto)


class TestDesempate(unittest.TestCase):
    def test_um_so_elegivel(self):
        r = sl.selecionar(sem_os_outros("rf", rf={"dG": 0.03}))
        self.assertEqual((r.algoritmo, r.desafiante_escolhido, r.desempate_aplicado), ("rf", True, False))

    def test_maior_delta_vence_quando_a_diferenca_nao_e_menor_que_a_margem(self):
        casos = ((0.0051, "rf", False), (0.0049, "svm", True), (0.0, "svm", True), (0.02, "rf", False))
        for diferenca, esperado, desempate in casos:
            with self.subTest(diferenca=diferenca):
                r = sl.selecionar(sem_os_outros("rf", "svm", rf={"dG": 0.04 + diferenca}, svm={"dG": 0.04}))
                self.assertEqual((r.algoritmo, r.desempate_aplicado), (esperado, desempate))

    def test_diferenca_exatamente_igual_a_margem_nao_dispara_o_desempate(self):
        # Valores com diferença exata em binário: 0,5 − 0,25 = 0,25 (≥ 0,005) e 0,0625 − 0,0625 = 0 (< 0,005).
        r = sl.selecionar(sem_os_outros("rf", "svm", rf={"dG": 0.5}, svm={"dG": 0.25}))
        self.assertEqual((r.algoritmo, r.desempate_aplicado, r.diferenca_entre_os_dois_primeiros), ("rf", False, 0.25))
        # Com a configuração real a fronteira é "menor que": uma diferença igual à margem, em ponto flutuante exato,
        # fica do lado do maior Δ. Usa-se uma entrada em que a diferença calculada é exatamente a margem.
        d2 = 0.125
        d1 = d2 + C.selecao.margem_desempate
        diferenca = d1 - d2
        r = sl.selecionar(sem_os_outros("rf", "svm", rf={"dG": d1}, svm={"dG": d2}))
        self.assertEqual(r.desempate_aplicado, diferenca < C.selecao.margem_desempate)

    def test_ordem_de_simplicidade_lr_svm_rf_xgb(self):
        pares = (("svm", "rf", "svm"), ("rf", "xgb", "rf"), ("svm", "xgb", "svm"))
        for a, b, esperado in pares:
            with self.subTest(par=(a, b)):
                e = sem_os_outros(a, b, **{a: {"dG": 0.041}, b: {"dG": 0.042}})
                self.assertEqual(sl.selecionar(e).algoritmo, esperado)

    def test_empate_exato_vence_o_mais_simples(self):
        self.assertEqual(sl.selecionar(sem_os_outros("rf", "xgb", rf={"dG": 0.05}, xgb={"dG": 0.05})).algoritmo, "rf")
        self.assertEqual(sl.selecionar(entrada(rf={"dG": 0.05}, xgb={"dG": 0.05}, svm={"dG": 0.05})).algoritmo, "svm")

    def test_so_os_dois_primeiros_entram_no_desempate(self):
        # xgb 0,060 e svm 0,050 diferem em 0,010 (≥ margem): vence xgb, ainda que rf (0,0499) seja mais simples.
        r = sl.selecionar(entrada(xgb={"dG": 0.060}, svm={"dG": 0.050}, rf={"dG": 0.0499}))
        self.assertEqual((r.elegiveis, r.algoritmo, r.desempate_aplicado), (("xgb", "svm", "rf"), "xgb", False))

    def test_diferenca_registrada_e_justificativa(self):
        r = sl.selecionar(sem_os_outros("rf", "svm", rf={"dG": 0.045}, svm={"dG": 0.043}))
        self.assertAlmostEqual(r.diferenca_entre_os_dois_primeiros, 0.002, places=12)
        self.assertTrue(any("vence o mais simples" in j for j in r.justificativa))


class TestEntradaEJson(unittest.TestCase):
    def test_parametros_diferentes_da_configuracao_sao_recusados(self):
        casos = (("delta", 0.01), ("fracao_min_repeticoes_positivas", 0.5), ("margem_desempate", 0.01),
                 ("delta_minimo_faixa", -0.1), ("padrao", "rf"), ("ordem_simplicidade", ["rf", "lr_tuning"]))
        for chave, valor in casos:
            e = entrada()
            e["parametros"][chave] = valor
            with self.subTest(chave=chave), self.assertRaisesRegex(sl.EntradaInvalida, "parâmetros"):
                sl.selecionar(e)

    def test_entradas_malformadas(self):
        def com(editar):
            e = entrada()
            editar(e)
            return e

        casos = {
            "versao": com(lambda e: e.update(versao_esquema=2)),
            "desafiante_ausente": com(lambda e: e["desafiantes"].pop("xgb")),
            "delta_texto": com(lambda e: e["desafiantes"]["rf"]["contrastes"]["G"].update(delta="0,03")),
            "nan": com(lambda e: e["desafiantes"]["rf"]["contrastes"]["G"].update(delta=float("nan"))),
            "contagem": com(lambda e: e["desafiantes"]["rf"]["contrastes"]["F"].update(n_repeticoes_positivas=11)),
            "incoerente": com(lambda e: e["desfechos_estimaveis"].update(F=False)),
            "unidades": com(lambda e: e["unidades"].update(concluidas=-1)),
        }
        for nome, e in casos.items():
            with self.subTest(caso=nome), self.assertRaises(sl.EntradaInvalida):
                sl.selecionar(e)
        with self.assertRaises(sl.EntradaInvalida):
            sl.selecionar([1, 2])  # type: ignore[arg-type]
        self.assertEqual(sl.EntradaInvalida.chave_saida, "integridade")

    def test_selecionar_nao_altera_a_entrada(self):
        e = entrada(rf={"dG": 0.05})
        antes = copy.deepcopy(e)
        sl.selecionar(e)
        self.assertEqual(e, antes)

    def test_resultado_serializa_em_json_e_traz_c1_a_c4_explicitos(self):
        r = sl.selecionar(sem_os_outros("rf", rf={"dG": 0.05}))
        d = json.loads(json.dumps(r.como_dict()))
        self.assertEqual(d["algoritmo_recomendado"], "rf")
        for m in DESAFIANTES:
            self.assertEqual(set(d["criterios"][m]), set(sl.CRITERIOS))
            self.assertEqual(d["elegibilidade"][m], all(c["cumprido"] for c in d["criterios"][m].values()))
        self.assertIn("otimista por seleção", d["otimismo_por_selecao"])

    def test_texto_da_selecao_respeita_o_vocabulario_controlado(self):
        r = sl.selecionar(entrada(rf={"dG": 0.05}, svm={"dG": 0.049}))
        texto = json.dumps(r.como_dict(), ensure_ascii=False).lower()
        for vetada in C.saidas.vocabulario_vetado:
            self.assertNotIn(vetada.lower(), texto)


class TestRecalculoAPartirDoRelatorio(unittest.TestCase):
    def test_entrada_da_consolidacao_reproduz_a_decisao_depois_de_ida_e_volta_por_json(self):
        carga = sn.carga_definitiva()
        ctx = un.ContextoExecucao.de_dados(carga)
        base = mt.BaseAvaliacao.de_contexto(ctx)
        lista = sn.resultados_sinteticos(ctx, sn.pontuador_padrao({"svm": 3.2}))  # SVM bem melhor que a LR com ajuste
        c = co.consolidar(co.parear(base, lista))
        entrada_ = sl.montar_entrada(c, falhas_tecnicas_pendentes=0, unidades_esperadas=250, unidades_concluidas=250)
        direto = sl.selecionar(entrada_)
        refeito = sl.selecionar(json.loads(json.dumps(entrada_)))
        self.assertEqual(direto.como_dict(), refeito.como_dict())
        g = c.contraste("svm", "lr_tuning", "G")
        self.assertEqual(entrada_["desafiantes"]["svm"]["contrastes"]["G"]["delta"], g.delta)
        self.assertGreater(g.delta, 0.0)
        self.assertEqual(set(entrada_["desafiantes"]), {"rf", "xgb", "svm"})
        self.assertEqual(direto.padrao, "lr_tuning")
        # a decisão acompanha o que a regra manda para os números calculados
        def delta(m, d):
            return c.contraste(m, "lr_tuning", d)

        esperado_eleg = tuple(sorted(
            (m for m in DESAFIANTES
             if delta(m, "G").delta >= 0.02 and delta(m, "G").fracao_repeticoes_positivas >= 0.8
             and delta(m, "F").delta >= 0.0),
            key=lambda m: -delta(m, "G").delta))
        self.assertEqual(direto.elegiveis, esperado_eleg)

    def test_selecao_independe_de_tempo_e_do_bootstrap(self):
        arvore = ast.parse(MODULO.read_text(encoding="utf-8"))
        modulos = {a.name for n in ast.walk(arvore) if isinstance(n, ast.Import) for a in n.names} | \
                  {n.module or "" for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom)}
        importados = {a.name for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom) for a in n.names}
        for proibido in ("time", "datetime", "numpy"):
            self.assertNotIn(proibido, modulos)
        self.assertNotIn("bootstrap", importados)


if __name__ == "__main__":
    unittest.main()
