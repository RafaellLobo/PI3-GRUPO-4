"""Testes de ``relatorios.py`` e ``figuras.py`` (P9, Seções 11, 12 e 14.5). Só dados sintéticos; nada é treinado."""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sinteticos as sn  # noqa: E402
from src.benchmark_s6 import bootstrap as bs  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import consolidacao as co  # noqa: E402
from src.benchmark_s6 import figuras as fg  # noqa: E402
from src.benchmark_s6 import manifesto as mf  # noqa: E402
from src.benchmark_s6 import metricas as mt  # noqa: E402
from src.benchmark_s6 import privacidade as pv  # noqa: E402
from src.benchmark_s6 import relatorios as rl  # noqa: E402
from src.benchmark_s6 import selecao as sl  # noqa: E402
from src.benchmark_s6 import unidades as un  # noqa: E402

C = cfg.CONFIGURACAO


class BaseRelatorios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.carga = sn.carga_definitiva()
        cls.ctx = un.ContextoExecucao.de_dados(cls.carga)
        cls.base = mt.BaseAvaliacao.de_contexto(cls.ctx)
        cls.p = co.parear(cls.base, sn.resultados_sinteticos(cls.ctx))
        cls.c = co.consolidar(cls.p)
        cls.b = bs.bootstrap(cls.p, cls.c, n_replicas=1000)
        cls.entrada = sl.montar_entrada(cls.c, 0, 250, 250)
        cls.selecao = sl.selecionar(cls.entrada)
        cls.plano = cls.ctx.plano.resumo(cls.ctx.amostra)
        cls.info_base = {"pacientes": cls.base.n_pacientes, "nodulos": cls.base.amostra.n}
        cls.figs = fg.gerar_todas(cls.c, cls.b, cls.info_base)
        cls.tabs = rl.tabelas(cls.c, cls.b, cls.selecao, cls.plano)

    def dados(self, hashes=None) -> rl.DadosRelatorio:
        y = self.base.amostra.y
        faixa = self.base.contagem_faixa()
        return rl.DadosRelatorio(
            rl.Populacao(self.base.n_pacientes, self.base.amostra.n, int((y == 0).sum()), int((y == 1).sum()), faixa),
            rl.Identificacao("definitivo", False, cfg.sha256_configuracao(), "a" * 40, "main", {"python": "3.9.13"},
                             {"tabela": "1" * 64}, 250),
            self.c, self.b, self.entrada, self.selecao, self.plano,
            hashes if hashes is not None else {k: "0" * 64 for k in self.tabs}, sorted(self.figs))


def ler_csv(bruto: bytes):
    return list(csv.reader(io.StringIO(bruto.decode("utf-8"))))


class TestTabelas(BaseRelatorios):
    def test_nomes_cobrem_os_artefatos_versionados_da_configuracao(self):
        esperados = set(C.saidas.artefatos_versionados) - {"relatorio_benchmark.json", "resumo_benchmark.md",
                                                            "relatorio_execucao.json", "manifesto_benchmark.json"}
        self.assertEqual(set(self.tabs), esperados)

    def test_formas_e_contagens(self):
        n = lambda nome: len(ler_csv(self.tabs[nome])) - 1  # noqa: E731
        self.assertEqual(n("denominadores.csv"), 50 * 2)
        self.assertEqual(n("auc_por_fold.csv"), 5 * 50 * 2)
        self.assertEqual(n("auc_por_repeticao.csv"), 5 * 10 * 2)
        self.assertEqual(n("contrastes.csv"), 7 * 2)  # 3 da família + 4 descritivos, em G e F
        self.assertEqual(n("bootstrap_delta.csv"), self.b.n_replicas)
        self.assertEqual(n("hiperparametros_escolhidos.csv"), 250)
        self.assertEqual(len(ler_csv(self.tabs["bootstrap_delta.csv"])[0]), 1 + 14)

    def test_estados_dd_e_ri_aparecem_nos_csv(self):
        self.assertEqual(ler_csv(self.tabs["denominadores.csv"])[0][-1], "estado")
        self.assertEqual(ler_csv(self.tabs["auc_por_repeticao.csv"])[0][-1], "estado")

    def test_contrastes_trazem_os_dois_intervalos_so_na_familia(self):
        linhas = ler_csv(self.tabs["contrastes.csv"])
        cab = linhas[0]
        for l in linhas[1:]:
            r = dict(zip(cab, l))
            if r["grupo"] == "familia":
                self.assertTrue(r["bonferroni_baixo"] and r["ic95_baixo"])
            else:
                self.assertEqual(r["bonferroni_baixo"], "")
                self.assertTrue(r["ic95_baixo"])
        self.assertEqual([c for c in cab if c.startswith("delta_r")], [f"delta_r{r:02d}" for r in range(10)])

    def test_metricas_descritivas_sem_probabilidade_no_svm(self):
        linhas = ler_csv(self.tabs["metricas_descritivas.csv"])
        cab = linhas[0]
        svm = {dict(zip(cab, l))["metrica"] for l in linhas[1:] if l[1] == "svm"}
        self.assertEqual(svm, set(C.metricas.descritivas_regra_nativa))

    def test_configuracao_copiada_e_identica_a_canonica(self):
        self.assertEqual(self.tabs["config_benchmark.json"], cfg.bytes_arquivo())
        self.assertEqual(s4_sha(self.tabs["config_benchmark.json"]), cfg.sha256_configuracao())

    def test_plano_de_dobras_so_com_fingerprints_e_contagens(self):
        plano = json.loads(self.tabs["plano_folds.json"])
        self.assertEqual(plano["fingerprint_plano"], self.ctx.plano.fingerprint)
        self.assertEqual(len(plano["dobras"]), 50)

    def test_bytes_deterministicos(self):
        again = rl.tabelas(self.c, self.b, self.selecao, self.plano)
        self.assertEqual(again, self.tabs)
        self.assertTrue(all(b"\r" not in v for k, v in self.tabs.items() if k.endswith((".csv", ".json"))))

    def test_celulas_nao_finitas_sao_recusadas(self):
        with self.assertRaises(ValueError):
            rl.csv_bytes(["a"], [[float("nan")]])
        with self.assertRaises(ValueError):
            rl.csv_bytes(["a", "b"], [[1]])


def s4_sha(bruto: bytes) -> str:
    from src.benchmark_s6 import adaptadores_s4 as s4
    return s4.sha256_canonico(json.loads(bruto))


class TestRelatorioBenchmark(BaseRelatorios):
    def test_esquema_e_recalculo_da_selecao_a_partir_do_json(self):
        r = json.loads(json.dumps(rl.relatorio_benchmark(self.dados())))
        self.assertEqual(r["versao_esquema"], rl.VERSAO_ESQUEMA)
        refeita = sl.selecionar(r["selecao"]["entrada"])
        self.assertEqual(refeita.como_dict(), r["selecao"]["resultado"])
        self.assertEqual(r["identificacao"]["sha256_configuracao"], cfg.sha256_configuracao())
        self.assertEqual({e["modelo"] for e in r["estimativas"]}, set(sn.CODIGOS))
        self.assertEqual(r["desvios"], "nenhum")
        self.assertEqual(r["adendos"], [])
        for chave in ("o_que_o_intervalo_nao_captura", "bonferroni", "otimismo_por_selecao", "teste_e_s4"):
            self.assertIn(chave, r["limites"])

    def test_vencedor_e_rotulado_otimista_e_so_ele(self):
        r = rl.relatorio_benchmark(self.dados())
        rotulados = {e["modelo"] for e in r["estimativas"] if e["rotulo"]}
        self.assertEqual(rotulados, {self.selecao.algoritmo})

    def test_estados_dd_ri_ne_rd_iq(self):
        r = rl.relatorio_benchmark(self.dados())
        for d in ("G", "F"):
            self.assertEqual(set(r["estados"][d]), {"DD", "RI", "NE", "dobras", "repeticoes", "repeticoes_validas",
                                                    "RD", "fracao_RD", "IQ"})

    def test_json_passa_na_varredura(self):
        pv.varrer_json(rl.relatorio_benchmark(self.dados()), "relatorio_benchmark.json",
                       pv.identificadores_da_amostra(self.base.amostra))


class TestResumo(BaseRelatorios):
    def test_quinze_secoes_da_p9(self):
        md = rl.resumo_md(self.dados()).decode("utf-8")
        titulos = re.findall(r"^## (\d+)\. (.+)$", md, flags=re.M)
        self.assertEqual([int(n) for n, _ in titulos], list(range(1, 16)))
        for trecho in ("Identificação", "Desvios", "População", "AUC por modelo", "Contrastes", "Sensibilidade",
                       "LR com ajuste contra LR fixa", "Hiperparâmetros", "Métricas descritivas", "Regra de seleção",
                       "Otimismo", "não captura", "Falhas técnicas", "Teste e independência", "Figuras"):
            self.assertIn(trecho, md)
        self.assertIn("otimista por seleção", md)
        self.assertIn("Algoritmo recomendado", md)

    def test_passa_na_checagem_de_vocabulario_e_privacidade(self):
        md = rl.resumo_md(self.dados()).decode("utf-8")
        pv.varrer_texto(md, "resumo_benchmark.md", pv.identificadores_da_amostra(self.base.amostra))
        for vetada in C.saidas.vocabulario_vetado:
            self.assertNotIn(vetada.lower(), md.lower())

    def test_deterministico_e_lf(self):
        a, b = rl.resumo_md(self.dados()), rl.resumo_md(self.dados())
        self.assertEqual(a, b)
        self.assertNotIn(b"\r", a)

    def test_lista_hashes_e_commit_e_nao_afirma_validacao_clinica(self):
        md = rl.resumo_md(self.dados({"a.csv": "f" * 64})).decode("utf-8")
        self.assertIn("f" * 64, md)
        self.assertIn("a" * 40, md)
        self.assertIn("não afirma validação clínica", md)

    def test_svm_sem_metricas_de_probabilidade_no_resumo(self):
        linhas = [l for l in rl.resumo_md(self.dados()).decode("utf-8").splitlines() if l.startswith("| SVM |")]
        usadas = " ".join(linhas)
        for proibida in ("brier", "log_loss", "calibracao", "p_0_5"):
            self.assertNotIn(proibida, usadas)
        self.assertIn("regra nativa, não comparável", usadas)


class TestRelatorioExecucao(unittest.TestCase):
    def test_tempos_falhas_e_sem_mensagens_nem_caminhos(self):
        r = rl.relatorio_execucao(
            modo="implementacao", processos=4, duracoes={"r00-k0-rf": 10.0, "r00-k1-rf": 14.0, "r00-k0-lr_fixa": 0.5},
            reaproveitadas=["r00-k2-rf"], executadas=["r00-k0-rf"],
            falhas=[{"unidade": "r00-k3-rf", "chave_saida": "falha_tecnica", "categoria": "FalhaAjuste"}],
            avisos={"rf": 0}, duracao_total_s=30.0)
        self.assertEqual(r["duracao_por_modelo"]["rf"],
                         {"unidades": 2, "soma_s": 24.0, "media_s": 12.0, "maxima_s": 14.0})
        self.assertEqual(r["unidades"], {"executadas": 1, "reaproveitadas": 1, "retomada": True, "falhas": 1})
        pv.varrer_json(r, "relatorio_execucao.json")
        self.assertNotIn("mensagem", json.dumps(r))


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #


def dpi_do_png(bruto: bytes) -> float:
    assert bruto[:8] == b"\x89PNG\r\n\x1a\n"
    i = 8
    while i < len(bruto):
        n, tipo = struct.unpack(">I4s", bruto[i:i + 8])
        if tipo == b"pHYs":
            ppm = struct.unpack(">II", bruto[i + 8:i + 16])[0]
            return ppm * 0.0254
        i += 12 + n
    raise AssertionError("PNG sem pHYs")


class TestFiguras(BaseRelatorios):
    def test_f1_a_f8_em_png_svg_e_csv(self):
        self.assertEqual(tuple(sorted(self.figs)), fg.nomes_esperados())
        self.assertEqual(len(self.figs), 24)
        for i in range(1, 9):
            self.assertTrue(self.figs[f"figuras/F{i}.png"].startswith(b"\x89PNG"))
            self.assertIn(b"<svg", self.figs[f"figuras/F{i}.svg"][:400])
            self.assertGreater(len(ler_csv(self.figs[f"figuras/dados/F{i}.csv"])), 1)
        self.assertEqual([f.codigo for f in C.saidas.figuras], [f"F{i}" for i in range(1, 9)])

    def test_png_a_300_dpi(self):
        for i in range(1, 9):
            self.assertAlmostEqual(dpi_do_png(self.figs[f"figuras/F{i}.png"]), 300, delta=0.1)

    def test_bytes_deterministicos_inclusive_svg(self):
        again = fg.gerar_todas(self.c, self.b, self.info_base)
        self.assertEqual(again, self.figs)

    def test_svg_sem_data_nem_caminho_local(self):
        for i in range(1, 9):
            svg = self.figs[f"figuras/F{i}.svg"].decode("utf-8")
            self.assertNotIn("<dc:date>", svg)
            pv.varrer_texto(svg, f"F{i}.svg", pv.identificadores_da_amostra(self.base.amostra))

    def test_paleta_okabe_ito_e_cor_fixa_por_modelo(self):
        okabe = {"#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"}
        self.assertTrue(set(fg.COR.values()) <= okabe)
        self.assertEqual(len(set(fg.COR.values())), 5)
        self.assertEqual(set(fg.COR), set(sn.CODIGOS))

    def test_csv_das_figuras_so_agregados(self):
        ids = pv.identificadores_da_amostra(self.base.amostra)
        for nome, bruto in self.figs.items():
            if nome.endswith(".csv"):
                pv.varrer_csv(bruto.decode("utf-8"), nome, ids)

    def test_f1_nao_usa_intervalo_de_bonferroni(self):
        cab = ler_csv(self.figs["figuras/dados/F1.csv"])[0]
        self.assertFalse([c for c in cab if "bonferroni" in c])
        self.assertIn("ic95_descritivo_baixo", cab)
        linhas = ler_csv(self.figs["figuras/dados/F1.csv"])[1:]
        self.assertEqual(len(linhas), 10)  # 5 modelos × G e F
        tabela = {(l[0], l[1]): l for l in linhas}
        ic = self.b.intervalo_theta("rf", "G")
        self.assertEqual(float(tabela[("rf", "G")][3]), ic.limite_baixo)

    def test_f2_traz_os_dois_intervalos_e_a_amplitude_entre_repeticoes(self):
        linhas = ler_csv(self.figs["figuras/dados/F2.csv"])
        cab = linhas[0]
        for esperado in ("bonferroni_baixo", "ic95_baixo", "minimo_repeticoes", "maximo_repeticoes"):
            self.assertIn(esperado, cab)
        self.assertEqual(len(linhas) - 1, 6)

    def test_f3_ate_50_pontos_por_contraste_e_desfecho(self):
        linhas = ler_csv(self.figs["figuras/dados/F3.csv"])[1:]
        por = {}
        for l in linhas:
            por[(l[0], l[2])] = por.get((l[0], l[2]), 0) + 1
        self.assertEqual(set(por.values()), {50})
        self.assertEqual(len(por), 6)

    def test_f4_destaca_bordas_e_cobre_todas_as_configuracoes(self):
        linhas = ler_csv(self.figs["figuras/dados/F4.csv"])[1:]
        self.assertEqual(len([l for l in linhas if l[0] == "svm"]), 30)
        self.assertEqual(len([l for l in linhas if l[0] == "xgb"]), 32)
        self.assertEqual({l[0] for l in linhas}, {"lr_tuning", "svm", "rf", "xgb"})  # a LR fixa não tem grade
        self.assertEqual(sum(int(l[3]) for l in linhas if l[0] == "rf"), 50)
        self.assertTrue(any(l[4] == "true" for l in linhas) and any(l[4] == "false" for l in linhas))

    def test_f5_rotula_a_faixa_como_dispersao_entre_dobras(self):
        textos = []

        def capturar(fig, nome, tabela):
            if nome == "F5":
                textos.extend(t.get_text() for t in fig.axes[0].get_legend().get_texts())
            fg.plt.close(fig)
            return {}

        with fg.matplotlib.rc_context(fg._estilo()), mock.patch.object(fg, "_bytes", side_effect=capturar):
            fg.f5(self.c, self.info_base)
        self.assertTrue(any("dispersão entre dobras — não é intervalo de confiança" in t for t in textos), textos)
        self.assertEqual(fg.ROTULO_FAIXA_ROC, "dispersão entre dobras — não é intervalo de confiança")
        curvas = ler_csv(self.figs["figuras/dados/F5.csv"])[1:]
        self.assertEqual(len(curvas), 5 * 101)

    def test_f6_sem_svm(self):
        linhas = ler_csv(self.figs["figuras/dados/F6.csv"])[1:]
        self.assertEqual({l[0] for l in linhas}, {"lr_fixa", "lr_tuning", "rf", "xgb"})

    def test_f7_denominadores_do_desfecho_f(self):
        linhas = ler_csv(self.figs["figuras/dados/F7.csv"])[1:]
        self.assertEqual(len(linhas), 50)
        self.assertEqual(sum(int(l[2]) for l in linhas), sum(d.nodulos for d in self.c.denominadores
                                                             if d.desfecho == "F"))

    def test_f8_traz_histograma_e_taxa_de_descarte_por_contraste(self):
        linhas = ler_csv(self.figs["figuras/dados/F8.csv"])[1:]
        self.assertEqual(len([l for l in linhas if l[3] == "descarte"]), 6)
        for a in ("rf", "xgb", "svm"):
            for d in ("G", "F"):
                bins = [l for l in linhas if (l[0], l[2], l[3]) == (a, d, "bin")]
                self.assertEqual(sum(int(l[6]) for l in bins), self.b.n_replicas)

    def test_ic_nao_relatado_e_explicito_na_figura(self):
        # Desfecho F não estimável: F1 e F2 geram sem quebrar e sem IC.
        import dataclasses
        base = dataclasses.replace(self.base, na_faixa=self.base.na_faixa & False)
        p = co.parear(base, list(self.p.resultados.values()))
        c = co.consolidar(p)
        b = bs.bootstrap(p, c, n_replicas=50)
        saidas = fg.gerar_todas(c, b, self.info_base)
        self.assertEqual(tuple(sorted(saidas)), fg.nomes_esperados())
        linhas = ler_csv(saidas["figuras/dados/F1.csv"])[1:]
        self.assertTrue(all(l[5] == "false" and l[6] == "NE" for l in linhas if l[1] == "F"))


class TestConjunto(BaseRelatorios):
    def test_todos_os_artefatos_versionados_gravados_passam_na_varredura(self):
        pasta = Path(tempfile.mkdtemp(prefix="rel-s6-"))
        self.addCleanup(shutil.rmtree, pasta, ignore_errors=True)
        arquivos = {**self.tabs, **self.figs}
        for nome, bruto in arquivos.items():
            mf.gravar_atomico(pasta / nome, bruto)
        hashes = {k: v["sha256_lf"] for k, v in mf.hashes_de_saidas(pasta, arquivos).items()}
        d = self.dados(hashes)
        mf.gravar_atomico(pasta / "relatorio_benchmark.json", mf.bytes_json(rl.relatorio_benchmark(d)))
        mf.gravar_atomico(pasta / "resumo_benchmark.md", rl.resumo_md(d))
        mf.gravar_atomico(pasta / "relatorio_execucao.json", mf.bytes_json(rl.relatorio_execucao(
            modo="definitivo", processos=4, duracoes={"r00-k0-rf": 1.0}, reaproveitadas=[], executadas=["r00-k0-rf"],
            falhas=[], avisos={}, duracao_total_s=1.0)))
        varridos = pv.varrer_diretorio(pasta, pv.identificadores_da_amostra(self.base.amostra))
        self.assertEqual(set(C.saidas.artefatos_versionados) - {"manifesto_benchmark.json"},
                         {v for v in varridos if "/" not in v})
        self.assertEqual(len(varridos), 14 + 24)


if __name__ == "__main__":
    unittest.main()
