"""Testes de ``src/benchmark_s6/privacidade.py`` (P9, Seções 8.3, 12 e 14.5). Só dados sintéticos."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sinteticos as sn  # noqa: E402
from src.benchmark_s6 import adaptadores_s4 as s4  # noqa: E402
from src.benchmark_s6 import config as cfg  # noqa: E402
from src.benchmark_s6 import privacidade as pv  # noqa: E402

C = cfg.CONFIGURACAO


class TestTexto(unittest.TestCase):
    def test_texto_limpo_passa(self):
        pv.varrer_texto("Diferença estimada de 0,01; intervalo descritivo; compatível com ausência de diferença.", "x")

    def test_padrao_de_identificador_da_configuracao(self):
        for ruim in ("LIDC-IDRI-0001", "paciente LIDC-IDRI-9999 aqui", "x/LIDC-IDRI-0123/y"):
            with self.subTest(texto=ruim), self.assertRaisesRegex(pv.PrivacidadeViolada, "identificador"):
                pv.varrer_texto(ruim, "x")
        pv.varrer_texto("LIDC-IDRI-12", "x")  # fora do padrão de 4 dígitos
        self.assertEqual(C.saidas.padrao_varredura, r"LIDC-IDRI-\d{4}")

    def test_identificadores_da_amostra(self):
        carga = sn.carga_definitiva(n_pacientes=30)
        ids = pv.identificadores_da_amostra(carga.amostra())
        self.assertIn(carga.nodulos()[0], ids)
        self.assertIn(carga.grupos()[0], ids)
        for i in (carga.nodulos()[3], carga.grupos()[3]):
            with self.assertRaisesRegex(pv.PrivacidadeViolada, "individual"):
                pv.varrer_texto(f"dobra com {i}", "x", ids)
        pv.varrer_texto("sem identificador", "x", ids)

    def test_cada_termo_vetado_e_recusado_sem_diferenciar_caixa(self):
        for termo in C.saidas.vocabulario_vetado:
            for variante in (termo, termo.upper(), termo.capitalize(), f"um resultado {termo} aqui"):
                with self.subTest(termo=variante), self.assertRaisesRegex(pv.PrivacidadeViolada, "vocabulário"):
                    pv.varrer_texto(variante, "x")
        pv.varrer_texto("significativo", "x", vocabulario=False)  # a exceção é explícita e só por parâmetro

    def test_caminhos_locais(self):
        for prefixo in (str(Path.home()), str(s4.REPO_ROOT), tempfile.gettempdir()):
            for texto in (prefixo, prefixo.replace("\\", "/"), prefixo.upper()):
                with self.subTest(texto=texto), self.assertRaisesRegex(pv.PrivacidadeViolada, "caminho local"):
                    pv.varrer_texto(f"gravado em {texto}/saida", "x")
        pv.varrer_texto("reports/sprint6/benchmark/figuras/F1.png", "x")  # caminho relativo é permitido


class TestJsonECsv(unittest.TestCase):
    def test_chaves_de_dados_individuais(self):
        for chave in ("nodulo_ids", "pontuacoes", "posicoes", "patient_id", "y_score"):
            with self.subTest(chave=chave), self.assertRaisesRegex(pv.PrivacidadeViolada, "dado individual"):
                pv.varrer_json({"a": {chave: [1, 2]}}, "r.json")

    def test_valores_e_chaves_sao_varridos_em_profundidade(self):
        with self.assertRaises(pv.PrivacidadeViolada):
            pv.varrer_json({"a": [{"b": "LIDC-IDRI-0001"}]}, "r.json")
        with self.assertRaises(pv.PrivacidadeViolada):
            pv.varrer_json({"LIDC-IDRI-0001": 1}, "r.json")
        with self.assertRaises(pv.PrivacidadeViolada):
            pv.varrer_json({"texto": ["ok", "resultado demonstrado"]}, "r.json")
        pv.varrer_json({"a": [1, 2.5, None, True], "b": {"c": "agregado"}}, "r.json")

    def test_csv_cabecalho_e_conteudo(self):
        pv.varrer_csv("modelo,auc\nrf,0.97\n", "t.csv")
        with self.assertRaisesRegex(pv.PrivacidadeViolada, "coluna"):
            pv.varrer_csv("nodulo_id,auc\nx,0.9\n", "t.csv")
        with self.assertRaises(pv.PrivacidadeViolada):
            pv.varrer_csv("modelo,nota\nrf,LIDC-IDRI-0002\n", "t.csv")


class TestArquivos(unittest.TestCase):
    def setUp(self):
        self.pasta = Path(tempfile.mkdtemp(prefix="priv-s6-"))
        self.addCleanup(shutil.rmtree, self.pasta, ignore_errors=True)

    def escrever(self, nome: str, conteudo, binario: bool = False) -> Path:
        caminho = self.pasta / nome
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_bytes(conteudo if binario else conteudo.encode("utf-8"))
        return caminho

    def test_varre_todos_os_tipos_e_devolve_os_caminhos(self):
        self.escrever("a.json", json.dumps({"x": 1}))
        self.escrever("b.csv", "a,b\n1,2\n")
        self.escrever("c.md", "# Resumo\n\nDiferença estimada.\n")
        self.escrever("figuras/F1.svg", "<svg><text>AUC</text></svg>")
        self.escrever("figuras/F1.png", b"\x89PNG\r\n\x1a\n\x00\x00binario", binario=True)
        self.assertEqual(pv.varrer_diretorio(self.pasta), ("a.json", "b.csv", "c.md", "figuras/F1.png",
                                                           "figuras/F1.svg"))

    def test_recusa_em_cada_tipo(self):
        casos = {"a.json": '{"x": "LIDC-IDRI-0001"}', "b.csv": "a\nLIDC-IDRI-0001\n", "c.md": "demonstrado\n",
                 "d.svg": "<svg>LIDC-IDRI-0003</svg>"}
        for nome, conteudo in casos.items():
            with self.subTest(arquivo=nome):
                limpo = Path(tempfile.mkdtemp(prefix="priv-s6-"))
                self.addCleanup(shutil.rmtree, limpo, ignore_errors=True)
                (limpo / nome).write_text(conteudo, encoding="utf-8")
                with self.assertRaises(pv.PrivacidadeViolada):
                    pv.varrer_diretorio(limpo)
        self.escrever("e.png", b"\x89PNG\x00LIDC-IDRI-0009\x00", binario=True)  # busca no conteúdo do binário
        with self.assertRaises(pv.PrivacidadeViolada):
            pv.varrer_arquivo(self.pasta / "e.png")

    def test_ids_da_amostra_em_qualquer_arquivo(self):
        carga = sn.carga_definitiva(n_pacientes=20)
        ids = pv.identificadores_da_amostra(carga.amostra())
        self.escrever("t.csv", f"a\n{carga.grupos()[0]}\n")
        with self.assertRaises(pv.PrivacidadeViolada):
            pv.varrer_diretorio(self.pasta, ids)

    def test_unica_excecao_de_vocabulario_e_a_configuracao_canonica(self):
        config = (cfg.bytes_arquivo())
        self.escrever(pv.ARQUIVO_COM_VOCABULARIO_VETADO, config, binario=True)
        pv.varrer_diretorio(self.pasta)  # lista o vocabulário vetado por definição: passa
        self.assertIn("significativo", config.decode("utf-8"))
        self.escrever("outro.json", json.dumps({"vocabulario_vetado": ["significativo"]}))
        with self.assertRaises(pv.PrivacidadeViolada):
            pv.varrer_diretorio(self.pasta)  # o mesmo texto em qualquer outro arquivo é recusado

    def test_config_canonica_nao_tem_identificador_nem_caminho(self):
        self.escrever(pv.ARQUIVO_COM_VOCABULARIO_VETADO, cfg.bytes_arquivo(), binario=True)
        pv.varrer_arquivo(self.pasta / pv.ARQUIVO_COM_VOCABULARIO_VETADO)

    def test_json_invalido_e_diretorio_inexistente(self):
        self.escrever("x.json", "{")
        with self.assertRaisesRegex(pv.PrivacidadeViolada, "inválido"):
            pv.varrer_diretorio(self.pasta)
        with self.assertRaises(pv.PrivacidadeViolada):
            pv.varrer_diretorio(self.pasta / "nao_existe")
        self.assertEqual(pv.PrivacidadeViolada.chave_saida, "privacidade_ou_vocabulario")
        self.assertIn(pv.PrivacidadeViolada.chave_saida, C.execucao.codigos_saida)


if __name__ == "__main__":
    unittest.main()
