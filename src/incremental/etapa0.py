"""Etapa 0, item 2 — procedimento ajustado aplicado ao treino completo (Seção 7.1 do protocolo).

Aplica UMA vez a seleção da Seção 5.4 (``selecao.selecionar_treino_completo``) aos 417 nódulos do
treino oficial e registra a lista obtida **apenas de forma descritiva**, junto com a diferença em
relação à lista declarada da Seção 5.1. A lista não substitui a lista fixa, não é avaliada na
validação nem no teste e não autoriza M2, M3 ou M4.

* Atividade: ``etapa0_prerequisitos`` (``config.Etapa.E0``). Não exige decisões e lê só ``train``.
* Nenhum modelo de avaliação é ajustado e nenhuma métrica de desempenho é calculada.
* O relatório não traz identificadores nem predições; o manifesto é gravado ao lado.
* A comparação usa ``config.PAINEL_DECLARADO`` (Seção 5.1) como referência descritiva. Isso NÃO
  é a aprovação da decisão P3 e não libera nenhum modelo com painel.

Ponto de entrada: ``executar_etapa0_item2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from . import config as cfg
from . import dados
from . import guardas
from . import piloto
from . import rastreabilidade as rst
from . import selecao

ATIVIDADE = cfg.Etapa.E0.value
PARTICOES: Tuple[str, ...] = ("train",)
ROTULO = "descritivo_etapa0_item2"
FORMATO_RELATORIO = "relatorio-etapa0-item2-cartao4"
VERSAO = 1
ARQUIVO_RELATORIO = "relatorio_etapa0_item2.json"
ARQUIVO_MANIFESTO = "manifesto_etapa0_item2.json"
AVISO = ("Registro DESCRITIVO da Etapa 0, item 2 (protocolo, Seção 7.1): lista do procedimento ajustado "
         "aplicado uma vez ao treino oficial completo. Não substitui a lista fixa da Seção 5.1, não é "
         "avaliada na validação nem no teste e não calcula nenhuma métrica de desempenho.")


@dataclass(frozen=True)
class ResultadoEtapa0:
    relatorio: Mapping[str, Any]
    caminho_relatorio: Path
    caminho_manifesto: Path
    manifesto: Mapping[str, Any]


def seeds() -> Dict[str, int]:
    espec = cfg.ESPEC_SELECAO
    return {"estabilidade": espec.seed_estabilidade, "interna_base": espec.seed_interna_base,
            "logistica_selecao": espec.random_state}


def comparar_listas(ajustada: Tuple[str, ...], declarada: Tuple[str, ...]) -> Dict[str, Any]:
    """Diferença entre a lista ajustada e a declarada (conjuntos; ordem da lista declarada e do esquema)."""
    comum = tuple(f for f in declarada if f in set(ajustada))
    so_ajustada = tuple(f for f in ajustada if f not in set(declarada))
    so_declarada = tuple(f for f in declarada if f not in set(ajustada))
    uniao = len(set(ajustada) | set(declarada))
    return {
        "lista_declarada": list(declarada), "lista_ajustada": list(ajustada),
        "em_comum": list(comum), "so_na_ajustada": list(so_ajustada), "so_na_declarada": list(so_declarada),
        "n_em_comum": len(comum), "jaccard": (len(comum) / uniao) if uniao else None,
        "referencia": "config.PAINEL_DECLARADO (Seção 5.1); não é a aprovação de P3",
    }


def _relatorio(carga: dados.DadosCarregados, amostra: Any, sel: selecao.SelecaoFold,
               decisoes: Mapping[str, cfg.Decisao]) -> dict:
    avisos: Dict[Tuple[str, str], int] = {}
    for f in sel.folds:
        for categoria, mensagem, n in f.warnings:
            avisos[(categoria, mensagem)] = avisos.get((categoria, mensagem), 0) + n
    return {
        "formato": FORMATO_RELATORIO, "versao_formato": VERSAO, "rotulo": ROTULO, "aviso": AVISO,
        "atividade": ATIVIDADE, "hash_configuracao": cfg.hash_configuracao(decisoes),
        "sha256_dados": dict(carga.sha256_entradas), "particoes_lidas": list(carga.particoes),
        "seeds": seeds(), "amostra": piloto._resumo_amostra(amostra, amostra.y),
        "fora_desta_modalidade": {"modelos_ajustados": False, "metricas_de_desempenho": False,
                                  "particoes_nao_lidas": ["validation", "test"]},
        "selecao": sel.como_dict(),
        "comparacao_com_lista_declarada": comparar_listas(sel.s_k, cfg.PAINEL_DECLARADO),
        "convergencia": {"folds_nao_convergidos": [f.fold_estabilidade for f in sel.folds if not f.convergiu]},
        "warnings": [{"categoria": c, "mensagem": m, "contagem": n} for (c, m), n in sorted(avisos.items())],
    }


def executar_etapa0_item2(destino: Union[str, Path], *,
                          finalidade: rst.Finalidade = rst.Finalidade.DEFINITIVA,
                          decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
                          repo: Union[str, Path] = guardas.REPO_ROOT,
                          fonte: Optional[dados.FonteDados] = None) -> ResultadoEtapa0:
    """Executa o item 2 e grava relatório e manifesto em ``destino`` (diretório existente do repositório).

    Lê só ``train``. ``fonte`` existe para testes com dados sintéticos; o padrão é a fonte oficial.
    """
    repo, destino = Path(repo), Path(destino)
    caminho_relatorio, caminho_manifesto = piloto._exigir_destino(destino, repo, ARQUIVO_RELATORIO,
                                                                  ARQUIVO_MANIFESTO)
    autorizacao = piloto._autorizar(ATIVIDADE, (), decisoes, repo)
    fonte = fonte or dados.fonte_oficial()
    entradas = {"tabela": fonte.tabela.caminho, "split": fonte.split.caminho}
    execucao = rst.Execucao(ATIVIDADE, finalidade, autorizacao, entradas, seeds(),
                            {"item": "etapa0_item2", "particoes": list(PARTICOES), "rotulo": ROTULO},
                            decisoes=decisoes, repo=repo)
    relatorio = None
    try:
        with execucao:
            carga, frame, amostra = piloto._carregar_treino(ATIVIDADE, PARTICOES, autorizacao, decisoes, repo, fonte)
            sel = selecao.selecionar_treino_completo(amostra, carga.X(PARTICOES[0], carga.features), carga.features)
            relatorio = _relatorio(carga, amostra, sel, decisoes)
            piloto._gravar_json(relatorio, caminho_relatorio)
            execucao.registrar_saida("relatorio", caminho_relatorio)
    finally:
        if execucao.manifesto is not None:
            rst.escrever_manifesto(execucao.manifesto, caminho_manifesto, repo)
    return ResultadoEtapa0(MappingProxyType(relatorio), caminho_relatorio, caminho_manifesto,
                           MappingProxyType(execucao.manifesto))


__all__ = ["ATIVIDADE", "AVISO", "ResultadoEtapa0", "comparar_listas", "executar_etapa0_item2", "seeds"]
