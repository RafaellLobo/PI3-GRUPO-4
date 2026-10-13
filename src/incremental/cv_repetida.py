"""CV repetida exploratória 10 × 5 — Cartão 4, Sprint 4 (Fase C).

Executa a atividade ``config.ATIVIDADE_CV_REPETIDA_EXPLORATORIA`` — distinta
das Etapas 1–5 (em particular da Etapa 2 oficial) e do piloto, nunca
definitiva — sobre a partição ``train``, com a estrutura da Seção 7.2:

1. autorização pelas guardas (atividade e modelos; nenhuma decisão própria,
   M2–M4 exigiriam P3) e sessão de checkpoints da própria atividade;
2. ``Execucao`` EXPLORATORIA: carga só de ``train``, uma única ``Amostra``
   e um plano por repetição r = 0…9 (folds externos com seed 42 + r,
   agrupados por paciente; estabilidade e internos como no piloto);
3. em cada (r, k): a execução por fold do piloto (``piloto._executar_fold``)
   — S_k calculado UMA vez e reutilizado em M2v, M3v e M4v; M0, M1, M2v,
   M3v e M4v ajustados só no treino externo e aplicados só à avaliação
   externa. Nenhuma lógica de seleção ou ajuste é copiada aqui;
4. cada unidade (r, k, modelo) é um checkpoint; as 250 identidades
   esperadas são verificadas antes de qualquer ajuste, e a retomada confere
   a seleção recalculada com a gravada;
5. consolidação por repetição (``piloto._consolidar_repeticao``): 417
   predições OOF por modelo, AUC OOF e Δ (M3v − M0, M4v − M1) em CADA
   repetição. Entre repetições, só os dez valores, a média e o
   mínimo–máximo, como sensibilidade à partição — nunca IC, erro-padrão ou
   teste, porque os 50 folds e as 10 repetições não são independentes;
6. estabilidade da seleção: frequência de cada candidata nos 50 treinos
   externos e Jaccard descritivo entre cada S_k e o painel declarado da
   Seção 5.1 (só nomes: nenhum modelo de painel fixo é ajustado);
7. relatório agregado ``preliminar_exploratorio`` e manifesto exploratório;
   tempos num arquivo à parte, no diretório local de checkpoints.

Fora desta modalidade, sem parâmetro que os habilite: permutações, spline,
M2–M4 com lista fixa e as partições ``validation`` e ``test``. Sem paralelismo.

Ponto de entrada: ``executar_cv_repetida`` ou
``python -m src.incremental.cv_repetida --checkpoints DIR --destino DIR
--confirmo-preliminar-exploratorio``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from . import checkpoints as ck
from . import config as cfg
from . import dados
from . import folds as fl
from . import guardas
from . import piloto
from . import rastreabilidade as rst

ATIVIDADE = cfg.ATIVIDADE_CV_REPETIDA_EXPLORATORIA
PARTICOES: Tuple[str, ...] = ("train",)
#: r = 0…n_repeticoes − 1; a repetição r usa a seed externa 42 + r (Seção 7.2).
REPETICOES: Tuple[int, ...] = tuple(range(cfg.ESPEC_CV.n_repeticoes))
MODELOS: Tuple[str, ...] = piloto.MODELOS
CONTRASTES: Tuple[Tuple[str, str], ...] = piloto.CONTRASTES
ROTULO = piloto.ROTULO
FORMATO_RELATORIO = "relatorio-cv-repetida-exploratoria-cartao4"
FORMATO_PAYLOAD = "payload-cv-repetida-exploratoria-cartao4"
VERSAO = 1
ARQUIVO_RELATORIO = "relatorio_cv_repetida_exploratoria.json"
ARQUIVO_MANIFESTO = "manifesto_cv_repetida_exploratoria.json"
#: Referência do Jaccard descritivo: a lista DECLARADA da Seção 5.1, só como nomes.
PAINEL_JACCARD: Tuple[str, ...] = cfg.PAINEL_DECLARADO
UNIDADE_DE_RESUMO = "repeticao"
AVISO = ("Resultado preliminar exploratório da atividade cv_repetida_exploratoria: CV aninhada repetida "
         "fora das Etapas 1-5 do protocolo, sem aprovação do grupo ou do professor (P1-P5 pendentes), sem "
         "validação nem teste. A unidade de resumo é a repetição; média e mínimo-máximo entre repetições "
         "descrevem sensibilidade à partição, não incerteza amostral, porque os folds e as repetições "
         "reutilizam os mesmos nódulos. Sem p-valor, intervalo de confiança, erro-padrão ou teste de "
         "significância; AUC, contrastes e Jaccard são apenas descritivos e nenhum modelo de lista fixa é "
         "ajustado.")

if set(MODELOS) & set(cfg.MODELOS_COM_PAINEL):  # pragma: no cover — proteção de edição
    raise RuntimeError("a CV repetida exploratória não usa M2, M3 nem M4")


class CvRepetidaInvalida(piloto.PilotoInvalido):
    """Inconsistência própria da CV repetida (repetições, consolidação); nada é corrigido."""


@dataclass(frozen=True)
class ResultadoCvRepetida:
    relatorio: Mapping[str, Any]
    caminho_relatorio: Path
    caminho_manifesto: Path
    manifesto: Mapping[str, Any]
    caminho_tempos: Path


# --------------------------------------------------------------------------- #
# Configuração da execução
# --------------------------------------------------------------------------- #


def _validar_repeticoes(repeticoes: Any) -> Tuple[int, ...]:
    """Repetições inteiras, distintas, crescentes e dentro de 0…n_repeticoes − 1."""
    limite = cfg.ESPEC_CV.n_repeticoes
    if not isinstance(repeticoes, tuple) or not repeticoes:
        raise CvRepetidaInvalida("repetições devem ser uma tupla não vazia")
    if any(type(r) is not int or not 0 <= r < limite for r in repeticoes):
        raise CvRepetidaInvalida(f"repetições devem ser inteiros em [0, {limite})")
    if list(repeticoes) != sorted(set(repeticoes)):
        raise CvRepetidaInvalida("repetições devem ser distintas e crescentes")
    return repeticoes


def seeds_manifesto(repeticoes: Sequence[int]) -> Dict[str, int]:
    """Seeds do manifesto: a externa de cada repetição e as comuns a todas, todas de ``config``."""
    comuns = {k: v for k, v in piloto.seeds_repeticao(repeticoes[0]).items() if k != "cv_externa"}
    return {**{f"cv_externa_r{r:02d}": piloto.seeds_repeticao(r)["cv_externa"] for r in repeticoes}, **comuns}


def _parametros(repeticoes: Sequence[int]) -> Dict[str, Any]:
    return {"repeticoes": list(repeticoes), "particoes": list(PARTICOES), "modelos": list(MODELOS),
            "contrastes": [f"{a}-{b}" for a, b in CONTRASTES], "rotulo": ROTULO,
            "n_folds_externos": cfg.ESPEC_CV.n_folds_externos, "unidade_de_resumo": UNIDADE_DE_RESUMO,
            "painel_jaccard": list(PAINEL_JACCARD)}


# --------------------------------------------------------------------------- #
# Consolidação entre repetições (só descritiva)
# --------------------------------------------------------------------------- #


def _descrever(valores: Sequence[float]) -> dict:
    """Os valores na ordem das repetições, a média e o mínimo–máximo; nada inferencial."""
    valores = [float(v) for v in valores]
    return {"valores_por_repeticao": valores, "media": math.fsum(valores) / len(valores),
            "minimo": min(valores), "maximo": max(valores)}


def _jaccard(conjunto: Sequence[str], referencia: Sequence[str]) -> float:
    a, b = set(conjunto), set(referencia)
    return len(a & b) / len(a | b)


def _consolidar(carga: dados.DadosCarregados, amostra: fl.Amostra, decisoes: Mapping[str, cfg.Decisao],
                repeticoes: Tuple[int, ...], por_repeticao: List[Tuple[Any, list, dict]]) -> dict:
    if [ctx.repeticao for ctx, _, _ in por_repeticao] != list(repeticoes):
        raise CvRepetidaInvalida("consolidação com repetições fora de ordem ou ausentes")
    frequencia, tamanhos, jaccards = Counter(), [], []
    avisos_selecao, avisos_modelos = Counter(), Counter()
    refits_sem_convergencia = n_refits = ajustes_sem_convergencia = n_ajustes = 0
    relatorio_repeticoes = []
    for ctx, por_fold, repeticao in por_repeticao:
        folds = []
        for resumo, (payloads, selecao_gravada) in zip(repeticao["folds"], por_fold):
            jaccard = _jaccard(resumo["s_k"], PAINEL_JACCARD)
            folds.append({**resumo, "s_k_tamanho": len(resumo["s_k"]), "jaccard_painel_declarado": jaccard})
            frequencia.update(resumo["s_k"])
            tamanhos.append(len(resumo["s_k"]))
            jaccards.append(jaccard)
            estabilidade = selecao_gravada["registro"]["folds_estabilidade"]
            n_refits += len(estabilidade)
            refits_sem_convergencia += sum(not f["convergiu"] for f in estabilidade)
            for payload in payloads.values():
                ajuste = payload["parametros_ajustados"]
                if ajuste["tipo"] == "logistica":
                    n_ajustes += 1
                    ajustes_sem_convergencia += ajuste["n_iter"] >= cfg.ESPEC_AVALIACAO.max_iter
        for w in repeticao["warnings"]["selecao"]:
            avisos_selecao[(w["categoria"], w["mensagem"])] += w["contagem"]
        for w in repeticao["warnings"]["modelos"]:
            avisos_modelos[(w["categoria"], w["mensagem"])] += w["contagem"]
        relatorio_repeticoes.append({
            "repeticao": ctx.repeticao, "seed_cv_externa": ctx.seeds["cv_externa"],
            "auc_oof": repeticao["auc_oof"], "contrastes_descritivos": repeticao["contrastes_descritivos"],
            "folds": folds, "folds_com_s_k_vazio": repeticao["folds_com_s_k_vazio"],
            "frequencia_entre_folds_externos": repeticao["frequencia_entre_folds_externos"],
            "warnings": repeticao["warnings"],
        })
    n_treinos = len(tamanhos)
    return {
        "formato": FORMATO_RELATORIO, "versao_formato": VERSAO, "rotulo": ROTULO, "aviso": AVISO,
        "atividade": ATIVIDADE, "finalidade": rst.Finalidade.EXPLORATORIA.value,
        "hash_configuracao": cfg.hash_configuracao(decisoes), "sha256_dados": dict(carga.sha256_entradas),
        "particoes_lidas": list(carga.particoes), "repeticoes": list(repeticoes),
        "n_folds_externos": cfg.ESPEC_CV.n_folds_externos, "seeds": seeds_manifesto(repeticoes),
        "amostra": piloto._resumo_amostra(amostra, amostra.y),
        "modelos": list(MODELOS), "contrastes": [f"{a}-{b}" for a, b in CONTRASTES],
        "fora_desta_modalidade": {"modelos_com_painel": list(cfg.MODELOS_COM_PAINEL), "permutacoes": False,
                                  "spline": False, "cv_com_lista_fixa": False,
                                  "particoes_nao_lidas": ["validation", "test"]},
        "decisoes_pendentes": [c for c, d in decisoes.items() if not d.aprovada],
        "por_repeticao": relatorio_repeticoes,
        "entre_repeticoes": {
            "unidade_de_resumo": UNIDADE_DE_RESUMO, "n_repeticoes": len(repeticoes),
            "leitura": "sensibilidade à partição; não é incerteza amostral nem estimativa independente",
            "auc_oof": {m: _descrever([r["auc_oof"][m] for r in relatorio_repeticoes]) for m in MODELOS},
            "contrastes_descritivos": {
                f"{a}-{b}": _descrever([r["contrastes_descritivos"][f"{a}-{b}"] for r in relatorio_repeticoes])
                for a, b in CONTRASTES},
        },
        "selecao": {
            "n_treinos_externos": n_treinos,
            "frequencia_nos_treinos_externos": {f: frequencia.get(f, 0) for f in carga.features},
            "tamanho_s_k": {"minimo": min(tamanhos), "maximo": max(tamanhos),
                            "media": math.fsum(tamanhos) / n_treinos,
                            "contagem_por_tamanho": {str(t): n for t, n in sorted(Counter(tamanhos).items())}},
            "treinos_externos_com_s_k_vazio": sum(t == 0 for t in tamanhos),
            "jaccard_painel_declarado": {"painel": list(PAINEL_JACCARD), "minimo": min(jaccards),
                                         "maximo": max(jaccards), "media": math.fsum(jaccards) / n_treinos},
        },
        "convergencia": {"refits_selecao": n_refits, "refits_selecao_sem_convergencia": refits_sem_convergencia,
                         "ajustes_avaliacao_logistica": n_ajustes,
                         "ajustes_avaliacao_sem_convergencia": ajustes_sem_convergencia},
        "warnings": {
            "selecao": [{"categoria": c, "mensagem": m, "contagem": n} for (c, m), n in sorted(avisos_selecao.items())],
            "modelos": [{"categoria": c, "mensagem": m, "contagem": n} for (c, m), n in sorted(avisos_modelos.items())],
        },
    }


# --------------------------------------------------------------------------- #
# Ponto de entrada
# --------------------------------------------------------------------------- #


def executar_cv_repetida(diretorio_checkpoints: Union[str, Path], destino: Union[str, Path], *,
                         decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
                         repo: Union[str, Path] = guardas.REPO_ROOT,
                         fonte: Optional[dados.FonteDados] = None) -> ResultadoCvRepetida:
    """Executa (ou retoma) a CV repetida exploratória e grava relatório e manifesto em ``destino``.

    Mesmas regras de destino e checkpoints do piloto (``piloto.executar_piloto``);
    os checkpoints ficam em ``<diretorio_checkpoints>/cv_repetida_exploratoria/``
    e nunca se misturam com os de outra atividade. ``fonte`` existe para testes
    com dados sintéticos; o padrão é a fonte oficial.
    """
    repo, destino = Path(repo), Path(destino)
    repeticoes = _validar_repeticoes(REPETICOES)
    caminho_relatorio, caminho_manifesto = piloto._exigir_destino(destino, repo, ARQUIVO_RELATORIO,
                                                                  ARQUIVO_MANIFESTO)
    autorizacao = piloto._autorizar(ATIVIDADE, MODELOS, decisoes, repo)
    fonte = fonte or dados.fonte_oficial()
    entradas = {"tabela": fonte.tabela.caminho, "split": fonte.split.caminho}
    sessao = ck.SessaoCheckpoints(diretorio_checkpoints, ATIVIDADE, autorizacao, entradas,
                                  decisoes=decisoes, repo=repo)
    tempos: Dict[str, Any] = {"selecao_s": {}, "unidades_s": {}}
    execucao = rst.Execucao(ATIVIDADE, rst.Finalidade.EXPLORATORIA, autorizacao, entradas,
                            seeds_manifesto(repeticoes), _parametros(repeticoes), decisoes=decisoes, repo=repo)
    relatorio = None
    try:
        with execucao:
            carga, frame, amostra = piloto._carregar_treino(ATIVIDADE, PARTICOES, autorizacao, decisoes, repo, fonte)
            contextos = []
            for r in repeticoes:
                tempos_r = tempos["selecao_s"].setdefault(f"r{r:03d}", {})
                contextos.append(piloto._Contexto(
                    carga, frame, amostra, fl.plano_repeticao(amostra, r), sessao, piloto.seeds_repeticao(r),
                    decisoes, repo, {"selecao_s": tempos_r, "unidades_s": tempos["unidades_s"]},
                    atividade=ATIVIDADE, repeticao=r, formato_payload=FORMATO_PAYLOAD))
            esperadas = [i for ctx in contextos for k in range(len(ctx.plano.externos))
                         for i in piloto._identidades_do_fold(ctx, k).values()]
            retomada = sessao.verificar(esperadas)  # aborta em slot corrompido, incompatível ou inesperado
            tempos["unidades_reaproveitadas"] = list(retomada.concluidas)
            por_repeticao = []
            for ctx in contextos:
                por_fold = [piloto._executar_fold(ctx, k) for k in range(len(ctx.plano.externos))]
                por_repeticao.append((ctx, por_fold, piloto._consolidar_repeticao(ctx, por_fold)))
            relatorio = _consolidar(carga, amostra, decisoes, repeticoes, por_repeticao)
            piloto._gravar_json(relatorio, caminho_relatorio)
            execucao.registrar_saida("relatorio", caminho_relatorio)
    finally:
        if execucao.manifesto is not None:
            rst.escrever_manifesto(execucao.manifesto, caminho_manifesto, repo)
    manifesto = execucao.manifesto
    caminho_tempos = piloto._gravar_tempos(tempos, manifesto, sessao, ATIVIDADE)
    return ResultadoCvRepetida(MappingProxyType(relatorio), caminho_relatorio, caminho_manifesto,
                               MappingProxyType(manifesto), caminho_tempos)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CV repetida exploratória 10 x 5 da Sprint 4 (somente treino, nunca definitiva).")
    parser.add_argument("--checkpoints", type=Path, required=True,
                        help="diretório local não versionado para checkpoints e tempos")
    parser.add_argument("--destino", type=Path, required=True,
                        help="diretório existente no repositório para relatório e manifesto")
    parser.add_argument("--confirmo-preliminar-exploratorio", action="store_true",
                        help="confirma que o resultado é preliminar e exploratório")
    args = parser.parse_args(argv)
    if not args.confirmo_preliminar_exploratorio:
        parser.error("confirme com --confirmo-preliminar-exploratorio")
    resultado = executar_cv_repetida(args.checkpoints, args.destino)
    entre = resultado.relatorio["entre_repeticoes"]
    medias = {m: round(d["media"], 4) for m, d in entre["auc_oof"].items()}
    contrastes = {c: {k: round(d[k], 4) for k in ("media", "minimo", "maximo")}
                  for c, d in entre["contrastes_descritivos"].items()}
    print(f"{ROTULO}: AUC OOF média entre repetições {medias}; contrastes {contrastes}")
    print(f"relatório: {resultado.caminho_relatorio.name}; manifesto: {resultado.caminho_manifesto.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
