"""Orquestrador mínimo do piloto exploratório — Cartão 4, Sprint 4 (Incremento B).

Executa a atividade ``config.ATIVIDADE_PILOTO_EXPLORATORIO`` — distinta das
Etapas 1–5 e nunca definitiva — sobre a partição ``train``:

1. autorização pelas guardas (atividade e modelos) e sessão de checkpoints;
2. ``Execucao`` EXPLORATORIA: carga só de ``train`` (``dados.carregar``),
   ``Amostra`` e plano da repetição do piloto (r = 0, cinco folds externos
   agrupados por paciente, com estabilidade e internos);
3. por fold externo k: S_k calculado UMA vez (``selecao``) e reutilizado em
   M2v, M3v e M4v; M0, M1, M2v, M3v e M4v ajustados só no treino externo k e
   aplicados só à avaliação externa k (``modelos``);
4. cada unidade (fold, modelo) é um checkpoint local. Na retomada, a seleção
   recalculada tem de coincidir com a gravada no S_k E no hash do registro
   completo; divergência aborta;
5. consolidação SEMPRE a partir dos payloads gravados: uma predição OOF por
   nódulo e modelo, AUC global por modelo e contrastes descritivos M3v − M0 e
   M4v − M1; AUC por fold só como complemento;
6. relatório agregado ``preliminar_exploratorio`` (sem IDs, predições
   individuais, p-valores ou significância), registrado como saída da
   ``Execucao``; manifesto exploratório gravado ao lado. Tempos ficam num
   arquivo à parte, no diretório local de checkpoints, fora das saídas
   registradas: o relatório é idêntico entre execuções e retomadas.

Os passos 1–5 recebem atividade, repetição e formato do payload pelo
``_Contexto`` (nada disso é constante global nas funções internas), para que
outra atividade exploratória reutilize a execução por fold e a consolidação
por repetição sem copiá-las. O piloto continua fixo em r = 0.

Fora desta modalidade, sem parâmetro que os habilite: permutações, spline,
M2–M4 fixos e as partições ``validation`` e ``test``. Inconsistências de
configuração, folds, seleção, checkpoints ou autorizações levantam exceção;
nada é recuperado em silêncio. Sem paralelismo.

Ponto de entrada: ``executar_piloto`` ou
``python -m src.incremental.piloto --checkpoints DIR --destino DIR
--confirmo-preliminar-exploratorio``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score

from . import checkpoints as ck
from . import config as cfg
from . import dados
from . import folds as fl
from . import guardas
from . import modelos
from . import rastreabilidade as rst
from . import selecao

ATIVIDADE = cfg.ATIVIDADE_PILOTO_EXPLORATORIO
PARTICOES: Tuple[str, ...] = ("train",)
#: Piloto = repetição cuja seed é ``ESPEC_CV.seed_piloto`` (r = 0).
REPETICAO = cfg.ESPEC_CV.seeds_repeticoes().index(cfg.ESPEC_CV.seed_piloto)
MODELOS_FIXOS_PILOTO: Tuple[str, ...] = ("M0", "M1")
MODELOS: Tuple[str, ...] = MODELOS_FIXOS_PILOTO + cfg.MODELOS_VARIAVEIS
CONTRASTES: Tuple[Tuple[str, str], ...] = (("M3v", "M0"), ("M4v", "M1"))
ROTULO = "preliminar_exploratorio"
FORMATO_RELATORIO = "relatorio-piloto-exploratorio-cartao4"
FORMATO_PAYLOAD = "payload-piloto-exploratorio-cartao4"
VERSAO = 1
ARQUIVO_RELATORIO = "relatorio_piloto_exploratorio.json"
ARQUIVO_MANIFESTO = "manifesto_piloto_exploratorio.json"
AVISO = ("Resultado preliminar exploratório da atividade piloto_exploratorio: fora das Etapas 1-5 do "
         "protocolo, sem aprovação do grupo ou do professor, sem validação nem teste. Sem p-valor, "
         "intervalo de confiança ou teste de significância; AUC e contrastes são apenas descritivos.")

_CHAVES_PAYLOAD = ("formato", "versao", "atividade", "repeticao", "fold", "modelo", "features",
                   "nodulos_avaliacao", "proba", "parametros_ajustados", "warnings", "selecao")

if set(MODELOS) & set(cfg.MODELOS_COM_PAINEL):  # pragma: no cover — proteção de edição
    raise RuntimeError("o piloto exploratório não usa M2, M3 nem M4")


class PilotoInvalido(RuntimeError):
    """Inconsistência de entrada, folds, checkpoints ou predições; nada é corrigido."""


class SelecaoDivergente(PilotoInvalido):
    """Seleção recalculada diferente da gravada (S_k ou registro completo)."""


@dataclass(frozen=True)
class ResultadoPiloto:
    relatorio: Mapping[str, Any]
    caminho_relatorio: Path
    caminho_manifesto: Path
    manifesto: Mapping[str, Any]
    caminho_tempos: Path


# --------------------------------------------------------------------------- #
# Configuração da execução
# --------------------------------------------------------------------------- #


def seeds_repeticao(repeticao: int) -> Dict[str, int]:
    """Seeds de uma unidade da repetição ``repeticao``, todas de ``config``."""
    selecao_espec = cfg.ESPEC_SELECAO
    return {
        "cv_externa": cfg.ESPEC_CV.seeds_repeticoes()[repeticao],
        "estabilidade": selecao_espec.seed_estabilidade,
        "interna_base": selecao_espec.seed_interna_base,
        "logistica_selecao": selecao_espec.random_state,
        "logistica_avaliacao": cfg.ESPEC_AVALIACAO.random_state,
    }


def seeds_piloto() -> Dict[str, int]:
    """Seeds da unidade, todas de ``config``."""
    return seeds_repeticao(REPETICAO)


def _parametros() -> Dict[str, Any]:
    return {"repeticao": REPETICAO, "particoes": list(PARTICOES), "modelos": list(MODELOS),
            "contrastes": [f"{a}-{b}" for a, b in CONTRASTES], "rotulo": ROTULO,
            "n_folds_externos": cfg.ESPEC_CV.n_folds_externos}


def _exigir_destino(destino: Path, repo: Path, arquivo_relatorio: str = ARQUIVO_RELATORIO,
                    arquivo_manifesto: str = ARQUIVO_MANIFESTO) -> Tuple[Path, Path]:
    if not destino.is_dir():
        raise PilotoInvalido("diretório de destino inexistente")
    relativo = rst.caminho_relativo(destino, repo)
    if relativo == "data" or relativo.startswith("data/"):
        raise PilotoInvalido("o relatório exploratório não é gravado em data/")
    relatorio, manifesto = destino / arquivo_relatorio, destino / arquivo_manifesto
    existentes = [p.name for p in (relatorio, manifesto) if p.exists()]
    if existentes:
        raise rst.SobrescritaRecusada(f"já existem no destino: {existentes}; nada é sobrescrito")
    return relatorio, manifesto


def _autorizar(atividade: str, modelos_usados: Sequence[str], decisoes: Mapping[str, cfg.Decisao],
               repo: Path) -> guardas.Autorizacao:
    autorizacao = guardas.exigir_autorizacao(atividade, decisoes, repo)
    if guardas.exigir_autorizacao(atividade, decisoes, repo, modelos=modelos_usados) != autorizacao:
        raise PilotoInvalido("os modelos exigiriam decisões além das da atividade")
    return autorizacao


# --------------------------------------------------------------------------- #
# Unidades: payloads e validação
# --------------------------------------------------------------------------- #


def _resumir_warnings(capturados) -> List[dict]:
    contagem = Counter((w.category.__name__, rst._sanear_mensagem(str(w.message))) for w in capturados)
    return [{"categoria": c, "mensagem": m, "contagem": n} for (c, m), n in sorted(contagem.items())]


def _parametros_ajustados(ajustado: modelos.ModeloAjustado) -> dict:
    passos = ajustado.estimador.named_steps
    if modelos.PASSO_DUMMY in passos:
        return {"tipo": "dummy", "prior": [float(v) for v in passos[modelos.PASSO_DUMMY].class_prior_]}
    scaler, logreg = passos[modelos.PASSO_SCALER], passos[modelos.PASSO_LOGREG]
    nomes = ajustado.features
    return {
        "tipo": "logistica",
        "coeficientes": {n: float(c) for n, c in zip(nomes, logreg.coef_.ravel())},
        "intercepto": float(logreg.intercept_[0]),
        "scaler_media": {n: float(v) for n, v in zip(nomes, scaler.mean_)},
        "scaler_escala": {n: float(v) for n, v in zip(nomes, scaler.scale_)},
        "n_iter": int(np.max(logreg.n_iter_)),
    }


def _validar_payload(ctx: "_Contexto", payload: Any, modelo: str, k: int, features: Tuple[str, ...],
                     ids_avaliacao: List[str]) -> dict:
    onde = f"r{ctx.repeticao:03d}-f{k:02d}-{modelo}"
    if not isinstance(payload, dict) or set(payload) != set(_CHAVES_PAYLOAD):
        raise PilotoInvalido(f"{onde}: payload com chaves inesperadas")
    esperado = {"formato": ctx.formato_payload, "versao": VERSAO, "atividade": ctx.atividade,
                "repeticao": ctx.repeticao, "fold": k, "modelo": modelo}
    if {c: payload[c] for c in esperado} != esperado:
        raise PilotoInvalido(f"{onde}: payload de outra unidade ou formato")
    if payload["features"] != list(features):
        raise PilotoInvalido(f"{onde}: features gravadas diferem das esperadas")
    if payload["nodulos_avaliacao"] != ids_avaliacao:
        raise PilotoInvalido(f"{onde}: nódulos de avaliação diferem do fold externo")
    proba = payload["proba"]
    if not (isinstance(proba, list) and len(proba) == len(ids_avaliacao)
            and all(type(p) is float and math.isfinite(p) and 0.0 <= p <= 1.0 for p in proba)):
        raise PilotoInvalido(f"{onde}: probabilidades ausentes, fora de [0, 1] ou desalinhadas")
    return payload


def _validar_registro_selecao(gravado: Any, k: int, fingerprint: str, amostra: fl.Amostra,
                              repeticao: int) -> Tuple[str, ...]:
    if not (isinstance(gravado, dict) and set(gravado) == {"sha256", "s_k", "registro"}):
        raise PilotoInvalido(f"fold {k}: seleção gravada com forma inválida")
    registro = gravado["registro"]
    if rst.sha256_canonico(registro) != gravado["sha256"]:
        raise PilotoInvalido(f"fold {k}: hash do registro de seleção não confere")
    esperado = {"repeticao": repeticao, "fold_externo": k, "fingerprint_folds": fingerprint,
                "sha256_amostra": amostra.sha256, "hash_configuracao": cfg.hash_configuracao()}
    if {c: registro.get(c) for c in esperado} != esperado:
        raise PilotoInvalido(f"fold {k}: registro de seleção de outro fold, amostra ou configuração")
    if gravado["s_k"] != registro["s_k"]:
        raise PilotoInvalido(f"fold {k}: S_k gravado difere do registro de seleção")
    return tuple(registro["s_k"])


# --------------------------------------------------------------------------- #
# Execução de um fold externo
# --------------------------------------------------------------------------- #


@dataclass
class _Contexto:
    """Estado de UMA repetição de uma atividade; as funções internas não leem globais de unidade."""

    carga: dados.DadosCarregados
    frame: Any
    amostra: fl.Amostra
    plano: fl.PlanoRepeticao
    sessao: ck.SessaoCheckpoints
    seeds: Dict[str, int]
    decisoes: Mapping[str, cfg.Decisao]
    repo: Path
    tempos: Dict[str, Any]
    atividade: str
    repeticao: int
    formato_payload: str

    def __post_init__(self) -> None:
        if self.plano.repeticao != self.repeticao or self.sessao.atividade != self.atividade \
                or self.carga.atividade != self.atividade:
            raise PilotoInvalido("contexto com repetição ou atividade incoerentes")


def _selecao_do_fold(ctx: _Contexto, k: int, fingerprint: str, gravados: Mapping[str, dict],
                     pendentes: Sequence[str]) -> Tuple[Optional[selecao.SelecaoFold], dict]:
    """Seleção do fold: recalculada só se algum Mv estiver pendente, sempre conferida com a gravada."""
    variaveis_gravados = {m: gravados[m] for m in cfg.MODELOS_VARIAVEIS if m in gravados}
    for payload in variaveis_gravados.values():
        _validar_registro_selecao(payload["selecao"], k, fingerprint, ctx.amostra, ctx.repeticao)
    shas = {p["selecao"]["sha256"] for p in variaveis_gravados.values()}
    if len(shas) > 1:
        raise SelecaoDivergente(f"fold {k}: M2v, M3v e M4v gravados com seleções diferentes")
    if not any(m in pendentes for m in cfg.MODELOS_VARIAVEIS):
        return None, next(iter(variaveis_gravados.values()))["selecao"]
    inicio = time.monotonic()
    sel = selecao.selecionar_fold_externo(ctx.amostra, ctx.carga.X(PARTICOES[0], ctx.carga.features),
                                          ctx.carga.features, ctx.plano, k)
    ctx.tempos["selecao_s"][str(k)] = round(time.monotonic() - inicio, 6)
    registro = sel.como_dict()
    gravado = {"sha256": rst.sha256_canonico(registro), "s_k": list(sel.s_k), "registro": registro}
    for modelo, payload in variaveis_gravados.items():
        if payload["selecao"]["s_k"] != gravado["s_k"]:
            raise SelecaoDivergente(f"fold {k}: S_k recalculado difere do gravado em {modelo}")
        if payload["selecao"]["sha256"] != gravado["sha256"]:
            raise SelecaoDivergente(f"fold {k}: registro de seleção recalculado difere do gravado em {modelo}")
    return sel, gravado


def _executar_unidade(ctx: _Contexto, identidade: ck.IdentidadeUnidade, construido: modelos.ModeloConstruido,
                      externa: fl.Divisao, selecao_gravada: Optional[dict]) -> dict:
    frame, alvo = ctx.frame, cfg.COLUNA_ALVO
    colunas = list(construido.features)
    treino, avaliacao = frame.iloc[externa.treino], frame.iloc[externa.avaliacao]
    inicio = time.monotonic()
    try:
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter("always")
            ajustado = modelos.ajustar(construido, treino[colunas], treino[alvo])
            proba = ajustado.prever_proba(avaliacao[colunas])
    except BaseException as erro:
        ctx.sessao.registrar_falha(identidade, erro)
        raise
    ctx.tempos["unidades_s"][identidade.slot] = round(time.monotonic() - inicio, 6)
    payload = {
        "formato": ctx.formato_payload, "versao": VERSAO, "atividade": ctx.atividade,
        "repeticao": ctx.repeticao, "fold": externa.fold_externo, "modelo": construido.nome,
        "features": colunas,
        "nodulos_avaliacao": avaliacao[cfg.COLUNA_NODULO].tolist(),
        "proba": [float(p) for p in proba],
        "parametros_ajustados": _parametros_ajustados(ajustado),
        "warnings": _resumir_warnings(capturados),
        "selecao": selecao_gravada,
    }
    resumo = {"n_treino": len(externa.treino), "n_avaliacao": len(externa.avaliacao),
              "n_features": len(colunas), "auc_fold": _auc(ctx.amostra.y[externa.avaliacao], payload["proba"])}
    if selecao_gravada is not None:
        resumo.update(s_k_tamanho=len(selecao_gravada["s_k"]), sha256_selecao=selecao_gravada["sha256"])
    ctx.sessao.salvar(identidade, payload, resumo)
    return payload


def _auc(rotulos: np.ndarray, proba: Sequence[float]) -> float:
    if set(np.asarray(rotulos).tolist()) != {0, 1}:
        raise PilotoInvalido("AUC indefinida: avaliação sem as duas classes")
    return float(roc_auc_score(rotulos, np.asarray(proba, dtype=np.float64)))


def _identidades_do_fold(ctx: _Contexto, k: int) -> Dict[str, ck.IdentidadeUnidade]:
    fingerprint = fl.fingerprint_unidade(ctx.amostra, ctx.plano, k)
    return {m: ctx.sessao.identidade(ctx.repeticao, k, m, ctx.seeds, fingerprint) for m in MODELOS}


def _executar_fold(ctx: _Contexto, k: int) -> Tuple[Dict[str, dict], dict]:
    externa = ctx.plano.externos[k]
    fingerprint = fl.fingerprint_unidade(ctx.amostra, ctx.plano, k)
    identidades = _identidades_do_fold(ctx, k)
    ids_avaliacao = ctx.frame[cfg.COLUNA_NODULO].iloc[externa.avaliacao].tolist()
    gravados: Dict[str, dict] = {}
    for modelo, identidade in identidades.items():
        if ctx.sessao.estado(identidade) == ck.ESTADO_CONCLUIDA:
            gravados[modelo] = ctx.sessao.carregar(identidade)
    pendentes = [m for m in MODELOS if m not in gravados]

    sel, selecao_gravada = _selecao_do_fold(ctx, k, fingerprint, gravados, pendentes)
    s_k = tuple(selecao_gravada["s_k"])
    especificacoes = {"M0": cfg.modelo_fixo("M0"), "M1": cfg.modelo_fixo("M1"),
                      **{m: cfg.modelo_variavel(m, s_k) for m in cfg.MODELOS_VARIAVEIS}}
    for modelo, payload in gravados.items():
        _validar_payload(ctx, payload, modelo, k, especificacoes[modelo].features, ids_avaliacao)
        if (modelo in cfg.MODELOS_VARIAVEIS) != (payload["selecao"] is not None):
            raise PilotoInvalido(f"fold {k}: {modelo} gravado com seleção indevida ou ausente")

    if pendentes:
        construidos = {m: modelos.construir_fixo(m, ctx.atividade, ctx.decisoes, ctx.repo)
                       for m in MODELOS_FIXOS_PILOTO if m in pendentes}
        if sel is not None:
            construidos.update(modelos.modelos_do_fold(sel, ctx.atividade, ctx.decisoes, ctx.repo))
        for modelo in pendentes:
            construido = construidos[modelo]
            if construido.features != especificacoes[modelo].features:
                raise PilotoInvalido(f"fold {k}: composição de {modelo} diverge do S_k do fold")
            gravada = selecao_gravada if modelo in cfg.MODELOS_VARIAVEIS else None
            payload = _executar_unidade(ctx, identidades[modelo], construido, externa, gravada)
            gravados[modelo] = _validar_payload(ctx, payload, modelo, k, construido.features, ids_avaliacao)
    return gravados, selecao_gravada


# --------------------------------------------------------------------------- #
# Consolidação e relatório
# --------------------------------------------------------------------------- #


def _consolidar_repeticao(ctx: _Contexto, por_fold: List[Tuple[Dict[str, dict], dict]]) -> dict:
    """Agregados de UMA repetição, sempre a partir dos payloads gravados.

    Devolve AUC OOF por modelo (uma predição por nódulo, cobertura exigida),
    resumo por fold, frequência de S_k entre os folds externos e warnings
    agregados de seleção e modelos.
    """
    amostra, plano = ctx.amostra, ctx.plano
    if len(por_fold) != len(plano.externos):
        raise PilotoInvalido("consolidação com número de folds diferente do plano")
    posicao = {nodulo: i for i, nodulo in enumerate(amostra.ids.tolist())}
    oof = {m: np.full(amostra.n, np.nan) for m in MODELOS}
    vistos = {m: np.zeros(amostra.n, dtype=np.int64) for m in MODELOS}
    folds_relatorio = []
    for k, (payloads, selecao_gravada) in enumerate(por_fold):
        externa = plano.externos[k]
        auc_fold = {}
        for modelo in MODELOS:
            payload = payloads[modelo]
            indices = np.array([posicao[n] for n in payload["nodulos_avaliacao"]], dtype=np.int64)
            if not np.array_equal(indices, externa.avaliacao):
                raise PilotoInvalido(f"fold {k}, {modelo}: predições fora da avaliação externa do fold")
            oof[modelo][indices] = payload["proba"]
            vistos[modelo][indices] += 1
            auc_fold[modelo] = _auc(amostra.y[indices], payload["proba"])
        registro = selecao_gravada["registro"]
        folds_relatorio.append({
            "fold": k, "fingerprint_folds": registro["fingerprint_folds"],
            "n_treino": len(externa.treino), "n_avaliacao": len(externa.avaliacao),
            "n_pacientes_avaliacao": len(set(amostra.grupos[externa.avaliacao].tolist())),
            "auc": auc_fold,
            "s_k": list(selecao_gravada["s_k"]), "s_k_vazio": not selecao_gravada["s_k"],
            "sha256_selecao": selecao_gravada["sha256"],
            "C_por_fold_estabilidade": [f["C"] for f in registro["folds_estabilidade"]],
        })
    for modelo in MODELOS:
        if not np.all(vistos[modelo] == 1) or not np.all(np.isfinite(oof[modelo])):
            raise PilotoInvalido(f"{modelo}: cobertura OOF incompleta ou duplicada")
    auc_oof = {m: _auc(amostra.y, oof[m]) for m in MODELOS}
    frequencia = Counter(nome for f in folds_relatorio for nome in f["s_k"])
    avisos_selecao, avisos_modelos = Counter(), Counter()
    for payloads, selecao_gravada in por_fold:
        for f in selecao_gravada["registro"]["folds_estabilidade"]:
            for w in f["warnings"]:
                avisos_selecao[(w["categoria"], w["mensagem"])] += w["contagem"]
        for payload in payloads.values():
            for w in payload["warnings"]:
                avisos_modelos[(w["categoria"], w["mensagem"])] += w["contagem"]
    return {
        "auc_oof": auc_oof,
        "contrastes_descritivos": {f"{a}-{b}": auc_oof[a] - auc_oof[b] for a, b in CONTRASTES},
        "folds": folds_relatorio,
        "frequencia_entre_folds_externos": dict(sorted(frequencia.items())),
        "folds_com_s_k_vazio": sum(f["s_k_vazio"] for f in folds_relatorio),
        "warnings": {
            "selecao": [{"categoria": c, "mensagem": m, "contagem": n} for (c, m), n in sorted(avisos_selecao.items())],
            "modelos": [{"categoria": c, "mensagem": m, "contagem": n} for (c, m), n in sorted(avisos_modelos.items())],
        },
    }


def _consolidar(ctx: _Contexto, por_fold: List[Tuple[Dict[str, dict], dict]]) -> dict:
    repeticao = _consolidar_repeticao(ctx, por_fold)
    amostra, rotulos = ctx.amostra, ctx.amostra.y
    return {
        "formato": FORMATO_RELATORIO, "versao_formato": VERSAO, "rotulo": ROTULO, "aviso": AVISO,
        "atividade": ATIVIDADE, "finalidade": rst.Finalidade.EXPLORATORIA.value,
        "hash_configuracao": cfg.hash_configuracao(ctx.decisoes), "sha256_dados": dict(ctx.carga.sha256_entradas),
        "particoes_lidas": list(ctx.carga.particoes), "repeticao": REPETICAO, "seeds": dict(ctx.seeds),
        "amostra": _resumo_amostra(amostra, rotulos),
        "modelos": list(MODELOS),
        "fora_desta_modalidade": {"modelos_com_painel": list(cfg.MODELOS_COM_PAINEL), "permutacoes": False,
                                  "spline": False, "particoes_nao_lidas": ["validation", "test"]},
        "decisoes_pendentes": [c for c, d in ctx.decisoes.items() if not d.aprovada],
        "auc_oof": repeticao["auc_oof"],
        "contrastes_descritivos": repeticao["contrastes_descritivos"],
        "folds": repeticao["folds"],
        "selecao": {"frequencia_entre_folds_externos": repeticao["frequencia_entre_folds_externos"],
                    "folds_com_s_k_vazio": repeticao["folds_com_s_k_vazio"]},
        "warnings": repeticao["warnings"],
    }


def _resumo_amostra(amostra: fl.Amostra, rotulos: np.ndarray) -> dict:
    return {"n_nodulos": amostra.n, "n_pacientes": len(set(amostra.grupos.tolist())),
            "classe_0": int((rotulos == 0).sum()), "classe_1": int((rotulos == 1).sum())}


def _gravar_json(conteudo: Mapping[str, Any], destino: Path) -> None:
    rst._exigir_sem_dados_individuais(dict(conteudo), destino.name)
    ck._publicar(rst.serializar(conteudo), destino)


# --------------------------------------------------------------------------- #
# Carga e encerramento compartilhados
# --------------------------------------------------------------------------- #


def _carregar_treino(atividade: str, particoes: Tuple[str, ...], autorizacao: guardas.Autorizacao,
                     decisoes: Mapping[str, cfg.Decisao], repo: Path, fonte: dados.FonteDados
                     ) -> Tuple[dados.DadosCarregados, Any, fl.Amostra]:
    """Carga de ``train`` pela porta única de ``dados``; contagens e amostra conferidas."""
    carga = dados.carregar(atividade, particoes, fonte.nomes_pool, decisoes, repo, fonte)
    if (carga.atividade, carga.particoes, carga.autorizacao) != (atividade, particoes, autorizacao):
        raise PilotoInvalido("carga com atividade, partições ou autorização inesperadas")
    frame = carga.frame(particoes[0])
    esperado = fonte.particoes[particoes[0]]
    if (len(frame), frame[cfg.COLUNA_PACIENTE].nunique()) != (esperado["nodulos"], esperado["pacientes"]):
        raise PilotoInvalido("partição de treino com contagens inesperadas")
    amostra = fl.criar_amostra(frame[cfg.COLUNA_NODULO], frame[cfg.COLUNA_ALVO],
                               frame[cfg.COLUNA_PACIENTE], X=frame)
    return carga, frame, amostra


def _gravar_tempos(tempos: Dict[str, Any], manifesto: Mapping[str, Any], sessao: ck.SessaoCheckpoints,
                   atividade: str) -> Path:
    tempos.update(sha256_manifesto=manifesto["sha256_manifesto"], duracao_total_s=manifesto["tempo"]["duracao_s"])
    caminho = sessao.diretorio / f"tempos-{atividade}-{manifesto['sha256_manifesto'][:16]}.json"
    _gravar_json(tempos, caminho)
    return caminho


# --------------------------------------------------------------------------- #
# Ponto de entrada
# --------------------------------------------------------------------------- #


def executar_piloto(diretorio_checkpoints: Union[str, Path], destino: Union[str, Path], *,
                    decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
                    repo: Union[str, Path] = guardas.REPO_ROOT,
                    fonte: Optional[dados.FonteDados] = None) -> ResultadoPiloto:
    """Executa (ou retoma) o piloto exploratório e grava relatório e manifesto em ``destino``.

    ``destino``: diretório existente dentro de ``repo`` (fora de ``data/``),
    sem relatório nem manifesto do piloto. ``diretorio_checkpoints``: local e
    não versionado (``checkpoints.SessaoCheckpoints``); guarda as predições
    individuais e o arquivo de tempos. ``fonte`` existe para testes com dados
    sintéticos; o padrão é a fonte oficial.
    """
    repo, destino = Path(repo), Path(destino)
    caminho_relatorio, caminho_manifesto = _exigir_destino(destino, repo)
    autorizacao = _autorizar(ATIVIDADE, MODELOS, decisoes, repo)
    fonte = fonte or dados.fonte_oficial()
    entradas = {"tabela": fonte.tabela.caminho, "split": fonte.split.caminho}
    sessao = ck.SessaoCheckpoints(diretorio_checkpoints, ATIVIDADE, autorizacao, entradas,
                                  decisoes=decisoes, repo=repo)
    seeds = seeds_piloto()
    tempos: Dict[str, Any] = {"selecao_s": {}, "unidades_s": {}}
    execucao = rst.Execucao(ATIVIDADE, rst.Finalidade.EXPLORATORIA, autorizacao, entradas, seeds,
                            _parametros(), decisoes=decisoes, repo=repo)
    relatorio = None
    try:
        with execucao:
            carga, frame, amostra = _carregar_treino(ATIVIDADE, PARTICOES, autorizacao, decisoes, repo, fonte)
            plano = fl.plano_repeticao(amostra, REPETICAO)
            ctx = _Contexto(carga, frame, amostra, plano, sessao, seeds, decisoes, repo, tempos,
                            atividade=ATIVIDADE, repeticao=REPETICAO, formato_payload=FORMATO_PAYLOAD)
            esperadas = [i for k in range(len(plano.externos)) for i in _identidades_do_fold(ctx, k).values()]
            retomada = sessao.verificar(esperadas)  # aborta em slot corrompido, incompatível ou inesperado
            tempos["unidades_reaproveitadas"] = list(retomada.concluidas)
            por_fold = [_executar_fold(ctx, k) for k in range(len(plano.externos))]
            relatorio = _consolidar(ctx, por_fold)
            _gravar_json(relatorio, caminho_relatorio)
            execucao.registrar_saida("relatorio", caminho_relatorio)
    finally:
        if execucao.manifesto is not None:
            rst.escrever_manifesto(execucao.manifesto, caminho_manifesto, repo)
    manifesto = execucao.manifesto
    caminho_tempos = _gravar_tempos(tempos, manifesto, sessao, ATIVIDADE)
    return ResultadoPiloto(MappingProxyType(relatorio), caminho_relatorio, caminho_manifesto,
                           MappingProxyType(manifesto), caminho_tempos)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Piloto exploratório da Sprint 4 (somente treino, nunca definitivo).")
    parser.add_argument("--checkpoints", type=Path, required=True,
                        help="diretório local não versionado para checkpoints e tempos")
    parser.add_argument("--destino", type=Path, required=True,
                        help="diretório existente no repositório para relatório e manifesto")
    parser.add_argument("--confirmo-preliminar-exploratorio", action="store_true",
                        help="confirma que o resultado é preliminar e exploratório")
    args = parser.parse_args(argv)
    if not args.confirmo_preliminar_exploratorio:
        parser.error("confirme com --confirmo-preliminar-exploratorio")
    resultado = executar_piloto(args.checkpoints, args.destino)
    print(f"{ROTULO}: AUC OOF {resultado.relatorio['auc_oof']}; "
          f"contrastes {resultado.relatorio['contrastes_descritivos']}")
    print(f"relatório: {resultado.caminho_relatorio.name}; manifesto: {resultado.caminho_manifesto.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
