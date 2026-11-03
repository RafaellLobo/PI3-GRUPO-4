"""Orquestrador da Etapa 3 — validação oficial (Seções 6, 7.3, 8 e 9 do protocolo).

Avalia M0–M4 **com a lista fixa** da Seção 5.1: ajusta cada modelo nos 417 nódulos de ``train`` e o
avalia nos 92 nódulos / 62 pacientes de ``validation``. Atividade ``etapa3_validacao``
(``config.Etapa.E3``), com leitura de ``train`` e ``validation`` e NUNCA de ``test``.

Sequência (nada ocorre antes da anterior):

1. destino livre; autorização das guardas para a atividade e para M0–M4 (exige P1, P2 e P3 aprovadas,
   com evidência no git; sem elas, ``BloqueioDecisao``/``AprovacaoNaoVerificada`` e NENHUM dado é lido);
2. sessão de checkpoints local (fora do repositório) e ``rastreabilidade.Execucao``;
3. carga de ``train`` + ``validation`` pela porta única ``dados.carregar``; contagens, classes e
   disjunção de pacientes conferidas;
4. invariante do M1 (Seção 6.2): M1 reajustado no treino tem de reproduzir ``baseline_metrics.json``
   (coeficientes, intercepto, média e escala do scaler). Só parâmetros; falha antes de qualquer predição;
5. uma unidade por modelo (checkpoint): ajuste no treino, predição na validação. Unidade concluída é
   reaproveitada na retomada (conferida contra o esperado); falha não conta como concluída;
6. consolidação SEMPRE a partir dos payloads gravados: AUC por modelo, Δ dos contrastes da Seção 8.1
   (M3−M0 primário, M4−M1 secundário, M1−M0 e M2−M0 descritivos), bootstrap pareado por paciente
   (``bootstrap.bootstrap_pareado``, Seção 8.2), leitura da Seção 9 e métricas descritivas da
   Seção 10.2 (Δ Brier, Δ log-loss, calibração, recall e especificidade em 0,5);
7. relatório agregado (sem identificadores, predições, p-valor ou significância), saída registrada na
   ``Execucao`` e manifesto gravado ao lado. Execução DEFINITIVA exige git limpo e ambiente esperado.

Fora desta etapa e NÃO implementado aqui: sensibilidade com spline (M0s/M3s, Seção 10.1),
circularidade (10.3), máscaras reduzidas e faixas de diâmetro (10.4), sensibilidades da lista
reproduzida (P3) e de ``C = 0,1`` (P7). O relatório declara a omissão; nenhuma delas muda a leitura da
Seção 9. A Etapa 3 não ajusta seleção: a lista é a fixa.

Ponto de entrada: ``executar_etapa3`` ou
``python -m src.incremental.validacao --checkpoints DIR --destino DIR --confirmo-etapa3-oficial``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import bootstrap as bs
from . import checkpoints as ck
from . import config as cfg
from . import dados
from . import guardas
from . import modelos
from . import piloto
from . import rastreabilidade as rst

ATIVIDADE = cfg.Etapa.E3.value
PARTICOES: Tuple[str, ...] = ("train", "validation")
MODELOS: Tuple[str, ...] = cfg.MODELOS_FIXOS
#: Contrastes da Seção 8.1, na ordem de prioridade; Δ = A − B.
CONTRASTES: Tuple[Tuple[str, str], ...] = (("M3", "M0"), ("M4", "M1"), ("M1", "M0"), ("M2", "M0"))
PAPEL_DO_CONTRASTE: Mapping[str, str] = MappingProxyType(
    {"M3-M0": "primario", "M4-M1": "secundario", "M1-M0": "descritivo", "M2-M0": "descritivo"})
REPETICAO = 0
FOLD = 0
ROTULO = "validacao_oficial_etapa3"
FORMATO_RELATORIO = "relatorio-etapa3-validacao-cartao4"
FORMATO_PAYLOAD = "payload-etapa3-validacao-cartao4"
VERSAO = 1
ARQUIVO_RELATORIO = "relatorio_etapa3_validacao.json"
ARQUIVO_MANIFESTO = "manifesto_etapa3_validacao.json"
CAMINHO_BASELINE = Path("reports") / "sprint3" / "baseline" / "baseline_metrics.json"
TOLERANCIAS: Mapping[str, Mapping[str, float]] = MappingProxyType({
    "scaler_mean": MappingProxyType({"rtol": 1e-10, "atol": 1e-12}),
    "scaler_scale": MappingProxyType({"rtol": 1e-10, "atol": 1e-12}),
    "coeficientes": MappingProxyType({"rtol": 1e-6, "atol": 1e-8}),
    "intercepto": MappingProxyType({"rtol": 1e-6, "atol": 1e-8}),
})
AVISO = ("Etapa 3 do protocolo (validação oficial), lista fixa da Seção 5.1. Resultados descritivos e "
         "condicionais, lidos exclusivamente pelo IC95% (Seções 8.3 e 9); nenhum resultado altera "
         "modelo, features, especificação ou critérios.")
LIMITACOES_DECLARADAS: Tuple[str, ...] = (
    "Sensibilidade com spline do diâmetro (Seção 10.1, M0s e M3s) não calculada nesta execução.",
    "Circularidade e dependência do tamanho (Seção 10.3) e máscaras reduzidas/faixas (Seção 10.4) "
    "não calculadas nesta execução.",
    "Sensibilidades da lista reproduzida (P3) e de C = 0,1 (P7) não calculadas nesta execução.",
    "A leitura da Seção 9 exige a comparação com a CV aninhada (Etapa 2) e, só como descrição, o teste "
    "(Etapa 5); ambas ficam fora deste relatório.",
)

_CHAVES_PAYLOAD = ("formato", "versao", "atividade", "modelo", "features", "nodulos_avaliacao", "proba",
                   "parametros_ajustados", "warnings", "n_treino")


class ValidacaoInvalida(RuntimeError):
    """Inconsistência de dados, checkpoints ou predições; nada é corrigido."""


class InvarianteM1Violado(ValidacaoInvalida):
    """Parâmetros de M1 diferentes de ``baseline_metrics.json`` (Seção 6.2)."""


@dataclass(frozen=True)
class ResultadoEtapa3:
    relatorio: Mapping[str, Any]
    caminho_relatorio: Path
    caminho_manifesto: Path
    manifesto: Mapping[str, Any]


def seeds() -> Dict[str, int]:
    return {"logistica_avaliacao": cfg.ESPEC_AVALIACAO.random_state, "bootstrap": cfg.ESPEC_BOOTSTRAP.seed}


# --------------------------------------------------------------------------- #
# Invariante do M1 (Seção 6.2)
# --------------------------------------------------------------------------- #


def _comparar(obtido: Mapping[str, float], referencia: Mapping[str, float], tol: Mapping[str, float]) -> dict:
    if set(obtido) != set(referencia):
        raise InvarianteM1Violado("chaves diferentes de baseline_metrics.json")
    termos = {n: {"obtido": float(obtido[n]), "referencia": float(referencia[n]),
                  "diferenca_absoluta": abs(float(obtido[n]) - float(referencia[n])),
                  "dentro_da_tolerancia": bool(np.isclose(obtido[n], referencia[n], rtol=tol["rtol"],
                                                          atol=tol["atol"]))} for n in sorted(obtido)}
    return {"tolerancia": dict(tol), "termos": termos,
            "dentro_da_tolerancia": all(t["dentro_da_tolerancia"] for t in termos.values())}


def verificar_invariante_m1(ajustado: modelos.ModeloAjustado, caminho_baseline: Path) -> dict:
    """Compara só PARÂMETROS ajustados de M1 com ``baseline_metrics.json``. Nenhuma predição é feita."""
    referencia = json.loads(Path(caminho_baseline).read_bytes().decode("utf-8"))
    nomes = list(cfg.FEATURES_M1)
    if ajustado.nome != "M1" or list(ajustado.features) != nomes or list(referencia["features"]) != nomes:
        raise InvarianteM1Violado("features de M1 diferentes das de baseline_metrics.json")
    scaler = ajustado.estimador.named_steps[modelos.PASSO_SCALER]
    logreg = ajustado.estimador.named_steps[modelos.PASSO_LOGREG]
    n_vistos = int(np.atleast_1d(scaler.n_samples_seen_)[0])
    if n_vistos != ajustado.n_treino or n_vistos != referencia["prevencao_vazamento"]["scaler_n_samples_seen"]:
        raise InvarianteM1Violado(f"o padronizador viu {n_vistos} linhas; a referência, "
                                  f"{referencia['prevencao_vazamento']['scaler_n_samples_seen']}")
    ref = referencia["coeficientes"]
    pares = {
        "coeficientes": ({n: float(c) for n, c in zip(nomes, logreg.coef_.ravel())}, ref["por_feature"]),
        "intercepto": ({"intercepto": float(logreg.intercept_[0])}, {"intercepto": ref["intercepto"]}),
        "scaler_mean": ({n: float(v) for n, v in zip(nomes, scaler.mean_)}, ref["scaler_mean"]),
        "scaler_scale": ({n: float(v) for n, v in zip(nomes, scaler.scale_)}, ref["scaler_scale"]),
    }
    grupos = {nome: _comparar(o, r, TOLERANCIAS[nome]) for nome, (o, r) in pares.items()}
    resultado = {"modelo": "M1", "n_linhas_de_ajuste": n_vistos, "grupos": grupos,
                 "dentro_da_tolerancia": all(g["dentro_da_tolerancia"] for g in grupos.values()),
                 "observacao": "Só parâmetros ajustados; nenhuma predição nem métrica de desempenho."}
    if not resultado["dentro_da_tolerancia"]:
        fora = [f"{g}/{n}" for g, d in grupos.items() for n, t in d["termos"].items()
                if not t["dentro_da_tolerancia"]]
        raise InvarianteM1Violado(f"parâmetros de M1 fora da tolerância: {fora}")
    return resultado


# --------------------------------------------------------------------------- #
# Unidades
# --------------------------------------------------------------------------- #


def _validar_payload(payload: Any, modelo: str, features: Tuple[str, ...], ids_avaliacao: List[str],
                     n_treino: int) -> dict:
    onde = f"{ATIVIDADE}-{modelo}"
    if not isinstance(payload, dict) or set(payload) != set(_CHAVES_PAYLOAD):
        raise ValidacaoInvalida(f"{onde}: payload com chaves inesperadas")
    esperado = {"formato": FORMATO_PAYLOAD, "versao": VERSAO, "atividade": ATIVIDADE, "modelo": modelo,
                "features": list(features), "nodulos_avaliacao": ids_avaliacao, "n_treino": n_treino}
    if {c: payload[c] for c in esperado} != esperado:
        raise ValidacaoInvalida(f"{onde}: payload de outra unidade, features, nódulos ou treino")
    proba = payload["proba"]
    if not (isinstance(proba, list) and len(proba) == len(ids_avaliacao)
            and all(type(p) is float and math.isfinite(p) and 0.0 <= p <= 1.0 for p in proba)):
        raise ValidacaoInvalida(f"{onde}: probabilidades ausentes, fora de [0, 1] ou desalinhadas")
    return payload


def _executar_unidade(sessao: ck.SessaoCheckpoints, identidade: ck.IdentidadeUnidade,
                      construido: modelos.ModeloConstruido, treino: Any, avaliacao: Any) -> dict:
    colunas = list(construido.features)
    try:
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter("always")
            ajustado = modelos.ajustar(construido, treino[colunas], treino[cfg.COLUNA_ALVO])
            proba = ajustado.prever_proba(avaliacao[colunas])
    except BaseException as erro:
        sessao.registrar_falha(identidade, erro)
        raise
    payload = {
        "formato": FORMATO_PAYLOAD, "versao": VERSAO, "atividade": ATIVIDADE, "modelo": construido.nome,
        "features": colunas, "nodulos_avaliacao": avaliacao[cfg.COLUNA_NODULO].tolist(),
        "proba": [float(p) for p in proba], "parametros_ajustados": piloto._parametros_ajustados(ajustado),
        "warnings": piloto._resumir_warnings(capturados), "n_treino": len(treino),
    }
    sessao.salvar(identidade, payload, {"n_treino": len(treino), "n_avaliacao": len(avaliacao),
                                        "n_features": len(colunas)})
    return payload


# --------------------------------------------------------------------------- #
# Consolidação
# --------------------------------------------------------------------------- #


def leitura_secao_9(inferior: float, superior: float) -> Dict[str, Any]:
    """Classificação da Seção 9 pelo IC95% [L, U] de Δ (texto do protocolo, sem acréscimo)."""
    if inferior > 0:
        codigo, texto = "L_maior_que_0", "evidência de ganho de discriminação do alvo radiológico"
    elif superior < 0:
        codigo, texto = "U_menor_que_0", "evidência de perda de discriminação"
    else:
        codigo, texto = "L_menor_ou_igual_a_0_menor_ou_igual_a_U", \
                        "ganho não demonstrado — inconclusivo, o que não equivale a ausência de ganho"
    return {"codigo": codigo, "texto_do_protocolo": texto}


def _referencia_exploratoria(ic: bs.Intervalo) -> Dict[str, Any]:
    ref = cfg.REFERENCIA_EXPLORATORIA_DELTA_AUC

    def lado(valor: float) -> str:
        return "acima" if valor > ref else "abaixo_ou_igual"

    return {"referencia": ref, "ponto": lado(ic.ponto), "limite_inferior": lado(ic.limite_inferior),
            "limite_superior": lado(ic.limite_superior),
            "nota": "referência exploratória de magnitude, só descritiva (Seção 9)"}


def _calibracao(y: np.ndarray, proba: np.ndarray) -> Dict[str, Any]:
    """Intercepto e inclinação de calibração: regressão logística de y sobre logit(p), sem regularização."""
    q = np.clip(proba, bs.EPS, 1.0 - bs.EPS)
    logito = np.log(q / (1.0 - q)).reshape(-1, 1)
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        modelo = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(logito, y)
    return {"intercepto": float(modelo.intercept_[0]), "inclinacao": float(modelo.coef_[0][0]),
            "convergiu": bool(int(np.max(modelo.n_iter_)) < 1000),
            "n_warnings": len(capturados)}


def _recall_especificidade(y: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    classe = (proba >= cfg.THRESHOLD).astype(np.int64)
    return {"threshold": cfg.THRESHOLD, "recall_classe_1": float(((classe == 1) & (y == 1)).sum() / (y == 1).sum()),
            "especificidade_classe_0": float(((classe == 0) & (y == 0)).sum() / (y == 0).sum())}


def _consolidar(payloads: Mapping[str, dict], y: np.ndarray, pacientes: np.ndarray,
                espec: cfg.EspecBootstrap) -> Tuple[bs.ResultadoBootstrap, Dict[str, Any]]:
    probas = {m: np.asarray(payloads[m]["proba"], dtype=np.float64) for m in MODELOS}
    resultado = bs.bootstrap_pareado(pacientes, y, probas, CONTRASTES, espec)
    modelos_rel = {}
    for m in MODELOS:
        p = payloads[m]
        aval = {k: v.como_dict() for k, v in ((met, resultado.por_modelo[met][m]) for met in bs.METRICAS)}
        modelos_rel[m] = {
            "n_features": len(p["features"]), "features": list(p["features"]),
            "n_iter": p["parametros_ajustados"].get("n_iter"),
            "metricas": aval, "calibracao": _calibracao(y, probas[m]),
            "descritivas_limiar_0_5": _recall_especificidade(y, probas[m]),
            "warnings": p["warnings"],
        }
    contrastes_rel = {}
    for a, b in CONTRASTES:
        par = f"{a}-{b}"
        auc = resultado.contrastes["auc"][par]
        contrastes_rel[par] = {
            "papel": PAPEL_DO_CONTRASTE[par], "a": a, "b": b,
            "auc": auc.como_dict(),
            "leitura_secao_9": leitura_secao_9(auc.limite_inferior, auc.limite_superior),
            "referencia_exploratoria_delta_auc": _referencia_exploratoria(auc),
            "brier": resultado.contrastes["brier"][par].como_dict(),
            "log_loss": resultado.contrastes["log_loss"][par].como_dict(),
        }
    return resultado, {"modelos": modelos_rel, "contrastes": contrastes_rel}


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #


def _especificacoes(decisoes: Mapping[str, cfg.Decisao]) -> Dict[str, cfg.EspecModelo]:
    return {m: cfg.modelo_fixo(m, decisoes) for m in MODELOS}


def _conferir_cargas(treino: Any, validacao: Any, fonte: dados.FonteDados) -> None:
    for nome, frame in (("train", treino), ("validation", validacao)):
        esperado = fonte.particoes[nome]
        obtido = {"nodulos": len(frame), "pacientes": int(frame[cfg.COLUNA_PACIENTE].nunique()),
                  "classe_0": int((frame[cfg.COLUNA_ALVO] == 0).sum()),
                  "classe_1": int((frame[cfg.COLUNA_ALVO] == 1).sum())}
        if obtido != {k: esperado[k] for k in obtido}:
            raise ValidacaoInvalida(f"partição {nome} com contagens inesperadas")
    if set(treino[cfg.COLUNA_PACIENTE]) & set(validacao[cfg.COLUNA_PACIENTE]):
        raise ValidacaoInvalida("paciente presente em train e validation")
    if set(treino[cfg.COLUNA_NODULO]) & set(validacao[cfg.COLUNA_NODULO]):
        raise ValidacaoInvalida("nódulo presente em train e validation")


def executar_etapa3(diretorio_checkpoints: Union[str, Path], destino: Union[str, Path], *,
                    finalidade: rst.Finalidade = rst.Finalidade.DEFINITIVA,
                    decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
                    repo: Union[str, Path] = guardas.REPO_ROOT,
                    fonte: Optional[dados.FonteDados] = None,
                    caminho_baseline: Optional[Union[str, Path]] = None,
                    espec_bootstrap: cfg.EspecBootstrap = cfg.ESPEC_BOOTSTRAP) -> ResultadoEtapa3:
    """Executa (ou retoma) a Etapa 3 e grava relatório e manifesto em ``destino``.

    ``finalidade`` DEFINITIVA (padrão) exige git limpo, ambiente esperado e autorização do HEAD atual.
    ``fonte``, ``caminho_baseline`` e ``espec_bootstrap`` existem para testes com dados sintéticos; os
    padrões são a fonte oficial, ``baseline_metrics.json`` do repositório e a Seção 8.2.
    """
    repo, destino = Path(repo), Path(destino)
    caminho_relatorio, caminho_manifesto = piloto._exigir_destino(destino, repo, ARQUIVO_RELATORIO,
                                                                  ARQUIVO_MANIFESTO)
    autorizacao = piloto._autorizar(ATIVIDADE, MODELOS, decisoes, repo)  # P1, P2 e P3; antes de ler dados
    especificacoes = _especificacoes(decisoes)
    baseline = Path(caminho_baseline) if caminho_baseline is not None else repo / CAMINHO_BASELINE
    fonte = fonte or dados.fonte_oficial()
    entradas = {"tabela": fonte.tabela.caminho, "split": fonte.split.caminho, "baseline_metrics": baseline}
    sessao = ck.SessaoCheckpoints(diretorio_checkpoints, ATIVIDADE, autorizacao, entradas,
                                  decisoes=decisoes, repo=repo)
    execucao = rst.Execucao(
        ATIVIDADE, finalidade, autorizacao, entradas, seeds(),
        {"particoes": list(PARTICOES), "modelos": list(MODELOS), "rotulo": ROTULO,
         "contrastes": [f"{a}-{b}" for a, b in CONTRASTES], "n_replicas": espec_bootstrap.n_replicas},
        decisoes=decisoes, repo=repo)
    relatorio = None
    try:
        with execucao:
            relatorio = _executar(sessao, autorizacao, especificacoes, decisoes, repo, fonte, baseline,
                                  espec_bootstrap, finalidade)
            piloto._gravar_json(relatorio, caminho_relatorio)
            execucao.registrar_saida("relatorio", caminho_relatorio)
    finally:
        if execucao.manifesto is not None:
            rst.escrever_manifesto(execucao.manifesto, caminho_manifesto, repo)
    return ResultadoEtapa3(MappingProxyType(relatorio), caminho_relatorio, caminho_manifesto,
                           MappingProxyType(execucao.manifesto))


def _executar(sessao: ck.SessaoCheckpoints, autorizacao: guardas.Autorizacao,
              especificacoes: Mapping[str, cfg.EspecModelo], decisoes: Mapping[str, cfg.Decisao], repo: Path,
              fonte: dados.FonteDados, baseline: Path, espec_bootstrap: cfg.EspecBootstrap,
              finalidade: rst.Finalidade) -> dict:
    features = cfg.uniao_ordenada(*(especificacoes[m].features for m in MODELOS))
    carga = dados.carregar(ATIVIDADE, PARTICOES, features, decisoes, repo, fonte)
    if (carga.atividade, carga.particoes, carga.autorizacao) != (ATIVIDADE, PARTICOES, autorizacao):
        raise ValidacaoInvalida("carga com atividade, partições ou autorização inesperadas")
    treino, validacao = carga.frame("train"), carga.frame("validation")
    _conferir_cargas(treino, validacao, fonte)
    ids_avaliacao = validacao[cfg.COLUNA_NODULO].tolist()
    fingerprint = ck.fingerprint_folds({"treino": treino[cfg.COLUNA_NODULO].tolist(),
                                        "avaliacao": ids_avaliacao})
    identidades = {m: sessao.identidade(REPETICAO, FOLD, m, seeds(), fingerprint) for m in MODELOS}
    sessao.verificar(list(identidades.values()))  # aborta em slot corrompido ou inesperado
    construidos = {m: modelos.construir_fixo(m, ATIVIDADE, decisoes, repo) for m in MODELOS}
    for m in MODELOS:
        if construidos[m].features != especificacoes[m].features:
            raise ValidacaoInvalida(f"{m}: composição diverge da especificação")

    # Invariante do M1 (Seção 6.2): reajuste do M1 só para conferir parâmetros, antes de qualquer avaliação.
    invariante = verificar_invariante_m1(
        modelos.ajustar(construidos["M1"], treino[list(construidos["M1"].features)], treino[cfg.COLUNA_ALVO]),
        baseline)

    payloads: Dict[str, dict] = {}
    for m in MODELOS:
        if sessao.estado(identidades[m]) == ck.ESTADO_CONCLUIDA:
            payloads[m] = _validar_payload(sessao.carregar(identidades[m]), m, construidos[m].features,
                                           ids_avaliacao, len(treino))
        else:
            payloads[m] = _validar_payload(
                _executar_unidade(sessao, identidades[m], construidos[m], treino, validacao),
                m, construidos[m].features, ids_avaliacao, len(treino))

    y = validacao[cfg.COLUNA_ALVO].to_numpy(dtype=np.int64)
    resultado, partes = _consolidar(payloads, y, validacao[cfg.COLUNA_PACIENTE].to_numpy(), espec_bootstrap)
    return {
        "formato": FORMATO_RELATORIO, "versao_formato": VERSAO, "rotulo": ROTULO, "aviso": AVISO,
        "atividade": ATIVIDADE, "finalidade": finalidade.value,
        "hash_configuracao": cfg.hash_configuracao(decisoes), "sha256_dados": dict(carga.sha256_entradas),
        "particoes_lidas": list(carga.particoes), "seeds": seeds(),
        "autorizacao": autorizacao.como_dict(),
        "decisoes_exigidas": list(autorizacao.decisoes_exigidas),
        "amostra": {
            "treino": {"n_nodulos": len(treino), "n_pacientes": int(treino[cfg.COLUNA_PACIENTE].nunique()),
                       "classe_0": int((treino[cfg.COLUNA_ALVO] == 0).sum()),
                       "classe_1": int((treino[cfg.COLUNA_ALVO] == 1).sum())},
            "validacao": {"n_nodulos": len(validacao), "n_pacientes": int(validacao[cfg.COLUNA_PACIENTE].nunique()),
                          "classe_0": int((y == 0).sum()), "classe_1": int((y == 1).sum())},
        },
        "fora_desta_modalidade": {"particoes_nao_lidas": ["test"], "selecao_de_features": False,
                                  "permutacoes": False, "spline": False, "teste_de_hipotese": False},
        "unidades": {"total": len(MODELOS), "modelos": list(MODELOS)},
        "invariante_m1": invariante,
        "bootstrap": resultado.como_dict(),
        "modelos": partes["modelos"],
        "contrastes": partes["contrastes"],
        "limitacoes_declaradas": list(LIMITACOES_DECLARADAS),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Etapa 3 do Cartão 4 (validação oficial, lista fixa).")
    parser.add_argument("--checkpoints", type=Path, required=True,
                        help="diretório local não versionado para os checkpoints")
    parser.add_argument("--destino", type=Path, required=True,
                        help="diretório existente no repositório para relatório e manifesto")
    parser.add_argument("--confirmo-etapa3-oficial", action="store_true",
                        help="confirma a execução oficial da Etapa 3 (lê treino e validação)")
    args = parser.parse_args(argv)
    if not args.confirmo_etapa3_oficial:
        parser.error("confirme com --confirmo-etapa3-oficial")
    resultado = executar_etapa3(args.checkpoints, args.destino)
    primario = resultado.relatorio["contrastes"]["M3-M0"]
    print(f"Etapa 3 concluída: M3-M0 {primario['auc']}; leitura {primario['leitura_secao_9']['codigo']}")
    print(f"relatório: {resultado.caminho_relatorio.name}; manifesto: {resultado.caminho_manifesto.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["ATIVIDADE", "CONTRASTES", "InvarianteM1Violado", "MODELOS", "ResultadoEtapa3", "ValidacaoInvalida",
           "executar_etapa3", "leitura_secao_9", "seeds", "verificar_invariante_m1"]
