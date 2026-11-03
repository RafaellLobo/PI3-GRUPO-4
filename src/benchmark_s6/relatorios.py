"""Tabelas, relatório científico e relatório de execução (P9, Seções 12 e 14.5).

Tudo o que sai daqui é AGREGADO e determinístico (bytes idênticos para a mesma
consolidação), exceto ``relatorio_execucao.json``, que traz tempos. Os CSV usam
``\\n`` e ``repr`` de ``float``; os JSON, a forma canônica da S4. Nenhuma função
lê dados, calcula AUC ou grava arquivos: devolvem bytes, e quem grava é o
coordenador (``manifesto.gravar_atomico``).

O texto de ``resumo_benchmark.md`` segue o vocabulário controlado da P9, Seção 8.3:
nada de "significativo", "superior", "comprovado" etc.; só "diferença estimada",
"intervalo descritivo" e afins. ``privacidade.varrer_texto`` confere.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import bootstrap as bs
from . import config as cfg
from . import consolidacao as co
from . import manifesto as mf
from . import modelos
from . import selecao as sl

VERSAO_ESQUEMA = 1
NOME_MODELO = {"lr_fixa": "LR fixa", "lr_tuning": "LR com ajuste", "svm": "SVM", "rf": "RF", "xgb": "XGBoost"}
NOME_DESFECHO = {"G": "G (AUC geral)", "F": "F (AUC na faixa de 12 a 30 mm)"}


# --------------------------------------------------------------------------- #
# Insumos
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Populacao:
    pacientes: int
    nodulos: int
    classe_0: int
    classe_1: int
    faixa: Mapping[str, int]


@dataclass(frozen=True)
class Identificacao:
    """Quem gerou os números: tudo aqui vem do manifesto/da execução, sem caminho local."""

    modo: str
    oficial: bool
    sha256_configuracao: str
    head: str
    branch: Optional[str]
    ambiente: Mapping[str, Any]
    sha256_entradas: Mapping[str, str]
    n_unidades: int


@dataclass(frozen=True, eq=False)
class DadosRelatorio:
    populacao: Populacao
    identificacao: Identificacao
    consolidacao: co.Consolidacao
    bootstrap: bs.ResultadoBootstrap
    selecao_entrada: Mapping[str, Any]
    selecao: sl.ResultadoSelecao
    plano_resumo: Mapping[str, Any]
    #: Nomes e SHA-256 (LF) dos demais artefatos já gerados, listados no resumo (sem o próprio resumo).
    hashes_de_artefatos: Mapping[str, str]
    figuras: Sequence[str]


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def _celula(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("valor não finito em CSV")
        return repr(v)
    return str(v)


def csv_bytes(cabecalho: Sequence[str], linhas: Sequence[Sequence[Any]]) -> bytes:
    saida = io.StringIO()
    escritor = csv.writer(saida, lineterminator="\n")
    escritor.writerow(cabecalho)
    for linha in linhas:
        if len(linha) != len(cabecalho):
            raise ValueError("linha de CSV com número de colunas diferente do cabeçalho")
        escritor.writerow([_celula(v) for v in linha])
    return saida.getvalue().encode("utf-8")


def _configuracao_texto(configuracao: Sequence[Tuple[str, Any]]) -> str:
    return ";".join(f"{nome}={valor!r}" for nome, valor in configuracao)


def tabelas(c: co.Consolidacao, b: bs.ResultadoBootstrap, selecao: sl.ResultadoSelecao, plano_resumo: Mapping[str, Any]
            ) -> Dict[str, bytes]:
    """Os CSV e JSON de agregados da P9, Seção 14.5, indexados pelo nome do arquivo."""
    saidas: Dict[str, bytes] = {}
    saidas["config_benchmark.json"] = cfg.bytes_arquivo()
    saidas["plano_folds.json"] = mf.bytes_json(dict(plano_resumo))
    saidas["denominadores.csv"] = csv_bytes(
        ["repeticao", "dobra", "desfecho", "nodulos", "pacientes", "positivos", "negativos", "valida", "estado"],
        [(d.repeticao, d.dobra, d.desfecho, d.nodulos, d.pacientes, d.positivos, d.negativos, d.valida,
          "" if d.valida else "DD") for d in c.denominadores])
    saidas["auc_por_fold.csv"] = csv_bytes(
        ["modelo", "repeticao", "dobra", "desfecho", "auc", "valida"],
        [(a.modelo, a.repeticao, a.dobra, a.desfecho, a.auc, a.valida) for a in c.auc_por_dobra])
    saidas["auc_por_repeticao.csv"] = csv_bytes(
        ["modelo", "repeticao", "desfecho", "n_dobras_validas", "valida", "auc", "estado"],
        [(a.modelo, a.repeticao, a.desfecho, a.n_dobras_validas, a.valida, a.auc, "" if a.valida else "RI")
         for a in c.auc_por_repeticao])
    ic = {(i.a, i.b, i.desfecho): i for i in b.intervalos_contraste}
    cab = ["a", "b", "desfecho", "grupo", "estimavel", "n_repeticoes_validas", "delta", "minimo_repeticoes",
           "maximo_repeticoes", "n_repeticoes_positivas", "fracao_repeticoes_positivas", "ic_relatado",
           "motivo_sem_ic", "ic95_baixo", "ic95_alto", "bonferroni_baixo", "bonferroni_alto"] + \
        [f"delta_r{r:02d}" for r in c.repeticoes]
    linhas = []
    for x in c.contrastes:
        i = ic[(x.a, x.b, x.desfecho)]
        linhas.append([x.a, x.b, x.desfecho, x.grupo, x.estimavel, x.n_repeticoes_validas, x.delta, x.minimo,
                       x.maximo, x.n_repeticoes_positivas, x.fracao_repeticoes_positivas, i.relatado, i.motivo,
                       i.ic95_baixo, i.ic95_alto, i.bonferroni_baixo, i.bonferroni_alto, *x.delta_por_repeticao])
    saidas["contrastes.csv"] = csv_bytes(cab, linhas)
    saidas["bootstrap_resumo.json"] = mf.bytes_json(bootstrap_resumo(b))
    chaves = [(a, bb, d) for a, bb, _ in co.pares_de_contraste() for d in co.DESFECHOS]
    colunas = ["replica"] + [f"{a}_contra_{bb}_{d}" for a, bb, d in chaves]
    saidas["bootstrap_delta.csv"] = csv_bytes(
        colunas, [[n] + [None if np.isnan(b.delta_replicas[k][n]) else float(b.delta_replicas[k][n]) for k in chaves]
                  for n in range(b.n_replicas)])
    saidas["hiperparametros_escolhidos.csv"] = csv_bytes(
        ["modelo", "repeticao", "dobra", "indice_configuracao", "n_configuracoes", "configuracao", "na_borda",
         "n_ajustes"],
        [(e.modelo, e.repeticao, e.dobra, e.indice_configuracao, e.n_configuracoes, _configuracao_texto(e.configuracao),
          e.na_borda, e.ajustes) for e in c.escolhas])
    linhas = [("dobra", d.modelo, d.repeticao, d.dobra, d.metrica, d.rotulo, None, d.valor, None, None, None)
              for d in c.descritivas]
    linhas += [("modelo", s.modelo, None, None, s.metrica, s.rotulo, s.n_dobras, None, s.media, s.minimo, s.maximo)
               for s in c.resumo_descritivas]
    saidas["metricas_descritivas.csv"] = csv_bytes(
        ["nivel", "modelo", "repeticao", "dobra", "metrica", "rotulo", "n_dobras", "valor", "media", "minimo",
         "maximo"], linhas)
    saidas["selecao_algoritmo.json"] = mf.bytes_json(selecao.como_dict())
    return saidas


def bootstrap_resumo(b: bs.ResultadoBootstrap) -> Dict[str, Any]:
    return {
        "versao_esquema": VERSAO_ESQUEMA, "n_replicas": b.n_replicas, "semente": b.semente,
        "gerador": cfg.CONFIGURACAO.bootstrap.gerador, "unidade_de_reamostragem": cfg.CONFIGURACAO.bootstrap.unidade,
        "reajuste": False, "percentis_ic95_descritivo": list(b.percentis_descritivos),
        "percentis_familia_bonferroni": list(b.percentis_familia),
        "descartes": [d._asdict() for d in b.descartes],
        "intervalos_theta": [i._asdict() for i in b.intervalos_theta],
        "intervalos_contraste": [i._asdict() for i in b.intervalos_contraste],
    }


# --------------------------------------------------------------------------- #
# relatorio_benchmark.json
# --------------------------------------------------------------------------- #

LIMITES_DO_INTERVALO = (
    ("Variabilidade do treinamento: não há reajuste nas réplicas", "O intervalo tende a ser estreito demais."),
    ("Variabilidade da escolha de hiperparâmetros", "O intervalo tende a ser estreito demais."),
    ("Dependência entre dobras e repetições, cujos treinos se sobrepõem",
     "A partição só aparece como mínimo e máximo entre repetições."),
    ("A validação cruzada estima o desempenho esperado do procedimento, não de um modelo final",
     "Não há modelo final validado."),
    ("Os pacientes não são amostra aleatória de uma população clínica; o alvo é suspeita radiológica",
     "Não se generaliza para a prática clínica."),
    ("Reuso dos pacientes de validação depois da Etapa 3 da S4, e as escolhas analíticas",
     "Declarado como limitação."),
)

LIMITES_DE_BONFERRONI = (
    "O ajuste pressupõe que os intervalos tenham a cobertura declarada; os daqui são condicionais e tendem a ser "
    "estreitos, então a correção NÃO garante controle do erro de família: é uma convenção de largura.",
    "É conservador com contrastes correlacionados: os seis compartilham o comparador, e o desfecho F é um subconjunto "
    "do G.",
    "Não cobre outras escolhas analíticas (grades, desfechos, regras de validade), nem os contrastes descritivos.",
    "Exige caudas extremas no bootstrap: com B = 10.000, cada cauda de 0,417% se apoia em cerca de 42 réplicas.",
)

DECLARACAO_TESTE = (
    "A partição histórica de teste não foi usada na modelagem, no ajuste, na validação cruzada, nas métricas, no "
    "bootstrap nem na seleção; seus identificadores só entram na verificação estrutural pública da S4. Este "
    "benchmark é independente das métricas da S4: não reaproveita nenhum resultado dela.")


def relatorio_benchmark(d: DadosRelatorio) -> Dict[str, Any]:
    """Todos os números do resumo, em esquema versionado, suficientes para recalcular a seleção."""
    c, b = d.consolidacao, d.bootstrap
    escolhido = d.selecao.algoritmo
    ic_theta = {(i.modelo, i.desfecho): i for i in b.intervalos_theta}
    estimativas = []
    for e in c.estimativas:
        i = ic_theta[(e.modelo, e.desfecho)]
        estimativas.append({
            "modelo": e.modelo, "desfecho": e.desfecho, "estimavel": e.estimavel, "theta": e.theta,
            "n_repeticoes_validas": e.n_repeticoes_validas, "minimo_repeticoes": e.minimo_repeticoes,
            "maximo_repeticoes": e.maximo_repeticoes, "ic_relatado": i.relatado, "motivo_sem_ic": i.motivo,
            "ic95_descritivo_baixo": i.limite_baixo, "ic95_descritivo_alto": i.limite_alto,
            "rotulo": "otimista por seleção" if e.modelo == escolhido else ""})
    ic_ct = {(i.a, i.b, i.desfecho): i for i in b.intervalos_contraste}
    contrastes = []
    for x in c.contrastes:
        i = ic_ct[(x.a, x.b, x.desfecho)]
        contrastes.append({
            "a": x.a, "b": x.b, "desfecho": x.desfecho, "grupo": x.grupo, "estimavel": x.estimavel, "delta": x.delta,
            "n_repeticoes_validas": x.n_repeticoes_validas, "minimo_repeticoes": x.minimo,
            "maximo_repeticoes": x.maximo,
            "n_repeticoes_positivas": x.n_repeticoes_positivas,
            "fracao_repeticoes_positivas": x.fracao_repeticoes_positivas,
            "delta_por_repeticao": list(x.delta_por_repeticao), "ic_relatado": i.relatado, "motivo_sem_ic": i.motivo,
            "ic95_baixo": i.ic95_baixo, "ic95_alto": i.ic95_alto, "bonferroni_baixo": i.bonferroni_baixo,
            "bonferroni_alto": i.bonferroni_alto})
    frequencia: Dict[str, Dict[str, int]] = {}
    borda: Dict[str, int] = {}
    for e in c.escolhas:
        chave = _configuracao_texto(e.configuracao) or "(sem ajuste)"
        frequencia.setdefault(e.modelo, {})
        frequencia[e.modelo][chave] = frequencia[e.modelo].get(chave, 0) + 1
        borda[e.modelo] = borda.get(e.modelo, 0) + (1 if e.na_borda else 0)
    estados = {e.desfecho: {"DD": e.dobras_degeneradas, "RI": e.repeticoes_invalidas, "NE": e.nao_estimavel,
                            "dobras": e.n_dobras, "repeticoes": e.n_repeticoes,
                            "repeticoes_validas": e.repeticoes_validas} for e in c.estados}
    for dd in b.descartes:
        estados[dd.desfecho].update({"RD": dd.descartadas, "fracao_RD": dd.fracao, "IQ": dd.iq})
    i = d.identificacao
    return {
        "versao_esquema": VERSAO_ESQUEMA,
        "identificacao": {"modo": i.modo, "oficial": i.oficial, "sha256_configuracao": i.sha256_configuracao,
                          "head": i.head, "branch": i.branch, "ambiente": dict(i.ambiente),
                          "sha256_entradas": dict(i.sha256_entradas), "n_unidades": i.n_unidades},
        "adendos": [], "desvios": "nenhum",
        "populacao": {"pacientes": d.populacao.pacientes, "nodulos": d.populacao.nodulos,
                      "classe_0": d.populacao.classe_0, "classe_1": d.populacao.classe_1,
                      "faixa": dict(d.populacao.faixa)},
        "estados": estados, "falhas_tecnicas": 0,
        "estimativas": estimativas, "contrastes": contrastes,
        "hiperparametros": {"frequencia": {m: dict(sorted(f.items())) for m, f in sorted(frequencia.items())},
                            "dobras_na_borda_da_grade": dict(sorted(borda.items()))},
        "avisos": [a._asdict() | {"unidades_com_convergencia": list(a.unidades_com_convergencia)} for a in c.avisos],
        "metricas_descritivas": [s._asdict() for s in c.resumo_descritivas],
        "selecao": {"entrada": dict(d.selecao_entrada), "resultado": d.selecao.como_dict()},
        "bootstrap": {"n_replicas": b.n_replicas, "semente": b.semente,
                      "descartes": [x._asdict() for x in b.descartes]},
        "limites": {"o_que_o_intervalo_nao_captura": [list(x) for x in LIMITES_DO_INTERVALO],
                    "bonferroni": list(LIMITES_DE_BONFERRONI), "otimismo_por_selecao": sl.DECLARACAO_OTIMISMO,
                    "teste_e_s4": DECLARACAO_TESTE},
        "figuras": list(d.figuras),
    }


# --------------------------------------------------------------------------- #
# resumo_benchmark.md
# --------------------------------------------------------------------------- #


def _f(v: Optional[float], casas: int = 4) -> str:
    return "—" if v is None else f"{v:.{casas}f}"


def _ic(baixo: Optional[float], alto: Optional[float], motivo: str = "") -> str:
    return f"não relatado ({motivo})" if baixo is None else f"[{baixo:.4f}; {alto:.4f}]"


def resumo_md(d: DadosRelatorio) -> bytes:
    """Relatório científico: os 15 itens da P9, Seção 12, em Markdown (LF)."""
    c, b, i, p = d.consolidacao, d.bootstrap, d.identificacao, d.populacao
    ic_theta = {(x.modelo, x.desfecho): x for x in b.intervalos_theta}
    ic_ct = {(x.a, x.b, x.desfecho): x for x in b.intervalos_contraste}
    L: List[str] = []
    add = L.append
    add("# Benchmark de classificadores da Sprint 6 — resumo")
    add("")
    add("Relatório gerado pelo pipeline `src/benchmark_s6` (pré-registro P9). Texto descritivo e condicional: "
        "não afirma validação clínica nem desempenho em teste.")
    add("")
    add("## 1. Identificação")
    add("")
    add(f"- Modo: `{i.modo}`; dados oficiais: {'sim' if i.oficial else 'não'}; unidades: {i.n_unidades}.")
    add(f"- Commit da execução: `{i.head}` (branch `{i.branch}`).")
    add(f"- SHA-256 da configuração canônica: `{i.sha256_configuracao}`.")
    for nome, sha in sorted(i.sha256_entradas.items()):
        add(f"- SHA-256 da entrada `{nome}`: `{sha}`.")
    add(f"- Ambiente: {', '.join(f'{k} {v}' for k, v in sorted(i.ambiente.items()))}.")
    add("- SHA-256 das saídas (LF) dos demais artefatos; o do próprio resumo e o do manifesto ficam em "
        "`manifesto_benchmark.json`:")
    for nome, sha in sorted(d.hashes_de_artefatos.items()):
        add(f"  - `{nome}`: `{sha}`")
    add("")
    add("## 2. Desvios em relação ao pré-registro")
    add("")
    add("Adendos: nenhum. Desvios: nenhum.")
    add("")
    add("## 3. População e denominadores")
    add("")
    add(f"Base: {p.pacientes} pacientes e {p.nodulos} nódulos ({p.classe_0} benignos e {p.classe_1} malignos). "
        f"Faixa do desfecho F (12 ≤ D < 30 mm): {p.faixa['nodulos']} nódulos, {p.faixa['pacientes']} pacientes, "
        f"{p.faixa['classe_1']} malignos e {p.faixa['classe_0']} benignos.")
    add("")
    add("| Desfecho | Dobras | DD | Repetições | RI | Repetições válidas | NE | RD | IQ |")
    add("|---|--:|--:|--:|--:|--:|:-:|--:|:-:|")
    desc = {x.desfecho: x for x in b.descartes}
    for e in c.estados:
        add(f"| {e.desfecho} | {e.n_dobras} | {e.dobras_degeneradas} | {e.n_repeticoes} | {e.repeticoes_invalidas} | "
            f"{e.repeticoes_validas} | {'sim' if e.nao_estimavel else 'não'} | {desc[e.desfecho].descartadas} | "
            f"{'sim' if desc[e.desfecho].iq else 'não'} |")
    add("")
    add("Os denominadores por dobra estão em `denominadores.csv`. Falhas técnicas pendentes: 0.")
    add("")
    add("## 4. AUC por modelo")
    add("")
    add("θ̂ é a média, entre as repetições válidas, da média das AUC das dobras válidas. O intervalo é o de 95% "
        "descritivo do bootstrap por paciente, sem correção de multiplicidade.")
    add("")
    add("| Modelo | Desfecho | θ̂ | Intervalo descritivo de 95% | Rótulo |")
    add("|---|---|--:|---|---|")
    for e in c.estimativas:
        x = ic_theta[(e.modelo, e.desfecho)]
        rotulo = "otimista por seleção" if e.modelo == d.selecao.algoritmo else ""
        theta = _f(e.theta) if e.estimavel else "não estimável"
        add(f"| {NOME_MODELO[e.modelo]} | {e.desfecho} | {theta} | {_ic(x.limite_baixo, x.limite_alto, x.motivo)} | "
            f"{rotulo} |")
    add("")
    add("## 5. Contrastes da família principal")
    add("")
    add("Diferença estimada contra a LR com ajuste. Dois intervalos: Bonferroni (seis contrastes, percentis "
        f"{b.percentis_familia[0]:.5f} e {b.percentis_familia[1]:.5f}) e 95% sem correção.")
    add("")
    add("| Contraste | Desfecho | Diferença estimada | Bonferroni | 95% sem correção |")
    add("|---|---|--:|---|---|")
    for x in (y for y in c.contrastes if y.grupo == co.GRUPO_FAMILIA):
        i2 = ic_ct[(x.a, x.b, x.desfecho)]
        add(f"| {NOME_MODELO[x.a]} − {NOME_MODELO[x.b]} | {x.desfecho} | {_f(x.delta)} | "
            f"{_ic(i2.bonferroni_baixo, i2.bonferroni_alto, i2.motivo)} | "
            f"{_ic(i2.ic95_baixo, i2.ic95_alto, i2.motivo)} |")
    add("")
    add("## 6. Sensibilidade à partição")
    add("")
    add("Diferença estimada de cada repetição: média, mínimo, máximo e número de repetições com diferença positiva.")
    add("")
    add("| Contraste | Desfecho | Média | Mínimo | Máximo | Repetições com Δ > 0 |")
    add("|---|---|--:|--:|--:|--:|")
    for x in c.contrastes:
        add(f"| {NOME_MODELO[x.a]} − {NOME_MODELO[x.b]} | {x.desfecho} | {_f(x.delta)} | {_f(x.minimo)} | "
            f"{_f(x.maximo)} | {x.n_repeticoes_positivas} de {x.n_repeticoes_validas} |")
    add("")
    add("## 7. LR com ajuste contra LR fixa")
    add("")
    for dsf in co.DESFECHOS:
        x = c.contraste("lr_tuning", "lr_fixa", dsf)
        i2 = ic_ct[("lr_tuning", "lr_fixa", dsf)]
        add(f"- Desfecho {dsf}: diferença estimada {_f(x.delta)}, intervalo descritivo de 95% "
            f"{_ic(i2.ic95_baixo, i2.ic95_alto, i2.motivo)} (contraste descritivo, fora da família).")
    add("")
    add("## 8. Hiperparâmetros escolhidos e bordas da grade")
    add("")
    add("Frequência das configurações escolhidas nas 50 dobras (detalhe em `hiperparametros_escolhidos.csv`).")
    add("")
    add("| Modelo | Dobras com escolha na borda da grade | Configuração mais frequente (n) |")
    add("|---|--:|---|")
    por_modelo: Dict[str, Dict[str, int]] = {}
    borda: Dict[str, int] = {}
    for e in c.escolhas:
        por_modelo.setdefault(e.modelo, {})
        chave = _configuracao_texto(e.configuracao) or "(sem ajuste)"
        por_modelo[e.modelo][chave] = por_modelo[e.modelo].get(chave, 0) + 1
        borda[e.modelo] = borda.get(e.modelo, 0) + (1 if e.na_borda else 0)
    for m in modelos.codigos():
        top = sorted(por_modelo[m].items(), key=lambda kv: (-kv[1], kv[0]))[0]
        add(f"| {NOME_MODELO[m]} | {borda.get(m, 0)} | {top[0]} ({top[1]}) |")
    add("")
    add("## 9. Métricas descritivas (somente desfecho G)")
    add("")
    add("Média, mínimo e máximo entre as dobras. Não entram em contrastes nem na seleção. O SVM não tem métricas de "
        "probabilidade: só sensibilidade e especificidade pela regra nativa (`decision_function` ≥ 0), "
        "que não é comparável.")
    add("")
    add("| Modelo | Métrica | Dobras | Média | Mínimo | Máximo | Observação |")
    add("|---|---|--:|--:|--:|--:|---|")
    for s in c.resumo_descritivas:
        add(f"| {NOME_MODELO[s.modelo]} | {s.metrica} | {s.n_dobras} | {_f(s.media)} | {_f(s.minimo)} | "
            f"{_f(s.maximo)} | {s.rotulo} |")
    add("")
    add("## 10. Regra de seleção")
    add("")
    add("Padrão: LR com ajuste. Critérios: C1 (diferença estimada em G ≥ 0,02), C2 (diferença positiva em pelo menos "
        "80% das repetições válidas), C3 (F estimável e diferença em F ≥ 0) e C4 (sem falha técnica pendente, G "
        "estimável e sem aviso de não convergência). O intervalo e a taxa de descarte não são critério.")
    add("")
    add("| Desafiante | C1 | C2 | C3 | C4 | Elegível |")
    add("|---|:-:|:-:|:-:|:-:|:-:|")
    for a in d.selecao.avaliacoes:
        marcas = {x.codigo: ("sim" if x.cumprido else "não") for x in a.criterios}
        add(f"| {NOME_MODELO[a.modelo]} | {marcas['C1']} | {marcas['C2']} | {marcas['C3']} | {marcas['C4']} | "
            f"{'sim' if a.elegivel else 'não'} |")
    add("")
    add(f"**Algoritmo recomendado: {NOME_MODELO[d.selecao.algoritmo]}.**")
    add("")
    for j in d.selecao.justificativa:
        add(f"- {j}")
    add("")
    add("## 11. Otimismo da seleção")
    add("")
    add(sl.DECLARACAO_OTIMISMO)
    add("")
    add("## 12. O que o intervalo não captura e limites do ajuste de Bonferroni")
    add("")
    add("| Não captura | Consequência |")
    add("|---|---|")
    for a, cons in LIMITES_DO_INTERVALO:
        add(f"| {a} | {cons} |")
    add("")
    for t in LIMITES_DE_BONFERRONI:
        add(f"- {t}")
    add("")
    add("## 13. Falhas técnicas e avisos")
    add("")
    add("Falhas técnicas pendentes: 0.")
    add("")
    add("| Modelo | Unidades | Unidades com aviso | Avisos de convergência |")
    add("|---|--:|--:|--:|")
    for a in c.avisos:
        add(f"| {NOME_MODELO[a.modelo]} | {a.n_unidades} | {a.unidades_com_aviso} | {a.avisos_convergencia} |")
    add("")
    add("## 14. Teste e independência em relação à S4")
    add("")
    add(DECLARACAO_TESTE)
    add("")
    add("## 15. Figuras")
    add("")
    for nome in d.figuras:
        add(f"- `{nome}`")
    add("")
    return "\n".join(L).encode("utf-8")


# --------------------------------------------------------------------------- #
# relatorio_execucao.json
# --------------------------------------------------------------------------- #


def relatorio_execucao(*, modo: str, processos: int, duracoes: Mapping[str, float], reaproveitadas: Sequence[str],
                       executadas: Sequence[str], falhas: Sequence[Mapping[str, str]], avisos: Mapping[str, int],
                       duracao_total_s: float, extras: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Tempos por unidade e por modelo, processos, falhas, avisos e retomadas. SEM caminhos e sem mensagens.

    ``duracoes`` mapeia o rótulo da unidade (``r00-k0-svm``) ao tempo de relógio em segundos; ``falhas`` traz só
    ``unidade``, ``chave_saida`` e ``categoria`` (a mensagem fica no arquivo local de falha).
    """
    por_modelo: Dict[str, List[float]] = {}
    for rotulo, s in duracoes.items():
        por_modelo.setdefault(rotulo.rsplit("-", 1)[1], []).append(float(s))
    return {
        "versao_esquema": VERSAO_ESQUEMA, "modo": modo, "processos": processos,
        "unidades": {"executadas": len(executadas), "reaproveitadas": len(reaproveitadas),
                     "retomada": bool(reaproveitadas), "falhas": len(falhas)},
        "duracao_total_s": float(duracao_total_s),
        "duracao_por_modelo": {m: {"unidades": len(v), "soma_s": float(sum(v)), "media_s": float(np.mean(v)),
                                   "maxima_s": float(max(v))} for m, v in sorted(por_modelo.items())},
        "duracao_por_unidade": {k: float(v) for k, v in sorted(duracoes.items())},
        "falhas": [dict(f) for f in falhas], "avisos_por_modelo": dict(sorted(avisos.items())),
        **(dict(extras) if extras else {}),
    }


__all__ = [
    "DECLARACAO_TESTE", "DadosRelatorio", "Identificacao", "LIMITES_DE_BONFERRONI", "LIMITES_DO_INTERVALO",
    "NOME_MODELO", "Populacao", "VERSAO_ESQUEMA", "bootstrap_resumo", "csv_bytes", "relatorio_benchmark",
    "relatorio_execucao", "resumo_md", "tabelas",
]
