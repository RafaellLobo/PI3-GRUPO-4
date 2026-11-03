"""Regra de seleção do algoritmo (P9, Seção 10).

**A regra é recalculável a partir de um JSON.** ``selecionar`` recebe só um
mapeamento de tipos JSON (a ``entrada``, gravada em ``relatorio_benchmark.json``) e
devolve a decisão; nenhum objeto do pipeline participa. ``montar_entrada`` extrai
essa entrada da consolidação. Assim, qualquer pessoa refaz a seleção lendo o
relatório versionado. Os limiares vêm da configuração canônica; a entrada
repete os que foram usados, e divergência é recusada (a regra não muda por
dentro de um relatório).

**Padrão.** A LR com ajuste (``lr_tuning``). Um desafiante (RF, XGB ou SVM) só é
elegível se cumprir os quatro critérios:

* C1, magnitude: ``Δ̂(a, G) ≥ δ`` (δ = 0,02, critério operacional sem
  interpretação clínica);
* C2, consistência: ``Δ(a, G) > 0`` em pelo menos 80% das repetições válidas;
* C3, sem piora na faixa: o desfecho F é estimável e ``Δ̂(a, F) ≥ 0``;
* C4, integridade técnica: nenhuma falha técnica pendente nas 250 unidades,
  G estimável e nenhum aviso de não convergência nas unidades do desafiante.

Com mais de um elegível, vence o de maior ``Δ̂(G)``; se a diferença entre os
DOIS primeiros for menor que ``margem_desempate`` (0,005, estritamente), vence o
mais simples, na ordem LR < SVM < RF < XGB. Sem elegível, fica a LR com ajuste.

O IC do bootstrap e a taxa de descarte (IQ) NÃO são critério. Os limiares são
comparados literalmente (``≥``), sem tolerância numérica.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import config as cfg
from . import consolidacao as co

VERSAO_ESQUEMA = 1
CRITERIOS = ("C1", "C2", "C3", "C4")

DECLARACAO_OTIMISMO = (
    "O algoritmo escolhido sai dos mesmos resultados de validação cruzada usados para estimá-lo; por isso a sua "
    "estimativa de AUC é otimista por seleção, e nenhuma validação cruzada externa à seleção nem conjunto "
    "independente a corrige. A escolha vale como recomendação para estas três características nesta base, sem "
    "validação no teste ou externa. Uma correção formal do viés não foi aplicada, e isso fica registrado como "
    "limitação.")


class EntradaInvalida(ValueError):
    """Entrada de seleção malformada ou com parâmetros diferentes da configuração. Chave de saída: integridade."""

    chave_saida = "integridade"


def parametros() -> Dict[str, Any]:
    """Parâmetros da regra, da configuração canônica."""
    s = cfg.CONFIGURACAO.selecao
    return {"padrao": s.padrao, "delta": s.delta, "fracao_min_repeticoes_positivas": s.fracao_min_repeticoes_positivas,
            "delta_minimo_faixa": s.delta_minimo_faixa, "margem_desempate": s.margem_desempate,
            "ordem_simplicidade": list(s.ordem_simplicidade)}


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #


def montar_entrada(c: co.Consolidacao, falhas_tecnicas_pendentes: int, unidades_esperadas: int,
                   unidades_concluidas: int) -> Dict[str, Any]:
    """Entrada da regra a partir da consolidação (tipos JSON)."""
    espec = cfg.CONFIGURACAO.contrastes
    avisos = {a.modelo: a for a in c.avisos}
    desafiantes: Dict[str, Any] = {}
    for a in espec.desafiantes:
        contrastes = {}
        for d in co.DESFECHOS:
            ct = c.contraste(a, espec.comparador, d)
            contrastes[d] = {"estimavel": ct.estimavel, "delta": ct.delta,
                             "n_repeticoes_validas": ct.n_repeticoes_validas,
                             "n_repeticoes_positivas": ct.n_repeticoes_positivas,
                             "fracao_repeticoes_positivas": ct.fracao_repeticoes_positivas}
        desafiantes[a] = {"avisos_convergencia": avisos[a].avisos_convergencia, "contrastes": contrastes}
    return {"versao_esquema": VERSAO_ESQUEMA, "parametros": parametros(),
            "unidades": {"esperadas": unidades_esperadas, "concluidas": unidades_concluidas,
                         "falhas_tecnicas_pendentes": falhas_tecnicas_pendentes},
            "desfechos_estimaveis": {e.desfecho: not e.nao_estimavel for e in c.estados},
            "desafiantes": desafiantes}


def _exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise EntradaInvalida(mensagem)


def _numero(v: Any, onde: str, opcional: bool = False) -> Optional[float]:
    if v is None and opcional:
        return None
    _exigir(type(v) in (int, float) and v == v and abs(v) != float("inf"), f"{onde}: número finito esperado")
    return float(v)


def _validar(entrada: Any) -> None:
    _exigir(isinstance(entrada, Mapping), "entrada deve ser um mapeamento")
    _exigir(set(entrada) == {"versao_esquema", "parametros", "unidades", "desfechos_estimaveis", "desafiantes"},
            "chaves da entrada inválidas")
    _exigir(entrada["versao_esquema"] == VERSAO_ESQUEMA, "versao_esquema desconhecida")
    _exigir(entrada["parametros"] == parametros(),
            "os parâmetros da entrada diferem da configuração canônica; a regra não muda por dentro de um relatório")
    u = entrada["unidades"]
    _exigir(isinstance(u, Mapping) and set(u) == {"esperadas", "concluidas", "falhas_tecnicas_pendentes"}
            and all(type(v) is int and v >= 0 for v in u.values()), "unidades inválidas")
    _exigir(isinstance(entrada["desfechos_estimaveis"], Mapping)
            and set(entrada["desfechos_estimaveis"]) == set(co.DESFECHOS)
            and all(type(v) is bool for v in entrada["desfechos_estimaveis"].values()),
            "desfechos_estimaveis inválidos")
    desafiantes = entrada["desafiantes"]
    _exigir(isinstance(desafiantes, Mapping) and set(desafiantes) == set(cfg.CONFIGURACAO.contrastes.desafiantes),
            "desafiantes diferentes da configuração")
    for a, dados in desafiantes.items():
        _exigir(isinstance(dados, Mapping) and set(dados) == {"avisos_convergencia", "contrastes"},
                f"{a}: chaves inválidas")
        _exigir(type(dados["avisos_convergencia"]) is int and dados["avisos_convergencia"] >= 0,
                f"{a}: avisos_convergencia inválido")
        _exigir(isinstance(dados["contrastes"], Mapping) and set(dados["contrastes"]) == set(co.DESFECHOS),
                f"{a}: contrastes devem cobrir G e F")
        for d, ct in dados["contrastes"].items():
            onde = f"{a}/{d}"
            _exigir(isinstance(ct, Mapping) and set(ct) == {"estimavel", "delta", "n_repeticoes_validas",
                                                            "n_repeticoes_positivas", "fracao_repeticoes_positivas"},
                    f"{onde}: chaves inválidas")
            _exigir(type(ct["estimavel"]) is bool, f"{onde}: estimavel inválido")
            _numero(ct["delta"], f"{onde}/delta", opcional=not ct["estimavel"])
            _numero(ct["fracao_repeticoes_positivas"], f"{onde}/fracao", opcional=not ct["estimavel"])
            _exigir(type(ct["n_repeticoes_validas"]) is int and type(ct["n_repeticoes_positivas"]) is int
                    and 0 <= ct["n_repeticoes_positivas"] <= ct["n_repeticoes_validas"], f"{onde}: contagens inválidas")
            _exigir(ct["estimavel"] == (ct["delta"] is not None) and
                    ct["estimavel"] == entrada["desfechos_estimaveis"][d], f"{onde}: estimabilidade incoerente")


# --------------------------------------------------------------------------- #
# Decisão
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Criterio:
    """Um critério para um desafiante: o valor observado, o limiar e se foi cumprido."""

    codigo: str
    descricao: str
    cumprido: bool
    valor: Any
    limiar: Any


@dataclass(frozen=True)
class AvaliacaoDesafiante:
    modelo: str
    criterios: Tuple[Criterio, ...]

    @property
    def elegivel(self) -> bool:
        return all(c.cumprido for c in self.criterios)


@dataclass(frozen=True)
class ResultadoSelecao:
    algoritmo: str
    padrao: str
    elegiveis: Tuple[str, ...]
    avaliacoes: Tuple[AvaliacaoDesafiante, ...]
    desempate_aplicado: bool
    diferenca_entre_os_dois_primeiros: Optional[float]
    justificativa: Tuple[str, ...]
    otimismo_por_selecao: str

    @property
    def desafiante_escolhido(self) -> bool:
        return self.algoritmo != self.padrao

    def como_dict(self) -> Dict[str, Any]:
        return {
            "versao_esquema": VERSAO_ESQUEMA, "algoritmo_recomendado": self.algoritmo, "padrao": self.padrao,
            "desafiante_escolhido": self.desafiante_escolhido, "elegiveis": list(self.elegiveis),
            "desempate_aplicado": self.desempate_aplicado,
            "diferenca_entre_os_dois_primeiros": self.diferenca_entre_os_dois_primeiros,
            "criterios": {a.modelo: {c.codigo: {"descricao": c.descricao, "cumprido": c.cumprido, "valor": c.valor,
                                                "limiar": c.limiar} for c in a.criterios} for a in self.avaliacoes},
            "elegibilidade": {a.modelo: a.elegivel for a in self.avaliacoes},
            "justificativa": list(self.justificativa), "otimismo_por_selecao": self.otimismo_por_selecao,
        }


def _avaliar(modelo: str, dados: Mapping[str, Any], entrada: Mapping[str, Any]) -> AvaliacaoDesafiante:
    p = entrada["parametros"]
    g, f = dados["contrastes"]["G"], dados["contrastes"]["F"]
    unidades = entrada["unidades"]
    c1 = g["estimavel"] and g["delta"] >= p["delta"]
    c2 = g["estimavel"] and g["fracao_repeticoes_positivas"] >= p["fracao_min_repeticoes_positivas"]
    c3 = f["estimavel"] and f["delta"] >= p["delta_minimo_faixa"]
    sem_ft = unidades["falhas_tecnicas_pendentes"] == 0 and unidades["concluidas"] == unidades["esperadas"]
    c4 = sem_ft and entrada["desfechos_estimaveis"]["G"] and dados["avisos_convergencia"] == 0
    return AvaliacaoDesafiante(modelo, (
        Criterio("C1", "magnitude: Δ estimado em G não menor que δ", bool(c1), g["delta"], p["delta"]),
        Criterio("C2", "consistência: fração das repetições válidas com Δ > 0 em G", bool(c2),
                 g["fracao_repeticoes_positivas"], p["fracao_min_repeticoes_positivas"]),
        Criterio("C3", "sem piora na faixa: F estimável e Δ estimado em F não menor que o mínimo", bool(c3),
                 f["delta"], p["delta_minimo_faixa"]),
        Criterio("C4", "integridade técnica: sem falha técnica pendente, G estimável e sem aviso de não convergência",
                 bool(c4), {"falhas_tecnicas_pendentes": unidades["falhas_tecnicas_pendentes"],
                            "g_estimavel": entrada["desfechos_estimaveis"]["G"],
                            "avisos_convergencia": dados["avisos_convergencia"]}, "nenhum")))


def selecionar(entrada: Mapping[str, Any]) -> ResultadoSelecao:
    """Aplica a regra da P9, Seção 10, à entrada (tipos JSON). Não consulta nada além da entrada e da configuração."""
    _validar(entrada)
    entrada = copy.deepcopy(dict(entrada))
    p = entrada["parametros"]
    ordem = {m: i for i, m in enumerate(p["ordem_simplicidade"])}
    avaliacoes = tuple(_avaliar(m, entrada["desafiantes"][m], entrada) for m in sorted(entrada["desafiantes"]))
    elegiveis = sorted((a.modelo for a in avaliacoes if a.elegivel),
                       key=lambda m: (-entrada["desafiantes"][m]["contrastes"]["G"]["delta"], ordem[m]))
    desempate, diferenca = False, None
    justificativa: List[str] = []
    escolhido = p["padrao"]
    for a in avaliacoes:
        falhou = [c.codigo for c in a.criterios if not c.cumprido]
        justificativa.append(f"{a.modelo}: elegível (C1 a C4 cumpridos)." if not falhou else
                             f"{a.modelo}: não elegível; critério(s) não cumprido(s): {', '.join(falhou)}.")
    if not elegiveis:
        justificativa.append(f"Nenhum desafiante cumpriu C1 a C4; fica o padrão da regra, {p['padrao']}.")
    elif len(elegiveis) == 1:
        escolhido = elegiveis[0]
        justificativa.append(f"Único desafiante elegível: {escolhido}.")
    else:
        primeiro, segundo = elegiveis[0], elegiveis[1]
        d1 = entrada["desafiantes"][primeiro]["contrastes"]["G"]["delta"]
        d2 = entrada["desafiantes"][segundo]["contrastes"]["G"]["delta"]
        diferenca = d1 - d2
        if diferenca < p["margem_desempate"]:
            desempate = True
            escolhido = min((primeiro, segundo), key=lambda m: ordem[m])
            justificativa.append(
                f"Elegíveis {', '.join(elegiveis)}. Os dois primeiros ({primeiro}, {segundo}) diferem em Δ(G) por "
                f"{diferenca:.6f}, menos que {p['margem_desempate']}: vence o mais simples, {escolhido}.")
        else:
            escolhido = primeiro
            justificativa.append(f"Elegíveis {', '.join(elegiveis)}. Vence o de maior Δ(G), {primeiro}; a diferença "
                                 f"para o segundo é {diferenca:.6f}, não menor que {p['margem_desempate']}.")
    return ResultadoSelecao(escolhido, p["padrao"], tuple(elegiveis), avaliacoes, desempate, diferenca,
                            tuple(justificativa), DECLARACAO_OTIMISMO)


__all__ = [
    "AvaliacaoDesafiante", "CRITERIOS", "Criterio", "DECLARACAO_OTIMISMO", "EntradaInvalida", "ResultadoSelecao",
    "VERSAO_ESQUEMA", "montar_entrada", "parametros", "selecionar",
]
