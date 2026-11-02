"""Plano de dobras do benchmark S6 (P9, Seção 4; sementes da decisão D12).

Reimplementa, presa à configuração canônica do benchmark, a validação de
dobras de ``src/incremental/folds.py`` (``_dividir``, ``_validar_nivel`` e
``plano_repeticao``), que está presa à configuração da S4 (P9, Seção 14.2).
Da S4 só vêm nomes públicos, pelo adaptador: ``Amostra`` (ids, rótulos e
grupos na ordem canônica), ``fingerprint_folds`` e ``sha256_canonico``.

Dois níveis, ambos com ``StratifiedGroupKFold(shuffle=True)``, estratificado
pelo alvo e agrupado por ``patient_id``:

* EXTERNO — ``n_dobras_externas`` (5) dobras sobre a amostra inteira, semente
  ``42 + r`` na repetição r;
* INTERNO — ``n_dobras_internas`` (5) dobras sobre o TREINO da dobra externa
  k, semente ``1000 + 10r + k``.

Repetições, dobras e índices começam em 0. O modo ``implementacao`` gera só
r = 0; o modo ``definitivo``, r = 0…9 (50 dobras externas).

Toda divisão é validada (P9, Seção 4):

* treino e avaliação não vazios, disjuntos e cobrindo exatamente as linhas
  do nível;
* nenhum paciente nos dois lados;
* as duas classes nos dois lados;
* cada linha avaliada exatamente uma vez por repetição (externo) e por
  dobra externa (interno);
* nenhuma linha nem paciente da avaliação externa em qualquer divisão
  interna.

Avisos do divisor são tratados como erro. Uma violação levanta
``DobrasInvalidas`` (falha técnica, P9, Seção 6).

O plano é gerado UMA vez e não conhece modelos: ``dobra_para_modelo``
devolve, para qualquer modelo da configuração, o MESMO objeto da dobra, com
o mesmo fingerprint. Os índices são POSIÇÕES na amostra completa, em ordem
crescente e somente leitura. Só fingerprints (``fingerprint_folds``) e
contagens agregadas saem do módulo por ``PlanoDobras.resumo``; nenhum
identificador. O módulo não lê arquivos, não treina e não conhece
características.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from . import adaptadores_s4 as s4
from . import config as cfg
from .dados import MODO_DEFINITIVO, MODO_IMPLEMENTACAO, DadosBenchmark

NIVEL_EXTERNO = "externo"
NIVEL_INTERNO = "interno"
MODOS = (MODO_IMPLEMENTACAO, MODO_DEFINITIVO)


class DobrasInvalidas(ValueError):
    """Entrada incompatível ou divisão que viola a P9, Seção 4 (falha técnica)."""

    chave_saida: str = "falha_tecnica"


class PlanoDivergente(DobrasInvalidas):
    """Plano recebido difere do regerado a partir da amostra e da configuração."""

    chave_saida = "integridade"


def _somente_leitura(valores: np.ndarray) -> np.ndarray:
    copia = np.array(valores, dtype=np.int64, copy=True)
    copia.setflags(write=False)
    return copia


def repeticoes_do_modo(modo: str) -> Tuple[int, ...]:
    """r = 0 no modo ``implementacao``; r = 0…n_repeticoes − 1 no ``definitivo``."""
    if modo == MODO_IMPLEMENTACAO:
        return (0,)
    if modo == MODO_DEFINITIVO:
        return tuple(range(cfg.CONFIGURACAO.dobras.n_repeticoes))
    raise DobrasInvalidas(f"modo inválido: {modo!r}; use {MODOS}")


# --------------------------------------------------------------------------- #
# Estruturas
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class Divisao:
    """Uma divisão treino/avaliação, com posições na amostra completa."""

    nivel: str
    repeticao: int
    dobra_externa: int
    dobra_interna: Optional[int]
    semente: int
    treino: np.ndarray = field(repr=False)
    avaliacao: np.ndarray = field(repr=False)

    @property
    def rotulo(self) -> str:
        base = f"r{self.repeticao:02d}-k{self.dobra_externa}"
        return base if self.dobra_interna is None else f"{base}-i{self.dobra_interna}"


@dataclass(frozen=True, eq=False)
class DobraExterna:
    """Dobra externa ``(r, k)`` com as suas dobras internas: a unidade de dados de ``(r, k, modelo)``."""

    sha256_amostra: str
    externa: Divisao
    internas: Tuple[Divisao, ...]
    #: ``fingerprint_folds`` da divisão externa e de todas as internas.
    fingerprint: str

    @property
    def repeticao(self) -> int:
        return self.externa.repeticao

    @property
    def dobra(self) -> int:
        return self.externa.dobra_externa

    def cv_interno(self) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """Divisões internas em posições RELATIVAS ao treino externo (``cv`` do ``GridSearchCV``).

        Para usar com ``X[self.externa.treino]``: cada par aponta linhas dessa
        submatriz, na ordem crescente de ``externa.treino``.
        """
        treino_externo = self.externa.treino
        pares = []
        for interna in self.internas:
            locais = []
            for lado in (interna.treino, interna.avaliacao):
                posicoes = np.searchsorted(treino_externo, lado)
                if np.any(posicoes >= len(treino_externo)) or not np.array_equal(treino_externo[posicoes], lado):
                    raise DobrasInvalidas(f"{interna.rotulo}: linha fora do treino externo")
                locais.append(_somente_leitura(posicoes))
            pares.append((locais[0], locais[1]))
        return tuple(pares)


@dataclass(frozen=True, eq=False)
class PlanoDobras:
    """Plano completo de um modo, validado. Igual para todos os modelos."""

    modo: str
    repeticoes: Tuple[int, ...]
    n_linhas: int
    sha256_amostra: str
    sha256_configuracao: str
    dobras: Mapping[Tuple[int, int], DobraExterna] = field(repr=False)
    #: ``sha256_canonico`` do modo, das sementes e dos fingerprints de todas as dobras.
    fingerprint: str

    def dobra(self, r: int, k: int) -> DobraExterna:
        if type(r) is not int or type(k) is not int or (r, k) not in self.dobras:
            raise DobrasInvalidas(f"dobra inexistente no plano {self.modo}: r={r!r}, k={k!r}")
        return self.dobras[(r, k)]

    def externas(self, r: int) -> Tuple[Divisao, ...]:
        if type(r) is not int or r not in self.repeticoes:
            raise DobrasInvalidas(f"repetição inexistente no plano {self.modo}: {r!r}")
        return tuple(d.externa for (rr, _), d in self.dobras.items() if rr == r)

    def resumo(self, amostra: s4.Amostra) -> Dict[str, Any]:
        """Conteúdo de ``plano_folds.json`` (P9, Seção 14.5): fingerprints e contagens agregadas.

        Nenhum identificador nem hash da amostra: só sementes, fingerprints e
        contagens de nódulos, pacientes e classes por lado de cada divisão.
        """
        _exigir_amostra_do_plano(amostra, self)

        def contagem(posicoes: np.ndarray) -> Dict[str, int]:
            y = amostra.y[posicoes]
            return {"nodulos": int(len(posicoes)), "pacientes": len(set(amostra.grupos[posicoes].tolist())),
                    "classe_0": int(np.sum(y == 0)), "classe_1": int(np.sum(y == 1))}

        def divisao(d: Divisao) -> Dict[str, Any]:
            return {"rotulo": d.rotulo, "semente": d.semente, "fingerprint": _fingerprint_divisao(amostra, d),
                    "treino": contagem(d.treino), "avaliacao": contagem(d.avaliacao)}

        espec = cfg.CONFIGURACAO.dobras
        return {
            "modo": self.modo,
            "divisor": espec.divisor,
            "estratificacao": espec.estratificacao,
            "grupo": espec.grupo,
            "n_dobras_externas": espec.n_dobras_externas,
            "n_dobras_internas": espec.n_dobras_internas,
            "repeticoes": list(self.repeticoes),
            "sha256_configuracao": self.sha256_configuracao,
            "fingerprint_plano": self.fingerprint,
            "dobras": [{"repeticao": r, "dobra_externa": k, "fingerprint_dobra": d.fingerprint,
                        "externa": divisao(d.externa), "internas": [divisao(i) for i in d.internas]}
                       for (r, k), d in self.dobras.items()],
        }


# --------------------------------------------------------------------------- #
# Divisão e validação (espelho de ``_dividir`` e ``_validar_nivel`` da S4)
# --------------------------------------------------------------------------- #


def _exigir_amostra(amostra: Any) -> s4.Amostra:
    if not isinstance(amostra, s4.Amostra):
        raise TypeError("amostra deve ser a Amostra da S4 (DadosBenchmark.amostra())")
    return amostra


def _exigir_amostra_do_plano(amostra: Any, plano: Any) -> None:
    _exigir_amostra(amostra)
    if not isinstance(plano, PlanoDobras):
        raise TypeError("plano deve ser PlanoDobras")
    if plano.sha256_amostra != amostra.sha256 or plano.n_linhas != amostra.n:
        raise DobrasInvalidas("plano de outra amostra")


def _dividir(amostra: s4.Amostra, base: np.ndarray, n_splits: int, semente: int,
             contexto: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """``StratifiedGroupKFold`` sobre as linhas ``base``; devolve posições globais."""
    divisor = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semente)
    marcador = np.zeros((len(base), 1))  # o divisor só usa o número de linhas de X
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            partes = list(divisor.split(marcador, amostra.y[base], amostra.grupos[base]))
    except (ValueError, Warning) as exc:
        raise DobrasInvalidas(f"{contexto}: o divisor recusou a entrada: {exc}") from exc
    return [(_somente_leitura(np.sort(base[tr])), _somente_leitura(np.sort(base[te]))) for tr, te in partes]


def _indices_validos(amostra: s4.Amostra, lado: Any) -> bool:
    return (isinstance(lado, np.ndarray) and lado.ndim == 1 and lado.dtype == np.int64 and len(lado) > 0
            and not lado.flags.writeable and lado[0] >= 0 and lado[-1] < amostra.n
            and bool(np.all(np.diff(lado) > 0)))


def _validar_nivel(amostra: s4.Amostra, base: np.ndarray, partes: List[Tuple[np.ndarray, np.ndarray]],
                   n_esperado: int, contexto: str) -> None:
    if len(partes) != n_esperado:
        raise DobrasInvalidas(f"{contexto}: {len(partes)} dobras; esperadas {n_esperado}")
    conjunto_base = set(base.tolist())
    avaliadas: List[int] = []
    for i, (treino, avaliacao) in enumerate(partes):
        onde = f"{contexto}, dobra {i}"
        for lado in (treino, avaliacao):
            if not _indices_validos(amostra, lado):
                raise DobrasInvalidas(f"{onde}: treino ou avaliação vazio ou com índices inválidos")
        if np.intersect1d(treino, avaliacao).size:
            raise DobrasInvalidas(f"{onde}: linha em treino e avaliação")
        if set(treino.tolist()) | set(avaliacao.tolist()) != conjunto_base or \
                len(treino) + len(avaliacao) != len(base):
            raise DobrasInvalidas(f"{onde}: treino ∪ avaliação difere das linhas do nível")
        if set(amostra.grupos[treino].tolist()) & set(amostra.grupos[avaliacao].tolist()):
            raise DobrasInvalidas(f"{onde}: paciente em treino e avaliação")
        for nome, lado in (("treino", treino), ("avaliação", avaliacao)):
            if set(amostra.y[lado].tolist()) != {0, 1}:
                raise DobrasInvalidas(f"{onde}: {nome} sem as duas classes")
        avaliadas.extend(avaliacao.tolist())
    if sorted(avaliadas) != sorted(base.tolist()):
        raise DobrasInvalidas(f"{contexto}: linhas não avaliadas exatamente uma vez")


def _exigir_sem_vazamento(amostra: s4.Amostra, externa: Divisao, internas: Tuple[Divisao, ...]) -> None:
    """Nenhuma linha nem paciente da avaliação externa em qualquer divisão interna."""
    pacientes_avaliacao = set(amostra.grupos[externa.avaliacao].tolist())
    for interna in internas:
        usadas = np.concatenate([interna.treino, interna.avaliacao])
        if np.isin(usadas, externa.avaliacao).any() or pacientes_avaliacao & set(amostra.grupos[usadas].tolist()):
            raise DobrasInvalidas(f"{interna.rotulo}: usa dados da avaliação externa")


# --------------------------------------------------------------------------- #
# Fingerprints (``checkpoints.fingerprint_folds`` da S4)
# --------------------------------------------------------------------------- #


def _fingerprint_divisao(amostra: s4.Amostra, d: Divisao) -> str:
    return s4.fingerprint_folds({"treino": amostra.ids[d.treino].tolist(),
                                 "avaliacao": amostra.ids[d.avaliacao].tolist()})


def _fingerprint_dobra(amostra: s4.Amostra, externa: Divisao, internas: Tuple[Divisao, ...]) -> str:
    """Divisão externa e TODAS as internas: o resultado de ``(r, k, modelo)`` depende de todas."""
    composicao = {}
    for d in (externa, *internas):
        composicao[f"{d.rotulo}/treino"] = amostra.ids[d.treino].tolist()
        composicao[f"{d.rotulo}/avaliacao"] = amostra.ids[d.avaliacao].tolist()
    return s4.fingerprint_folds(composicao)


def _fingerprint_plano(modo: str, repeticoes: Tuple[int, ...],
                       dobras: Mapping[Tuple[int, int], DobraExterna]) -> str:
    espec = cfg.CONFIGURACAO.dobras
    return s4.sha256_canonico({
        "modo": modo,
        "divisor": espec.divisor,
        "repeticoes": list(repeticoes),
        "n_dobras_externas": espec.n_dobras_externas,
        "n_dobras_internas": espec.n_dobras_internas,
        "sementes_externas": [espec.sementes_externas()[r] for r in repeticoes],
        "sementes_internas": [espec.semente_interna(r, k) for r in repeticoes
                              for k in range(espec.n_dobras_externas)],
        "dobras": {f"r{r:02d}-k{k}": d.fingerprint for (r, k), d in dobras.items()},
    })


# --------------------------------------------------------------------------- #
# Geração e verificação
# --------------------------------------------------------------------------- #


def _gerar_dobra(amostra: s4.Amostra, externa: Divisao) -> DobraExterna:
    espec = cfg.CONFIGURACAO.dobras
    r, k = externa.repeticao, externa.dobra_externa
    semente = espec.semente_interna(r, k)
    contexto = f"interno {externa.rotulo}"
    partes = _dividir(amostra, externa.treino, espec.n_dobras_internas, semente, contexto)
    _validar_nivel(amostra, externa.treino, partes, espec.n_dobras_internas, contexto)
    internas = tuple(Divisao(NIVEL_INTERNO, r, k, i, semente, tr, te) for i, (tr, te) in enumerate(partes))
    _exigir_sem_vazamento(amostra, externa, internas)
    return DobraExterna(amostra.sha256, externa, internas, _fingerprint_dobra(amostra, externa, internas))


def gerar_plano(amostra: s4.Amostra, modo: str) -> PlanoDobras:
    """Plano validado do ``modo``, a partir da ``Amostra`` na ordem canônica.

    Usa sempre a configuração canônica (n de dobras, repetições e sementes).
    Determinístico: a mesma amostra gera o mesmo plano e o mesmo fingerprint.
    """
    _exigir_amostra(amostra)
    repeticoes = repeticoes_do_modo(modo)
    espec = cfg.CONFIGURACAO.dobras
    base = _somente_leitura(np.arange(amostra.n))
    dobras: Dict[Tuple[int, int], DobraExterna] = {}
    for r in repeticoes:
        semente = espec.sementes_externas()[r]
        contexto = f"externo r{r:02d}"
        partes = _dividir(amostra, base, espec.n_dobras_externas, semente, contexto)
        _validar_nivel(amostra, base, partes, espec.n_dobras_externas, contexto)
        for k, (tr, te) in enumerate(partes):
            dobras[(r, k)] = _gerar_dobra(amostra, Divisao(NIVEL_EXTERNO, r, k, None, semente, tr, te))
    dobras_ro = MappingProxyType(dobras)
    return PlanoDobras(modo, repeticoes, amostra.n, amostra.sha256, cfg.sha256_configuracao(cfg.CONFIGURACAO),
                       dobras_ro, _fingerprint_plano(modo, repeticoes, dobras_ro))


def plano_dos_dados(dados: DadosBenchmark) -> PlanoDobras:
    """Plano do modo da carga: r = 0 para ``implementacao``, r = 0…9 para ``definitivo``."""
    if not isinstance(dados, DadosBenchmark):
        raise TypeError("dados deve ser DadosBenchmark")
    return gerar_plano(dados.amostra(), dados.modo)


def validar_dobra(amostra: s4.Amostra, dobra: DobraExterna) -> DobraExterna:
    """Revalida UMA dobra externa recebida, sem o plano.

    Confere, com os mesmos validadores que geraram o plano:

    * amostra: a dobra é desta ``Amostra``;
    * metadados: ``r`` e ``k`` dentro do plano definitivo, níveis, índices e
      sementes (``42 + r`` e ``1000 + 10r + k``) da configuração;
    * divisão externa: índices válidos e somente leitura, treino e avaliação
      disjuntos, nenhum paciente nos dois lados, as duas classes nos dois lados;
    * internas: cinco divisões do treino externo, cada uma válida, cobrindo-o
      com cada linha avaliada uma vez, sem nenhuma linha nem paciente da
      avaliação externa;
    * fingerprint da dobra recalculado.

    A cobertura da REPETIÇÃO (cada linha avaliada uma vez entre as cinco
    dobras externas) só é conferível com o plano: ``validar_plano``.
    Violação: ``DobrasInvalidas``; fingerprint divergente: ``PlanoDivergente``.

    Uma cópia por ``pickle`` volta com os índices graváveis (numpy 1.23) e é
    RECUSADA: não há, por ora, caminho validado para enviar dobras a outro
    processo.
    """
    _exigir_amostra(amostra)
    if not isinstance(dobra, DobraExterna):
        raise TypeError("dobra deve ser DobraExterna")
    if dobra.sha256_amostra != amostra.sha256:
        raise DobrasInvalidas("dobra de outra amostra")
    espec = cfg.CONFIGURACAO.dobras
    ext = dobra.externa
    if not isinstance(ext, Divisao) or not isinstance(dobra.internas, tuple) or \
            not all(isinstance(i, Divisao) for i in dobra.internas):
        raise DobrasInvalidas("dobra com divisões malformadas")
    r, k = ext.repeticao, ext.dobra_externa
    if type(r) is not int or type(k) is not int or not 0 <= r < espec.n_repeticoes \
            or not 0 <= k < espec.n_dobras_externas:
        raise DobrasInvalidas(f"dobra fora do plano: r={r!r}, k={k!r}")
    if (ext.nivel, ext.dobra_interna, ext.semente) != (NIVEL_EXTERNO, None, espec.sementes_externas()[r]):
        raise DobrasInvalidas(f"r{r:02d}-k{k}: metadados ou semente externa inválidos")
    if len(dobra.internas) != espec.n_dobras_internas or any(
            (i.nivel, i.repeticao, i.dobra_externa, i.dobra_interna, i.semente) !=
            (NIVEL_INTERNO, r, k, n, espec.semente_interna(r, k)) for n, i in enumerate(dobra.internas)):
        raise DobrasInvalidas(f"r{r:02d}-k{k}: metadados ou semente interna inválidos")
    onde = f"externo {ext.rotulo}"
    for lado in (ext.treino, ext.avaliacao):
        if not _indices_validos(amostra, lado):
            raise DobrasInvalidas(f"{onde}: treino ou avaliação vazio ou com índices inválidos")
    if np.intersect1d(ext.treino, ext.avaliacao).size:
        raise DobrasInvalidas(f"{onde}: linha em treino e avaliação")
    if set(amostra.grupos[ext.treino].tolist()) & set(amostra.grupos[ext.avaliacao].tolist()):
        raise DobrasInvalidas(f"{onde}: paciente em treino e avaliação")
    for nome, lado in (("treino", ext.treino), ("avaliação", ext.avaliacao)):
        if set(amostra.y[lado].tolist()) != {0, 1}:
            raise DobrasInvalidas(f"{onde}: {nome} sem as duas classes")
    _validar_nivel(amostra, ext.treino, [(i.treino, i.avaliacao) for i in dobra.internas],
                   espec.n_dobras_internas, f"interno {ext.rotulo}")
    _exigir_sem_vazamento(amostra, ext, dobra.internas)
    if dobra.fingerprint != _fingerprint_dobra(amostra, ext, dobra.internas):
        raise PlanoDivergente(f"{ext.rotulo}: fingerprint da dobra não confere")
    return dobra


def validar_plano(amostra: s4.Amostra, plano: PlanoDobras) -> PlanoDobras:
    """Revalida um plano recebido sem regerá-lo: invariantes, sementes e fingerprints.

    Um plano montado ou alterado à mão não passa: cada dobra passa por
    ``validar_dobra``, cada repetição é conferida como um todo, e os
    fingerprints são recalculados.
    """
    _exigir_amostra_do_plano(amostra, plano)
    espec = cfg.CONFIGURACAO.dobras
    if plano.repeticoes != repeticoes_do_modo(plano.modo):
        raise DobrasInvalidas(f"repetições do plano não são as do modo {plano.modo}")
    if plano.sha256_configuracao != cfg.sha256_configuracao(cfg.CONFIGURACAO):
        raise PlanoDivergente("plano gerado com outra configuração")
    esperadas = [(r, k) for r in plano.repeticoes for k in range(espec.n_dobras_externas)]
    if list(plano.dobras) != esperadas:
        raise DobrasInvalidas("dobras do plano fora da ordem ou incompletas")
    base = np.arange(amostra.n, dtype=np.int64)
    for r in plano.repeticoes:
        partes = []
        for k in range(espec.n_dobras_externas):
            d = plano.dobras[(r, k)]
            if not isinstance(d, DobraExterna):
                raise DobrasInvalidas(f"r{r:02d}-k{k}: dobra malformada")
            validar_dobra(amostra, d)
            if (d.externa.repeticao, d.externa.dobra_externa) != (r, k):
                raise DobrasInvalidas(f"r{r:02d}-k{k}: dobra registrada na posição de outra")
            partes.append((d.externa.treino, d.externa.avaliacao))
        _validar_nivel(amostra, base, partes, espec.n_dobras_externas, f"externo r{r:02d}")
    if plano.fingerprint != _fingerprint_plano(plano.modo, plano.repeticoes, plano.dobras):
        raise PlanoDivergente("fingerprint do plano não confere")
    return plano


def verificar_plano(amostra: s4.Amostra, plano: PlanoDobras) -> PlanoDobras:
    """Revalida o plano e exige que seja idêntico ao regerado (modo ``verificar``)."""
    validar_plano(amostra, plano)
    regerado = gerar_plano(amostra, plano.modo)
    if regerado.fingerprint != plano.fingerprint:
        raise PlanoDivergente("plano difere do regerado a partir da amostra e da configuração")
    return plano


def dobra_para_modelo(plano: PlanoDobras, r: int, k: int, modelo: str) -> DobraExterna:
    """Dobra ``(r, k)`` para a unidade ``(r, k, modelo)``: o MESMO objeto para todos os modelos.

    O modelo só é conferido contra a configuração; ele não influencia a
    divisão. Não existe caminho para gerar dobras por modelo.
    """
    if not isinstance(plano, PlanoDobras):
        raise TypeError("plano deve ser PlanoDobras")
    codigos = tuple(m.codigo for m in cfg.CONFIGURACAO.modelos)
    if modelo not in codigos:
        raise DobrasInvalidas(f"modelo fora da configuração: {modelo!r}; use {codigos}")
    return plano.dobra(r, k)


__all__ = [
    "DobraExterna", "DobrasInvalidas", "Divisao", "MODOS", "NIVEL_EXTERNO", "NIVEL_INTERNO",
    "PlanoDivergente", "PlanoDobras", "dobra_para_modelo", "gerar_plano", "plano_dos_dados",
    "repeticoes_do_modo", "validar_dobra", "validar_plano", "verificar_plano",
]
