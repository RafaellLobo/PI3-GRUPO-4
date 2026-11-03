"""Folds agrupados por paciente da CV aninhada do Cartão 4 — Sprint 4.

Três níveis, todos com ``StratifiedGroupKFold(shuffle=True)`` agrupado por
``patient_id`` e com parâmetros lidos de ``config``:

* EXTERNO — ``ESPEC_CV.n_folds_externos`` folds sobre o treino oficial, seed
  ``ESPEC_CV.seeds_repeticoes()[r]`` (42 + r) na repetição r (protocolo,
  Seção 7.2);
* ESTABILIDADE — ``ESPEC_SELECAO.n_folds_estabilidade`` folds dentro do
  TREINO de cada fold externo, seed ``ESPEC_SELECAO.seed_estabilidade``
  (Seção 5.4: idêntica ao procedimento original);
* INTERNO — ``ESPEC_SELECAO.n_folds_interna`` folds dentro do TREINO de cada
  fold de estabilidade, seed ``ESPEC_SELECAO.seed_interna(j + 1)``.

Índices dos folds começam em 0 em todos os níveis, como nos checkpoints. O
fold de estabilidade ``j`` corresponde ao fold ``j + 1`` do procedimento
original (``enumerate(..., start=1)``), daí ``seed_interna(j + 1)`` = 100 + j + 1.

Cada nível divide apenas as linhas do treino do nível acima, nunca as de
avaliação. Os índices devolvidos são sempre POSIÇÕES na amostra completa
(``Amostra``), em ordem crescente e somente leitura.

Toda divisão é validada:

* treino e avaliação disjuntos e não vazios, cobrindo exatamente as linhas
  do nível;
* nenhum paciente nos dois lados;
* as duas classes presentes nos dois lados;
* cada linha avaliada exatamente uma vez no conjunto de folds do nível.

Warnings do splitter são tratados como erro. Entradas incompatíveis — fora
da ordem canônica por ``nodulo_id``, desalinhadas, com IDs repetidos ou
rótulo não binário — são recusadas, nunca reorganizadas.

O módulo não treina nada, não lê dados e não conhece features.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from . import config as cfg
from .checkpoints import fingerprint_folds

NIVEL_EXTERNO = "externo"
NIVEL_ESTABILIDADE = "estabilidade"
NIVEL_INTERNO = "interno"


class FoldsInvalidos(ValueError):
    """Entrada incompatível ou divisão que viola uma invariante dos folds."""


# --------------------------------------------------------------------------- #
# Amostra
# --------------------------------------------------------------------------- #


def _somente_leitura(valores: np.ndarray) -> np.ndarray:
    copia = np.array(valores, copy=True)
    copia.setflags(write=False)
    return copia


@dataclass(frozen=True, eq=False)
class Amostra:
    """Identificadores, rótulos e grupos alinhados, na ordem canônica.

    Recusa: comprimentos diferentes, arrays não unidimensionais,
    identificadores ou grupos vazios ou não textuais, ``nodulo_id`` repetido
    ou fora da ordem crescente (``config.ORDEM_LINHAS``), rótulo que não seja
    inteiro 0/1 e ausência de uma das classes.
    """

    ids: np.ndarray
    y: np.ndarray
    grupos: np.ndarray
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        ids = np.asarray(self.ids, dtype=object)
        grupos = np.asarray(self.grupos, dtype=object)
        y = np.asarray(self.y)
        for nome, arr in (("ids", ids), ("y", y), ("grupos", grupos)):
            if arr.ndim != 1:
                raise FoldsInvalidos(f"{nome} deve ser unidimensional")
        if not len(ids) == len(y) == len(grupos) or len(ids) == 0:
            raise FoldsInvalidos("ids, y e grupos devem ter o mesmo comprimento, não nulo")
        for nome, arr in (("ids", ids), ("grupos", grupos)):
            if not all(type(v) is str and v for v in arr):
                raise FoldsInvalidos(f"{nome}: todo valor deve ser texto não vazio")
        lista = list(ids)
        if len(set(lista)) != len(lista):
            raise FoldsInvalidos("ids repetidos")
        if lista != sorted(lista):
            raise FoldsInvalidos(f"ids fora da ordem canônica por {cfg.ORDEM_LINHAS[0]}; nada é reordenado")
        if y.dtype.kind not in "iu":
            raise FoldsInvalidos(f"y deve ser inteiro 0/1; dtype {y.dtype}")
        valores = set(np.unique(y).tolist())
        if not valores <= {0, 1}:
            raise FoldsInvalidos("y deve conter apenas 0 e 1")
        if valores != {0, 1}:
            raise FoldsInvalidos("y deve conter as duas classes")
        object.__setattr__(self, "ids", _somente_leitura(ids))
        object.__setattr__(self, "y", _somente_leitura(y.astype(np.int64)))
        object.__setattr__(self, "grupos", _somente_leitura(grupos))
        conteudo = json.dumps({"ids": lista, "y": self.y.tolist(), "grupos": list(grupos)},
                              ensure_ascii=False, separators=(",", ":"))
        object.__setattr__(self, "sha256", hashlib.sha256(conteudo.encode("utf-8")).hexdigest())

    @property
    def n(self) -> int:
        return len(self.ids)


def criar_amostra(ids: Iterable[str], y: Iterable[int], grupos: Iterable[str], X: Any = None) -> Amostra:
    """``Amostra`` a partir de arrays ou objetos pandas, exigindo alinhamento.

    Se algum argumento tiver ``index`` (Series ou DataFrame), todos os que
    tiverem precisam ter o MESMO índice: posições desalinhadas nunca são
    convertidas em silêncio. ``X``, se dado, só é usado para conferir
    comprimento e índice.
    """
    objetos = {"ids": ids, "y": y, "grupos": grupos}
    if X is not None:
        objetos["X"] = X
    # Índice pandas: atributo não chamável com ``equals`` (``list.index`` é um método).
    indices = {nome: getattr(obj, "index") for nome, obj in objetos.items()
               if hasattr(getattr(obj, "index", None), "equals") and not callable(getattr(obj, "index"))}
    if indices:
        referencia = next(iter(indices.values()))
        diferentes = [nome for nome, indice in indices.items() if not indice.equals(referencia)]
        if diferentes:
            raise FoldsInvalidos(f"índices desalinhados: {sorted(indices)}")
    if X is not None and len(X) != len(ids):
        raise FoldsInvalidos("X e ids têm comprimentos diferentes")
    return Amostra(np.asarray(ids, dtype=object), np.asarray(y), np.asarray(grupos, dtype=object))


# --------------------------------------------------------------------------- #
# Divisões
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class Divisao:
    """Uma divisão treino/avaliação, com posições na amostra completa."""

    nivel: str
    repeticao: int
    fold_externo: int
    fold_estabilidade: Optional[int]
    fold_interno: Optional[int]
    seed: int
    treino: np.ndarray = field(repr=False)
    avaliacao: np.ndarray = field(repr=False)
    sha256_amostra: str = field(repr=False)

    @property
    def rotulo(self) -> str:
        partes = [f"r{self.repeticao:02d}", f"k{self.fold_externo}"]
        if self.fold_estabilidade is not None:
            partes.append(f"e{self.fold_estabilidade}")
        if self.fold_interno is not None:
            partes.append(f"i{self.fold_interno}")
        return "-".join(partes)


def _dividir(amostra: Amostra, base: np.ndarray, n_splits: int, seed: int,
             contexto: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """``StratifiedGroupKFold`` sobre as linhas ``base``; devolve posições globais."""
    divisor = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    marcador = np.zeros((len(base), 1))  # o splitter não usa X além do número de linhas
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            partes = list(divisor.split(marcador, amostra.y[base], amostra.grupos[base]))
    except (ValueError, Warning) as exc:
        raise FoldsInvalidos(f"{contexto}: o splitter recusou a entrada: {exc}") from exc
    return [(_somente_leitura(np.sort(base[tr]).astype(np.int64)),
             _somente_leitura(np.sort(base[te]).astype(np.int64))) for tr, te in partes]


def _validar_nivel(amostra: Amostra, base: np.ndarray, partes: List[Tuple[np.ndarray, np.ndarray]],
                   n_esperado: int, contexto: str) -> None:
    if len(partes) != n_esperado:
        raise FoldsInvalidos(f"{contexto}: {len(partes)} folds; esperados {n_esperado}")
    conjunto_base = set(base.tolist())
    avaliadas: List[int] = []
    for i, (treino, avaliacao) in enumerate(partes):
        onde = f"{contexto}, fold {i}"
        if len(treino) == 0 or len(avaliacao) == 0:
            raise FoldsInvalidos(f"{onde}: treino ou avaliação vazio")
        if set(treino.tolist()) & set(avaliacao.tolist()):
            raise FoldsInvalidos(f"{onde}: linha em treino e avaliação")
        if set(treino.tolist()) | set(avaliacao.tolist()) != conjunto_base or \
                len(treino) + len(avaliacao) != len(base):
            raise FoldsInvalidos(f"{onde}: treino ∪ avaliação difere das linhas do nível")
        if set(amostra.grupos[treino].tolist()) & set(amostra.grupos[avaliacao].tolist()):
            raise FoldsInvalidos(f"{onde}: paciente em treino e avaliação")
        for lado, idx in (("treino", treino), ("avaliação", avaliacao)):
            if set(amostra.y[idx].tolist()) != {0, 1}:
                raise FoldsInvalidos(f"{onde}: {lado} sem as duas classes")
        avaliadas.extend(avaliacao.tolist())
    if sorted(avaliadas) != sorted(base.tolist()):
        raise FoldsInvalidos(f"{contexto}: linhas não avaliadas exatamente uma vez")


def _exigir_da_amostra(amostra: Amostra, divisao: Divisao, nivel: str) -> None:
    if not isinstance(amostra, Amostra):
        raise TypeError("amostra deve ser Amostra")
    if not isinstance(divisao, Divisao) or divisao.nivel != nivel:
        raise FoldsInvalidos(f"esperada uma divisão de nível {nivel!r}")
    if divisao.sha256_amostra != amostra.sha256:
        raise FoldsInvalidos(f"{divisao.rotulo}: divisão de outra amostra")
    # A divisão recebida é revalidada: uma Divisao montada à mão não entra.
    for lado in (divisao.treino, divisao.avaliacao):
        if not (isinstance(lado, np.ndarray) and lado.ndim == 1 and lado.dtype == np.int64 and len(lado)
                and lado[0] >= 0 and lado[-1] < amostra.n and np.all(np.diff(lado) > 0)):
            raise FoldsInvalidos(f"{divisao.rotulo}: índices inválidos")
    if np.intersect1d(divisao.treino, divisao.avaliacao).size or \
            set(amostra.grupos[divisao.treino].tolist()) & set(amostra.grupos[divisao.avaliacao].tolist()):
        raise FoldsInvalidos(f"{divisao.rotulo}: linha ou paciente em treino e avaliação")


def folds_externos(amostra: Amostra, repeticao: int) -> Tuple[Divisao, ...]:
    """Folds externos da repetição ``repeticao`` (seed ``ESPEC_CV.seeds_repeticoes()[r]``)."""
    if not isinstance(amostra, Amostra):
        raise TypeError("amostra deve ser Amostra")
    seeds = cfg.ESPEC_CV.seeds_repeticoes()
    if type(repeticao) is not int or not 0 <= repeticao < len(seeds):
        raise FoldsInvalidos(f"repeticao deve ser inteiro em [0, {len(seeds)})")
    base = _somente_leitura(np.arange(amostra.n, dtype=np.int64))
    n = cfg.ESPEC_CV.n_folds_externos
    contexto = f"externo r{repeticao:02d}"
    partes = _dividir(amostra, base, n, seeds[repeticao], contexto)
    _validar_nivel(amostra, base, partes, n, contexto)
    return tuple(Divisao(NIVEL_EXTERNO, repeticao, k, None, None, seeds[repeticao], tr, te, amostra.sha256)
                 for k, (tr, te) in enumerate(partes))


def folds_estabilidade(amostra: Amostra, externa: Divisao) -> Tuple[Divisao, ...]:
    """Folds de estabilidade dentro do TREINO de ``externa`` (seed ``ESPEC_SELECAO.seed_estabilidade``)."""
    _exigir_da_amostra(amostra, externa, NIVEL_EXTERNO)
    espec = cfg.ESPEC_SELECAO
    contexto = f"estabilidade {externa.rotulo}"
    partes = _dividir(amostra, externa.treino, espec.n_folds_estabilidade, espec.seed_estabilidade, contexto)
    _validar_nivel(amostra, externa.treino, partes, espec.n_folds_estabilidade, contexto)
    return tuple(Divisao(NIVEL_ESTABILIDADE, externa.repeticao, externa.fold_externo, j, None,
                         espec.seed_estabilidade, tr, te, amostra.sha256)
                 for j, (tr, te) in enumerate(partes))


def folds_internos(amostra: Amostra, estabilidade: Divisao) -> Tuple[Divisao, ...]:
    """Folds internos dentro do TREINO de ``estabilidade`` (seed ``seed_interna(j + 1)``)."""
    _exigir_da_amostra(amostra, estabilidade, NIVEL_ESTABILIDADE)
    espec = cfg.ESPEC_SELECAO
    seed = espec.seed_interna(estabilidade.fold_estabilidade + 1)
    contexto = f"interno {estabilidade.rotulo}"
    partes = _dividir(amostra, estabilidade.treino, espec.n_folds_interna, seed, contexto)
    _validar_nivel(amostra, estabilidade.treino, partes, espec.n_folds_interna, contexto)
    return tuple(Divisao(NIVEL_INTERNO, estabilidade.repeticao, estabilidade.fold_externo,
                         estabilidade.fold_estabilidade, i, seed, tr, te, amostra.sha256)
                 for i, (tr, te) in enumerate(partes))


# --------------------------------------------------------------------------- #
# Plano completo de uma repetição e fingerprints
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class PlanoRepeticao:
    """Todos os folds de uma repetição, validados nos três níveis."""

    repeticao: int
    sha256_amostra: str
    externos: Tuple[Divisao, ...]
    estabilidade: Mapping[int, Tuple[Divisao, ...]]
    internos: Mapping[Tuple[int, int], Tuple[Divisao, ...]]

    def divisoes_do_fold(self, k: int) -> Tuple[Divisao, ...]:
        """Divisão externa ``k`` seguida de todas as divisões aninhadas nela."""
        aninhadas = [d for j, est in enumerate(self.estabilidade[k]) for d in (est, *self.internos[(k, j)])]
        return (self.externos[k], *aninhadas)


def plano_repeticao(amostra: Amostra, repeticao: int) -> PlanoRepeticao:
    """Externos, estabilidade e internos da repetição, com checagens entre níveis.

    É o ponto de entrada recomendado: só ele confere que nenhuma divisão
    aninhada toca a avaliação EXTERNA. As funções de cada nível validam o
    próprio nível e a divisão-pai recebida, mas não conhecem o nível de cima
    dela.
    """
    externos = folds_externos(amostra, repeticao)
    estabilidade: Dict[int, Tuple[Divisao, ...]] = {}
    internos: Dict[Tuple[int, int], Tuple[Divisao, ...]] = {}
    for externa in externos:
        pacientes_avaliacao = set(amostra.grupos[externa.avaliacao].tolist())
        estabilidade[externa.fold_externo] = folds_estabilidade(amostra, externa)
        for est in estabilidade[externa.fold_externo]:
            internos[(externa.fold_externo, est.fold_estabilidade)] = folds_internos(amostra, est)
        # Nenhuma linha ou paciente da avaliação externa em qualquer nível aninhado.
        for d in (*estabilidade[externa.fold_externo],
                  *(i for j in range(len(estabilidade[externa.fold_externo]))
                    for i in internos[(externa.fold_externo, j)])):
            usadas = np.concatenate([d.treino, d.avaliacao])
            if np.isin(usadas, externa.avaliacao).any() or \
                    pacientes_avaliacao & set(amostra.grupos[usadas].tolist()):
                raise FoldsInvalidos(f"{d.rotulo}: usa dados da avaliação externa")
    return PlanoRepeticao(repeticao, amostra.sha256, externos,
                          MappingProxyType(estabilidade), MappingProxyType(internos))


def fingerprint_divisao(amostra: Amostra, divisao: Divisao) -> str:
    """Fingerprint (``checkpoints.fingerprint_folds``) de uma divisão isolada."""
    _exigir_da_amostra(amostra, divisao, divisao.nivel)
    return fingerprint_folds({"treino": amostra.ids[divisao.treino].tolist(),
                              "avaliacao": amostra.ids[divisao.avaliacao].tolist()})


def fingerprint_unidade(amostra: Amostra, plano: PlanoRepeticao, k: int) -> str:
    """Fingerprint da unidade ``(repetição, fold externo k)`` para os checkpoints.

    Cobre a divisão externa e TODAS as aninhadas (estabilidade e internas),
    porque o resultado da unidade depende de todas elas. Só o hash sai daqui:
    nenhum identificador é gravado.
    """
    if not isinstance(plano, PlanoRepeticao) or plano.sha256_amostra != amostra.sha256:
        raise FoldsInvalidos("plano de outra amostra")
    if type(k) is not int or k not in plano.estabilidade:
        raise FoldsInvalidos(f"fold externo inexistente: {k!r}")
    composicao = {}
    for d in plano.divisoes_do_fold(k):
        composicao[f"{d.rotulo}/treino"] = amostra.ids[d.treino].tolist()
        composicao[f"{d.rotulo}/avaliacao"] = amostra.ids[d.avaliacao].tolist()
    return fingerprint_folds(composicao)
