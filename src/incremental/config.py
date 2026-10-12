"""Configuração centralizada do Cartão 4 — Sprint 4.

Fonte única de nomes de features, composição de modelos, especificações,
seeds, limiares e estado das decisões pendentes do protocolo
(``docs/sprint4/protocolo_valor_incremental.md``).

Princípios:

* tudo é imutável: dataclasses congeladas, tuplas e ``MappingProxyType``;
* nenhuma lista de features é derivada por prefixo (protocolo, Seção 5.5);
* nenhuma decisão aberta recebe valor inventado: P1–P5 e a revisão do protocolo
  nascem PENDENTES, e o código que depende delas levanta ``BloqueioDecisao``;
* não há painel fixo padrão: M2, M3 e M4 só existem depois de P3 aprovada;
* ``hash_configuracao`` identifica a configuração de forma determinística, para
  ser gravado em todo artefato.

O módulo usa apenas a biblioteca padrão; o numpy é importado sob demanda só para
gerar a grade de ``C`` exatamente como ``np.logspace``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple

PROTOCOLO: str = "docs/sprint4/protocolo_valor_incremental.md"

# --------------------------------------------------------------------------- #
# Colunas e pool de candidatas
# --------------------------------------------------------------------------- #

COLUNA_NODULO: str = "nodulo_id"
COLUNA_PACIENTE: str = "patient_id"
COLUNA_ALVO: str = "target_binary"

#: Metade da chave canônica do nódulo — nunca é preditor (Seção 5.5).
COLUNA_PROIBIDA: str = "original_nodule_idx"
PREFIXO_RADIOMICA: str = "original_"
#: Tamanho exigido do pool de candidatas da seleção (Seção 5.5).
N_POOL_CANDIDATAS: int = 107
#: Número de features de um painel fixo (Seção 5.1).
N_PAINEL: int = 8

#: Decisão arquitetural: o pipeline corrigido usa a ordem canônica das linhas,
#: tabela ordenada por ``nodulo_id`` com ordenação estável, como no baseline.
ORDEM_LINHAS: Tuple[str, str] = (COLUNA_NODULO, "mergesort")

# --------------------------------------------------------------------------- #
# Features dos modelos de referência e painéis candidatos
# --------------------------------------------------------------------------- #

DIAMETRO: str = "original_shape_Maximum3DDiameter"

#: M1 — baseline oficial da Sprint 3, na ordem de ``BASELINE_FEATURES``.
FEATURES_M1: Tuple[str, ...] = (
    "original_shape_MeshVolume",
    DIAMETRO,
    "original_shape_Sphericity",
)

#: Lista declarada na conclusão do notebook de Juan, na ordem da Seção 5.1.
PAINEL_DECLARADO: Tuple[str, ...] = (
    "original_glcm_Imc2",
    "original_shape_Sphericity",
    "original_shape_SurfaceVolumeRatio",
    "original_firstorder_90Percentile",
    "original_firstorder_RootMeanSquared",
    "original_shape_Elongation",
    "original_firstorder_Skewness",
    "original_ngtdm_Coarseness",
)

#: Lista da reprodução fiel, na ordem de
#: ``reports/sprint4/feature_selection/reproducao_juan/features_selecionadas_lasso.csv``
#: (SHA-256 ``d45c6882…``). Candidata, NÃO padrão (auditoria, Seção 13).
PAINEL_REPRODUZIDO: Tuple[str, ...] = (
    "original_shape_Sphericity",
    "original_shape_SurfaceVolumeRatio",
    "original_glcm_Imc1",
    "original_glcm_Imc2",
    "original_firstorder_RootMeanSquared",
    "original_firstorder_90Percentile",
    "original_shape_Elongation",
    "original_glcm_ClusterShade",
)

#: Painéis candidatos, só para consulta. Nenhum é usado sem P3 aprovada.
PAINEIS_CANDIDATOS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "declarado": PAINEL_DECLARADO,
    "reproduzido": PAINEL_REPRODUZIDO,
})

# --------------------------------------------------------------------------- #
# Especificações (Seções 5.3, 5.4, 6.1, 7.2, 8.2 e 10.1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EspecRegressaoLogistica:
    """Modelo de avaliação (Seção 6.1), idêntico ao baseline da Sprint 3.

    ``solver="lbfgs"`` é explícito; no baseline ele era o padrão do
    scikit-learn 1.3.2.
    """

    C: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    class_weight: Optional[str] = None
    max_iter: int = 1000
    random_state: int = 42

    def parametros(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GradeLog:
    """Grade logarítmica, gerada exatamente como ``np.logspace(inicio, fim, n)``."""

    inicio: float
    fim: float
    n: int

    def valores(self) -> Tuple[float, ...]:
        import numpy as np  # sob demanda: o módulo não depende de numpy

        return tuple(float(v) for v in np.logspace(self.inicio, self.fim, self.n))


@dataclass(frozen=True)
class EspecSelecao:
    """Procedimento de seleção ajustado para a CV aninhada (Seções 5.3 e 5.4).

    Grade, seeds, limiar de coeficiente, regra 3 de 5 e pool são idênticos ao
    procedimento original; a diferença — filtros ajustados em cada treino
    interno — é de arquitetura, não de parâmetro.
    """

    limiar_variancia: float = 0.0
    #: Remove a coluna posterior de cada par com |r de Pearson| ESTRITAMENTE
    #: maior que o limiar; mantém a que aparece primeiro na ordem do pool.
    limiar_correlacao: float = 0.90
    penalty: str = "l1"
    solver: str = "liblinear"
    max_iter: int = 5000
    random_state: int = 42
    grade_C: GradeLog = GradeLog(-4.0, 2.0, 13)
    scoring: str = "roc_auc"
    #: Feature selecionada no fold se |coef| > limiar.
    limiar_coeficiente: float = 1e-12
    n_folds_estabilidade: int = 5
    #: Seed da CV de estabilidade dentro de cada treino (Seção 5.4: idêntica ao original).
    seed_estabilidade: int = 42
    n_folds_interna: int = 4
    #: Seed da CV interna do fold de estabilidade j (j = 1…5): base + j.
    seed_interna_base: int = 100
    #: Lista do treino = features selecionadas em pelo menos este número de folds.
    limiar_estabilidade: int = 3

    def seed_interna(self, j: int) -> int:
        if not 1 <= j <= self.n_folds_estabilidade:
            raise ValueError(f"fold de estabilidade fora de 1…{self.n_folds_estabilidade}: {j}")
        return self.seed_interna_base + j


@dataclass(frozen=True)
class EspecCVAninhada:
    """CV aninhada (Seção 7.2) e decisão de medir paralelismo no piloto."""

    n_folds_externos: int = 5
    #: Piloto = repetição r = 0.
    seed_piloto: int = 42
    #: Repetição r usa ``seed_base_repeticoes + r``, r = 0…n_repeticoes - 1.
    seed_base_repeticoes: int = 42
    n_repeticoes: int = 10
    #: Decisão arquitetural: o piloto mede tempo com 1, 2 e 4 workers.
    workers_a_medir_no_piloto: Tuple[int, ...] = (1, 2, 4)

    def seeds_repeticoes(self) -> Tuple[int, ...]:
        return tuple(self.seed_base_repeticoes + r for r in range(self.n_repeticoes))


@dataclass(frozen=True)
class EspecBootstrap:
    """Bootstrap pareado por paciente (Seção 8.2)."""

    n_replicas: int = 2000
    seed: int = 42
    percentis: Tuple[float, float] = (2.5, 97.5)
    unidade: str = COLUNA_PACIENTE
    pareado: bool = True
    retreino: bool = False
    descartar_replicas_com_uma_classe: bool = True


@dataclass(frozen=True)
class EspecSpline:
    """Parâmetros PROPOSTOS na Seção 10.1; uso condicionado à aprovação de P2."""

    coluna: str = DIAMETRO
    n_knots: int = 4
    degree: int = 3
    knots: str = "quantile"
    extrapolation: str = "linear"
    include_bias: bool = False


ESPEC_AVALIACAO: EspecRegressaoLogistica = EspecRegressaoLogistica()
ESPEC_SELECAO: EspecSelecao = EspecSelecao()
ESPEC_CV: EspecCVAninhada = EspecCVAninhada()
ESPEC_BOOTSTRAP: EspecBootstrap = EspecBootstrap()
#: Proposta do protocolo — use ``spline_confirmado()``, nunca este objeto direto.
SPLINE_PROPOSTO: EspecSpline = EspecSpline()

#: Threshold explícito sobre ``predict_proba`` (Seção 6.1).
THRESHOLD: float = 0.5
#: M2v com S_k vazio prevê a prevalência do treino externo (Seção 6.3).
DUMMY_ESTRATEGIA: str = "prior"
#: Métrica principal (Seção 8.1).
METRICA_PRINCIPAL: str = "roc_auc"
#: Referência exploratória de magnitude, só descritiva (Seção 9).
REFERENCIA_EXPLORATORIA_DELTA_AUC: float = 0.02
#: Nenhum p-valor é calculado (Seção 8.3).
CALCULA_P_VALOR: bool = False

# --------------------------------------------------------------------------- #
# Invariantes esperadas e ambiente (Seções 2, 4 e 12)
# --------------------------------------------------------------------------- #

COORTE_ESPERADA: Mapping[str, int] = MappingProxyType(
    {"nodulos": 607, "pacientes": 416, "classe_0": 310, "classe_1": 297}
)
PARTICOES_ESPERADAS: Mapping[str, Mapping[str, int]] = MappingProxyType({
    "train": MappingProxyType({"pacientes": 291, "nodulos": 417, "classe_0": 209, "classe_1": 208}),
    "validation": MappingProxyType({"pacientes": 62, "nodulos": 92, "classe_0": 49, "classe_1": 43}),
    "test": MappingProxyType({"pacientes": 63, "nodulos": 98, "classe_0": 52, "classe_1": 46}),
})
AMBIENTE_ESPERADO: Mapping[str, str] = MappingProxyType({
    "python": "3.9.13",
    "scikit-learn": "1.3.2",
    "numpy": "1.23.5",
    "pandas": "1.5.3",
    "scipy": "1.10.1",
})

# --------------------------------------------------------------------------- #
# Etapas e isolamento (Seção 7)
# --------------------------------------------------------------------------- #


class Etapa(str, Enum):
    """Etapas do protocolo, na ordem de execução (Seção 7)."""

    E0 = "etapa0_prerequisitos"
    E1 = "etapa1_piloto"
    E2 = "etapa2_cv_aninhada"
    E3 = "etapa3_validacao"
    E4 = "etapa4_congelamento"
    E5 = "etapa5_teste"

    @property
    def ordem(self) -> int:
        return tuple(Etapa).index(self)


#: Partições cujas linhas cada etapa pode ler (Seção 7.1, item 4).
PARTICOES_POR_ETAPA: Mapping[Etapa, Tuple[str, ...]] = MappingProxyType({
    Etapa.E0: ("train",),
    Etapa.E1: ("train",),
    Etapa.E2: ("train",),
    Etapa.E3: ("train", "validation"),
    Etapa.E4: (),
    Etapa.E5: ("train", "test"),
})

# --------------------------------------------------------------------------- #
# Decisões pendentes (Seção 13) — nenhuma aprovação é presumida
# --------------------------------------------------------------------------- #


class StatusDecisao(str, Enum):
    PENDENTE = "pendente"
    APROVADA = "aprovada"


_HASH_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Decisao:
    """Decisão do protocolo. Só fica APROVADA com o commit da revisão que a registra.

    A existência desse commit no histórico é verificada pelas guardas da
    execução, não aqui.
    """

    codigo: str
    descricao: str
    prazo: str
    status: StatusDecisao = StatusDecisao.PENDENTE
    commit_revisao_protocolo: Optional[str] = None
    #: Conteúdo aprovado, quando houver (P3: o painel; P2: um ``EspecSpline``).
    valor: Any = None
    #: Primeira etapa que exige a decisão (prazo da Seção 13). A exigência é
    #: cumulativa: vale para essa etapa e para todas as seguintes.
    etapa_limite: Optional[Etapa] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, StatusDecisao):
            raise TypeError(f"{self.codigo}: status deve ser StatusDecisao")
        if self.etapa_limite is not None and not isinstance(self.etapa_limite, Etapa):
            raise TypeError(f"{self.codigo}: etapa_limite deve ser Etapa")
        if self.status is StatusDecisao.APROVADA:
            if not (isinstance(self.commit_revisao_protocolo, str)
                    and _HASH_COMMIT.fullmatch(self.commit_revisao_protocolo)):
                raise ValueError(
                    f"{self.codigo}: decisão aprovada exige o hash completo (40 hex) do "
                    "commit da revisão do protocolo"
                )
        elif self.commit_revisao_protocolo is not None or self.valor is not None:
            raise ValueError(f"{self.codigo}: decisão pendente não pode ter commit nem valor")

    @property
    def aprovada(self) -> bool:
        return self.status is StatusDecisao.APROVADA


DECISOES: Mapping[str, Decisao] = MappingProxyType({
    "P1": Decisao(
        "P1",
        "Número de permutações K, esquema (por nódulo ou por paciente) e tratamento dos "
        "folds de estabilidade e internos sob rótulos permutados (Seção 7.2)",
        "antes da Etapa 1",
        etapa_limite=Etapa.E1,
    ),
    "P2": Decisao(
        "P2",
        "Confirmação dos parâmetros do spline (Seção 10.1)",
        "antes da Etapa 1",
        etapa_limite=Etapa.E1,
    ),
    "P3": Decisao(
        "P3",
        "Regra para divergência na reprodução fiel e lista fixa de M2, M3 e M4 "
        "(Seção 7.1, item 1); bloqueia também a CV descritiva com lista fixa da Etapa 2",
        "antes da Etapa 3",
        etapa_limite=Etapa.E3,
    ),
    "P4": Decisao(
        "P4",
        "Explicação da divergência entre M1 = 0,9778 (Experimento 13) e 0,9762 (baseline)",
        "antes da Etapa 5",
        etapa_limite=Etapa.E5,
    ),
    "P5": Decisao(
        "P5",
        "Versionamento dos registros dos Experimentos 10 e 13 e do notebook de Juan",
        "antes da Etapa 4",
        etapa_limite=Etapa.E4,
    ),
    "REVISAO_PROTOCOLO": Decisao(
        "REVISAO_PROTOCOLO",
        "Revisão do protocolo pelo grupo e pelo professor",
        "antes da Etapa 4",
        etapa_limite=Etapa.E4,
    ),
})

#: Atividade que antecipa, na Etapa 2, um uso da lista fixa: além das decisões
#: da Etapa 2, exige P3 (Seção 7.2, CV descritiva com as oito features fixas).
ATIVIDADE_CV_LISTA_FIXA: str = "etapa2_cv_lista_fixa"
#: Verificações puramente estruturais (invariantes de dados, composição e
#: validação de folds, sem ajuste de modelo nem métrica): não exigem decisão.
ATIVIDADE_VERIFICACOES_ESTRUTURAIS: str = "verificacoes_estruturais"


def _exigidas_ate(etapa: Etapa) -> Tuple[str, ...]:
    return tuple(c for c, d in DECISOES.items()
                 if d.etapa_limite is not None and d.etapa_limite.ordem <= etapa.ordem)


#: Decisões exigidas por etapa ou atividade, derivadas de ``etapa_limite``: cada
#: decisão vale a partir do seu prazo e em todas as etapas seguintes (Seção 13).
DECISOES_POR_ETAPA: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    ATIVIDADE_VERIFICACOES_ESTRUTURAIS: (),
    **{e.value: _exigidas_ate(e) for e in Etapa},
    ATIVIDADE_CV_LISTA_FIXA: tuple(c for c in DECISOES if c in {*_exigidas_ate(Etapa.E2), "P3"}),
})


class BloqueioDecisao(RuntimeError):
    """Uso de algo que depende de uma decisão do protocolo ainda pendente."""


def exigir_aprovada(codigo: str, decisoes: Mapping[str, Decisao] = DECISOES) -> Decisao:
    decisao = decisoes[codigo]
    if not decisao.aprovada:
        raise BloqueioDecisao(
            f"{codigo} está pendente ({decisao.descricao}; prazo: {decisao.prazo}). "
            "Registre a decisão em revisão do protocolo antes de prosseguir."
        )
    return decisao


# --------------------------------------------------------------------------- #
# Validação de nomes e união ordenada
# --------------------------------------------------------------------------- #


def validar_nomes(nomes: Iterable[str], contexto: str) -> Tuple[str, ...]:
    """Exige nomes explícitos, radiômicos, sem duplicata e sem ``original_nodule_idx``."""
    nomes = tuple(nomes)
    for nome in nomes:
        if not isinstance(nome, str) or not nome.startswith(PREFIXO_RADIOMICA):
            raise ValueError(f"{contexto}: nome inválido {nome!r}")
        if nome == COLUNA_PROIBIDA:
            raise ValueError(f"{contexto}: {COLUNA_PROIBIDA} é identificador, não feature")
    if len(set(nomes)) != len(nomes):
        raise ValueError(f"{contexto}: nomes duplicados")
    return nomes


def uniao_ordenada(*grupos: Iterable[str]) -> Tuple[str, ...]:
    """Concatena preservando a primeira ocorrência; cada feature entra uma única vez."""
    vistos: dict = {}
    for grupo in grupos:
        for nome in grupo:
            vistos.setdefault(nome, None)
    return tuple(vistos)


def painel_fixo(decisoes: Mapping[str, Decisao] = DECISOES) -> Tuple[str, ...]:
    """Painel de M2, M3 e M4. Levanta ``BloqueioDecisao`` enquanto P3 estiver pendente."""
    decisao = exigir_aprovada("P3", decisoes)
    painel = decisao.valor
    if not isinstance(painel, tuple):
        raise ValueError("P3 aprovada deve trazer o painel como tupla de nomes")
    painel = validar_nomes(painel, "painel de P3")
    if len(painel) != N_PAINEL:
        raise ValueError(f"painel de P3 deve ter {N_PAINEL} features; tem {len(painel)}")
    return painel


def spline_confirmado(decisoes: Mapping[str, Decisao] = DECISOES) -> EspecSpline:
    """Parâmetros do spline. Levanta ``BloqueioDecisao`` enquanto P2 estiver pendente."""
    decisao = exigir_aprovada("P2", decisoes)
    if not isinstance(decisao.valor, EspecSpline):
        raise ValueError("P2 aprovada deve trazer um EspecSpline")
    return decisao.valor


# --------------------------------------------------------------------------- #
# Composição dos modelos (Seções 6 e 6.3)
# --------------------------------------------------------------------------- #

MODELOS_FIXOS: Tuple[str, ...] = ("M0", "M1", "M2", "M3", "M4")
MODELOS_VARIAVEIS: Tuple[str, ...] = ("M2v", "M3v", "M4v")
#: Modelos que exigem o painel fixo de P3.
MODELOS_COM_PAINEL: Tuple[str, ...] = ("M2", "M3", "M4")

#: Preditores de referência de cada modelo, antes do termo radiômico.
REFERENCIA: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "M0": (DIAMETRO,),
    "M1": FEATURES_M1,
    "M2": (),
    "M3": (DIAMETRO,),
    "M4": FEATURES_M1,
    "M2v": (),
    "M3v": (DIAMETRO,),
    "M4v": FEATURES_M1,
})


class TipoModelo(str, Enum):
    FIXO = "fixo"
    VARIAVEL = "variavel"


#: Marca de que um M2, M3 ou M4 foi montado por ``modelo_fixo`` depois de
#: ``painel_fixo`` aceitar P3; vai em ``EspecModelo._autorizacao`` junto com o
#: painel aceito. É PROTEÇÃO CONTRA USO ACIDENTAL — construção
#: direta ou ``dataclasses.replace`` —, não mecanismo de segurança: código que
#: importe este nome privado consegue contorná-la. As guardas efetivas de
#: execução (decisão aprovada com commit existente e ancestral do HEAD,
#: isolamento de dados por etapa) ficam para o Incremento 2.
_AUTORIZACAO_PAINEL_P3 = object()


@dataclass(frozen=True)
class EspecModelo:
    """Composição de um modelo, validada na construção.

    Use ``modelo_fixo`` e ``modelo_variavel``. Invariantes:

    * o nome define o tipo: M0–M4 são fixos; M2v–M4v são variáveis;
    * features explícitas, válidas e sem duplicata; os preditores de
      referência do modelo vêm primeiro, na ordem de ``REFERENCIA``;
    * M0 e M1 têm exatamente seus preditores de referência;
    * M2, M3 e M4 exigem a marca de ``modelo_fixo``, que carrega o painel
      aceito por P3, e features iguais à referência ∪ esse painel; o termo
      radiômico tem de 1 a ``N_PAINEL`` features e M2 tem exatamente ``N_PAINEL``;
    * ``usa_dummy`` é verdadeiro se, e somente se, o modelo é M2v sem features.
    """

    nome: str
    tipo: TipoModelo
    features: Tuple[str, ...]
    usa_dummy: bool = False
    _autorizacao: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        nome = self.nome
        if nome in MODELOS_FIXOS:
            tipo_esperado = TipoModelo.FIXO
        elif nome in MODELOS_VARIAVEIS:
            tipo_esperado = TipoModelo.VARIAVEL
        else:
            raise ValueError(f"modelo desconhecido: {nome!r}")
        if self.tipo is not tipo_esperado:
            raise ValueError(f"{nome}: tipo deve ser {tipo_esperado.value}")
        if not isinstance(self.features, tuple):
            raise TypeError(f"{nome}: features deve ser tupla")
        validar_nomes(self.features, nome)
        if not isinstance(self.usa_dummy, bool):
            raise TypeError(f"{nome}: usa_dummy deve ser bool")
        if self.usa_dummy != (nome == "M2v" and not self.features):
            raise ValueError(f"{nome}: Dummy só em M2v e somente com S_k vazio")

        referencia = REFERENCIA[nome]
        if self.features[:len(referencia)] != referencia:
            raise ValueError(f"{nome}: os preditores de referência {referencia} devem vir primeiro")
        termo_radiomico = self.features[len(referencia):]
        if nome in ("M0", "M1") and termo_radiomico:
            raise ValueError(f"{nome}: não admite termo radiômico")
        if nome in MODELOS_COM_PAINEL:
            marca = self._autorizacao
            if not (isinstance(marca, tuple) and len(marca) == 2 and marca[0] is _AUTORIZACAO_PAINEL_P3):
                raise BloqueioDecisao(
                    f"{nome} só pode ser montado por modelo_fixo, depois de P3 aprovada"
                )
            if self.features != uniao_ordenada(referencia, marca[1]):
                raise ValueError(f"{nome}: features diferentes da referência ∪ painel aprovado em P3")
            if not 1 <= len(termo_radiomico) <= N_PAINEL:
                raise ValueError(f"{nome}: termo radiômico deve ter de 1 a {N_PAINEL} features")
            if nome == "M2" and len(self.features) != N_PAINEL:
                raise ValueError(f"M2 deve ter {N_PAINEL} features")

    @property
    def n_features(self) -> int:
        return len(self.features)


def modelo_fixo(nome: str, decisoes: Mapping[str, Decisao] = DECISOES) -> EspecModelo:
    """M0 e M1 sempre; M2, M3 e M4 só com P3 aprovada."""
    if nome not in MODELOS_FIXOS:
        raise ValueError(f"modelo fixo desconhecido: {nome!r}")
    if nome in MODELOS_COM_PAINEL:
        painel = painel_fixo(decisoes)
        features = uniao_ordenada(REFERENCIA[nome], painel)
        return EspecModelo(nome, TipoModelo.FIXO, features,
                           _autorizacao=(_AUTORIZACAO_PAINEL_P3, painel))
    return EspecModelo(nome, TipoModelo.FIXO, REFERENCIA[nome])


def modelo_variavel(nome: str, s_k: Iterable[str]) -> EspecModelo:
    """M2v = S_k; M3v = diâmetro ∪ S_k; M4v = M1 ∪ S_k (Seção 6.3).

    S_k pode ser vazio e pode conter features de M0 ou M1, que entram uma única
    vez. Com S_k vazio, só M2v usa Dummy; M3v e M4v mantêm seus preditores de
    referência e coincidem com M0 e M1 naquele fold.
    """
    if nome not in MODELOS_VARIAVEIS:
        raise ValueError(f"modelo variável desconhecido: {nome!r}")
    s_k = validar_nomes(s_k, f"S_k de {nome}")
    features = uniao_ordenada(REFERENCIA[nome], s_k)
    usa_dummy = nome == "M2v" and not s_k
    return EspecModelo(nome, TipoModelo.VARIAVEL, features, usa_dummy=usa_dummy)


# --------------------------------------------------------------------------- #
# Snapshot e hash determinístico
# --------------------------------------------------------------------------- #


def _jsonavel(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        # Campos privados (como a marca de autorização) não descrevem a configuração.
        return {"__tipo__": type(obj).__name__,
                **{f.name: _jsonavel(getattr(obj, f.name)) for f in fields(obj)
                   if not f.name.startswith("_")}}
    if isinstance(obj, Mapping):
        return {str(_jsonavel(k)): _jsonavel(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_jsonavel(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"valor não serializável na configuração: {type(obj).__name__}")


def snapshot(decisoes: Mapping[str, Decisao] = DECISOES) -> dict:
    """Configuração completa como dicionário serializável em JSON."""
    return _jsonavel({
        "protocolo": PROTOCOLO,
        "colunas": {"nodulo": COLUNA_NODULO, "paciente": COLUNA_PACIENTE, "alvo": COLUNA_ALVO,
                    "proibida": COLUNA_PROIBIDA, "prefixo_radiomica": PREFIXO_RADIOMICA},
        "n_pool_candidatas": N_POOL_CANDIDATAS,
        "n_painel": N_PAINEL,
        "ordem_linhas": ORDEM_LINHAS,
        "diametro": DIAMETRO,
        "features_m1": FEATURES_M1,
        "paineis_candidatos": PAINEIS_CANDIDATOS,
        "referencia_modelos": REFERENCIA,
        "modelos_fixos": MODELOS_FIXOS,
        "modelos_variaveis": MODELOS_VARIAVEIS,
        "espec_avaliacao": ESPEC_AVALIACAO,
        "espec_selecao": ESPEC_SELECAO,
        "espec_cv": ESPEC_CV,
        "espec_bootstrap": ESPEC_BOOTSTRAP,
        "spline_proposto": SPLINE_PROPOSTO,
        "threshold": THRESHOLD,
        "dummy_estrategia": DUMMY_ESTRATEGIA,
        "metrica_principal": METRICA_PRINCIPAL,
        "referencia_exploratoria_delta_auc": REFERENCIA_EXPLORATORIA_DELTA_AUC,
        "calcula_p_valor": CALCULA_P_VALOR,
        "coorte_esperada": COORTE_ESPERADA,
        "particoes_esperadas": PARTICOES_ESPERADAS,
        "ambiente_esperado": AMBIENTE_ESPERADO,
        "particoes_por_etapa": PARTICOES_POR_ETAPA,
        "decisoes": decisoes,
        "decisoes_por_etapa": DECISOES_POR_ETAPA,
    })


def hash_configuracao(decisoes: Mapping[str, Decisao] = DECISOES) -> str:
    """SHA-256 do JSON canônico da configuração (chaves ordenadas, sem NaN)."""
    texto = json.dumps(snapshot(decisoes), sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()
