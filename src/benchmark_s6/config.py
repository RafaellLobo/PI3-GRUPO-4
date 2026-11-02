"""Configuração canônica do benchmark de classificadores da Sprint 6 (P9).

Fonte única dos parâmetros pré-registrados em
``docs/sprint6/preregistro_p9_benchmark.md`` (PR #11, versão 1.0, commit
``31931d0``). Cada bloco cita a seção do pré-registro de onde vem.

Princípios:

* tudo é imutável: dataclasses congeladas, tuplas e ``MappingProxyType``;
* toda especificação é validada na construção; uma configuração incoerente
  com a P9 (modelo desconhecido, SVM com métrica de probabilidade, teste
  entre as partições lidas, parâmetro alterável por tempo…) levanta
  ``ConfiguracaoInvalida``;
* nenhum dado é lido e nenhum modelo é construído: estimadores são nomes;
* ``conteudo_canonico`` devolve a configuração como JSON simples, e
  ``sha256_configuracao`` é o ``valor_sha256`` do futuro registro de aprovação
  P9 — igual a ``guardas.hash_valor_decisao`` e a
  ``rastreabilidade.sha256_canonico`` desse conteúdo (P9, Seção 14.4);
* o arquivo canônico (``docs/sprint6/config_benchmark_p9.json``) é o JSON de
  ``rastreabilidade.json_canonico`` seguido de ``\\n``; é gerado e conferido
  sem nunca ser sobrescrito.

Valores derivados (grade de ``C`` materializada, percentis de Bonferroni,
número de ajustes) NÃO entram no JSON; só as especificações que os geram.
Assim o hash não depende de aritmética de ponto flutuante de bibliotecas.
"""

from __future__ import annotations

import collections.abc
import itertools
import json
import math
import os
import re
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from . import adaptadores_s4 as s4

PREREGISTRO: str = "docs/sprint6/preregistro_p9_benchmark.md"
VERSAO_PREREGISTRO: str = "1.0"
ARQUIVO_CONFIGURACAO: str = "docs/sprint6/config_benchmark_p9.json"
#: Versão do esquema do JSON canônico; muda só com mudança de estrutura.
VERSAO_ESQUEMA: int = 1

COLUNA_NODULO: str = "nodulo_id"
COLUNA_PACIENTE: str = "patient_id"
COLUNA_ALVO: str = "target_binary"


class ConfiguracaoInvalida(ValueError):
    """Especificação incoerente com a P9 ou estruturalmente inválida."""


class ConfiguracaoDivergente(RuntimeError):
    """Arquivo de configuração diferente da configuração canônica gerada pelo código."""


# --------------------------------------------------------------------------- #
# Validação elementar
# --------------------------------------------------------------------------- #

_CODIGO = re.compile(r"[a-z][a-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise ConfiguracaoInvalida(mensagem)


def _inteiro(valor: Any, nome: str, minimo: Optional[int] = None) -> int:
    _exigir(type(valor) is int, f"{nome} deve ser int")
    if minimo is not None:
        _exigir(valor >= minimo, f"{nome} deve ser ≥ {minimo}")
    return valor


def _real(valor: Any, nome: str) -> float:
    _exigir(type(valor) in (int, float) and math.isfinite(valor), f"{nome} deve ser real finito")
    return float(valor)


def _texto(valor: Any, nome: str) -> str:
    _exigir(type(valor) is str and valor != "", f"{nome} deve ser texto não vazio")
    return valor


def _escalar(valor: Any, nome: str) -> Any:
    """Valor admitido num parâmetro fixo: texto, bool, int, real finito ou ``None``."""
    if valor is None or type(valor) in (str, bool, int):
        return valor
    return _real(valor, nome)


def _instancia(valor: Any, tipo: type, nome: str) -> Any:
    _exigir(isinstance(valor, tipo), f"{nome} deve ser {tipo.__name__}")
    return valor


def _tupla_de(valores: Any, tipo: type, nome: str) -> tuple:
    """Tupla (imutável) cujos itens são todos ``tipo``; lista é recusada."""
    _exigir(isinstance(valores, tuple) and all(isinstance(v, tipo) for v in valores),
            f"{nome} deve ser tupla de {tipo.__name__}")
    return valores


def _tupla_textos(valores: Any, nome: str, vazia: bool = False) -> Tuple[str, ...]:
    _exigir(isinstance(valores, tuple), f"{nome} deve ser tupla")
    _exigir(vazia or len(valores) > 0, f"{nome} não pode ser vazia")
    for v in valores:
        _texto(v, nome)
    _exigir(len(set(valores)) == len(valores), f"{nome} com repetição")
    return valores


# --------------------------------------------------------------------------- #
# Grades e modelos (P9, Seção 5)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GradeLog:
    """Grade ``numpy.logspace(inicio, fim, n)``; no JSON só a especificação."""

    inicio: int
    fim: int
    n: int

    def __post_init__(self) -> None:
        _inteiro(self.inicio, "GradeLog.inicio")
        _inteiro(self.fim, "GradeLog.fim")
        _inteiro(self.n, "GradeLog.n", minimo=2)
        _exigir(self.fim > self.inicio, "GradeLog: fim deve ser maior que inicio")

    def valores(self) -> Tuple[float, ...]:
        import numpy as np  # sob demanda: só a materialização usa numpy

        return tuple(float(v) for v in np.logspace(self.inicio, self.fim, self.n))

    def __len__(self) -> int:
        return self.n


@dataclass(frozen=True)
class ParametroGrade:
    """Um hiperparâmetro ajustado, com valores NA ORDEM DE SIMPLICIDADE."""

    nome: str
    valores: Union[Tuple[Union[int, float], ...], GradeLog]

    def __post_init__(self) -> None:
        _texto(self.nome, "ParametroGrade.nome")
        if isinstance(self.valores, GradeLog):
            return
        _exigir(isinstance(self.valores, tuple) and len(self.valores) > 0,
                f"{self.nome}: valores devem ser tupla não vazia ou GradeLog")
        for v in self.valores:
            _exigir(type(v) in (int, float) and math.isfinite(v),
                    f"{self.nome}: valor de grade deve ser número finito (não bool)")
        _exigir(len(set(self.valores)) == len(self.valores), f"{self.nome}: valores repetidos")

    def materializar(self) -> Tuple[Union[int, float], ...]:
        return self.valores.valores() if isinstance(self.valores, GradeLog) else self.valores


PAPEIS: Tuple[str, ...] = ("referencia_historica", "comparador", "desafiante")
PONTUACOES: Tuple[str, ...] = ("predict_proba", "decision_function")


@dataclass(frozen=True)
class EspecModelo:
    """Modelo do benchmark (P9, Seção 5.1). ``estimador`` é só o nome da classe."""

    codigo: str
    papel: str
    estimador: str
    padronizacao: bool
    fixos: Mapping[str, Any]
    grade: Tuple[ParametroGrade, ...]
    pontuacao: str
    metricas_probabilisticas: bool
    calibracao: bool = False
    parada_antecipada: bool = False

    def __post_init__(self) -> None:
        _exigir(type(self.codigo) is str and _CODIGO.fullmatch(self.codigo) is not None,
                f"código de modelo inválido: {self.codigo!r}")
        _exigir(self.papel in PAPEIS, f"{self.codigo}: papel inválido {self.papel!r}")
        _texto(self.estimador, f"{self.codigo}.estimador")
        for nome in ("padronizacao", "metricas_probabilisticas", "calibracao", "parada_antecipada"):
            _exigir(type(getattr(self, nome)) is bool, f"{self.codigo}.{nome} deve ser bool")
        _exigir(isinstance(self.fixos, Mapping), f"{self.codigo}: fixos deve ser mapeamento")
        fixos = {}
        for chave, valor in self.fixos.items():
            fixos[_texto(chave, f"{self.codigo}: nome de parâmetro fixo")] = \
                _escalar(valor, f"{self.codigo}.{chave}")
        object.__setattr__(self, "fixos", MappingProxyType(fixos))
        _tupla_de(self.grade, ParametroGrade, f"{self.codigo}.grade")
        nomes = [p.nome for p in self.grade]
        _exigir(len(set(nomes)) == len(nomes), f"{self.codigo}: hiperparâmetro repetido na grade")
        _exigir(not set(nomes) & set(fixos), f"{self.codigo}: parâmetro fixo e ajustado ao mesmo tempo")
        _exigir(self.pontuacao in PONTUACOES, f"{self.codigo}: pontuação inválida {self.pontuacao!r}")
        # P9, Seções 7.1 e 7.4: métricas de probabilidade se e somente se a
        # pontuação é probabilidade; o SVM fica sem elas e sem calibração.
        _exigir(self.metricas_probabilisticas == (self.pontuacao == "predict_proba"),
                f"{self.codigo}: métricas de probabilidade exigem pontuação predict_proba, e vice-versa")
        _exigir(not self.calibracao, f"{self.codigo}: calibração fora do escopo (P9, D10)")
        _exigir(not self.parada_antecipada, f"{self.codigo}: parada antecipada fora do escopo (P9, Seção 5.1)")
        _exigir((self.papel == "referencia_historica") == (not self.grade),
                f"{self.codigo}: só a referência histórica não tem grade")

    @property
    def n_configuracoes(self) -> int:
        return math.prod(len(p.valores) for p in self.grade) if self.grade else 1

    @property
    def n_hiperparametros(self) -> int:
        return len(self.grade)

    def configuracoes(self) -> Tuple[Mapping[str, Any], ...]:
        """Configurações na ordem de simplicidade: produto na ordem declarada.

        O primeiro hiperparâmetro varia mais devagar; a primeira configuração
        é a mais simples (P9, Seção 5.2).
        """
        if not self.grade:
            return (MappingProxyType({}),)
        nomes = [p.nome for p in self.grade]
        return tuple(MappingProxyType(dict(zip(nomes, combinacao)))
                     for combinacao in itertools.product(*(p.materializar() for p in self.grade)))

    def valores_de(self, nome: str) -> Tuple[Union[int, float], ...]:
        """Valores materializados de um hiperparâmetro da grade (vazio se não houver)."""
        for p in self.grade:
            if p.nome == nome:
                return p.materializar()
        return ()

    def ajustes_por_dobra_externa(self, n_dobras_internas: int) -> int:
        """Ajustes do ajuste interno mais o reajuste; a referência é ajustada uma vez."""
        return 1 if not self.grade else self.n_configuracoes * n_dobras_internas + 1


# --------------------------------------------------------------------------- #
# Blocos da especificação
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EspecArquivo:
    caminho: str
    sha256: str
    sha256_lf: str

    def __post_init__(self) -> None:
        _texto(self.caminho, "EspecArquivo.caminho")
        _exigir(not Path(self.caminho).is_absolute() and "\\" not in self.caminho,
                "EspecArquivo.caminho deve ser relativo, em POSIX")
        for nome in ("sha256", "sha256_lf"):
            _exigir(type(getattr(self, nome)) is str and _SHA256.fullmatch(getattr(self, nome)) is not None,
                    f"EspecArquivo.{nome} inválido")


_CHAVES_CONTAGEM = ("pacientes", "nodulos", "classe_0", "classe_1")


def _contagem(valor: Mapping[str, int], nome: str) -> Mapping[str, int]:
    _exigir(isinstance(valor, Mapping) and set(valor) == set(_CHAVES_CONTAGEM),
            f"{nome}: chaves esperadas {_CHAVES_CONTAGEM}")
    for chave in _CHAVES_CONTAGEM:
        _inteiro(valor[chave], f"{nome}.{chave}", minimo=0)
    _exigir(valor["classe_0"] + valor["classe_1"] == valor["nodulos"], f"{nome}: classes ≠ nódulos")
    return MappingProxyType({k: valor[k] for k in _CHAVES_CONTAGEM})


@dataclass(frozen=True)
class EspecDados:
    """P9, Seção 3."""

    tabela: EspecArquivo
    split: EspecArquivo
    particoes: Tuple[str, ...]
    particao_excluida: str
    contagens_particoes: Mapping[str, Mapping[str, int]]
    contagem_base: Mapping[str, int]
    ordem_linhas: Tuple[str, str]
    coluna_nodulo: str = COLUNA_NODULO
    coluna_paciente: str = COLUNA_PACIENTE
    coluna_alvo: str = COLUNA_ALVO

    def __post_init__(self) -> None:
        _instancia(self.tabela, EspecArquivo, "dados.tabela")
        _instancia(self.split, EspecArquivo, "dados.split")
        _tupla_textos(self.particoes, "particoes")
        _exigir(self.particao_excluida == "test", "a partição excluída deve ser 'test'")
        _exigir(self.particao_excluida not in self.particoes, "o teste não pode estar entre as partições lidas")
        _exigir(isinstance(self.contagens_particoes, Mapping)
                and tuple(self.contagens_particoes) == self.particoes,
                "contagens_particoes deve cobrir as partições lidas, na mesma ordem")
        contagens = {p: _contagem(self.contagens_particoes[p], f"partição {p}") for p in self.particoes}
        object.__setattr__(self, "contagens_particoes", MappingProxyType(contagens))
        base = _contagem(self.contagem_base, "contagem_base")
        for chave in _CHAVES_CONTAGEM:
            _exigir(sum(c[chave] for c in contagens.values()) == base[chave],
                    f"contagem_base.{chave} difere da soma das partições")
        object.__setattr__(self, "contagem_base", base)
        _exigir(self.ordem_linhas == (COLUNA_NODULO, "mergesort"), "ordem das linhas deve ser (nodulo_id, mergesort)")


@dataclass(frozen=True)
class EspecFaixa:
    """Desfecho F: 12 ≤ D < 30 mm (P9, Seção 7.2; D2 pendente de confirmação documental)."""

    variavel: str
    inferior: float
    superior: float
    inclui_inferior: bool
    inclui_superior: bool
    unidade: str
    contagem_esperada: Mapping[str, int]

    def __post_init__(self) -> None:
        _texto(self.variavel, "faixa.variavel")
        _exigir(_real(self.inferior, "faixa.inferior") < _real(self.superior, "faixa.superior"),
                "faixa: inferior deve ser menor que superior")
        _exigir(self.inclui_inferior is True and self.inclui_superior is False,
                "faixa deve ser semiaberta [inferior, superior)")
        _texto(self.unidade, "faixa.unidade")
        object.__setattr__(self, "contagem_esperada", _contagem(self.contagem_esperada, "faixa.contagem_esperada"))

    def contem(self, valor: float) -> bool:
        return self.inferior <= valor < self.superior


@dataclass(frozen=True)
class EspecDobras:
    """P9, Seção 4; sementes da decisão D12."""

    n_dobras_externas: int
    n_repeticoes: int
    semente_externa_base: int
    n_dobras_internas: int
    semente_interna_base: int
    semente_interna_passo_repeticao: int
    embaralhar: bool
    estratificacao: str
    grupo: str
    divisor: str = "sklearn.model_selection.StratifiedGroupKFold"
    avisos_do_divisor_sao_erro: bool = True

    def __post_init__(self) -> None:
        _inteiro(self.n_dobras_externas, "n_dobras_externas", minimo=2)
        _inteiro(self.n_repeticoes, "n_repeticoes", minimo=1)
        _inteiro(self.semente_externa_base, "semente_externa_base", minimo=0)
        _inteiro(self.n_dobras_internas, "n_dobras_internas", minimo=2)
        _inteiro(self.semente_interna_base, "semente_interna_base", minimo=0)
        _inteiro(self.semente_interna_passo_repeticao, "semente_interna_passo_repeticao", minimo=1)
        _exigir(self.embaralhar is True, "as dobras devem ser embaralhadas (shuffle=True)")
        _exigir(self.estratificacao == COLUNA_ALVO, "estratificação só pelo alvo (P9, D8)")
        _exigir(self.grupo == COLUNA_PACIENTE, "agrupamento por patient_id")
        _exigir(self.avisos_do_divisor_sao_erro is True, "avisos do divisor devem ser tratados como erro")
        _exigir(self.semente_interna_passo_repeticao >= self.n_dobras_externas,
                "passo das sementes internas menor que o número de dobras externas: sementes colidem")
        externas, internas = set(self.sementes_externas()), set(self.sementes_internas())
        _exigir(len(internas) == self.n_repeticoes * self.n_dobras_externas, "sementes internas repetidas")
        _exigir(not externas & internas, "sementes externas e internas colidem")

    def sementes_externas(self) -> Tuple[int, ...]:
        return tuple(self.semente_externa_base + r for r in range(self.n_repeticoes))

    def semente_interna(self, r: int, k: int) -> int:
        _exigir(type(r) is int and 0 <= r < self.n_repeticoes, f"repetição fora de 0…{self.n_repeticoes - 1}")
        _exigir(type(k) is int and 0 <= k < self.n_dobras_externas, f"dobra fora de 0…{self.n_dobras_externas - 1}")
        return self.semente_interna_base + self.semente_interna_passo_repeticao * r + k

    def sementes_internas(self) -> Tuple[int, ...]:
        return tuple(self.semente_interna_base + self.semente_interna_passo_repeticao * r + k
                     for r in range(self.n_repeticoes) for k in range(self.n_dobras_externas))


@dataclass(frozen=True)
class EspecAjusteInterno:
    """P9, Seção 5.2."""

    busca: str
    scoring: str
    criterio: str
    tolerancia_empate: float
    desempate: str
    reajuste_no_treino_externo: bool
    error_score: str
    return_train_score: bool
    n_jobs: int
    registrar_borda_da_grade: bool

    def __post_init__(self) -> None:
        _exigir(self.busca == "sklearn.model_selection.GridSearchCV", "busca deve ser GridSearchCV")
        _exigir(self.scoring == "roc_auc", "scoring interno deve ser roc_auc")
        _exigir(self.criterio == "media_auc_dobras_internas", "critério interno inválido")
        _exigir(0 < _real(self.tolerancia_empate, "tolerancia_empate") < 1e-6, "tolerância de empate fora de (0, 1e-6)")
        _exigir(self.desempate == "primeira_na_ordem_de_simplicidade", "desempate inválido")
        _exigir(self.reajuste_no_treino_externo is True, "reajuste no treino externo é obrigatório")
        _exigir(self.error_score == "raise", "error_score deve ser 'raise'")
        _exigir(self.return_train_score is False, "return_train_score deve ser False")
        _exigir(_inteiro(self.n_jobs, "n_jobs") == 1, "n_jobs do ajuste interno deve ser 1")
        _exigir(self.registrar_borda_da_grade is True, "a borda da grade deve ser registrada")


ESTADOS: Tuple[str, ...] = ("FT", "AVISO", "DD", "RI", "NE", "RD", "IQ")


@dataclass(frozen=True)
class EspecValidade:
    """P9, Seções 6 e 7.3; decisão D3."""

    min_positivos: int
    min_negativos: int
    contagem: str
    min_dobras_validas_por_repeticao: int
    min_repeticoes_validas: int
    max_fracao_descarte_bootstrap: float
    estados: Tuple[str, ...]
    imputacao_de_auc: bool = False

    def __post_init__(self) -> None:
        _inteiro(self.min_positivos, "min_positivos", minimo=1)
        _inteiro(self.min_negativos, "min_negativos", minimo=1)
        _exigir(self.contagem == "nodulos_distintos", "a contagem de validade é por nódulos distintos")
        _inteiro(self.min_dobras_validas_por_repeticao, "min_dobras_validas_por_repeticao", minimo=1)
        _inteiro(self.min_repeticoes_validas, "min_repeticoes_validas", minimo=1)
        _exigir(0 < _real(self.max_fracao_descarte_bootstrap, "max_fracao_descarte_bootstrap") < 1,
                "max_fracao_descarte_bootstrap fora de (0, 1)")
        _exigir(self.estados == ESTADOS, f"estados devem ser {ESTADOS}")
        _exigir(self.imputacao_de_auc is False, "AUC nunca é imputada")


@dataclass(frozen=True)
class EspecDesfecho:
    codigo: str
    descricao: str
    recorte: Optional[str]

    def __post_init__(self) -> None:
        _exigir(self.codigo in ("G", "F"), f"desfecho desconhecido: {self.codigo!r}")
        _texto(self.descricao, "desfecho.descricao")
        _exigir((self.codigo == "F") == (self.recorte == "faixa"), "só o desfecho F recorta pela faixa")


@dataclass(frozen=True)
class EspecMetricas:
    """P9, Seção 7."""

    principal: str
    unidade: str
    agregacao: str
    desfechos: Tuple[EspecDesfecho, ...]
    descritivas_probabilisticas: Tuple[str, ...]
    descritivas_regra_nativa: Tuple[str, ...]
    corte_log_loss: float
    limiar_probabilidade: float
    limiar_decisao_nativa: float
    desfecho_das_descritivas: str

    def __post_init__(self) -> None:
        _exigir(self.principal == "roc_auc", "métrica principal deve ser roc_auc")
        _exigir(self.unidade == "dobra_externa", "AUC calculada por dobra externa")
        _exigir(self.agregacao == "media_das_dobras_validas_e_media_das_repeticoes_validas", "agregação inválida")
        _tupla_de(self.desfechos, EspecDesfecho, "metricas.desfechos")
        _exigir(tuple(d.codigo for d in self.desfechos) == ("G", "F"), "desfechos devem ser (G, F)")
        _tupla_textos(self.descritivas_probabilisticas, "descritivas_probabilisticas")
        _tupla_textos(self.descritivas_regra_nativa, "descritivas_regra_nativa")
        _exigir(0 < _real(self.corte_log_loss, "corte_log_loss") < 1e-6, "corte do log-loss fora de (0, 1e-6)")
        _exigir(_real(self.limiar_probabilidade, "limiar_probabilidade") == 0.5, "limiar de probabilidade deve ser 0,5")
        _exigir(_real(self.limiar_decisao_nativa, "limiar_decisao_nativa") == 0.0, "regra nativa do SVM: decision_function ≥ 0")
        _exigir(self.desfecho_das_descritivas == "G", "métricas descritivas no desfecho G")


@dataclass(frozen=True)
class EspecBootstrap:
    """P9, Seção 8.2; decisões D4 e D12."""

    n_replicas: int
    gerador: str
    semente: int
    unidade: str
    reajuste: bool
    metodo: str
    peso: str
    #: α dos intervalos descritivos (θ̂ e contrastes descritivos): IC de 95%.
    alfa_descritivo: float

    def __post_init__(self) -> None:
        _inteiro(self.n_replicas, "n_replicas", minimo=1000)
        _exigir(self.gerador == "numpy.random.default_rng", "gerador deve ser numpy.random.default_rng")
        _inteiro(self.semente, "semente do bootstrap", minimo=0)
        _exigir(self.unidade == COLUNA_PACIENTE, "reamostragem por paciente")
        _exigir(self.reajuste is False, "o bootstrap não reajusta modelos")
        _exigir(self.metodo == "percentil", "IC por percentis")
        _exigir(self.peso == "multiplicidade_do_paciente", "peso pela multiplicidade do paciente")
        _exigir(0 < _real(self.alfa_descritivo, "alfa_descritivo") < 1, "α descritivo fora de (0, 1)")

    def percentis_descritivos(self) -> Tuple[float, float]:
        cauda = 100 * self.alfa_descritivo / 2
        return (cauda, 100 - cauda)


@dataclass(frozen=True)
class EspecContrastes:
    """P9, Seção 9; decisão D4."""

    comparador: str
    desafiantes: Tuple[str, ...]
    desfechos: Tuple[str, ...]
    alfa: float
    correcao: str
    descritivos: Tuple[Tuple[str, str], ...]

    def __post_init__(self) -> None:
        _texto(self.comparador, "comparador")
        _tupla_textos(self.desafiantes, "desafiantes")
        _exigir(self.comparador not in self.desafiantes, "o comparador não pode ser desafiante")
        _exigir(self.desfechos == ("G", "F"), "a família cobre os desfechos (G, F)")
        _exigir(0 < _real(self.alfa, "alfa") < 1, "alfa fora de (0, 1)")
        _exigir(self.correcao == "bonferroni", "correção deve ser Bonferroni")
        _exigir(isinstance(self.descritivos, tuple) and all(
            isinstance(par, tuple) and len(par) == 2 and par[0] != par[1] for par in self.descritivos),
            "descritivos devem ser pares distintos")
        familia = {(a, self.comparador) for a in self.desafiantes}
        _exigir(not familia & {tuple(p) for p in self.descritivos}, "contraste descritivo repete a família")

    @property
    def n_familia(self) -> int:
        return len(self.desafiantes) * len(self.desfechos)

    def percentis_familia(self) -> Tuple[float, float]:
        cauda = 100 * self.alfa / (2 * self.n_familia)
        return (cauda, 100 - cauda)


@dataclass(frozen=True)
class EspecSelecao:
    """P9, Seção 10; decisão D5. δ é operacional, sem interpretação clínica."""

    padrao: str
    delta: float
    natureza_delta: str
    fracao_min_repeticoes_positivas: float
    delta_minimo_faixa: float
    exige_faixa_estimavel: bool
    exige_g_estimavel: bool
    exige_sem_falha_tecnica: bool
    exige_sem_aviso_convergencia: bool
    margem_desempate: float
    ordem_simplicidade: Tuple[str, ...]
    ic_e_criterio: bool = False
    descarte_bootstrap_e_criterio: bool = False

    def __post_init__(self) -> None:
        _texto(self.padrao, "padrao")
        _exigir(_real(self.delta, "delta") > 0, "δ deve ser positivo")
        _exigir(self.natureza_delta == "operacional_sem_interpretacao_clinica",
                "δ é critério operacional, sem interpretação clínica (P9, Seção 10.2)")
        _exigir(0 < _real(self.fracao_min_repeticoes_positivas, "fracao_min_repeticoes_positivas") <= 1,
                "fração mínima de repetições fora de (0, 1]")
        _real(self.delta_minimo_faixa, "delta_minimo_faixa")
        for nome in ("exige_faixa_estimavel", "exige_g_estimavel", "exige_sem_falha_tecnica",
                     "exige_sem_aviso_convergencia"):
            _exigir(getattr(self, nome) is True, f"C3/C4: {nome} deve ser True")
        _exigir(_real(self.margem_desempate, "margem_desempate") >= 0, "margem de desempate negativa")
        _tupla_textos(self.ordem_simplicidade, "ordem_simplicidade")
        _exigir(self.ordem_simplicidade[0] == self.padrao, "a ordem de simplicidade começa pelo padrão")
        _exigir(self.ic_e_criterio is False, "o IC não é critério de seleção (P9, Seção 10.1)")
        _exigir(self.descarte_bootstrap_e_criterio is False,
                "a taxa de descarte do bootstrap não é critério de seleção (P9, D3 e D5)")


@dataclass(frozen=True)
class EspecFigura:
    codigo: str
    conteudo: str
    intervalo: str

    def __post_init__(self) -> None:
        _exigir(re.fullmatch(r"F[1-8]", self.codigo or "") is not None, f"figura inválida: {self.codigo!r}")
        _texto(self.conteudo, f"{self.codigo}.conteudo")
        _texto(self.intervalo, f"{self.codigo}.intervalo")
        if self.codigo == "F1":
            _exigir(self.intervalo == "ic95_descritivo_sem_correcao",
                    "F1 usa IC de 95% descritivo; Bonferroni só nos contrastes (F2)")
        if self.codigo == "F5":
            _exigir(self.intervalo == "dispersao_entre_dobras_nao_e_intervalo_de_confianca",
                    "a faixa de F5 é dispersão entre dobras, não intervalo de confiança")


@dataclass(frozen=True)
class EspecSaidas:
    """P9, Seções 11, 12 e 14.5."""

    figuras: Tuple[EspecFigura, ...]
    formatos_figura: Tuple[str, ...]
    dpi: int
    paleta: str
    pontos_roc: int
    decis_calibracao: int
    min_nodulos_por_decil: int
    artefatos_versionados: Tuple[str, ...]
    artefatos_locais: Tuple[str, ...]
    vocabulario_vetado: Tuple[str, ...]
    padrao_varredura: str

    def __post_init__(self) -> None:
        _tupla_de(self.figuras, EspecFigura, "saidas.figuras")
        _exigir(tuple(f.codigo for f in self.figuras) == tuple(f"F{i}" for i in range(1, 9)),
                "figuras obrigatórias F1…F8, nessa ordem")
        _exigir(self.formatos_figura == ("png", "svg"), "formatos das figuras: png e svg")
        _exigir(_inteiro(self.dpi, "dpi") == 300, "figuras a 300 dpi")
        _exigir(self.paleta == "okabe_ito", "paleta Okabe–Ito")
        _inteiro(self.pontos_roc, "pontos_roc", minimo=2)
        _inteiro(self.decis_calibracao, "decis_calibracao", minimo=2)
        _inteiro(self.min_nodulos_por_decil, "min_nodulos_por_decil", minimo=1)
        _tupla_textos(self.artefatos_versionados, "artefatos_versionados")
        _tupla_textos(self.artefatos_locais, "artefatos_locais")
        _tupla_textos(self.vocabulario_vetado, "vocabulario_vetado")
        _texto(self.padrao_varredura, "padrao_varredura")


@dataclass(frozen=True)
class EspecExecucao:
    """P9, Seções 13, 14.3 e 15."""

    modos: Tuple[str, ...]
    codigos_saida: Mapping[str, int]
    teto_alarme_horas: float
    altera_parametros_por_tempo: bool
    n_jobs_estimador: int
    threads_blas: int
    ambiente: Mapping[str, str]
    registros_exigidos: Tuple[str, ...]
    implementacao_metricas_externas: bool = False

    def __post_init__(self) -> None:
        _exigir(self.modos == ("configuracao", "implementacao", "definitivo", "verificar"), "modos do coordenador")
        _exigir(isinstance(self.codigos_saida, Mapping) and dict(self.codigos_saida) == {
            "concluido": 0, "bloqueio_aprovacao": 2, "falha_tecnica": 3, "integridade": 4,
            "privacidade_ou_vocabulario": 5}, "códigos de saída diferentes da P9")
        object.__setattr__(self, "codigos_saida", MappingProxyType(dict(self.codigos_saida)))
        _exigir(_real(self.teto_alarme_horas, "teto_alarme_horas") > 0, "teto de alarme deve ser positivo")
        _exigir(self.altera_parametros_por_tempo is False,
                "nenhum parâmetro muda por tempo; só por adendo aprovado (P9, Seção 15)")
        _exigir(_inteiro(self.n_jobs_estimador, "n_jobs_estimador") == 1, "n_jobs do estimador deve ser 1")
        _exigir(_inteiro(self.threads_blas, "threads_blas") == 1, "BLAS limitado a uma thread")
        _exigir(isinstance(self.ambiente, Mapping) and set(self.ambiente) == {
            "python", "scikit-learn", "numpy", "pandas", "scipy", "matplotlib", "xgboost"},
            "ambiente incompleto")
        object.__setattr__(self, "ambiente",
                           MappingProxyType({k: _texto(v, f"ambiente.{k}") for k, v in self.ambiente.items()}))
        _exigir(self.registros_exigidos == ("P9", "S4_E3_REGISTRADA"), "registros exigidos: P9 e S4_E3_REGISTRADA")
        _exigir(self.implementacao_metricas_externas is False,
                "o modo implementacao não produz métrica externa (P9, Seção 14.3)")


@dataclass(frozen=True)
class EspecReferencias:
    preregistro: str
    versao_preregistro: str
    arquivo_configuracao: str
    emenda: str
    commit_emenda: str
    arquivos_s4: Mapping[str, str]
    simbolos_s4: Mapping[str, Tuple[str, ...]]

    def __post_init__(self) -> None:
        _exigir(self.preregistro == PREREGISTRO, "pré-registro inesperado")
        _texto(self.versao_preregistro, "versao_preregistro")
        _exigir(self.arquivo_configuracao == ARQUIVO_CONFIGURACAO, "arquivo de configuração inesperado")
        _texto(self.emenda, "emenda")
        _exigir(type(self.commit_emenda) is str and _COMMIT.fullmatch(self.commit_emenda) is not None,
                "commit da emenda deve ter 40 hex")
        _instancia(self.arquivos_s4, collections.abc.Mapping, "arquivos_s4")
        _instancia(self.simbolos_s4, collections.abc.Mapping, "simbolos_s4")
        _exigir(dict(self.arquivos_s4) == dict(s4.ARQUIVOS_S4), "arquivos da S4 diferentes dos fixados no adaptador")
        _exigir({k: tuple(v) for k, v in self.simbolos_s4.items()} == dict(s4.SIMBOLOS_S4),
                "símbolos da S4 diferentes dos importados pelo adaptador")
        for nomes in self.simbolos_s4.values():
            _exigir(all(not n.startswith("_") for n in nomes), "símbolo privado da S4")
        object.__setattr__(self, "arquivos_s4", MappingProxyType(dict(self.arquivos_s4)))
        object.__setattr__(self, "simbolos_s4", MappingProxyType({k: tuple(v) for k, v in self.simbolos_s4.items()}))


# --------------------------------------------------------------------------- #
# Configuração completa
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConfiguracaoBenchmark:
    versao_esquema: int
    referencias: EspecReferencias
    dados: EspecDados
    caracteristicas: Tuple[str, ...]
    faixa: EspecFaixa
    dobras: EspecDobras
    modelos: Tuple[EspecModelo, ...]
    ajuste_interno: EspecAjusteInterno
    validade: EspecValidade
    metricas: EspecMetricas
    bootstrap: EspecBootstrap
    contrastes: EspecContrastes
    selecao: EspecSelecao
    saidas: EspecSaidas
    execucao: EspecExecucao

    def __post_init__(self) -> None:
        _exigir(self.versao_esquema == VERSAO_ESQUEMA, f"versao_esquema deve ser {VERSAO_ESQUEMA}")
        for nome, tipo in (("referencias", EspecReferencias), ("dados", EspecDados), ("faixa", EspecFaixa),
                           ("dobras", EspecDobras), ("ajuste_interno", EspecAjusteInterno),
                           ("validade", EspecValidade), ("metricas", EspecMetricas),
                           ("bootstrap", EspecBootstrap), ("contrastes", EspecContrastes),
                           ("selecao", EspecSelecao), ("saidas", EspecSaidas), ("execucao", EspecExecucao)):
            _instancia(getattr(self, nome), tipo, nome)
        _tupla_de(self.modelos, EspecModelo, "modelos")
        _exigir(self.caracteristicas == tuple(s4.FEATURES_M1),
                "características devem ser exatamente as três do baseline, na ordem de FEATURES_M1")
        _exigir(self.faixa.variavel in self.caracteristicas, "a variável da faixa deve ser uma das características")
        for chave in _CHAVES_CONTAGEM:
            _exigir(self.faixa.contagem_esperada[chave] <= self.dados.contagem_base[chave],
                    f"faixa.{chave} maior que a base")
        _exigir(self.validade.min_dobras_validas_por_repeticao <= self.dobras.n_dobras_externas,
                "mínimo de dobras válidas maior que o número de dobras")
        _exigir(self.validade.min_repeticoes_validas <= self.dobras.n_repeticoes,
                "mínimo de repetições válidas maior que o número de repetições")

        codigos = tuple(m.codigo for m in self.modelos)
        _exigir(len(set(codigos)) == len(codigos), "códigos de modelo repetidos")
        papeis = {m.codigo: m.papel for m in self.modelos}
        _exigir(list(papeis.values()).count("referencia_historica") == 1, "exatamente uma referência histórica")
        _exigir([c for c, p in papeis.items() if p == "comparador"] == [self.contrastes.comparador],
                "o comparador dos contrastes deve ser o único modelo com papel 'comparador'")
        _exigir(set(self.contrastes.desafiantes) == {c for c, p in papeis.items() if p == "desafiante"},
                "os desafiantes dos contrastes devem ser os modelos com papel 'desafiante'")
        for par in self.contrastes.descritivos:
            _exigir(set(par) <= set(codigos), f"contraste descritivo com modelo desconhecido: {par}")
        _exigir(self.selecao.padrao == self.contrastes.comparador, "o padrão da seleção é o comparador")
        _exigir(set(self.selecao.ordem_simplicidade) == {self.contrastes.comparador, *self.contrastes.desafiantes},
                "a ordem de simplicidade deve conter o comparador e todos os desafiantes")
        n_hiper = [self._modelo(c).n_hiperparametros for c in self.selecao.ordem_simplicidade]
        _exigir(n_hiper == sorted(n_hiper), "a ordem de simplicidade não pode ter menos hiperparâmetros depois de mais")

        referencia = self._modelo([c for c, p in papeis.items() if p == "referencia_historica"][0])
        comparador = self._modelo(self.contrastes.comparador)
        _exigir(referencia.estimador == comparador.estimador, "referência e comparador são o mesmo estimador")
        _exigir(referencia.fixos.get("C") == 1.0, "a referência histórica usa C = 1 (Seção 6.1 da S4)")
        _exigir({k: v for k, v in referencia.fixos.items() if k != "C"} == dict(comparador.fixos),
                "comparador = referência sem C fixo")
        _exigir(1.0 in comparador.valores_de("C"), "a grade do comparador inclui C = 1")
        for m in self.modelos:
            for nome in ("n_jobs",):
                if nome in m.fixos:
                    _exigir(m.fixos[nome] == self.execucao.n_jobs_estimador, f"{m.codigo}: n_jobs ≠ 1")

    def _modelo(self, codigo: str) -> EspecModelo:
        for m in self.modelos:
            if m.codigo == codigo:
                return m
        raise ConfiguracaoInvalida(f"modelo inexistente: {codigo!r}")

    def modelo(self, codigo: str) -> EspecModelo:
        return self._modelo(codigo)

    def ajustes_por_dobra_externa(self) -> int:
        return sum(m.ajustes_por_dobra_externa(self.dobras.n_dobras_internas) for m in self.modelos)

    def ajustes_totais(self) -> int:
        return self.ajustes_por_dobra_externa() * self.dobras.n_dobras_externas * self.dobras.n_repeticoes

    def n_unidades(self) -> int:
        return self.dobras.n_repeticoes * self.dobras.n_dobras_externas * len(self.modelos)


# --------------------------------------------------------------------------- #
# Construção da configuração pré-registrada
# --------------------------------------------------------------------------- #

_LOGREG = "sklearn.linear_model.LogisticRegression"
_FIXOS_LOGREG = {"penalty": "l2", "solver": "lbfgs", "class_weight": None, "max_iter": 1000, "random_state": 42}


def _arquivo(verificado: Any) -> EspecArquivo:
    return EspecArquivo(s4.caminho_relativo_repo(verificado.caminho), verificado.sha256, verificado.sha256_lf)


def construir_configuracao() -> ConfiguracaoBenchmark:
    """A configuração exatamente como pré-registrada na P9 v1.0."""
    modelos = (
        EspecModelo("lr_fixa", "referencia_historica", _LOGREG, True, {"C": 1.0, **_FIXOS_LOGREG}, (),
                    "predict_proba", True),
        EspecModelo("lr_tuning", "comparador", _LOGREG, True, dict(_FIXOS_LOGREG),
                    (ParametroGrade("C", GradeLog(-3, 3, 13)),), "predict_proba", True),
        EspecModelo("svm", "desafiante", "sklearn.svm.SVC", True,
                    {"kernel": "rbf", "probability": False, "class_weight": None, "random_state": 42},
                    (ParametroGrade("C", (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)),
                     ParametroGrade("gamma", (0.01, 0.03, 0.1, 0.3, 1.0))),
                    "decision_function", False),
        EspecModelo("rf", "desafiante", "sklearn.ensemble.RandomForestClassifier", False,
                    {"n_estimators": 500, "criterion": "gini", "bootstrap": True, "class_weight": None,
                     "n_jobs": 1, "random_state": 42},
                    (ParametroGrade("min_samples_leaf", (20, 10, 5, 3, 1)),
                     ParametroGrade("max_features", (1, 2, 3))),
                    "predict_proba", True),
        EspecModelo("xgb", "desafiante", "xgboost.XGBClassifier", False,
                    {"objective": "binary:logistic", "tree_method": "hist", "subsample": 0.8,
                     "colsample_bytree": 1.0, "reg_lambda": 1.0, "n_jobs": 1, "random_state": 42},
                    (ParametroGrade("max_depth", (1, 2, 3, 4)),
                     ParametroGrade("n_estimators", (100, 300)),
                     ParametroGrade("learning_rate", (0.03, 0.1)),
                     ParametroGrade("min_child_weight", (5, 1))),
                    "predict_proba", True),
    )
    return ConfiguracaoBenchmark(
        versao_esquema=VERSAO_ESQUEMA,
        referencias=EspecReferencias(
            preregistro=PREREGISTRO,
            versao_preregistro=VERSAO_PREREGISTRO,
            arquivo_configuracao=ARQUIVO_CONFIGURACAO,
            emenda="docs/sprint4/emenda_01_auditoria_teste.md",
            commit_emenda="573bc1729f7dec33c73675ccc9c48c97d5e50c5a",
            arquivos_s4=s4.ARQUIVOS_S4,
            simbolos_s4=s4.SIMBOLOS_S4,
        ),
        dados=EspecDados(
            tabela=_arquivo(s4.TABELA_OFICIAL),
            split=_arquivo(s4.SPLIT_OFICIAL),
            particoes=("train", "validation"),
            particao_excluida="test",
            contagens_particoes={
                "train": {"pacientes": 291, "nodulos": 417, "classe_0": 209, "classe_1": 208},
                "validation": {"pacientes": 62, "nodulos": 92, "classe_0": 49, "classe_1": 43},
            },
            contagem_base={"pacientes": 353, "nodulos": 509, "classe_0": 258, "classe_1": 251},
            ordem_linhas=(COLUNA_NODULO, "mergesort"),
        ),
        caracteristicas=tuple(s4.FEATURES_M1),
        faixa=EspecFaixa(
            variavel="original_shape_Maximum3DDiameter", inferior=12.0, superior=30.0,
            inclui_inferior=True, inclui_superior=False, unidade="mm",
            contagem_esperada={"pacientes": 187, "nodulos": 226, "classe_0": 91, "classe_1": 135},
        ),
        dobras=EspecDobras(
            n_dobras_externas=5, n_repeticoes=10, semente_externa_base=42,
            n_dobras_internas=5, semente_interna_base=1000, semente_interna_passo_repeticao=10,
            embaralhar=True, estratificacao=COLUNA_ALVO, grupo=COLUNA_PACIENTE,
        ),
        modelos=modelos,
        ajuste_interno=EspecAjusteInterno(
            busca="sklearn.model_selection.GridSearchCV", scoring="roc_auc",
            criterio="media_auc_dobras_internas", tolerancia_empate=1e-9,
            desempate="primeira_na_ordem_de_simplicidade", reajuste_no_treino_externo=True,
            error_score="raise", return_train_score=False, n_jobs=1, registrar_borda_da_grade=True,
        ),
        validade=EspecValidade(
            min_positivos=5, min_negativos=5, contagem="nodulos_distintos",
            min_dobras_validas_por_repeticao=4, min_repeticoes_validas=8,
            max_fracao_descarte_bootstrap=0.01, estados=ESTADOS,
        ),
        metricas=EspecMetricas(
            principal="roc_auc", unidade="dobra_externa",
            agregacao="media_das_dobras_validas_e_media_das_repeticoes_validas",
            desfechos=(EspecDesfecho("G", "AUC geral por dobra externa", None),
                       EspecDesfecho("F", "AUC por dobra externa na faixa de 12 a 30 mm", "faixa")),
            descritivas_probabilisticas=("brier", "log_loss", "intercepto_calibracao", "inclinacao_calibracao",
                                         "sensibilidade_p_0_5", "especificidade_p_0_5"),
            descritivas_regra_nativa=("sensibilidade_regra_nativa", "especificidade_regra_nativa"),
            corte_log_loss=1e-15, limiar_probabilidade=0.5, limiar_decisao_nativa=0.0,
            desfecho_das_descritivas="G",
        ),
        bootstrap=EspecBootstrap(
            n_replicas=10_000, gerador="numpy.random.default_rng", semente=42, unidade=COLUNA_PACIENTE,
            reajuste=False, metodo="percentil", peso="multiplicidade_do_paciente", alfa_descritivo=0.05,
        ),
        contrastes=EspecContrastes(
            comparador="lr_tuning", desafiantes=("rf", "xgb", "svm"), desfechos=("G", "F"),
            alfa=0.05, correcao="bonferroni",
            descritivos=(("lr_tuning", "lr_fixa"), ("rf", "xgb"), ("rf", "svm"), ("xgb", "svm")),
        ),
        selecao=EspecSelecao(
            padrao="lr_tuning", delta=0.02, natureza_delta="operacional_sem_interpretacao_clinica",
            fracao_min_repeticoes_positivas=0.8, delta_minimo_faixa=0.0, exige_faixa_estimavel=True,
            exige_g_estimavel=True, exige_sem_falha_tecnica=True, exige_sem_aviso_convergencia=True,
            margem_desempate=0.005, ordem_simplicidade=("lr_tuning", "svm", "rf", "xgb"),
        ),
        saidas=EspecSaidas(
            figuras=(
                EspecFigura("F1", "auc_por_modelo_e_desfecho", "ic95_descritivo_sem_correcao"),
                EspecFigura("F2", "floresta_dos_seis_contrastes", "bonferroni_e_ic95_e_amplitude_entre_repeticoes"),
                EspecFigura("F3", "delta_por_dobra", "sem_intervalo"),
                EspecFigura("F4", "frequencia_das_configuracoes_escolhidas", "sem_intervalo"),
                EspecFigura("F5", "roc_media_vertical_desfecho_g",
                            "dispersao_entre_dobras_nao_e_intervalo_de_confianca"),
                EspecFigura("F6", "calibracao_por_decis_r0_desfecho_g", "sem_intervalo"),
                EspecFigura("F7", "denominadores_desfecho_f_por_dobra", "sem_intervalo"),
                EspecFigura("F8", "histograma_delta_bootstrap", "percentis_do_ic"),
            ),
            formatos_figura=("png", "svg"), dpi=300, paleta="okabe_ito", pontos_roc=101,
            decis_calibracao=10, min_nodulos_por_decil=10,
            artefatos_versionados=(
                "config_benchmark.json", "plano_folds.json", "denominadores.csv", "auc_por_fold.csv",
                "auc_por_repeticao.csv", "contrastes.csv", "bootstrap_resumo.json", "bootstrap_delta.csv",
                "hiperparametros_escolhidos.csv", "metricas_descritivas.csv", "selecao_algoritmo.json",
                "relatorio_benchmark.json", "resumo_benchmark.md", "relatorio_execucao.json",
                "manifesto_benchmark.json",
            ),
            artefatos_locais=("checkpoints", "predicoes_fora_da_dobra", "atribuicao_de_dobras",
                              "ajuste_interno_detalhado"),
            vocabulario_vetado=("significativo", "significância", "superior", "comprovado", "demonstrado",
                                "estatisticamente", "p-valor"),
            padrao_varredura=r"LIDC-IDRI-\d{4}",
        ),
        execucao=EspecExecucao(
            modos=("configuracao", "implementacao", "definitivo", "verificar"),
            codigos_saida={"concluido": 0, "bloqueio_aprovacao": 2, "falha_tecnica": 3, "integridade": 4,
                           "privacidade_ou_vocabulario": 5},
            teto_alarme_horas=3.0, altera_parametros_por_tempo=False, n_jobs_estimador=1, threads_blas=1,
            ambiente={"python": "3.9.13", "scikit-learn": "1.3.2", "numpy": "1.23.5", "pandas": "1.5.3",
                      "scipy": "1.10.1", "matplotlib": "3.7.3", "xgboost": "2.1.4"},
            registros_exigidos=("P9", "S4_E3_REGISTRADA"),
        ),
    )


CONFIGURACAO: ConfiguracaoBenchmark = construir_configuracao()

# --------------------------------------------------------------------------- #
# Serialização canônica e hash
# --------------------------------------------------------------------------- #


def _jsonavel(obj: Any) -> Any:
    if isinstance(obj, GradeLog):
        return {"tipo": "logspace", "inicio": obj.inicio, "fim": obj.fim, "n": obj.n}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonavel(getattr(obj, f.name)) for f in fields(obj) if not f.name.startswith("_")}
    if isinstance(obj, Mapping):
        return {_texto(k, "chave"): _jsonavel(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_jsonavel(v) for v in obj]
    if obj is None or type(obj) in (bool, int, str):
        return obj
    if type(obj) is float:
        return _real(obj, "valor")
    raise ConfiguracaoInvalida(f"valor não serializável: {type(obj).__name__}")


def conteudo_canonico(configuracao: ConfiguracaoBenchmark = CONFIGURACAO) -> Dict[str, Any]:
    """A configuração como JSON simples (dicionários, listas e escalares)."""
    if not isinstance(configuracao, ConfiguracaoBenchmark):
        raise TypeError("configuracao deve ser ConfiguracaoBenchmark")
    return _jsonavel(configuracao)


def texto_canonico(configuracao: ConfiguracaoBenchmark = CONFIGURACAO) -> str:
    return s4.json_canonico(conteudo_canonico(configuracao))


def sha256_configuracao(configuracao: ConfiguracaoBenchmark = CONFIGURACAO) -> str:
    """``valor_sha256`` do registro P9: SHA-256 do JSON canônico do conteúdo."""
    return s4.sha256_canonico(conteudo_canonico(configuracao))


def bytes_arquivo(configuracao: ConfiguracaoBenchmark = CONFIGURACAO) -> bytes:
    """Bytes do arquivo canônico: JSON canônico, UTF-8, terminado por ``\\n``."""
    return (texto_canonico(configuracao) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- #
# Arquivo canônico: geração e conferência, sem sobrescrita
# --------------------------------------------------------------------------- #


def _sem_duplicatas(pares):
    chaves = [k for k, _ in pares]
    if len(set(chaves)) != len(chaves):
        raise ConfiguracaoDivergente("JSON com chave repetida")
    return dict(pares)


def _constante_proibida(nome: str) -> Any:
    raise ConfiguracaoDivergente(f"JSON com constante não finita: {nome}")


def ler_arquivo(caminho: Union[str, Path]) -> Dict[str, Any]:
    """Lê o arquivo exigindo a forma canônica (só CRLF → LF é tolerado)."""
    bruto = Path(caminho).read_bytes()
    normalizado = bruto.replace(b"\r\n", b"\n")
    try:
        conteudo = json.loads(normalizado.decode("utf-8"), object_pairs_hook=_sem_duplicatas,
                              parse_constant=_constante_proibida)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfiguracaoDivergente(f"{Path(caminho).name}: JSON ilegível: {exc}") from exc
    if normalizado != (s4.json_canonico(conteudo) + "\n").encode("utf-8"):
        raise ConfiguracaoDivergente(f"{Path(caminho).name}: arquivo fora da forma canônica")
    return conteudo


def conferir_arquivo(caminho: Union[str, Path],
                     configuracao: ConfiguracaoBenchmark = CONFIGURACAO) -> str:
    """Exige que o arquivo seja a configuração canônica gerada pelo código; devolve o SHA-256."""
    conteudo = ler_arquivo(caminho)
    esperado = sha256_configuracao(configuracao)
    obtido = s4.sha256_canonico(conteudo)
    if obtido != esperado:
        raise ConfiguracaoDivergente(
            f"{Path(caminho).name}: SHA-256 {obtido} difere da configuração do código ({esperado})")
    return esperado


def gerar_arquivo(destino: Union[str, Path],
                  configuracao: ConfiguracaoBenchmark = CONFIGURACAO) -> Tuple[str, bool]:
    """Grava o arquivo canônico se ausente; nunca sobrescreve.

    Devolve ``(sha256, criado)``. Se o arquivo já existir, só o confere: igual
    → ``criado=False``; diferente → ``ConfiguracaoDivergente``.
    """
    destino = Path(destino)
    sha = sha256_configuracao(configuracao)
    if destino.exists():
        return conferir_arquivo(destino, configuracao), False
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(f".{destino.name}.{os.getpid()}.tmp")
    try:
        with temporario.open("xb") as arquivo:
            arquivo.write(bytes_arquivo(configuracao))
            arquivo.flush()
            os.fsync(arquivo.fileno())
        try:
            os.link(temporario, destino)
        except FileExistsError:
            return conferir_arquivo(destino, configuracao), False
    finally:
        if temporario.exists():
            temporario.unlink()
    conferir_arquivo(destino, configuracao)
    return sha, True
