"""Acesso controlado aos dados oficiais do Cartão 4 — Sprint 4 (Incremento 2B).

Única porta de leitura de alvo e features para o pipeline corrigido. A ordem
de cada carga é sempre a mesma, e nenhum byte de dado é lido antes dos três
primeiros passos:

1. a atividade é validada por ``guardas.normalizar_atividade``. Só atividades
   ligadas a uma etapa com partições em ``config.PARTICOES_POR_ETAPA`` leem
   alvo e features; verificações estruturais e a Etapa 4 não leem; o
   piloto exploratório lê só ``train``;
2. as partições pedidas, sempre explícitas, têm de estar entre as permitidas
   para a etapa. Validação só na Etapa 3 e teste só na Etapa 5;
3. ``guardas.exigir_autorizacao`` é chamada AQUI, com a própria atividade,
   antes de qualquer leitura de arquivo. Passar o nome de uma etapa não
   basta: a carga exige as decisões da etapa aprovadas e com evidência no
   repositório. Não existe parâmetro que aceite uma autorização pronta. Para
   dados oficiais, o repositório tem de ser o do projeto: aprovações
   fictícias num repositório temporário só valem para fontes sintéticas;
4. os bytes da tabela e do split são lidos UMA vez e conferidos contra o
   SHA-256 da fonte. Todo o processamento seguinte usa esses mesmos bytes.
   Conteúdo com o hash oficial só é aceito dentro da fonte oficial, mesmo que
   tenha sido copiado para outro caminho;
5. primeiro só os identificadores (``nodulo_id``, ``patient_id``) são
   lidos, para checar as invariantes estruturais e localizar as linhas;
6. alvo e features são lidos só das linhas das partições autorizadas. As
   demais são puladas com ``skiprows`` e nunca chegam a um DataFrame. Isso só
   é seguro com uma linha física por registro, o que é verificado antes.
   Depois da leitura, a sequência de identificadores é comparada com a
   esperada.

Colunas derivadas do rótulo (``malignancy_*``, ``exclusion_reason``,
contagens de anotações) e metadados de processamento nunca são lidos, de
nenhuma partição. O pool de candidatas é o conjunto ``original_*`` sem
``original_nodule_idx`` (protocolo, Seção 5.5), na ordem do cabeçalho, e tem
de coincidir com os 107 nomes versionados na reprodução auditada.

O módulo não escreve arquivos, não ajusta modelos e não calcula métricas.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..radiomics import ids as ids_extracao
from . import config as cfg
from . import guardas

REPO_ROOT: Path = guardas.REPO_ROOT
DIRETORIO_DADOS: Path = REPO_ROOT / "data"

COLUNA_SPLIT: str = "split"
PARTICOES: Tuple[str, ...] = ("train", "validation", "test")
COLUNAS_ID: Tuple[str, str] = (cfg.COLUNA_NODULO, cfg.COLUNA_PACIENTE)
#: Nomes que nunca são features, ainda que pedidos como tal.
NAO_FEATURES: FrozenSet[str] = frozenset(
    {cfg.COLUNA_NODULO, cfg.COLUNA_PACIENTE, cfg.COLUNA_ALVO, cfg.COLUNA_PROIBIDA, COLUNA_SPLIT}
)

#: Atividade → etapa cujas partições ela pode ler. Verificações estruturais
#: ficam de fora: não leem alvo nem features. A CV com lista fixa é uma
#: atividade da Etapa 2. O piloto exploratório reutiliza SÓ a política de
#: leitura da Etapa 1 (``train``); suas decisões vêm da própria atividade.
ETAPA_DE_LEITURA: Mapping[str, cfg.Etapa] = MappingProxyType({
    **{e.value: e for e in cfg.Etapa},
    cfg.ATIVIDADE_CV_LISTA_FIXA: cfg.Etapa.E2,
    cfg.ATIVIDADE_PILOTO_EXPLORATORIO: cfg.Etapa.E1,
})

#: Conversor de float do ``read_csv``: o PADRÃO do pandas, o mesmo do baseline
#: da Sprint 3 e da reprodução auditada. Ele pode diferir em 1 ulp da conversão
#: exata (``"round_trip"``); trocar exige decisão, porque quebra a paridade.
PRECISAO_FLOAT_CSV: Optional[str] = None

_SHA256 = re.compile(r"[0-9a-f]{64}")

# --------------------------------------------------------------------------- #
# Exceções
# --------------------------------------------------------------------------- #


class AcessoNegado(PermissionError):
    """Atividade ou partição sem permissão de leitura."""


class DadosInvalidos(ValueError):
    """Integridade ou invariante estrutural violada; nada é devolvido."""


class FeatureInvalida(ValueError):
    """Nome fora do pool de candidatas, duplicado ou identificador."""


# --------------------------------------------------------------------------- #
# Fontes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArquivoVerificado:
    """Arquivo de entrada e seus SHA-256: bruto e com CRLF → LF."""

    caminho: Path
    sha256: str
    sha256_lf: str

    def __post_init__(self) -> None:
        if not isinstance(self.caminho, Path):
            raise TypeError("caminho deve ser Path")
        for valor in (self.sha256, self.sha256_lf):
            if not (isinstance(valor, str) and _SHA256.fullmatch(valor)):
                raise ValueError(f"SHA-256 inválido para {self.caminho.name}: {valor!r}")


def _contagens(valor: Mapping[str, int], chaves: Tuple[str, ...], contexto: str) -> Mapping[str, int]:
    if not isinstance(valor, Mapping) or set(valor) != set(chaves):
        raise ValueError(f"{contexto}: chaves esperadas {chaves}")
    for chave in chaves:
        if type(valor[chave]) is not int or valor[chave] < 0:
            raise ValueError(f"{contexto}: {chave} deve ser inteiro não negativo")
    return MappingProxyType(dict(valor))


_CHAVES_COORTE = ("nodulos", "pacientes", "classe_0", "classe_1")


@dataclass(frozen=True)
class FonteDados:
    """Entradas e expectativas de uma carga. Use ``fonte_oficial()``.

    Fontes sintéticas existem só para testes. Uma fonte que aponte para
    ``data/`` ou para o split oficial tem de ser idêntica à oficial
    (``_exigir_fonte_permitida``).
    """

    tabela: ArquivoVerificado
    split: ArquivoVerificado
    nomes_pool: FrozenSet[str]
    coorte: Mapping[str, int]
    particoes: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        if not isinstance(self.nomes_pool, frozenset):
            raise TypeError("nomes_pool deve ser frozenset")
        cfg.validar_nomes(sorted(self.nomes_pool), "pool esperado")
        if len(self.nomes_pool) != cfg.N_POOL_CANDIDATAS or cfg.COLUNA_PROIBIDA in self.nomes_pool:
            raise ValueError(f"pool esperado deve ter {cfg.N_POOL_CANDIDATAS} features, "
                             f"sem {cfg.COLUNA_PROIBIDA}")
        object.__setattr__(self, "coorte", _contagens(self.coorte, _CHAVES_COORTE, "coorte"))
        if not isinstance(self.particoes, Mapping) or set(self.particoes) != set(PARTICOES):
            raise ValueError(f"partições esperadas devem ser {PARTICOES}")
        particoes = {p: _contagens(self.particoes[p], _CHAVES_COORTE, f"partição {p}") for p in PARTICOES}
        for chave in _CHAVES_COORTE:
            if sum(particoes[p][chave] for p in PARTICOES) != self.coorte[chave]:
                raise ValueError(f"soma das partições difere da coorte em {chave}")
        object.__setattr__(self, "particoes", MappingProxyType(particoes))


#: Pins das entradas oficiais, copiados de ``sha256_entradas`` em
#: ``reports/sprint4/feature_selection/reproducao_juan/relatorio_reproducao.json``
#: (commit 0ef4039), o mesmo conteúdo usado na reprodução auditada.
TABELA_OFICIAL = ArquivoVerificado(
    DIRETORIO_DADOS / "modeling_table_sprint3.csv",
    "5e92e7e860ad8edd327570c929cd263cd780be98902e57017551ad0080da60ab",
    "fd01a1daf19503e84ec40df20e66190f99b4a460180879bdb5d279bd0b1506ea",
)
SPLIT_OFICIAL = ArquivoVerificado(
    REPO_ROOT / "reports" / "sprint3" / "baseline" / "patient_split.csv",
    "ed120c2b5c0f308d41c11cfc32dae0e897a179c71d6ea803588720ba1ba4f65f",
    "812fe3ffe51f11257a9aa74529e1bfc19ed58285a11142f0e8b5739c1d1a4363",
)
#: Os 107 nomes do pool: coluna ``feature`` da frequência de seleção
#: versionada (todas as candidatas, ordenadas por frequência — por isso só o
#: CONJUNTO é usado; a ordem vem do cabeçalho da tabela).
POOL_VERSIONADO = ArquivoVerificado(
    REPO_ROOT / "reports" / "sprint4" / "feature_selection" / "reproducao_juan"
    / "frequencia_selecao_features.csv",
    "8e85ee1770f9a53b896941b22b5ebd660c2d623100a3df096d552dbd4ff38ed2",
    "8e85ee1770f9a53b896941b22b5ebd660c2d623100a3df096d552dbd4ff38ed2",
)


def _ler_bytes(caminho: Path) -> bytes:
    """Única leitura de arquivo do módulo (somente leitura)."""
    try:
        return caminho.read_bytes()
    except OSError as exc:
        raise DadosInvalidos(f"não foi possível ler {caminho.name}: {exc}") from exc


def _ler_verificado(arquivo: ArquivoVerificado) -> bytes:
    conteudo = _ler_bytes(arquivo.caminho)
    if hashlib.sha256(conteudo).hexdigest() == arquivo.sha256:
        return conteudo
    if hashlib.sha256(conteudo.replace(b"\r\n", b"\n")).hexdigest() == arquivo.sha256_lf:
        return conteudo
    raise DadosInvalidos(f"{arquivo.caminho.name}: SHA-256 diferente do fixado; o arquivo mudou")


def fonte_oficial() -> FonteDados:
    """Fonte oficial. Lê só o CSV versionado com os nomes do pool, não ``data/``."""
    texto = _ler_verificado(POOL_VERSIONADO).decode("utf-8")
    nomes = [linha["feature"] for linha in csv.DictReader(io.StringIO(texto))]
    if len(nomes) != len(set(nomes)):
        raise DadosInvalidos("nomes duplicados no pool versionado")
    return FonteDados(
        tabela=TABELA_OFICIAL,
        split=SPLIT_OFICIAL,
        nomes_pool=frozenset(nomes),
        coorte=cfg.COORTE_ESPERADA,
        particoes=cfg.PARTICOES_ESPERADAS,
    )


def _dentro_de(caminho: Path, diretorio: Path) -> bool:
    try:
        caminho.resolve().relative_to(diretorio.resolve())
        return True
    except ValueError:
        return False


def _toca_caminho_oficial(fonte: Optional[FonteDados]) -> bool:
    """Fonte padrão, ou com algum arquivo em ``data/`` ou no split oficial (sem ler nada)."""
    if fonte is None:
        return True
    if not isinstance(fonte, FonteDados):
        raise TypeError("fonte deve ser FonteDados")
    return any(
        _dentro_de(a.caminho, DIRETORIO_DADOS) or a.caminho.resolve() == SPLIT_OFICIAL.caminho.resolve()
        for a in (fonte.tabela, fonte.split)
    )


def _exigir_fonte_permitida(fonte: FonteDados) -> FonteDados:
    """Impede combinar a tabela ou o split oficiais com pins ou expectativas trocados."""
    if _toca_caminho_oficial(fonte) and fonte != fonte_oficial():
        raise AcessoNegado("fonte com arquivos oficiais tem de ser idêntica a fonte_oficial()")
    return fonte


def _conteudo_oficial(conteudo: bytes, oficial: ArquivoVerificado) -> bool:
    return (hashlib.sha256(conteudo).hexdigest() == oficial.sha256
            or hashlib.sha256(conteudo.replace(b"\r\n", b"\n")).hexdigest() == oficial.sha256_lf)


def _exigir_repo_do_projeto(repo: Path) -> None:
    """Dados oficiais só com aprovações do repositório do projeto.

    Sem isso, um repositório temporário com registros fictícios (como os dos
    testes) liberaria validação e teste reais.
    """
    try:
        mesmo = Path(repo).resolve() == REPO_ROOT.resolve()
    except (TypeError, OSError):
        mesmo = False
    if not mesmo:
        raise AcessoNegado("dados oficiais exigem as aprovações do repositório do projeto (repo=REPO_ROOT)")


# --------------------------------------------------------------------------- #
# Esquema de features e validação de S_k
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EsquemaFeatures:
    """As 107 candidatas, na ordem do cabeçalho da tabela."""

    nomes: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.nomes, tuple):
            raise TypeError("nomes deve ser tupla")
        cfg.validar_nomes(self.nomes, "esquema")
        if len(self.nomes) != cfg.N_POOL_CANDIDATAS or cfg.COLUNA_PROIBIDA in self.nomes:
            raise ValueError(f"esquema deve ter {cfg.N_POOL_CANDIDATAS} features, sem {cfg.COLUNA_PROIBIDA}")

    def validar(self, nomes: Iterable[str], contexto: str = "S_k",
                vazio_permitido: bool = True) -> Tuple[str, ...]:
        """Valida um subconjunto e o devolve NA ORDEM DO ESQUEMA.

        Recusa string solta, nome que não seja ``str``, identificador ou alvo,
        nome fora do pool e duplicata. A ordem de entrada não importa: o
        resultado é determinístico.
        """
        if isinstance(nomes, (str, bytes)):
            raise TypeError(f"{contexto}: informe uma coleção de nomes, não uma string")
        nomes = tuple(nomes)
        pool = set(self.nomes)
        for nome in nomes:
            if type(nome) is not str:
                raise FeatureInvalida(f"{contexto}: nome inválido {nome!r}")
            if nome in NAO_FEATURES:
                raise FeatureInvalida(f"{contexto}: {nome} é identificador ou alvo, não feature")
            if nome not in pool:
                raise FeatureInvalida(f"{contexto}: {nome} não está entre as {len(pool)} candidatas")
        if len(set(nomes)) != len(nomes):
            raise FeatureInvalida(f"{contexto}: nomes duplicados")
        if not nomes and not vazio_permitido:
            raise FeatureInvalida(f"{contexto}: nenhuma feature informada")
        pedidos = set(nomes)
        return tuple(nome for nome in self.nomes if nome in pedidos)


# --------------------------------------------------------------------------- #
# Leitura estrutural (somente identificadores)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class _Estrutura:
    tabela: bytes
    esquema: EsquemaFeatures
    #: Identificadores na ordem do arquivo, com a partição de cada linha.
    ids: pd.DataFrame
    sha256: Mapping[str, str]


def _cabecalho(conteudo: bytes) -> List[str]:
    try:
        primeira = conteudo.split(b"\n", 1)[0].rstrip(b"\r").decode("utf-8")
        return next(csv.reader([primeira]))
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise DadosInvalidos("cabeçalho da tabela ilegível") from exc


def _linhas_fisicas(conteudo: bytes) -> int:
    """Número de linhas físicas, recusando o que desalinha ``skiprows``.

    ``skiprows`` conta linhas físicas. Linha em branco ou ``\\r`` isolado
    desalinhariam as linhas puladas; um campo com quebra de linha é detectado
    depois, comparando com o número de registros.
    """
    corpo = conteudo.replace(b"\r\n", b"\n")
    if b"\r" in corpo:
        raise DadosInvalidos("tabela com \\r isolado; leitura seletiva insegura")
    linhas = corpo[:-1].split(b"\n") if corpo.endswith(b"\n") else corpo.split(b"\n")
    if any(not linha.strip() for linha in linhas):
        raise DadosInvalidos("tabela com linha em branco; leitura seletiva insegura")
    return len(linhas)


def _ler_split(conteudo: bytes) -> Dict[str, str]:
    try:
        split = pd.read_csv(io.BytesIO(conteudo), dtype=str, keep_default_na=False, na_filter=False)
    except (ValueError, pd.errors.ParserError) as exc:
        raise DadosInvalidos(f"split ilegível: {exc}") from exc
    if list(split.columns) != [cfg.COLUNA_PACIENTE, COLUNA_SPLIT]:
        raise DadosInvalidos(f"split deve ter exatamente as colunas {cfg.COLUNA_PACIENTE},{COLUNA_SPLIT}")
    pacientes = split[cfg.COLUNA_PACIENTE]
    if (pacientes == "").any() or pacientes.duplicated().any():
        raise DadosInvalidos("split com patient_id vazio ou repetido (paciente em mais de uma partição)")
    desconhecidas = set(split[COLUNA_SPLIT]) - set(PARTICOES)
    if desconhecidas:
        raise DadosInvalidos(f"split com partições desconhecidas: {sorted(desconhecidas)}")
    return dict(zip(pacientes, split[COLUNA_SPLIT]))


def _ler_estrutura(fonte: FonteDados) -> _Estrutura:
    tabela = _ler_verificado(fonte.tabela)
    split = _ler_verificado(fonte.split)
    # Conteúdo oficial copiado para outro caminho continua oficial: exige a
    # fonte oficial inteira (split e expectativas), antes de qualquer DataFrame.
    if ((_conteudo_oficial(tabela, TABELA_OFICIAL) or _conteudo_oficial(split, SPLIT_OFICIAL))
            and fonte != fonte_oficial()):
        raise AcessoNegado("conteúdo oficial fora da fonte oficial")
    particao_do_paciente = _ler_split(split)

    n_linhas_fisicas = _linhas_fisicas(tabela)
    cabecalho = _cabecalho(tabela)
    if len(set(cabecalho)) != len(cabecalho):
        raise DadosInvalidos("cabeçalho com colunas repetidas")
    faltam = [c for c in (*COLUNAS_ID, cfg.COLUNA_ALVO, cfg.COLUNA_PROIBIDA) if c not in cabecalho]
    if faltam:
        raise DadosInvalidos(f"colunas ausentes na tabela: {faltam}")
    pool = tuple(c for c in cabecalho
                 if c.startswith(cfg.PREFIXO_RADIOMICA) and c != cfg.COLUNA_PROIBIDA)
    if len(pool) != cfg.N_POOL_CANDIDATAS or set(pool) != fonte.nomes_pool:
        raise DadosInvalidos(
            f"pool da tabela ({len(pool)} colunas) difere das {cfg.N_POOL_CANDIDATAS} candidatas versionadas")

    try:
        ids = pd.read_csv(io.BytesIO(tabela), usecols=list(COLUNAS_ID), dtype=str,
                          keep_default_na=False, na_filter=False)
    except (ValueError, pd.errors.ParserError) as exc:
        raise DadosInvalidos(f"tabela ilegível: {exc}") from exc
    ids = ids[list(COLUNAS_ID)]
    if n_linhas_fisicas != len(ids) + 1:
        raise DadosInvalidos("tabela com registro em mais de uma linha física; leitura seletiva insegura")

    nodulos, pacientes = ids[cfg.COLUNA_NODULO], ids[cfg.COLUNA_PACIENTE]
    if (nodulos == "").any() or (pacientes == "").any():
        raise DadosInvalidos("identificador vazio na tabela")
    if nodulos.duplicated().any():
        raise DadosInvalidos("nodulo_id repetido na tabela")
    for nodulo, paciente in zip(nodulos, pacientes):
        try:
            paciente_do_id = ids_extracao.decompor_id_extracao(nodulo)[0]
        except ValueError as exc:
            raise DadosInvalidos("nodulo_id fora do formato de extração da Sprint 3") from exc
        if paciente_do_id != paciente:
            raise DadosInvalidos("nodulo_id e patient_id incoerentes na mesma linha")
    if set(pacientes) != set(particao_do_paciente):
        raise DadosInvalidos("pacientes da tabela e do split diferem")

    ids = ids.assign(**{COLUNA_SPLIT: pacientes.map(particao_do_paciente)})
    grupos = {p: set(ids.loc[ids[COLUNA_SPLIT] == p, cfg.COLUNA_PACIENTE]) for p in PARTICOES}
    for i, a in enumerate(PARTICOES):
        for b in PARTICOES[i + 1:]:
            if grupos[a] & grupos[b]:
                raise DadosInvalidos(f"pacientes compartilhados entre {a} e {b}")
    for p in PARTICOES:
        obtido = {"nodulos": int((ids[COLUNA_SPLIT] == p).sum()), "pacientes": len(grupos[p])}
        esperado = {k: fonte.particoes[p][k] for k in obtido}
        if obtido != esperado:
            raise DadosInvalidos(f"partição {p}: esperado {esperado}, obtido {obtido}")
    total = {"nodulos": len(ids), "pacientes": int(pacientes.nunique())}
    if total != {k: fonte.coorte[k] for k in total}:
        raise DadosInvalidos(f"coorte: obtido {total}")

    sha = {"tabela": hashlib.sha256(tabela).hexdigest(), "split": hashlib.sha256(split).hexdigest()}
    return _Estrutura(tabela, EsquemaFeatures(pool), ids, MappingProxyType(sha))


@dataclass(frozen=True)
class RelatorioEstrutura:
    """Resultado das verificações estruturais — sem alvo e sem features."""

    esquema: EsquemaFeatures
    contagens: Mapping[str, Mapping[str, int]]


def verificar_estrutura(fonte: Optional[FonteDados] = None) -> RelatorioEstrutura:
    """Integridade e invariantes estruturais lendo só identificadores e split.

    Liberada sem decisões (``ATIVIDADE_VERIFICACOES_ESTRUTURAIS``). Contagens
    de classe exigem o alvo e, por isso, só são conferidas por ``carregar``.
    """
    guardas.exigir_autorizacao(cfg.ATIVIDADE_VERIFICACOES_ESTRUTURAIS)
    estrutura = _ler_estrutura(_exigir_fonte_permitida(fonte or fonte_oficial()))
    ids = estrutura.ids
    contagens = {
        p: MappingProxyType({
            "nodulos": int((ids[COLUNA_SPLIT] == p).sum()),
            "pacientes": int(ids.loc[ids[COLUNA_SPLIT] == p, cfg.COLUNA_PACIENTE].nunique()),
        })
        for p in PARTICOES
    }
    return RelatorioEstrutura(estrutura.esquema, MappingProxyType(contagens))


def esquema_features(fonte: Optional[FonteDados] = None) -> EsquemaFeatures:
    """As 107 candidatas na ordem do cabeçalho, depois das verificações estruturais."""
    return verificar_estrutura(fonte).esquema


# --------------------------------------------------------------------------- #
# Carga autorizada
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DadosCarregados:
    """Partições autorizadas, cada uma ordenada por ``nodulo_id``.

    Cada acesso devolve uma CÓPIA; partições não carregadas levantam
    ``AcessoNegado``.

    ``autorizacao`` cobre a LEITURA da atividade, não o uso de um modelo:
    ter as 107 candidatas carregadas não autoriza M2, M3 ou M4. O uso de cada
    modelo continua exigindo ``guardas.autorizar_modelo``.
    """

    atividade: str
    autorizacao: guardas.Autorizacao
    particoes: Tuple[str, ...]
    esquema: EsquemaFeatures
    features: Tuple[str, ...]
    sha256_entradas: Mapping[str, str]
    _frames: Mapping[str, pd.DataFrame] = field(repr=False, compare=False)

    def frame(self, particao: str) -> pd.DataFrame:
        if particao not in self.particoes:
            raise AcessoNegado(f"partição {particao!r} não foi carregada nesta atividade")
        return self._frames[particao].copy()

    def X(self, particao: str, colunas: Iterable[str]) -> pd.DataFrame:
        """Features NA ORDEM PEDIDA (a do modelo); só nomes carregados, sem duplicata."""
        if isinstance(colunas, (str, bytes)):
            raise TypeError("colunas deve ser uma coleção de nomes, não uma string")
        pedidas = tuple(colunas)
        self.esquema.validar(pedidas, "colunas de X", vazio_permitido=False)
        faltam = [c for c in pedidas if c not in self.features]
        if faltam:
            raise FeatureInvalida(f"features não carregadas: {faltam}")
        return self.frame(particao)[list(pedidas)]

    def y(self, particao: str) -> pd.Series:
        return self.frame(particao)[cfg.COLUNA_ALVO]

    def grupos(self, particao: str) -> pd.Series:
        return self.frame(particao)[cfg.COLUNA_PACIENTE]


def _particoes_pedidas(particoes: Iterable[str], permitidas: Tuple[str, ...], chave: str) -> Tuple[str, ...]:
    if isinstance(particoes, (str, bytes)):
        raise TypeError("particoes deve ser uma coleção de nomes, não uma string")
    pedidas = tuple(particoes)
    if not pedidas:
        raise AcessoNegado("nenhuma partição pedida")
    for p in pedidas:
        if type(p) is not str or p not in PARTICOES:
            raise AcessoNegado(f"partição desconhecida: {p!r}")
    if len(set(pedidas)) != len(pedidas):
        raise AcessoNegado("partição pedida mais de uma vez")
    negadas = [p for p in pedidas if p not in permitidas]
    if negadas:
        raise AcessoNegado(f"{chave}: partições {negadas} não permitidas; permitidas: {list(permitidas)}")
    return tuple(p for p in PARTICOES if p in pedidas)


def _ler_linhas_autorizadas(estrutura: _Estrutura, particoes: Tuple[str, ...],
                            features: Tuple[str, ...]) -> pd.DataFrame:
    ids = estrutura.ids
    permitida = ids[COLUNA_SPLIT].isin(particoes).to_numpy()
    # Linha física = posição do registro + 1 (cabeçalho na linha 0).
    puladas = [int(i) + 1 for i in np.flatnonzero(~permitida)]
    colunas = [*COLUNAS_ID, cfg.COLUNA_ALVO, *features]
    tipos = {cfg.COLUNA_NODULO: str, cfg.COLUNA_PACIENTE: str, cfg.COLUNA_ALVO: str,
             **{f: "float64" for f in features}}
    try:
        lido = pd.read_csv(io.BytesIO(estrutura.tabela), usecols=colunas, skiprows=puladas,
                           dtype=tipos, keep_default_na=True, float_precision=PRECISAO_FLOAT_CSV)
    except (ValueError, TypeError, pd.errors.ParserError) as exc:
        raise DadosInvalidos(f"valor não numérico nas partições autorizadas: {exc}") from exc
    lido = lido[colunas].copy()

    esperado = ids.loc[permitida, list(COLUNAS_ID)].reset_index(drop=True)
    if not lido[list(COLUNAS_ID)].reset_index(drop=True).equals(esperado):
        raise DadosInvalidos("linhas lidas desalinhadas dos identificadores esperados")

    alvo = lido[cfg.COLUNA_ALVO]
    if alvo.isna().any() or not alvo.isin(["0", "1"]).all():
        raise DadosInvalidos("target_binary deve ser 0 ou 1 em todas as linhas autorizadas")
    lido[cfg.COLUNA_ALVO] = alvo.astype(np.int64)
    if features and not np.isfinite(lido[list(features)].to_numpy(dtype=np.float64)).all():
        raise DadosInvalidos("features com NaN ou infinito nas partições autorizadas")
    lido[COLUNA_SPLIT] = ids.loc[permitida, COLUNA_SPLIT].to_numpy()
    return lido


def carregar(atividade: guardas.Atividade, particoes: Iterable[str], features: Iterable[str],
             decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
             repo: Path = guardas.REPO_ROOT,
             fonte: Optional[FonteDados] = None) -> DadosCarregados:
    """Carrega alvo e features das partições pedidas, se a atividade puder lê-las.

    ``particoes`` e ``features`` são sempre explícitas. As features voltam na
    ordem do esquema. ``decisoes`` e ``repo`` vão para
    ``guardas.exigir_autorizacao``, chamada antes de qualquer leitura.
    """
    chave = guardas.normalizar_atividade(atividade)
    etapa = ETAPA_DE_LEITURA.get(chave)
    if etapa is None:
        raise AcessoNegado(f"{chave}: atividade sem permissão de ler alvo ou features")
    permitidas = cfg.PARTICOES_POR_ETAPA[etapa]
    if not permitidas:
        raise AcessoNegado(f"{chave}: a {etapa.name} não lê partições")
    pedidas = _particoes_pedidas(particoes, permitidas, chave)
    if isinstance(features, (str, bytes)):
        raise TypeError("features deve ser uma coleção de nomes, não uma string")
    features = tuple(features)
    # Conteúdo oficial em outro caminho é recusado em _ler_estrutura; logo, toda
    # carga de dados oficiais passa por aqui com o repositório do projeto.
    if _toca_caminho_oficial(fonte):
        _exigir_repo_do_projeto(repo)

    # Antes de QUALQUER leitura de arquivo, inclusive do pool versionado.
    autorizacao = guardas.exigir_autorizacao(chave, decisoes, repo)

    fonte = _exigir_fonte_permitida(fonte or fonte_oficial())
    estrutura = _ler_estrutura(fonte)
    features = estrutura.esquema.validar(features, "features", vazio_permitido=False)
    lido = _ler_linhas_autorizadas(estrutura, pedidas, features)

    frames = {}
    for p in pedidas:
        bloco = lido[lido[COLUNA_SPLIT] == p].drop(columns=COLUNA_SPLIT)
        bloco = bloco.sort_values(cfg.ORDEM_LINHAS[0], kind=cfg.ORDEM_LINHAS[1]).reset_index(drop=True)
        classes = {"classe_0": int((bloco[cfg.COLUNA_ALVO] == 0).sum()),
                   "classe_1": int((bloco[cfg.COLUNA_ALVO] == 1).sum())}
        esperado = {k: fonte.particoes[p][k] for k in classes}
        if classes != esperado:
            raise DadosInvalidos(f"partição {p}: classes esperadas {esperado}, obtidas {classes}")
        frames[p] = bloco
    return DadosCarregados(chave, autorizacao, pedidas, estrutura.esquema, features,
                           estrutura.sha256, MappingProxyType(frames))
