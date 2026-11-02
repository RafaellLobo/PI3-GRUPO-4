"""Carregamento seguro dos dados do benchmark S6 (P9, Seções 3 e 14.3, passos 5 a 7).

Duas portas, sem parâmetro de partição livre:

* ``carregar_implementacao``: lê SÓ ``train`` (291 pacientes / 417 nódulos).
  Não exige aprovação (P9, Seção 14.3: o modo ``implementacao`` não consulta
  aprovações) e nunca lê alvo ou características da validação.
* ``carregar_definitivo``: lê ``train`` + ``validation`` (353 / 509). Chama
  ``aprovacoes.autorizar_definitivo`` AQUI, antes de qualquer leitura de
  arquivo; não existe parâmetro que aceite uma autorização pronta. Confere
  a faixa do desfecho F (226 / 187 / 135 / 91) antes de devolver os dados.

As duas portas públicas NÃO aceitam fonte, repositório, configuração nem
partição: usam sempre a fonte oficial (pins da configuração canônica,
conferidos com os da S4), a configuração canônica e as aprovações do
repositório do projeto. Fontes sintéticas existem só para testes, por
``_carregar_implementacao_sintetico`` e ``_carregar_definitivo_sintetico``,
que recusam qualquer arquivo em ``data/`` ou o split oficial e marcam o
resultado com ``oficial=False``; nenhum outro módulo do pacote as chama.

A partição ``test`` nunca é lida por nenhuma das duas. Ordem de cada carga:

1. a fonte é fixada sem ler nada;
2. (definitivo) autorização das guardas;
3. (fonte oficial) ``verificar_estrutura`` da S4: a ÚNICA leitura dos
   identificadores do teste, que confere as contagens das três partições
   sem ler alvo nem características (P9, Seção 3, e decisão D11);
4. os bytes da tabela e do split são lidos UMA vez e conferidos contra o
   SHA-256 fixado; todo o resto usa esses bytes. Conteúdo com o hash oficial
   em outro caminho é recusado;
5. primeira fase: só ``nodulo_id`` e ``patient_id``, para validar
   identificadores, partições e contagens e localizar as linhas;
6. segunda fase: só as linhas das partições lidas, e só os identificadores,
   o alvo e as três características. As demais linhas são puladas com
   ``skiprows`` e nunca chegam a um DataFrame (seguro só com uma linha
   física por registro, verificado antes). A sequência de identificadores
   lida tem de ser a esperada, e nenhuma linha de outra partição pode
   aparecer;
7. alvo 0/1, características ``float64`` finitas, contagens de pacientes,
   nódulos e classes, e ordem canônica por ``nodulo_id`` (estável).

Espelha ``src/incremental/dados.py`` (``carregar`` e as proteções da fonte
oficial), que está preso às atividades e decisões da S4 (P9, Seção 14.2).
O conversor de ``float`` do ``read_csv`` é o padrão do pandas, o mesmo da S4
e do baseline. O módulo não escreve arquivos, não ajusta modelos e não
calcula métricas.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import adaptadores_s4 as s4
from . import aprovacoes as ap
from . import config as cfg

REPO_ROOT: Path = s4.REPO_ROOT
DIRETORIO_DADOS: Path = REPO_ROOT / "data"
COLUNA_SPLIT: str = "split"
COLUNA_PARTICAO: str = "particao"
PARTICOES_CONHECIDAS: Tuple[str, ...] = ("train", "validation", "test")
MODO_IMPLEMENTACAO: str = "implementacao"
MODO_DEFINITIVO: str = "definitivo"
#: Partições lidas por modo (P9, Seção 14.3, passo 6). O teste não aparece.
PARTICOES_POR_MODO: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    MODO_IMPLEMENTACAO: ("train",),
    MODO_DEFINITIVO: ("train", "validation"),
})
#: Conversor de float do ``read_csv``: o padrão do pandas, como na S4
#: (``dados.PRECISAO_FLOAT_CSV``) e no baseline; trocar quebra a paridade.
PRECISAO_FLOAT_CSV: Optional[str] = None
_CHAVES_CONTAGEM = ("pacientes", "nodulos", "classe_0", "classe_1")

# --------------------------------------------------------------------------- #
# Exceções
# --------------------------------------------------------------------------- #


class AcessoNegado(PermissionError):
    """Leitura não permitida: fonte, repositório ou partição."""

    chave_saida: str = "bloqueio_aprovacao"


class DadosInvalidos(ValueError):
    """Integridade ou invariante violada; nada é devolvido."""

    chave_saida: str = "integridade"


# --------------------------------------------------------------------------- #
# Fonte
# --------------------------------------------------------------------------- #


def _contagem(valor: Any, contexto: str) -> Mapping[str, int]:
    if not isinstance(valor, Mapping) or set(valor) != set(_CHAVES_CONTAGEM):
        raise ValueError(f"{contexto}: chaves esperadas {_CHAVES_CONTAGEM}")
    for chave in _CHAVES_CONTAGEM:
        if type(valor[chave]) is not int or valor[chave] < 0:
            raise ValueError(f"{contexto}: {chave} deve ser inteiro não negativo")
    if valor["classe_0"] + valor["classe_1"] != valor["nodulos"]:
        raise ValueError(f"{contexto}: classes não somam os nódulos")
    return MappingProxyType({k: valor[k] for k in _CHAVES_CONTAGEM})


@dataclass(frozen=True)
class FonteBenchmark:
    """Arquivos, pins e contagens esperadas. Use ``fonte_oficial()``.

    Fontes sintéticas existem só para testes; uma fonte que toque ``data/``
    ou o split oficial tem de ser idêntica à oficial.
    """

    tabela: s4.ArquivoVerificado
    split: s4.ArquivoVerificado
    #: Contagens esperadas de ``train`` e ``validation``.
    contagens: Mapping[str, Mapping[str, int]]
    #: Contagem esperada da faixa do desfecho F na base train + validation.
    faixa: Mapping[str, int]

    def __post_init__(self) -> None:
        for nome in ("tabela", "split"):
            if not isinstance(getattr(self, nome), s4.ArquivoVerificado):
                raise TypeError(f"{nome} deve ser ArquivoVerificado")
        if not isinstance(self.contagens, Mapping) or tuple(self.contagens) != PARTICOES_POR_MODO[MODO_DEFINITIVO]:
            raise ValueError("contagens devem cobrir ('train', 'validation'), nessa ordem")
        object.__setattr__(self, "contagens", MappingProxyType(
            {p: _contagem(self.contagens[p], f"contagem de {p}") for p in self.contagens}))
        object.__setattr__(self, "faixa", _contagem(self.faixa, "contagem da faixa"))


def fonte_oficial(configuracao: cfg.ConfiguracaoBenchmark = cfg.CONFIGURACAO) -> FonteBenchmark:
    """Fonte oficial com os pins e contagens da configuração canônica. Não lê nada."""
    dados = configuracao.dados
    fonte = FonteBenchmark(
        tabela=s4.ArquivoVerificado(REPO_ROOT / dados.tabela.caminho, dados.tabela.sha256, dados.tabela.sha256_lf),
        split=s4.ArquivoVerificado(REPO_ROOT / dados.split.caminho, dados.split.sha256, dados.split.sha256_lf),
        contagens={p: dict(dados.contagens_particoes[p]) for p in dados.particoes},
        faixa=dict(configuracao.faixa.contagem_esperada),
    )
    for nosso, da_s4 in ((fonte.tabela, s4.TABELA_OFICIAL), (fonte.split, s4.SPLIT_OFICIAL)):
        if (nosso.caminho.resolve(), nosso.sha256, nosso.sha256_lf) != \
                (da_s4.caminho.resolve(), da_s4.sha256, da_s4.sha256_lf):
            raise DadosInvalidos(f"{nosso.caminho.name}: pins da configuração diferem dos da S4")
    return fonte


def _dentro_de(caminho: Path, diretorio: Path) -> bool:
    try:
        caminho.resolve().relative_to(diretorio.resolve())
        return True
    except ValueError:
        return False


def _toca_caminho_oficial(fonte: FonteBenchmark) -> bool:
    """Algum arquivo em ``data/`` ou no split oficial (sem ler nada)."""
    return any(_dentro_de(a.caminho, DIRETORIO_DADOS) or a.caminho.resolve() == s4.SPLIT_OFICIAL.caminho.resolve()
               for a in (fonte.tabela, fonte.split))


def _exigir_fonte_sintetica(fonte: Any) -> FonteBenchmark:
    """Fonte de teste: nunca toca ``data/`` nem o split oficial.

    A fonte oficial, mesmo idêntica a ``fonte_oficial()``, só entra pelas
    funções públicas, que fixam também o repositório das aprovações.
    """
    if not isinstance(fonte, FonteBenchmark):
        raise TypeError("fonte deve ser FonteBenchmark")
    if _toca_caminho_oficial(fonte):
        raise AcessoNegado("arquivos oficiais só pelas funções públicas carregar_implementacao e "
                           "carregar_definitivo, nunca por uma fonte sintética")
    return fonte


def _conteudo_oficial(conteudo: bytes, oficial: s4.ArquivoVerificado) -> bool:
    return (hashlib.sha256(conteudo).hexdigest() == oficial.sha256
            or hashlib.sha256(conteudo.replace(b"\r\n", b"\n")).hexdigest() == oficial.sha256_lf)


def _ler_bytes(caminho: Path) -> bytes:
    """Única leitura de arquivo do módulo (somente leitura)."""
    try:
        return caminho.read_bytes()
    except OSError as exc:
        raise DadosInvalidos(f"não foi possível ler {caminho.name}: {exc}") from exc


def _ler_verificado(arquivo: s4.ArquivoVerificado) -> bytes:
    conteudo = _ler_bytes(arquivo.caminho)
    if not _conteudo_oficial(conteudo, arquivo):
        raise DadosInvalidos(f"{arquivo.caminho.name}: SHA-256 diferente do fixado; o arquivo mudou")
    return conteudo


def _verificacao_estrutural_s4() -> None:
    """``verificar_estrutura`` pública da S4, sobre a fonte oficial.

    Única leitura dos identificadores de todas as partições, inclusive do
    teste, sem alvo nem características (P9, Seção 3, e D11).
    """
    s4.verificar_arquivos_s4()
    try:
        s4.verificar_estrutura()
    except (ValueError, PermissionError, RuntimeError, OSError) as exc:
        raise DadosInvalidos(f"verificação estrutural da S4 falhou: {type(exc).__name__}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Primeira fase: identificadores
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class _Estrutura:
    tabela: bytes
    #: ``nodulo_id``, ``patient_id`` e partição, na ordem do arquivo.
    ids: pd.DataFrame
    sha256: Mapping[str, str]
    #: Partição de cada paciente, direto do split (conferência independente).
    particao_do_paciente: Mapping[str, str]


def _cabecalho(conteudo: bytes) -> List[str]:
    try:
        primeira = conteudo.split(b"\n", 1)[0].rstrip(b"\r").decode("utf-8")
        return next(csv.reader([primeira]))
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise DadosInvalidos("cabeçalho da tabela ilegível") from exc


def _linhas_fisicas(conteudo: bytes) -> int:
    """Linhas físicas, recusando o que desalinha ``skiprows`` (espelho da S4)."""
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
    desconhecidas = set(split[COLUNA_SPLIT]) - set(PARTICOES_CONHECIDAS)
    if desconhecidas:
        raise DadosInvalidos(f"split com partições desconhecidas: {sorted(desconhecidas)}")
    return dict(zip(pacientes, split[COLUNA_SPLIT]))


def _ler_estrutura(fonte: FonteBenchmark, oficial: bool, caracteristicas: Tuple[str, ...]) -> _Estrutura:
    tabela = _ler_verificado(fonte.tabela)
    split = _ler_verificado(fonte.split)
    # Conteúdo oficial copiado para outro caminho continua oficial.
    if not oficial and (_conteudo_oficial(tabela, s4.TABELA_OFICIAL) or _conteudo_oficial(split, s4.SPLIT_OFICIAL)):
        raise AcessoNegado("conteúdo oficial fora da fonte oficial")
    particao_do_paciente = _ler_split(split)

    n_linhas = _linhas_fisicas(tabela)
    cabecalho = _cabecalho(tabela)
    if len(set(cabecalho)) != len(cabecalho):
        raise DadosInvalidos("cabeçalho com colunas repetidas")
    exigidas = (cfg.COLUNA_NODULO, cfg.COLUNA_PACIENTE, cfg.COLUNA_ALVO, *caracteristicas)
    faltam = [c for c in exigidas if c not in cabecalho]
    if faltam:
        raise DadosInvalidos(f"colunas ausentes na tabela: {faltam}")

    try:
        ids = pd.read_csv(io.BytesIO(tabela), usecols=[cfg.COLUNA_NODULO, cfg.COLUNA_PACIENTE], dtype=str,
                          keep_default_na=False, na_filter=False)
    except (ValueError, pd.errors.ParserError) as exc:
        raise DadosInvalidos(f"tabela ilegível: {exc}") from exc
    ids = ids[[cfg.COLUNA_NODULO, cfg.COLUNA_PACIENTE]]
    if n_linhas != len(ids) + 1:
        raise DadosInvalidos("tabela com registro em mais de uma linha física; leitura seletiva insegura")
    nodulos, pacientes = ids[cfg.COLUNA_NODULO], ids[cfg.COLUNA_PACIENTE]
    if (nodulos == "").any() or (pacientes == "").any():
        raise DadosInvalidos("identificador vazio na tabela")
    if nodulos.duplicated().any():
        raise DadosInvalidos("nodulo_id repetido na tabela")
    if set(pacientes) != set(particao_do_paciente):
        raise DadosInvalidos("pacientes da tabela e do split diferem")
    ids = ids.assign(**{COLUNA_PARTICAO: pacientes.map(particao_do_paciente)})
    sha = {"tabela": hashlib.sha256(tabela).hexdigest(), "split": hashlib.sha256(split).hexdigest()}
    return _Estrutura(tabela, ids, MappingProxyType(sha), MappingProxyType(dict(particao_do_paciente)))


# --------------------------------------------------------------------------- #
# Segunda fase: linhas das partições lidas
# --------------------------------------------------------------------------- #


def _ler_linhas(estrutura: _Estrutura, particoes: Tuple[str, ...],
                caracteristicas: Tuple[str, ...]) -> pd.DataFrame:
    ids = estrutura.ids
    lidas = ids[COLUNA_PARTICAO].isin(particoes).to_numpy()
    # Linha física = posição do registro + 1 (cabeçalho na linha 0).
    puladas = [int(i) + 1 for i in np.flatnonzero(~lidas)]
    colunas = [cfg.COLUNA_NODULO, cfg.COLUNA_PACIENTE, cfg.COLUNA_ALVO, *caracteristicas]
    tipos = {cfg.COLUNA_NODULO: str, cfg.COLUNA_PACIENTE: str, cfg.COLUNA_ALVO: str,
             **{c: "float64" for c in caracteristicas}}
    try:
        lido = pd.read_csv(io.BytesIO(estrutura.tabela), usecols=colunas, skiprows=puladas, dtype=tipos,
                           keep_default_na=True, float_precision=PRECISAO_FLOAT_CSV)
    except (ValueError, TypeError, pd.errors.ParserError) as exc:
        raise DadosInvalidos(f"valor não numérico nas partições lidas: {exc}") from exc
    lido = lido[colunas].copy()

    esperado = ids.loc[lidas, [cfg.COLUNA_NODULO, cfg.COLUNA_PACIENTE]].reset_index(drop=True)
    if not lido[[cfg.COLUNA_NODULO, cfg.COLUNA_PACIENTE]].reset_index(drop=True).equals(esperado):
        raise DadosInvalidos("linhas lidas desalinhadas dos identificadores esperados")
    lido[COLUNA_PARTICAO] = ids.loc[lidas, COLUNA_PARTICAO].to_numpy()
    # Conferência independente da máscara: a partição de cada linha lida,
    # rederivada do split pelo patient_id. Uma linha de teste aborta.
    rederivada = lido[cfg.COLUNA_PACIENTE].map(estrutura.particao_do_paciente)
    if not rederivada.isin(particoes).all() or not rederivada.equals(lido[COLUNA_PARTICAO]):
        raise DadosInvalidos("linha de partição não autorizada na matriz; execução abortada")

    alvo = lido[cfg.COLUNA_ALVO]
    if alvo.isna().any() or not alvo.isin(["0", "1"]).all():
        raise DadosInvalidos("target_binary deve ser 0 ou 1 em todas as linhas lidas")
    lido[cfg.COLUNA_ALVO] = alvo.astype(np.int64)
    if not np.isfinite(lido[list(caracteristicas)].to_numpy(dtype=np.float64)).all():
        raise DadosInvalidos("características com NaN ou infinito nas partições lidas")
    return lido[[cfg.COLUNA_NODULO, cfg.COLUNA_PACIENTE, COLUNA_PARTICAO, cfg.COLUNA_ALVO, *caracteristicas]]


def _contar(frame: pd.DataFrame) -> Dict[str, int]:
    return {"pacientes": int(frame[cfg.COLUNA_PACIENTE].nunique()), "nodulos": len(frame),
            "classe_0": int((frame[cfg.COLUNA_ALVO] == 0).sum()),
            "classe_1": int((frame[cfg.COLUNA_ALVO] == 1).sum())}


def _exigir_contagens_estruturais(estrutura: _Estrutura, fonte: FonteBenchmark) -> None:
    """Pacientes e nódulos de train e validation pelos identificadores, sem alvo.

    Vale nos dois modos: no ``implementacao`` a validação é contada, não lida.
    """
    ids = estrutura.ids
    for p, esperado in fonte.contagens.items():
        obtido = {"pacientes": int(ids.loc[ids[COLUNA_PARTICAO] == p, cfg.COLUNA_PACIENTE].nunique()),
                  "nodulos": int((ids[COLUNA_PARTICAO] == p).sum())}
        if obtido != {k: esperado[k] for k in obtido}:
            raise DadosInvalidos(f"partição {p}: esperado {dict(esperado)}, obtido {obtido} (identificadores)")


def _exigir_contagens(frame: pd.DataFrame, fonte: FonteBenchmark,
                      particoes: Tuple[str, ...]) -> Dict[str, Mapping[str, int]]:
    """Pacientes, nódulos e classes das partições lidas."""
    contagens: Dict[str, Mapping[str, int]] = {}
    for p in particoes:
        obtido = _contar(frame[frame[COLUNA_PARTICAO] == p])
        if obtido != dict(fonte.contagens[p]):
            raise DadosInvalidos(f"partição {p}: esperado {dict(fonte.contagens[p])}, obtido {obtido}")
        contagens[p] = MappingProxyType(obtido)
    contagens["base"] = MappingProxyType(_contar(frame))
    return contagens


def _exigir_faixa(frame: pd.DataFrame, fonte: FonteBenchmark,
                  configuracao: cfg.ConfiguracaoBenchmark) -> Mapping[str, int]:
    """Conferência da faixa do desfecho F na base (P9, Seção 14.3, passo 7; D2)."""
    faixa = configuracao.faixa
    valores = frame[faixa.variavel].to_numpy(dtype=np.float64)
    dentro = frame[(valores >= faixa.inferior) & (valores < faixa.superior)]
    obtido = _contar(dentro)
    if obtido != dict(fonte.faixa):
        raise DadosInvalidos(
            f"faixa {faixa.inferior:g} ≤ {faixa.variavel} < {faixa.superior:g}: esperado {dict(fonte.faixa)}, "
            f"obtido {obtido}; nenhum ajuste é feito")
    return MappingProxyType(obtido)


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DadosBenchmark:
    """Base lida, em ordem canônica por ``nodulo_id``. Cada acesso devolve uma cópia."""

    modo: str
    particoes: Tuple[str, ...]
    #: ``True`` só para a fonte oficial lida pelas funções públicas. Resultado
    #: definitivo reportável exige ``oficial`` (fontes sintéticas são de teste).
    oficial: bool
    caracteristicas: Tuple[str, ...]
    sha256_entradas: Mapping[str, str]
    contagens: Mapping[str, Mapping[str, int]]
    autorizacao: Optional[ap.AutorizacaoDefinitiva]
    _frame: pd.DataFrame = field(repr=False, compare=False)

    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    def X(self) -> np.ndarray:
        """Características na ordem da configuração (``FEATURES_M1``), ``float64``."""
        return self._frame[list(self.caracteristicas)].to_numpy(dtype=np.float64, copy=True)

    def y(self) -> np.ndarray:
        return self._frame[cfg.COLUNA_ALVO].to_numpy(dtype=np.int64, copy=True)

    def grupos(self) -> np.ndarray:
        return self._frame[cfg.COLUNA_PACIENTE].to_numpy(dtype=object, copy=True)

    def nodulos(self) -> np.ndarray:
        return self._frame[cfg.COLUNA_NODULO].to_numpy(dtype=object, copy=True)

    def amostra(self) -> s4.Amostra:
        """``folds.criar_amostra`` da S4: confere ordem canônica, alinhamento e alvo 0/1."""
        return s4.criar_amostra(self.nodulos(), self.y(), self.grupos())


# --------------------------------------------------------------------------- #
# Pontos de entrada
# --------------------------------------------------------------------------- #


def _carregar(modo: str, fonte: FonteBenchmark, oficial: bool, configuracao: cfg.ConfiguracaoBenchmark,
              autorizacao: Optional[ap.AutorizacaoDefinitiva]) -> DadosBenchmark:
    particoes = PARTICOES_POR_MODO[modo]
    if modo == MODO_DEFINITIVO and not isinstance(autorizacao, ap.AutorizacaoDefinitiva):
        raise AcessoNegado("validation só com autorização das guardas")
    caracteristicas = tuple(configuracao.caracteristicas)
    if oficial:
        _verificacao_estrutural_s4()
    estrutura = _ler_estrutura(fonte, oficial, caracteristicas)
    _exigir_contagens_estruturais(estrutura, fonte)
    lido = _ler_linhas(estrutura, particoes, caracteristicas)
    contagens = _exigir_contagens(lido, fonte, particoes)
    ordem, algoritmo = configuracao.dados.ordem_linhas
    base = lido.sort_values(ordem, kind=algoritmo).reset_index(drop=True)
    if modo == MODO_DEFINITIVO:
        contagens["faixa"] = _exigir_faixa(base, fonte, configuracao)
    dados = DadosBenchmark(modo, particoes, oficial, caracteristicas, estrutura.sha256,
                           MappingProxyType(contagens), autorizacao, base)
    try:
        dados.amostra()
    except (ValueError, TypeError) as exc:
        raise DadosInvalidos(f"base fora da ordem canônica ou desalinhada: {exc}") from exc
    return dados


def _validar_configuracao(configuracao: Any) -> cfg.ConfiguracaoBenchmark:
    if not isinstance(configuracao, cfg.ConfiguracaoBenchmark):
        raise TypeError("configuracao deve ser ConfiguracaoBenchmark")
    if tuple(configuracao.dados.particoes) != PARTICOES_POR_MODO[MODO_DEFINITIVO] or \
            configuracao.dados.particao_excluida != "test":
        raise AcessoNegado("a configuração tem de ler só train e validation e excluir test")
    return configuracao


def _autorizar(manifesto_s4: Optional[str], repo: Union[str, Path], configuracao: cfg.ConfiguracaoBenchmark,
               commit_p9: Optional[str], commit_s4_e3: Optional[str]) -> ap.AutorizacaoDefinitiva:
    autorizacao = ap.autorizar_definitivo(manifesto_s4, repo=repo, configuracao=configuracao,
                                          commit_p9=commit_p9, commit_s4_e3=commit_s4_e3)
    if not isinstance(autorizacao, ap.AutorizacaoDefinitiva) or \
            autorizacao.sha256_configuracao != cfg.sha256_configuracao(configuracao):
        raise AcessoNegado("autorização inválida para esta configuração")
    return autorizacao


# Produção: SÓ a fonte oficial, a configuração canônica e o repositório do
# projeto. Não há parâmetro de fonte, repositório, configuração ou partição.


def carregar_implementacao() -> DadosBenchmark:
    """Modo ``implementacao``: só ``train`` oficial, sem aprovação (P9, Seção 14.3)."""
    configuracao = _validar_configuracao(cfg.CONFIGURACAO)
    return _carregar(MODO_IMPLEMENTACAO, fonte_oficial(configuracao), True, configuracao, None)


def carregar_definitivo(manifesto_s4: Optional[str], commit_p9: Optional[str] = None,
                        commit_s4_e3: Optional[str] = None) -> DadosBenchmark:
    """Modo ``definitivo``: ``train`` + ``validation`` oficiais, só com P9 e S4_E3_REGISTRADA.

    As aprovações vêm sempre do repositório do projeto (``REPO_ROOT``).
    ``aprovacoes.autorizar_definitivo`` é chamada aqui, antes de qualquer
    leitura de arquivo; um bloqueio levanta ``aprovacoes.BloqueioAprovacao``.
    """
    configuracao = _validar_configuracao(cfg.CONFIGURACAO)
    fonte = fonte_oficial(configuracao)
    autorizacao = _autorizar(manifesto_s4, REPO_ROOT, configuracao, commit_p9, commit_s4_e3)
    return _carregar(MODO_DEFINITIVO, fonte, True, configuracao, autorizacao)


# Testes: fontes SINTÉTICAS, que nunca tocam data/ nem o split oficial. O
# resultado sai com ``oficial=False``. Nenhum outro módulo do pacote as usa
# (verificado por teste).


def _carregar_implementacao_sintetico(fonte: FonteBenchmark) -> DadosBenchmark:
    configuracao = _validar_configuracao(cfg.CONFIGURACAO)
    return _carregar(MODO_IMPLEMENTACAO, _exigir_fonte_sintetica(fonte), False, configuracao, None)


def _carregar_definitivo_sintetico(manifesto_s4: Optional[str], repo: Union[str, Path], fonte: FonteBenchmark,
                                   commit_p9: Optional[str] = None,
                                   commit_s4_e3: Optional[str] = None) -> DadosBenchmark:
    configuracao = _validar_configuracao(cfg.CONFIGURACAO)
    fonte = _exigir_fonte_sintetica(fonte)
    autorizacao = _autorizar(manifesto_s4, repo, configuracao, commit_p9, commit_s4_e3)
    return _carregar(MODO_DEFINITIVO, fonte, False, configuracao, autorizacao)
