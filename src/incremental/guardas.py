"""Guardas científicas de execução do Cartão 4 — Sprint 4 (Incremento 2A).

Toda atividade experimental chama ``exigir_autorizacao`` (ou
``autorizar_modelo``) imediatamente antes de executar. A guarda consulta
``config.DECISOES_POR_ETAPA`` e recusa a execução se alguma decisão exigida:

* estiver pendente;
* for inconsistente com a definição canônica de ``config.DECISOES``;
* tiver hash de commit inválido, inexistente, que não seja commit ou que não
  seja ancestral do HEAD;
* não tiver registro explícito de aprovação no protocolo.

A existência do commit NÃO basta. O protocolo (Seção 13) exige que cada
decisão seja "registrada em revisão deste documento, com data". A guarda
procura esse registro, em ``DOCUMENTOS_APROVACAO``, como uma linha própria no
formato de ``formatar_registro``::

    APROVACAO-DECISAO codigo=P3 status=aprovada data=AAAA-MM-DD valor_sha256=<64 hex>

Só conta a linha que começa na coluna 0, fora de bloco de código e de
comentário HTML (``_linhas_de_registro``). Exemplos, como o acima, nunca contam.

``valor_sha256`` liga o registro ao conteúdo aprovado (``hash_valor_decisao``).
O registro tem de:

1. existir uma única vez no commit da aprovação;
2. ter sido introduzido por esse commit, ou seja, estar ausente de todos os pais;
3. ter data igual ou anterior à data do commit;
4. permanecer idêntico e único no HEAD, sem ter sido revogado nem alterado.

Sem evidência verificável, a guarda falha de forma conservadora. O formato do
registro é uma PROPOSTA desta implementação: o protocolo ainda não contém
nenhum registro desse tipo, e por isso toda atividade que exige decisões
permanece bloqueada.

Verificações puramente estruturais e a Etapa 0 não exigem decisões e não
consultam o git.

As verificações são refeitas a cada chamada. Nada é guardado entre chamadas, e
a guarda não confia na marca de autorização de ``EspecModelo``:
``autorizar_modelo`` rederiva a composição esperada a partir das decisões
verificadas.

O módulo lê apenas objetos do git (commits e documentos versionados). Não lê
dados de pacientes, não lê a cópia de trabalho, não escreve arquivos e não
altera ``config``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

from . import config as cfg

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Documentos onde um registro de aprovação é aceito (Seção 13 do protocolo).
DOCUMENTOS_APROVACAO: Tuple[str, ...] = (cfg.PROTOCOLO,)
#: Início de linha que identifica um registro de aprovação.
MARCADOR_REGISTRO: str = "APROVACAO-DECISAO"
#: Único status de registro que autoriza.
STATUS_REGISTRO_APROVADA: str = "aprovada"

_REGISTRO = re.compile(
    r"APROVACAO-DECISAO"
    r" codigo=(?P<codigo>[A-Z][A-Z0-9_]*)"
    r" status=(?P<status>[a-z_]+)"
    r" data=(?P<data>\d{4}-\d{2}-\d{2})"
    r" valor_sha256=(?P<valor_sha256>[0-9a-f]{64})"
)
_HASH_COMMIT = re.compile(r"[0-9a-f]{40}")
#: Abertura de bloco cercado do Markdown, já sem o recuo.
_CERCA = re.compile(r"(?P<cerca>`{3,}|~{3,})(?P<info>.*)")
_TIMEOUT_GIT_S = 60

# --------------------------------------------------------------------------- #
# Exceções — todas as recusas por decisão herdam de BloqueioDecisao
# --------------------------------------------------------------------------- #


class AtividadeDesconhecida(ValueError):
    """Atividade ausente de ``config.DECISOES_POR_ETAPA``."""


class ModeloDesconhecido(ValueError):
    """Nome de modelo fora de ``MODELOS_FIXOS`` e ``MODELOS_VARIAVEIS``."""


class DecisaoInconsistente(cfg.BloqueioDecisao):
    """Decisão divergente da definição canônica ou com conteúdo inválido."""


class AprovacaoNaoVerificada(cfg.BloqueioDecisao):
    """Decisão marcada como aprovada sem evidência verificável no repositório."""


# --------------------------------------------------------------------------- #
# Registros de resultado
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VerificacaoDecisao:
    """Evidência aceita para uma decisão aprovada."""

    codigo: str
    commit: str
    documento: str
    data: str
    valor_sha256: str


@dataclass(frozen=True)
class Autorizacao:
    """Resultado de uma autorização concedida no momento da chamada.

    ``head`` é ``None`` quando a atividade não exige decisões, porque nesse caso
    o git não é consultado.
    """

    atividade: str
    decisoes_exigidas: Tuple[str, ...]
    verificacoes: Tuple[VerificacaoDecisao, ...]
    head: Optional[str]
    hash_configuracao: str

    def como_dict(self) -> dict:
        return {
            "atividade": self.atividade,
            "decisoes_exigidas": list(self.decisoes_exigidas),
            "verificacoes": [vars(v).copy() for v in self.verificacoes],
            "head": self.head,
            "hash_configuracao": self.hash_configuracao,
        }


# --------------------------------------------------------------------------- #
# Atividades e decisões exigidas
# --------------------------------------------------------------------------- #

Atividade = Union[str, cfg.Etapa]


def normalizar_atividade(atividade: Atividade) -> str:
    """Chave exata de ``DECISOES_POR_ETAPA``; qualquer outra coisa é recusada.

    Não há normalização de caixa, espaços ou nomes de ``Etapa``: uma atividade
    escrita de forma incorreta nunca cai numa entrada sem decisões.
    """
    chave = atividade.value if isinstance(atividade, cfg.Etapa) else atividade
    if type(chave) is not str or chave not in cfg.DECISOES_POR_ETAPA:
        raise AtividadeDesconhecida(f"atividade desconhecida: {atividade!r}")
    return chave


def _validar_modelos(modelos: Iterable[str]) -> Tuple[str, ...]:
    """Nomes exatos de modelos; recusa string solta e nomes desconhecidos.

    Sem isso, ``"M2"`` seria iterado como ``"M"`` e ``"2"`` e P3 deixaria de ser
    exigida.
    """
    if isinstance(modelos, (str, bytes)):
        raise TypeError("modelos deve ser uma coleção de nomes, não uma string")
    nomes = tuple(modelos)
    conhecidos = cfg.MODELOS_FIXOS + cfg.MODELOS_VARIAVEIS
    for nome in nomes:
        if type(nome) is not str or nome not in conhecidos:
            raise ModeloDesconhecido(f"modelo desconhecido: {nome!r}")
    return nomes


def decisoes_exigidas(atividade: Atividade, modelos: Iterable[str] = ()) -> Tuple[str, ...]:
    """Decisões da atividade em ``DECISOES_POR_ETAPA``; M2, M3 e M4 somam P3.

    A ordem segue ``config.DECISOES``.
    """
    exigidas = set(cfg.DECISOES_POR_ETAPA[normalizar_atividade(atividade)])
    if any(nome in cfg.MODELOS_COM_PAINEL for nome in _validar_modelos(modelos)):
        exigidas.add("P3")
    return tuple(c for c in cfg.DECISOES if c in exigidas)


# --------------------------------------------------------------------------- #
# Consistência das decisões (sem git)
# --------------------------------------------------------------------------- #


def hash_valor_decisao(valor: Any) -> str:
    """SHA-256 do JSON canônico do conteúdo aprovado (``null`` quando não há)."""
    try:
        texto = json.dumps(cfg._jsonavel(valor), sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DecisaoInconsistente(f"conteúdo da decisão não serializável: {exc}") from exc
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def formatar_registro(codigo: str, data: str, valor: Any = None) -> str:
    """Linha de registro que a guarda aceita. Apenas formata; não grava nada."""
    linha = (f"{MARCADOR_REGISTRO} codigo={codigo} status={STATUS_REGISTRO_APROVADA} "
             f"data={data} valor_sha256={hash_valor_decisao(valor)}")
    _ler_registro(linha, "formatar_registro")
    return linha


def _validar_consistencia(decisoes: Mapping[str, cfg.Decisao]) -> None:
    """Recusa mapas que divergem de ``config.DECISOES`` além de status, commit e valor.

    Refaz as invariantes de ``Decisao``, porque o objeto pode ter sido alterado
    depois da construção (por exemplo, com ``object.__setattr__``).
    """
    if not isinstance(decisoes, Mapping):
        raise DecisaoInconsistente("decisões devem ser um mapeamento código → Decisao")
    if set(decisoes) != set(cfg.DECISOES):
        faltam = sorted(set(cfg.DECISOES) - set(decisoes))
        sobram = sorted(set(decisoes) - set(cfg.DECISOES))
        raise DecisaoInconsistente(f"decisões divergentes do protocolo: faltam {faltam}, sobram {sobram}")
    for codigo, canonica in cfg.DECISOES.items():
        decisao = decisoes[codigo]
        if not isinstance(decisao, cfg.Decisao):
            raise DecisaoInconsistente(f"{codigo}: não é Decisao")
        for atributo in ("codigo", "descricao", "prazo", "etapa_limite"):
            if getattr(decisao, atributo) != getattr(canonica, atributo):
                raise DecisaoInconsistente(f"{codigo}: {atributo} diverge da definição em config")
        if not isinstance(decisao.status, cfg.StatusDecisao):
            raise DecisaoInconsistente(f"{codigo}: status inválido")
        commit = decisao.commit_revisao_protocolo
        if decisao.aprovada:
            if not (isinstance(commit, str) and _HASH_COMMIT.fullmatch(commit)):
                raise DecisaoInconsistente(
                    f"{codigo}: hash de commit inválido ({commit!r}); exige 40 hex minúsculos")
        elif commit is not None or decisao.valor is not None:
            raise DecisaoInconsistente(f"{codigo}: decisão pendente com commit ou valor")


def _validar_conteudo(codigo: str, decisoes: Mapping[str, cfg.Decisao]) -> None:
    """Conteúdo aprovado serializável e, em P2 e P3, com a forma definida em ``config``."""
    hash_valor_decisao(decisoes[codigo].valor)
    try:
        if codigo == "P2":
            cfg.spline_confirmado(decisoes)
        elif codigo == "P3":
            cfg.painel_fixo(decisoes)
    except (TypeError, ValueError) as exc:
        raise DecisaoInconsistente(f"{codigo}: conteúdo aprovado inválido: {exc}") from exc


def _hash_configuracao(decisoes: Mapping[str, cfg.Decisao]) -> str:
    try:
        return cfg.hash_configuracao(decisoes)
    except (TypeError, ValueError) as exc:
        raise DecisaoInconsistente(f"configuração não serializável: {exc}") from exc


def _exigir_aprovadas(exigidas: Tuple[str, ...], decisoes: Mapping[str, cfg.Decisao],
                      atividade: str) -> None:
    pendentes = [c for c in exigidas if not decisoes[c].aprovada]
    if pendentes:
        detalhes = "; ".join(f"{c} ({decisoes[c].prazo})" for c in pendentes)
        raise cfg.BloqueioDecisao(
            f"{atividade}: decisões pendentes — {detalhes}. Registre cada decisão em "
            "revisão do protocolo antes de prosseguir."
        )


# --------------------------------------------------------------------------- #
# Acesso ao git — somente leitura de objetos, sem cache entre chamadas
# --------------------------------------------------------------------------- #


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    executavel = shutil.which("git")
    if executavel is None:
        raise AprovacaoNaoVerificada("git indisponível: não é possível verificar aprovações")
    # Variáveis GIT_* (GIT_DIR, GIT_WORK_TREE...) poderiam apontar para outro
    # repositório; refs/replace poderiam exibir outro conteúdo no lugar do commitado.
    ambiente = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    try:
        return subprocess.run([executavel, "--no-replace-objects", "-C", str(repo), *args],
                              capture_output=True,
                              env=ambiente, timeout=_TIMEOUT_GIT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AprovacaoNaoVerificada(f"falha ao consultar o git: {exc}") from exc


def _saida(processo: subprocess.CompletedProcess) -> str:
    return processo.stdout.decode("utf-8", errors="replace").strip()


def _head(repo: Path) -> str:
    """HEAD de ``repo``, que tem de ser a RAIZ do repositório.

    Sem essa exigência, um diretório qualquer dentro de outro repositório git
    (por exemplo, um home versionado) seria verificado contra o repositório
    que o contém.
    """
    processo = _git(repo, "rev-parse", "--show-toplevel", "HEAD^{commit}")
    linhas = _saida(processo).splitlines()
    if processo.returncode != 0 or len(linhas) != 2 or not _HASH_COMMIT.fullmatch(linhas[1]):
        raise AprovacaoNaoVerificada(f"não foi possível ler o HEAD de {repo}")
    if os.path.normcase(Path(linhas[0]).resolve()) != os.path.normcase(Path(repo).resolve()):
        raise AprovacaoNaoVerificada(
            f"{repo} não é a raiz de um repositório git (raiz encontrada: {linhas[0]})")
    return linhas[1]


class _LeitorGit:
    """Leituras de uma única chamada da guarda; o cache morre com a chamada."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self._documentos: Dict[Tuple[str, str], Optional[str]] = {}

    def exigir_commit(self, codigo: str, commit: str) -> None:
        processo = _git(self.repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
        if processo.returncode != 0 or _saida(processo) != commit:
            tipo = _saida(_git(self.repo, "cat-file", "-t", commit))
            situacao = f"objeto {tipo}, não commit" if tipo else "inexistente"
            raise AprovacaoNaoVerificada(f"{codigo}: commit {commit} {situacao} no repositório")

    def exigir_ancestral(self, codigo: str, commit: str, head: str) -> None:
        processo = _git(self.repo, "merge-base", "--is-ancestor", commit, head)
        if processo.returncode == 1:
            raise AprovacaoNaoVerificada(f"{codigo}: commit {commit} não é ancestral do HEAD {head}")
        if processo.returncode != 0:
            raise AprovacaoNaoVerificada(f"{codigo}: falha ao verificar ancestralidade de {commit}")

    def pais(self, commit: str) -> Tuple[str, ...]:
        processo = _git(self.repo, "rev-list", "--parents", "-n", "1", commit)
        partes = _saida(processo).split()
        if processo.returncode != 0 or not partes or partes[0] != commit:
            raise AprovacaoNaoVerificada(f"falha ao ler os pais de {commit}")
        return tuple(partes[1:])

    def data(self, commit: str) -> _dt.date:
        processo = _git(self.repo, "log", "-1", "--no-show-signature", "--format=%cs", commit)
        try:
            if processo.returncode != 0:
                raise ValueError
            return _dt.date.fromisoformat(_saida(processo))
        except ValueError as exc:
            raise AprovacaoNaoVerificada(f"falha ao ler a data do commit {commit}") from exc

    def documento(self, commit: str, caminho: str) -> Optional[str]:
        """Conteúdo versionado do documento no commit; ``None`` se não existir."""
        chave = (commit, caminho)
        if chave not in self._documentos:
            processo = _git(self.repo, "cat-file", "blob", f"{commit}:{caminho}")
            if processo.returncode != 0:
                self._documentos[chave] = None
            else:
                try:
                    self._documentos[chave] = processo.stdout.decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    raise AprovacaoNaoVerificada(
                        f"{caminho} em {commit} não é UTF-8 válido") from exc
        return self._documentos[chave]


# --------------------------------------------------------------------------- #
# Registros de aprovação
# --------------------------------------------------------------------------- #


def _ler_registro(linha: str, contexto: str) -> Dict[str, str]:
    encontrado = _REGISTRO.fullmatch(linha)
    if encontrado is None:
        raise AprovacaoNaoVerificada(f"{contexto}: registro de aprovação malformado: {linha!r}")
    registro = encontrado.groupdict()
    try:
        _dt.date.fromisoformat(registro["data"])
    except ValueError as exc:
        raise AprovacaoNaoVerificada(f"{contexto}: data inválida no registro: {linha!r}") from exc
    return registro


def _fecha_cerca(linha: str, cerca: Tuple[str, int]) -> bool:
    """Fechamento CommonMark: até 3 espaços de recuo (tab = 4), mesmo caractere,
    comprimento igual ou maior que o da abertura e nada além de espaços."""
    expandida = linha.expandtabs(4)
    resto = expandida.lstrip(" ")
    if len(expandida) - len(resto) > 3:
        return False
    resto = resto.rstrip()
    caractere, comprimento = cerca
    return len(resto) >= comprimento and set(resto) == {caractere}


def _linhas_de_registro(texto: str) -> Iterator[str]:
    """Linhas que podem ser registros reais de aprovação.

    Só conta a linha que começa NA COLUNA 0 pelo marcador, fora de bloco
    cercado (``` ou ~~~) e fora de comentário HTML. Fica de fora, portanto,
    todo exemplo em bloco de código (cercado ou indentado), citação, lista,
    tabela ou comentário.

    Cada erro de interpretação só pode levar a ignorar registros, e assim a
    falhar fechado, nunca a aceitar um exemplo:

    * a cerca abre com qualquer recuo, mas fecha só pela regra estrita do
      CommonMark;
    * cerca sem fechamento vai até o fim do documento;
    * ``<!--`` sem ``-->`` na mesma linha esconde tudo até o próximo ``-->``.
    """
    cerca: Optional[Tuple[str, int]] = None
    em_comentario = False
    for linha in texto.splitlines():
        if cerca is not None:
            if _fecha_cerca(linha, cerca):
                cerca = None
            continue
        if em_comentario:
            # Fecha no "-->", mas pode reabrir na mesma linha ("x --> y <!--").
            if "-->" in linha:
                em_comentario = linha.rfind("<!--") > linha.rfind("-->")
            continue
        abertura = _CERCA.fullmatch(linha.strip())
        if abertura and not (abertura["cerca"][0] == "`" and "`" in abertura["info"]):
            cerca = (abertura["cerca"][0], len(abertura["cerca"]))
            continue
        if linha.rfind("<!--") > linha.rfind("-->"):
            em_comentario = True
            continue
        if linha.startswith(MARCADOR_REGISTRO):
            yield linha.rstrip()


def _registros(leitor: _LeitorGit, commit: str, codigo: str,
               estrito: bool = True) -> List[Tuple[str, Dict[str, str]]]:
    """Registros de ``codigo`` nos documentos de aprovação do commit.

    No modo estrito, toda linha que começa pelo marcador tem de estar bem
    formada, de qualquer decisão. O modo tolerante só serve para os pais do
    commit de aprovação, onde a pergunta é apenas se o registro já existia.
    """
    encontrados = []
    for caminho in DOCUMENTOS_APROVACAO:
        texto = leitor.documento(commit, caminho)
        if texto is None:
            continue
        for linha in _linhas_de_registro(texto):
            try:
                registro = _ler_registro(linha, f"{caminho} em {commit[:12]}")
            except AprovacaoNaoVerificada:
                if estrito:
                    raise
                continue
            if registro["codigo"] == codigo:
                encontrados.append((caminho, registro))
    return encontrados


def verificar_aprovacao(codigo: str, decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
                        repo: Union[str, Path] = REPO_ROOT) -> VerificacaoDecisao:
    """Evidência de uma decisão aprovada, contra o HEAD atual de ``repo``.

    Levanta ``BloqueioDecisao`` sem ela.
    """
    _validar_consistencia(decisoes)
    if codigo not in cfg.DECISOES:
        raise DecisaoInconsistente(f"decisão desconhecida: {codigo!r}")
    cfg.exigir_aprovada(codigo, decisoes)
    _validar_conteudo(codigo, decisoes)
    leitor = _LeitorGit(Path(repo))
    head = _head(leitor.repo)
    verificacao = _verificar(codigo, decisoes[codigo], leitor, head)
    if _head(leitor.repo) != head:
        raise AprovacaoNaoVerificada("o HEAD mudou durante a verificação; repita a verificação")
    return verificacao


def _verificar(codigo: str, decisao: cfg.Decisao, leitor: _LeitorGit,
               head: str) -> VerificacaoDecisao:
    """Regras de evidência; ``head`` vem sempre de ``_head`` na mesma chamada pública."""
    commit = decisao.commit_revisao_protocolo
    leitor.exigir_commit(codigo, commit)
    leitor.exigir_ancestral(codigo, commit, head)

    no_commit = _registros(leitor, commit, codigo)
    if not no_commit:
        raise AprovacaoNaoVerificada(
            f"{codigo}: o commit {commit} existe, mas não contém registro explícito da "
            f"aprovação em {', '.join(DOCUMENTOS_APROVACAO)}; a existência do commit não "
            "comprova a aprovação científica"
        )
    if len(no_commit) > 1:
        raise AprovacaoNaoVerificada(f"{codigo}: {len(no_commit)} registros no commit {commit}")
    documento, registro = no_commit[0]

    if registro["status"] != STATUS_REGISTRO_APROVADA:
        raise AprovacaoNaoVerificada(f"{codigo}: registro com status {registro['status']!r}")
    esperado = hash_valor_decisao(decisao.valor)
    if registro["valor_sha256"] != esperado:
        raise AprovacaoNaoVerificada(
            f"{codigo}: o conteúdo da decisão não corresponde ao registro aprovado "
            f"(registro {registro['valor_sha256'][:12]}…, decisão {esperado[:12]}…)"
        )
    if _dt.date.fromisoformat(registro["data"]) > leitor.data(commit):
        raise AprovacaoNaoVerificada(f"{codigo}: registro datado depois do commit {commit}")
    for pai in leitor.pais(commit):
        if any(r == registro for _, r in _registros(leitor, pai, codigo, estrito=False)):
            raise AprovacaoNaoVerificada(
                f"{codigo}: o registro já existia em {pai}; informe o commit que o introduziu")
    if _registros(leitor, head, codigo) != [(documento, registro)]:
        raise AprovacaoNaoVerificada(
            f"{codigo}: o registro aprovado em {commit} foi revogado, alterado ou "
            f"duplicado até o HEAD {head}"
        )
    return VerificacaoDecisao(codigo, commit, documento, registro["data"], registro["valor_sha256"])


# --------------------------------------------------------------------------- #
# Pontos de entrada
# --------------------------------------------------------------------------- #


def exigir_autorizacao(atividade: Atividade, decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
                       repo: Union[str, Path] = REPO_ROOT,
                       modelos: Iterable[str] = ()) -> Autorizacao:
    """Autoriza a atividade agora ou levanta ``BloqueioDecisao``.

    Deve ser chamada imediatamente antes da execução; não há cache entre
    chamadas.
    """
    chave = normalizar_atividade(atividade)
    exigidas = decisoes_exigidas(chave, modelos)
    _validar_consistencia(decisoes)
    _exigir_aprovadas(exigidas, decisoes, chave)
    for codigo in exigidas:
        _validar_conteudo(codigo, decisoes)
    hash_config = _hash_configuracao(decisoes)
    if not exigidas:
        return Autorizacao(chave, (), (), None, hash_config)

    leitor = _LeitorGit(Path(repo))
    head = _head(leitor.repo)
    verificacoes = tuple(_verificar(c, decisoes[c], leitor, head) for c in exigidas)
    if _head(leitor.repo) != head:
        raise AprovacaoNaoVerificada("o HEAD mudou durante a verificação; repita a autorização")
    return Autorizacao(chave, exigidas, verificacoes, head, hash_config)


def _composicao_esperada(espec: cfg.EspecModelo,
                         decisoes: Mapping[str, cfg.Decisao]) -> Tuple[cfg.TipoModelo, Tuple[str, ...], bool]:
    nome = espec.nome
    referencia = cfg.REFERENCIA[nome]
    if nome in cfg.MODELOS_FIXOS:
        if nome in cfg.MODELOS_COM_PAINEL:
            return cfg.TipoModelo.FIXO, cfg.uniao_ordenada(referencia, cfg.painel_fixo(decisoes)), False
        return cfg.TipoModelo.FIXO, referencia, False
    features = espec.features
    if not isinstance(features, tuple) or features[:len(referencia)] != referencia:
        raise DecisaoInconsistente(f"{nome}: preditores de referência ausentes ou fora de ordem")
    try:
        esperado = cfg.modelo_variavel(nome, features[len(referencia):])
    except (TypeError, ValueError) as exc:
        raise DecisaoInconsistente(f"{nome}: composição inválida: {exc}") from exc
    return esperado.tipo, esperado.features, esperado.usa_dummy


def autorizar_modelo(espec: cfg.EspecModelo, atividade: Atividade,
                     decisoes: Mapping[str, cfg.Decisao] = cfg.DECISOES,
                     repo: Union[str, Path] = REPO_ROOT) -> Autorizacao:
    """Autoriza o uso do modelo na atividade, revalidando tudo no momento da chamada.

    Ignora a marca interna de ``EspecModelo``: M2, M3 e M4 exigem P3 com
    evidência, e as features são comparadas com referência ∪ painel de P3.
    Modelos variáveis são reconstruídos com ``modelo_variavel``.
    """
    if not isinstance(espec, cfg.EspecModelo):
        raise TypeError("autorizar_modelo exige um EspecModelo")
    if espec.nome not in cfg.MODELOS_FIXOS + cfg.MODELOS_VARIAVEIS:
        raise DecisaoInconsistente(f"modelo desconhecido: {espec.nome!r}")
    autorizacao = exigir_autorizacao(atividade, decisoes, repo, modelos=(espec.nome,))
    tipo, features, usa_dummy = _composicao_esperada(espec, decisoes)
    if (espec.tipo, espec.features, espec.usa_dummy) != (tipo, features, usa_dummy):
        raise DecisaoInconsistente(
            f"{espec.nome}: composição diverge da derivada das decisões verificadas")
    return autorizacao
