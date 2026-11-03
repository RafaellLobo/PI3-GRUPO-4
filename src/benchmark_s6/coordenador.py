"""Comando coordenador do benchmark S6 (P9, Seção 14.3).

Uso, a partir da raiz do repositório::

    python -m src.benchmark_s6 --modo configuracao [--gerar ARQUIVO | --conferir ARQUIVO]
    python -m src.benchmark_s6 --modo implementacao --checkpoints DIR_LOCAL [--destino DIR] [--processos N]
    python -m src.benchmark_s6 --modo definitivo --checkpoints DIR_LOCAL [--destino DIR] [--processos N] \\
        --manifesto-s4 reports/sprint4/.../manifesto.json
    python -m src.benchmark_s6 --modo verificar --checkpoints DIR_LOCAL [--destino DIR] \\
        --manifesto-s4 reports/sprint4/.../manifesto.json

O destino padrão é ``reports/sprint6/benchmark``.

**Modos.**

* ``configuracao``: gera ou confere a configuração canônica (Seção 14.4); não lê dados.
* ``implementacao``: só ``train``, 25 unidades (``r = 0``), invariante da LR fixa e
  manifesto exploratório. NÃO calcula, grava nem divulga nenhuma métrica externa:
  ``_implementacao`` não chama ``metricas``, ``consolidacao``, ``bootstrap``, ``selecao``
  nem ``figuras`` (um teste confere o código e outro, a execução). Só usa ``executor``,
  ``invariante`` e ``manifesto``. Artefatos em ``DESTINO/implementacao/``.
* ``definitivo``: ``train`` + ``validation``, 250 unidades. Antes de qualquer
  treinamento exige ambiente, configuração canônica versionada, árvore git limpa,
  aprovações P9 e ``S4_E3_REGISTRADA`` (a carga as exige antes de ler a validação),
  dados oficiais e as contagens 353/509 e 226/187/135/91. Só com as 250 unidades
  concluídas agrega (G e F), faz o bootstrap, seleciona, gera figuras e
  relatórios, varre a privacidade e grava os artefatos; o manifesto é o último.
  Falha técnica pendente deixa a execução ``incompleta``, sem relatório final.
* ``verificar``: sem treinar, relê o manifesto, confere hashes, git, código,
  ambiente e configuração, recalcula tudo a partir dos checkpoints e exige bytes
  idênticos aos versionados. ``relatorio_execucao.json`` (tempos) só tem o hash
  conferido contra o manifesto. Os artefatos exploratórios do ``implementacao``
  só têm os hashes conferidos.

**Códigos de saída (P9, Seção 14.3):** 0 concluído · 2 bloqueio de aprovação ·
3 falha técnica · 4 integridade · 5 privacidade ou vocabulário · 64 erro de uso
(``EX_USAGE``; o 2 do ``argparse`` é reservado ao bloqueio de aprovação).

**Paralelismo.** Entre unidades, por processos (``--processos``, padrão 1); o
resultado não depende dele. O ponto de entrada tem a guarda ``__main__`` exigida
pelo ``spawn`` do Windows.

``main`` aceita ``dependencias`` para os testes sintéticos (carga, git, executor e
ambiente injetáveis). A produção usa ``dependencias_de_producao()``, sem nenhum
parâmetro de fonte de dados, aprovação ou configuração na linha de comando.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy
import pandas
import scipy
import sklearn

from . import adaptadores_s4 as s4
from . import checkpoints as ck
from . import config as cfg
from . import dados as dd
from . import executor as ex
from . import invariante as inv
from . import manifesto as mf
from . import modelos
from . import privacidade as pv
from . import relatorios as rl
from . import unidades as un

MODOS_IMPLEMENTADOS = ("configuracao", "implementacao", "definitivo", "verificar")
MODOS_COM_DADOS = ("implementacao", "definitivo", "verificar")

#: Erro de argumentos da linha de comando. Fora dos códigos da P9: o 2, que o
#: ``argparse`` usaria, é reservado ao bloqueio de aprovação (P9, Seção 14.3).
CODIGO_ERRO_USO = 64

DESTINO_PADRAO = "reports/sprint6/benchmark"
SUBDIRETORIO_IMPLEMENTACAO = "implementacao"
ARQUIVO_CONFIG_VERSIONADA = s4.REPO_ROOT / "docs" / "sprint6" / "config_benchmark_p9.json"
ARQUIVO_INVARIANTE = "invariante_lr_fixa.json"
ARQUIVO_EXECUCAO = "relatorio_execucao.json"
#: Artefatos cuja regeneração não é determinística (tempos): o ``verificar`` só confere o hash contra o manifesto.
NAO_DETERMINISTICOS = (ARQUIVO_EXECUCAO,)


#: Texto da mensagem de erro por chave de saída (``erro de integridade`` é o formato desde o modo ``configuracao``).
ROTULO_DE_SAIDA = {"concluido": "concluído", "bloqueio_aprovacao": "bloqueio de aprovação",
                   "falha_tecnica": "falha técnica", "integridade": "erro de integridade",
                   "privacidade_ou_vocabulario": "erro de privacidade ou vocabulário"}


class AmbienteDivergente(RuntimeError):
    """Versão instalada diferente da fixada pela P9, Seção 13. Saída 4."""

    chave_saida = "integridade"


class PreVooRecusado(RuntimeError):
    """Condição do pré-voo do modo definitivo não satisfeita (git sujo, dados não oficiais…). Saída 4."""

    chave_saida = "integridade"


class DestinoOcupado(RuntimeError):
    """O diretório de destino já tem artefatos; nada é sobrescrito. Saída 4."""

    chave_saida = "integridade"


class ExecucaoIncompleta(RuntimeError):
    """Falha técnica pendente: nenhum agregado, seleção ou relatório final. Saída 3 (ou 4 se houver integridade)."""

    def __init__(self, mensagem: str, chave_saida: str = "falha_tecnica"):
        super().__init__(mensagem)
        self.chave_saida = chave_saida


class _ErroDeUso(Exception):
    """Argumentos inválidos; vira o código ``CODIGO_ERRO_USO``."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        raise _ErroDeUso(message)


# --------------------------------------------------------------------------- #
# Dependências (injetáveis só nos testes)
# --------------------------------------------------------------------------- #


def ambiente_instalado() -> Dict[str, str]:
    """Versões instaladas, com os nomes de ``EspecExecucao.ambiente`` (``xgboost``: ``""`` se ausente)."""
    disponivel, versao = modelos.xgboost_disponivel()
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "xgboost": versao if disponivel else ""}


def _carregar_producao(modo: str, manifesto_s4: Optional[str]) -> dd.DadosBenchmark:
    if modo == "implementacao":
        return dd.carregar_implementacao()
    return dd.carregar_definitivo(manifesto_s4)


def _carregador_dos_workers_producao(modo: str, manifesto_s4: Optional[str]) -> Callable[[], dd.DadosBenchmark]:
    """Carregador SERIALIZÁVEL (função de módulo ou ``partial``) que cada worker chama uma vez."""
    if modo == "implementacao":
        return dd.carregar_implementacao
    return functools.partial(dd.carregar_definitivo, manifesto_s4)


@dataclass(frozen=True)
class Dependencias:
    """Tudo o que o coordenador pede ao mundo. A produção usa ``dependencias_de_producao``."""

    carregar: Callable[[str, Optional[str]], dd.DadosBenchmark]
    carregador_dos_workers: Callable[[str, Optional[str]], Callable[[], dd.DadosBenchmark]]
    estado_git: Callable[[], Any]
    executar: Callable[..., ex.RelatorioExecucao]
    ambiente: Callable[[], Mapping[str, str]]
    #: Só dados oficiais valem no definitivo (P9, Seção 14.3); os testes sintéticos o desligam.
    exigir_oficial: bool = True
    #: ``None`` = ``B`` da configuração (10.000). Só os testes passam outro valor.
    n_replicas: Optional[int] = None
    config_versionada: Path = ARQUIVO_CONFIG_VERSIONADA
    #: Repositório do código (``verificar`` compara o HEAD do manifesto com o dele).
    repo: Path = s4.REPO_ROOT
    baseline: Path = inv.CAMINHO_BASELINE
    codigo: Callable[[], Mapping[str, str]] = ck.hashes_do_codigo
    descricao_do_ambiente: Callable[[], Mapping[str, Any]] = ck.descricao_do_ambiente


def dependencias_de_producao() -> Dependencias:
    return Dependencias(carregar=_carregar_producao, carregador_dos_workers=_carregador_dos_workers_producao,
                        estado_git=lambda: s4.estado_git(), executar=ex.executar, ambiente=ambiente_instalado)


# --------------------------------------------------------------------------- #
# Pré-voo
# --------------------------------------------------------------------------- #


def verificar_ambiente(dep: Dependencias) -> None:
    """P9, Seção 13: cada pacote na versão fixada; divergência aborta (integridade)."""
    instalado = dep.ambiente()
    divergentes = {nome: (instalado.get(nome, ""), esperado)
                   for nome, esperado in cfg.CONFIGURACAO.execucao.ambiente.items()
                   if instalado.get(nome) != esperado}
    if divergentes:
        raise AmbienteDivergente("ambiente diferente do fixado pela P9: " + "; ".join(
            f"{n} {o or 'ausente'} (esperado {e})" for n, (o, e) in sorted(divergentes.items())))


def conferir_configuracao(dep: Dependencias, exigir_arquivo: bool) -> List[str]:
    """A configuração gerada pelo código tem de ser idêntica à canônica versionada. Devolve notas de registro."""
    caminho = dep.config_versionada
    if caminho.is_file():
        cfg.conferir_arquivo(caminho)
        return []
    if exigir_arquivo:
        raise cfg.ConfiguracaoDivergente(f"{caminho.name}: configuração canônica versionada ausente")
    return [f"{caminho.name} ausente neste commit: configuração conferida só contra o código"]


def exigir_destino_livre(destino: Path) -> None:
    """Nenhum artefato é sobrescrito: o destino não pode ter arquivos (``implementacao/`` de outra execução, sim)."""
    if not destino.exists():
        return
    if not destino.is_dir():
        raise DestinoOcupado(f"{destino.name} não é diretório")
    ocupados = [p.name for p in destino.iterdir() if p.name != SUBDIRETORIO_IMPLEMENTACAO]
    if ocupados:
        raise DestinoOcupado(f"{destino.name} já contém artefatos ({ocupados[:3]}); nada é sobrescrito")


def exigir_pre_voo_definitivo(dep: Dependencias, dados: dd.DadosBenchmark, estado: Any) -> None:
    if dados.modo != dd.MODO_DEFINITIVO:
        raise PreVooRecusado(f"dados do modo {dados.modo!r}, não do definitivo")
    if dep.exigir_oficial and not dados.oficial:
        raise PreVooRecusado("o definitivo exige dados oficiais (oficial=True)")
    if not estado.limpo:
        raise PreVooRecusado("o definitivo exige árvore git limpa")


# --------------------------------------------------------------------------- #
# Artefatos
# --------------------------------------------------------------------------- #


def sha256_lf(bruto: bytes) -> str:
    return hashlib.sha256(bruto.replace(b"\r\n", b"\n")).hexdigest()


def ambiente_plano(descricao: Mapping[str, Any]) -> Dict[str, Any]:
    """``{python, sistema, <biblioteca>: versão…}`` de ``descricao_do_ambiente`` (sem processador nem caminho)."""
    return {"python": descricao["python"], "sistema": descricao["sistema"], **dict(descricao["versoes"])}


def gerar_artefatos_definitivos(dados: dd.DadosBenchmark, resultados: Sequence[un.ResultadoUnidade], *,
                                head: str, branch: Optional[str], descricao_ambiente: Mapping[str, Any],
                                n_replicas: Optional[int]) -> Dict[str, bytes]:
    """Todos os artefatos DETERMINÍSTICOS do definitivo (tudo menos ``relatorio_execucao.json`` e o manifesto).

    Usada pelo ``definitivo`` e pelo ``verificar``: a mesma função, os mesmos bytes. Só aqui se importam e usam as
    métricas externas (importação tardia: o modo ``implementacao`` nunca chega a elas).
    """
    from . import bootstrap as bs, consolidacao as co, figuras as fg, metricas as mt, selecao as sl

    contexto = un.ContextoExecucao.de_dados(dados)
    base = mt.BaseAvaliacao.de_contexto(contexto)
    pareados = co.parear(base, resultados)
    consolidacao = co.consolidar(pareados)
    resultado_bootstrap = bs.bootstrap(pareados, consolidacao, n_replicas)
    n_esperadas = len(un.unidades_do_modo(contexto.modo))
    entrada = sl.montar_entrada(consolidacao, falhas_tecnicas_pendentes=0, unidades_esperadas=n_esperadas,
                                unidades_concluidas=len(resultados))
    selecao = sl.selecionar(entrada)
    plano = contexto.plano.resumo(contexto.amostra)
    info_base = {"pacientes": base.n_pacientes, "nodulos": base.amostra.n}
    y = base.amostra.y
    populacao = rl.Populacao(base.n_pacientes, base.amostra.n, int((y == 0).sum()), int((y == 1).sum()),
                             base.contagem_faixa())
    identificacao = rl.Identificacao(
        modo=contexto.modo, oficial=bool(dados.oficial), sha256_configuracao=contexto.sha256_configuracao, head=head,
        branch=branch, ambiente=ambiente_plano(descricao_ambiente), sha256_entradas=dict(dados.sha256_entradas),
        n_unidades=n_esperadas)

    artefatos: Dict[str, bytes] = {}
    artefatos.update(rl.tabelas(consolidacao, resultado_bootstrap, selecao, plano))
    artefatos.update(fg.gerar_todas(consolidacao, resultado_bootstrap, info_base))
    nomes_figuras = sorted(n for n in artefatos if n.startswith("figuras/") and not n.startswith("figuras/dados/"))

    def dados_do_relatorio(hashes: Mapping[str, str]) -> rl.DadosRelatorio:
        return rl.DadosRelatorio(populacao, identificacao, consolidacao, resultado_bootstrap, entrada, selecao, plano,
                                 hashes, nomes_figuras)

    artefatos["relatorio_benchmark.json"] = mf.bytes_json(rl.relatorio_benchmark(dados_do_relatorio({})))
    hashes = {nome: sha256_lf(bruto) for nome, bruto in sorted(artefatos.items())}
    artefatos["resumo_benchmark.md"] = rl.resumo_md(dados_do_relatorio(hashes))
    artefatos["_avisos_por_modelo.json"] = json.dumps({a.modelo: a.avisos_convergencia
                                                       for a in consolidacao.avisos}).encode("utf-8")
    return artefatos


def relatorio_de_execucao(modo: str, processos: int, relatorio: ex.RelatorioExecucao,
                          resultados: Sequence[un.ResultadoUnidade]) -> Dict[str, Any]:
    """``relatorio_execucao.json``: tempos, processos, falhas, avisos e retomadas; sem mensagens nem caminhos."""
    avisos: Dict[str, int] = {}
    for r in resultados:
        avisos[r.unidade.modelo] = avisos.get(r.unidade.modelo, 0) + r.registro.avisos_convergencia
    return rl.relatorio_execucao(
        modo=modo, processos=processos, duracoes=dict(relatorio.duracoes), reaproveitadas=relatorio.reaproveitadas,
        executadas=relatorio.executadas, avisos=avisos, duracao_total_s=relatorio.duracao_total_s,
        falhas=[{"unidade": f.unidade.rotulo, "chave_saida": f.chave_saida, "categoria": f.categoria}
                for f in relatorio.falhas])


def _varrer_e_gravar(destino: Path, artefatos: Mapping[str, bytes], ids: Sequence[str]) -> None:
    """Varre TODOS os artefatos antes de gravar qualquer um; depois grava de forma atômica e varre o diretório."""
    for nome, bruto in sorted(artefatos.items()):
        pv.varrer_bytes(nome, bruto, ids)
    for nome, bruto in sorted(artefatos.items()):
        mf.gravar_atomico(destino / nome, bruto)


# --------------------------------------------------------------------------- #
# Modos
# --------------------------------------------------------------------------- #


def _executar_unidades(args: argparse.Namespace, dep: Dependencias, dados: dd.DadosBenchmark, modo: str
                       ) -> Tuple[ck.SessaoCheckpoints, ex.RelatorioExecucao]:
    sessao = ck.SessaoCheckpoints(args.checkpoints, dados)
    relatorio = dep.executar(sessao, dep.carregador_dos_workers(modo, args.manifesto_s4), processos=args.processos)
    if relatorio.falhas:
        chaves = {f.chave_saida for f in relatorio.falhas}
        raise ExecucaoIncompleta(
            f"execução incompleta: {len(relatorio.falhas)} unidade(s) com falha (primeiras: "
            f"{[f.unidade.rotulo + ':' + f.categoria for f in relatorio.falhas[:3]]}); nenhum resultado agregado",
            "integridade" if "integridade" in chaves else "falha_tecnica")
    if not relatorio.concluida:
        raise ExecucaoIncompleta("execução incompleta: unidades sem resultado")
    return sessao, relatorio


def _implementacao(args: argparse.Namespace, dep: Dependencias) -> int:
    verificar_ambiente(dep)
    notas = conferir_configuracao(dep, exigir_arquivo=False)
    estado = dep.estado_git()
    destino = Path(args.destino) / SUBDIRETORIO_IMPLEMENTACAO
    exigir_destino_livre(destino)
    dados = dep.carregar("implementacao", None)
    if dados.modo != dd.MODO_IMPLEMENTACAO or "validation" in dados.particoes or "test" in dados.particoes:
        raise PreVooRecusado("o modo implementacao lê só a partição train")
    sessao, relatorio = _executar_unidades(args, dep, dados, "implementacao")
    # Sem métricas externas: só o invariante da LR fixa (parâmetros ajustados) e o relatório de execução.
    invariante = inv.verificar(dados.X(), dados.y(), dep.baseline)
    artefatos = {
        ARQUIVO_EXECUCAO: mf.bytes_json(relatorio_de_execucao("implementacao", args.processos, relatorio,
                                                              relatorio.resultados)),
        ARQUIVO_INVARIANTE: mf.bytes_json(invariante),
    }
    ids = pv.identificadores_da_amostra(sessao.contexto.amostra)
    _varrer_e_gravar(destino, artefatos, ids)
    manifesto = mf.montar_manifesto(
        modo="implementacao", oficial=bool(dados.oficial), elegivel=False,
        motivos=["modo implementacao é exploratório: sem métricas externas, nunca elegível", *notas],
        estado_git=estado, sha256_configuracao=sessao.contexto.sha256_configuracao,
        entradas=dict(dados.sha256_entradas), codigo=dep.codigo(), ambiente=dep.descricao_do_ambiente(),
        saidas=mf.hashes_de_saidas(destino, artefatos))
    mf.escrever_manifesto(destino, manifesto)
    pv.varrer_diretorio(destino, ids)
    print(f"implementacao concluida: {len(relatorio.resultados)} unidades; invariante da LR fixa ok; destino "
          f"{SUBDIRETORIO_IMPLEMENTACAO}/")
    return cfg.CONFIGURACAO.execucao.codigos_saida["concluido"]


def _definitivo(args: argparse.Namespace, dep: Dependencias) -> int:
    verificar_ambiente(dep)
    conferir_configuracao(dep, exigir_arquivo=True)
    estado = dep.estado_git()
    if not estado.limpo:
        raise PreVooRecusado("o definitivo exige árvore git limpa")
    destino = Path(args.destino)
    exigir_destino_livre(destino)
    dados = dep.carregar("definitivo", args.manifesto_s4)  # as aprovações são exigidas aqui, antes de ler a validação
    exigir_pre_voo_definitivo(dep, dados, estado)
    sessao, relatorio = _executar_unidades(args, dep, dados, "definitivo")
    atual = dep.estado_git()
    if atual.head != estado.head or not atual.limpo:
        raise PreVooRecusado("o git mudou durante a execução; nenhum artefato é gerado")
    descricao = dep.descricao_do_ambiente()
    artefatos = gerar_artefatos_definitivos(dados, relatorio.resultados, head=estado.head, branch=estado.branch,
                                            descricao_ambiente=descricao, n_replicas=dep.n_replicas)
    artefatos.pop("_avisos_por_modelo.json")
    artefatos[ARQUIVO_EXECUCAO] = mf.bytes_json(relatorio_de_execucao("definitivo", args.processos, relatorio,
                                                                       relatorio.resultados))
    ids = pv.identificadores_da_amostra(sessao.contexto.amostra)
    _varrer_e_gravar(destino, artefatos, ids)
    manifesto = mf.montar_manifesto(
        modo="definitivo", oficial=bool(dados.oficial), elegivel=bool(dados.oficial),
        motivos=[] if dados.oficial else ["dados não oficiais (somente testes sintéticos)"], estado_git=estado,
        sha256_configuracao=sessao.contexto.sha256_configuracao, entradas=dict(dados.sha256_entradas),
        codigo=dep.codigo(), ambiente=descricao, saidas=mf.hashes_de_saidas(destino, artefatos))
    mf.escrever_manifesto(destino, manifesto)
    pv.varrer_diretorio(destino / "figuras", ids)
    print(f"definitivo concluido: {len(relatorio.resultados)} unidades; {len(artefatos) + 1} artefatos")
    return cfg.CONFIGURACAO.execucao.codigos_saida["concluido"]


def _comparar(destino: Path, regenerados: Mapping[str, bytes], manifesto: Mapping[str, Any]) -> None:
    for nome, bruto in sorted(regenerados.items()):
        if nome in NAO_DETERMINISTICOS:
            continue
        versionado = (destino / nome).read_bytes()
        if not nome.endswith((".png",)):  # o checkout do Windows pode trocar LF por CRLF em texto
            versionado = versionado.replace(b"\r\n", b"\n")
        if versionado != bruto:
            raise mf.ManifestoInvalido(f"{nome}: o recálculo a partir dos checkpoints difere do versionado")
    faltam = set(manifesto["saidas"]) - set(regenerados) - set(NAO_DETERMINISTICOS)
    if faltam:
        raise mf.ManifestoInvalido(f"saídas do manifesto sem recálculo: {sorted(faltam)}")


def _verificar(args: argparse.Namespace, dep: Dependencias) -> int:
    verificar_ambiente(dep)
    destino = Path(args.destino)
    exploratorio = (destino / SUBDIRETORIO_IMPLEMENTACAO / mf.ARQUIVO_MANIFESTO).is_file() and \
        not (destino / mf.ARQUIVO_MANIFESTO).is_file()
    if exploratorio:  # só os hashes: os tempos do relatório de execução não se recalculam
        destino = destino / SUBDIRETORIO_IMPLEMENTACAO
    manifesto = mf.ler_manifesto(destino)
    mf.verificar_saidas(destino, manifesto)
    mf.verificar_git(manifesto, dep.repo)
    mf.verificar_ambiente_e_codigo(manifesto, dep.codigo(), dep.descricao_do_ambiente())
    if manifesto["modo"] != "definitivo":
        print(f"verificar: manifesto exploratório ({manifesto['modo']}); hashes, git, código e ambiente conferem")
        return cfg.CONFIGURACAO.execucao.codigos_saida["concluido"]
    conferir_configuracao(dep, exigir_arquivo=True)
    dados = dep.carregar("definitivo", args.manifesto_s4)
    sessao = ck.SessaoCheckpoints(args.checkpoints, dados)
    retomada = sessao.verificar()
    if retomada.a_executar:
        raise ExecucaoIncompleta(f"{len(retomada.a_executar)} unidade(s) sem checkpoint concluído; nada a verificar")
    resultados = [sessao.carregar(u) for u in sessao.contexto.unidades()]
    regenerados = gerar_artefatos_definitivos(
        dados, resultados, head=manifesto["git"]["head"], branch=manifesto["git"]["branch"],
        descricao_ambiente=manifesto["ambiente"], n_replicas=dep.n_replicas)
    regenerados.pop("_avisos_por_modelo.json")
    _comparar(destino, regenerados, manifesto)
    pv.varrer_diretorio(destino, pv.identificadores_da_amostra(sessao.contexto.amostra))
    print(f"verificar: {len(manifesto['saidas'])} saídas conferem; recálculo idêntico; git, código e ambiente ok")
    return cfg.CONFIGURACAO.execucao.codigos_saida["concluido"]


def _configuracao(args: argparse.Namespace) -> int:
    codigos = cfg.CONFIGURACAO.execucao.codigos_saida
    s4.verificar_arquivos_s4()
    if args.gerar is not None:
        sha, criado = cfg.gerar_arquivo(args.gerar)
        print(f"{'gerado' if criado else 'já existente e idêntico'}: {args.gerar}")
    elif args.conferir is not None:
        sha = cfg.conferir_arquivo(args.conferir)
        print(f"conferido: {args.conferir}")
    else:
        sha = cfg.sha256_configuracao()
    print(f"sha256_configuracao={sha}")
    return codigos["concluido"]


# --------------------------------------------------------------------------- #
# Linha de comando
# --------------------------------------------------------------------------- #


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="python -m src.benchmark_s6",
                     description="Benchmark de classificadores da Sprint 6 (pré-registro P9).")
    parser.add_argument("--modo", required=True, choices=MODOS_IMPLEMENTADOS, help="modo de execução")
    acao = parser.add_mutually_exclusive_group()
    acao.add_argument("--gerar", metavar="ARQUIVO", type=Path,
                      help="(configuracao) grava o JSON canônico se ausente; se existir, só confere")
    acao.add_argument("--conferir", metavar="ARQUIVO", type=Path,
                      help="(configuracao) exige que ARQUIVO seja a configuração canônica do código")
    parser.add_argument("--checkpoints", metavar="DIR_LOCAL", type=Path,
                        help="diretório local, fora do repositório e do git (obrigatório fora do modo configuracao)")
    parser.add_argument("--destino", metavar="DIR", type=Path, default=Path(DESTINO_PADRAO),
                        help=f"diretório dos artefatos (padrão: {DESTINO_PADRAO})")
    parser.add_argument("--processos", metavar="N", type=int, default=1, help="processos entre unidades (padrão: 1)")
    parser.add_argument("--manifesto-s4", metavar="CAMINHO", dest="manifesto_s4", default=None,
                        help="manifesto definitivo da Etapa 3 da S4, relativo ao repositório (definitivo e verificar)")
    return parser


def _validar_uso(args: argparse.Namespace) -> None:
    if args.modo == "configuracao":
        if args.checkpoints is not None or args.processos != 1 or args.manifesto_s4 is not None:
            raise _ErroDeUso("--checkpoints, --processos e --manifesto-s4 não valem no modo configuracao")
        return
    if args.gerar is not None or args.conferir is not None:
        raise _ErroDeUso("--gerar e --conferir só valem no modo configuracao")
    if args.checkpoints is None:
        raise _ErroDeUso(f"o modo {args.modo} exige --checkpoints")
    if args.processos < 1:
        raise _ErroDeUso("--processos deve ser inteiro ≥ 1")
    if args.modo == "implementacao" and args.manifesto_s4 is not None:
        raise _ErroDeUso("o modo implementacao não usa --manifesto-s4 (não lê validação)")


def codigo_de_saida(exc: BaseException) -> int:
    """Código da P9 para uma exceção: ``chave_saida`` da exceção; sem ela, falha técnica (3)."""
    codigos = cfg.CONFIGURACAO.execucao.codigos_saida
    if isinstance(exc, (cfg.ConfiguracaoDivergente, cfg.ConfiguracaoInvalida, s4.ArquivoS4Divergente,
                        FileNotFoundError)):
        return codigos["integridade"]
    chave = getattr(exc, "chave_saida", None)
    return codigos[chave] if chave in codigos else codigos["falha_tecnica"]


def main(argv: Optional[Sequence[str]] = None, dependencias: Optional[Dependencias] = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        _validar_uso(args)
    except _ErroDeUso as exc:
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: erro: {exc}", file=sys.stderr)
        return CODIGO_ERRO_USO
    dep = dependencias or dependencias_de_producao()
    try:
        if args.modo == "configuracao":
            return _configuracao(args)
        if args.modo == "implementacao":
            return _implementacao(args, dep)
        if args.modo == "definitivo":
            return _definitivo(args, dep)
        return _verificar(args, dep)
    except Exception as exc:  # toda recusa vira um código da P9; nenhuma passa em silêncio
        codigo = codigo_de_saida(exc)
        chave = next(k for k, v in cfg.CONFIGURACAO.execucao.codigos_saida.items() if v == codigo)
        print(f"{ROTULO_DE_SAIDA[chave]} ({type(exc).__name__}): {exc}", file=sys.stderr)
        return codigo


__all__ = [
    "AmbienteDivergente", "CODIGO_ERRO_USO", "Dependencias", "DestinoOcupado", "ExecucaoIncompleta",
    "MODOS_IMPLEMENTADOS", "PreVooRecusado", "ambiente_instalado", "codigo_de_saida", "conferir_configuracao",
    "dependencias_de_producao", "gerar_artefatos_definitivos", "main", "relatorio_de_execucao", "verificar_ambiente",
]
