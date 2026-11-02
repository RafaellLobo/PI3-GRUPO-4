"""Comando coordenador do benchmark S6 (P9, Seção 14.3).

Nesta etapa só existe o modo ``configuracao`` (P9, Seção 14.4): gera ou
confere a configuração canônica e imprime seu SHA-256. Não lê dados, não
ajusta modelos e não consulta aprovações. Os modos ``implementacao``,
``definitivo`` e ``verificar`` ainda não existem e são recusados pelo parser.

Uso, a partir da raiz do repositório::

    python -m src.benchmark_s6 --modo configuracao
    python -m src.benchmark_s6 --modo configuracao --gerar docs/sprint6/config_benchmark_p9.json
    python -m src.benchmark_s6 --modo configuracao --conferir docs/sprint6/config_benchmark_p9.json

Códigos de saída (P9, Seção 14.3): 0 concluído; 4 integridade (arquivo
divergente da configuração do código, arquivo da S4 alterado ou
configuração inválida). Erro de argumentos sai com 64 (``EX_USAGE`` de
``sysexits.h``), e não com o 2 do ``argparse``, que a P9 reserva ao bloqueio
de aprovação.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import adaptadores_s4 as s4
from . import config as cfg

MODOS_IMPLEMENTADOS = ("configuracao",)

#: Erro de argumentos da linha de comando. Fora dos códigos da P9: o 2, que o
#: ``argparse`` usaria, é reservado ao bloqueio de aprovação (P9, Seção 14.3).
CODIGO_ERRO_USO = 64


class _ErroDeUso(Exception):
    """Argumentos inválidos; vira o código ``CODIGO_ERRO_USO``."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        raise _ErroDeUso(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m src.benchmark_s6",
        description="Benchmark de classificadores da Sprint 6 (pré-registro P9).")
    parser.add_argument("--modo", required=True, choices=MODOS_IMPLEMENTADOS,
                        help="modo de execução; nesta etapa, só 'configuracao'")
    acao = parser.add_mutually_exclusive_group()
    acao.add_argument("--gerar", metavar="ARQUIVO", type=Path,
                      help="grava o JSON canônico se ausente; se existir, só confere")
    acao.add_argument("--conferir", metavar="ARQUIVO", type=Path,
                      help="exige que ARQUIVO seja a configuração canônica do código")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except _ErroDeUso as exc:
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: erro: {exc}", file=sys.stderr)
        return CODIGO_ERRO_USO
    codigos = cfg.CONFIGURACAO.execucao.codigos_saida
    try:
        s4.verificar_arquivos_s4()
        if args.gerar is not None:
            sha, criado = cfg.gerar_arquivo(args.gerar)
            print(f"{'gerado' if criado else 'já existente e idêntico'}: {args.gerar}")
        elif args.conferir is not None:
            sha = cfg.conferir_arquivo(args.conferir)
            print(f"conferido: {args.conferir}")
        else:
            sha = cfg.sha256_configuracao()
    except (cfg.ConfiguracaoDivergente, cfg.ConfiguracaoInvalida, s4.ArquivoS4Divergente,
            FileNotFoundError) as exc:
        print(f"erro de integridade: {exc}", file=sys.stderr)
        return codigos["integridade"]
    print(f"sha256_configuracao={sha}")
    return codigos["concluido"]
