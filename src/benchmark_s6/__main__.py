"""Ponto de entrada: ``python -m src.benchmark_s6``.

A guarda ``__main__`` é obrigatória: com ``multiprocessing`` em ``spawn`` (o
único método do Windows), cada worker reimporta o módulo principal, e sem ela
o coordenador seria iniciado de novo dentro de cada worker.
"""

import sys

from .coordenador import main

if __name__ == "__main__":
    sys.exit(main())
