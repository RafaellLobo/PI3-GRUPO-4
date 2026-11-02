"""Benchmark de classificadores da Sprint 6 — protocolo complementar à S4.

Implementa o pré-registro P9 (``docs/sprint6/preregistro_p9_benchmark.md``):
LR fixa e com ajuste, RF, XGBoost e SVM, nas três características do
baseline, em 353 pacientes / 509 nódulos, por CV aninhada 10 × 5 agrupada por
paciente. O teste nunca é lido.

Este pacote NÃO modifica ``src/incremental``; o único ponto de contato com a
S4 é ``adaptadores_s4``, que importa apenas nomes públicos.

Módulos implementados nesta etapa:

* ``adaptadores_s4`` — nomes públicos da S4 e SHA-256 fixados dos arquivos;
* ``config`` — configuração canônica imutável, JSON canônico e SHA-256;
* ``coordenador`` — comando ``python -m src.benchmark_s6``; por ora, só o
  modo ``configuracao``, que não lê dados nem ajusta modelos.
"""
