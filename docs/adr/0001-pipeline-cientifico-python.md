# 0001 - Manter o pipeline científico modular em Python

Status: Aceito

Data: 2026-11-23

Revisado em: 2026-11-24

Registro retrospectivo de uma decisão já implementada. A data indica sua documentação neste ADR, não a adoção original nem a aprovação de uma arquitetura nova.

## Contexto

O projeto combina seleção/máscaras, processamento de imagem, estatística e avaliação de classificadores. Os produtores existentes são notebooks, scripts e módulos Python, com entradas e saídas de etapas científicas. É necessário executar e testar etapas de modo rastreável, respeitando protocolos diferentes. Não há aplicação web ou serviço hospedado implementado neste checkout.

## Decisão

Manter a arquitetura de pipeline local em Python, separando `src/radiomics/`, `src/incremental/` e `src/benchmark_s6/`. Usar a CLI `python -m src.benchmark_s6` como ponto de entrada do P9 e preservar notebooks para as análises históricas. Usar Conda/pip com `environment.yml`; para o P9 definitivo, prevalecem as versões exatas do pré-registro/configuração, inclusive Python 3.9.13 e scikit-learn 1.3.2.

A opção reaproveita o ecossistema de imagem e modelagem já utilizado e permite testes sintéticos dos componentes. A arquitetura permanece voltada à execução científica local, sem acrescentar uma camada de aplicação que o fluxo atual não utiliza.

## Alternativa descartada

Concentrar toda a execução em um único notebook facilitaria a exploração inicial, mas misturaria estado de sessão, parâmetros e etapas de produção. A alternativa foi descartada nesta análise arquitetural porque módulos e CLI permitem testes isolados, retomada e conferência de artefatos sem depender da ordem manual de células. Os notebooks históricos permanecem como evidência de suas etapas.

## Consequências

**Benefícios:** módulos especializados permitem manutenção e testes isolados; a CLI registra modos e argumentos; bibliotecas científicas atendem às etapas existentes sem adicionar outra stack.

**Custos e limites:** o operador precisa de terminal e ambiente científico; notebooks históricos têm dependências adicionais; o YAML não fixa todas as transitivas nem todos os ambientes históricos. Não existe uma interface de uso clínico ou um deploy pronto, e não se promete execução idêntica entre plataformas.

**Rastreabilidade:** [CLI](../../src/benchmark_s6/__main__.py), [coordenador](../../src/benchmark_s6/coordenador.py), [ambiente](../../environment.yml), [configuração](../sprint6/config_benchmark_p9.json), [RF-04/RF-05 e RNF-05/RNF-06](../requisitos.md).
