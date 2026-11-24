# Requisitos do pipeline radiômico PI3 Grupo 4

Registro em 23/11/2026, revisado em 24/11/2026, baseado nos módulos, scripts e protocolos existentes. O sistema é um pipeline científico operado pela equipe via terminal/notebooks; não há requisito de cadastro, interface web ou atendimento clínico. Os critérios mensuráveis abaixo definem o comportamento esperado no escopo indicado, sem criar metas de latência, desempenho clínico ou novas análises.

Prioridade **alta**: sua ausência compromete a validade do estudo, a proteção dos dados ou a preservação dos resultados. **Média**: capacidade de apoio, comunicação ou reconstrução de etapas históricas. **Baixa**: consulta auxiliar que pode ser adiada sem retirar as guardas obrigatórias da execução científica.

As prioridades são de planejamento; não flexibilizam regras do protocolo. Por isso, a geração do split histórico (RF-03) e a consulta avulsa da configuração (RF-04) têm prioridade menor que o isolamento por paciente (RNF-01) e a verificação obrigatória de versões no modo definitivo (RNF-05). As regras de domínio permanecem altas porque definem quais resultados são admissíveis.

## Requisitos funcionais

| ID | Prioridade | O que o sistema deve fazer | Critério de aceitação e evidência |
|---|---|---|---|
| RF-01 | Média | Selecionar nódulos e consolidar máscaras pelas regras do piloto. | Aplicar as regras documentadas e consenso de pelo menos 50% dos anotadores, preservando a identificação do nódulo. [Seleção/máscaras](sprint2/pipeline_selecao_mascaras.md), [código](../src/radiomics/). |
| RF-02 | Alta | Construir a tabela de modelagem com alvo binário e as características extraídas. | Relacionar nódulo/paciente e rótulos conforme RD-01; registrar descartes e origem sem publicar a tabela individual. [Construtor](../scripts/build_modeling_table.py), [dados](../data/README.md). |
| RF-03 | Média | Disponibilizar split por paciente para as etapas supervisionadas. | Um paciente pertence a uma única partição; preservar o split oficial existente. [Script](../scripts/make_patient_split.py), [baseline](sprint3/baseline_regressao_logistica.md). |
| RF-04 | Baixa | Conferir a configuração canônica do benchmark por CLI sem carregar dados de pacientes. | `--modo configuracao --conferir` retorna 0 para o JSON canônico e informa SHA-256; divergência deve ser recusada. [Coordenador](../src/benchmark_s6/coordenador.py), [configuração](sprint6/config_benchmark_p9.json). |
| RF-05 | Alta | Avaliar os cinco modelos do P9 nas mesmas divisões, com ajuste interno quando previsto. | Modo definitivo organiza 5 modelos × 10 repetições × 5 dobras externas = 250 unidades; ajuste de hiperparâmetros usa somente treino externo. [Pré-registro](sprint6/preregistro_p9_benchmark.md), [executor](../src/benchmark_s6/executor.py). |
| RF-06 | Alta | Agregar AUC em G e F, incerteza e contrastes no protocolo P9. | Gerar os dois desfechos e bootstrap por paciente de 10.000 réplicas, sem reajuste de modelos nas réplicas; distinguir intervalos descritivos dos contrastes corrigidos. [Configuração](sprint6/config_benchmark_p9.json), [bootstrap](../src/benchmark_s6/bootstrap.py). |
| RF-07 | Alta | Selecionar o algoritmo pela regra predefinida e registrar a decisão. | Manter LR com ajuste como padrão; exigir todos os critérios C1–C4 do pré-registro para recomendar desafiante, sem escolher apenas pela maior AUC pontual. [Seleção](../src/benchmark_s6/selecao.py), [testes sintéticos](../tests/benchmark_s6/test_selecao.py). |
| RF-08 | Média | Exportar resultados agregados, figuras e manifesto, e permitir sua conferência. | Registrar saídas e hashes; gravar o manifesto após as saídas válidas; `verificar` deve conferir hashes e recálculo a partir de checkpoints sob suas guardas. [Manifesto](../src/benchmark_s6/manifesto.py), [coordenador](../src/benchmark_s6/coordenador.py). |

## Requisitos não funcionais

| ID | Prioridade | Qualidade e medida verificável | Evidência |
|---|---|---|---|
| RNF-01 | Alta | Isolamento entre grupos: interseção de pacientes entre treino e avaliação igual a **zero**, em cada divisão validada. | [Dobras P9](../src/benchmark_s6/folds.py), [testes](../tests/benchmark_s6/test_folds.py). |
| RNF-02 | Alta | Rastreabilidade: entradas/configurações/saídas controladas pelo protocolo devem ter hashes SHA-256 de **64 caracteres hexadecimais** e identificação de versão; mudanças incompatíveis invalidam a retomada. | [Checkpoints](../src/benchmark_s6/checkpoints.py), [manifesto](../src/benchmark_s6/manifesto.py). |
| RNF-03 | Alta | Privacidade das novas saídas P9: **zero identificadores individuais detectados** pelas regras de varredura nos artefatos versionáveis; violação retorna código **5**. Checkpoints/predições individuais ficam fora do Git. | [Privacidade](../src/benchmark_s6/privacidade.py), [.gitignore](../.gitignore). A regra não apaga as exceções históricas da emenda E10/E11. |
| RNF-04 | Alta | Preservação: **zero sobrescritas** de artefatos congelados pelo coordenador; destino ocupado deve ser recusado com código **4**. | `exigir_destino_livre` no [coordenador](../src/benchmark_s6/coordenador.py). |
| RNF-05 | Alta | Compatibilidade do P9 definitivo: conferir **as sete versões** declaradas na configuração (Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib e XGBoost); divergência bloqueia a execução com código **4**. | [Configuração](sprint6/config_benchmark_p9.json), `verificar_ambiente` no [coordenador](../src/benchmark_s6/coordenador.py). Não promete igualdade entre plataformas. |
| RNF-06 | Média | Verificabilidade mínima sem dados privados: ajuda da CLI, conferência canônica e suíte sintética de seleção devem terminar com código **0**, sem treinamento ou tabelas individuais. | Comandos e resultado da verificação local no [README](../README.md); [testes](../tests/benchmark_s6/test_selecao.py). |

## Requisitos de domínio

| ID | Prioridade | Regra obrigatória | Evidência |
|---|---|---|---|
| RD-01 | Alta | O alvo é suspeição radiológica: mediana ≤ 2 → classe 0; ≥ 4 → classe 1; 2,5/3/3,5 ficam fora da modelagem binária. Não equivale a histopatologia. | [Política de coorte/alvo](protocolo_coorte_target_sprint2.md). |
| RD-02 | Alta | No P9, somente treino+validação entram no ajuste/avaliação: **509 nódulos/353 pacientes** e três características de forma. O teste histórico já consultado não vira evidência independente. | [Pré-registro](sprint6/preregistro_p9_benchmark.md), [emenda](sprint4/emenda_01_auditoria_teste.md). |
| RD-03 | Alta | F é subconjunto de G: **12 ≤ diâmetro < 30 mm**, **226 nódulos/187 pacientes**; recorta apenas a avaliação. Não representar G/F como faixas disjuntas. | [Configuração](sprint6/config_benchmark_p9.json), [relatório](../reports/sprint6/benchmark/resumo_benchmark.md). |
| RD-04 | Alta | C1 exige ΔAUC G ≥ **0,02**; C2 exige ΔG positivo em ≥ **80%** das repetições válidas; C3 exige F estimável e ΔF ≥ **0**; C4 exige ausência de falha técnica pendente/aviso de convergência e G estimável. Todos são necessários. | Seção 10 do [pré-registro](sprint6/preregistro_p9_benchmark.md); o IC não integra essa decisão. |
| RD-05 | Alta | Uma execução definitiva depende dos **dois registros** de aprovação `P9` e `S4_E3_REGISTRADA`, da configuração e das guardas; resultado exploratório não pode ser promovido automaticamente a definitivo. | Seção 19 do [pré-registro](sprint6/preregistro_p9_benchmark.md), [coordenador](../src/benchmark_s6/coordenador.py). |

## Decisões relacionadas

[ADR 0001](adr/0001-pipeline-cientifico-python.md), [ADR 0002](adr/0002-artefatos-arquivos-dados-locais.md) e [ADR 0003](adr/0003-validacao-por-paciente.md). Requisitos e ADRs descrevem o projeto existente; não aprovam mudanças de coorte, modelos, dados ou protocolo.
