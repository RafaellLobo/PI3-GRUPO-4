# PI3 — Grupo 4: Radiômica e Suspeição Radiológica de Nódulos Pulmonares

## Sobre o projeto

Projeto acadêmico de radiômica aplicada a imagens de Tomografia Computadorizada (TC) de nódulos pulmonares do LIDC-IDRI. O repositório reúne o piloto de extração da Sprint 2, o baseline de Regressão Logística da Sprint 3 e a infraestrutura e os resultados exploratórios da Sprint 4.

O alvo é a **suspeição radiológica de malignidade**, derivada da mediana dos escores de radiologistas (`malignancy`, escala de 1 a 5). **Não representa malignidade confirmada por histopatologia nem diagnóstico de câncer.** Pela [política de coorte e alvo](docs/protocolo_coorte_target_sprint2.md), medianas ≤ 2 recebem classe 0, medianas ≥ 4 recebem classe 1 e medianas 2,5 / 3,0 / 3,5 ficam fora da modelagem binária. O LIDC-IDRI não fornece classificação Lung-RADS nativa.

## Objetivo

Investigar a extração e a estabilidade de atributos radiômicos e sua capacidade de discriminar a suspeição radiológica dos nódulos. Na Sprint 4, a questão é se a radiômica acrescenta discriminação além do tamanho do nódulo; os resultados existentes ainda não sustentam uma conclusão sobre esse valor incremental.

## Estado das etapas e limites de interpretação

| Sprint | Entregas existentes | Limites |
|---|---|---|
| 2 | Seleção de nódulos, máscaras por consenso, extração piloto, comparação `binCount = 64` × `binWidth = 25` e verificação populacional | Piloto de 44 nódulos; a comparação mede sensibilidade à discretização, não superioridade preditiva. |
| 3 | Reconstrução do alvo, split por paciente e baseline com volume, diâmetro e esfericidade | Base modelável de 607 nódulos / 416 pacientes; cobertura inferior à coorte teórica de 616 / 422. |
| 4 | Verificação de M0/M1, reprodução e auditoria da seleção de features, piloto exploratório (1 × 5) e CV repetida exploratória (10 × 5) | Os artefatos exploratórios presentes usam somente o treino oficial (417 nódulos / 291 pacientes). Não foram encontradas neste checkout evidências de execução das Etapas 1–5 oficiais; o protocolo versionado continua como proposta. |

**O conjunto oficial de teste já foi utilizado.** Esse uso está documentado no [baseline da Sprint 3](docs/sprint3/baseline_regressao_logistica.md); o [protocolo da Sprint 4](docs/sprint4/protocolo_valor_incremental.md), Seção 4.1, também referencia desempenho previamente observado de M0/M1 nesse split no Experimento 13. Conforme o histórico informado pelo grupo, o uso anterior inclui ainda os Experimentos 10, 11, 12, 14 e 17. As evidências desses usos informados pelo grupo ainda precisam ser localizadas e consolidadas na auditoria; essa informação não equivale à verificação dos respectivos registros experimentais no checkout. O teste não é mais um conjunto intocado: sua reutilização não constitui nova evidência independente para hipóteses desenvolvidas após esse uso, nem validação externa. A separação por paciente deve ser preservada, com esse uso prévio declarado nas avaliações.

Nos resultados exploratórios da Sprint 4 presentes no checkout analisado, não foram encontrados intervalo de confiança, controle por permutação, análise com spline ou avaliação nas partições oficiais de validação e teste. Esses artefatos não demonstram independência em relação ao tamanho nem utilidade clínica. As decisões científicas e pendências documentadas estão no [checkpoint do Cartão 4](docs/sprint4/estado_cartao4.md).

Resultados e documentos das Sprints seguintes e dos experimentos posteriores precisam ser localizados, indexados e vinculados ao repositório; sua ausência neste checkout não comprova que as atividades não tenham sido realizadas.

## Estrutura do repositório

```text
.
├── data/                  # Entradas e tabelas derivadas locais; README e placeholder versionados
├── docs/                  # Documentação técnica
│   ├── sprint2/           # Piloto e pipeline de seleção e máscaras
│   ├── sprint3/           # Baseline e convenção de identificação
│   └── sprint4/           # Proposta de protocolo e checkpoint do Cartão 4
├── notebooks/
│   └── sprint2/           # Quatro notebooks, incluindo um histórico em historico/
├── reports/
│   ├── sprint2/           # Configurações, validações, features, comparação e históricos
│   ├── sprint3/baseline/  # Split por paciente, métricas e matriz de confusão
│   └── sprint4/           # Referências M0/M1, seleção e relatórios exploratórios
├── scripts/               # Inspeção, validação, manifesto, tabela de modelagem e baseline
│   └── sprint4/           # Verificação de referências e reprodução da seleção
├── src/
│   ├── radiomics/         # Seleção, máscaras e identificadores de nódulos
│   └── incremental/       # Configuração, guardas, dados, folds, seleção, modelos e rastreabilidade
├── tests/incremental/     # Testes da infraestrutura da Sprint 4
├── .gitattributes         # EOL determinístico para artefatos científicos
├── .gitignore
├── environment.yml        # Especificação do ambiente Conda
├── LICENSE
└── README.md
```

## Documentação de referência

- [Índice de experimentos](docs/indice_experimentos.md): centraliza objetivos, produtores, artefatos e pendências de rastreabilidade das atividades documentadas.
- [Guia de reprodução](docs/guia_reproducao.md): reúne ambientes, entradas, sequência de execução, saídas e limitações de reprodução das Sprints 2–4.
- [Coorte e variável-alvo](docs/protocolo_coorte_target_sprint2.md): política científica de elegibilidade, alvo e divisão dos dados.
- [Piloto radiômico da Sprint 2](docs/sprint2/piloto_radiomico.md): referência canônica do piloto, resultados, limitações e mapa dos quatro notebooks; [pipeline de seleção e máscaras](docs/sprint2/pipeline_selecao_mascaras.md): comportamento extraído do notebook e divergências históricas documentadas.
- [Baseline da Sprint 3](docs/sprint3/baseline_regressao_logistica.md): dados, modelo, split, métricas e limitações; [identificação de nódulos](docs/sprint3/convencao_identificacao_nodulos.md): formatos históricos e chave canônica `(scan_id, original_nodule_idx)`.
- [Checkpoint do Cartão 4](docs/sprint4/estado_cartao4.md): estado de execução, resultados exploratórios e pendências da Sprint 4. A [proposta de protocolo](docs/sprint4/protocolo_valor_incremental.md) aguarda revisão do grupo e do professor; o checkpoint registra avanços posteriores e trechos da proposta que precisam de revisão.
- [Auditoria da seleção](reports/sprint4/feature_selection/auditoria_reproducao_juan.md) e [comparação com o pacote de Juan](reports/sprint4/feature_selection/comparacao_pacote_juan.md): proveniência, divergências e distinção entre evidências verificadas e declarações externas.
- [Validação inicial do LIDC-IDRI](docs/lidc_initial_validation.md): inspeção DICOM/XML e configuração local do `pylidc`.

## Código, experimentos e resultados disponíveis

| Etapa | Código ou notebook | Artefatos e disponibilidade |
|---|---|---|
| S2 — extração piloto | [Notebook v2](notebooks/sprint2/PI3_G4_sprint2_v2_extração_piloto.ipynb); biblioteca em [src/radiomics/](src/radiomics/) | [Configurações](reports/sprint2/config/), [features CSV/Parquet](reports/sprint2/features/) e [validações](reports/sprint2/validacao/) |
| S2 — discretização | [Comparação binCount/binWidth](notebooks/sprint2/PI3_G4_comparacao_binCount_binWidth.ipynb) | [Comparação e proveniência](reports/sprint2/comparacao/comparacao_v2_v3_discretizacao.json) |
| S2 — população | [Verificação do valor central 3](notebooks/sprint2/PI3_G4_verificacao_populacional_valor3.ipynb) | [Verificação populacional](reports/sprint2/validacao/verificacao_populacional_valor_central_3.json) |
| S3 — tabela de modelagem | [build_modeling_table.py](scripts/build_modeling_table.py) | Saída local não versionada: `data/modeling_table_sprint3.csv` |
| S3 — split por paciente | [make_patient_split.py](scripts/make_patient_split.py) | [patient_split.csv](reports/sprint3/baseline/patient_split.csv) |
| S3 — baseline | [run_baseline.py](scripts/run_baseline.py) | [Métricas](reports/sprint3/baseline/baseline_metrics.json) e [matriz de confusão](reports/sprint3/baseline/confusion_matrix.csv) |
| S4 — referências M0/M1 | [validate_reference_models.py](scripts/sprint4/validate_reference_models.py) | [Verificação dos modelos de referência](reports/sprint4/reference_models/validacao_modelos_referencia.json) |
| S4 — reprodução da seleção | [reproduce_juan_feature_selection.py](scripts/sprint4/reproduce_juan_feature_selection.py) | [Três CSVs e relatório de reprodução](reports/sprint4/feature_selection/reproducao_juan/) |
| S4 — piloto exploratório | [piloto.py](src/incremental/piloto.py) | [Relatório e manifesto do piloto](reports/sprint4/piloto_exploratorio/) |
| S4 — CV repetida exploratória | [cv_repetida.py](src/incremental/cv_repetida.py) | [Relatório e manifesto da CV repetida](reports/sprint4/cv_repetida_exploratoria/) |

O [histórico da Sprint 2](reports/sprint2/historico/) preserva `v1`, `v3_oficial` e documentos `.docx`; o [notebook v3 histórico](notebooks/sprint2/historico/PI3_G4_piloto_v3_binWidth_validacao.ipynb) completa os quatro notebooks versionados. Esses materiais têm os papéis e as limitações descritos no documento canônico; não substituem a referência atual.

A sequência da Sprint 3 é tabela de modelagem → split por paciente → baseline. **`run_baseline.py` treina e sobrescreve os relatórios da Sprint 3**; não deve ser executado para simples inspeção. Os módulos de piloto e CV repetida da Sprint 4 documentam suas interfaces e exigem destinos explícitos; checkpoints individuais ficam em armazenamento local não versionado. Não sobrescreva artefatos históricos nem execute etapas bloqueadas por decisões pendentes.

## Como começar

### Pré-requisitos

- Git
- Miniconda ou Anaconda com Conda
- Windows, plataforma utilizada nas validações registradas

Não é necessário instalar o Python 3.9 globalmente, pois ele é criado pelo `environment.yml`.

### Clonar o repositório

```bash
git clone https://github.com/RafaellLobo/PI3-GRUPO-4.git
cd PI3-GRUPO-4
```

### Criar o ambiente

Na raiz do repositório clonado, execute:

```bash
conda --no-plugins env create --solver classic -f environment.yml
```

No ambiente Windows testado, `--no-plugins` foi necessário para evitar a interferência do plugin de Terms of Service do Conda sobre canais globais não utilizados. As dependências Conda do projeto são obtidas do `conda-forge`, conforme definido em `environment.yml`.

### Ativar o ambiente

```bash
conda activate pi3-radiomics
```

### Validar o ambiente

```bash
python scripts/validate_environment.py
```

O script importa as principais dependências e exibe as versões instaladas, retornando erro caso algum import falhe.

### Diferenças entre ambientes

O ambiente Conda usa Python 3.9, NumPy, SciPy, pandas, Matplotlib, scikit-learn, pydicom, pylidc, SimpleITK, PyRadiomics e SQLAlchemy, conforme [environment.yml](environment.yml). As versões registradas variam entre etapas:

| Contexto | Versões documentadas |
|---|---|
| Ambiente Conda e baseline da Sprint 3 | YAML: Python `3.9`, NumPy `1.23.5`, SciPy `1.10.1`, pandas `1.5.3`, scikit-learn `1.3.2`, PyRadiomics `3.1.0`. O baseline registra Python `3.9.23`. |
| Reprodução e exploração da Sprint 4 | `.venv` local: Python `3.9.13`, NumPy `1.23.5`, SciPy `1.10.1`, pandas `1.5.3`, scikit-learn `1.3.2`; ambiente esperado em [config.py](src/incremental/config.py). |
| Extração e comparação da Sprint 2 no Colab | [Configuração v2](reports/sprint2/config/config_piloto_v2.json): Python `3.13.15`, NumPy `2.1.3`, PyRadiomics `3.1.1.dev111+g8ed579383`, fixado no commit `8ed579383`. |

O YAML não reproduz, por si só, o ambiente histórico do Colab. Os notebooks também instalam `idc-index` e `pyarrow`, não declarados nele, e usam adaptações de compatibilidade próprias. O [registro da Sprint 4](docs/sprint4/protocolo_valor_incremental.md), Seção 12, relata bloqueio de importação do NumPy no Conda pelo Controle de Aplicativo do Windows e o uso da `.venv` nessa máquina.

Não há `requirements.txt`, lockfile ou receita versionada de criação dessa `.venv`. O validador acima verifica imports e informa versões; não garante correspondência com todos os ambientes históricos. Sementes, versões e hashes de cada etapa devem ser preservados conforme sua documentação e seus manifestos.

## Dados locais e fontes externas

O dataset completo LIDC-IDRI não é distribuído pelo Git. DICOMs, anotações locais e tabelas completas de entrada ou derivadas ficam em `data/`, ignorado pelo Git, com documentação em [data/README.md](data/README.md). Uma amostra DICOM local não representa a coorte completa.

| Entrada local | Origem ou uso |
|---|---|
| `data/base_radiomica_oficial_422.csv` | Entrada da Sprint 3: 958 linhas e 107 features radiômicas. O Drive do projeto é referenciado em `data/README.md`; o notebook original da extração completa não está versionado. |
| `data/modeling_table_sprint3.csv` | Derivada pelo script de construção da tabela: 607 nódulos / 416 pacientes. A reconstrução do alvo usa o banco de anotações embutido no `pylidc`, sem ler DICOM. |
| `data/vies_malignidade_mascara_pequena.csv` | Entrada de alvo da reprodução da seleção da Sprint 4; a obtenção externa não está documentada em `data/README.md`. |

As três tabelas foram encontradas localmente na auditoria, mas **não acompanham um clone**. Os notebooks da Sprint 2 usam `/content/drive/MyDrive/PI3_Grupo4` e `/content/lidc`, incluindo arquivos `.tar` no Drive; esses caminhos e conteúdos não estão disponíveis automaticamente na máquina local. A configuração de DICOM do `pylidc` também é local, conforme a validação inicial.

Os registros dos Experimentos 10 e 13, notebooks de Juan, pacote ZIP de auditoria, logs e checkpoints individuais da Sprint 4 são referenciados na documentação, mas não estão versionados. Relatórios sobre esses materiais não equivalem à disponibilidade dos arquivos originais. Os resultados agregados e manifestos listados acima estão no repositório.

Há CSVs/Parquet históricos da Sprint 2 já versionados em `reports/`, com atributos por nódulo e identificadores LIDC. Sua presença não significa que novas tabelas médicas possam ser adicionadas ao Git: preserve os registros históricos e mantenha dados locais, credenciais e configurações pessoais fora do versionamento. As regras de `.gitattributes` protegem a integridade dos hashes registrados; não normalize ou regrave esses artefatos como parte de alterações documentais.

## Equipe

PI3 — Grupo 4.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
