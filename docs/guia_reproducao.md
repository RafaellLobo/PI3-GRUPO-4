# Guia de ambiente e reprodução — Sprints 2, 3 e 4

Este guia organiza os procedimentos identificados no [README](../README.md), no [environment.yml](../environment.yml), no [índice de experimentos](indice_experimentos.md) e nos produtores e documentos técnicos ali relacionados. Os comandos abaixo foram conferidos por leitura do código e da documentação; não foram executados para elaborar este guia. Nenhuma instalação, treinamento ou reprodução científica foi realizada nesta atividade.

O alvo é a suspeição radiológica dos nódulos, conforme o [protocolo de coorte e variável-alvo](protocolo_coorte_target_sprint2.md). Preserve coorte, alvo, split, sementes, configurações e artefatos históricos.

## 1. Escolher o alcance da atividade

| Alcance | O que permite afirmar | Condições e limites |
|---|---|---|
| Inspeção de artefatos existentes | O conteúdo documentado e os registros de proveniência podem ser examinados. | Consultar os JSONs, tabelas e documentos do índice não equivale a reproduzir seus resultados. |
| Reprodução parcial / por etapa | Uma atividade específica tem produtor e entradas identificados. | Depende do ambiente, dos arquivos locais e das guardas da atividade; exige verificar as saídas da nova execução. A existência do script não comprova sucesso no ambiente atual. |
| Reprodução integral | A cadeia completa, desde as entradas originais até os resultados, foi refeita e conferida. | Não está demonstrada por este guia. Faltam materiais externos, produtores históricos e receitas completas de ambiente; os registros locais não cobrem todas as etapas oficiais propostas. |

Os procedimentos que gravam resultados são documentação de execução futura, não uma rotina de inspeção. As restrições de preservação abaixo devem ser consideradas antes de qualquer reexecução.

## 2. Pré-requisitos e ambientes

Os comandos locais são apresentados para a raiz do repositório. `python` deve corresponder ao ambiente da atividade. O projeto registra validações no Windows, usa Git para rastreabilidade e Conda/pip no ambiente declarativo; a Sprint 2 utiliza notebooks no Colab com acesso ao Drive e aos dados necessários.

| Contexto | Evidência de versões | Situação de reprodução |
|---|---|---|
| Sprint 2 — Colab histórico | Os [configs v2](../reports/sprint2/config/config_piloto_v2.json) e [v3](../reports/sprint2/config/config_piloto_v3.json) registram Python `3.13.15`, NumPy `2.1.3`, pandas `2.2.3`, SimpleITK `2.5.6`, pyarrow `18.1.0`, pylidc `0.2.3` e PyRadiomics `3.1.1.dev111+g8ed579383`, commit `8ed579383`. | Registro histórico. Os notebooks contêm instalações e adaptações de compatibilidade, mas não há receita congelada de todo o runtime. Um Colab atual não é presumido equivalente. |
| Sprint 3 — Conda `pi3-radiomics` | O YAML declara Python `3.9`, NumPy `1.23.5`, SciPy `1.10.1`, pandas `1.5.3` e scikit-learn `1.3.2`. O [baseline](sprint3/baseline_regressao_logistica.md) registra Python `3.9.23`. | Há receita declarativa de criação, não executada nesta elaboração. `python=3.9` não fixa o patch histórico nem todas as dependências transitivas. |
| Sprint 4 — `.venv` local | O [relatório da seleção](../reports/sprint4/feature_selection/reproducao_juan/relatorio_reproducao.json) e [config.py](../src/incremental/config.py) registram Python `3.9.13`, NumPy `1.23.5`, SciPy `1.10.1`, pandas `1.5.3` e scikit-learn `1.3.2`. Essas versões também foram encontradas nos metadados da `.venv` inspecionada. | Ambiente local presente, com registros de execuções anteriores; não há receita versionada completa de criação dessa `.venv`. A leitura dos metadados não verifica imports nem funcionamento científico. |

O YAML também fixa pydicom `2.4.4`, pylidc `0.2.3`, SimpleITK `2.5.5`, PyRadiomics `3.1.0`, SQLAlchemy `1.4.54`, greenlet `3.2.1`, requests `2.32.5` e Matplotlib `3.7.3`; `pip` e `ipykernel` não têm versão fixada nele. Os metadados locais consultados coincidem com essas versões fixadas. Isso não torna o YAML equivalente ao Colab histórico, inclusive quanto a PyRadiomics e SimpleITK.

Os notebooks de extração/comparação instalam também `idc-index` e `pyarrow`, ausentes do YAML e da `.venv` inspecionada. Há instalações sem versão fixada nos notebooks. Não foi localizado `requirements.txt` ou lockfile que feche todos os ambientes.

### Preparação Conda documentada

Os comandos preservados do README são:

```text
conda --no-plugins env create --solver classic -f environment.yml
conda activate pi3-radiomics
python scripts/validate_environment.py
```

O README explica o uso de `--no-plugins` no Windows; o YAML usa `conda-forge` e `nodefaults`. O [validador de ambiente](../scripts/validate_environment.py) importa dependências e informa versões, mas não exige igualdade com todos os ambientes históricos nem cobre todas as dependências dos notebooks.

A [proposta da Sprint 4](sprint4/protocolo_valor_incremental.md), Seção 12, registra bloqueio do NumPy no Conda pelo Controle de Aplicativo do Windows e uso da `.venv` naquela máquina. Não há procedimento versionado suficiente para apresentar aqui uma criação equivalente da `.venv`; as versões esperadas devem ser conferidas antes de usar seus comandos.

## 3. Entradas e disponibilidade

| Entrada | Origem ou produtor documentado | Disponibilidade e uso |
|---|---|---|
| DICOMs e `.tar` da Sprint 2 | Os notebooks usam `/content/drive/MyDrive/PI3_Grupo4/dicom_tar` e `/content/lidc`. A [validação inicial](lidc_initial_validation.md) descreve a configuração local do `pylidc`. | Materiais completos externos não verificados. Uma amostra DICOM local não representa a coorte inteira. |
| `data/base_radiomica_oficial_422.csv` | [data/README.md](../data/README.md) referencia `08_extracao_completa_oficial_422/CSV E JSON/base_radiomica_oficial_422.csv` no Drive. | Cópia local não versionada encontrada; entrada da Sprint 3 e da reprodução da seleção. O notebook original da extração completa não está versionado, conforme a [convenção de identificação](sprint3/convencao_identificacao_nodulos.md). |
| `data/modeling_table_sprint3.csv` | Produzida por [build_modeling_table.py](../scripts/build_modeling_table.py), a partir da base oficial e das anotações embutidas no `pylidc`, sem ler DICOM. | Local não versionada, encontrada neste checkout; entrada do split, baseline e atividades da Sprint 4. |
| `data/vies_malignidade_mascara_pequena.csv` | O script e o relatório de reprodução da seleção registram seu uso e hash. | Local não versionada, encontrada neste checkout; necessária à reprodução fiel da seleção. Sua origem e obtenção não estão consolidadas em `data/README.md`. |
| Split e referência do baseline | [patient_split.csv](../reports/sprint3/baseline/patient_split.csv) e [baseline_metrics.json](../reports/sprint3/baseline/baseline_metrics.json). | Versionados. A seleção e os manifestos exploratórios registram o split; a verificação M0/M1 também exige o JSON do baseline. |
| Notebooks de Juan, pacote ZIP, logs e checkpoints anteriores | Declarados fora do repositório no [checkpoint do Cartão 4](sprint4/estado_cartao4.md), Seção 9. | Referências externas, sem disponibilidade dos originais verificada nesta elaboração. |

Os arquivos locais em `data/` não acompanham um clone. A presença das três tabelas neste checkout não demonstra a disponibilidade atual de suas fontes externas. Os caminhos do Drive são referências documentadas, não links de download verificados.

## 4. Sprint 2 — sequência dos notebooks históricos

Fonte canônica: [piloto radiômico](sprint2/piloto_radiomico.md), incluindo o mapa de notebooks e saídas. Preserve a ordem declarada no notebook v2: instalação → compatibilidade → Drive → parâmetros → obtenção dos dados → escrita de `.pylidcrc` → importação de `pylidc` → extração. O notebook informa que `pylidc` lê a configuração no import.

| Atividade | Sequência e pré-condições confirmadas | Saídas e localização |
|---|---|---|
| Piloto v2 | Usar o [notebook v2](../notebooks/sprint2/PI3_G4_sprint2_v2_extração_piloto.ipynb), suas células de instalação/compatibilidade e os dados exigidos. | O notebook grava configuração em `BASE/config` e tabelas, descartes e validação em `BASE/features`; há inventário em `BASE/logs`. Cópias preservadas estão em [config](../reports/sprint2/config/), [features](../reports/sprint2/features/) e [validação](../reports/sprint2/validacao/). |
| Comparação v2 × v3 | O [notebook de comparação](../notebooks/sprint2/PI3_G4_comparacao_binCount_binWidth.ipynb) exige o CSV e o config v2 em seus caminhos do Drive antes da extração v3. A comparação definitiva documentada exige a mesma sessão e ambiente, sem reinício nem reinstalação entre as extrações. | Produz config e CSV v3 e `BASE/features/comparacao_v2_v3_discretizacao.json`. A cópia de referência está em [reports/sprint2/comparacao/](../reports/sprint2/comparacao/). |
| Verificação populacional | O [notebook populacional](../notebooks/sprint2/PI3_G4_verificacao_populacional_valor3.ipynb) usa o banco de anotações do `pylidc`, sem baixar imagens; executar sua própria sequência de compatibilidade e cálculo. É independente da extração DICOM. | A célula de exportação aponta para `/content/drive/MyDrive/PI3_Grupo4/logs/verificacao_populacional_valor_central_3.json`; a cópia versionada está em [validação](../reports/sprint2/validacao/verificacao_populacional_valor_central_3.json). |
| v3_oficial histórico | O [notebook histórico](../notebooks/sprint2/historico/PI3_G4_piloto_v3_binWidth_validacao.ipynb) inclui extração e teste de determinismo. Seu papel atual é controle de regressão, segundo o documento canônico. | Configuração, CSV/Parquet, descartes e validação estão preservados em [historico/v3_oficial/](../reports/sprint2/historico/v3_oficial/). Não é o lado v3 da comparação definitiva. |

Nos notebooks de extração e comparação, `BASE` é `/content/drive/MyDrive/PI3_Grupo4`; seus diretórios de saída não são `reports/` do checkout. Não há sincronização automática documentada entre esses locais. As células escrevem configurações e resultados nos caminhos definidos e podem substituir arquivos existentes no Drive.

A presença de células de instalação não garante reconstrução do runtime histórico nem o cumprimento automático da condição de mesma sessão. A v1 e os DOCX são registros históricos; consulte as limitações no índice e no documento canônico, sem tratá-los como referência atual ou inventar um comando de reprodução da v1.

### Verificações parciais locais

Os seguintes comandos estão documentados nos próprios scripts e verificam aspectos existentes sem reextrair as features:

```text
python scripts/validate_nodule_ids.py
python scripts/validate_selection_masks.py
```

O [validador de IDs](../scripts/validate_nodule_ids.py) tem blocos opcionais: tabelas locais ausentes ou NumPy indisponível podem gerar `SKIP`. O [validador de seleção/máscaras](../scripts/validate_selection_masks.py) depende de NumPy e `pylidc`; a validação da ROI pode gerar `SKIP` sem DICOM. Os scripts não gravam novos resultados científicos. Uma saída bem-sucedida com verificações omitidas não comprova reprodução integral.

## 5. Sprint 3 — sequência implementada e sobrescrita

A cadeia consta do [baseline oficial](sprint3/baseline_regressao_logistica.md). Todos os comandos abaixo gravam em caminhos fixos; não possuem opção documentada de destino alternativo. A execução do baseline também treina e avalia o teste oficial. O checkpoint da Sprint 4 orienta não reexecutá-lo sobre os relatórios históricos. A tabela registra a sequência implementada, não uma autorização para substituir esses arquivos.

| Ordem | Comando existente | Entradas | Saídas fixas |
|---|---|---|---|
| 1 | `python scripts/build_modeling_table.py` | Base radiômica oficial e banco de anotações do `pylidc`. | `data/modeling_table_sprint3.csv`, substituída se já existir. |
| 2 | `python scripts/make_patient_split.py` | Tabela de modelagem. | `reports/sprint3/baseline/patient_split.csv`, substituído se já existir. |
| 3 | `python scripts/run_baseline.py` | Tabela de modelagem e split. | `reports/sprint3/baseline/baseline_metrics.json` e `confusion_matrix.csv`, substituídos se já existirem. |

Para inspeção, consulte os artefatos existentes em [reports/sprint3/baseline/](../reports/sprint3/baseline/). Uma reexecução que preserve todos os históricos exige resolver a estratégia de preservação antes da escrita; esses scripts não oferecem o redirecionamento necessário por argumento. Não altere coorte, split, features ou sementes para contornar uma divergência.

O relatório canônico registra verificações anteriores de determinismo, com alcance explícito; isso é evidência histórica, não uma reexecução feita por este guia. A cobertura incompleta da extração e as linhas ausentes permanecem como limitações documentadas.

## 6. Sprint 4 — atividades com produtores disponíveis

Consulte o [checkpoint](sprint4/estado_cartao4.md), a [proposta de protocolo](sprint4/protocolo_valor_incremental.md) e os produtores abaixo. A reprodução fiel da seleção original preserva limitações metodológicas auditadas; o piloto e a CV usam o procedimento ajustado. Essas atividades não são intercambiáveis.

### Verificação M0/M1 e reprodução fiel da seleção

| Atividade | Comando confirmado | Saídas e cuidado necessário |
|---|---|---|
| Verificação dos modelos de referência | `python scripts/sprint4/validate_reference_models.py` | Ajusta M0/M1 no treino e sobrescreve `reports/sprint4/reference_models/validacao_modelos_referencia.json`. Não calcula novas métricas de validação/teste. Não reexecutar para simples inspeção do [relatório existente](../reports/sprint4/reference_models/validacao_modelos_referencia.json). |
| Reprodução fiel da seleção | `python scripts/sprint4/reproduce_juan_feature_selection.py --out-dir DIR_SAIDA --n-jobs 1` | Em `DIR_SAIDA`: `frequencia_selecao_features.csv`, `resumo_folds_selecao.csv`, `features_selecionadas_lasso.csv` e `relatorio_reproducao.json`. Ajusta LASSO, Random Forest e diagnóstico; não é inspeção de arquivos. |

`DIR_SAIDA` é um marcador a substituir por um destino distinto do histórico, não um diretório encontrado no checkout. O [script da seleção](../scripts/sprint4/reproduce_juan_feature_selection.py) proíbe saída em `data/` e `reports/sprint3/`, cria o destino se necessário e recusa substituir seus arquivos sem `--sobrescrever`; não use essa opção para atualizar os históricos. O padrão aponta para [reproducao_juan/](../reports/sprint4/feature_selection/reproducao_juan/), que já contém resultados.

A opção documentada `--notebook` é apenas para registrar o hash de um notebook externo; ele não é necessário para executar o script versionado. `--n-jobs 1` permite captar os warnings no processo principal. Entradas e configurações completas estão no script, no JSON de execução e na [auditoria](../reports/sprint4/feature_selection/auditoria_reproducao_juan.md).

### Piloto e CV repetida exploratórios

Após conferir ambiente, entradas e diretórios, os módulos documentam estas interfaces:

```text
python -m src.incremental.piloto --checkpoints DIR_CHECKPOINTS --destino DIR_DESTINO --confirmo-preliminar-exploratorio
python -m src.incremental.cv_repetida --checkpoints DIR_CHECKPOINTS --destino DIR_DESTINO --confirmo-preliminar-exploratorio
```

São comandos de ajuste de modelos e gravação de resultados, não comandos de inspeção. `DIR_CHECKPOINTS` e `DIR_DESTINO` são marcadores; não representam caminhos já preparados. Conforme [piloto.py](../src/incremental/piloto.py), [cv_repetida.py](../src/incremental/cv_repetida.py) e [checkpoints.py](../src/incremental/checkpoints.py):

- Os dois diretórios precisam existir. O destino dos relatórios fica dentro do repositório, fora de `data/`, e não pode conter os nomes de relatório/manifesto que a atividade produzirá; a sobrescrita é recusada.
- Os checkpoints ficam fora de `data/`, em diretório ignorado pelo Git ou fora de qualquer repositório. Podem conter predições individuais e não devem ser publicados nos relatórios.
- A retomada confere código, entradas, configuração, ambiente e composição dos folds. Checkpoints incompatíveis ou corrompidos não são reutilizados silenciosamente.

| Atividade | Saídas em `DIR_DESTINO` | Saídas locais e referência preservada |
|---|---|---|
| Piloto exploratório | `relatorio_piloto_exploratorio.json` e `manifesto_piloto_exploratorio.json`. | Checkpoints sob `DIR_CHECKPOINTS/piloto_exploratorio/` e arquivo de tempos no diretório da sessão. Referência: [piloto_exploratorio/](../reports/sprint4/piloto_exploratorio/). |
| CV repetida exploratória | `relatorio_cv_repetida_exploratoria.json` e `manifesto_cv_repetida_exploratoria.json`. | Checkpoints sob `DIR_CHECKPOINTS/cv_repetida_exploratoria/` e arquivo de tempos no diretório da sessão. Referência: [cv_repetida_exploratoria/](../reports/sprint4/cv_repetida_exploratoria/). |

Ambas as atividades usam somente o treino oficial e permanecem exploratórias. Preserve as sementes de [config.py](../src/incremental/config.py) e dos módulos: `42` no piloto e `42 + r` nas repetições da CV. Manifestos e relatórios registram configuração, entradas e warnings; não promovem a exploração a resultado definitivo.

## 7. Decisões e limitações de reprodução

- **Etapas oficiais da Sprint 4:** não foram localizados neste checkout registros de execução das Etapas 1–5 oficiais. O checkpoint registra como pendentes as decisões P1–P5 e a revisão do protocolo: permutações, spline, lista fixa, reconciliação de M1 e proveniência. Preserve as guardas; este guia não define essas decisões nem fornece comandos para contorná-las.
- **Entradas e ambientes incompletos:** disponibilidade do Drive, DICOMs completos e originais externos não foi verificada. A extração completa da Sprint 3 não tem seu notebook original versionado; o Colab histórico e a `.venv` não têm receitas completas congeladas.
- **Alcance dos validadores:** imports, metadados, hashes ou verificações com `SKIP` não comprovam toda a cadeia científica. Nesta elaboração foram lidos código, documentação e metadados; não foram testados imports científicos ou reexecutados resultados.
- **Uso anterior do teste:** o histórico documentado e as informações do grupo registradas no README devem acompanhar qualquer interpretação. Reutilizar o teste oficial não constitui nova evidência independente de hipóteses formuladas após esse uso nem validação externa.
- **Preservação e proveniência:** mantenha as entradas locais fora do Git e respeite [.gitattributes](../.gitattributes), pois conversões ou regravações podem invalidar hashes. A [.gitignore](../.gitignore) protege os dados locais em `data/`, `scripts/output/` e `.claude/settings.local.json`; não distribua checkpoints ou dados individuais.

O [índice de experimentos](indice_experimentos.md) centraliza os vínculos e as pendências de materiais posteriores. Arquivos não localizados não demonstram que os respectivos experimentos deixaram de ser realizados.
