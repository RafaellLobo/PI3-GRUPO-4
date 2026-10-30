# Índice de experimentos e artefatos — PI3-GRUPO-4

Este índice relaciona atividades, produtores, artefatos e lacunas verificáveis no checkout analisado, incluindo o README com alterações locais. Não substitui os relatórios técnicos nem atribui números de experimento a atividades sem correspondência documentada. As fontes externas não foram consultadas e nenhum experimento foi executado para produzir este levantamento.

O alvo científico é a suspeição radiológica dos nódulos, conforme o [protocolo de coorte e variável-alvo](protocolo_coorte_target_sprint2.md). Métricas e conclusões devem ser consultadas nos documentos e artefatos originais indicados abaixo.

## Estados de rastreabilidade

| Estado | Significado neste índice |
|---|---|
| Disponível no repositório | Arquivo ou diretório encontrado e rastreado pelo Git. Não implica aprovação científica nem execução de outras etapas. |
| Referenciado externamente | Material citado por uma fonte local, mas fora dos arquivos versionados; sua disponibilidade externa não foi verificada. |
| Local não versionado | Arquivo encontrado neste checkout, ignorado pelo Git e ausente de um clone novo. |
| Não localizado | Produtor, artefato ou vínculo não identificado nos materiais inspecionados. Não significa que a atividade não tenha sido realizada. |

## Sprint 2 — piloto, comparação e validação

Referências: [piloto radiômico](sprint2/piloto_radiomico.md), fonte canônica, e [pipeline de seleção e máscaras](sprint2/pipeline_selecao_mascaras.md). A expressão "comparação definitiva" abaixo segue o documento do piloto e não representa aprovação de uma discretização para modelagem.

| Atividade / natureza | Objetivo | Script ou notebook produtor | Artefatos e estado |
|---|---|---|---|
| Piloto v2 — referência canônica do piloto | Extrair atributos com máscaras por consenso e validar coorte, alvo e qualidade da extração. | [Notebook de extração v2](../notebooks/sprint2/PI3_G4_sprint2_v2_extração_piloto.ipynb) | Disponíveis no repositório: [configuração v2](../reports/sprint2/config/config_piloto_v2.json), [CSV](../reports/sprint2/features/piloto_piloto_v2_consenso50.csv), [Parquet](../reports/sprint2/features/piloto_piloto_v2_consenso50.parquet), [validação](../reports/sprint2/validacao/validacao_piloto_v2_consenso50.json) e [descartes](../reports/sprint2/validacao/descartes_piloto_v2_consenso50.csv). |
| Comparação definitiva v2 × v3 — sensibilidade à discretização | Comparar `binCount = 64` e `binWidth = 25` mantendo as condições comuns; não avalia superioridade preditiva. | [Notebook de comparação](../notebooks/sprint2/PI3_G4_comparacao_binCount_binWidth.ipynb) | Disponíveis no repositório: [configuração v3](../reports/sprint2/config/config_piloto_v3.json), [CSV v3](../reports/sprint2/features/piloto_piloto_v3_consenso50.csv) e [comparação com proveniência](../reports/sprint2/comparacao/comparacao_v2_v3_discretizacao.json). O lado v2 é o da linha anterior. |
| Verificação populacional — validação do piloto | Examinar a frequência de alvos indeterminados e comparar amostra sequencial e amostras aleatórias. | [Notebook de verificação populacional](../notebooks/sprint2/PI3_G4_verificacao_populacional_valor3.ipynb) | Disponível no repositório: [JSON de critérios e verificação populacional](../reports/sprint2/validacao/verificacao_populacional_valor_central_3.json). |
| v3_oficial — histórico e controle de regressão/determinismo | Preservar a rodada histórica e verificar repetibilidade; não é o lado v3 da comparação definitiva. | [Notebook histórico v3](../notebooks/sprint2/historico/PI3_G4_piloto_v3_binWidth_validacao.ipynb) | Disponíveis no repositório: [configuração](../reports/sprint2/historico/v3_oficial/config_piloto_v3_oficial.json), [CSV](../reports/sprint2/historico/v3_oficial/piloto_piloto_v3_oficial_consenso50.csv), [Parquet](../reports/sprint2/historico/v3_oficial/piloto_piloto_v3_oficial_consenso50.parquet), [validação e determinismo](../reports/sprint2/historico/v3_oficial/validacao_piloto_v3_oficial_consenso50.json) e [descartes](../reports/sprint2/historico/v3_oficial/descartes_piloto_v3_oficial_consenso50.csv). |
| v1 — histórico anterior às correções | Manter a rastreabilidade da rodada anterior às correções de alvo e estabilidade. | Produtor específico da v1 não localizado; não se presume equivalência com o notebook v2 atual. | Disponíveis no repositório: [diretório v1](../reports/sprint2/historico/v1/), com configuração, CSV/Parquet, validação e descartes. Não é referência atual de alvo. |
| Diagnóstico da coorte e manifesto de download — suporte exploratório/logístico | Inspecionar critérios de elegibilidade e preparar o manifesto das séries CT. | [explore_cohort_criteria.py](../scripts/explore_cohort_criteria.py) e [generate_download_manifest.py](../scripts/generate_download_manifest.py) | Código disponível no repositório. Saídas locais não versionadas encontradas: `scripts/output/cohort_diagnostic_raw.csv` e `scripts/output/lidc_manifest_422.tcia`. |

O código de [seleção e máscaras](../src/radiomics/) e o [validador de equivalência](../scripts/validate_selection_masks.py) apoiam a rastreabilidade do piloto. Os [documentos DOCX históricos](../reports/sprint2/historico/docx/) são preservados, com suas limitações registradas no [aviso desse diretório](../reports/sprint2/historico/docx/README.md); não substituem o Markdown canônico.

## Sprint 3 — cadeia do baseline oficial

Referências: [baseline de Regressão Logística](sprint3/baseline_regressao_logistica.md) e [convenção de identificação dos nódulos](sprint3/convencao_identificacao_nodulos.md).

| Etapa / natureza | Objetivo | Script produtor | Artefatos e estado |
|---|---|---|---|
| Tabela de modelagem — preparação do baseline | Reconstruir o alvo a partir das anotações do scan correto, preservar os atributos de entrada e selecionar as linhas modeláveis. | [build_modeling_table.py](../scripts/build_modeling_table.py) | Código disponível no repositório; saída local não versionada encontrada: `data/modeling_table_sprint3.csv`. A etapa lê o banco de anotações embutido no `pylidc`, sem ler DICOM. |
| Split oficial por paciente | Persistir a divisão de pacientes entre treino, validação e teste. | [make_patient_split.py](../scripts/make_patient_split.py) | Disponível no repositório: [patient_split.csv](../reports/sprint3/baseline/patient_split.csv). |
| Baseline oficial de Regressão Logística | Estabelecer a referência com volume, diâmetro e esfericidade e registrar avaliação e incerteza. | [run_baseline.py](../scripts/run_baseline.py) | Disponíveis no repositório: [baseline_metrics.json](../reports/sprint3/baseline/baseline_metrics.json), com configurações, métricas e intervalos, e [confusion_matrix.csv](../reports/sprint3/baseline/confusion_matrix.csv). |

A entrada da extração completa e seu produtor original estão discriminados na seção de dependências. O baseline já avaliou o teste oficial; esse uso está documentado no relatório canônico. A reutilização desse conjunto não constitui nova evidência independente para hipóteses desenvolvidas após seu uso.

## Sprint 4 — verificação, reprodução e exploração

Referências: [checkpoint do Cartão 4](sprint4/estado_cartao4.md), [proposta de protocolo](sprint4/protocolo_valor_incremental.md), [auditoria da seleção](../reports/sprint4/feature_selection/auditoria_reproducao_juan.md) e [comparação com o pacote de Juan](../reports/sprint4/feature_selection/comparacao_pacote_juan.md).

| Atividade / natureza | Objetivo | Script ou módulo produtor | Artefatos e estado |
|---|---|---|---|
| Verificação de M0/M1 — pré-requisito da Etapa 0 | Verificar invariantes e reproduzir o ajuste do M1 canônico no treino, sem calcular novas métricas de validação ou teste. | [validate_reference_models.py](../scripts/sprint4/validate_reference_models.py) | Disponível no repositório: [validacao_modelos_referencia.json](../reports/sprint4/reference_models/validacao_modelos_referencia.json), com configuração, ambiente e hashes de entradas. |
| Reprodução fiel da seleção — procedimento original auditado | Reproduzir a seleção descrita no notebook de Juan e registrar convergência e divergências, preservando as limitações metodológicas originais. | [reproduce_juan_feature_selection.py](../scripts/sprint4/reproduce_juan_feature_selection.py) | Disponíveis no repositório: [frequências](../reports/sprint4/feature_selection/reproducao_juan/frequencia_selecao_features.csv), [resumo dos folds](../reports/sprint4/feature_selection/reproducao_juan/resumo_folds_selecao.csv), [features selecionadas](../reports/sprint4/feature_selection/reproducao_juan/features_selecionadas_lasso.csv) e [relatório com parâmetros e hashes](../reports/sprint4/feature_selection/reproducao_juan/relatorio_reproducao.json). Não equivale à aprovação da lista fixa de features. |
| Piloto exploratório — uma repetição e cinco folds | Avaliar no treino oficial o procedimento ajustado de seleção e modelos, com predições fora do fold e contrastes descritivos. | [piloto.py](../src/incremental/piloto.py) | Disponíveis no repositório: [relatório](../reports/sprint4/piloto_exploratorio/relatorio_piloto_exploratorio.json) e [manifesto](../reports/sprint4/piloto_exploratorio/manifesto_piloto_exploratorio.json). Checkpoints individuais referenciados externamente. |
| CV repetida exploratória — dez repetições e cinco folds | Descrever sensibilidade à partição e estabilidade da seleção, com consolidação por repetição sobre o treino oficial. | [cv_repetida.py](../src/incremental/cv_repetida.py) | Disponíveis no repositório: [relatório](../reports/sprint4/cv_repetida_exploratoria/relatorio_cv_repetida_exploratoria.json) e [manifesto](../reports/sprint4/cv_repetida_exploratoria/manifesto_cv_repetida_exploratoria.json). Checkpoints individuais referenciados externamente. |

As especificações comuns estão em [config.py](../src/incremental/config.py). A [infraestrutura incremental](../src/incremental/) implementa guardas, acesso aos dados, folds, seleção, modelos, checkpoints e manifestos; há [testes próprios](../tests/incremental/). A reprodução fiel do procedimento original e o procedimento ajustado da exploração são atividades distintas.

No checkout analisado, não foram localizados artefatos de execução das Etapas 1–5 oficiais. A proposta e as pendências estão documentadas no checkpoint; os relatórios exploratórios acima não comprovam a conclusão dessas etapas nem valor clínico ou independência em relação ao tamanho.

## Dependências locais e materiais externos

Os caminhos em código simples nesta seção identificam arquivos locais ou referências externas; não são links para artefatos distribuídos pelo Git.

| Material / uso | Evidência e dependência documentada | Estado |
|---|---|---|
| DICOMs e arquivos `.tar` para a extração da Sprint 2 | Os notebooks de extração e comparação usam `/content/drive/MyDrive/PI3_Grupo4/dicom_tar` e `/content/lidc`; o acesso a DICOM pelo `pylidc` exige a [configuração local documentada](lidc_initial_validation.md). | Referenciados externamente; a amostra local não demonstra disponibilidade da coorte completa. |
| `data/base_radiomica_oficial_422.csv` — entrada da Sprint 3 e da reprodução da seleção | [data/README.md](../data/README.md) aponta para `08_extracao_completa_oficial_422/CSV E JSON/base_radiomica_oficial_422.csv` no Drive. A convenção de identificação registra que o notebook original da extração completa não está versionado. | Origem referenciada externamente; cópia local não versionada encontrada. Produtor original não localizado no checkout. |
| `data/modeling_table_sprint3.csv` — entrada do split, baseline e atividades da Sprint 4 | Derivada por `scripts/build_modeling_table.py`; o relatório da seleção e os manifestos exploratórios registram seu hash e o do split utilizado. A verificação M0/M1 também usa o JSON do baseline. | Local não versionado; produtores e registros de proveniência disponíveis no repositório. |
| `data/vies_malignidade_mascara_pequena.csv` — alvo da reprodução da seleção | O script e o relatório da reprodução registram a entrada e seu hash. Sua obtenção não está descrita em `data/README.md`. | Local não versionado; produtor e origem externa não consolidados na documentação inspecionada. |
| Notebooks de Juan, pacote ZIP, scripts auxiliares, logs e checkpoints individuais | O checkpoint, Seção 9, declara esses materiais fora do repositório; as auditorias distinguem saídas históricas e regenerações recentes. O notebook é opcional no script versionado de reprodução, para registro de hash. | Referenciados externamente. Os relatórios locais não equivalem à disponibilidade dos originais. |

Versões, sementes e hashes devem ser consultados nos JSONs de cada atividade, no [environment.yml](../environment.yml) e nas referências canônicas. O ambiente Conda, o Colab histórico e a `.venv` da Sprint 4 têm diferenças documentadas; não se presume equivalência entre eles. As regras de [.gitattributes](../.gitattributes) preservam os artefatos com hashes registrados.

## Referências numeradas sem cadeia completa no checkout

Esta seção registra referências existentes, sem associá-las automaticamente aos produtores das tabelas anteriores nem atribuir métricas sem o artefato original.

| Referência | Evidência local e objetivo identificável | Produtor e resultados próprios | Estado |
|---|---|---|---|
| Experimento 10 | A proposta da Sprint 4, Seção 7.2, menciona controle por permutação de rótulos; o checkpoint, pendência P5, solicita seus registros. | Não localizados no checkout. | Referenciado externamente. |
| Experimento 13 | A proposta, Seção 4.1, descreve revalidação de M0/M1 no split oficial; o checkpoint registra pendência de reconciliação com o baseline. | Não localizados no checkout; não confundir com `validate_reference_models.py`, que tem relatório próprio e escopo distinto. | Referenciado externamente. |
| Experimentos 11, 12 e 14 | O [README atual](../README.md) registra, como histórico informado pelo grupo, uso anterior do teste oficial; não consolida objetivo, código e artefatos de cada experimento. | Não localizados no checkout. | Menção documental disponível; cadeia experimental não localizada. |

A menção no README comprova o registro de uma informação do grupo, não a verificação independente das execuções. O mesmo cuidado aplica-se ao Experimento 17, listado nas pendências abaixo.

## Pendências — Sprint 5 e Experimentos 17–26

| Item | Evidência no checkout analisado | Estado de rastreabilidade | Pendência |
|---|---|---|---|
| Sprint 5 | Não foram identificados diretórios, documentos ou artefatos específicos nos materiais inspecionados. | Não localizado. | Localizar entregas e identificar seus objetivos, natureza científica, produtores e resultados; vincular as fontes ao índice. |
| Experimento 17 | O README menciona uso anterior do teste oficial conforme histórico informado pelo grupo; não fornece a cadeia de artefatos do experimento. | Referência no README disponível; produtor e artefatos não localizados. | Localizar registros, confirmar objetivo e vínculo com a sprint, e consolidar as evidências do uso do teste. |
| Experimentos 18–26 | Não foram identificados produtores, configurações, resultados ou referências específicas que permitam caracterizar esses experimentos. | Não localizado. | Localizar e indexar cada experimento, incluindo código, entradas, configurações, resultados e manifestos quando existentes. |

A ausência desses materiais neste checkout não permite concluir que as atividades não foram realizadas. A consolidação deve preservar a distinção entre resultados oficiais, exploratórios e históricos, sem renumerar experimentos ou alterar evidências anteriores.
