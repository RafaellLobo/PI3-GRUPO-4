# Protocolo de Valor Incremental da Radiômica — Sprint 4

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Card relacionado:** S4 — Cartão 4: valor incremental da radiômica além do tamanho do nódulo
**Escopo do documento:** protocolo pré-experimental — define hipótese, modelos, etapas, análises e critérios de leitura **antes** de qualquer resultado de M2, M3 ou M4
**Política de alvo:** definida em [`docs/protocolo_coorte_target_sprint2.md`](../protocolo_coorte_target_sprint2.md) — este documento **não** redefine a regra
**Baseline de referência:** [`docs/sprint3/baseline_regressao_logistica.md`](../sprint3/baseline_regressao_logistica.md)
**Identificação de nódulos:** [`docs/sprint3/convencao_identificacao_nodulos.md`](../sprint3/convencao_identificacao_nodulos.md)
**Status:** proposta — aguardando revisão do grupo e do professor; **não é protocolo congelado**; nenhum modelo deste protocolo foi treinado
**Data:** 10/10/2026

---

## Aviso metodológico obrigatório

A variável `malignancy` do LIDC-IDRI é uma **avaliação subjetiva de suspeita de malignidade**, atribuída por radiologistas em escala de 1 a 5. **Não é confirmação anatomopatológica de câncer.** O alvo binário derivado dela mede **suspeita radiológica**, não malignidade histológica confirmada.

Duas propriedades do alvo condicionam toda a leitura deste protocolo:

1. **Circularidade com o tamanho.** O radiologista enxerga o nódulo — inclusive sua dimensão — ao atribuir a nota. Um modelo que usa tamanho pode estar reproduzindo o critério do avaliador, e não prevendo malignidade.
2. **Exclusão dos casos intermediários.** Medianas 2,5 / 3,0 / 3,5 são removidas pela regra oficial. O problema modelado — separar o claramente benigno do claramente maligno — é mais fácil do que o problema clínico real.

Nenhum resultado obtido sob este protocolo deve ser apresentado como desempenho diagnóstico ou como evidência de relevância clínica.

---

## 1. Pergunta científica

> **As características radiômicas acrescentam capacidade de discriminação da suspeita radiológica de malignidade além do tamanho do nódulo?**

A pergunta é motivada por três evidências já registradas no projeto:

- o baseline da Sprint 3 obteve AUC-ROC de 0,9762 no teste com apenas três atributos de forma, dois deles de tamanho;
- o estudo de máscaras reduzidas (Samara) observou que um modelo com **somente o diâmetro** atinge AUC-ROC de 0,9641, contra 0,9644 do modelo de três features — em um split **reconstruído**, não no split canônico; a revalidação no split oficial (Experimento 13) obteve AUC-ROC de 0,9772 para o diâmetro isolado e de 0,9778 para a reprodução do baseline (Seção 4.1);
- o feedback da Sprint 3 pediu explicitamente que o grupo testasse se o resultado é dominado pelo tamanho e se a radiômica acrescenta algo além dele.

### 1.1 Hipóteses do contraste primário

Com Δ = AUC(M3) − AUC(M0):

- **H0:** Δ ≤ 0 — as oito radiômicas não acrescentam discriminação ao diâmetro;
- **H1:** Δ > 0.

A leitura de H0/H1 é feita por intervalo de confiança, conforme a Seção 9. Nenhum p-valor é calculado (Seção 8.3).

## 2. Coorte

A coorte é a **base modelável oficial**, preservada sem alteração:

| Base | Nódulos | Pacientes | Classe 0 | Classe 1 |
|---|---:|---:|---:|---:|
| Base modelável (`data/modeling_table_sprint3.csv`) | 607 | 416 | 310 | 297 |

Nenhum nódulo é adicionado ou removido por este protocolo — **inclusive os de máscara reduzida** (Seção 11.2). A diferença em relação à coorte teórica de 616 nódulos / 422 pacientes (2 pacientes não extraídos e 19 linhas ausentes não explicadas) é herdada e está documentada no baseline, Seção 2.1.

## 3. Variável-alvo

`target_binary` de `data/modeling_table_sprint3.csv`, construída pela regra oficial:

```
0   se mediana de malignancy ≤ 2
1   se mediana de malignancy ≥ 4
NaN se mediana ∈ {2.5, 3.0, 3.5}   (excluído)
```

com exigência de ≥ 3 annotations com escore de malignidade.

### 3.1 Compatibilidade com o alvo usado na seleção de features

A seleção das oito features (Seção 5) recuperou o alvo de `data/vies_malignidade_mascara_pequena.csv`, e não da tabela canônica. Uma auditoria somente-leitura sobre os **291 pacientes / 417 nódulos do treino oficial** verificou:

| Verificação | Resultado |
|---|---|
| Correspondência 1:1 por `nodulo_id` | 417/417 nódulos canônicos presentes; junção `one_to_one` sem falha |
| Linhas adicionais no arquivo da seleção | 235, todas com mediana 2,5 / 3,0 / 3,5 — exatamente as excluídas pela regra oficial |
| Medianas | 417/417 idênticas |
| Target binário (regra oficial sobre as medianas da seleção) | 417/417 idêntico; nenhuma divergência individual |

O alvo usado na seleção é, portanto, **equivalente ao alvo canônico no treino oficial**.

## 4. Split

Arquivo oficial `reports/sprint3/baseline/patient_split.csv`, **sem reconstrução e sem alteração**:

| Partição | Pacientes | Nódulos | Classe 0 | Classe 1 |
|---|---:|---:|---:|---:|
| train | 291 | 417 | 209 | 208 |
| validation | 62 | 92 | 49 | 43 |
| test | 63 | 98 | 52 | 46 |

Papéis:

- **train** — único conjunto usado para seleção, ajuste de transformações e treinamento de qualquer modelo;
- **validation** — avaliação independente das oito features fixas (Etapa 3);
- **test** — avaliação final pré-especificada, executada uma única vez após o congelamento (Etapas 4 e 5).

### 4.1 O que já é conhecido antes deste protocolo

O teste **não é cego** para todos os modelos, e isso é declarado aqui antes de qualquer execução:

| Modelo | Resultado já conhecido | Onde |
|---|---|---|
| M1 — baseline canônico | AUC-ROC 0,9440 (validação) e **0,9762** [0,9477 – 0,9946] (teste), com matrizes de confusão | `reports/sprint3/baseline/baseline_metrics.json` — artefato versionado |
| M1 — reprodução da Samara | AUC-ROC 0,9778, **no teste oficial** | Experimento 13 (Samara) — revalidação com o split oficial: ajuste no treino oficial, avaliação no teste oficial, sem uso da validação |
| M0 | AUC-ROC 0,9772 com diâmetro isolado, **no teste oficial** | Experimento 13 (Samara) — mesmo desenho |
| M0 e M1 (histórico) | AUC-ROC 0,9641 (diâmetro isolado) e 0,9644 (três features), **em split reconstruído**, não no canônico | estudo de máscaras reduzidas (Samara) |
| M2, M3, M4 | **Nenhum.** Nenhum artefato do projeto registra desempenho desses modelos em validação ou teste | — |

**O valor canônico do M1 continua sendo 0,9762**, porque é o que está no artefato versionado e reprodutível do baseline. Os valores do Experimento 13 são uma revalidação independente e não o substituem. Segundo o DoD do Experimento 13, a revalidação usou treino e teste oficiais e não usou a validação; os valores de M0 e M1 acima são, portanto, **AUCs no teste oficial**. A reprodução da Samara (0,9778) **não coincide** com o baseline canônico (0,9762) na mesma partição de teste. A causa dessa diferença não está documentada neste protocolo; possíveis origens — ambiente, versão de bibliotecas, tabela de entrada, pré-processamento — não foram verificadas. Os registros dos Experimentos 10 e 13 **não estão versionados** no repositório.

Consequências:

- a comparação **M1 − M0** é descritiva e não cega: o nível de ambos já foi observado, inclusive no teste oficial;
- os contrastes **M3 − M0** e **M4 − M1** são os únicos com termo radiômico ainda não observado, e por isso são os únicos tratados como avaliação pré-especificada;
- a seleção das oito features calculou AUCs apenas em validação cruzada **interna ao treino**, durante o tuning do LASSO; nenhum desses valores toca validação ou teste.

## 5. Features

### 5.1 As oito features radiômicas — candidatas fixas

Lista declarada, em ordem fixa:

| # | Feature | Classe PyRadiomics |
|---:|---|---|
| 1 | `original_glcm_Imc2` | textura (GLCM) |
| 2 | `original_shape_Sphericity` | forma |
| 3 | `original_shape_SurfaceVolumeRatio` | forma |
| 4 | `original_firstorder_90Percentile` | primeira ordem |
| 5 | `original_firstorder_RootMeanSquared` | primeira ordem |
| 6 | `original_shape_Elongation` | forma |
| 7 | `original_firstorder_Skewness` | primeira ordem |
| 8 | `original_ngtdm_Coarseness` | textura (NGTDM) |

As oito existem em `data/modeling_table_sprint3.csv`, são `float64` e têm **417/417 valores finitos** no treino oficial, sem `NaN`, sem `Inf` e sem coluna constante. Nas 417 linhas de treino, seus valores — e os das demais 99 radiômicas — são **idênticos** aos de `data/base_radiomica_oficial_422.csv`, a tabela sobre a qual a seleção foi feita (diferença máxima absoluta 0,0).

**Esta lista é congelada.** Nenhum resultado obtido sob este protocolo — de CV, validação ou teste — pode adicionar, remover ou substituir uma feature.

### 5.2 Declaração × reprodução independente

O estado de evidência das oito features precisa ser lido com precisão:

| Nível | Situação |
|---|---|
| **Declarada** | As oito aparecem na conclusão em markdown do notebook de seleção de Juan (`S4 — Revisão e execução da seleção de features radiômicas`), como as features selecionadas pelo LASSO em ≥ 3 de 5 folds. |
| **Procedimento legível** | O código do notebook descreve o procedimento por completo (Seção 5.3) e restringe a seleção ao treino oficial — há `assert` exigindo 291 pacientes e disjunção com validação e teste. |
| **Não comprovado por saída** | O notebook disponível **não contém saídas de execução** (nenhuma célula tem `execution_count`; origem Colab). Os números citados na conclusão — 35–39 features após o filtro de correlação, 7–13 selecionadas pelo LASSO por fold, 16 estáveis no Random Forest, Jaccard 0,263 — **não têm saída nem CSV exportado** que os comprove. O notebook e seus CSVs de saída **não estão versionados** no repositório. |
| **Reprodução independente** | **Ainda não documentada.** A reprodução fiel do procedimento original (Seção 5.3) é pré-requisito da Etapa 3 (Seção 7.1). |

### 5.3 Procedimento original de seleção, como lido no notebook

Aplicado sobre as 107 radiômicas `original_*`, usando somente o treino oficial:

1. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grupos = `patient_id`;
2. em cada fold de treino: `VarianceThreshold(threshold=0.0)` — remoção apenas de constantes;
3. remoção de redundância: para cada par com `|correlação de Pearson| > 0,90`, calculada só no treino do fold, remove-se a coluna que aparece **depois** na ordem das colunas — o resultado depende da ordem das colunas;
4. `StandardScaler` → `LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42)`, com `C` escolhido por `GridSearchCV` em `np.logspace(-4, 2, 13)`, `scoring="roc_auc"`, CV interna `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=100 + fold)` agrupada por paciente;
5. feature selecionada no fold se `|coef| > 1e-12`;
6. **lista final:** features selecionadas em ≥ 3 dos 5 folds;
7. Random Forest (top-15, 500 árvores) apenas como checagem de consistência — **não define a lista**.

Pontos não registrados no notebook: versões de Python e bibliotecas; os warnings foram suprimidos (`warnings.filterwarnings("ignore")`), de modo que não se sabe se houve não convergência.

**Onde os filtros são ajustados no procedimento original.** Os passos 2 e 3 são ajustados sobre o treino **inteiro** de cada fold de estabilidade, **antes** do `GridSearchCV` do passo 4. A CV interna que escolhe `C` recebe, portanto, colunas filtradas com dados que incluem seus próprios folds internos de avaliação. Isso não afeta validação nem teste oficiais, que nunca entram na seleção, mas significa que a escolha de `C` não é estritamente aninhada em relação aos filtros. O procedimento original é mantido sem alteração para a **reprodução fiel** da Etapa 0, cujo objetivo é verificar a lista declarada.

### 5.4 Procedimento ajustado para a CV aninhada

Para avaliar o procedimento por CV aninhada (Etapas 1 e 2), usa-se uma versão **metodologicamente ajustada** do procedimento da Seção 5.3, em que todo filtro e toda transformação são ajustados apenas no treino em que são usados:

| Passo | Procedimento original (Seção 5.3) | Procedimento ajustado |
|---|---|---|
| Filtro de variância | ajustado no treino do fold de estabilidade, antes da CV interna | componente do pipeline avaliado pela CV interna: ajustado em cada **treino interno** |
| Filtro de correlação (abs(r) > 0,90) | idem | idem — ajustado em cada **treino interno**, mantendo a regra de ordem das colunas |
| `StandardScaler` | já dentro do pipeline | sem mudança — ajustado em cada **treino interno** |
| Escolha de `C` | `GridSearchCV`, CV interna de 4 folds agrupada | sem mudança, mas agora sobre o pipeline completo: filtros → scaler → L1 |
| Conjunto do fold de estabilidade | coeficientes ≠ 0 após o filtro | pipeline completo reajustado no treino do fold de estabilidade com o `C` escolhido; selecionadas as features com coeficiente ≠ 0 |
| Grade, seeds, limiar `1e-12`, regra ≥ 3/5, pool de 107 | — | idênticos ao original |

As duas versões **não são intercambiáveis**:

- a **reprodução fiel** (procedimento original) é a referência para verificar a lista declarada das oito features;
- a **CV aninhada** avalia o procedimento **ajustado**. Seu resultado descreve esse procedimento, e não exatamente o procedimento que produziu as oito features fixas.

Aplicado ao treino oficial completo, o procedimento ajustado pode produzir uma lista diferente da declarada. Essa lista é registrada na Etapa 0 **apenas de forma descritiva** e não substitui a lista fixa (Seção 5.1).

### 5.5 Descoberta automática de features — proibida sem exclusão explícita

`data/modeling_table_sprint3.csv` tem **108** colunas com prefixo `original_`: as 107 radiômicas **e `original_nodule_idx`**, que é metade da chave canônica do nódulo (Seção 6 da convenção de identificação), não uma feature. Um filtro `startswith("original_")` aplicado a essa tabela incluiria um identificador como preditor.

Regras:

- os conjuntos de M0–M4, e as partes fixas de M3v e M4v, são **listas explícitas por nome** (Seções 5.1, 6 e 6.3), nunca derivadas por prefixo;
- o pool de 107 candidatas usado para refazer a seleção na CV aninhada é definido por nome, na ordem do cabeçalho de `data/base_radiomica_oficial_422.csv`, porque o filtro de correlação depende da ordem (Seção 5.3, passo 3);
- se qualquer código derivar colunas por prefixo, deve excluir `original_nodule_idx` explicitamente e **abortar** se o resultado não tiver exatamente 107 colunas, na ordem acima. Essa ordem coincide com a de `modeling_table_sprint3.csv` depois de removida `original_nodule_idx` — verificado.

## 6. Modelos

| Modelo | Features | Nº de features |
|---|---|---:|
| **M0** — diâmetro | `original_shape_Maximum3DDiameter` | 1 |
| **M1** — baseline oficial | `original_shape_MeshVolume`, `original_shape_Maximum3DDiameter`, `original_shape_Sphericity` | 3 |
| **M2** — radiômica | as oito da Seção 5.1 | 8 |
| **M3** — diâmetro + radiômica | M0 ∪ M2 | 9 |
| **M4** — baseline + radiômica | M1 ∪ M2 | **10** |

`original_shape_Sphericity` pertence a M1 e a M2; em M4 entra **uma única vez**, de modo que M4 tem 10 features distintas, e não 11.

M2, M3 e M4 designam **sempre** modelos com a lista fixa da Seção 5.1. Os modelos cujo conjunto radiômico é selecionado dentro de cada fold da CV aninhada têm nomes próprios (Seção 6.3).

### 6.1 Especificação comum

Idêntica ao baseline, para que as diferenças entre modelos venham só do conjunto de features:

| Componente | Valor |
|---|---|
| Pipeline | `StandardScaler` → `LogisticRegression`, num único estimador |
| `C` | 1.0 |
| `penalty` | `l2` |
| `solver` | `lbfgs` |
| `class_weight` | `None` |
| `max_iter` | 1000 |
| `random_state` | 42 |
| Tuning | nenhum |
| Transformação de features | nenhuma além da padronização |
| Imputação | nenhuma — `NaN` ou `Inf` em qualquer feature abortam a execução |
| Threshold | 0,5, aplicado explicitamente sobre `predict_proba` |

A seleção usou regressão logística L1 com `C` ajustado; os modelos avaliados usam L2 com `C = 1`. A diferença é deliberada: a seleção escolhe o conjunto, e a avaliação mantém a especificação do baseline.

Warnings do scikit-learn **não são suprimidos**: são capturados e gravados no artefato de cada etapa.

### 6.2 Invariante de reprodução do M1

Antes de qualquer avaliação, M1 é reajustado no treino e seus coeficientes, intercepto, `scaler_mean` e `scaler_scale` são comparados aos de `baseline_metrics.json`. A comparação usa **somente parâmetros ajustados no treino**: não exige calcular nenhuma métrica de validação ou de teste. Divergência acima da tolerância numérica aborta a execução.

### 6.3 Modelos com seleção variável — somente na CV aninhada

Na CV aninhada (Etapas 1 e 2), o conjunto radiômico de cada fold é o selecionado pelo procedimento ajustado (Seção 5.4) **naquele treino externo**, e não a lista fixa. Sendo S_k o conjunto selecionado no treino externo k:

| Modelo | Features | Onde é usado |
|---|---|---|
| M0, M1 | fixos, como na tabela da Seção 6 | Etapas 1, 2, 3 e 5 |
| **M2v** | S_k | somente Etapas 1 e 2 |
| **M3v** | diâmetro ∪ S_k | somente Etapas 1 e 2 |
| **M4v** | features de M1 ∪ S_k | somente Etapas 1 e 2 |
| M2, M3, M4 | lista fixa da Seção 5.1 | Etapas 3 e 5; na CV, apenas como comparação descritiva (Seção 7.2) |

Regras:

- S_k pode ter tamanho diferente de oito e pode conter features de M0 ou M1, inclusive o próprio diâmetro; nesse caso, a feature entra **uma única vez**. O conjunto e o número de features de cada modelo são registrados por fold;
- se S_k for vazio, M3v coincide com M0 e M4v com M1 naquele fold, e M2v prevê a prevalência do treino externo. A ocorrência é registrada;
- a especificação de ajuste é a da Seção 6.1;
- resultados de M2v–M4v descrevem o **procedimento**; resultados de M2–M4 descrevem as **oito features fixas**. Nenhum dos dois é usado como estimativa do outro.

## 7. Etapas

As etapas são sequenciais. Cada uma só começa depois que a anterior tiver sido concluída e documentada.

| Etapa | Dados | Objetivo |
|---|---|---|
| 0 — Pré-requisitos | treino | reprodução fiel da seleção original e invariantes |
| 1 — CV aninhada piloto | treino | verificar implementação, convergência e custo |
| 2 — CV aninhada completa | treino | avaliar o **procedimento ajustado** de seleção + modelagem (M2v–M4v) |
| 3 — Validação oficial | treino → validação | avaliar as **oito features fixas** em dados independentes |
| 4 — Congelamento | — | fixar protocolo, código e relatório antes do teste |
| 5 — Teste oficial | treino → teste | avaliação final pré-especificada, executada uma vez |

### 7.1 Etapa 0 — pré-requisitos

1. **Reprodução fiel do procedimento original.** O procedimento da Seção 5.3 é reexecutado **sem alteração** sobre os 417 nódulos de treino, no ambiente do projeto, com versões registradas e warnings capturados, e a lista obtida é comparada à da Seção 5.1. Essa reprodução é **pré-requisito para avaliar M2, M3 e M4 na validação** (Etapa 3).
   - **Lista idêntica:** a declaração passa a ter reprodução documentada.
   - **Lista diferente:** a lista da Seção 5.1 **permanece** como conjunto fixo — ela foi definida antes de qualquer resultado de desempenho, e trocá-la depois exigiria justificar a troca. A divergência é registrada como limitação, e a lista reproduzida é avaliada **apenas como sensibilidade**. A decisão de seguir adiante é registrada pelo grupo antes da Etapa 3.
2. **Procedimento ajustado no treino completo.** O procedimento da Seção 5.4 é aplicado aos mesmos 417 nódulos, e a lista resultante é registrada **apenas de forma descritiva**, junto com sua diferença em relação à lista fixa. Ela não substitui a lista fixa e não é avaliada na validação nem no teste.
3. **Invariantes estruturais:** 607 nódulos / 416 pacientes; composição das partições igual à Seção 4; zero pacientes compartilhados entre partições; listas de features da Seção 5.5; invariante do M1 (Seção 6.2).
4. **Isolamento:** o código das Etapas 0 a 2 lê apenas linhas de treino. O código da Etapa 3 lê treino e validação. **Só o código da Etapa 5 lê linhas de teste.**

### 7.2 Etapas 1 e 2 — CV aninhada: avaliação do procedimento

**O que estima.** Desempenho do **procedimento ajustado** (Seção 5.4) — "selecionar features e ajustar o modelo" —, por meio dos modelos M2v, M3v e M4v (Seção 6.3). Não estima o desempenho das oito features fixas, nem exatamente o do procedimento original que as produziu. Em cada fold, o conjunto selecionado pode diferir da lista declarada.

**Por que a seleção é refeita.** As oito features foram selecionadas com **todo** o treino oficial. Avaliá-las por CV dentro desse mesmo treino seria leakage: cada fold de avaliação já teria influenciado a escolha das features. Por isso, em cada **treino externo**, todo o procedimento da Seção 5.4 é refeito do zero — estabilidade em 5 folds e LASSO com CV interna, com filtros e scaler ajustados em cada treino interno —, sem acesso ao fold externo de avaliação.

Estrutura:

```
Treino oficial (291 pacientes / 417 nódulos)
└─ CV externa: StratifiedGroupKFold(5), grupos = patient_id
   ├─ treino externo k → procedimento ajustado da Seção 5.4
   │                     (CV de estabilidade de 5 folds; em cada uma, CV interna
   │                      de 4 folds com filtros → scaler → L1 ajustados em cada
   │                      treino interno; tudo agrupado por paciente)
   │                   → conjunto S_k
   │                   → ajuste de M0, M1, M2v, M3v e M4v no treino externo k
   └─ fold externo k   → predições out-of-fold
```

M0 e M1 não dependem da seleção, mas são ajustados nos mesmos folds para que os contrastes sejam pareados. Os contrastes da CV são **M3v − M0** e **M4v − M1**.

**Etapa 1 — piloto.** Uma repetição de 5 folds externos (`random_state = 42`). Serve para verificar implementação, convergência, invariantes e tempo de execução. O piloto **só autoriza correções técnicas** — erro de código, não convergência, invariante violada. Nenhuma escolha de modelo, feature, parâmetro ou critério pode ser alterada com base nos contrastes observados no piloto. Os resultados do piloto são registrados como piloto.

**Etapa 2 — completa.** 10 repetições × 5 folds externos, com `random_state = 42 + r`, r = 0…9. A repetição r = 0 reproduz a partição do piloto.

**Agregação — os 50 folds não são observações independentes.** Os folds de uma mesma repetição compartilham dados de treino, e as repetições reutilizam os mesmos 417 nódulos. Por isso:

- a unidade de resumo é a **repetição**: em cada uma, as predições out-of-fold dos 417 nódulos são reunidas e se calcula uma AUC por modelo e um Δ por contraste (M3v − M0 e M4v − M1);
- reporta-se a média dos 10 Δ e seu intervalo mínimo–máximo, descrito como **sensibilidade à partição**, e não como incerteza amostral;
- **não** se calcula IC, erro padrão, teste t ou qualquer estatística que trate os 50 folds ou as 10 repetições como amostras independentes;
- valores por fold são reportados apenas de forma descritiva.

Saídas adicionais da Etapa 2:

- **estabilidade da seleção:** frequência com que cada uma das 107 features é selecionada nos 50 treinos externos e índice de Jaccard entre cada conjunto selecionado e a lista da Seção 5.1;
- **CV com as oito features fixas — comparação descritiva:** M2, M3 e M4 com a lista fixa, nas mesmas partições externas, sem refazer a seleção. Essa CV é **otimista por construção**, porque as oito foram escolhidas com todo o treino, inclusive com os nódulos de cada fold externo de avaliação, e é rotulada como tal. A diferença entre ela e a CV aninhada **não é uma medida exata do otimismo da seleção**, porque mistura outras fontes de diferença:
  - o procedimento original (Seção 5.3) e o ajustado (Seção 5.4) não são idênticos;
  - na CV aninhada, a seleção é feita com cerca de 4/5 do treino;
  - o conjunto S_k varia entre folds, enquanto a lista fixa não varia.

  Essa diferença é reportada apenas como indicação descritiva da possível influência da seleção, e a CV com features fixas nunca é apresentada como desempenho;
- **controle por permutação de rótulos — descritivo:** a CV aninhada, com o procedimento ajustado, é repetida sobre K permutações de `target_binary` no treino, na partição externa da repetição r = 0, com seeds de permutação fixas. Reporta-se a **distribuição** das AUCs sob permutação, por modelo (quantis e histograma), e não um valor isolado. O Experimento 10 documentou comportamento **bimodal** da AUC sob permutação de rótulos. Por isso:
  - a distribuição sob permutação **não** é presumida centrada em 0,5;
  - nenhum valor isolado, nem o afastamento de 0,5 de uma única permutação, é tomado como prova de vazamento ou como prova de ausência de vazamento;
  - um padrão inesperado — por exemplo, AUCs sob permutação comparáveis às obtidas com os rótulos reais — motiva investigação documentada, não conclusão automática.

  K e o esquema de permutação — por nódulo ou por paciente — são definidos antes da Etapa 1 (Seção 13).

### 7.3 Etapa 3 — validação oficial: avaliação das oito features fixas

**O que estima.** Desempenho dos modelos M0–M4 **com a lista fixa** da Seção 5.1, ajustados nos 417 nódulos de treino e avaliados nos 92 nódulos / 62 pacientes de validação.

Não há leakage de seleção: a lista foi escolhida sem acesso à validação.

A Etapa 2 e a Etapa 3 respondem a perguntas diferentes e **nenhuma substitui a outra**:

| | CV aninhada (Etapa 2) | Validação oficial (Etapa 3) |
|---|---|---|
| Objeto avaliado | o procedimento **ajustado** de seleção + modelo | as oito features fixas + modelo |
| Modelos radiômicos | M2v, M3v, M4v | M2, M3, M4 |
| Contrastes | M3v − M0, M4v − M1 | M3 − M0, M4 − M1 |
| Conjunto de features | varia por fold (S_k) | fixo (Seção 5.1) |
| Dados de avaliação | folds do próprio treino | partição independente |
| Incerteza reportada | sensibilidade à partição (mín.–máx.) | IC95% por bootstrap pareado (Seção 8) |

**Pré-requisito:** M2, M3 e M4 só são avaliados na validação depois da reprodução da Etapa 0 (Seção 7.1, item 1).

Os resultados da validação são registrados integralmente, mas **não alteram** modelo, features, especificação ou critérios. Só problemas técnicos — não convergência, invariante violada — justificam correção antes do teste, e qualquer correção é documentada.

### 7.4 Etapa 4 — congelamento

Antes do teste, ficam versionados na branch, em um commit identificado:

- este protocolo, inclusive revisões feitas desde a proposta;
- o código das Etapas 0 a 5;
- os relatórios das Etapas 0 a 3.

O hash desse commit é gravado no artefato da Etapa 5.

### 7.5 Etapa 5 — teste oficial: avaliação final pré-especificada

- os modelos são os da Etapa 3, ajustados nos 417 nódulos de treino — sem reajuste com validação;
- o teste é avaliado **uma única vez**, para M0–M4 e para a sensibilidade da Seção 10.1, com todos os contrastes da Seção 8 reportados;
- qualquer reexecução — por exemplo, por falha técnica — é documentada com a causa, e nenhum resultado do teste é usado para alterar modelo, feature ou critério.

Ressalva já declarada na Seção 4.1: o teste é cego apenas para os modelos com termo radiômico.

## 8. Contrastes e inferência

### 8.1 Contrastes

| Prioridade | Contraste | Pergunta |
|---|---|---|
| **Primário** | **M3 − M0** | a radiômica acrescenta ao diâmetro? |
| **Secundário** | **M4 − M1** | a radiômica acrescenta ao baseline oficial? |
| Descritivo | M1 − M0 | volume e esfericidade acrescentam ao diâmetro? Não cego (Seção 4.1) |
| Descritivo | M2 vs. M0 | radiômica isolada vs. diâmetro isolado |

Esses contrastes, com M2–M4 de lista fixa, são os da validação (Etapa 3) e do teste (Etapa 5). Na CV aninhada, os contrastes correspondentes são **M3v − M0** e **M4v − M1** (Seção 6.3): eles avaliam o procedimento, e não as oito features fixas.

Métrica principal: **AUC-ROC**, calculada sobre `predict_proba`. Como o contraste primário é único, não há correção de multiplicidade. Os demais contrastes não são confirmatórios.

### 8.2 Intervalo de confiança — bootstrap pareado por paciente

Aplicado na validação (Etapa 3) e no teste (Etapa 5):

| Parâmetro | Valor |
|---|---|
| Unidade de reamostragem | `patient_id` — pacientes sorteados com reposição, cada um entrando com **todos** os seus nódulos |
| Pareamento | em cada réplica, a **mesma** amostra de pacientes é usada para todos os modelos, e Δ é calculado dentro da réplica |
| Réplicas | B = 2000 |
| Seed | 42 |
| IC | percentil, limites 2,5 e 97,5 (IC95%) |
| Réplicas degeneradas | réplicas com uma única classe são descartadas e **contadas** no artefato |
| Retreino no bootstrap | **não** — as probabilidades são as do modelo já ajustado |

Reamostrar por nódulo subestimaria a largura dos intervalos, porque nódulos do mesmo paciente não são independentes. O IC reflete a incerteza da **amostra de avaliação**, não a variabilidade do treinamento (Seção 11).

### 8.3 Sem p-valor

Nenhum p-valor é reportado. O protocolo não define um teste de hipótese formal para a diferença pareada de AUC com nódulos agrupados por paciente. Derivar um p-valor da distribuição bootstrap sem esse método definido produziria um número sem taxa de erro conhecida. A leitura da Seção 9 é feita **exclusivamente pelo IC95%**.

## 9. Critérios de interpretação — definidos antes dos resultados

Seja [L, U] o IC95% bootstrap de Δ no **teste**.

| Condição | Leitura |
|---|---|
| L > 0 | evidência de ganho de discriminação **do alvo radiológico** |
| L ≤ 0 ≤ U | ganho **não demonstrado** — inconclusivo, o que não equivale a ausência de ganho |
| U < 0 | evidência de perda de discriminação |

A mesma régua vale para M4 − M1, lida como resultado secundário.

**Coerência com as etapas anteriores.** A conclusão do teste é sempre reportada junto com:

- o sinal e o IC de Δ na validação (Etapa 3), que avalia o mesmo objeto do teste — as oito features fixas;
- a média e o intervalo mínimo–máximo do contraste correspondente na CV aninhada (M3v − M0 ou M4v − M1, Etapa 2), que avalia **outro objeto**: o procedimento ajustado.

Discordância de sinal entre teste e validação é declarada explicitamente e qualifica a conclusão como **instável**, mesmo quando L > 0 no teste. Discordância em relação à CV aninhada também é declarada e discutida, mas é lida como diferença entre o desempenho do procedimento e o do conjunto fixo, e não como contradição direta.

**Referência exploratória ΔAUC = 0,02.** Reporta-se se a estimativa pontual e os limites do IC ficam acima ou abaixo de 0,02, **apenas como referência descritiva de magnitude**. Esse valor não foi derivado de critério clínico, não define relevância clínica e não altera a leitura da tabela acima.

**Teto da AUC.** Com o M1 já em 0,94–0,98, o espaço para ganho em AUC é pequeno. Δ pequeno não implica ausência de informação adicional; por isso a Seção 10 inclui métricas de probabilidade.

## 10. Análises secundárias pré-especificadas

Todas são descritivas ou de sensibilidade; nenhuma é confirmatória.

### 10.1 Sensibilidade — representação flexível do tamanho

Motivação: algumas das oito features dependem fortemente da escala do nódulo. `SurfaceVolumeRatio`, por exemplo, decresce aproximadamente como o inverso do tamanho. Parte de um "ganho" de M3 sobre M0 pode refletir a **forma não linear** da relação entre tamanho e alvo, que um único termo linear do diâmetro não captura.

Modelos:

- **M0s:** spline do diâmetro;
- **M3s:** spline do diâmetro + as oito features fixas;
- na CV aninhada, **M3vs:** spline do diâmetro + S_k.

**A mesma representação de tamanho é usada em M0s, M3s e M3vs**, de modo que o contraste compara modelos que diferem apenas pelo termo radiômico, com o tamanho representado da mesma forma flexível nos dois lados.

| Componente | Valor |
|---|---|
| Transformação | `SplineTransformer(n_knots=4, degree=3, knots="quantile", extrapolation="linear", include_bias=False)` sobre `original_shape_Maximum3DDiameter` |
| Ajuste dos nós | somente no treino (no treino oficial ou no treino externo de cada fold) |
| Demais etapas | colunas do spline e termo radiômico (oito features fixas ou S_k; nenhum em M0s) → `StandardScaler` → mesma `LogisticRegression` da Seção 6.1 |
| Onde | Etapa 2 (M3vs − M0s) e Etapas 3 e 5 (M3s − M0s, com o mesmo bootstrap pareado) |

Leitura — descritiva, sem inferência causal:

- **Se o ganho diminuir ou desaparecer** em M3s − M0s, em relação a M3 − M0, o resultado é **compatível** com a hipótese de que parte do ganho se deve a uma relação não linear entre tamanho e alvo, não capturada pelo termo linear. Isso não demonstra que essa seja a explicação, nem que as oito features não carreguem outra informação: o spline também altera o ajuste do modelo como um todo, e diferenças pequenas cabem na incerteza dos intervalos.
- **Se o ganho persistir**, ele não é explicado **por esta representação específica** do tamanho. Isso não demonstra que as oito features carreguem informação independente do tamanho: o spline é uma entre muitas representações possíveis, e as features podem codificar tamanho de outras formas — por exemplo, pela dependência da máscara ou por combinação entre features.
- Em nenhum dos casos o resultado estabelece independência em relação ao tamanho, nem relação causal entre feature e alvo.

### 10.2 Métricas de probabilidade e calibração

Para os contrastes da Seção 8.1, com o mesmo bootstrap pareado:

- Δ log-loss e Δ Brier score — regras de pontuação próprias, mais sensíveis que a AUC perto do teto;
- intercepto e inclinação de calibração, por modelo;
- recall (classe 1) e especificidade (classe 0) com threshold 0,5 — apenas descritivos.

### 10.3 Circularidade e dependência do tamanho

Calculadas **somente no treino**:

- correlação de Spearman entre `original_shape_Maximum3DDiameter` e a mediana de malignidade (a) nos 417 nódulos modeláveis e (b) nos 652 nódulos dos pacientes de treino, incluindo os intermediários, com as medianas de `data/vies_malignidade_mascara_pequena.csv`. A versão (b) evita que a exclusão dos intermediários infle a correlação;
- correlação de Spearman entre cada uma das oito features e o diâmetro.

### 10.4 Máscaras reduzidas e faixas de diâmetro

Descritivo, sem inferência por subgrupo:

- número de nódulos com `mascara_pequena` por partição e por classe;
- probabilidade média prevista e erros por classe no subgrupo, para M0, M1 e M3;
- os mesmos resumos por faixa de diâmetro: < 10 mm, 10–20 mm e > 20 mm.

Subgrupos com poucos nódulos ou uma única classe não têm AUC calculada.

## 11. Limitações

### 11.1 Do alvo

1. **Suspeita radiológica, não malignidade histológica confirmada** (aviso inicial).
2. **Circularidade com o tamanho:** parte da associação entre tamanho e alvo pode vir do critério do próprio avaliador. Valor incremental da radiômica, aqui, significa capacidade de explicar a **percepção radiológica** além do tamanho, e não a biologia do nódulo.
3. **Exclusão dos intermediários:** remove os casos difíceis; as métricas não se transferem para o problema clínico completo.

### 11.2 Do tamanho e das máscaras reduzidas

4. **Dominância do tamanho:** o diâmetro isolado já se aproxima do baseline (Seção 1), o que deixa pouco espaço para ganho em AUC.
5. **Viés das máscaras reduzidas — coorte preservada.** Na base de 958 nódulos, 115 têm menos de 50 voxels após a resegmentação. Segundo o estudo de máscaras reduzidas, esse grupo concentra nódulos menores (diâmetro mediano de 8 mm contra 15 mm), tem textura menos estável sob perturbação (CV 0,18 contra 0,10) e malignidade mais baixa. No treino modelável são 40 nódulos, 35 da classe 0 e 5 da classe 1. **Os nódulos são mantidos**: o estudo concluiu que não são dado de má qualidade a descartar automaticamente, e removê-los alteraria a distribuição do alvo. Em consequência:
   - parte da discriminação pode vir de o modelo classificar bem nódulos pequenos por serem pequenos;
   - features de textura e de primeira ordem — cinco das oito — são menos confiáveis nesse grupo;
   - a análise com exclusão desses nódulos **não faz parte deste protocolo**.
6. **Features dependentes de escala:** três das oito são de forma (`Sphericity`, `SurfaceVolumeRatio`, `Elongation`), e `SurfaceVolumeRatio` e `ngtdm_Coarseness` dependem do tamanho da região. As oito não são "independentes do tamanho" por construção; a Seção 10.1 e a Seção 10.3 existem por isso.
7. **Medida do diâmetro:** `Maximum3DDiameter` é medido na máscara de consenso de 50% após a resegmentação, e não pelos radiologistas; em máscaras reduzidas, pode subestimar o tamanho físico.

### 11.3 Da seleção de features

8. **Reprodução fiel ainda não documentada** (Seção 5.2) — pré-requisito da Etapa 3. Além disso, a CV aninhada avalia o procedimento **ajustado** (Seção 5.4), e não exatamente o procedimento original que produziu as oito features.
9. **Filtro de correlação dependente da ordem das colunas:** a representante mantida em cada grupo redundante é arbitrária.
10. **Estabilidade entre anotadores não avaliada:** não se sabe se as oito features são estáveis quando outro radiologista desenha o contorno. O estudo correspondente está em aberto, e seu cruzamento com as oito features deve ser feito quando ele for concluído.
11. **Sem transformação de features:** a análise exploratória do grupo indicou assimetria relevante em boa parte das radiômicas. Este protocolo mantém só a padronização, por coerência com o baseline e com a seleção; transformações não são avaliadas aqui.

### 11.4 Da inferência

12. **Partições pequenas:** a validação tem 43 positivos em 62 pacientes, e o teste, 46 positivos em 63 pacientes. ICs de Δ largos e resultado inconclusivo são desfechos plausíveis.
13. **IC sem variabilidade de treino:** o bootstrap reamostra a avaliação, não o ajuste.
14. **CV aninhada sem IC formal:** a dispersão entre repetições descreve sensibilidade à partição, não incerteza amostral (Seção 7.2).
15. **Teste parcialmente não cego:** M0 e M1 já tiveram desempenho observado, inclusive no split oficial (Seção 4.1).
16. **Sem p-valor e sem teste formal** (Seção 8.3).
17. **Coeficientes não interpretáveis:** features correlacionadas dividem a contribuição no ajuste, e os coeficientes estão no espaço padronizado. Nenhum ranking de importância é derivado deles.

### 11.5 Herdadas

18. Discretização de intensidade herdada da extração oficial; a escolha entre `binCount` e `binWidth` não é reavaliada.
19. Cobertura de extração: 607 de 616 nódulos, com 19 linhas ausentes não explicadas.
20. Generalização externa não avaliada: todos os dados vêm do LIDC-IDRI.

## 12. Reprodutibilidade

| Item | Valor |
|---|---|
| Seeds | 42 em ajuste, CV piloto e bootstrap; `42 + r` (r = 0…9) nas repetições da CV completa; `100 + fold` na CV interna da seleção, como no notebook de Juan, tanto no procedimento original quanto no ajustado; seeds de permutação fixadas junto com K (Seção 13, P1) |
| Ambiente | `.venv` do projeto: Python 3.9.13, scikit-learn 1.3.2 (mesma versão do baseline), numpy 1.23.5, pandas 1.5.3, scipy 1.10.1. O ambiente conda `pi3-radiomics` não carrega o numpy nesta máquina por bloqueio do Controle de Aplicativo do Windows |
| Registro obrigatório por etapa | versões de bibliotecas, seeds, listas de features, composição das partições, warnings capturados, réplicas bootstrap descartadas e hash do commit |
| Dados | `data/` não é versionado (dado derivado de imagem médica). Os artefatos em `reports/` não contêm valores de feature por nódulo |
| Artefatos existentes | `reports/sprint3/baseline/*` e as tabelas em `data/` são lidos e nunca modificados |

## 13. Decisões pendentes antes da execução

| # | Decisão | Prazo |
|---|---|---|
| P1 | Número de permutações K e esquema de permutação — por nódulo ou por paciente — do controle da Seção 7.2 | antes da Etapa 1 |
| P2 | Confirmação dos parâmetros do spline da Seção 10.1 | antes da Etapa 1 |
| P3 | Confirmação da regra para divergência na reprodução fiel (Seção 7.1, item 1) | antes da Etapa 3 |
| P4 | Explicar a divergência entre M1 = 0,9778 (Experimento 13, Samara) e M1 = 0,9762 (baseline canônico), ambos no teste oficial. A partição do Experimento 13 já está confirmada como teste oficial (Seção 4.1) | antes da Etapa 5 |
| P5 | Versionamento dos registros dos Experimentos 10 e 13 e do notebook de seleção de Juan, com suas saídas | antes da Etapa 4 |

Qualquer decisão tomada sobre esses pontos é registrada em revisão deste documento, com data, antes da etapa indicada.

## 14. Fora do escopo deste protocolo

- implementação de scripts e notebooks, e execução de qualquer etapa;
- alteração da coorte, do alvo, do split ou da lista de oito features;
- exclusão ou reponderação dos nódulos de máscara reduzida;
- outros algoritmos de classificação, tuning de hiperparâmetros e seleção de threshold;
- teste de hipótese formal para AUC pareada com dados agrupados (Seção 8.3).

---

_Este documento registra o protocolo pré-experimental do Cartão 4 da Sprint 4. Qualquer alteração após o início da Etapa 1 deve ser registrada em nova revisão, com data, motivo e indicação de quais resultados já eram conhecidos no momento da alteração._
