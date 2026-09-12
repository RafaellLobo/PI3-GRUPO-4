# Baseline de Regressão Logística — Sprint 3

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Card relacionado:** S3 — Implementação e avaliação formal do baseline
**Escopo do documento:** documentação técnica canônica do baseline oficial da Sprint 3
**Política de alvo:** definida em [`docs/protocolo_coorte_target_sprint2.md`](../protocolo_coorte_target_sprint2.md) — este documento **não** redefine a regra, apenas a aplica
**Status:** consolidado — baseline executado, validado e persistido
**Data:** 11/09/2026

---

## Aviso metodológico obrigatório

A variável `malignancy` do LIDC-IDRI é uma **avaliação subjetiva de probabilidade/aparência de malignidade**, atribuída por radiologistas em escala de 1 a 5. **Não é confirmação anatomopatológica de câncer.** O alvo binário derivado dela, e o modelo treinado sobre esse alvo, produzem um **score de risco baseado em características radiológicas disponíveis no dataset** — não um diagnóstico. O LIDC-IDRI não fornece classificação Lung-RADS nativa.

Nenhuma métrica deste documento deve ser apresentada como desempenho diagnóstico.

---

## 1. Objetivo

Formalizar o baseline de Regressão Logística da Sprint 3: um modelo deliberadamente mínimo — três atributos de forma, um classificador linear, nenhum tuning — que estabelece a **referência quantitativa** contra a qual modelos posteriores devem ser comparados.

O baseline não é o modelo final e não pretende ser competitivo. Seu valor está em ser reprodutível, auditável e honesto quanto à incerteza.

## 2. Base utilizada

| Base | Nódulos | Pacientes |
|---|---:|---:|
| Coorte teórica do protocolo (Seção 5.4) | 616 | 422 |
| Base radiômica efetivamente disponível para modelagem | **607** | **416** |

Distribuição de classes na base modelável: **classe 0 = 310**, **classe 1 = 297** (51,1% / 48,9%). O balanço é saudável e nenhuma técnica de rebalanceamento foi aplicada, conforme previsto no protocolo.

### 2.1 Diferença entre a coorte teórica e a cobertura efetiva

Recomputando a regra oficial diretamente das annotations sobre os 1018 scans do banco do `pylidc`, obtém-se exatamente **616 nódulos / 422 pacientes** — reprodução íntegra da Seção 5.4 do protocolo. A regra de elegibilidade e de consolidação do alvo está, portanto, correta e reprodutível.

A diferença até 607 é **cobertura de extração**, não divergência metodológica, e decompõe-se em duas causas independentes:

| Etapa | Nódulos | Pacientes |
|---|---:|---:|
| Coorte teórica (universo completo) | 616 | 422 |
| Esperado nos 420 scans efetivamente extraídos | 610 | 417 |
| Obtido | **607** | **416** |

- **Cobertura de scans (−6 nódulos, −5 pacientes).** Dois pacientes com nódulo modelável nunca foram extraídos: `LIDC-IDRI-0578` e `LIDC-IDRI-0626`. O restante decorre da escolha de série em pacientes com mais de uma reconstrução CT, em que o scan selecionado rende menos nódulos modeláveis que a alternativa.
- **Extração parcial (−3 nódulos, −1 paciente).** Em 14 scans, o número de linhas extraídas é menor que o de clusters selecionados: 977 esperados contra 958 presentes, 19 linhas ausentes. Três delas eram modeláveis, e uma custou o paciente `LIDC-IDRI-0380` inteiro.

A perda corresponde a 1,5% dos nódulos e não distorce o balanço de classes. As 19 linhas ausentes permanecem **não explicadas** — não foi possível determinar, a partir dos artefatos versionados, se resultaram de descarte por salvaguarda de intensidade ou de falha silenciosa na extração. Esse ponto fica registrado como pendência, não como resultado.

### 2.2 Reconstrução do alvo

O alvo **precisou ser reconstruído a partir do scan correto**. A base radiômica da Sprint 3 foi persistida com um identificador cuja numeração não corresponde ao índice original do clustering, o que impede associar o rótulo por correspondência direta. O mecanismo está detalhado na Seção 3.

## 3. Rastreabilidade do target

A convenção de identificação de nódulos — os dois formatos textuais, a diferença semântica entre eles e a identidade canônica `(scan_id, original_nodule_idx)` — está documentada em [`docs/sprint3/convencao_identificacao_nodulos.md`](convencao_identificacao_nodulos.md). Esta seção registra como ela foi aplicada para recuperar o alvo.

### 3.1 O que `Nxx` significa

O `nodulo_id` da base oficial tem o formato `{patient_id}_N{pos}_scan{id}`. O notebook de extração da Sprint 3 gerou esse `N{pos}` a partir de:

```python
selecionados, _ = selection.selecionar_clusters(scan, clusters=clusters)
for idx, (_, anns) in enumerate(selecionados):
```

Portanto **`Nxx` é a posição dentro da lista filtrada `selecionados`**, não o `nodule_idx` original devolvido por `Scan.cluster_annotations()`. Como `selecionar_clusters` remove os clusters inelegíveis e o `enumerate` reindexa o que sobra, as duas numerações divergem sempre que há descarte antes da posição em questão.

### 3.2 Magnitude da divergência

**448 das 958 linhas (46,8%)** têm `Nxx != original_nodule_idx`. Os deslocamentos observados vão de +1 a +15, sempre não negativos — coerente com filtragem. Exemplos:

| `nodulo_id` | `Nxx` | `original_nodule_idx` | deslocamento |
|---|---:|---:|---:|
| `LIDC-IDRI-0078_N02_scan1` | 2 | 3 | +1 |
| `LIDC-IDRI-0132_N01_scan7` | 1 | 3 | +2 |
| `LIDC-IDRI-0136_N02_scan8` | 2 | 5 | +3 |
| `LIDC-IDRI-0154_N00_scan11` | 0 | 1 | +1 |

### 3.3 Associação correta

A chave válida é **`scan_id` + posição em `selecionados`**. O `scan_id` embutido no `nodulo_id` corresponde exatamente a `pylidc.Scan.id` — verificado coerente com `patient_id` em 958/958 linhas — e é ele que **remove a ambiguidade de pacientes com múltiplas séries**, problema que nenhuma chave baseada apenas em `patient_id` consegue resolver.

Com essa chave, o mapeamento é inequívoco em **958/958 linhas, com 0 falhas**.

### 3.4 `cohort_diagnostic_raw.csv` não serve como fonte de alvo para esta base

O arquivo `scripts/output/cohort_diagnostic_raw.csv` indexa por `nodule_index` **original** e **não registra o scan**. Usá-lo como fonte direta de target para a base radiômica da Sprint 3 faria com que as linhas em que as duas numerações divergem — 448 das 958, conforme a Seção 3.2 — fossem **associadas ao cluster incorreto**, podendo gerar target incorreto. Nem toda associação equivocada resulta necessariamente em rótulo diferente: dois clusters distintos podem coincidir na classe. Some-se a isso a ambiguidade irresolúvel nos pacientes multi-série, que aquele arquivo não permite desfazer.

O alvo é, por isso, recuperado diretamente das annotations do scan correto, aplicando as mesmas funções `has_characteristics` e `binarize_malignancy` de [`scripts/explore_cohort_criteria.py`](../../scripts/explore_cohort_criteria.py) — importadas, não reimplementadas, de modo que exista uma única definição de cada regra no projeto.

Materialização em [`scripts/build_modeling_table.py`](../../scripts/build_modeling_table.py) → `data/modeling_table_sprint3.csv`.

## 4. Features do baseline

Exclusivamente três atributos de forma:

| Feature | Grandeza |
|---|---|
| `original_shape_MeshVolume` | volume |
| `original_shape_Maximum3DDiameter` | diâmetro |
| `original_shape_Sphericity` | esfericidade |

Nenhuma outra feature foi utilizada. Não houve seleção de features: o conjunto foi fixado por decisão de escopo do card, não por procedimento estatístico. As três colunas foram verificadas sem `NaN` e sem `Inf` antes do treinamento.

## 5. Split

Divisão **por `patient_id`**, nunca por nódulo — um paciente pode ter até 7 nódulos na base, e eles não são observações independentes. Método: `GroupShuffleSplit` em dois estágios, `groups=patient_id`, `random_state=42`, conforme Seção 6 do protocolo.

| Partição | Pacientes | Nódulos | Classe 0 | Classe 1 | % classe 1 |
|---|---:|---:|---:|---:|---:|
| train | 291 | 417 | 209 | 208 | 49,9% |
| validation | 62 | 92 | 49 | 43 | 46,7% |
| test | 63 | 98 | 52 | 46 | 46,9% |
| **Total** | **416** | **607** | **310** | **297** | **48,9%** |

**Vazamento de paciente entre partições: 0**, verificado nas três interseções par a par (train/validation, train/test, validation/test).

As proporções de nódulos ficaram em 68,7 / 15,2 / 16,1 em vez de 70/15/15 exatos. Sob agrupamento por paciente, as proporções-alvo são aproximadas e não garantidas: a alocação ocorre em blocos de tamanho variável — pacientes têm de 1 a 7 nódulos —, de modo que a contagem de nódulos por partição depende de quais pacientes caem em cada lado. O desvio observado é pequeno e compatível com esse mecanismo.

O split está persistido em `reports/sprint3/baseline/patient_split.csv`, gerado por [`scripts/make_patient_split.py`](../../scripts/make_patient_split.py). É versionado de propósito: é ele que torna auditável qualquer métrica reportada.

## 6. Modelo

Pipeline: **`StandardScaler` → `LogisticRegression`**, nesta ordem, num único estimador.

| Parâmetro | Valor |
|---|---|
| `C` | 1.0 |
| `penalty` | `l2` |
| `class_weight` | `None` |
| `random_state` | 42 |
| `max_iter` | 1000 |
| threshold de classificação | 0.5 (fixo) |
| scikit-learn | 1.3.2 |

Nenhum tuning de hiperparâmetro foi realizado. O threshold é aplicado explicitamente sobre `predict_proba`, não via `predict()`, para que o corte faça parte do contrato registrado.

**O scaler foi ajustado apenas sobre o treino.** Por viver dentro do `Pipeline`, ele recebe `fit_transform` somente nos dados de `train`; validação e teste passam exclusivamente por `transform`, com as médias e desvios aprendidos no treino. Isso cumpre a Seção 6.3 do protocolo. A evidência é verificável e está gravada no JSON: `scaler.n_samples_seen_ = 417`, idêntico ao número de nódulos de treino — o script aborta se divergir.

O modelo convergiu em 15 iterações, bem abaixo do limite de 1000. **Nenhum warning do scikit-learn** foi emitido durante o ajuste ou a avaliação.

Implementação em [`scripts/run_baseline.py`](../../scripts/run_baseline.py).

## 7. Resultados

Treinamento exclusivamente na partição `train`. Threshold = 0.5.

### 7.1 Validation (n = 92)

| Métrica | Valor |
|---|---:|
| AUC-ROC | 0,9440 |
| Recall / Sensibilidade (classe 1) | 0,9070 |
| Especificidade (classe 0) | 0,8776 |
| F1-score (classe 1) | 0,8864 |

```
              previsto 0   previsto 1
   real 0            43            6
   real 1             4           39
```

### 7.2 Test (n = 98)

| Métrica | Valor |
|---|---:|
| AUC-ROC | 0,9762 |
| Recall / Sensibilidade (classe 1) | 0,8478 |
| Especificidade (classe 0) | 0,9615 |
| F1-score (classe 1) | 0,8966 |

```
              previsto 0   previsto 1
   real 0            50            2
   real 1             7           39
```

Os perfis de erro das duas partições são opostos: validation erra mais para falso positivo (6 FP / 4 FN), test erra mais para falso negativo (2 FP / 7 FN). Com n ≈ 95 em cada partição, contagens dessa ordem envolvem poucas unidades, e a diferença observada é compatível com variação amostral entre duas amostras pequenas. Nenhum teste de hipótese foi conduzido para comparar as partições, de modo que não se afirma aqui nem que há diferença real de comportamento, nem que não há.

Artefatos: `reports/sprint3/baseline/baseline_metrics.json` e `reports/sprint3/baseline/confusion_matrix.csv`.

## 8. Intervalos de confiança

Bootstrap **percentil em nível de paciente** sobre a partição `test`:

- **1000 réplicas**, todas válidas (`n_bootstrap_valid = 1000`, nenhuma descartada);
- **seed 42**;
- unidade de reamostragem: `patient_id` — 63 pacientes sorteados com reposição por réplica, cada paciente entrando com todos os seus nódulos;
- **modelo não retreinado**: probabilidades e predições são as do modelo já ajustado; nenhum `fit` de `StandardScaler` ou `LogisticRegression` ocorre dentro do laço;
- limites: percentis 2,5 e 97,5.

| Métrica | Estimativa | IC 95% | Largura |
|---|---:|---|---:|
| AUC-ROC | 0,9762 | [0,9477 – 0,9946] | 0,047 |
| Recall (classe 1) | 0,8478 | [0,7347 – 0,9524] | 0,218 |
| Especificidade (classe 0) | 0,9615 | [0,8980 – 1,0000] | 0,102 |
| F1 (classe 1) | 0,8966 | [0,8172 – 0,9575] | 0,140 |

A reamostragem é por paciente, e não por nódulo, justamente porque nódulos do mesmo paciente não são independentes; reamostrar nódulos individualmente subestimaria a largura dos intervalos.

O limite superior de 1,0000 na especificidade é comportamento esperado de intervalo percentil próximo do teto e **não** deve ser lido como evidência de especificidade perfeita.

## 9. Coeficientes

No espaço padronizado pelo `StandardScaler` — **não interpretáveis em unidades originais** (mm, mm³):

| Termo | Coeficiente |
|---|---:|
| `original_shape_Maximum3DDiameter` | +2,694712 |
| `original_shape_MeshVolume` | +0,979678 |
| `original_shape_Sphericity` | −0,556207 |
| intercepto | +0,944129 |

O JSON registra também `scaler_mean` e `scaler_scale`, o que permite reconstruir os coeficientes em unidades originais sem reexecutar o treino.

O coeficiente de `Maximum3DDiameter` tem a maior magnitude entre os três. Isso descreve **este ajuste específico**, com estas três features e estes dados de treino — não estabelece importância causal nem hierarquia de relevância entre os atributos.

Duas ressalvas limitam a leitura desses valores. Primeira: `MeshVolume` e `Maximum3DDiameter` medem ambos o tamanho da lesão e são, por construção, grandezas relacionadas; features correlacionadas dividem entre si a contribuição no ajuste, de modo que a magnitude individual de cada coeficiente não é estável nem separável. A correlação entre as features não foi quantificada nesta atividade. Segunda: os valores estão no espaço padronizado e não expressam efeito por unidade física.

Uma leitura plausível — não verificada aqui — é que o tamanho da lesão esteja associado ao alvo porque também influencia a percepção radiológica de malignidade: os radiologistas enxergam a dimensão do nódulo ao atribuir o escore de 1 a 5. Isso **não** caracteriza vazamento no sentido do split, cuja disjunção por paciente está verificada em zero; é uma hipótese sobre como o alvo foi construído, que convém ter em mente ao interpretar tanto os coeficientes quanto a magnitude das métricas.

## 10. Limitações

1. **O alvo não é diagnóstico anatomopatológico.** É a mediana de escores subjetivos de malignidade atribuídos por radiologistas. Todo resultado aqui é score de risco radiológico.
2. **A AUC alta não deve ser interpretada como desempenho clínico diagnóstico.** Valores de 0,94–0,98 com três features medem a separação entre classes de um alvo derivado de escores radiológicos, não capacidade diagnóstica. A hipótese levantada na Seção 9 — associação entre tamanho e percepção de malignidade — oferece uma explicação possível para a magnitude, mas não foi testada nesta atividade.
3. **Os coeficientes não indicam importância causal.** As duas features de tamanho são grandezas relacionadas, e sua correlação não foi quantificada aqui; a magnitude individual de cada coeficiente descreve este ajuste e não deve ser lida como ranking de relevância (Seção 9).
4. **O IC do Recall é relativamente largo** — [0,7347 – 0,9524], amplitude de 0,218. Com 46 positivos distribuídos em 63 pacientes, cada falso negativo a mais ou a menos move a métrica em cerca de 2 pontos. Afirmações do tipo "o baseline detecta 85% dos nódulos de alto risco" são frágeis sob esse intervalo.
5. **Os intervalos representam a incerteza deste split de teste**, não a variabilidade entre diferentes seeds de particionamento. `GroupShuffleSplit` não estratifica — ignora `y` por construção —, e o protocolo (Seção 6.2) já documenta variância relevante entre folds. Quantificar essa segunda fonte de incerteza exigiria validação cruzada agrupada, fora do escopo desta atividade.
6. **O conjunto de teste já foi avaliado** para estabelecer este baseline e deve ser tratado como **congelado** nas próximas comparações. Reavaliá-lo repetidamente ao longo do desenvolvimento corroeria sua função de estimativa não enviesada.
7. **Decisões futuras e qualquer tuning devem usar treino e validação**, reservando o teste para a avaliação final do modelo escolhido.
8. **Cobertura incompleta da extração:** 607 dos 616 nódulos da coorte teórica, com 19 linhas ausentes ainda não explicadas (Seção 2.1).
9. **Generalização externa não foi avaliada.** Todos os dados vêm do LIDC-IDRI; nenhum conjunto externo foi utilizado.

## 11. Conclusão

O baseline de Regressão Logística descrito neste documento fica registrado como a **referência quantitativa oficial da Sprint 3**. Ele é reprodutível a partir dos scripts versionados, tem cada invariante estrutural verificada antes da escrita de qualquer artefato, e reporta sua própria incerteza.

Seu papel é servir de referência de comparação: modelos subsequentes — com mais features, outras classes de atributos ou outros algoritmos — devem demonstrar ganho **sobre a partição de validação**, com intervalos de confiança que sustentem a diferença, antes de qualquer avaliação final no teste congelado.

Nenhuma afirmação de superioridade clínica ou de generalização externa é feita ou sustentada por estes resultados.

---

## Reprodutibilidade

| Etapa | Script | Artefato |
|---|---|---|
| Tabela de modelagem | `scripts/build_modeling_table.py` | `data/modeling_table_sprint3.csv` |
| Split por paciente | `scripts/make_patient_split.py` | `reports/sprint3/baseline/patient_split.csv` |
| Baseline e IC | `scripts/run_baseline.py` | `reports/sprint3/baseline/baseline_metrics.json`, `confusion_matrix.csv` |

Seed fixa `42` em todas as operações estocásticas. Ambiente: conda env `pi3-radiomics`, Python 3.9.23, scikit-learn 1.3.2. Os três scripts abortam sem gravar nada se qualquer invariante falhar.

Quanto a determinismo, apenas o que foi efetivamente verificado: `make_patient_split.py` e `run_baseline.py` foram reexecutados e produziram artefatos **byte a byte idênticos** (comparação de hash). `build_modeling_table.py` não teve reexecução comparada dessa forma — seu resultado depende do banco de anotações do `pylidc` e das funções de seleção, ambos determinísticos por construção, mas a identidade byte a byte não foi medida e portanto não é afirmada aqui.

`data/modeling_table_sprint3.csv` é dado derivado de imagem médica e **não é versionado**, conforme a política do repositório. Os artefatos em `reports/sprint3/baseline/` contêm: o split (`patient_id` e partição atribuída), os parâmetros experimentais e do modelo (features, hiperparâmetros, threshold, seed, versão do scikit-learn, coeficientes e estatísticas do scaler), a composição das partições e as métricas com seus intervalos de confiança. **Não contêm imagens, dados DICOM nem a tabela radiômica completa** — nenhum valor de feature por nódulo é gravado em `reports/`.

---

_Este documento registra as decisões e resultados do baseline oficial da Sprint 3. Qualquer alteração de modelo, features, split ou threshold deve ser registrada em nova revisão, mantendo o histórico das decisões anteriores._
