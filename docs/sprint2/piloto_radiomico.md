# Piloto de Extração Radiômica — Sprint 2

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Escopo do documento:** documentação técnica canônica do piloto de extração radiômica da Sprint 2
**Política de alvo:** definida em [`docs/protocolo_coorte_target_sprint2.md`](../protocolo_coorte_target_sprint2.md) — este documento **não** redefine a regra, apenas a aplica
**Status:** consolidado — Bloqueador 1 (regra de target) e Bloqueador 2 (comparação de discretização) **resolvidos**

---

## Aviso metodológico obrigatório

A variável `malignancy` do LIDC-IDRI é uma **avaliação subjetiva de probabilidade/aparência de malignidade**, atribuída por radiologistas em escala de 1 a 5. **Não é confirmação anatomopatológica de câncer.** Qualquer alvo derivado dela — e qualquer modelo treinado sobre esse alvo — produz um score de risco baseado em características radiológicas disponíveis no dataset, não um diagnóstico. O LIDC-IDRI não fornece classificação Lung-RADS nativa.

---

## Estado dos bloqueadores

| Bloqueador | Assunto | Estado | Onde está documentado |
|---|---|---|---|
| 1 | Regra de consolidação do alvo a partir da mediana de `malignancy` | **Resolvido** | Seções 4, 8, 9 e 10 |
| 2 | Comparação `binCount = 64` × `binWidth = 25` HU contaminada por geometria de recorte | **Resolvido** | Seção 12 |

Ambos foram corrigidos no código, reexecutados sobre dados reais e revalidados nos artefatos. Nenhuma seção deste documento depende de resultado pendente.

---

## 1. Objetivo do piloto

Verificar, em escala reduzida e antes de comprometer tempo de processamento sobre a coorte inteira, se o pipeline converte de ponta a ponta um nódulo anotado do LIDC-IDRI em um vetor de atributos quantitativos utilizável para modelagem. Concretamente, o piloto responde a três perguntas:

1. A consolidação das anotações em máscara e a extração de features executam sem falha sobre dados reais?
2. A tabela resultante é íntegra — sem `NaN`, `Inf` ou colunas constantes — e fisicamente plausível?
3. A execução é reprodutível, isto é, a mesma entrada produz a mesma saída numérica?

O piloto **não** tem por objetivo produzir resultados de modelagem, nem fechar decisões metodológicas de configuração.

## 2. Escopo

| Item | Valor |
|---|---|
| Pacientes avaliados | 25 (os primeiros em ordem alfabética de identificador, `LIDC-IDRI-0001`–`LIDC-IDRI-0025`) |
| Forma de seleção | sequencial, **não aleatória** — ver Seção 10 |
| Coorte completa (referência) | 1.010 pacientes / 1.018 exames na base de anotações do `pylidc` |
| Fora do escopo | extração sobre a coorte completa, seleção de atributos, treinamento de modelo, decisão final de discretização |

## 3. Configuração experimental

Duas versões executadas sobre os mesmos 25 pacientes, idênticas exceto pelo modo de discretização de intensidade:

| Parâmetro | `piloto_v2` | `piloto_v3` |
|---|---|---|
| Espaçamento (reamostragem) | `[1.0, 1.0, 1.0]` mm isotrópico | idem |
| Interpolador | `sitkBSpline` | idem |
| Modo de discretização | `binCount = 64` | `binWidth = 25` HU |
| Máscara | `consenso50` | idem |
| Mínimo de anotadores | 3 | idem |
| Diâmetro mínimo | 3,0 mm | idem |
| Margem de contexto antes da reamostragem | 4 voxels | idem |
| Máx. tentativas de extração | 2 | idem |
| Classes habilitadas | `shape`, `firstorder`, `glcm`, `glrlm`, `glszm` | idem |
| PyRadiomics | `3.1.1.dev111+g8ed579383` (commit fixado `8ed579383`) | idem |

Ambas as tabelas foram extraídas na **mesma sessão Colab**, com o mesmo ambiente de execução — condição verificada e registrada nos artefatos (Seção 12.2). Existe ainda uma terceira tabela histórica, `piloto_v3_oficial`, usada **apenas como controle externo de regressão** (Seção 12.6).

**Ressalva sobre discretização:** `binWidth = 25` é registrado aqui como **a configuração da v3 testada**. A comparação entre `binCount = 64` e `binWidth = 25` está validada quanto à *sensibilidade* dos atributos (Seção 12), mas isso **não** torna `binWidth = 25` uma decisão metodológica aprovada — a escolha definitiva de discretização segue em aberto (Seção 14).

Parâmetros de alvo registrados nos configs (`mediana_max_benigno = 2.0`, `mediana_min_maligno = 4.0`, `regra_rotulo = mediana`) seguem o protocolo oficial.

## 4. Coorte e variável-alvo

**Fonte oficial e normativa da política de alvo:** [`docs/protocolo_coorte_target_sprint2.md`](../protocolo_coorte_target_sprint2.md), Seções 4, 5.2 e 5.3. O resumo abaixo existe só para leitura deste documento; em qualquer divergência, o protocolo prevalece.

Elegibilidade: nódulo (cluster de annotations) com **≥ 3 anotadores** com características completas. O diâmetro mínimo de 3 mm é mantido como *assert* de sanidade, não como filtro efetivo.

Consolidação: mediana dos escores de `malignancy` das annotations do cluster, binarizada em três faixas — duas classes e uma zona indefinida —, com a causa da indefinição registrada em `exclusion_reason`:

| Mediana | `alvo` | `indeterminado` | `exclusion_reason` |
|---|---|---|---|
| ≤ 2.0 | 0 | `False` | `included` |
| ≥ 4.0 | 1 | `False` | `included` |
| = 3.0 | `NaN` | `True` | `consensus_indeterminate` |
| 2.5 ou 3.5 | `NaN` | `True` | `fractional_median_even_raters` |

As duas causas de indefinição têm naturezas distintas — indecisão real de consenso *versus* efeito estrutural de contagem par de observadores — e por isso são **sempre reportadas separadamente**, nunca agregadas em um único número.

Nódulos indeterminados **não são descartados**: permanecem na tabela com `alvo = NaN` e `exclusion_reason` preenchido, preservando a possibilidade de revisão metodológica futura sem reprocessar a extração.

## 5. Segmentação / máscara utilizada no piloto

Máscara `consenso50`: os contornos dos radiologistas do cluster são consolidados em uma única região de interesse por consenso de 50% (`clevel = 0.5`), via `pylidc.utils.consensus`.

Antes da extração, a caixa envolvente do nódulo é recortada do volume e expandida em **4 voxels em cada direção**, dando vizinhança à interpolação B-spline usada na reamostragem para 1×1×1 mm. Sem essa margem, a interpolação cúbica pode produzir *overshoot* nas bordas do recorte — foi exatamente essa a causa do defeito histórico descrito na Seção 12.1. A margem está presente em **ambos os lados** do experimento definitivo (`margem_contexto_voxels = 4`).

Duas salvaguardas rodam sobre os voxels dentro da máscara antes de acionar o extrator: descarte se houver intensidade não finita, e descarte se houver intensidade fora da faixa fisiológica esperada (|HU| > 5000). No piloto, **nenhum nódulo foi descartado por qualquer uma delas**.

`consenso50` é configuração **provisória** — a estratégia definitiva de consolidação de máscara é decisão em aberto (Seção 14).

## 6. Extração radiômica

Para cada nódulo elegível, um extrator PyRadiomics é instanciado **isoladamente por chamada** (não reutilizado ao longo do laço), sobre a sub-imagem reamostrada e a máscara correspondente, com as cinco classes de atributos habilitadas.

A tabela produzida tem uma linha por nódulo, com identificadores e metadados de alvo à frente e as features numéricas em seguida. Os escores brutos de cada radiologista (`malignancy_escores`) e a mediana (`malignancy_mediana`) são preservados em coluna própria, de modo que **a coluna de alvo pode ser recalculada sem reextrair features** caso a política mude.

Colunas de rastreabilidade de alvo presentes em cada linha: `malignancy_escores`, `malignancy_mediana`, `valor_central`, `alvo`, `indeterminado`, `exclusion_reason`.

## 7. Resultado da extração

Valores registrados em `validacao_piloto_v2_consenso50.json` e `validacao_piloto_v3_oficial_consenso50.json` — **idênticos entre as configurações**, como esperado, já que a discretização não altera quais nódulos são elegíveis. Os mesmos 44 `nodule_id` aparecem também no lado `v3` do experimento definitivo (Seção 12.2):

| Métrica | Valor |
|---|---|
| Pacientes avaliados | 25 |
| Pacientes com ao menos um nódulo extraído | 20 |
| Nódulos extraídos | **44** |
| Nódulos descartados | **30** — todos por `leitores_insuficientes` |
| Falhas de extração | **0** |
| Features por nódulo | **88** |

Composição das 88 features: `shape` 14 · `firstorder` 18 · `glcm` 24 · `glrlm` 16 · `glszm` 16.

Integridade da tabela: **0** colunas com `NaN`, **0** com `Inf`, **0** colunas constantes, **0** linhas com `NaN`.

## 8. Distribuição corrigida do target

Sobre os 44 nódulos extraídos, aplicada a regra oficial:

| Categoria | n | % dos 44 |
|---|---|---|
| `alvo = 0` (baixo risco) | **5** | 11,4% |
| `alvo = 1` (alto risco) | **10** | 22,7% |
| `consensus_indeterminate` (mediana 3.0) | **17** | 38,6% |
| `fractional_median_even_raters` (medianas 2.5 / 3.5) | **12** | 27,3% |
| **Total indeterminado** | **29** | **65,9%** |
| **Base modelável do piloto** (`alvo` 0 ou 1) | **15** | 34,1% |

Identidades verificadas nos artefatos: `5 + 10 + 29 = 44` e `17 + 12 = 29`.

Leitura: **no piloto, a maioria dos nódulos extraídos não recebe alvo binário.** A base efetivamente modelável são 15 dos 44 nódulos. Isso não é defeito de extração — os 44 nódulos foram extraídos com sucesso e suas 88 features são válidas; é consequência direta da política de alvo, que recusa rotular casos sem consenso suficiente.

## 9. Validação populacional

Verificação executada sobre a base de anotações embutida no `pylidc`, que cobre os 1.018 exames independentemente da presença local de imagem DICOM — nenhuma imagem foi baixada e nenhuma extração radiômica foi executada para esta seção. Critérios de elegibilidade idênticos aos do pipeline.

| Métrica | Valor | % dos elegíveis |
|---|---|---|
| Pacientes com ao menos um nódulo elegível | **696** | — |
| Nódulos elegíveis | **1.392** | 100% |
| `alvo = 0` | **315** | 22,6% |
| `alvo = 1` | **301** | 21,6% |
| `consensus_indeterminate` (mediana 3.0) | **513** | 36,9% |
| `fractional_median_even_raters` (2.5 / 3.5) | **263** | 18,9% |
| **Total indeterminado** | **776** | **55,7%** |
| **Base modelável** | **616 nódulos** | 44,3% |
| Pacientes na base modelável | **422** | — |

Identidades verificadas: `315 + 301 + 776 = 1.392`, `513 + 263 = 776`, `315 + 301 = 616`.

Balanceamento da base modelável: **51,1% / 48,9%** — saudável, sem necessidade de técnicas de rebalanceamento nesta fase.

**A indeterminação de alvo é característica estrutural do LIDC-IDRI, não artefato de amostra pequena:** mais da metade dos nódulos elegíveis da população não recebe rótulo binário sob a política oficial.

Convergência da estimativa com o tamanho da amostra (amostras aleatórias, semente fixa `42`, intervalo de confiança de Wilson):

| Pacientes sorteados | Nódulos | `consensus` | `fractional` | Indeterminados | % | IC 95% |
|---|---|---|---|---|---|---|
| 25 | 55 | 20 | 8 | 28 | 50,9% | 38,1 – 63,6 |
| 50 | 109 | 47 | 21 | 68 | 62,4% | 53,0 – 70,9 |
| 100 | 194 | 77 | 27 | 104 | 53,6% | 46,6 – 60,5 |
| 200 | 370 | 127 | 67 | 194 | 52,4% | 47,3 – 57,5 |
| 500 | 972 | 370 | 187 | 557 | 57,3% | 54,2 – 60,4 |
| 696 (censo completo) | 1.392 | 513 | 263 | 776 | 55,7% | 53,1 – 58,3 |

## 10. Avaliação da amostra sequencial

Os 25 pacientes do piloto foram escolhidos por ordem alfabética de identificador, não por sorteio. Isolando exatamente esses pacientes dentro do censo completo:

| Base | Nódulos elegíveis | `consensus` | `fractional` | Indeterminados | % |
|---|---|---|---|---|---|
| Amostra sequencial (25 pacientes) | 54 | 21 | 15 | **36** | **66,7%** |
| População elegível (696 pacientes) | 1.392 | 513 | 263 | 776 | 55,7% |
| **Diferença** | — | — | — | — | **+10,9 p.p.** |

> **A amostra sequencial do piloto apresenta sobre-representação de casos indeterminados em relação à população elegível.**

Essa leitura é sustentada pelos intervalos de confiança da Seção 9: os 66,7% da amostra sequencial ficam **acima** do limite superior do IC 95% de uma amostra aleatória de 25 pacientes (38,1 – 63,6) e também do IC 95% da própria população (53,1 – 58,3).

O escopo desta conclusão é deliberadamente estreito: a análise avaliou **especificamente a distribuição da indeterminação do target**. Ela não permite afirmar, em um sentido ou no outro, que a amostra sequencial seja genericamente representativa ou não representativa quanto a outras propriedades — tamanho de nódulo, textura, características de aquisição ou perfil de paciente.

Implicação prática: métricas de proporção de alvo obtidas sobre o piloto **não devem ser extrapoladas** para a coorte completa; use os números da Seção 9 para dimensionamento.

## 11. Determinismo e qualidade

**Reprodutibilidade da mesma configuração.** O teste registrado em `validacao_piloto_v3_oficial_consenso50.json` executa a extração duas vezes sobre a mesma sub-amostra de 10 pacientes, em execuções independentes:

| Verificação | Resultado |
|---|---|
| Pacientes testados | 10 |
| Nódulos na execução A / B | 12 / 12 |
| Mesmos identificadores nas duas execuções | Sim |
| Maior diferença absoluta entre features | **0,0** |
| Determinismo confirmado | Sim |

Diferença numérica **nula**, e não apenas abaixo de um limiar de tolerância: as duas execuções produzem exatamente os mesmos valores de ponto flutuante para cada uma das 88 features de cada nódulo testado.

Escopo do resultado: ele atesta a reprodutibilidade **de uma mesma configuração entre execuções**. Não diz nada sobre equivalência entre configurações diferentes de discretização — assunto tratado na Seção 12.

**Correções de estabilidade incorporadas ao pipeline**, ambas presentes no código atual:

1. **Extrator isolado por chamada** — `RadiomicsFeatureExtractor` é instanciado uma vez por nódulo, em vez de reutilizado ao longo do laço, eliminando contaminação de estado entre nódulos.
2. **Margem de contexto na reamostragem** — a caixa envolvente é expandida em 4 voxels antes do recorte, evitando *overshoot* numérico da interpolação B-spline nas bordas.

**Qualidade da tabela produzida:** 0 `NaN`, 0 `Inf`, 0 colunas constantes, 0 linhas incompletas; `Sphericity` dentro de `[0, 1]`; nenhum nódulo descartado por intensidade não finita ou fora de faixa HU.

**Coerência de target verificada nos artefatos** (CSV e Parquet das versões v2, v3 de comparação e v3 oficial):

- nenhum nódulo com mediana 2.5 ou 3.5 possui alvo binário;
- todo nódulo com mediana 3.0 está indeterminado;
- `indeterminado == alvo.isna()`;
- `exclusion_reason != "included"` se e somente se `alvo` é `NaN`;
- `alvo = 0 ⟺ mediana ≤ 2.0` e `alvo = 1 ⟺ mediana ≥ 4.0`;
- as contagens dos JSONs de validação conferem com os CSVs correspondentes.

## 12. Experimento de discretização — `binCount = 64` × `binWidth = 25` HU

### 12.1 Defeito histórico, agora corrigido

A comparação `v2 × v3` executada originalmente era **inválida**. A tabela `v2` daquela época havia sido gravada **antes** da introdução de `margem_contexto_voxels = 4`, enquanto a `v3` já a possuía. Os dois lados diferiam, portanto, em **geometria de recorte e discretização ao mesmo tempo** — o efeito da discretização ficava confundido com o da margem, e nenhum dos dois podia ser isolado.

Evidências do defeito, levantadas na auditoria:

- diferenças de atributos de `shape` de até **66,25** entre os dois lados, em 31 dos 44 nódulos — forma não pode variar com discretização, logo a divergência denunciava outra causa;
- intensidade fisicamente impossível em `LIDC-IDRI-0018_N02`: `firstorder_Minimum = −9,43e+18` HU, com `Energy = 4,20e+38`, decorrente de *overshoot* da interpolação B-spline sem margem de contexto;
- o `comparacao_v2_v3_discretizacao.json` histórico registrava `shape_max_diff_absoluta = 0.0`, valor que não correspondia às tabelas efetivamente versionadas.

> **A comparação histórica não deve ser usada como evidência do efeito isolado da discretização.**

### 12.2 Experimento definitivo

Reexecutado com ambos os lados extraídos **na mesma sessão Colab**, sem reinício de runtime nem reinstalação de dependências entre as duas extrações.

| Item | Valor |
|---|---|
| Pacientes | 25 |
| Nódulos — lado `v2` / lado `v3` | 44 / 44 |
| Mesmos `nodule_id` nos dois lados | `true` |
| `v2` — discretização | `binCount = 64` |
| `v3` — discretização | `binWidth = 25` HU |

Parâmetros mantidos **idênticos** nos dois lados — é o que permite atribuir a diferença observada à discretização e não a outro fator:

`espacamento = [1.0, 1.0, 1.0]` · `interpolador = sitkBSpline` · `mascara = consenso50` · `min_anotadores = 3` · `diametro_min_mm = 3.0` · `margem_contexto_voxels = 4`

Proveniência de ambiente registrada no artefato:

| Campo | Valor |
|---|---|
| `mesma_sessao_colab` | `true` |
| `ambiente_identico` | `true` |
| `runtime_id` | `ef406b45-e751-4634-9b06-91198b4f4378` |

### 12.3 Controle de forma

| Métrica | Valor |
|---|---|
| `shape_max_diff_absoluta` | `4.547473508864641e-13` |
| `tolerancia_shape` | `1e-6` |
| `shape_invariante` | `true` |

Os 14 atributos de `shape` permanecem **invariantes dentro da tolerância numérica definida** — precisamente o controle que a comparação histórica não passava.

### 12.4 Efeito sobre as demais classes

| Classe | Spearman médio | Spearman mínimo | Diferença média |
|---|---|---|---|
| `firstorder` | 0,962 | 0,568 | 4,3% |
| `glcm` | 0,704 | 0,214 | 44,2% |
| `glrlm` | 0,756 | 0,106 | 44,2% |
| `glszm` | 0,666 | −0,049 | 74,6% |

### 12.5 Conclusão

> Mantendo coorte, máscaras, geometria, interpolação e ambiente computacional constantes, a alteração da estratégia de discretização de `binCount = 64` para `binWidth = 25` HU produziu impacto relevante principalmente nas características radiômicas de **textura**, enquanto as características de **forma** permaneceram invariantes dentro da tolerância numérica definida.

**Escopo da conclusão.** Este experimento avalia **sensibilidade à discretização**, não superioridade preditiva. Ele **não** estabelece que `binWidth = 25` seja superior a `binCount = 64`, nem o contrário. A escolha definitiva de discretização permanece em aberto (Seção 14).

### 12.6 Papel da `v3_oficial`

A `v3_oficial` versionada foi utilizada **apenas como controle externo de regressão**, e **não** como lado do experimento definitivo: seu config não possui proveniência completa de ambiente (campos `ambiente` e `runtime_id` ausentes), logo não é possível comprovar que tenha sido produzida no mesmo runtime que a `v2`. Ela serve para confirmar que a nova extração não regrediu em relação ao histórico, não para sustentar a conclusão da Seção 12.5.

## 13. Limitações

1. **`malignancy` é avaliação subjetiva**, não confirmação anatomopatológica (ver aviso no topo). Todo alvo derivado é score de risco radiológico.
2. **Escala do piloto.** 25 pacientes, 20 com nódulo, 44 nódulos. Serve para validar o pipeline; não sustenta conclusão sobre desempenho de modelo.
3. **Base modelável pequena no piloto.** Apenas 15 dos 44 nódulos recebem alvo binário — insuficiente para qualquer avaliação estatística.
4. **A amostra sequencial do piloto apresenta sobre-representação de casos indeterminados em relação à população elegível** (+10,9 p.p.), conforme Seção 10. Proporções de alvo do piloto não são extrapoláveis.
5. **A comparação de discretização mede sensibilidade, não desempenho.** A Seção 12 estabelece *quanto* os atributos mudam entre `binCount = 64` e `binWidth = 25` HU, sobre 44 nódulos. Ela não avalia qual configuração produz melhor capacidade preditiva — isso exigiria modelagem sobre uma base modelável de tamanho adequado, fora do escopo do piloto.
6. **`binWidth = 25` não é decisão aprovada.** É a configuração da v3 testada. A escolha definitiva de discretização permanece em aberto.
7. **Máscara `consenso50` é provisória.** Estratégia de consolidação ainda não fechada pelo grupo.
8. **Divergência CSV × Parquet nas features.** As duas serializações do mesmo resultado não são bit-a-bit idênticas nas colunas de features: o CSV trunca casas decimais frente ao `float64` do Parquet, com erro **relativo** máximo da ordem de 1e-13. É condição **pré-existente** dos artefatos, não introduzida pelas correções de target, e não afeta nenhuma coluna de identificação ou de alvo — essas são idênticas nas duas. Para trabalho numérico, prefira o Parquet.
9. **Artefatos `v1` não foram atualizados.** A versão v1 do piloto é registro histórico, anterior às correções de estabilidade e à política de alvo oficial; seus CSV/Parquet/JSON permanecem com a regra antiga e **não devem ser usados como referência**.
10. **Limitação de estratificação por paciente.** Na base modelável da população, apenas 42 dos 422 pacientes possuem nódulos de ambas as classes, o que limita estratificação perfeita por grupo e gera variância entre folds (ver protocolo, Seção 6.2).
11. **Documentos `.docx` históricos.** Os arquivos em `01_piloto_experimental/DOCUMENTAÇÃO/` precedem a correção da regra de target e contêm números e interpretações desatualizados. São tratados como documentos históricos; **este Markdown é a referência canônica** do piloto.

## 14. Decisões ainda abertas

| # | Decisão | Estado |
|---|---|---|
| 1 | **Discretização de intensidade** — `binCount = 64` ou `binWidth = 25` HU | Em aberto. A sensibilidade dos atributos às duas configurações já está medida (Seção 12); o que falta é o critério de escolha, que depende de desempenho preditivo. Nada neste documento pressupõe uma escolha |
| 2 | **Estratégia de consolidação de máscara** — consenso, união, interseção ou leitor único | Em aberto; `consenso50` é provisório |
| 3 | **Unidade de análise da extração** — por cluster consolidado (uma ROI de consenso) ou por annotation individual | Em aberto (protocolo, Seção 8.1). Deve respeitar o split por paciente em qualquer caso |
| 4 | **Destino dos nódulos indeterminados** na modelagem — permanecem na tabela com `alvo = NaN`; se e como serão aproveitados na Sprint 3 | Em aberto |
| 5 | **Disponibilidade de Lung-RADS no LIDC-IDRI** | Pendência registrada desde a Sprint 1; o LIDC-IDRI não fornece a classificação nativamente |
| 6 | **Permanência dos `.docx` no PR** | A ser decidida na etapa de reorganização da documentação |

Já **fechados** pelo protocolo oficial e, portanto, fora desta lista: a regra de consolidação (`mediana`), as fronteiras do alvo (`≤ 2.0` / `≥ 4.0`), o tratamento das medianas 2.5/3.0/3.5 e o mínimo de 3 anotadores.

---

## Rastreabilidade e reprodutibilidade

| Notebook | Produz |
|---|---|
| `01_piloto_experimental/NOTBOOKS/PI3_G4_sprint2_v2_extração_piloto.ipynb` | `config_piloto_v2.json`, `piloto_piloto_v2_consenso50.csv/.parquet`, `validacao_piloto_v2_consenso50.json`, `descartes_piloto_v2_consenso50.csv` |
| `01_piloto_experimental/NOTBOOKS/PI3_G4_piloto_v3_binWidth_validacao.ipynb` | `config_piloto_v3_oficial.json`, `piloto_piloto_v3_oficial_consenso50.csv/.parquet`, `validacao_piloto_v3_oficial_consenso50.json`, `descartes_piloto_v3_oficial_consenso50.csv` — controle externo de regressão (Seção 12.6) |
| `01_piloto_experimental/NOTBOOKS/PI3_G4_comparacao_binCount_binWidth.ipynb` | `config_piloto_v3.json`, `piloto_piloto_v3_consenso50.csv`, `comparacao_v2_v3_discretizacao.json` — experimento definitivo de discretização (Seção 12) |
| `01_piloto_experimental/NOTBOOKS/PI3_G4_verificacao_populacional_valor3.ipynb` | `verificacao_populacional_valor_central_3.json` (Seções 9 e 10) |

Artefatos em `01_piloto_experimental/JSON E CSV/`. Semente fixa `42` em todas as operações estocásticas. PyRadiomics fixado no commit `8ed579383`.

**Fontes deste documento:** [`docs/protocolo_coorte_target_sprint2.md`](../protocolo_coorte_target_sprint2.md) (política de alvo e coorte); os quatro notebooks corrigidos acima; os JSONs de configuração e validação corrigidos; e a auditoria documental dos `.docx` históricos. Somente informação já validada nos artefatos foi consolidada aqui.
