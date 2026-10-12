# Auditoria da Reprodução da Seleção de Features — Sprint 4

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Card relacionado:** S4 — Cartão 4, Etapa 0: reprodução independente da seleção de features
**Protocolo:** [`docs/sprint4/protocolo_valor_incremental.md`](../../../docs/sprint4/protocolo_valor_incremental.md) (commit `9e9b3bd`), Seções 5.2, 5.3 e 7.1
**Objeto auditado:** notebook de seleção de Juan, *S4 — Revisão e execução da seleção de features radiômicas*
**Resultado:** **reprodução divergente** — 6 das 8 features declaradas reproduzidas; 2 divergentes
**Status:** registro de auditoria — **a lista definitiva de features permanece pendente de decisão do grupo**
**Execução:** 10/10/2026 · **Registro:** 11/10/2026
**Revisão:** 11/10/2026 — incorpora a comparação com o pacote enviado por Juan ([`comparacao_pacote_juan.md`](comparacao_pacote_juan.md)) e os esclarecimentos de Juan (Seção 16). Os resultados verificados das Seções 3 a 10 e os hashes da Seção 15 não mudaram.
**Revisão:** 12/10/2026 — registra o recebimento do notebook da regeneração recente (Seção 16) e a limitação metodológica dos filtros (Seção 2.4)
**Revisão:** 12/10/2026 — registra a preservação versionada da reprodução no commit `0ef4039` (Seção 17). Os hashes históricos da Seção 15 não mudaram.

---

## Aviso de escopo

Esta auditoria reexecutou **somente o procedimento de seleção**, sobre o treino oficial. Nenhum modelo M0–M4 foi ajustado ou avaliado, nenhuma métrica de validação ou de teste foi calculada e nenhuma linha de validação ou de teste entrou em ajuste, filtro ou seleção.

O alvo é a suspeita radiológica de malignidade (mediana de escores de radiologistas), não malignidade histológica confirmada.

Neste documento:
- **"verificado"** designa o que foi medido nas execuções descritas aqui ou na comparação com o pacote de Juan;
- **"hipótese"** designa explicação possível que **não** foi testada;
- **"declarado por Juan"** designa informação dada pelo responsável pela seleção original, registrada como declaração e não como verificação independente.

Nada aqui atribui a divergência a erro de quem conduziu a seleção original: a evidência disponível não permite essa conclusão.

---

## 1. Objetivo

Verificar se o procedimento de seleção do notebook de Juan, reexecutado **sem alteração de algoritmo ou de parâmetros** no ambiente do projeto, devolve as oito features declaradas na conclusão do notebook (protocolo, Seção 5.1):

| # | Feature declarada |
|---:|---|
| 1 | `original_glcm_Imc2` |
| 2 | `original_shape_Sphericity` |
| 3 | `original_shape_SurfaceVolumeRatio` |
| 4 | `original_firstorder_90Percentile` |
| 5 | `original_firstorder_RootMeanSquared` |
| 6 | `original_shape_Elongation` |
| 7 | `original_firstorder_Skewness` |
| 8 | `original_ngtdm_Coarseness` |

A reprodução fiel é pré-requisito para avaliar M2, M3 e M4 na validação (protocolo, Seção 7.1, item 1).

## 2. Método

### 2.1 Procedimento reproduzido

O script de reprodução replica, linha a linha, as células de código do notebook (carga, binarização do alvo, split oficial, seleção, frequências e conjunto final), conforme descrito na Seção 5.3 do protocolo:

1. pool de 107 radiômicas `original_*` de `data/base_radiomica_oficial_422.csv`;
2. alvo de `data/vies_malignidade_mascara_pequena.csv`, binarizado pela regra oficial (≤ 2 → 0; ≥ 4 → 1; 2,5 / 3,0 / 3,5 excluídos);
3. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grupos = `patient_id`, sobre o treino oficial;
4. em cada fold: `VarianceThreshold(0.0)`; remoção de redundância por `|r de Pearson| > 0,90`, mantendo a coluna que aparece primeiro; `StandardScaler` → `LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42)`, com `C` escolhido por `GridSearchCV` em `logspace(-4, 2, 13)`, `scoring="roc_auc"`, CV interna `StratifiedGroupKFold(4, shuffle=True, random_state=100 + fold)`;
5. feature selecionada no fold se `|coef| > 1e-12`; lista final = selecionadas em **≥ 3 de 5 folds**;
6. Random Forest (500 árvores, `random_state = 42 + fold`, `class_weight="balanced_subsample"`), top-15 por fold, apenas como checagem de consistência.

### 2.2 Diferenças deliberadas em relação ao notebook

Nenhuma altera o cálculo:

| Diferença | Motivo |
|---|---|
| Caminhos apontam para os arquivos do projeto | o notebook lia os arquivos da pasta do Colab |
| Warnings **registrados** em vez de suprimidos | o notebook usa `warnings.filterwarnings("ignore")` |
| `n_jobs` recebido por argumento | permitir uma segunda execução em processo único (Seção 9) |
| Saídas gravadas fora do repositório | não criar artefatos de dados no repositório. Vale para as execuções A e B; as saídas agregadas da versão preservada estão versionadas (Seção 17). |

Não foi aplicada nenhuma das correções previstas para a CV aninhada (protocolo, Seção 5.4). Nenhum parâmetro foi alterado para tentar reproduzir a lista declarada.

### 2.3 Isolamento

Como no notebook original, a base radiômica e o arquivo de alvo são carregados inteiros em memória, e as partições são separadas pelo split oficial. As linhas de validação e de teste são usadas apenas na verificação de disjunção de pacientes entre partições e descartadas em seguida. A matriz que entra nos filtros, no LASSO e no Random Forest é construída **somente** com as linhas de treino.

Antes da seleção, o script verificou que essas linhas são exatamente os nódulos do treino oficial e abortaria em caso contrário (Seção 4). As checagens prévias também leem a tabela canônica inteira, apenas para comparar IDs, alvo, ordem e valores das linhas de treino.

### 2.4 Limitação metodológica do procedimento original

No procedimento original, os filtros são ajustados em todo o treino de cada fold de estabilidade, **antes** e **fora** da `GridSearchCV` interna:
- `VarianceThreshold` com `fit` sobre esse treino completo;
- o filtro de correlação calculado sobre o mesmo treino.

Só depois a CV interna de 4 folds escolhe `C` sobre a matriz já filtrada. Assim, cada fold de validação interna participou da definição do conjunto de features avaliado nele.
- Os filtros não usam o alvo, então não há vazamento de rótulo.
- A parte de avaliação de cada fold de estabilidade não entra em nenhum ajuste.
- Mesmo assim, a escolha de `C` não é estritamente independente dos filtros.

Esta reprodução **replica a limitação de propósito**, porque é uma reprodução fiel (Seção 2.2). A limitação vale também para a execução histórica e para a regeneração recente. Ela **não explica** a divergência entre essas execuções.

Essa forma de aplicar os filtros **não deve ser reutilizada na CV aninhada corrigida**: lá, os filtros entram no pipeline avaliado pela CV interna e são ajustados em cada treino interno, como prevê a Seção 5.4 do protocolo.

## 3. Ambiente, seeds e entradas

### 3.1 Ambiente

| Item | Valor |
|---|---|
| Python | 3.9.13 (`.venv` do projeto) |
| scikit-learn | 1.3.2 |
| numpy | 1.23.5 |
| pandas | 1.5.3 |
| scipy | 1.10.1 |
| joblib | 1.5.3 |
| Plataforma | Windows 10.0.26200 |
| Commit base do repositório | `9e9b3bd` (árvore limpa no momento da execução) |
| Tempo por execução | ≈ 111–113 s |

### 3.2 Seeds

| Uso | Seed |
|---|---|
| CV externa de estabilidade | 42 |
| CV interna do `GridSearchCV` | `100 + fold` (101 a 105) |
| `LogisticRegression` L1 | 42 |
| Random Forest | `42 + fold` (43 a 47) |

### 3.3 Hashes SHA-256 das entradas

| Arquivo | SHA-256 |
|---|---|
| `data/base_radiomica_oficial_422.csv` | `4068f7d661eac3bdfd33342a3759a0dfa80d42e47d4be4be00cd85b4e13563b1` |
| `data/vies_malignidade_mascara_pequena.csv` | `67f3fd2c3a55de75a58c40e7380480a5df0f5747ac017ec2b1f6ec091c3c6478` |
| `reports/sprint3/baseline/patient_split.csv` | `ed120c2b5c0f308d41c11cfc32dae0e897a179c71d6ea803588720ba1ba4f65f` |
| `data/modeling_table_sprint3.csv` (usado só nas checagens) | `5e92e7e860ad8edd327570c929cd263cd780be98902e57017551ad0080da60ab` |
| Notebook de Juan (cópia local, não versionada) | `48faaea60d2e6e3dc5af7955639b516e178adf93266d830e162f7dac570c34d8` |

Os arquivos em `data/` não são versionados. Não existe, no repositório, hash de referência dos arquivos que Juan usou no Colab (Seção 12).

## 4. Checagens prévias — verificado

Todas passaram; qualquer falha teria abortado a execução antes da seleção.

| Checagem | Resultado |
|---|---|
| IDs da base radiômica (formato da Sprint 3, duplicados, coerência com `patient_id`) | 958 linhas, 0 duplicados, 958/958 coerentes |
| IDs do arquivo de alvo | 958 linhas, 0 duplicados; mesmo conjunto de `nodulo_id` da base |
| Base modelável após binarização | 607 nódulos / 416 pacientes |
| **Treino oficial usado na seleção** | **291 pacientes / 417 nódulos**, idênticos ao treino canônico |
| Classes no treino | 209 (classe 0) / 208 (classe 1) |
| Alvo da seleção × alvo canônico | 417/417 idênticos |
| Pool de candidatas | 107 features; `original_nodule_idx` ausente |
| Ordem das 107 colunas × tabela canônica (sem `original_nodule_idx`) | idêntica |
| Valores das 107 features no treino × tabela canônica | idênticos; nenhum valor não finito |
| Oito features declaradas presentes no pool | sim |

## 5. Filtros e seleção por fold — verificado

Cada fold usa cerca de 4/5 do treino oficial.

| Fold | Nódulos / pacientes | Classes 0 / 1 | Após variância | Redundantes removidas | Após correlação | Melhor `C` | Selecionadas pelo LASSO |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 331 / 232 | 172 / 159 | 107 | 70 | 37 | 0,316 | 7 |
| 2 | 339 / 231 | 164 / 175 | 107 | 71 | 36 | 0,316 | 10 |
| 3 | 328 / 237 | 157 / 171 | 107 | 71 | 36 | 0,316 | 10 |
| 4 | 337 / 231 | 168 / 169 | 107 | 72 | 35 | 0,316 | 8 |
| 5 | 333 / 233 | 175 / 158 | 107 | 71 | 36 | 0,100 | 7 |

- O filtro de variância não removeu nenhuma feature em nenhum fold.
- O melhor `C` nunca ficou no limite da grade.
- **Nenhuma das oito features declaradas foi eliminada pelos filtros em nenhum fold**: todas chegaram ao LASSO nos cinco folds. A divergência, portanto, ocorre na etapa do LASSO, e não nos filtros.

## 6. Frequência de seleção pelo LASSO — verificado

Features selecionadas em pelo menos um fold:

| Frequência | Features |
|---|---|
| 5/5 | `shape_Sphericity`, `shape_SurfaceVolumeRatio` |
| 4/5 | `shape_Elongation`, `firstorder_90Percentile`, `firstorder_RootMeanSquared`, `glcm_Imc1`, `glcm_Imc2` |
| 3/5 | `glcm_ClusterShade` |
| 2/5 | `glcm_Id` |
| 1/5 | `firstorder_Skewness`, `ngtdm_Coarseness`, `shape_Flatness`, `shape_MajorAxisLength`, `glcm_Idmn`, `gldm_LargeDependenceLowGrayLevelEmphasis`, `ngtdm_Busyness` |

Pelo critério de estabilidade (≥ 3/5), a lista reproduzida tem **8 features** — as oito primeiras linhas acima.

## 7. Declaradas × reproduzidas — verificado

| Feature | Declarada | Reproduzida | Frequência no LASSO |
|---|:---:|:---:|---:|
| `original_shape_Sphericity` | ✓ | ✓ | 5/5 |
| `original_shape_SurfaceVolumeRatio` | ✓ | ✓ | 5/5 |
| `original_shape_Elongation` | ✓ | ✓ | 4/5 |
| `original_firstorder_90Percentile` | ✓ | ✓ | 4/5 |
| `original_firstorder_RootMeanSquared` | ✓ | ✓ | 4/5 |
| `original_glcm_Imc2` | ✓ | ✓ | 4/5 |
| `original_glcm_Imc1` | — | ✓ | 4/5 |
| `original_glcm_ClusterShade` | — | ✓ | 3/5 |
| `original_firstorder_Skewness` | ✓ | — | 1/5 |
| `original_ngtdm_Coarseness` | ✓ | — | 1/5 |

- **Seis coincidências:** `Sphericity`, `SurfaceVolumeRatio`, `Elongation`, `90Percentile`, `RootMeanSquared`, `Imc2`.
- **Duas divergências:** a reprodução inclui `glcm_Imc1` e `glcm_ClusterShade`; a lista declarada inclui `firstorder_Skewness` e `ngtdm_Coarseness`.
- Ambas as listas têm oito features. Nenhuma contém `original_shape_Maximum3DDiameter` ou `original_shape_MeshVolume`; ambas contêm `original_shape_Sphericity`, que também pertence a M1.

### 7.1 Coeficientes das quatro features divergentes

Coeficientes L1 no espaço padronizado; "—" = não selecionada no fold.

| Fold | `Skewness` | `Coarseness` | `Imc1` | `ClusterShade` |
|---:|---:|---:|---:|---:|
| 1 | −0,2834 | −0,0085 | — | — |
| 2 | — | — | +0,3273 | — |
| 3 | — | — | +0,2659 | −0,1815 |
| 4 | — | — | +0,1860 | −0,0151 |
| 5 | — | — | +1,0916 | −0,0139 |

Para referência, os valores absolutos de todos os coeficientes selecionados em todos os folds variam de 0,0085 a 2,01, com mediana 0,348.

- `Skewness` e `Coarseness` são selecionadas **apenas no fold 1**, e `Coarseness` com coeficiente próximo de zero.
- `Imc1` é selecionada nos folds 2 a 5, nunca no fold 1, com coeficientes não desprezíveis.
- `ClusterShade` atinge exatamente o limiar (3/5), e em duas dessas três seleções o coeficiente é próximo de zero.

O padrão é de **sensibilidade da seleção à partição**: o fold 1 favorece um subconjunto, e os folds 2 a 5 favorecem outro.

## 8. Random Forest e índice de Jaccard — verificado

Features no top-15 do Random Forest em ≥ 3 de 5 folds (**16 features**):

| Frequência no top-15 | Features |
|---|---|
| 5/5 | `glcm_Imc1`, `glcm_Imc2`, `shape_MajorAxisLength`, `shape_MeshVolume`, `ngtdm_Busyness`, `ngtdm_Coarseness`, `glrlm_LongRunEmphasis`, `glszm_LargeAreaEmphasis` |
| 4/5 | `firstorder_Skewness`, `glcm_Contrast`, `glrlm_GrayLevelNonUniformity` |
| 3/5 | `shape_Sphericity`, `shape_SurfaceVolumeRatio`, `firstorder_RootMeanSquared`, `glcm_Id`, `ngtdm_Contrast` |

| Medida | Reprodução | Declarado no notebook |
|---|---:|---:|
| Features estáveis no RF | 16 | 16 |
| Interseção LASSO × RF | 5 (`Sphericity`, `SurfaceVolumeRatio`, `Imc1`, `Imc2`, `RootMeanSquared`) | 5 |
| Índice de Jaccard LASSO × RF | 0,263 (5/19) | 0,263 |
| Features após o filtro de correlação, por fold | 35–37 | 35–39 |
| Features selecionadas pelo LASSO, por fold | 7–10 | 7–13 |

As três estatísticas pontuais do notebook — 16, 5 e 0,263 — são reproduzidas **exatamente**. As duas faixas declaradas contêm os valores reproduzidos, mas os extremos superiores declarados (39 e 13) não foram observados.

**Precisão acrescentada na revisão.** A coincidência é **só das contagens**. No pacote de Juan (regeneração recente), as 16 features estáveis do RF e a interseção de 5 são **outros conjuntos**:
- a interseção do pacote contém `Skewness` e `Coarseness`;
- a nossa contém `Imc1` e `RootMeanSquared`.

Detalhes em `comparacao_pacote_juan.md`, Seção 5.2.

O Random Forest é apenas checagem de consistência e não define a lista (protocolo, Seção 5.3).

## 9. Determinismo — verificado

| Execução | `n_jobs` | Resultado |
|---|---:|---|
| A | −1 (como no notebook) | lista reproduzida da Seção 7 |
| B | 1 (processo único) | idêntica |

Os três CSVs de saída (frequências, resumo por fold e lista final) são **idênticos byte a byte** entre A e B, assim como os blocos de checagens, detalhes por fold, resultado e comparação dos dois relatórios JSON. Os JSONs completos diferem apenas nos campos de execução (`n_jobs`, tempo e contagem de warnings capturados).

O resultado é determinístico neste ambiente e não depende do paralelismo.

## 10. Warnings e convergência — verificado

- A execução B captou **2 `ConvergenceWarning`** do liblinear. Na execução A, os mesmos warnings ocorreram dentro dos processos paralelos e não chegaram ao processo principal.
- Um diagnóstico separado, sobre o mesmo particionamento e os mesmos filtros, localizou os dois ajustes não convergidos: **ambos com `C = 100`**, um no fold externo 4 e outro no fold externo 5. Ao todo houve 260 ajustes na grade.
- `C = 100` **nunca** foi o valor escolhido. Os ajustes finais com o `C` escolhido convergiram em 10 a 13 iterações, sem warning.
- O notebook original suprime warnings; não se sabe se a execução de Juan teve não convergência.

## 11. Investigação da divergência

### 11.1 Causas descartadas dentro da reprodução — verificado

As verificações abaixo foram feitas **com os arquivos de entrada do projeto** (Seção 3.3) e **no ambiente da Seção 3.1**. Elas descartam causas de erro **dentro desta reprodução**. Não dizem nada sobre os arquivos nem sobre o ambiente que Juan usou no Colab.

| Possível causa, nesta reprodução | Evidência | Alcance da conclusão |
|---|---|---|
| Resultado não determinístico | execuções A e B idênticas byte a byte (Seção 9) | somente neste ambiente e com estes arquivos; não indica se uma execução em outro ambiente daria o mesmo resultado |
| Linhas, IDs ou alvo diferentes do treino oficial | checagens prévias: 417/417 nódulos e alvo idênticos ao treino canônico (Seção 4) | compatibilidade com a **base canônica do projeto** |
| Ordem ou valores das features diferentes | ordem e valores idênticos aos da tabela canônica (Seção 4) | compatibilidade com a **base canônica do projeto** |
| Eliminação das declaradas pelos filtros | as oito declaradas sobrevivem aos filtros em 5/5 folds (Seção 5) | nesta reprodução |
| Não convergência nos ajustes que definem a seleção | só `C = 100`, nunca escolhido; ajustes finais convergiram (Seção 10) | nesta reprodução |
| Parâmetro alterado na reprodução | o código replica o notebook; nenhuma correção da Seção 5.4 aplicada | inspeção do código |

**O que não foi verificado.** Ser compatível com a base canônica do projeto **não equivale** a ser equivalente aos arquivos que Juan usou. Não existe cópia nem hash dos arquivos dele no repositório. O texto do notebook relata 607 nódulos / 416 pacientes, o mesmo que os arquivos do projeto produzem, mas contagens iguais não garantem valores iguais: outra versão da base radiômica, com as mesmas linhas e valores diferentes de alguma feature, daria as mesmas contagens. Por isso a **hipótese H2 continua aberta** (Seção 11.2), assim como a H1, que trata do ambiente.

**Atualização na revisão:** a situação de H1 a H5 após o pacote e os esclarecimentos de Juan está na Seção 11.3.

### 11.2 Hipóteses não confirmadas — registro inicial

Esta tabela preserva as hipóteses como foram formuladas no registro inicial. A situação atual de cada uma está na Seção 11.3.

Nenhuma das hipóteses abaixo foi testada. Elas não são mutuamente exclusivas.

| # | Hipótese | Indício a favor | O que confirmaria ou descartaria |
|---|---|---|---|
| H1 | **Versões de bibliotecas diferentes no Colab** alteram a partição do `StratifiedGroupKFold` ou a numérica do liblinear | a seleção é sensível à partição (Seção 7.1); o notebook não registra versões | versões usadas no Colab; reexecução nessas versões |
| H2 | **Arquivos de entrada diferentes** no Colab (outra versão da base ou do alvo) | não há hash de referência dos arquivos de Juan | SHA-256 dos dois CSVs usados por Juan |
| H3 | **A lista escrita na conclusão do notebook e as estatísticas que a acompanham podem vir de execuções diferentes** | a reprodução bate exatamente com 16 / 5 / 0,263. Com a lista declarada e o RF desta reprodução, a interseção seria 6 e o Jaccard 6/18 = 0,333, e não 5 e 0,263; `Skewness` e `Coarseness` estão entre as 16 features estáveis do RF desta reprodução | os CSVs exportados pelo próprio notebook (`frequencia_selecao_features.csv`, `resumo_folds_selecao.csv`, `features_selecionadas_lasso.csv`) |
| H4 | **As faixas 35–39 e 7–13 do texto vêm de outra execução** (por exemplo, de uma versão anterior do notebook, que o próprio texto menciona) | os extremos 39 e 13 não foram observados | os mesmos CSVs exportados, ou o histórico de versões do notebook |

A hipótese H3 depende de o Random Forest da execução de Juan ser igual ao desta reprodução, o que também não está confirmado. Ela é um indício de inconsistência entre o texto e as estatísticas, **não uma prova**, e não permite atribuir a divergência a erro de quem conduziu a seleção.

### 11.3 Situação das hipóteses após o pacote e os esclarecimentos de Juan

Fontes:
- a comparação estática com o pacote enviado por Juan (`comparacao_pacote_juan.md`);
- os esclarecimentos de Juan (Seção 16).

Passam a ser distinguidas três execuções:

| Execução | O que se sabe |
|---|---|
| **Histórica** (Colab, conclusão do notebook) | só o texto da conclusão; segundo Juan, não há saídas nem logs, e as versões são desconhecidas |
| **Regeneração recente** (CSVs do pacote) | entradas idênticas às do projeto (verificado); versões declaradas por Juan; notebook declarado como o da regeneração recebido e idêntico ao do pacote (verificado); ligação entre esse código e os CSVs não comprovada |
| **Esta reprodução** | entradas, versões, código, saídas e hashes registrados (Seções 3 e 15) |

| # | Situação | Fundamento |
|---|---|---|
| H5 — os CSVs do pacote são uma regeneração recente, e não a saída histórica | **fato declarado e confirmado pelo responsável**; deixa de ser hipótese | declaração de Juan (Seção 16), compatível com o verificado na comparação. Data e ambiente da regeneração não foram verificados de forma independente. |
| H1 — versões de bibliotecas | **continua aberta** | entre a regeneração recente e esta reprodução, as entradas são idênticas, o notebook distribuído equivale ao auditado e a divergência começa na partição: a composição de pacientes difere nos 5 folds. As versões diferem muito (scikit-learn 1.8.0 × 1.3.2). O notebook declarado como o da regeneração é idêntico ao do pacote (Seção 16). Se essa declaração estiver correta, a diferença de código fica descartada, e restam o ambiente e as versões. Versões diferentes **podem** explicar os folds divergentes, mas a causa **não foi comprovada**, e a ligação entre o notebook e os CSVs também não. Para a execução histórica, as versões são desconhecidas. |
| H2 — arquivos de entrada diferentes | **descartada para a regeneração recente; não verificável para a execução histórica** | base, alvo e split do pacote são idênticos aos do projeto. O split é igual após normalizar só CRLF → LF e é byte a byte igual ao conteúdo versionado no Git. Não há registro dos arquivos lidos na execução histórica. |
| H3 — a lista e as estatísticas da conclusão vêm de execuções diferentes | **o indício original perdeu a base; para a execução histórica, não verificável** | na regeneração recente, a lista declarada e 16 / 5 / 0,263 saem de uma mesma execução, de forma consistente. A combinação do texto é, portanto, atingível por uma única execução. O indício usava o RF desta reprodução, que difere do RF do pacote (Seção 8). Sem saídas históricas, nada se conclui sobre a execução histórica. |
| H4 — as faixas 35–39 e 7–13 vêm de outra execução | **o indício original perdeu a base; para a execução histórica, não verificável** | a regeneração recente produz exatamente 35–39 e 7–13. Mesmo limite de H3. |

**Leitura conjunta.** Com as mesmas entradas e um procedimento equivalente:
- a regeneração recente devolve a lista declarada;
- esta reprodução devolve a lista com `Imc1` e `ClusterShade`.

Pelo que se verificou, **nenhuma das duas é incorreta**: são resultados do mesmo procedimento em ambientes diferentes, com partições diferentes. A lista resultante depende da partição e, possivelmente, do ambiente.

## 12. Fatos × hipóteses — resumo

| Afirmação | Situação |
|---|---|
| A reprodução usou exatamente os 417 nódulos / 291 pacientes do treino oficial | **verificado** |
| A reprodução devolve 8 features, 6 em comum com a lista declarada | **verificado** |
| `Skewness` e `Coarseness` são selecionadas em 1/5 folds; `Imc1` em 4/5 e `ClusterShade` em 3/5 | **verificado** |
| O resultado é determinístico neste ambiente | **verificado** |
| As estatísticas 16 / 5 / 0,263 do notebook são reproduzidas | **verificado** — só as contagens; os conjuntos diferem dos do pacote (Seção 8) |
| Entradas do pacote de Juan idênticas às do projeto | **verificado** (comparação, Seção 4) |
| Com essas entradas, a regeneração recente devolve a lista declarada, de forma internamente consistente | **verificado** nos CSVs do pacote (comparação, Seção 5) |
| Composição de pacientes dos folds diferente entre a regeneração recente e esta reprodução | **verificado** por contagens (comparação, Seção 6) |
| Os CSVs do pacote são regeneração recente, e não saída histórica | **declarado por Juan** (H5) |
| Não há saídas nem logs da execução histórica; versões históricas desconhecidas | **declarado por Juan** |
| A divergência decorre de versões de bibliotecas | hipótese (H1) — aberta |
| A divergência decorre de arquivos de entrada diferentes | hipótese (H2) — descartada para a regeneração recente; não verificável para a histórica |
| A lista declarada não corresponde à execução que gerou as estatísticas | hipótese (H3) — indício sem base; não verificável para a histórica |
| Houve erro na seleção original | **sem evidência** — não afirmado |

## 13. Impacto sobre M2, M3 e M4 e decisão do grupo

### 13.1 O que o protocolo prevê

A Seção 7.1, item 1, do protocolo define a regra para reprodução divergente: a lista declarada **permanece** como conjunto fixo; a divergência é registrada como limitação; a lista reproduzida entra **apenas como sensibilidade**; e a decisão de seguir adiante é registrada pelo grupo **antes da Etapa 3**. Essa regra ainda depende de confirmação do grupo (pendência P3, Seção 13 do protocolo).

### 13.2 Situação atual

- **A lista definitiva de features permanece pendente.** Até a decisão do grupo, nenhuma das duas listas está confirmada como o conjunto de M2, M3 e M4.
- **M2, M3 e M4 não devem ser avaliados na validação** (Etapa 3) antes dessa decisão.
- **A lista não pode ser escolhida com base em resultados de validação ou de teste.** Comparar o desempenho das duas listas em validação ou teste e adotar a melhor seria seleção com uso desses dados e invalidaria a avaliação. A decisão deve se apoiar apenas em evidência de procedimento — por exemplo, a resposta às hipóteses H1 a H4 — e em critérios definidos antes de qualquer resultado.
- **Recomendação de Juan — proposta, não decisão.** Juan recomenda usar esta reprodução controlada como principal evidência reproduzível para fundamentar a seleção do Cartão 4, ou seja, propor como lista fixa a lista reproduzida (Seção 7). Ela fica registrada como **proposta científica para discussão do grupo**, e **não como decisão aprovada**:
  - **Conflito com a regra atual do protocolo.** A Seção 7.1, item 1, do protocolo diz que a lista declarada **permanece** fixa em caso de divergência. Adotar a lista reproduzida exige decisão do grupo (P3) e **revisão formal do protocolo**, antes de qualquer avaliação de M2, M3 e M4.
  - **O que a sustenta.** Esta é a única das três execuções com entradas, versões, código, saídas e hashes registrados, e com determinismo verificado.
  - **O que a limita.** O determinismo vale só neste ambiente: com as mesmas entradas, a regeneração recente devolveu a lista declarada. `ClusterShade` atinge exatamente o limiar de 3/5. A preservação técnica da reprodução foi feita no commit `0ef4039` (Seção 17); ela atende à condição de reprodutibilidade, mas **não decide** a lista.
  - Discussão completa em `comparacao_pacote_juan.md`, Seção 14.1.
- **Composição dos modelos.** Com qualquer das duas listas, M3 tem 9 features e M4 tem 10, porque nenhuma das listas contém o diâmetro ou o volume e ambas contêm `Sphericity`, que já está em M1.
- **CV aninhada (Etapas 1 e 2).** Não depende da lista fixa, porque refaz a seleção em cada treino externo (protocolo, Seção 7.2). A reprodução fiel, porém, mostra que a seleção é sensível à partição, o que é esperado também na CV aninhada.

### 13.3 Pontos para a decisão do grupo

1. Confirmar ou revisar a regra da Seção 7.1 (pendência P3) **antes de qualquer avaliação de M2–M4 na validação**. Nessa decisão, considerar a recomendação de Juan (Seção 13.2) como proposta.
2. Pedir a Juan, se disponíveis — **situação após os esclarecimentos (Seção 16):**
   - as versões de scikit-learn, pandas e numpy usadas no Colab: **desconhecidas**, segundo Juan;
   - o SHA-256 de `base_radiomica_oficial_422.csv` e de `vies_malignidade_mascara_pequena.csv` usados por ele: os arquivos do pacote têm os **mesmos hashes** dos do projeto; os arquivos históricos não têm registro;
   - os três CSVs exportados pelo notebook na execução que gerou a conclusão: **não existem**; os CSVs do pacote são regeneração recente;
   - **novo:** o código ou notebook exato usado na regeneração recente. **Recebido em 12/10/2026** e idêntico ao do pacote; a proveniência exata dos CSVs continua sem comprovação (Seção 16).
3. Registrar a decisão, com data e justificativa, em revisão do protocolo.

## 14. Limitações e lacunas de evidência

1. **Os artefatos históricos das execuções A e B não estão versionados.** O script de reprodução original, o script de diagnóstico, a saída do diagnóstico, os dois relatórios JSON e os CSVs das execuções A e B ficaram em diretório temporário da sessão, fora do repositório. Este documento registra seus resultados e hashes (Seção 15), mas os arquivos em si podem não estar disponíveis para reexame. *Revisão:* a reprodução foi **preservada em novos artefatos versionados** (Seção 17). Os três CSVs versionados são byte a byte iguais aos históricos; o script e o JSON são novos, derivados dos originais.
2. **O notebook de Juan não está versionado** e a cópia auditada não contém saídas de execução. Não há como comparar fold a fold com a execução original. *Revisão:* Juan declarou que não existem saídas nem logs históricos, então a execução histórica **não pode ser verificada** com o material disponível.
3. **Uma única configuração de ambiente foi testada.** A sensibilidade a versões de bibliotecas (H1) não foi avaliada.
4. **O RF da execução de Juan é desconhecido**, o que limita o alcance da comparação de Jaccard (H3). *Revisão:* o RF da **regeneração recente** é conhecido pelo pacote (Seção 8); o da execução histórica continua desconhecido.
5. **A estabilidade usa 5 folds de uma única partição**, como no notebook. A sensibilidade observada sugere que outra partição poderia mudar as features próximas do limiar — em especial `ClusterShade`, que atinge exatamente 3/5. Isso não foi testado aqui, porque sairia do escopo da reprodução fiel.
6. **Estabilidade entre anotadores** das features, de qualquer das listas, continua não avaliada (protocolo, Seção 11.3).
7. **Filtros ajustados fora da CV interna** (Seção 2.4). Limitação metodológica do procedimento original, replicada aqui de propósito. Não deve ser reutilizada na CV aninhada corrigida.

## 15. Rastreabilidade das execuções históricas

Artefatos **históricos** das execuções A e B, gerados em 10/10/2026 no diretório temporário da sessão, fora do repositório. Eles **não são versionados**. A versão preservada e versionada da reprodução está na Seção 17.

| Artefato | SHA-256 |
|---|---|
| Script de reprodução | `066fee8da95dc6b0f9d41b485c036d6d64ed42c2f4a072bb7cc9dc67c859c66c` |
| Script de diagnóstico de convergência | `fa83f5b56a154aeabb9e2d1e32b05288973960b239eb06f4747fc5318b258b4d` |
| Saída do diagnóstico | `33b48f52b9164f9275438719111b79f40cead0884f99bbdc7e9c6ff5e880e652` |
| Execução A — relatório JSON | `0b9560a4594e72896d596c6bce95b4396fe9e1e4795bc072a6df9ddf2df0063e` |
| Execução B — relatório JSON | `f1f313d0dc9bf318ad7e73a584486e9b73beec9c5cb4585fee67bb4609cd3d17` |
| A e B — `frequencia_selecao_features.csv` | `8e85ee1770f9a53b896941b22b5ebd660c2d623100a3df096d552dbd4ff38ed2` |
| A e B — `resumo_folds_selecao.csv` | `96c866b19319737ad3221f4011b7ec6f695cd729b2ed4128574313a7a15e27bf` |
| A e B — `features_selecionadas_lasso.csv` | `d45c68824548443be72443c8e2acdf4605cd3952e60906b861bacf82f073db34` |

Nenhum desses artefatos contém identificadores de paciente ou de nódulo; todos trazem apenas nomes de features, contagens e coeficientes agregados.

## 16. Esclarecimentos de Juan

Recebidos em 11/10/2026, depois do registro inicial. São **declarações** do responsável pela seleção original. A conferência independente de cada uma está em `comparacao_pacote_juan.md`, Seção 15.

| # | Declaração de Juan | Efeito nesta auditoria |
|---|---|---|
| 1 | Os CSVs do ZIP foram regenerados recentemente e não são os arquivos históricos da execução original do Colab | H5 passa a fato declarado; o pacote é evidência sobre a regeneração recente, não sobre a execução histórica (Seção 11.3) |
| 2 | A regeneração usou Python 3.13.5, NumPy 2.3.5, pandas 2.2.3 e scikit-learn 1.8.0 | ambiente diferente do desta reprodução (Seção 3.1); base de H1 (Seção 11.3) |
| 3 | As versões da execução histórica no Colab são desconhecidas | H1 não pode ser testada para a execução histórica |
| 4 | O `features_selecionadas_lasso.csv` de quatro colunas recebeu processamento adicional a partir do CSV de frequências; não é a exportação direta do notebook | compatível com a diferença de formato verificada |
| 5 | Não há saídas nem logs históricos que permitam verificar a execução original | H2, H3 e H4 não são verificáveis para a execução histórica (Seção 11.3) |
| 6 | Recomenda esta reprodução controlada como principal evidência reproduzível para fundamentar a seleção do Cartão 4 | registrada como **proposta**, não como decisão (Seção 13.2) |
| 7 | Foi solicitado a enviar o código ou notebook exato usado na regeneração recente | **recebido em 12/10/2026** — ver abaixo |

**Notebook da regeneração recente.** Em 12/10/2026, Juan enviou `S4_Selecao_Features_Regeneracao_Recente.ipynb`, com SHA-256 `3a5e8e63c4b47ff8144ace04d63e6b4e7cb16e38e41ba045701d7eaa4a7f9835`.
- É **idêntico byte a byte** ao notebook do pacote (**verificado** por hash e por `cmp`).
- Tem 7 células de código, sem saídas e sem execução registrada.
- Em relação à cópia auditada (Seção 3.3), a lógica de código é a mesma; muda só um comentário.

O recebimento resolve a pendência do envio, mas **não comprova** que esse código gerou os CSVs regenerados. A proveniência exata das saídas continua em aberto. Detalhes em `comparacao_pacote_juan.md`, Seção 15.1.

## 17. Preservação versionada da reprodução

Commit `0ef403998836c3d891441b9626bdbefae5fcbff3` (`0ef4039`), de 12/10/2026, na branch `feat/s4-incremental-radiomics`.

### 17.1 Artefatos versionados

| Caminho | SHA-256 | Relação com os artefatos históricos (Seção 15) |
|---|---|---|
| [`scripts/sprint4/reproduce_juan_feature_selection.py`](../../../scripts/sprint4/reproduce_juan_feature_selection.py) | `e90180ac78f8d492cda4c4fec1fd7cd093aa1c687f20edd1be9cc31f0934cdc4` | **novo**; derivado do script de reprodução (`066fee8d…`) e do script de diagnóstico (`fa83f5b5…`) |
| [`reproducao_juan/frequencia_selecao_features.csv`](reproducao_juan/frequencia_selecao_features.csv) | `8e85ee1770f9a53b896941b22b5ebd660c2d623100a3df096d552dbd4ff38ed2` | **byte a byte igual** ao CSV das execuções A e B |
| [`reproducao_juan/resumo_folds_selecao.csv`](reproducao_juan/resumo_folds_selecao.csv) | `96c866b19319737ad3221f4011b7ec6f695cd729b2ed4128574313a7a15e27bf` | **byte a byte igual** ao CSV das execuções A e B |
| [`reproducao_juan/features_selecionadas_lasso.csv`](reproducao_juan/features_selecionadas_lasso.csv) | `d45c68824548443be72443c8e2acdf4605cd3952e60906b861bacf82f073db34` | **byte a byte igual** ao CSV das execuções A e B |
| [`reproducao_juan/relatorio_reproducao.json`](reproducao_juan/relatorio_reproducao.json) | `c141fd6fa5dbaf5c8572866ce9515506a6021e251bfdec97dddbdc5ff714b5ea` | **novo**; substitui os dois JSONs históricos e a saída do diagnóstico, sem caminhos absolutos |

Os caminhos `reproducao_juan/...` são relativos a este diretório: `reports/sprint4/feature_selection/reproducao_juan/`.

### 17.2 O que mudou no script em relação ao original

- **Núcleo da seleção (blocos 1–5) e chamadas `to_csv`:** idênticos aos do script original. Conferido por diff textual; as únicas linhas novas apenas registram as features pós-filtro de cada fold, sem alterar nenhum cálculo.
- **Portabilidade e rastreabilidade:**
  - `argparse`, com a raiz do repositório derivada do próprio arquivo;
  - notebook opcional, usado só para registrar o hash;
  - chaves de arquivo relativas, sem caminhos absolutos;
  - registro do estado do Git e do SHA-256 do próprio script;
  - travas contra gravar em `reports/sprint3/` e `data/` e contra sobrescrever saídas.
- **Diagnóstico de convergência:** foi incorporado e roda **depois** da seleção, com captura de warnings própria. Ele exige que a partição e os filtros sejam idênticos aos da seleção.

### 17.3 Execução que gerou os artefatos versionados

- **Uma única execução**, com `n_jobs=1`, sobre os 417 nódulos do treino oficial, no ambiente da Seção 3.1. A seleção levou 109,3 s e o diagnóstico 103,6 s.
- As saídas foram gravadas primeiro em diretório temporário. Depois da validação, foram copiadas sem alteração de bytes para `reproducao_juan/`. Por isso o campo `saida.diretorio` do JSON registra `<externo ao repositório>`.
- **Validação:**
  - os três CSVs coincidem com os hashes históricos;
  - os blocos de checagens, resumo e detalhes por fold, resultado, comparação com as declaradas e afirmações do notebook são idênticos aos do JSON histórico da execução B;
  - houve 2 `ConvergenceWarning` na seleção, como em B;
  - o diagnóstico reproduz a Seção 10: 260 ajustes, 2 não convergidos com `C = 100` e refits convergidos;
  - os coeficientes do refit são idênticos aos da seleção nos 5 folds;
  - as entradas ficaram intactas.

### 17.4 Estado do Git registrado no JSON

O JSON foi gerado **antes** do commit `0ef4039` e registra o estado daquele momento:

| Campo | Valor registrado |
|---|---|
| `commit_head` | `a70db2ef91b6291ddb79c7d7e46471197691fa9e` (base da execução) |
| `branch` | `feat/s4-incremental-radiomics` |
| `arvore_limpa` | `false` |
| `status_porcelain` | só o script, ainda não rastreado |
| `script_versionado_no_head` | `false` |
| `sha256_script` | `e90180ac…`, igual ao script versionado em `0ef4039` |

Portanto, o vínculo entre o JSON e o código se faz pelo `sha256_script`, e não pelo `commit_head`. O commit `0ef4039` contém exatamente o script com esse hash.

### 17.5 Fim de linha

Os CSVs e o JSON foram gerados com CRLF, padrão do pandas e do Python no Windows. Os hashes acima referem-se a esses bytes.

Duas regras no `.gitattributes`, versionadas no mesmo commit, preservam os bytes:
- `-text` para os CSVs e o JSON de `reproducao_juan/`, sem conversão de fim de linha;
- `text eol=lf` para o script.

Os bytes no commit são iguais aos registrados.

Uma reexecução em outro sistema pode gravar os CSVs com LF. Para comparar, o JSON registra também o SHA-256 após normalizar somente CRLF → LF:
- frequências: `5a2f6af164de29862972f0dc7cfc67f394183ce29092f3c826c7b6c9b3609ab5`;
- resumo dos folds: `6f3ba35091dddd1877a6ce4860e47149469527534760c84f72b018a3485b22ca`;
- lista final: `8f81208d07e2f894067bba3ab50c8fc4bb2679dfe8f77b291b905f2a8c792e33`.

### 17.6 O que a preservação não muda

- **Reprodução histórica ≠ pipeline corrigido.** O script versionado preserva o procedimento **original**, inclusive os filtros ajustados fora da CV interna (Seção 2.4). Ele **não** é o pipeline da CV aninhada corrigida (protocolo, Seção 5.4), não deve ser reutilizado como tal, e seus resultados não substituem os das Etapas 1 e 2.
- **Pendências científicas mantidas.** A divergência entre a lista declarada e a lista reproduzida continua sem decisão. A pendência P3 — regra para divergência e lista fixa de M2, M3 e M4 — segue com o grupo, assim como a revisão formal do protocolo (Seção 13). A preservação técnica não escolhe lista.

---

_Este documento registra a auditoria da reprodução da seleção de features do Cartão 4. Ele não altera o protocolo nem define a lista de features; qualquer decisão sobre a lista deve ser registrada em revisão do protocolo._
