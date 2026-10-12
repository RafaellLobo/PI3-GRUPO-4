# Estado do Cartão 4 — Checkpoint de Continuidade

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Card relacionado:** S4 — Cartão 4: valor incremental da radiômica além do tamanho do nódulo
**Protocolo:** [`protocolo_valor_incremental.md`](protocolo_valor_incremental.md) — **proposta aguardando revisão do grupo e do professor; não é protocolo congelado**
**Branch:** `feat/s4-incremental-radiomics`
**Registro:** 11/10/2026 · **HEAD na revisão mais recente:** `0ef4039`
**Revisão:** 11/10/2026 — comparação com o pacote de Juan registrada e esclarecimentos de Juan incorporados (Seção 2)
**Revisão:** 12/10/2026 — notebook da regeneração recente recebido (Seção 2.3); limitação metodológica dos filtros registrada (Seções 2.6 e 4)
**Revisão:** 12/10/2026 — reprodução preservada e versionada no commit `0ef4039` (Seções 1 e 2.1)

---

## Finalidade

Este documento registra onde o Cartão 4 parou, para que o trabalho possa ser retomado sem reconstruir o histórico. Ele **não** altera o protocolo, **não** define a lista de features e **não** registra nenhuma decisão do grupo.

O alvo é a suspeita radiológica de malignidade (mediana de escores de radiologistas), não malignidade histológica confirmada.

Neste documento:
- **"verificado"** designa o que foi medido e está registrado em artefato do repositório ou em relatório de auditoria;
- **"declarado por Juan"** designa informação dada pelo responsável pela seleção original, sem verificação independente;
- **"hipótese"** designa explicação possível não testada.

---

## 1. Concluído

| Item | Commit | Artefatos | Situação |
|---|---|---|---|
| Proposta de protocolo | `9e9b3bd` | `docs/sprint4/protocolo_valor_incremental.md` | versionado; continua como **proposta** |
| Reprodução e verificação de M0 e M1 (Etapa 0, invariante do M1) | `7f1a0f6` | `scripts/sprint4/validate_reference_models.py`; `reports/sprint4/reference_models/validacao_modelos_referencia.json` | versionado |
| Auditoria da seleção de Juan, comparação com o pacote e este checkpoint | `a70db2e` | `reports/sprint4/feature_selection/auditoria_reproducao_juan.md`; `reports/sprint4/feature_selection/comparacao_pacote_juan.md`; `docs/sprint4/estado_cartao4.md` | versionado; revisado depois |
| Preservação da reprodução da seleção (script, três CSVs, JSON de execução) e regras de fim de linha | `0ef4039` | `scripts/sprint4/reproduce_juan_feature_selection.py`; `reports/sprint4/feature_selection/reproducao_juan/`; `.gitattributes` | versionado; CSVs byte a byte iguais aos históricos (Seção 2.1) |

Resultado da verificação de M0 e M1 — **verificado**:

- **23 de 23 invariantes estruturais aprovadas**;
- **M1 do baseline reproduzido** dentro das tolerâncias explícitas: intercepto e média do scaler idênticos bit a bit, coeficientes com diferença máxima de 3,3 × 10⁻¹⁶;
- ajuste somente no treino oficial (417 nódulos / 291 pacientes); nenhuma predição e nenhuma métrica calculadas; as entradas não foram alteradas;
- o JSON registra o commit `9e9b3bd` como base da execução, porque o script ainda não estava versionado quando rodou.

## 2. Auditoria da seleção de features de Juan

### 2.1 Nossa reprodução — verificado

- A reprodução fiel do procedimento do notebook, sobre o treino oficial e no ambiente do projeto, devolveu **8 features, das quais 6 coincidem com as 8 declaradas**.
- Coincidem: `shape_Sphericity`, `shape_SurfaceVolumeRatio`, `shape_Elongation`, `firstorder_90Percentile`, `firstorder_RootMeanSquared` e `glcm_Imc2`.
- Divergem:
  - a reprodução inclui `glcm_Imc1` e `glcm_ClusterShade`;
  - a lista declarada inclui `firstorder_Skewness` e `ngtdm_Coarseness`.
- Resultado determinístico no ambiente testado.
- As **contagens** 16 / 5 / 0,263 do Random Forest e do índice de Jaccard coincidem com as do notebook. Os conjuntos de features por trás delas diferem dos do pacote de Juan (Seção 2.2).
- Relatório: [`auditoria_reproducao_juan.md`](../../reports/sprint4/feature_selection/auditoria_reproducao_juan.md) — revisado e versionado (Seção 1).
- **Artefatos históricos (não versionados).** Os scripts originais, a saída do diagnóstico, os JSONs e os CSVs das execuções A e B, de 10/10/2026, continuam só no diretório temporário da sessão. Seus hashes estão na Seção 15 da auditoria.
- **Artefatos versionados no commit `0ef4039`** (auditoria, Seção 17):

| Caminho | SHA-256 |
|---|---|
| [`scripts/sprint4/reproduce_juan_feature_selection.py`](../../scripts/sprint4/reproduce_juan_feature_selection.py) | `e90180ac78f8d492cda4c4fec1fd7cd093aa1c687f20edd1be9cc31f0934cdc4` |
| [`reports/sprint4/feature_selection/reproducao_juan/`](../../reports/sprint4/feature_selection/reproducao_juan/) `frequencia_selecao_features.csv` | `8e85ee1770f9a53b896941b22b5ebd660c2d623100a3df096d552dbd4ff38ed2` (= histórico) |
| `.../reproducao_juan/resumo_folds_selecao.csv` | `96c866b19319737ad3221f4011b7ec6f695cd729b2ed4128574313a7a15e27bf` (= histórico) |
| `.../reproducao_juan/features_selecionadas_lasso.csv` | `d45c68824548443be72443c8e2acdf4605cd3952e60906b861bacf82f073db34` (= histórico) |
| `.../reproducao_juan/relatorio_reproducao.json` | `c141fd6fa5dbaf5c8572866ce9515506a6021e251bfdec97dddbdc5ff714b5ea` |

- Os artefatos versionados vieram de **uma única execução**, com `n_jobs=1`, gravada em diretório temporário e copiada sem alteração de bytes depois da validação.
- O JSON foi gerado **antes** do commit. Ele registra como base `a70db2e`, com o script ainda não rastreado. O vínculo com o código se faz pelo `sha256_script` (`e90180ac…`), igual ao script versionado.
- O `.gitattributes` preserva os bytes: CSVs e JSON com `-text`; script com `eol=lf`.

### 2.2 Pacote enviado por Juan — comparação verificada

Juan enviou o arquivo `pacote_auditoria_selecao_features.zip` (SHA-256 `a88b2f0ba0e87410c2ec780877e96da8604810e7d28be9d929fd69321c01dda7`, 863 972 bytes). Relatório da comparação: `reports/sprint4/feature_selection/comparacao_pacote_juan.md` — versionado (Seção 1). O ZIP **não** é versionado.

Verificado na comparação estática, sem executar código do pacote:

- base radiômica e alvo do pacote têm **bytes idênticos** aos do projeto;
- o split do pacote é **idêntico após normalizar só CRLF → LF** e igual byte a byte ao conteúdo versionado no Git;
- os CSVs do pacote são **internamente consistentes** e produzem a lista declarada, 16 / 5 / 0,263 e as faixas 35–39 e 7–13;
- os pacientes por fold são iguais em número, mas os nódulos por fold diferem nos 5 folds, logo **a composição dos folds difere** entre o pacote e a nossa reprodução;
- o código do notebook do pacote equivale ao da cópia auditada; só um comentário muda;
- o `features_selecionadas_lasso.csv` do pacote não tem o formato da exportação do notebook.

### 2.3 Esclarecimentos de Juan — declarado por Juan

1. Os CSVs do ZIP foram **regenerados recentemente**; não são os arquivos históricos da execução original do Colab.
2. A regeneração usou Python 3.13.5, NumPy 2.3.5, pandas 2.2.3 e scikit-learn 1.8.0.
3. As versões da execução histórica no Colab são **desconhecidas**.
4. O CSV de quatro colunas recebeu processamento adicional a partir do CSV de frequências; não é a exportação direta do notebook.
5. **Não existem saídas nem logs históricos** que permitam verificar a execução original.
6. Juan recomenda a nossa reprodução controlada como principal evidência reproduzível para fundamentar a seleção do Cartão 4.
7. Foi pedido a Juan o código ou notebook exato da regeneração recente — **recebido em 12/10/2026.**

As declarações 1, 2, 4 e 5 são compatíveis com o que a comparação verificou. Nenhuma foi verificada de forma independente quanto à data ou ao ambiente.

**Notebook da regeneração recente — verificado.** O arquivo `S4_Selecao_Features_Regeneracao_Recente.ipynb` (SHA-256 `3a5e8e63c4b47ff8144ace04d63e6b4e7cb16e38e41ba045701d7eaa4a7f9835`) é idêntico byte a byte ao notebook do pacote. Ele tem 7 células de código, sem saídas e sem execução registrada. A lógica de código é a mesma da cópia auditada; muda só um comentário.

- **Resolvido:** o recebimento do código.
- **Continua aberto:** a comprovação de que esse código gerou os CSVs regenerados, isto é, a proveniência exata das saídas. Também não foi recebido o script do processamento adicional do CSV de quatro colunas.

### 2.4 Situação das hipóteses da auditoria

| Hipótese | Situação |
|---|---|
| H5 — os CSVs do pacote são regeneração recente | **fato declarado e confirmado pelo responsável** |
| H1 — versões de bibliotecas diferentes | **aberta.** Pode explicar os folds divergentes entre a regeneração recente e a nossa reprodução, mas não foi comprovada. Com o notebook recebido idêntico, a diferença de código fica descartada **se** a declaração de Juan estiver correta. |
| H2 — arquivos de entrada diferentes | descartada para a regeneração recente; **não verificável** para a execução histórica |
| H3 e H4 — conclusão e estatísticas de execuções diferentes | o indício original perdeu a base, porque a regeneração recente produz lista e estatísticas de forma consistente; **não verificáveis** para a execução histórica |

Com as mesmas entradas:
- a regeneração recente devolve a lista declarada;
- a nossa reprodução devolve a lista com `Imc1` e `ClusterShade`.

Pelo que se verificou, nenhuma das duas é incorreta. Nada disso permite afirmar que houve erro na seleção original.

### 2.5 Recomendação de Juan — proposta, não decisão

A recomendação de usar a lista reproduzida no nosso ambiente como lista fixa está registrada como **proposta científica para discussão do grupo, e não como decisão aprovada**. Ela conflita com a regra atual do protocolo (Seção 7.1, item 1: a lista declarada permanece fixa em caso de divergência). Adotá-la exige:
- decisão do grupo (P3);
- revisão formal do protocolo;
- preservação dos artefatos da reprodução no repositório — **concluída no commit `0ef4039`** (Seção 2.1). Essa preservação é técnica e não decide a lista.

As duas primeiras precisam acontecer antes de qualquer avaliação de M2, M3 e M4. Discussão completa: `comparacao_pacote_juan.md`, Seção 14.1.

### 2.6 Limitação metodológica do procedimento original — verificado no código

No procedimento original, os filtros de variância e de correlação são ajustados no treino inteiro de cada fold de estabilidade, **antes** e **fora** da `GridSearchCV` interna. Assim, os folds de validação interna participam da definição das features avaliadas neles.
- Os filtros não usam o alvo.
- A escolha de `C` não é estritamente independente dos filtros.

A limitação vale para as três execuções e não explica a divergência entre elas. Ela **não deve ser reutilizada na CV aninhada corrigida** (Seção 4; protocolo, Seção 5.4).

Detalhes:
- auditoria, Seção 2.4;
- comparação, Seção 7.1.

## 3. Decisões pendentes

Nenhuma das decisões abaixo foi tomada. **Nada aqui presume aprovação.**

| # | Pendência | Origem | Prazo |
|---|---|---|---|
| P3 | Confirmação, pelo grupo, da regra para divergência na reprodução fiel e, com ela, da **lista fixa de features** de M2, M3 e M4. Inclui avaliar a recomendação de Juan (Seção 2.5) e, se aceita, a revisão formal do protocolo. | protocolo, Seção 7.1, item 1, e Seção 13 | antes da Etapa 3 |
| — | **Proveniência exata dos CSVs regenerados.** O notebook foi recebido em 12/10/2026, mas não tem saídas; a ligação entre código e saídas não está comprovada, e o script do CSV de quatro colunas não foi recebido. | Seção 2.3 | a considerar na decisão sobre P3; não há meio previsto de comprovação |
| — | Registro da composição dos folds da regeneração recente (por exemplo, hash por fold), só para comparação interna — ainda não solicitado formalmente | comparação, Seção 12 | opcional; apoiaria a análise de H1 |
| P1 | Número de permutações K e esquema de permutação | protocolo, Seção 13 | antes da Etapa 1 |
| P2 | Parâmetros do spline | protocolo, Seção 13 | antes da Etapa 1 |
| P4 | Divergência entre M1 = 0,9778 (Experimento 13) e M1 = 0,9762 (baseline canônico) | protocolo, Seção 13 | antes da Etapa 5 |
| P5 | Versionamento dos registros dos Experimentos 10 e 13 e do notebook de Juan com suas saídas. Segundo Juan, não existem saídas históricas, e o notebook da regeneração recente foi recebido sem saídas (Seção 2.3). Como tratar essa parte de P5 depende de decisão do grupo. | protocolo, Seção 13 | antes da Etapa 4 |
| — | Revisão do protocolo pelo grupo e pelo professor. Na revisão formal, as Seções 5.2 e 11.3 (item 8) do protocolo, que ainda dizem que a reprodução independente não está documentada, devem passar a remeter à auditoria e à comparação. | protocolo, cabeçalho | antes do congelamento (Etapa 4) |

A lista de features **não pode** ser decidida com base em desempenho em validação ou teste. A decisão deve se apoiar em evidência de procedimento e em critérios definidos antes de qualquer resultado.

A divergência entre a lista declarada e a lista reproduzida continua sendo **pendência científica**. A preservação técnica da reprodução (commit `0ef4039`) não a resolve.

## 4. Próxima implementação independente

**Preparação e validação estrutural dos folds da CV aninhada** (estrutura das Etapas 1 e 2 do protocolo, Seção 7.2).

Ela não depende da lista fixa de features, porque a CV aninhada refaz a seleção em cada treino externo. Por isso pode avançar enquanto P3 está pendente.

**Escopo:**

- **dados:** somente o treino oficial (291 pacientes / 417 nódulos);
- **folds externos:** `StratifiedGroupKFold(5)`, grupos = `patient_id`, `random_state = 42 + r`, com r = 0 para o piloto e r = 0…9 para a CV completa;
- **folds internos:** dentro de cada treino externo, a CV de estabilidade de 5 folds e a CV interna de 4 folds, ambas agrupadas por paciente, com as seeds da Seção 12 do protocolo;
- **requisito para as etapas seguintes:** a estrutura de folds deve permitir que os filtros de variância e de correlação sejam ajustados **em cada treino interno**, dentro do pipeline avaliado pela CV interna (protocolo, Seção 5.4). O ajuste dos filtros antes e fora da `GridSearchCV`, como no procedimento original, **não deve ser reutilizado** (Seção 2.6).
- **separação de código:** `scripts/sprint4/reproduce_juan_feature_selection.py` preserva o procedimento **histórico** e não é base para o pipeline corrigido. A CV aninhada deve ser implementada à parte, sem reaproveitar o laço de seleção desse script.

**Verificações estruturais, sem ajuste de modelo:**

- cada nódulo do treino aparece exatamente uma vez como avaliação em cada repetição;
- nenhum paciente aparece ao mesmo tempo no treino e na avaliação de um fold, em qualquer nível;
- nenhuma linha de validação ou de teste é lida;
- a composição de classes é descrita por fold;
- as partições são reprodutíveis entre execuções.

**Fora do escopo:** seleção de features, ajuste de M0 a M4, predições e métricas.

Saídas apenas com contagens e hashes, sem identificadores de paciente ou nódulo.

A implementação começa só com autorização explícita.

## 5. Restrições em vigor

- **Não escolher features por desempenho**, em nenhum conjunto.
- **Não avaliar M2, M3 nem M4**, em CV, validação ou teste, sem autorização.
- **Não usar linhas de validação ou de teste na CV aninhada**, nem em qualquer ajuste, filtro ou seleção das Etapas 0 a 2.
- **Não modificar o protocolo sem decisão documentada**, registrada em revisão com data e motivo.
- Não alterar coorte, alvo, split, features nem os arquivos em `data/`.
- Não executar `run_baseline.py`, porque ele sobrescreve os relatórios da Sprint 3.
- Não copiar dados individuais de paciente ou nódulo para documentos ou relatórios.
- Commit e push só com autorização explícita.

## 6. Estado do Git no registro

| Situação | Arquivo |
|---|---|
| Versionado em `9e9b3bd` | `docs/sprint4/protocolo_valor_incremental.md` |
| Versionado em `7f1a0f6` | `scripts/sprint4/validate_reference_models.py`; `reports/sprint4/reference_models/validacao_modelos_referencia.json` |
| Versionado em `a70db2e` | `reports/sprint4/feature_selection/auditoria_reproducao_juan.md`; `reports/sprint4/feature_selection/comparacao_pacote_juan.md`; `docs/sprint4/estado_cartao4.md` |
| Versionado em `0ef4039` | `scripts/sprint4/reproduce_juan_feature_selection.py`; os quatro arquivos de `reports/sprint4/feature_selection/reproducao_juan/`; `.gitattributes` (regras de fim de linha) |
| Revisado, ainda não versionado | os três documentos de `a70db2e`, nesta revisão |
| Fora do repositório | artefatos históricos das execuções A e B e scripts auxiliares da comparação (diretório temporário); `pacote_auditoria_selecao_features.zip`; notebooks de Juan (cópia auditada e notebook da regeneração recente) |

Nada foi enviado ao remoto nesta branch.

---

_Checkpoint de continuidade do Cartão 4. Atualize a cada etapa concluída ou decisão registrada; decisões em si pertencem a revisões do protocolo._
