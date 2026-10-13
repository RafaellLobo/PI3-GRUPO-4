# Estado do Cartão 4 — Checkpoint de Continuidade

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Card relacionado:** S4 — Cartão 4: valor incremental da radiômica além do tamanho do nódulo
**Protocolo:** [`protocolo_valor_incremental.md`](protocolo_valor_incremental.md) — **proposta aguardando revisão do grupo e do professor; não é protocolo congelado**
**Branch:** `feat/s4-incremental-radiomics`
**Registro:** 11/10/2026 · **HEAD na revisão mais recente:** `bda673d`
**Revisão:** 11/10/2026 — comparação com o pacote de Juan registrada e esclarecimentos de Juan incorporados (Seção 2)
**Revisão:** 12/10/2026 — notebook da regeneração recente recebido (Seção 2.3); limitação metodológica dos filtros registrada (Seções 2.6 e 4; a antiga Seção 4, "Próxima implementação independente", foi concluída e, na revisão de 13/10/2026, substituída pelas Seções 1.1 e 7)
**Revisão:** 12/10/2026 — reprodução preservada e versionada no commit `0ef4039` (Seções 1 e 2.1)
**Revisão:** 13/10/2026 — infraestrutura experimental, piloto exploratório e CV repetida exploratória 10 × 5 registrados (Seções 1, 3 e 4); pendências por etapa e matriz da Definition of Done (Seções 5 e 6)

---

## Finalidade

Este documento registra onde o Cartão 4 parou, para que o trabalho possa ser retomado sem reconstruir o histórico. Ele **não** altera o protocolo, **não** define a lista de features e **não** registra nenhuma decisão do grupo.

O alvo é a suspeita radiológica de malignidade (mediana de escores de radiologistas), não malignidade histológica confirmada.

Neste documento:
- **"verificado"** designa o que foi medido e está registrado em artefato do repositório ou em relatório de auditoria;
- **"declarado por Juan"** designa informação dada pelo responsável pela seleção original, sem verificação independente;
- **"hipótese"** designa explicação possível não testada;
- **"exploratório"** designa resultado de atividade própria, fora das Etapas 1–5 do protocolo, executada antes das decisões P1–P5 e nunca elegível como resultado definitivo.

## Situação resumida

- **Desenvolvimento exploratório: concluído.** A infraestrutura experimental, o piloto exploratório (1 × 5) e a CV repetida exploratória (10 × 5) foram implementados, executados só com o treino oficial, auditados e versionados.
- **Cartão científico: em andamento.** Nenhuma etapa oficial do protocolo (1 a 5) foi executada. Não há intervalo de confiança, comparação estatística entre modelos, análise com spline, controle por permutação nem conclusão sobre o valor incremental além do diâmetro. Essas entregas dependem de P1–P3 e da revisão do protocolo (Seção 5).

---

## 1. Concluído

| Item | Commit | Artefatos | Situação |
|---|---|---|---|
| Proposta de protocolo | `9e9b3bd` | `docs/sprint4/protocolo_valor_incremental.md` | versionado; continua como **proposta** |
| Reprodução e verificação de M0 e M1 (Etapa 0, invariante do M1) | `7f1a0f6` | `scripts/sprint4/validate_reference_models.py`; `reports/sprint4/reference_models/validacao_modelos_referencia.json` | versionado |
| Auditoria da seleção de Juan, comparação com o pacote e este checkpoint | `a70db2e` | `reports/sprint4/feature_selection/auditoria_reproducao_juan.md`; `reports/sprint4/feature_selection/comparacao_pacote_juan.md`; `docs/sprint4/estado_cartao4.md` | versionado; revisado depois |
| Preservação da reprodução da seleção (script, três CSVs, JSON de execução) e regras de fim de linha | `0ef4039` | `scripts/sprint4/reproduce_juan_feature_selection.py`; `reports/sprint4/feature_selection/reproducao_juan/`; `.gitattributes` | versionado; CSVs byte a byte iguais aos históricos (Seção 2.1) |
| Rastreabilidade da reprodução versionada | `cf2bae0` | os três documentos de `a70db2e` | versionado |
| Infraestrutura experimental (Seção 1.1) | `6214489` … `5e62228` | `src/incremental/` e `tests/incremental/` | versionado |
| Piloto exploratório: atividade, orquestrador e resultados | `7cf2694`, `6a87e88`, `13dc7c9` | `src/incremental/piloto.py`; `reports/sprint4/piloto_exploratorio/` | versionado (Seção 3.2) |
| CV repetida exploratória: atividade | `165584c` | `src/incremental/config.py`; `src/incremental/dados.py` | versionado |
| Refatoração do piloto para reutilização por atividade e repetição | `03a922d` | `src/incremental/piloto.py` | versionado; relatório e payloads sintéticos idênticos byte a byte aos de antes |
| CV repetida exploratória 10 × 5: orquestrador | `9eef000` | `src/incremental/cv_repetida.py`; `tests/incremental/test_cv_repetida.py` | versionado |
| CV repetida exploratória 10 × 5: resultados auditados | `bda673d` | `reports/sprint4/cv_repetida_exploratoria/`; regra de fim de linha em `.gitattributes` | versionado (Seção 3.3) |

Resultado da verificação de M0 e M1 — **verificado**:

- **23 de 23 invariantes estruturais aprovadas**;
- **M1 do baseline reproduzido** dentro das tolerâncias explícitas: intercepto e média do scaler idênticos bit a bit, coeficientes com diferença máxima de 3,3 × 10⁻¹⁶;
- ajuste somente no treino oficial (417 nódulos / 291 pacientes); nenhuma predição e nenhuma métrica calculadas; as entradas não foram alteradas;
- o JSON registra o commit `9e9b3bd` como base da execução, porque o script ainda não estava versionado quando rodou.

### 1.1 Infraestrutura experimental — verificado

Implementada à parte de `scripts/sprint4/reproduce_juan_feature_selection.py`, sem reaproveitar o laço de seleção histórico (Seção 2.6).

| Módulo | Commit | Função |
|---|---|---|
| `config.py` | `6214489` | especificações de modelos, seleção, CV e bootstrap; decisões P1–P5 e revisão como **pendentes**; hash determinístico da configuração |
| `guardas.py` | `bf13cd0` | autorização por atividade; decisões pendentes bloqueiam a execução; M2, M3 e M4 exigem P3 |
| `dados.py` | `4dc93c8` | porta única de leitura; partições permitidas por atividade; SHA-256 fixados das entradas; validação e teste nunca lidos nas atividades de treino |
| `rastreabilidade.py` | `7c66415` | manifestos com hashes de código, entradas e saídas; varredura de dados individuais; atividades somente exploratórias nunca definitivas |
| `checkpoints.py` | `df7276b` | checkpoint por unidade (atividade, repetição, fold, modelo); retomada só com identidade idêntica; nada sobrescrito ou apagado |
| `folds.py` | `ddee3ad` | folds externos, de estabilidade e internos agrupados por paciente, com as seeds da Seção 12 do protocolo |
| `transformadores.py`, `selecao.py` | `224db3b`, `65a5385` | procedimento ajustado da Seção 5.4: filtros → scaler → L1 ajustados **em cada treino interno**; S_k = features em ≥ 3 de 5 folds de estabilidade |
| `modelos.py` | `5e62228` | M0, M1, M2v, M3v e M4v; M2–M4 de lista fixa bloqueados sem P3 |

Suíte de testes: 428 testes aprovados no commit `9eef000`, só com dados sintéticos.

---

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

A limitação vale para as três execuções e não explica a divergência entre elas. Ela **não foi reutilizada** no procedimento ajustado (Seção 1.1; protocolo, Seção 5.4).

Detalhes:
- auditoria, Seção 2.4;
- comparação, Seção 7.1.

---

## 3. Resultados exploratórios

### 3.1 Escopo comum

- **Dados:** somente o treino oficial — 417 nódulos de 291 pacientes (209 da classe 0, 208 da classe 1). As partições de validação e de teste **não foram lidas** em nenhuma execução nem auditoria.
- **Alvo:** suspeita radiológica de malignidade (mediana de escores de radiologistas), **não câncer confirmado**.
- **Natureza:** atividades próprias e somente exploratórias (`piloto_exploratorio` e `cv_repetida_exploratoria`), fora das Etapas 1–5 e executadas antes de P1–P5. Os manifestos registram `finalidade = exploratoria` e `elegivel_definitiva = false`; a rastreabilidade recusa promovê-los a resultado definitivo.
- **Objeto avaliado:** o **procedimento ajustado** de seleção + modelo (M2v, M3v e M4v, com S_k refeito em cada treino externo), comparado a M0 (diâmetro) e M1 (baseline). Não é a avaliação das oito features fixas (Etapa 3).
- **Fora dessas execuções:** permutações (P1), spline (P2), M2–M4 com lista fixa (P3), validação e teste.
- **Configuração:** `b43381f7…` no piloto e `c36335c1696b99f5d2344e80f14884721007da5810f6be6c8b892de60f4c5885` na CV repetida. A diferença vem só da inclusão da atividade `cv_repetida_exploratoria` no snapshot da configuração.

### 3.2 Piloto exploratório (1 repetição × 5 folds) — verificado

Fonte: [`reports/sprint4/piloto_exploratorio/relatorio_piloto_exploratorio.json`](../../reports/sprint4/piloto_exploratorio/relatorio_piloto_exploratorio.json) e o manifesto ao lado (commit `13dc7c9`).

- AUC OOF: M0 0,9457; M1 0,9437; M2v 0,9688; M3v 0,9686; M4v 0,9687.
- Δ M3v − M0 = +0,0229; Δ M4v − M1 = +0,0250.
- 7 `ConvergenceWarning` na busca interna de C; os refits que definem as features convergiram.

### 3.3 CV repetida exploratória (10 repetições × 5 folds) — verificado

Fonte: [`reports/sprint4/cv_repetida_exploratoria/relatorio_cv_repetida_exploratoria.json`](../../reports/sprint4/cv_repetida_exploratoria/relatorio_cv_repetida_exploratoria.json) (SHA-256 `a5a8eae289a1ed379913f52d1106db9905978b66e972bd554e8c35fadffc0118`) e [`manifesto_cv_repetida_exploratoria.json`](../../reports/sprint4/cv_repetida_exploratoria/manifesto_cv_repetida_exploratoria.json) (SHA-256 `ea2dd0afa911651e8fe8a6d9d5f403ad33419247c3a78feef8663337dc9e04a7`), commit `bda673d`; código em `9eef000`.

Estrutura (protocolo, Seção 7.2): r = 0…9, seed externa 42 + r; 5 folds externos agrupados por paciente; S_k calculado uma vez por fold externo e reutilizado em M2v, M3v e M4v. A unidade de resumo é a **repetição**: em cada uma, as 417 predições out-of-fold de cada modelo são reunidas e se calcula uma AUC e um Δ por contraste.

**AUC OOF entre as 10 repetições** (média e mínimo–máximo, só descritivos):

| Modelo | Média | Mínimo | Máximo |
|---|---|---|---|
| M0 | 0,9451 | 0,9415 | 0,9468 |
| M1 | 0,9435 | 0,9370 | 0,9451 |
| M2v | 0,9706 | 0,9683 | 0,9730 |
| M3v | 0,9704 | 0,9679 | 0,9729 |
| M4v | 0,9704 | 0,9678 | 0,9729 |

**Contrastes por repetição:**

| | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | média | mín.–máx. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Δ M3v − M0 | +0,0229 | +0,0258 | +0,0233 | +0,0267 | +0,0227 | +0,0262 | +0,0298 | +0,0267 | +0,0255 | +0,0232 | +0,0253 | +0,0227 a +0,0298 |
| Δ M4v − M1 | +0,0250 | +0,0268 | +0,0251 | +0,0284 | +0,0236 | +0,0273 | +0,0343 | +0,0274 | +0,0275 | +0,0234 | +0,0269 | +0,0234 a +0,0343 |

- Os 10 Δ são positivos nos dois contrastes. O mínimo–máximo descreve **sensibilidade à partição**, não incerteza amostral: as repetições reutilizam os mesmos 417 nódulos.
- M3v ≈ M2v e M4v ≈ M2v em todas as repetições: somar o diâmetro ou o M1 a S_k praticamente não muda a AUC.
- A repetição r = 0 é numericamente idêntica ao piloto exploratório (AUCs, S_k, C, predições e parâmetros ajustados). Só o hash da seleção difere, porque o registro inclui o hash da configuração.

**Estabilidade da seleção nos 50 treinos externos:**

> **Seleção dentro dos folds × seleção no treino completo.** Cada S_k abaixo foi selecionado **dentro de um treino externo** da CV (de 323 a 346 nódulos, cerca de 4/5 do treino), sem acesso ao fold de avaliação correspondente. Isso é diferente da seleção prevista na Etapa 0, item 2, do protocolo: o procedimento ajustado aplicado **uma única vez aos 417 nódulos do treino completo**, com a lista resultante registrada só de forma descritiva. Essa seleção no treino completo **não foi executada** (Seção 5). As frequências abaixo descrevem a estabilidade do procedimento entre partições; não substituem a lista do treino completo nem a lista fixa de M2–M4 (P3).

- |S_k| de 5 a 11 (média 8,58); nenhum S_k vazio.
- Frequência das features selecionadas ao menos uma vez:

| Feature | Freq. | Feature | Freq. |
|---|---|---|---|
| `shape_Sphericity` | 50 | `ngtdm_Coarseness` | 14 |
| `shape_SurfaceVolumeRatio` | 50 | `glcm_Id` | 11 |
| `firstorder_90Percentile` | 49 | `shape_MajorAxisLength` | 9 |
| `glcm_Imc2` | 49 | `firstorder_Kurtosis` | 8 |
| `shape_Elongation` | 45 | `shape_Flatness` | 6 |
| `firstorder_RootMeanSquared` | 42 | `ngtdm_Busyness` | 5 |
| `glcm_Imc1` | 32 | `glcm_Idmn` | 4 |
| `firstorder_Skewness` | 26 | `firstorder_10Percentile` | 3 |
| `glcm_ClusterShade` | 20 | `firstorder_Entropy`, `glcm_Correlation` | 2 cada |
| | | `glcm_JointEnergy`, `glrlm_LongRunLowGrayLevelEmphasis` | 1 cada |

- Índice de Jaccard entre cada S_k e a lista declarada da Seção 5.1 do protocolo (só nomes; nenhum modelo de lista fixa ajustado): média 0,653, mínimo 0,30, máximo 1,00.
- O diâmetro e o `MeshVolume` não entram em nenhum S_k: o filtro de correlação os remove antes do L1.

**Convergência e warnings:**

- 80 `ConvergenceWarning` do liblinear na busca interna de C (de 4 a 11 por repetição). Segundo a auditoria dos checkpoints — dado que não consta do relatório JSON —, eles se distribuem por 66 dos 250 folds de estabilidade.
- Os 250 refits que definem as features selecionadas convergiram; os 250 ajustes de avaliação não tiveram warnings nem atingiram o limite de iterações.
- O registro não guarda os scores por C; não é possível descartar que algum aviso tenha alterado a escolha de C e, por ela, features no limiar de 3 de 5.

**Auditoria (somente leitura):** 250 de 250 checkpoints íntegros; 417 de 417 predições OOF por modelo em cada repetição; nenhum paciente em treino e avaliação do mesmo fold; AUCs, contrastes, frequências e Jaccard recalculados com igualdade exata; relatório e manifesto sem dados individuais.

### 3.4 Leitura permitida

- **Pode-se dizer:** o procedimento ajustado de seleção + modelo teve, de forma consistente nas 10 partições do treino, AUC cerca de 0,025 maior que o diâmetro com termo linear, para o alvo de suspeita radiológica.
- **Não se pode dizer:** que há informação além do tamanho, que o efeito é independente do tamanho, que há valor clínico, nem que esse desempenho vale para as oito features fixas, que são outro objeto (Etapa 3).

## 4. Limitações dos resultados exploratórios

1. **Sem intervalo de confiança.** Pelo protocolo, a CV não tem IC; o IC95% por bootstrap pareado por paciente existe só na validação e no teste (protocolo, Seções 7.2 e 8.2).
2. **Sem inferência estatística nem comparação estatística entre modelos.** Não há p-valor por desenho (protocolo, Seção 8.3), e a leitura da Seção 9 do protocolo exige o IC no teste.
3. **Sem controle por permutação** (P1): não há verificação de vazamento por rótulos permutados.
4. **Sem spline** (P2): o Δ não separa efeito não linear do tamanho de informação adicional. S_k carrega muita informação de tamanho (por exemplo, `SurfaceVolumeRatio` decresce aproximadamente com o inverso do tamanho), e M3v ≈ M2v é coerente com isso.
5. **Sem avaliação oficial:** validação e teste não foram usados; nada aqui é desempenho em dados independentes.
6. **Executado antes de P1–P5 e da revisão do protocolo.** Ver estes números antes das decisões cria risco de decisões influenciadas pelos resultados; a Etapa 2 oficial terá de ser executada de novo.
7. **Teto de AUC:** o diâmetro isolado já fica perto de 0,945, o que deixa pouco espaço para ganho.
8. **Seleção instável fora do núcleo:** 6 features aparecem em ≥ 42 de 50 treinos externos; as demais variam.
9. **Avisos de convergência não auditáveis em detalhe** (Seção 3.3).
10. **Calibração não avaliada** (Seção 10.2 do protocolo).
11. As limitações do protocolo (protocolo, Seção 11), em especial a circularidade do alvo com o tamanho e a exclusão dos casos intermediários, continuam valendo.

---

## 5. Decisões pendentes e etapas bloqueadas

Nenhuma das decisões abaixo foi tomada. **Nada aqui presume aprovação.** No código, todas estão registradas como pendentes, e as guardas bloqueiam as atividades que as exigem.

| # | Pendência | Bloqueia | Prazo no protocolo |
|---|---|---|---|
| P1 | Número de permutações K, esquema (por nódulo ou por paciente) e tratamento dos folds de estabilidade e internos sob rótulos permutados | Etapas 1 e 2 oficiais (controle por permutação) e todas as seguintes | antes da Etapa 1 |
| P2 | Confirmação dos parâmetros do spline (protocolo, Seção 10.1) | Etapas 1 e 2 oficiais (M0s, M3vs) e todas as seguintes | antes da Etapa 1 |
| P3 | Regra para divergência na reprodução fiel e **lista fixa** de M2, M3 e M4, incluindo a recomendação de Juan (Seção 2.5 deste documento) | Etapa 3 e seguintes; CV descritiva com lista fixa da Etapa 2 | antes da Etapa 3 |
| P4 | Divergência entre M1 = 0,9778 (Experimento 13) e M1 = 0,9762 (baseline canônico) | Etapa 5 | antes da Etapa 5 |
| P5 | Versionamento dos registros dos Experimentos 10 e 13 e do notebook de Juan com suas saídas. Segundo Juan, não existem saídas históricas, e o notebook da regeneração recente foi recebido sem saídas (Seção 2.3 deste documento). Como tratar essa parte depende de decisão do grupo. | Etapa 4 e seguintes | antes da Etapa 4 |
| Revisão | Revisão do protocolo pelo grupo e pelo professor. Na revisão formal: as Seções 5.2 e 11.3 (item 8) do protocolo, que ainda dizem que a reprodução independente não está documentada, devem remeter à auditoria e à comparação; e deve ficar registrado que os resultados exploratórios da Seção 3 deste documento já eram conhecidos. | Etapa 4 e seguintes | antes do congelamento (Etapa 4) |
| — | **Proveniência exata dos CSVs regenerados de Juan.** O notebook foi recebido, mas não tem saídas; a ligação entre código e saídas não está comprovada, e o script do CSV de quatro colunas não foi recebido. | — (insumo para P3) | a considerar na decisão sobre P3 |
| — | Registro da composição dos folds da regeneração recente (por exemplo, hash por fold), só para comparação interna — ainda não solicitado formalmente | — (apoiaria a análise de H1) | opcional |

Situação das etapas:

| Etapa | Situação | Bloqueada por |
|---|---|---|
| 0 — Pré-requisitos | **parcial:** reprodução fiel (protocolo, Seção 7.1, item 1) e invariantes com o M1 (item 3) concluídas (Seções 1 e 2 deste documento); falta o item 2 — o procedimento ajustado aplicado **uma vez ao treino completo** (417 nódulos), com a lista registrada só de forma descritiva, sem substituir a lista fixa e sem avaliação na validação ou no teste. A seleção feita dentro dos folds da CV exploratória (Seção 3.3) não cumpre este item | — (o item 2 não exige decisão) |
| 1 — CV aninhada piloto (oficial) | não executada; existe só o piloto exploratório | P1, P2 |
| 2 — CV aninhada completa (oficial) | não executada; existe só a CV repetida exploratória | P1, P2 (P3 para a CV com lista fixa) |
| 3 — Validação oficial | não executada | P1, P2, P3 |
| 4 — Congelamento | não executada | P1, P2, P3, P5, revisão do protocolo |
| 5 — Teste oficial | não executado | todas |

A lista de features **não pode** ser decidida com base em desempenho em validação ou teste. A decisão deve se apoiar em evidência de procedimento e em critérios definidos antes de qualquer resultado.

A divergência entre a lista declarada e a lista reproduzida continua sendo **pendência científica**. A preservação técnica da reprodução (commit `0ef4039`) não a resolve.

## 6. Definition of Done do card

| Critério do card | Situação | Evidência | Falta |
|---|---|---|---|
| Protocolo comparativo prévio | **parcial** | `docs/sprint4/protocolo_valor_incremental.md` (`9e9b3bd`) | aprovação e revisão do protocolo; atualização das Seções 5.2 e 11.3 do protocolo |
| Script/notebook reproduzível | **parcial** | `src/incremental/`, com CLI para o piloto e a CV repetida; 428 testes; retomada e r = 0 reproduzidas byte a byte | código das Etapas 3 e 5, do bootstrap pareado, da spline, das permutações e das métricas da Seção 10.2 do protocolo |
| Métricas e intervalos de confiança | **parcial** | AUC e Δ por repetição, média e mínimo–máximo (Seção 3.3) | IC95% por bootstrap na validação e no teste; log-loss, Brier e calibração |
| Comparação estatística entre modelos | **pendente** | só contrastes descritivos | leitura do IC95% de Δ no teste (Seção 9 do protocolo) |
| Registro das features utilizadas | **concluído** (com ressalva) | S_k dos 50 treinos externos da CV, frequências e Jaccard (Seção 3.3); lista declarada × reproduzida (Seção 2) | lista do procedimento ajustado no treino completo (Etapa 0, item 2, só descritiva); lista fixa de M2–M4 (P3) |
| Conclusão escrita sobre valor incremental além do diâmetro | **pendente** | só a leitura exploratória da Seção 3.4 | conclusão pela Seção 9 do protocolo, com spline e IC |
| Limitações | **concluído** | protocolo, Seção 11; Seção 4 deste documento | — |
| Consistência entre divisões e robustez das features | **parcial** | consistência entre as 10 partições do treino; núcleo estável de features (Seção 3.3) | concordância entre CV, validação e teste; robustez entre anotadores (protocolo, Seção 11.3, item 10) |

**Conclusão: o desenvolvimento exploratório está concluído, mas o cartão científico permanece em andamento.** O encerramento depende das decisões da Seção 5 e das Etapas 2 a 5 oficiais, ou de uma redefinição explícita do escopo do card pelo professor.

## 7. Próximos passos

Implementação possível sem nenhuma decisão:
- Etapa 0, item 2: procedimento ajustado aplicado ao treino completo, com a lista registrada só de forma descritiva (protocolo, Seção 7.1);
- documentação e pequenos ajustes técnicos (teste de divergência de seleção em r > 0; conferência de seeds no contexto da execução);
- infraestrutura das Etapas 3 e 5 — bootstrap pareado por paciente (protocolo, Seção 8.2) e métricas da Seção 10.2 do protocolo — escrita e testada com dados sintéticos, **sem executar** com dados oficiais.

Dependem de decisão:
- reexecução oficial das Etapas 1 e 2, com spline e permutações (P1, P2);
- Etapa 3 com a lista fixa (P3); Etapas 4 e 5 (P4, P5 e revisão do protocolo).

## 8. Restrições em vigor

- **Não escolher features por desempenho**, em nenhum conjunto.
- **Não avaliar M2, M3 nem M4**, em CV, validação ou teste, sem autorização.
- **Não usar linhas de validação ou de teste na CV aninhada**, nem em qualquer ajuste, filtro ou seleção das Etapas 0 a 2.
- **Não modificar o protocolo sem decisão documentada**, registrada em revisão com data e motivo.
- **Não apresentar os resultados exploratórios como resultado das etapas oficiais**, nem como evidência de valor além do tamanho.
- Não alterar coorte, alvo, split, features nem os arquivos em `data/`.
- Não executar `run_baseline.py`, porque ele sobrescreve os relatórios da Sprint 3.
- Não copiar dados individuais de paciente ou nódulo para documentos ou relatórios.
- Commit e push só com autorização explícita.

## 9. Estado do Git no registro

| Situação | Arquivo |
|---|---|
| Versionado em `9e9b3bd` | `docs/sprint4/protocolo_valor_incremental.md` |
| Versionado em `7f1a0f6` | `scripts/sprint4/validate_reference_models.py`; `reports/sprint4/reference_models/validacao_modelos_referencia.json` |
| Versionado em `a70db2e` e atualizado em `cf2bae0` | `reports/sprint4/feature_selection/auditoria_reproducao_juan.md`; `reports/sprint4/feature_selection/comparacao_pacote_juan.md`; `docs/sprint4/estado_cartao4.md` |
| Versionado em `0ef4039` | `scripts/sprint4/reproduce_juan_feature_selection.py`; os quatro arquivos de `reports/sprint4/feature_selection/reproducao_juan/`; `.gitattributes` (regras de fim de linha) |
| Versionado em `6214489` … `5e62228` | `src/incremental/` (infraestrutura) e `tests/incremental/` |
| Versionado em `7cf2694`, `6a87e88` | atividade e orquestrador do piloto exploratório |
| Versionado em `13dc7c9` | `reports/sprint4/piloto_exploratorio/` (relatório e manifesto); regra de fim de linha |
| Versionado em `165584c`, `03a922d`, `9eef000` | atividade, refatoração do piloto e orquestrador da CV repetida exploratória |
| Versionado em `bda673d` | `reports/sprint4/cv_repetida_exploratoria/` (relatório e manifesto); regra de fim de linha |
| Fora do repositório | checkpoints do piloto e da CV repetida (contêm predições individuais); arquivos de tempos e logs de execução; artefatos históricos das execuções A e B e scripts auxiliares da comparação; `pacote_auditoria_selecao_features.zip`; notebooks de Juan |

Nada foi enviado ao remoto nesta branch.

---

_Checkpoint de continuidade do Cartão 4. Atualize a cada etapa concluída ou decisão registrada; decisões em si pertencem a revisões do protocolo._
