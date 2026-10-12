# Comparação do Pacote de Juan com a Reprodução da Seleção de Features — Sprint 4

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Card relacionado:** S4 — Cartão 4, Etapa 0: reprodução independente da seleção de features
**Documentos relacionados:**
- [`auditoria_reproducao_juan.md`](auditoria_reproducao_juan.md): nossa reprodução;
- [`docs/sprint4/estado_cartao4.md`](../../../docs/sprint4/estado_cartao4.md): checkpoint do Cartão 4;
- [`docs/sprint4/protocolo_valor_incremental.md`](../../../docs/sprint4/protocolo_valor_incremental.md): Seções 5.2, 5.3 e 7.1.

**Objeto comparado:** `pacote_auditoria_selecao_features.zip`, enviado por Juan
**Status:** registro de comparação — **a lista definitiva de features permanece pendente de decisão do grupo (P3)**
**Registro:** 11/10/2026
**Revisão:** 11/10/2026 — incorpora os esclarecimentos de Juan (Seção 15). Os resultados verificados das Seções 3 a 9 não mudaram.
**Revisão:** 12/10/2026 — registra o recebimento do notebook da regeneração recente (Seção 15.1) e a limitação metodológica dos filtros (Seção 7.1)

---

## Aviso de escopo

Esta comparação é **somente estática**:
- nenhum código do pacote foi executado;
- nenhuma seleção de features foi refeita;
- nenhum modelo foi ajustado nem avaliado;
- nenhuma métrica de validação ou de teste foi calculada.

O pacote foi aberto e lido em diretório temporário, fora do repositório. Nem o ZIP nem os CSVs dele foram copiados para o repositório. Este documento não contém identificadores de paciente ou de nódulo.

O alvo é a suspeita radiológica de malignidade (mediana de escores de radiologistas), não malignidade histológica confirmada.

Convenções:
- **"verificado"** designa o que foi medido nos arquivos;
- **"hipótese"** designa explicação possível que **não** foi testada;
- **"declarado por Juan"** designa informação dada pelo responsável pela seleção original. Ela é registrada como declaração, e não como verificação independente;
- **"execução do pacote"** designa os resultados contidos nos CSVs do ZIP. Juan confirmou depois que esses CSVs vêm de uma **regeneração recente**, e não da execução histórica no Colab que gerou a conclusão do notebook (Seção 15).

Nada aqui atribui a divergência a erro de quem conduziu a seleção original.

## 1. Objetivo

Registrar, com hashes e contagens agregadas, o que o pacote de Juan permite verificar sobre a divergência documentada na auditoria: 6 das 8 features declaradas reproduzidas e 2 divergentes. Também registrar o que o pacote **não** permite verificar.

## 2. Método

| Verificação | Como foi feita |
|---|---|
| Inventário e hashes | leitura de cada entrada do ZIP e cálculo do SHA-256 |
| Entradas | SHA-256 comparado com a Seção 3.3 da auditoria; no split, comparação após normalizar **somente** CRLF → LF |
| Consistência interna | leitura dos três CSVs do pacote com pandas, sem executar código do pacote; somas, regra ≥ 3/5, interseção e Jaccard recalculados a partir da tabela de frequências |
| Folds | contagens por fold do `resumo_folds_selecao.csv` do pacote comparadas com as da nossa execução B |
| Notebooks | extração das células de código e de markdown dos dois `.ipynb`, com `diff` textual |
| Versões | leitura de `versoes_ambiente_atual.txt` e do README do pacote |
| Formato | comparação entre os CSVs do pacote e as instruções de exportação do notebook |

Os três scripts auxiliares e suas saídas ficaram fora do repositório (Seção 13).

## 3. Inventário do pacote — verificado

**ZIP:** 863 972 bytes; SHA-256 `a88b2f0ba0e87410c2ec780877e96da8604810e7d28be9d929fd69321c01dda7`. Tem 11 entradas, todas na raiz, sem subdiretórios, compactadas com deflate.

| Arquivo | Bytes | SHA-256 | Em `sha256_artefatos.txt` |
|---|---:|---|:---:|
| `base_radiomica_oficial_422.csv` | 1 831 915 | `4068f7d661eac3bdfd33342a3759a0dfa80d42e47d4be4be00cd85b4e13563b1` | ✓ confere |
| `vies_malignidade_mascara_pequena.csv` | 35 261 | `67f3fd2c3a55de75a58c40e7380480a5df0f5747ac017ec2b1f6ec091c3c6478` | ✓ confere |
| `patient_split.csv` | 9 000 | `812fe3ffe51f11257a9aa74529e1bfc19ed58285a11142f0e8b5739c1d1a4363` | ✓ confere |
| `frequencia_selecao_features.csv` | 6 391 | `6f1abed3b3a013cb1d3f2b36e93ed671567b483c2076c3a7ed9da9a5f7c4112c` | ✓ confere |
| `resumo_folds_selecao.csv` | 356 | `9cac6e372c1269ee4d703cd76837b80cd39a7b4d402c939de15c9021ef1b5276` | ✓ confere |
| `features_selecionadas_lasso.csv` | 467 | `7dd55bac3c06251e9b93dd0cdbbaf9e5c87424e7a4fea17e5fe17c3e8bc36967` | ✓ confere |
| `features_divergentes_execucao_registrada.csv` | 175 | `2105aeae52ac18202c049557167dd4e3b99e20436a936f29e43b6c88de24b820` | não listado |
| `S4_Selecao_Features_Execucao_FINAL_SPLIT_OFICIAL.ipynb` | 19 899 | `3a5e8e63c4b47ff8144ace04d63e6b4e7cb16e38e41ba045701d7eaa4a7f9835` | não listado |
| `README_reproducao_selecao.txt` | 1 690 | `79ea5c02d7725b14c8aba4806b8f127650e4624bcf094e64482d1f9244f0e489` | não listado |
| `versoes_ambiente_atual.txt` | 346 | `bb375f0dbf1e490d74f2ebbe71bd53133c9009054ee327e797da5a237fba3b38` | não listado |
| `sha256_artefatos.txt` | 571 | `72aa7d0b7dc75eebf873236805f51a68910415c9f3329ec750fa3caa69dc48d3` | — |

- Os seis hashes declarados em `sha256_artefatos.txt` conferem com os bytes do pacote.
- As datas internas do ZIP vão de 11/10/2026 14:41 a 14:42. Elas não trazem fuso horário e indicam apenas a última modificação ou o empacotamento dos arquivos, **não** a data em que a seleção foi executada.
- Todos os CSVs do pacote usam LF e não têm BOM.

## 4. Arquivos de entrada — verificado

| Arquivo | Pacote | Auditoria (Seção 3.3) | Resultado |
|---|---|---|---|
| Base radiômica | `4068f7d6…` | `4068f7d6…` | **bytes idênticos** |
| Alvo (`vies_malignidade_mascara_pequena.csv`) | `67f3fd2c…` | `67f3fd2c…` | **bytes idênticos** |
| `patient_split.csv` | `812fe3ff…` (LF) | `ed120c2b…` (CRLF) | **idêntico após normalizar somente CRLF → LF** |

Detalhe do split:

| Cópia | Bytes | Quebras de linha | SHA-256 |
|---|---:|---|---|
| Pacote | 9 000 | 417 LF | `812fe3ff…` |
| Cópia de trabalho do repositório, usada na auditoria | 9 417 | 417 CRLF | `ed120c2b…` |
| Cópia de trabalho normalizada para LF | 9 000 | 417 LF | `812fe3ff…` |
| Conteúdo versionado no Git (`HEAD:reports/sprint3/baseline/patient_split.csv`) | — | LF | `812fe3ff…` |

A diferença de 417 bytes corresponde a um `\r` por linha. O Git registra o arquivo com LF no índice e o converte para CRLF na cópia de trabalho (`core.autocrlf=true`; o `.gitattributes` não fixa EOL para `reports/sprint3/`). **O split do pacote é byte a byte igual ao conteúdo versionado no repositório.**

A tabela `data/modeling_table_sprint3.csv`, usada nas checagens da auditoria, não faz parte do pacote. Para ela não há comparação.

**Alcance:** isso prova que os arquivos **do pacote** são iguais aos do projeto. Não prova que foram esses os arquivos lidos na execução original do Colab (Seção 8). Juan declarou que não há saídas nem logs históricos daquela execução (Seção 15), então essa prova não é possível com o material disponível.

## 5. Consistência interna dos CSVs do pacote — verificado

Valores recalculados a partir de `frequencia_selecao_features.csv` e `resumo_folds_selecao.csv` do pacote.

| Verificação | Resultado |
|---|---|
| Linhas da tabela de frequências | 107, sem features repetidas; mesmo conjunto de 107 nomes da nossa execução |
| Soma de `freq_lasso` = soma de `n_lasso_selecionadas` | 47 = 47 |
| Soma de `freq_rf_top15` = soma de `n_rf_top15` | 75 = 75 (5 × 15) |
| Soma de `freq_pos_filtros` = soma de `n_pos_correlacao` | 183 = 183 |
| `freq_lasso` ≤ `freq_pos_filtros` e `freq_rf_top15` ≤ `freq_pos_filtros` | em todas as 107 linhas |
| `pct_lasso` e `pct_rf_top15` = frequência / 5 | em todas as linhas |
| Colunas `estavel_*_3de5` = frequência ≥ 3 | em todas as linhas |
| Lista LASSO ≥ 3/5 obtida da tabela = lista do `features_selecionadas_lasso.csv` | sim, as mesmas 8 features |
| Essa lista = lista declarada no notebook (protocolo, Seção 5.1) | **sim, as mesmas 8** |
| Features estáveis no RF top-15 (≥ 3/5) | 16 |
| Interseção LASSO × RF | 5: `Sphericity`, `SurfaceVolumeRatio`, `Imc2`, `Skewness`, `Coarseness` |
| Jaccard | 5/19 = 0,263 |
| Faixa de features após o filtro de correlação | 35–39 |
| Faixa de features selecionadas pelo LASSO | 7–13 |
| `features_divergentes_execucao_registrada.csv` e tabela do README | coerentes com a tabela de frequências |

**Os CSVs do pacote são internamente consistentes.** A lista declarada, as estatísticas 16 / 5 / 0,263 e as faixas 35–39 e 7–13 do texto do notebook saem todas da mesma tabela de frequências e do mesmo resumo de folds.

**Alcance:** a consistência vale para a **regeneração recente** (Seção 15). Ela mostra que uma única execução do procedimento, no ambiente da regeneração, produz ao mesmo tempo a lista declarada e as estatísticas do texto. Não mostra o que a execução histórica produziu.

### 5.1 Frequências: pacote × nossa reprodução

Features com frequência LASSO diferente (14 de 107):

| Feature | LASSO no pacote | LASSO na reprodução |
|---|---:|---:|
| `glcm_Imc2` | 5 | 4 |
| `firstorder_90Percentile` | 5 | 4 |
| `firstorder_Skewness` | **3** | **1** |
| `ngtdm_Coarseness` | **3** | **1** |
| `glcm_Imc1` | **2** | **4** |
| `glcm_ClusterShade` | **2** | **3** |
| `glcm_Id` | 1 | 2 |
| `firstorder_10Percentile` | 1 | 0 |
| `firstorder_Entropy` | 1 | 0 |
| `firstorder_Kurtosis` | 1 | 0 |
| `glcm_Correlation` | 1 | 0 |
| `shape_LeastAxisLength` | 1 | 0 |
| `shape_MajorAxisLength` | 0 | 1 |
| `gldm_LargeDependenceLowGrayLevelEmphasis` | 0 | 1 |

Também diferem `freq_rf_top15`, em 7 features, e `freq_pos_filtros`, em 3. As quatro features divergentes estão em negrito.

### 5.2 Random Forest: mesmas contagens, conjuntos diferentes

| | Pacote | Reprodução |
|---|---|---|
| Features estáveis no RF | 16 | 16 |
| Só neste conjunto | `glszm_SizeZoneNonUniformityNormalized`, `glszm_ZonePercentage` | `firstorder_RootMeanSquared`, `ngtdm_Contrast` |
| Interseção LASSO × RF | `Sphericity`, `SurfaceVolumeRatio`, `Imc2`, **`Skewness`, `Coarseness`** | `Sphericity`, `SurfaceVolumeRatio`, `Imc2`, **`Imc1`, `RootMeanSquared`** |
| Jaccard | 5/19 = 0,263 | 5/19 = 0,263 |

**Correção de leitura da auditoria.** A Seção 8 da auditoria diz que as estatísticas 16 / 5 / 0,263 foram "reproduzidas exatamente". O pacote mostra que essa coincidência é **só numérica**: as contagens são iguais, mas os conjuntos de features por trás delas são diferentes. O raciocínio da hipótese H3 da auditoria usava o RF da nossa reprodução; o pacote traz o RF da sua própria execução (Seção 10).

## 6. Folds — verificado

Contagens da parte de treino de cada fold externo de estabilidade. A parte de avaliação é o complemento até 417 nódulos.

| Fold | Nódulos: pacote | Nódulos: reprodução | Pacientes: pacote | Pacientes: reprodução | Após correlação: pacote / reprodução | Melhor `C`: pacote / reprodução | LASSO: pacote / reprodução |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 333 | 331 | 232 | 232 | 39 / 37 | 0,316 / 0,316 | 9 / 7 |
| 2 | 331 | 339 | 231 | 231 | 37 / 36 | 0,316 / 0,316 | 8 / 10 |
| 3 | 335 | 328 | 237 | 237 | 37 / 36 | **1,000** / 0,316 | 13 / 10 |
| 4 | 335 | 337 | 231 | 231 | 35 / 35 | 0,316 / 0,316 | 10 / 8 |
| 5 | 334 | 333 | 233 | 233 | 35 / 36 | 0,316 / **0,100** | 7 / 7 |
| Soma | 1 668 | 1 668 | 1 164 | 1 164 | 183 / 180 | — | 47 / 42 |

- **Pacientes por fold: iguais nos cinco folds.**
- **Nódulos por fold: diferentes nos cinco folds.** Como cada nódulo pertence a um único paciente, conjuntos de pacientes iguais teriam o mesmo número de nódulos. Logo, **a composição de pacientes difere em todos os cinco folds** entre a execução do pacote e a nossa.
- As somas (1 668 = 4 × 417 e 1 164 = 4 × 291) confirmam, nas duas execuções, que cada nódulo e cada paciente do treino ficam fora do treino em exatamente um fold.
- A comparação usa apenas contagens. O pacote não informa quais pacientes estão em cada fold, então não é possível dizer **quantos** pacientes mudam de fold.

Com entradas idênticas (Seção 4) e código equivalente (Seção 7), a divergência já aparece **na partição**, antes dos filtros e do LASSO. Isso é compatível com a sensibilidade à partição descrita na Seção 7.1 da auditoria.

## 7. Notebook do pacote × cópia auditada — verificado

| Item | Cópia auditada | Notebook do pacote |
|---|---|---|
| Arquivo | `S4_—_Revisão_e_execução_da_seleção_de_features_radiômicas.ipynb` | `S4_Selecao_Features_Execucao_FINAL_SPLIT_OFICIAL.ipynb` |
| SHA-256 | `48faaea6…` | `3a5e8e63…` |
| Células | 15 (7 de código, 8 de markdown) | 15 (7 de código, 8 de markdown) |
| Saídas gravadas | nenhuma; `execution_count` vazio | nenhuma; `execution_count` vazio |
| Metadados | `kernelspec` Python 3, `language_info` sem versão, bloco `colab` | `kernelspec` Python 3, `language_info` sem versão, sem bloco `colab` |

**Código:** as 7 células são idênticas, exceto por:
- um comentário: "Split oficial da Sprint 3." passou a "Split oficial da Sprint 3 (obrigatório nesta versão final).";
- uma linha em branco a mais na mesma célula.

Parâmetros, seeds, caminhos relativos, filtros, grade de `C`, Random Forest, regra ≥ 3/5 e instruções de exportação são os mesmos. Por isso a aderência da nossa reprodução à cópia auditada (auditoria, Seção 2) vale também para o notebook do pacote.

**Markdown:**
- a primeira célula ganhou um cabeçalho com título, responsável, revisor, prazo e objetivo;
- duas frases foram reescritas para citar explicitamente o `patient_split.csv` oficial;
- a conclusão é a mesma nos dois: a lista declarada, "aproximadamente 35–39" features após a correlação, "7 a 13" no LASSO, 16 features no RF, interseção 5 e Jaccard 0,263.

**Nenhum dos dois notebooks tem saídas de execução.** Por isso não registram versões, warnings nem resultados intermediários da execução original.

**Ressalva:** a comparação vale para o **arquivo** de notebook do pacote. Em 12/10/2026, Juan enviou o notebook que declarou ter usado na regeneração recente. Ele é **idêntico byte a byte** ao notebook do pacote (Seção 15.1).

Como esse notebook não tem saídas nem execução registrada, o recebimento **não comprova** que foi esse código que gerou os CSVs regenerados. A proveniência exata das saídas continua sem comprovação.

### 7.1 Limitação metodológica do procedimento original — verificado no código

Na célula do laço de seleção, idêntica nos dois notebooks, os filtros são ajustados **antes** e **fora** da `GridSearchCV` interna:
- `VarianceThreshold(0.0)` é ajustado com `fit` sobre todo o treino do fold de estabilidade;
- o filtro de correlação (|r| > 0,90) é calculado sobre esse mesmo treino completo;
- só depois a `GridSearchCV`, com CV interna de 4 folds, recebe a matriz já filtrada.

Consequência: cada fold de validação **interna** participou da definição do conjunto de features avaliado nele.
- Os dois filtros **não usam o alvo**, então não há vazamento de rótulo.
- A parte de avaliação de cada fold de estabilidade não entra no ajuste.
- Mesmo assim, a escolha de `C` pela CV interna não é estritamente independente dos filtros.

Essa é uma **limitação metodológica do procedimento original**, que a reprodução fiel replica de propósito (auditoria, Seções 2.2 e 2.4). Ela vale igualmente para a execução histórica, para a regeneração recente e para a nossa reprodução, e por isso não explica a divergência entre elas. Essa forma de aplicar os filtros **não deve ser reutilizada na CV aninhada corrigida**: lá, os filtros entram no pipeline e são ajustados em cada treino interno, como prevê a Seção 5.4 do protocolo.

## 8. Versões de ambiente informadas — verificado o conteúdo, não o ambiente

Conteúdo de `versoes_ambiente_atual.txt`:

| Biblioteca | Pacote: ambiente "atual" de Juan | Nossa reprodução (`.venv`) |
|---|---|---|
| Python | 3.13.5 | 3.9.13 |
| NumPy | 2.3.5 | 1.23.5 |
| pandas | 2.2.3 | 1.5.3 |
| scikit-learn | 1.8.0 | 1.3.2 |
| SciPy, joblib, sistema operacional | não informados | 1.10.1, 1.5.3, Windows |

O próprio pacote faz duas ressalvas:

- `versoes_ambiente_atual.txt`: "o notebook original não registrou/persistiu as versões do ambiente Colab", e as versões listadas são "do ambiente atual usado para auditoria/regeneração dos artefatos nesta sessão". Por isso, "não tratar estas versões como comprovação das versões históricas do Colab original";
- README: "as versões históricas exatas não podem ser confirmadas a partir dos artefatos disponíveis".

**Estas versões não comprovam o ambiente da execução original.** O pacote também não diz se os CSVs foram gerados no Colab original ou regenerados depois:
- o README os chama de "artefatos da execução registrada";
- o arquivo de versões fala em "regeneração dos artefatos nesta sessão".

As duas leituras são compatíveis com o pacote, e só Juan pode esclarecer (Seção 12).

**Esclarecimento posterior — declarado por Juan (Seção 15):**
- os CSVs foram **regenerados recentemente** com Python 3.13.5, NumPy 2.3.5, pandas 2.2.3 e scikit-learn 1.8.0;
- as versões da execução histórica no Colab são **desconhecidas**.

A ambiguidade acima fica resolvida a favor da regeneração. As versões da tabela descrevem o ambiente da regeneração recente, e não o da execução histórica.

## 9. Diferenças de formato em relação à exportação do notebook — verificado

A célula de exportação do notebook, idêntica nas duas cópias, grava:

```python
pd.DataFrame({
    "feature": features_finais_lasso,
    "criterio": "LASSO >= 3/5 folds"
}).to_csv("features_selecionadas_lasso.csv", index=False)
```

`features_finais_lasso` é tirada de `tabela_estabilidade`, ordenada por `freq_lasso`, `freq_rf_top15` e `freq_pos_filtros`, em ordem decrescente.

| Aspecto | Exportação definida no notebook | `features_selecionadas_lasso.csv` do pacote |
|---|---|---|
| Colunas | `feature`, `criterio` | `feature`, **`freq_lasso`, `pct_lasso`**, `criterio` |
| Ordem das linhas | a da tabela de estabilidade: `Imc2`, `Sphericity`, `SurfaceVolumeRatio`, `90Percentile`, `RootMeanSquared`, `Elongation`, `Skewness`, `Coarseness` | `90Percentile`, `Imc2`, `Sphericity`, `SurfaceVolumeRatio`, `RootMeanSquared`, `Elongation`, `Skewness`, `Coarseness` — compatível com `freq_lasso` decrescente e nome em ordem alfabética |
| Conjunto de features | 8 | as mesmas 8 |

- **O `features_selecionadas_lasso.csv` do pacote não tem o formato que a célula de exportação do notebook produz.** O conjunto de features é o mesmo; mudam as colunas e a ordem.
- `frequencia_selecao_features.csv` e `resumo_folds_selecao.csv` do pacote têm as mesmas colunas, na mesma ordem, que a exportação do notebook e que a nossa reprodução.
- `features_divergentes_execucao_registrada.csv`, `README_reproducao_selecao.txt`, `versoes_ambiente_atual.txt` e `sha256_artefatos.txt` não são gerados pelo notebook.
- **Quebras de linha:**
  - os CSVs do pacote usam LF;
  - os da nossa reprodução usam CRLF, padrão do pandas no Windows;
  - por isso, os CSVs de saída não podem ser comparados por hash sem normalização. Mesmo normalizados, diferem em conteúdo (Seção 5.1).

Isso indica que pelo menos esse CSV passou por outro script ou por edição depois da exportação, ou veio de uma versão do código diferente da distribuída. A causa não foi verificada. A diferença de formato **não altera** o conjunto de features registrado, mas impede tratar o arquivo como saída direta e inalterada do notebook.

**Esclarecimento posterior — declarado por Juan (Seção 15):** o arquivo de quatro colunas recebeu processamento adicional a partir do CSV de frequências e não é a exportação direta do notebook. A declaração é **compatível com o que foi verificado**:
- o conjunto das 8 features coincide com a regra ≥ 3/5 aplicada à tabela de frequências do pacote;
- os valores de `freq_lasso` e `pct_lasso` coincidem com os dessa tabela;
- a ordem segue `freq_lasso` decrescente e depois o nome.

O script desse processamento não foi recebido.

## 10. Efeito sobre as hipóteses da auditoria

As hipóteses H1 a H4 estão na Seção 11.2 da auditoria. H5 foi formulada na primeira versão deste documento.

Como Juan confirmou que os CSVs são de uma regeneração recente e que não há saídas históricas (Seção 15), o pacote passa a ser evidência sobre **três execuções distintas**:

| Execução | O que se sabe |
|---|---|
| **Histórica** (Colab, conclusão do notebook) | só o texto da conclusão; entradas, versões, partição e saídas desconhecidas e, segundo Juan, sem registro disponível |
| **Regeneração recente** (CSVs do pacote) | entradas idênticas às do projeto; versões declaradas por Juan; notebook declarado como o da regeneração recebido e idêntico ao do pacote; ligação entre esse código e os CSVs não comprovada |
| **Nossa reprodução controlada** | entradas, versões, código, saídas e hashes registrados (auditoria, Seções 3 e 15) |

As conclusões abaixo **não se estendem à execução histórica**.

| # | Hipótese | Situação | Fundamento |
|---|---|---|---|
| H5 | Os CSVs do pacote foram regenerados depois da execução original e não são a saída histórica do Colab | **fato declarado e confirmado pelo responsável**, deixa de ser hipótese | declaração de Juan (Seção 15, itens 1 e 5). Compatível com o verificado: arquivo de versões que fala em "regeneração", datas internas do ZIP e formato da lista diferente do da exportação. Não houve verificação independente da data ou do ambiente da regeneração. |
| H1 | Versões de bibliotecas diferentes alteram a partição ou a numérica | **continua aberta** | entre a regeneração recente e a nossa reprodução, as entradas são idênticas (Seção 4), o notebook distribuído é equivalente ao auditado (Seção 7), e a divergência começa na partição (Seção 6). As versões diferem muito (Seção 8). O notebook que Juan declarou ter usado na regeneração é idêntico ao do pacote (Seção 15.1). Se essa declaração estiver correta, a diferença de código fica descartada, e restam o ambiente e as versões. Versões diferentes **podem** explicar os folds divergentes, mas isso **não foi testado**, e a ligação entre o notebook e os CSVs não está comprovada. Para a execução histórica, as versões são desconhecidas. |
| H2 | Arquivos de entrada diferentes | **descartada para a regeneração recente; não verificável para a execução histórica** | base, alvo e split do pacote são idênticos aos do projeto (Seção 4). Os arquivos lidos na execução histórica não têm registro. |
| H3 | A lista da conclusão e as estatísticas vêm de execuções diferentes | **o indício original perdeu a base; para a execução histórica, não verificável** | na regeneração recente, a lista declarada, 16 / 5 / 0,263 e as faixas saem de uma mesma execução, de forma consistente (Seção 5). Logo, a combinação do texto é atingível por uma única execução. O indício da auditoria dependia do nosso RF, que difere do RF do pacote (Seção 5.2). Isso **não prova** que a conclusão histórica veio de uma única execução, porque não há saídas históricas. |
| H4 | As faixas 35–39 e 7–13 vêm de outra execução | **o indício original perdeu a base; para a execução histórica, não verificável** | o resumo de folds da regeneração recente produz exatamente 35–39 e 7–13. Mesmo limite de H3. |

**Leitura conjunta.** Com as mesmas entradas e um procedimento equivalente:
- a regeneração recente, em Python 3.13.5 e scikit-learn 1.8.0, devolve a lista declarada;
- a nossa reprodução, em Python 3.9.13 e scikit-learn 1.3.2, devolve a lista com `Imc1` e `ClusterShade`.

As duas são resultados do mesmo procedimento em ambientes diferentes, com partições diferentes. **Nenhuma das duas é incorreta** pelo que se verificou. A diferença mostra que a lista resultante depende da partição e, possivelmente, do ambiente. Isso não permite inferir quais versões a execução histórica usou.

## 11. Fatos × hipóteses — resumo

| Afirmação | Situação |
|---|---|
| Base radiômica e alvo do pacote têm os mesmos bytes que os do projeto | **verificado** |
| O split do pacote é igual ao do projeto após normalizar somente CRLF → LF, e igual byte a byte ao conteúdo versionado no Git | **verificado** |
| Os seis hashes de `sha256_artefatos.txt` conferem | **verificado** |
| Os CSVs do pacote são internamente consistentes e produzem a lista declarada, 16 / 5 / 0,263 e as faixas 35–39 e 7–13 | **verificado** |
| Nossa reprodução tem as mesmas contagens 16 / 5 / 0,263, com conjuntos RF e interseção diferentes | **verificado** |
| Pacientes por fold iguais e nódulos por fold diferentes nos 5 folds, logo composições diferentes | **verificado** |
| O código do notebook do pacote equivale ao da cópia auditada (diferença só de comentário) | **verificado** |
| Nenhum dos notebooks tem saídas de execução | **verificado** |
| O CSV da lista final do pacote não tem o formato da exportação do notebook | **verificado** |
| Os CSVs do pacote vêm de regeneração recente, e não da execução histórica do Colab | **declarado por Juan**; compatível com o verificado (H5) |
| A regeneração usou Python 3.13.5, NumPy 2.3.5, pandas 2.2.3 e scikit-learn 1.8.0 | **declarado por Juan**; não verificado de forma independente |
| As versões da execução histórica são desconhecidas | **declarado por Juan** |
| O CSV de quatro colunas recebeu processamento adicional a partir do CSV de frequências | **declarado por Juan**; compatível com o verificado (Seção 9) |
| Não há saídas nem logs históricos que permitam verificar a execução original | **declarado por Juan** |
| O notebook recebido em 12/10/2026 como o da regeneração recente é idêntico byte a byte ao do pacote | **verificado** (Seção 15.1) |
| Esse notebook foi o código que gerou os CSVs regenerados | **declarado por Juan; não comprovado**, porque o notebook não tem saídas nem execução registrada |
| Os filtros são ajustados antes e fora da `GridSearchCV` interna | **verificado** no código; limitação metodológica (Seção 7.1) |
| A divergência entre a regeneração recente e a nossa reprodução decorre de versões de bibliotecas | hipótese (H1) |
| A execução histórica usou as mesmas entradas e produziu as mesmas saídas que a regeneração recente | **não verificável** com o material disponível |
| Houve erro na seleção original, na regeneração ou na nossa reprodução | **sem evidência** — não afirmado |

## 12. Perguntas a Juan

Situação após os esclarecimentos da Seção 15:

| # | Pergunta | Situação |
|---|---|---|
| 1 | **Proveniência dos CSVs:** execução histórica ou regeneração? | **respondida:** regeneração recente (Seção 15, item 1). A data e o sistema operacional da regeneração não foram informados. |
| 2 | **Versões** da regeneração e da execução histórica | **respondida:** regeneração em Python 3.13.5, NumPy 2.3.5, pandas 2.2.3 e scikit-learn 1.8.0; versões históricas desconhecidas. As versões de SciPy e joblib da regeneração não foram informadas. |
| 3 | **Formato da lista final:** como foi gerado o CSV de quatro colunas | **respondida em parte:** processamento adicional a partir do CSV de frequências. O script desse processamento não foi recebido. |
| 4 | **Notebook com saídas** da execução histórica | **respondida:** não existem saídas nem logs históricos disponíveis |
| 5 | **Entradas no Colab:** os arquivos históricos eram os do pacote? | **sem resposta possível:** sem registro histórico (item 4) |
| 6 | **Código exato da regeneração recente** (notebook ou script) | **recebimento resolvido; proveniência não comprovada.** O notebook chegou em 12/10/2026 e é idêntico ao do pacote (Seção 15.1). Sem saídas, ele não comprova a ligação com os CSVs. O script do processamento adicional que gerou o CSV de quatro colunas não foi recebido. |
| 7 | **Partições** da regeneração recente: registro da composição dos folds, por exemplo um hash da lista ordenada de `patient_id` de cada fold, só para comparação interna | **pendente:** ainda não solicitado formalmente |
| 8 | **Warnings** da execução histórica | **respondida:** sem logs históricos (item 4). Para a regeneração recente, não informado. |

## 13. Limitações e rastreabilidade

### 13.1 Limitações

1. A comparação é estática. A hipótese H1 só seria testada reexecutando a seleção nas versões do pacote, o que não foi feito e depende de autorização.
2. O pacote não permite reconstruir a composição dos folds, só as contagens.
3. Nada no pacote prova a data, o ambiente ou a origem da execução que gerou a conclusão do notebook. Juan declarou que não há saídas nem logs históricos, então **a execução histórica não pode ser verificada** com o material disponível.
4. O ZIP e os arquivos extraídos não estão versionados. Este documento registra seus hashes (Seção 3).
5. Para ler os CSVs, usamos pandas no `.venv` do projeto, sem executar código do pacote.
6. Data, ambiente e versões da regeneração recente são **declarados**, e não verificados de forma independente. O notebook declarado como o da regeneração foi recebido, mas sem saídas. Por isso, a ligação entre ele e os CSVs **não está comprovada**.
7. O procedimento original ajusta os filtros fora da CV interna (Seção 7.1). Essa limitação metodológica afeta igualmente as três execuções e não explica a divergência entre elas.

### 13.2 Artefatos desta comparação

Todos estão fora do repositório, em diretório temporário da sessão:

| Artefato | SHA-256 |
|---|---|
| Script de inventário do ZIP | `b8800e058ca0c759f25f044c7cc18fea8e85f04ca9048e7637d2a8e9614aa36f` |
| Script de consistência e comparação de frequências e folds | `3e8a2240c488b7416f3b18a11ee5599559422020c2f46277bf19d84992a0572a` |
| Script de comparação dos notebooks | `a7898953990284f1a6fe733716ae08884d5781429ce8c9e8efb40ad5c616362e` |
| Saída da consistência | `3be815a430a5adf694657153b75e33f3ac0bc3d8550a19f21bbc995e98dc4f34` |
| Saída da comparação dos notebooks | `6e0de33deaad214270dcbed68f8ce51a68bc8b1d2ad1573ddfb5ad408bc639eb` |

Os CSVs da nossa execução B usados na comparação mantêm os hashes da Seção 15 da auditoria. Nenhum desses artefatos contém identificadores de paciente ou de nódulo.

## 14. Situação da decisão

- **A lista definitiva de features continua pendente** de decisão do grupo (protocolo, Seção 7.1, item 1, e pendência P3). Este documento não escolhe lista.
- A decisão **não pode** se basear em desempenho em validação ou teste. As respostas da Seção 12 são evidência de procedimento e podem apoiá-la.
- M2, M3 e M4 continuam sem avaliação.
- A auditoria foi revisada para incorporar este registro e os esclarecimentos de Juan (auditoria, Seção 16).

### 14.1 Recomendação de Juan — proposta, não decisão

Juan recomenda usar a **nossa reprodução controlada** como principal evidência reproduzível para fundamentar a seleção do Cartão 4 (Seção 15, item 6). Na prática, isso significa propor como lista fixa a lista reproduzida no nosso ambiente: `Sphericity`, `SurfaceVolumeRatio`, `Elongation`, `90Percentile`, `RootMeanSquared`, `Imc2`, `Imc1` e `ClusterShade`.

Ela fica registrada como **proposta científica para discussão do grupo**, e **não como decisão aprovada**. Pontos para essa discussão:

1. **Conflito com a regra atual do protocolo.** A Seção 7.1, item 1, do protocolo diz que, em caso de divergência, a lista declarada **permanece** como conjunto fixo e a reproduzida entra apenas como sensibilidade. Adotar a lista reproduzida exige **decisão do grupo (P3) e revisão formal do protocolo**, com data e justificativa, antes de qualquer avaliação de M2, M3 e M4.
2. **O que sustenta a proposta.** Entre as três execuções, a nossa é a única com entradas, versões, código, saídas e hashes registrados, e com determinismo verificado. A execução histórica não tem registro. Na regeneração recente, o notebook foi recebido, mas a ligação dele com os CSVs não está comprovada.
3. **O que limita a proposta:**
   - a lista reproduzida também depende do ambiente: com as mesmas entradas, a regeneração recente devolveu a lista declarada;
   - `ClusterShade` atinge exatamente o limiar de 3/5 (auditoria, Seção 7.1);
   - nenhuma das duas listas é robusta a mudanças de partição ou de ambiente, pelo que se observou.
4. **Condição para a reprodutibilidade invocada.** Hoje, os scripts e saídas da nossa reprodução estão fora do repositório (auditoria, Seção 14, item 1). Para servir como evidência reproduzível, eles precisam ser preservados de forma definitiva.
5. **Critério admissível.** A escolha só pode usar evidência de procedimento — rastreabilidade, reprodutibilidade, estabilidade no treino — e critérios fixados antes de qualquer resultado. **Nunca** desempenho em validação ou teste.

## 15. Esclarecimentos de Juan

Recebidos em 11/10/2026, depois da primeira versão deste documento. São **declarações** do responsável pela seleção original. A coluna "Verificação independente" diz o que este registro conseguiu conferir.

| # | Declaração de Juan | Verificação independente |
|---|---|---|
| 1 | Os CSVs do ZIP foram regenerados recentemente e não são os arquivos históricos preservados da execução original do Colab | compatível com o verificado: o arquivo de versões fala em "regeneração", as datas internas do ZIP são de 11/10/2026 e a lista tem formato diferente do da exportação (Seções 3, 8 e 9). A data e o ambiente da regeneração não foram verificados. |
| 2 | A regeneração usou Python 3.13.5, NumPy 2.3.5, pandas 2.2.3 e scikit-learn 1.8.0 | coincide com `versoes_ambiente_atual.txt` (Seção 8); não verificável de forma independente |
| 3 | As versões usadas na execução histórica do Colab são desconhecidas | coerente com os notebooks sem saídas e sem versão em `language_info` (Seção 7) |
| 4 | O `features_selecionadas_lasso.csv` de quatro colunas recebeu processamento adicional a partir do CSV de frequências e não é a exportação direta do notebook | compatível com o verificado: conjunto, valores e ordem derivam da tabela de frequências (Seção 9) |
| 5 | Não existem saídas nem logs históricos que permitam verificar a execução original | coerente com o material recebido; nenhum arquivo do pacote é saída histórica |
| 6 | Recomenda nossa reprodução controlada como principal evidência reproduzível para fundamentar a seleção do Cartão 4 | não é fato verificável; registrada como **proposta** (Seção 14.1) |
| 7 | Foi solicitado a enviar o código ou notebook exato usado na regeneração recente | **recebido em 12/10/2026** (Seção 15.1) |

Os esclarecimentos **não alteram** nenhum resultado verificado das Seções 3 a 9. Eles mudam a interpretação desses resultados: o pacote passa a ser evidência sobre a regeneração recente, e não sobre a execução histórica (Seção 10).

### 15.1 Notebook da regeneração recente — recebido em 12/10/2026

| Item | Resultado |
|---|---|
| Arquivo recebido | `S4_Selecao_Features_Regeneracao_Recente.ipynb`, 19 899 bytes |
| SHA-256 | `3a5e8e63c4b47ff8144ace04d63e6b4e7cb16e38e41ba045701d7eaa4a7f9835` |
| Comparação com o notebook do pacote | **idêntico byte a byte** a `S4_Selecao_Features_Execucao_FINAL_SPLIT_OFICIAL.ipynb`, conferido por hash e por `cmp` (**verificado**) |
| Conteúdo | 7 células de código, sem saídas e sem `execution_count` (Seção 7) |
| Comparação com a cópia auditada | mesma lógica de código; muda só um comentário (Seção 7) |

**O que fica resolvido:** o recebimento do código declarado como o da regeneração recente.

**O que continua aberto:** a comprovação da **proveniência exata das saídas**. Um notebook sem saídas, enviado depois da regeneração, não prova que foi ele que produziu os CSVs. O script do processamento adicional do CSV de quatro colunas (Seção 9) também não foi recebido.

Essas lacunas não alteram nenhum resultado verificado. Elas limitam apenas o que o pacote permite afirmar sobre a origem dos CSVs.

---

_Este documento registra a comparação estática entre o pacote enviado por Juan e a reprodução independente da seleção de features. Ele não altera o protocolo, a auditoria nem a lista de features._
