# Emenda 1 ao Protocolo de Valor Incremental — Auditoria do Teste e Congelamento

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Card relacionado:** S6 [P0] — Auditoria do teste e congelamento do protocolo final
**Protocolo emendado:** [`protocolo_valor_incremental.md`](protocolo_valor_incremental.md), commit `9e9b3bd`
**Responsável:** Samara Ruiz · **Revisor:** Rafael Lazarin Lobo de Almeida
**Registro:** 30/10/2026 · **Prazo do card:** 31/10/2026, 23h59 (janela original 27–28/10, prorrogada no cartão)
**Situação:** proposta — **não congelada**. Depende de aprovação do grupo.

Registro de auditoria completo: `DoD_S6_P0_Auditoria_do_Teste_e_Protocolo_v1.docx` (Drive).

---

## Aviso de escopo

Esta emenda **não altera** coorte, alvo, split, lista de features, modelos ou critérios de
interpretação do protocolo. Ela corrige o registro do que já se sabe sobre o conjunto de teste,
fixa o papel dele daqui em diante e acrescenta o que faltava para o protocolo ser congelável.

Verificações desta emenda classificadas como **verificado** foram reexecutadas em 30/10/2026
(Python 3.13, scikit-learn 1.9.1) a partir dos arquivos canônicos, com SHA-256 conferido contra
`auditoria_reproducao_juan.md` §3.3.

---

## E1 — Seção 4.1: histórico completo de uso do teste

A tabela atual da Seção 4.1 lista o baseline e o Experimento 13. Está incompleta. Substituir por:

| Análise | Tipo | Partição usada | Evidência |
|---|:--:|---|---|
| Baseline (card 4.7) | A | teste (98 nódulos) | `baseline_metrics.json`; AUC reproduzido |
| Exp. 10 revalidado | A | teste | notebook reexecutado integralmente |
| Exp. 11, 12, 14 | A | teste | DoD da Pasta 13 |
| Exp. 15, 16 | B | base inteira (CV repetida) | DoD da Pasta 13 |
| Exp. 17 | A | teste (98 em comum) | DoD do Cartão 1 |
| Exp. 17.2, 20, 20.1, 20.2 | B | base inteira (607) | DoDs e relatório S4 |
| Exp. 21, 21.1 | B | base inteira (604) | DoDs |
| Cartão 2 — seleção | C | **só treino** | código versionado: linhas de validação/teste descartadas |
| Cartão 4 — piloto e CV 10×5 | C | **só treino** (417/291) | `relatorio_cv_repetida_exploratoria.json` |
| Exp. 26.0, 26.1 | B | base elegível inteira (264 na faixa) | **novo**: os 264 contêm 38 nódulos de teste |
| Exp. 26.0.1 a 28 | C | treino + validação (226) | verificado |

**Tipo A** = uso direto da partição de teste. **Tipo B** = base inteira em validação cruzada, em
que os pacientes de teste entram na análise. **Tipo C** = preserva o teste.

Acrescentar: o teste foi avaliado em **cinco** análises de tipo A e participou de **sete** de tipo
B; nenhuma correção de multiplicidade foi aplicada entre elas, porque cada uma foi planejada
isoladamente. Isso não se corrige depois e fica como limitação declarada.

**Verificado (novo):** a faixa de 12 a <30 mm tem 264 nódulos / 217 pacientes / 157 malignos;
treino+validação = **226 / 187 / 135** (coorte dos Exp. 26.0.1 a 28, exata); teste = **38 / 30 / 22**.
AUC bruto do diâmetro nos 264 = **0,8004**. Logo, 26.0 e 26.1 são tipo B.

---

## E2 — Decisão P4: encerrada

A divergência M1 = 0,9778 (Exp. 10 revalidado) × 0,9762 (baseline canônico) **está explicada**.

O notebook `PI3_G4_10_Testes_Validade_Baseline_REVALIDADO.ipynb` tem dois desvios em relação a
`scripts/run_baseline.py`:

1. usa `original_shape_VoxelVolume` no lugar de `original_shape_MeshVolume`;
2. ajusta `LogisticRegression` **sem** `StandardScaler` (o baseline usa `Pipeline([scaler, logreg])`).

Reexecutado com esses dois desvios, o notebook devolve **exatamente** os números do DoD da Pasta 13:

| Medida | Reexecução | DoD da Pasta 13 |
|---|---:|---:|
| AUC teste, 3 features | 0,9778 | 0,9778 |
| AUC teste, só diâmetro | 0,9772 | 0,9772 |
| Rótulos embaralhados — média / mín / máx | 0,2945 / 0,0196 / 0,9758 | 0,2945 / 0,0196 / 0,9758 |
| Spearman diâmetro × nota (N=958) | 0,6828 (p = 1,86e-132) | 0,6828 (p = 1,86e-132) |

**Decisão:** o valor canônico do M1 é **0,9762**, e só ele. Corrigir o DoD da Pasta 13 e a Seção 4.1.
P4 fica encerrada.

**Verificado em paralelo (A8):** o baseline canônico foi reproduzido a partir só dos três CSVs
canônicos, em scikit-learn **1.9.1**, com AUC de teste `0,9761705686` e de validação
`0,9439962031` — iguais em dez casas ao `baseline_metrics.json`, gerado em 1.3.2. O baseline é
reprodutível por terceiro e robusto a versão de biblioteca.

---

## E3 — Nova Seção 4.2: papel do conjunto de teste

1. A **evidência principal** de qualquer comparação do projeto é a validação cruzada agrupada por
   paciente, com IC por bootstrap pareado de pacientes.
2. Resultados no teste são **descrição complementar** e levam sempre a declaração: *"conjunto já
   avaliado em análises anteriores do projeto; não constitui evidência independente."*
3. **Nenhum** resultado do teste entra em escolha de modelo, característica, limiar, faixa,
   parâmetro ou critério de parada.
4. Números de desenhos diferentes não se misturam (Quadro 13 do relatório da Sprint 4).

---

## E4 — Seção 12: ambiente oficial

Ambiente oficial congelado: **Python 3.9.13, scikit-learn 1.3.2, numpy 1.23.5, pandas 1.5.3,
scipy 1.10.1** — o do baseline e o da reprodução auditada. Execução em outro ambiente é
sensibilidade e declara a versão.

Motivo (A5): o projeto já usou pelo menos cinco configurações — 1.3.2 (baseline e reprodução),
1.6.1 (Colab oficial da autora), 1.9.1 (comparação), 1.8.0 (regeneração de Juan) e uma
desconhecida (execução histórica de Juan). A auditoria 20.2.1 mostrou que a versão altera o
gerador de dobras do `StratifiedGroupKFold`, e o Experimento 20.2 passou de 3/5 para 5/5 sementes
por causa disso.

Cada artefato registra: versões, seeds, SHA-256 das entradas e hash do commit.

---

## E5 — Seção 8.1: regra de multiplicidade

- Um único contraste primário → sem correção (regra atual, mantida).
- Família fixada de antemão e pequena → **Bonferroni**.
- Família maior → **Holm**.
- A família é declarada **antes** de rodar, no artefato da etapa.

---

## E6 — Seção 9: menor efeito detectável obrigatório

Todo resultado, de validação ou de teste, é relatado junto do **menor efeito detectável** do
próprio contraste, estimado como meia-largura do IC. Um contraste cuja meia-largura exceda a
diferença discutida é relatado como **inconclusivo por falta de poder** — nunca como ausência
de efeito.

**Verificado (A7)**, por bootstrap de pacientes (B=2000, seed 42):

| Conjunto | n | AUC do diâmetro | IC 95% | Meia-largura |
|---|--:|--:|---|--:|
| Teste completo | 98 | 0,9772 | [0,9491; 0,9959] | **0,0234** |
| Teste na faixa 12–30 mm | 38 | 0,8452 | [0,6886; 0,9647] | **0,1381** |

---

## E7 — Nova Seção 7.6: orçamento de uso do teste

| Item | Valor |
|---|---|
| Aberturas restantes do teste | **uma**, na Etapa 5 |
| O que é calculado nessa abertura | todos os contrastes pré-especificados, de uma vez |
| Reexecução por falha técnica | permitida, com causa documentada, sem alterar modelo/feature/critério |
| Uso dos 38 nódulos da faixa | **descritivo apenas** — meia-largura 0,1381 torna invisível qualquer diferença da ordem discutida |

---

## E8 — Seção 5 / decisão P3: irreprodutibilidade da lista de features

Registrar o achado A4: a lista de 8 features **não é reprodutível entre ambientes**. A reprodução
fiel devolve 6 das 8 declaradas (troca `Skewness` e `Coarseness` por `Imc1` e `ClusterShade`); com
as mesmas entradas, a regeneração recente devolve a lista declarada; a divergência começa na
composição dos folds. A decisão entre as listas é **de procedimento, nunca de desempenho**, e
exige data e justificativa.

Verificado: a seleção usou **somente o treino oficial** (417/291) — não há contaminação por teste.
Limitação metodológica mantida em registro: os filtros são ajustados fora da CV interna.

---

## E9 — Seção 11: limitações acrescentadas

1. **19 linhas de extração ausentes**, sem explicação (607 de 616 nódulos da coorte teórica).
2. **Circularidade medida:** Spearman 0,6828 entre diâmetro e nota mediana de malignidade (N=958).
3. **No teste, o diâmetro isolado (0,9772) supera o baseline de três features de forma (0,9762).**
4. Nenhuma correção de multiplicidade entre as análises históricas que tocaram o teste.

---

## E10 — Nova Seção 15: governança de dados (LGPD)

- Arquivos com `patient_id`: `patient_split.csv`, `base_radiomica_oficial_422.csv`,
  `vies_malignidade_mascara_pequena.csv`, tabela de modelagem. Só o primeiro é versionado, e
  contém apenas paciente e partição.
- `data/` continua não versionado; artefatos de `reports/` não contêm valor de feature por nódulo.
- `.tar` de DICOM e checkpoints não vão para o Drive compartilhado.
- A mesma regra vale para anexo de e-mail e para a entrega da banca.

---

## E11 — Seção 13: decisões pendentes, atualizada

| # | Decisão | Situação | Prazo |
|---|---|---|---|
| P1 | Permutações: K e esquema | aberta — recomendação: por paciente | antes da Etapa 1 |
| P2 | Parâmetros da spline | aberta — recomendação: confirmar os atuais | antes da Etapa 1 |
| P3 | Lista fixa de features | aberta — ver E8 | antes da Etapa 3 |
| P4 | Divergência 0,9778 × 0,9762 | **ENCERRADA** por esta emenda (E2) | — |
| P5 | Versionar registros dos Exp. 10 e 13 e o notebook de seleção | aberta — **fazer agora**: A2 só foi possível porque o notebook estava no Drive | antes da Etapa 4 |
| P6 | **Nova.** Base da evidência principal: 416 pacientes ou só os 353 de treino+validação | recomendação: **416** | antes da Etapa 2 |
| P7 | **Nova.** Configuração única de regularização na comparação final | recomendação: a do protocolo para M0–M4; C=0,1 das linhas 26–28 fica como análise própria da faixa | antes da Etapa 2 |

---

## E12 — Seção 14: fora do escopo

O **Experimento 28** (anel perinodular de 5 mm, pré-registro de 30/10/2026) é **exploratório**:
não é análise confirmatória, não entra no congelamento e não toca o conjunto de teste.

---

## Congelamento

O protocolo é considerado congelado quando:

1. esta emenda for aprovada pelo grupo, com data;
2. responsável e revisor estiverem registrados no card (feito: Samara / Rafael);
3. o commit que contém a emenda aprovada for identificado e seu hash registrado.

Até lá permanecem **bloqueados**: execução e seleção confirmatória de modelos, e qualquer relato
do teste como evidência independente.

---

_"O primeiro princípio é que você não deve enganar a si mesmo — e você é a pessoa mais fácil de enganar." — Richard P. Feynman_
