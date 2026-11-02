# Emenda 1 ao Protocolo de Valor Incremental — Auditoria do Teste e Congelamento

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Card relacionado:** S6 [P0] — Auditoria do teste e congelamento do protocolo final
**Protocolo emendado:** [`protocolo_valor_incremental.md`](protocolo_valor_incremental.md), commit `9e9b3bd`
**Responsável:** Samara Ruiz · **Revisor:** Rafael Lazarin Lobo de Almeida
**Registro:** 30/10/2026 (revisão 1) · **Revisão 2:** 31/10/2026, após o comentário de revisão do PR #10
**Prazo do card:** 31/10/2026, 23h59 (janela original 27–28/10, prorrogada no cartão)
**Situação:** proposta — **não congelada**. Decisões P1 a P7 registradas na Seção E11; aguardam a confirmação do revisor.

Registro de auditoria completo: `DoD_S6_P0_Auditoria_do_Teste_e_Protocolo_v1.docx` (Drive).
Fundamentação das decisões P1 a P7: `Proposta_Tecnica_Decisoes_P1_P7.docx` (Drive).

---

## Aviso de escopo

Esta emenda **não altera** coorte, alvo, split, lista de características, modelos ou critérios de
interpretação do protocolo. Ela corrige o registro do que já se sabe sobre o conjunto de teste, fixa o
papel dele daqui em diante, registra as decisões pendentes e acrescenta o que faltava para o protocolo
ser congelável. A decisão P3 (E11) **confirma** a regra da Seção 7.1, item 1, sem alterá-la.

Verificações desta emenda classificadas como **verificado** foram reexecutadas em 30 e 31/10/2026
em **Python 3.13 / scikit-learn 1.9.1**, a partir dos arquivos canônicos, com SHA-256 conferido
contra `auditoria_reproducao_juan.md` §3.3. Pela regra da E4, esse ambiente **não é o oficial**: tais
verificações são de **sensibilidade**. A E4 lista o que precisa ser refeito no ambiente oficial.

---

## Registro da revisão (PR #10, comentário de 31/10/2026)

Os seis pontos do revisor foram conferidos contra os arquivos do repositório e do Drive. Todos
procederam, uns por erro de fato no texto da revisão 1, outros por imprecisão conceitual. O revisor
informou que os pontos servem de aviso e de histórico (foram contestados, e não apenas aceitos); mesmo
assim, os erros de fato estão corrigidos abaixo, porque o texto anterior estava errado.

| # | Ponto | Veredito | O que mudou |
|---|---|---|---|
| 1 | E1 — contagem A/B | procedente: erro de soma na revisão 1 | critério de contagem fixado; A = 6 e B = 10; frase sobre "cego" corrigida (E1) |
| 2 | E6 — menor efeito detectável | procedente: conceito errado | precisão do IC separada do menor efeito detectável da diferença pareada (E6, E7) |
| 3 | E4 — Python 3.9.13 × 3.9.23 | procedente: o baseline registra 3.9.23 | duas versões de patch registradas; critério de escolha (E4) |
| 4 | E10 — dados individuais versionados | procedente: 10 arquivos, não 4 | inventário corrigido; exceções históricas declaradas; P8 aberta (E10, E11) |
| 5 | P6 — 353 × 416 | procedente: a recomendação anterior contradizia a E7 | recomendação invertida para 353; a base de 416 não é executada (E11) |
| 6 | Coorte — 19 linhas × 616/607 | procedente: duas grandezas misturadas | decomposição separada; as 19 linhas foram identificadas (E9) |

---

## E1 — Seção 4.1: histórico completo de uso do teste

A tabela atual da Seção 4.1 lista o baseline e o Experimento 13. Está incompleta. Substituir por:

| Análise | Tipo | Partição usada | Evidência |
|---|:--:|---|---|
| Baseline (card 4.7) | A | teste (98 nódulos) | `baseline_metrics.json`; AUC reproduzido |
| Exp. 10 revalidado | A | teste | notebook reexecutado integralmente |
| Exp. 11, 12, 14 | A | teste | DoD da Pasta 13 (documental) |
| Exp. 15, 16 | B | base inteira (CV repetida) | DoD da Pasta 13 (documental) |
| Exp. 17 | A | teste (98 em comum) | DoD do Cartão 1 (documental) |
| Exp. 17.2, 20, 20.1, 20.2 | B | base inteira (607) | DoDs e relatório S4 (documental) |
| Exp. 21, 21.1 | B | base inteira (604) | DoDs (documental) |
| Cartão 2 — seleção | C | **só treino** | código versionado: linhas de validação/teste descartadas |
| Cartão 4 — piloto e CV 10×5 | C | **só treino** (417/291) | `relatorio_cv_repetida_exploratoria.json` |
| Exp. 26.0, 26.1 | B | base elegível inteira (264 na faixa) | **novo**: os 264 contêm 38 nódulos de teste |
| Exp. 26.0.1 a 28 | C | treino + validação (226) | verificado |

**Tipo A** = uso direto da partição de teste. **Tipo B** = base inteira em validação cruzada, em
que os pacientes de teste entram na análise. **Tipo C** = preserva o teste.

**Critério de contagem (corrigido na revisão 2).** Conta-se **uma análise por experimento numerado**;
o baseline conta como uma análise (card 4.7) e o Exp. 10 revalidado como outra. Linhas da tabela que
agrupam vários experimentos (por exemplo, "11, 12, 14") valem um item por experimento. Por esse critério:

- **tipo A: 6** (baseline, 10, 11, 12, 14, 17);
- **tipo B: 10** (15, 16, 17.2, 20, 20.1, 20.2, 21, 21.1, 26.0, 26.1).

A revisão 1 escrevia "cinco" e "sete". O texto de origem listava seis nomes de tipo A e foi mal somado;
o "sete" de tipo B não correspondia a nenhuma contagem da tabela. Contando por linha da tabela, e não
por experimento, seriam quatro de cada tipo; esse critério não é usado, porque esconde quantas vezes o
teste foi de fato tocado. Os números são **mínimos**: onde a evidência é "documental", a classificação
vem do DoD ou do relatório, sem reexecução do notebook. As análises 20.1, 20.2 e 21.1 reanalisam os
mesmos dados dos experimentos de origem; a contagem mede ocorrências, e não olhadas independentes.

Nenhuma correção de multiplicidade foi aplicada entre essas análises, porque cada uma foi planejada
isoladamente. Isso não se corrige depois e fica como limitação declarada.

**O teste não é cego nem para o termo radiômico (correção na revisão 2).** A revisão 1 repetia que o
teste "é cego apenas para modelos com termo radiômico ainda não avaliado". Pela própria tabela acima,
as análises de tipo B (17.2, 20, 20.1, 20.2, 21, 21.1, 26.0, 26.1) avaliaram, por validação cruzada na
base inteira, modelos com termo radiômico (107 características, LASSO dentro das dobras), e essa base
contém os pacientes de teste. O que **não** foi observado é o contraste específico do protocolo:
M2, M3 e M4 com a **lista fixa**, ajustados só no treino oficial e avaliados no teste (a Seção 4.1 do
protocolo registra que nenhum artefato traz desempenho desses modelos em validação ou teste). A redação
da Seção 4.1 (marcador sobre M3 − M0 e M4 − M1) e da Seção 7.5 passa a dizer isso (ver patch).

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
   paciente, com IC por bootstrap pareado de pacientes, **sobre a base definida em P6** (E11).
2. Resultados no teste são **descrição complementar** e levam sempre a declaração: *"conjunto já
   avaliado em análises anteriores do projeto; não constitui evidência independente."*
3. **Nenhum** resultado do teste entra em escolha de modelo, característica, limiar, faixa,
   parâmetro ou critério de parada.
4. Números de desenhos diferentes não se misturam (Quadro 13 do relatório da Sprint 4).

---

## E4 — Seção 12: ambiente oficial

**Ambiente oficial congelado:** scikit-learn **1.3.2**, numpy **1.23.5**, pandas **1.5.3**,
scipy **1.10.1** e **Python 3.9.x**, com o patch registrado em cada artefato. Execução em outro
ambiente é sensibilidade e declara a versão.

**Versões de Python já registradas (correção na revisão 2).** A revisão 1 dizia que o Python 3.9.13
era "o do baseline". Não é. Os registros são:

| Contexto | Python | Fonte |
|---|---|---|
| Baseline da Sprint 3 (conda `pi3-radiomics`) | **3.9.23** | `docs/sprint3/baseline_regressao_logistica.md` (Reprodutibilidade) |
| Sprint 4 (`.venv` local: reprodução da seleção, modelos de referência, `config.py`) | **3.9.13** | `relatorio_reproducao.json`; `docs/guia_reproducao.md` |
| `environment.yml` | `3.9` (sem patch) | `environment.yml` |

Em todos, scikit-learn 1.3.2, numpy 1.23.5, pandas 1.5.3 e scipy 1.10.1.

**Critério:** o ambiente oficial das Etapas 0 a 5 é o da `.venv` (Python **3.9.13**), porque é o único com
versões, código, saídas e hash do commit registrados juntos. O 3.9.23 do baseline fica como ambiente
histórico **equivalente por evidência indireta**: o baseline reproduz em dez casas decimais entre
scikit-learn 1.3.2 (3.9.23) e 1.9.1 (3.13), o que sugere que o patch do Python não pesa para esse
modelo. Isso **não** foi testado isoladamente para os demais modelos. **Decidido em 31/10/2026:** o `environment.yml`
passa a fixar `python=3.9.13`. Execução para o revisor: conferir que o canal conda ainda serve esse patch;
se não servir, o `environment.yml` mantém `3.9` e o registro do patch por artefato passa a ser obrigatório,
com a troca registrada aqui.

**Por que a versão importa (A5).** O projeto já usou pelo menos cinco configurações — 1.3.2 (baseline
e reprodução), 1.6.1 (Colab da autora), 1.9.1 (comparação), 1.8.0 (regeneração de Juan) e uma
desconhecida (execução histórica de Juan). A auditoria 20.2.1 mostrou que a versão altera o gerador de
dobras do `StratifiedGroupKFold`, e o Experimento 20.2 passou de 3/5 para 5/5 sementes por causa disso.

**Como registrar as verificações desta auditoria, feitas fora do ambiente oficial.** Ficam como
**sensibilidade**, com a versão declarada em cada uma. Duas delas ficam como **pendência registrada**, a
refazer no ambiente oficial **antes do relatório final**: a tabela de precisão da E6 (bootstrap, seed 42)
e a contagem das 19 linhas da E9 (feita em Python 3.13 com pylidc 0.2.3 e uma adaptação de compatibilidade
com numpy 2). Elas **não condicionam o congelamento**, porque são descritivas e não decidem modelo,
característica, limiar nem critério; o que delas depende é o texto de limitações do relatório. Já independem de
versão: as contagens de coorte e de partição. Os experimentos exploratórios 27 e 28 (Colab, scikit-learn
1.6.1) ficam fora do congelamento (E12) e declaram o ambiente.

Cada artefato registra: versões, seeds, SHA-256 das entradas, hash do commit e a lista de pacotes
(`conda list --explicit` ou `pip freeze`).

---

## E5 — Seção 8.1: regra de multiplicidade

- Um único contraste primário → sem correção (regra atual, mantida).
- Família fixada de antemão e pequena → **Bonferroni**.
- Família maior → **Holm**.
- A família é declarada **antes** de rodar, no artefato da etapa.

---

## E6 — Seção 9: precisão e menor efeito detectável (reescrita na revisão 2)

A revisão 1 chamava a meia-largura do IC 95% de uma AUC de "menor efeito detectável". São grandezas
diferentes:

- **Precisão da estimativa** = meia-largura do IC 95% de **uma** AUC. Diz quão estável é a AUC de um
  modelo naquela amostra.
- **Menor efeito detectável (MDE) de um contraste** = a menor **diferença pareada** de AUC entre dois
  modelos que o desenho distinguiria do acaso. Depende do erro-padrão da diferença (que cai quando os
  dois modelos são muito correlacionados, como M0 e M3), do nível α e do poder pretendido.

**Regra adotada.** Todo contraste, de validação cruzada, validação ou teste, é relatado com:

1. a diferença Δ e seu IC 95% por bootstrap pareado de pacientes (Seção 8.2);
2. o erro-padrão de Δ, EP(Δ) = desvio-padrão das réplicas bootstrap de Δ;
3. o **MDE** = (z₁₋α/₂ + z_poder) × EP(Δ), com α = 0,05 bilateral e poder = 0,80, isto é, fator
   ≈ 2,80 (aproximação normal). Como a meia-largura do IC de Δ é ≈ 1,96 × EP(Δ), o MDE vale
   ≈ 1,43 vezes essa meia-largura, **se** a distribuição bootstrap for aproximadamente normal.

α e poder são convenções propostas, **sem sustentação nas referências do projeto**; a fórmula é de
estatística básica e precisa de uma referência bibliográfica a ser indicada no relatório final (não
citada aqui para não inventar). O MDE é calculado a partir do EP observado, ou seja, **depois** dos
dados: descreve a sensibilidade do desenho, e não um poder planejado.

Um contraste cujo MDE exceda a diferença discutida (a referência exploratória de 0,02 da Seção 9 ou a
diferença observada) é relatado como **inconclusivo por falta de poder**, e nunca como ausência de efeito.

**O que a tabela da revisão 1 mede, de fato.** Por bootstrap de pacientes (B=2000, seed 42), em Python 3.13
/ scikit-learn 1.9.1 (sensibilidade; a refazer no ambiente oficial):

| Conjunto | n | AUC do diâmetro | IC 95% | Meia-largura (precisão de uma AUC) |
|---|--:|--:|---|--:|
| Teste completo | 98 | 0,9772 | [0,9491; 0,9959] | **0,0234** |
| Teste na faixa 12–30 mm | 38 | 0,8452 | [0,6886; 0,9647] | **0,1381** |

Esses valores **não são MDE** de nenhum contraste. Nenhum MDE foi calculado até aqui, porque os
contrastes M3 − M0 e M4 − M1 ainda não existem.

---

## E7 — Nova Seção 7.6: orçamento de uso do teste

| Item | Valor |
|---|---|
| Aberturas restantes do teste | **uma**, na Etapa 5 |
| O que é calculado nessa abertura | todos os contrastes pré-especificados, de uma vez |
| Reexecução por falha técnica | permitida, com causa documentada, sem alterar modelo/feature/critério |
| Uso dos 38 nódulos da faixa | **descritivo apenas** |

**Justificativa do uso descritivo dos 38 nódulos (revisão 2).** A decisão não depende da meia-largura de
0,1381, que é a precisão de **uma** AUC e pode superestimar a incerteza de uma diferença pareada. Ela
decorre do tamanho da amostra (38 nódulos, 30 pacientes, 22 malignos) e da regra de orçamento: cada
abertura adicional do teste é mais uma olhada num conjunto já consumido (E1).

**Consequência para P6:** análises novas em validação cruzada **não** podem incluir os pacientes de
teste — nem como análise principal, nem como análise de sensibilidade —, ou esta tabela deixa de
valer. Ver P6 na E11.

---

## E8 — Seção 5 / decisão P3: irreprodutibilidade da lista de features

Registrar o achado A4: a lista de 8 features **não é reprodutível entre ambientes**. A reprodução fiel
devolve 6 das 8 declaradas (troca `Skewness` e `Coarseness` por `Imc1` e `ClusterShade`); com as mesmas
entradas, a regeneração recente devolve a lista declarada; a divergência começa na composição dos folds.
A hipótese de versão de biblioteca (H1) continua **aberta**. A decisão entre as listas é **de
procedimento, nunca de desempenho**. Decidida em 31/10/2026 (E11, P3): **mantida a lista declarada**,
com a reproduzida avaliada como sensibilidade.

Verificado: a seleção usou **somente o treino oficial** (417/291) — não há contaminação por teste.
Limitação metodológica mantida em registro: os filtros são ajustados fora da CV interna.

---

## E9 — Seção 11: limitações acrescentadas

1. **19 linhas de extração ausentes, de causa não determinada** (ver decomposição abaixo).
2. **Circularidade medida:** Spearman 0,6828 entre diâmetro e nota mediana de malignidade (N=958).
3. **No teste, o diâmetro isolado (0,9772) supera o baseline de três features de forma (0,9762).**
4. Nenhuma correção de multiplicidade entre as análises históricas que tocaram o teste.

**Cobertura da coorte: duas grandezas distintas (correção na revisão 2).** A revisão 1 escrevia "19 linhas
ausentes (607 de 616)", o que mistura duas coisas. A decomposição documentada em
`docs/sprint3/baseline_regressao_logistica.md`, Seção 2.1, é:

| Etapa | Nódulos | Pacientes |
|---|--:|--:|
| Coorte teórica (regra oficial recomputada sobre 1018 scans) | 616 | 422 |
| Esperado nos 420 scans efetivamente extraídos | 610 | 417 |
| Obtido (base modelável) | **607** | **416** |

- **616 → 610 (−6 nódulos, −5 pacientes): cobertura de scans.** Dois pacientes com nódulo modelável nunca
  foram extraídos (`LIDC-IDRI-0578`, `LIDC-IDRI-0626`); o restante vem da escolha de série em pacientes com
  mais de uma reconstrução. *Fonte: documento do baseline; não reexecutado nesta revisão.*
- **610 → 607 (−3 nódulos, −1 paciente): extração parcial.** Em 14 scans, há menos linhas extraídas do que
  clusters selecionados.

**As 19 linhas são linhas da extração de qualquer classe; só 3 delas eram modeláveis.** Por isso 19 linhas
ausentes correspondem a −3 nódulos modeláveis, e não a −9 (a diferença 616 − 607 = 9 soma as duas causas).

**Verificado em 31/10/2026** (pylidc 0.2.3, banco de anotações, sem DICOM; Python 3.13, ambiente não
oficial): o código versionado `selecionar_clusters` devolve **977** clusters esperados nos 420 scans, contra
**958** linhas em `base_radiomica_oficial_422.csv`; a diferença está em **14 scans** (9 com 1 linha ausente
e 5 com 2), totalizando **19** linhas. Pela mediana de malignidade dessas 19: 14 com
mediana 3,0, 1 com 2,5 e 1 com 3,5 (as 16 são indeterminadas, fora da base modelável); **3 com mediana 2,0**
(classe 0, modeláveis). Os três modeláveis ausentes são de `LIDC-IDRI-0380` (único nódulo modelável do
paciente, que por isso saiu da base), `LIDC-IDRI-0402` e `LIDC-IDRI-0565`. Os diâmetros médios das 19 vão de
4,87 a 14,66 mm; não há padrão simples de posição ou de tamanho.

**Causa:** continua **não determinada**. Não há log de descarte da extração completa nos artefatos. Saber se
foi salvaguarda de intensidade ou falha silenciosa exige reextrair esses 14 scans a partir do DICOM, com
registro do motivo por nódulo. **Efeito conhecido:** −3 nódulos de classe 0 (310 de classe 0 e 297 de classe 1
na base obtida), 0,5% dos 607. Não há evidência de que isso altere os resultados; também não há teste que
mostre o contrário.

---

## E10 — Nova Seção 15: governança de dados (LGPD)

**Correção na revisão 2.** A revisão 1 afirmava que só `patient_split.csv` tem `patient_id` entre os arquivos
versionados e que `reports/` não tem valor de feature por nódulo. Isso estava errado. Inventário de
`reports/` na branch `main` em 31/10/2026 (repositório **público**, conforme a API do GitHub):

| Arquivos | Conteúdo individual |
|---|---|
| `patient_split.csv` | `patient_id` e partição |
| `reports/sprint2/features/piloto_piloto_v2_consenso50.csv` e `.parquet` | 44 nódulos; `patient_id`, notas individuais, 88 colunas `original_*` |
| `reports/sprint2/features/piloto_piloto_v3_consenso50.csv` | 44 nódulos; `patient_id`, mediana, 88 colunas `original_*` |
| `reports/sprint2/historico/v1/piloto_piloto_v1_consenso50.csv` e `.parquet` | 43 nódulos; idem |
| `reports/sprint2/historico/v3_oficial/piloto_piloto_v3_oficial_consenso50.csv` e `.parquet` | 44 nódulos; idem |
| `descartes_piloto_v1/v2/v3_oficial_consenso50.csv` (3 arquivos) | `nodulo_id`, que embute o `patient_id` (ex.: `LIDC-IDRI-0006_N03`) |

São **10 arquivos** além do `patient_split.csv`: os 4 CSVs apontados na revisão, os 3 `.parquet` de mesmo
conteúdo e os 3 arquivos de descarte. O inventário cobre `reports/`; `docs/` e `notebooks/` não foram
varridos.

Os identificadores são os do LIDC-IDRI, conjunto público e já desidentificado. Se isso torna ou não essas
linhas "dado pessoal" para a LGPD é questão jurídica que **não** é afirmada aqui. A política do projeto
é tratar valor por nódulo e identificador de paciente como restritos, e esses arquivos a contrariam.

**Política mantida para tudo o que for novo:**

- `data/` continua não versionado;
- `.tar` de DICOM e checkpoints não vão para o Drive compartilhado;
- arquivos com `patient_id` ou valor de feature por nódulo não entram em repositório público, e a mesma
  regra vale para anexo de e-mail e para a entrega da banca;
- antes de qualquer commit de `reports/`, varredura automática por `LIDC-IDRI-\d{4}`.

**Arquivos históricos:** ficam declarados como **exceções históricas** até a decisão P8 (E11). Removê-los
do `main` quebra duas rotinas de validação que os leem (`scripts/validate_nodule_ids.py` e
`scripts/validate_selection_masks.py`), e eles permanecem no histórico do Git mesmo depois de removidos.

---

## E11 — Seção 13: decisões pendentes, atualizada

**Como as decisões foram tomadas.** Em 31/10/2026, pela responsável, item a item, com as alternativas e os
impactos à vista. Os demais integrantes do grupo foram consultados havia duas ou três semanas e não
responderam. As decisões abaixo foram tomadas pela responsável (Samara) e estão **aguardando
a confirmação do revisor** (Rafael). O silêncio do grupo fica registrado; **não equivale a aprovação** do
grupo. Fundamentação, alternativas e impactos de cada uma: `Proposta_Tecnica_Decisoes_P1_P7.docx`.

| # | Decisão | Situação | Prazo |
|---|---|---|---|
| P1 | Permutações: K e esquema | **decidida**: K = 100; permutação por paciente, em blocos; seeds `10000 + k` | antes da Etapa 1 |
| P2 | Parâmetros da spline | **decidida**: manter os atuais do protocolo | antes da Etapa 1 |
| P3 | Lista fixa de features | **decidida**: **manter a lista declarada** da Seção 5.1; a lista reproduzida no ambiente oficial (`Imc1` e `ClusterShade` no lugar de `Skewness` e `Coarseness`) entra como sensibilidade, com os mesmos contrastes. Confirma a regra da Seção 7.1, item 1, sem alterá-la | antes da Etapa 3 |
| P4 | Divergência 0,9778 × 0,9762 | **ENCERRADA** por esta emenda (E2) | — |
| P5 | Versionar registros dos Exp. 10 e 13 e o notebook de seleção | **decidida**: versionar com saídas removidas, manifesto de SHA-256 e números agregados | antes da Etapa 4 |
| P6 | Base da evidência principal | **decidida**: **353 pacientes** (treino + validação), em todas as análises novas; a base de 416 **não** é executada | antes da Etapa 2 |
| P7 | Regularização | **decidida**: C = 1 (L2) para M0–M4; C = 0,1 como sensibilidade; linhas 26–28 mantêm C = 0,1 como análise própria | antes da Etapa 2 |
| P9 | **Nova.** Hiperparâmetros dos classificadores do benchmark (Random Forest, XGBoost, SVM) | **decidida**: declarados pelo revisor em anexo datado **antes** de treinar — grade, validação interna, sementes e ambiente —, com uso restrito a treino e validação. Sem essa declaração prévia, o resultado é exploratório e não entra na comparação final | antes do treinamento |
| P8 | **Nova.** Destino dos 10 arquivos históricos com dado individual (E10) | **decidida em parte**: os 10 arquivos ficam declarados como **exceção histórica**, em 31/10/2026, e a política nova vale para tudo o que for criado daqui em diante. O destino final — retirar do `main` com adaptação das duas rotinas de validação, ou manter — fica a decidir | antes da entrega final |

**Ressalvas registradas por decisão:**

- **P3** mantém a regra que o protocolo pré-especificou em 10/10/2026 justamente para o caso de a reprodução
  divergir. Inverter essa regra depois de ver qual lista a reprodução devolveu seria, em forma, uma mudança
  pós-hoc, ainda que o critério invocado fosse de procedimento. O custo da decisão fica declarado: a lista
  principal **não tem saída, log nem versão registrada** da execução que a produziu, e a reprodução no
  ambiente oficial devolve 6 das 8. A recomendação em contrário do autor da seleção original fica registrada
  e não foi seguida; o motivo é a regra pré-especificada, e não o mérito da reprodução. Nenhuma das quatro
  características divergentes é estável: `Skewness` e `Coarseness` aparecem em 1 de 5 dobras na reprodução,
  `Coarseness` com coeficiente próximo de zero; `Imc1` em 4 de 5 e `ClusterShade` exatamente no limiar de
  3 em 5, com coeficiente próximo de zero em duas dessas três. A lista reproduzida é avaliada como
  sensibilidade, com os mesmos contrastes.
- **P6** inverte a recomendação da revisão 1 (416 pacientes). A revisão 1 justificou os 416 por poder
  estatístico, mas usar o teste em nova validação cruzada gastaria a "abertura única" da E7. O teste já foi
  usado em 6 análises de tipo A e 10 de tipo B (E1): **a escolha dos 353 não restaura a independência do teste**,
  apenas evita mais uma olhada. A escolha é de procedimento, tomada antes de qualquer resultado das novas
  análises; ela **não** se baseia em desempenho observado. Custo: 509 nódulos em vez de 607, isto é, 16%
  menos nódulos do que a base inteira. A revisão 1 previa rodar a mesma análise nos 416 como sensibilidade;
  isso foi **retirado**, porque seria mais um uso de tipo B dos pacientes de teste e tornaria falsa a tabela
  da E7. O ganho de precisão que os 416 dariam fica **não medido**; se o grupo quiser estimá-lo, o caminho
  que não toca o teste é medir a largura do IC em subamostras dos próprios 353 e observar como ela varia
  com o número de pacientes.
- **P7** preserva o invariante do M1 (Seção 6.2): C = 1 reproduz o baseline canônico (0,9762). A decisão vale
  só para a regressão logística; os demais classificadores caem na P9.
- **P9** existe porque o protocolo especifica apenas a regressão logística (Seção 6.1). O cartão do benchmark
  vence em 03/11/2026 e depende de P3, P6, P7 e desta declaração prévia.

---

## E12 — Seção 14: fora do escopo

O **Experimento 28** (anel perinodular de 5 mm, pré-registro de 30/10/2026) é **exploratório**:
não é análise confirmatória, não entra no congelamento e não toca o conjunto de teste.

---

## Congelamento

O protocolo é considerado congelado quando:

1. esta emenda for aprovada pelo revisor, com data, **e a ausência de resposta do grupo estiver registrada**
   (E11). A aprovação formal do grupo **não** é condição: ela foi solicitada, não veio, e o congelamento é
   feito por responsável e revisor. Isso é uma limitação de governança, não de método, e deve ser dita à
   banca nesses termos;
2. responsável e revisor estiverem registrados no card (feito: Samara / Rafael);
3. o commit que contém a emenda aprovada for identificado e seu hash registrado.

As duas verificações da E4 marcadas como pendência **não** condicionam o congelamento; têm prazo próprio,
antes do relatório final.

Até lá permanecem **bloqueados**: execução e seleção confirmatória de modelos, e qualquer relato
do teste como evidência independente.

---

_"O primeiro princípio é que você não deve enganar a si mesmo — e você é a pessoa mais fácil de enganar." — Richard P. Feynman_
