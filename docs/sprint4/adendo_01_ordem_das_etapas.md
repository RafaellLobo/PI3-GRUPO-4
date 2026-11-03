# Adendo 1 ao Protocolo de Valor Incremental — ordem das Etapas 1 a 3 e registro das confirmações da equipe

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Protocolo adendado:** [`protocolo_valor_incremental.md`](protocolo_valor_incremental.md) (Seção 7: "as etapas são sequenciais") e [`emenda_01_auditoria_teste.md`](emenda_01_auditoria_teste.md)
**Data do registro:** 03/11/2026, **anterior a qualquer execução oficial das Etapas 0 (item 2) e 3**
**Origem:** [`adendo_reordenacao_s4_proposta.md`](../sprint6/adendo_reordenacao_s4_proposta.md) (proposta de 03/11/2026, branch `feat/s6-benchmark-executor`)
**Situação:** **registro da confirmação da equipe**, comunicada ao revisor (`RafaellLobo`) e transcrita aqui por solicitação dele em 03/11/2026.

---

## 1. Natureza deste registro

- A equipe respondeu e confirmou os cinco encaminhamentos da Seção 2. A confirmação chegou ao revisor e foi **relatada por ele ao agente que redigiu este documento**. O repositório **não contém** mensagem, ata ou assinatura individual dos membros: o que se registra aqui é o que o revisor transmitiu, e a fonte primária está fora do Git.
- Este documento **não altera** coorte, alvo, split, lista de oito características, modelos, parâmetros, critérios de interpretação (Seção 9) nem a régua de leitura. Altera apenas **a ordem de execução** das Etapas 1 a 3 (Seção 3).
- Ele **não é** os registros formais `APROVACAO-DECISAO`, que são lavrados em commits próprios na Seção 13 do protocolo (Seção 4 abaixo) e exigidos pelas guardas.

## 2. Confirmações da equipe (03/11/2026)

| # | Encaminhamento confirmado | Efeito neste repositório |
|---|---|---|
| 1 | A Etapa 3 pode **preceder** as Etapas 1 e 2, que **continuam obrigatórias** e ocorrem depois, mediante registro datado anterior à execução | Seção 3 deste adendo |
| 2 | **P1, P2 e P3** estão confirmadas conforme o protocolo e a Emenda 1 (E11), **preservando as ressalvas documentadas da P3** (recomendação contrária do autor da seleção original, registrada e não seguida; ausência de resposta do grupo, registrada) | Registros formais na Seção 13 do protocolo (Seção 4) |
| 3 | **Circularidade (Seção 10.3) e máscaras reduzidas (Seção 10.4)** serão tratadas **posteriormente, em atividade própria**; a circularidade é registrada como **limitação** (Seção 11.1, item 2) | Seção 5 |
| 4 | **P4 está encerrada**, com valor canônico **0,9762** (M1 do baseline da Sprint 3; Emenda 1, E2) | Registro formal na Seção 13 (Seção 4) |
| 5 | Há concordância com **D1–D12** do pré-registro P9. **D2** foi confirmada pela responsável pelo Exp. 26.0.1, com evidências em `piso_26_0_1.py` e `26_0_1_resumo.json`. O experimento usa o intervalo fechado [12, 30]; o P9 usa [12, 30); as contagens observadas coincidem | Seção 6 e Seção 19 do pré-registro P9 (PR #11) |

## 3. Reordenação (itens 1 do quadro acima)

**Regra nova, em substituição à sequência literal da Seção 7 apenas quanto à ordem:**

1. A Etapa 3 (validação oficial, lista fixa da Seção 5.1, M0–M4 e as sensibilidades da Seção 7.5/E7) pode ser executada **antes** das Etapas 1 e 2 oficiais.
2. As Etapas 0, 1 e 2 oficiais **continuam obrigatórias**, com todos os artefatos, análises e limitações do protocolo (CV aninhada 10 × 5, controle por permutação P1, spline P2, CV com lista fixa e sensibilidades P3/P7). Elas são executadas **depois da Etapa 3 e antes da Etapa 4 (congelamento) e da Etapa 5 (teste)**.
3. As decisões P1, P2 e P3 devem estar **registradas nas guardas antes da Etapa 3**, como já exigem.
4. Os resultados das Etapas 1 e 2 **não alteram** modelo, lista, critérios nem a Etapa 3 já registrada. Nenhuma leitura do teste ocorre fora da Etapa 5, e a Etapa 5 permanece descritiva (E3).

**Limitações a declarar nos relatórios (da proposta):**

1. A Etapa 1 verifica implementação, convergência, invariantes e custo. Executada depois, esse controle técnico prévio deixa de existir para o código compartilhado com a Etapa 3. Mitigações: o invariante do M1, os testes sintéticos e o item 2 da Etapa 0.
2. Se as Etapas 1 ou 2 revelarem **erro técnico** em código usado pela Etapa 3, esta deve ser **refeita e registrada de novo**, e o registro `S4_E3_REGISTRADA` perde a validade para o benchmark S6 já executado.
3. A validação é lida pelo benchmark S6 depois do registro da Etapa 3 (E7), antes de as Etapas 1 e 2 estarem concluídas. A ordem alterada é uma limitação de governança a declarar à banca.

## 4. Registros formais (o que este documento não faz)

| Decisão | Confirmação da equipe | Registro formal `APROVACAO-DECISAO` | Estado em `config.DECISOES` |
|---|---|---|---|
| P1 | confirmada (03/11/2026) | **a lavrar** em commit próprio, depois deste | a atualizar depois do commit de aprovação |
| P2 | confirmada | idem | idem |
| P3 | confirmada, com as ressalvas documentadas | idem | idem |
| P4 | encerrada (valor 0,9762) | idem | idem |
| P5, `REVISAO_PROTOCOLO` | não tratadas aqui | pendentes (exigidas a partir da Etapa 4) | pendentes |

Os `valor_sha256` são os de `guardas.hash_valor_decisao` sobre o valor aprovado: P1 = nenhum (`null`; K = 100, por paciente em blocos e as seeds `10000 + k` ficam registrados na Seção 13 e na Emenda 1, E11, e a Etapa 1 oficial ainda exige o detalhamento do esquema de blocos), P2 = `SPLINE_PROPOSTO`, P3 = `PAINEL_DECLARADO`, P4 = 0,9762.

## 5. Circularidade e máscaras reduzidas (Seções 10.3 e 10.4)

- **Não fazem parte** do conteúdo executado pela Etapa 3 oficial. Serão realizadas em **atividade própria, posterior**, que lerá `data/vies_malignidade_mascara_pequena.csv` e a marca `mascara_pequena`, hoje fora do carregador.
- A circularidade com o tamanho permanece registrada como **limitação** (Seção 11.1, item 2; Seção 11.2, item 6). O relatório da Etapa 3 declara a omissão em `limitacoes_declaradas`.
- A ausência dessas análises **não altera** a leitura da Seção 9.

## 6. Benchmark S6: faixa de diâmetro (D2)

A diferença entre o intervalo do Exp. 26.0.1 (fechado, [12, 30]) e o do P9 (semiaberto, [12, 30)) é documentada na Seção 19 do pré-registro P9. O P9 mantém **[12, 30)**; nenhum parâmetro foi alterado. A equivalência é **observada** pelas contagens, que o modo definitivo do benchmark confere (226 / 187 / 135) antes de qualquer ajuste. Os arquivos de evidência citados pela equipe **não estão no repositório** e não foram inspecionados por quem redigiu este registro.
