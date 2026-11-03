# Adendo proposto — reordenação das Etapas 1–3 da S4

**Situação:** **PROPOSTA — não aprovada.** Este texto não altera o protocolo da S4 (`docs/sprint4/protocolo_valor_incremental.md`, Seção 7, "as etapas são sequenciais") nem a Emenda 1 enquanto não houver aprovação formal, datada e anterior a qualquer ação afetada.
**Data da proposta:** 03/11/2026 · **Proponente:** usuário da sessão (conta `RafaellLobo`, revisor da Emenda 1); redigido pelo Claude Code a pedido. **Aprovação:** pendente do responsável científico e do revisor. Não há aprovação formal do grupo e não se presume nenhuma.

## 1. O que se propõe

Permitir que a **Etapa 3** da S4 (validação oficial, lista fixa, M0–M4) seja executada e registrada **antes** das Etapas 1 e 2 oficiais, que continuam **obrigatórias** e passam a ser executadas depois dela, **antes** da Etapa 4 (congelamento) e da Etapa 5 (teste).

## 2. Motivo

A Etapa 3 usa a lista fixa da Seção 5.1 e só depende de `train` (ajuste) e `validation` (avaliação). Não consome nenhum resultado das Etapas 1 e 2, que avaliam o **procedimento ajustado** de seleção (M2v–M4v). A Emenda 1 (E7 e revisão 3) condiciona o benchmark S6 apenas ao **registro da Etapa 3**; a Etapa 2 oficial exige o controle por permutação com K = 100 (cerca de 2 a 9 horas de computação), o que adia o benchmark S6 sem relação de dados com ele.

## 3. O que permanece obrigatório

- Etapas 0, 1 e 2 oficiais, com todos os artefatos, análises e limitações do protocolo (CV aninhada 10 × 5, controle por permutação P1, spline P2, CV com lista fixa e sensibilidades P3/P7).
- Decisões P1, P2 e P3 registradas nas guardas **antes** da Etapa 3, como hoje.
- Etapa 3 completa conforme as Seções 7.3, 8, 9 e 10 do protocolo.
- Nenhuma leitura do teste fora da Etapa 5; Etapa 5 apenas descritiva (Emenda 1, E3).
- Os resultados das Etapas 1 e 2 **não alteram** modelo, lista, critérios ou a Etapa 3 já registrada.

## 4. Limitações da reordenação (a declarar nos relatórios)

1. A Etapa 1 serve para verificar implementação, convergência, invariantes e custo; ao rodar a Etapa 3 antes, esse controle técnico prévio deixa de existir para o código compartilhado. Mitigação: invariante do M1, testes sintéticos e a Etapa 0.
2. Se as Etapas 1 ou 2 revelarem erro técnico em código usado pela Etapa 3, a Etapa 3 deve ser refeita e registrada de novo, e o registro `S4_E3_REGISTRADA` perde a validade para o benchmark S6 já executado.
3. A validação é lida pelo benchmark S6 depois da Etapa 3, como já previsto na E7.
4. A ordem alterada é uma limitação de governança a declarar à banca.

## 5. Como seria a aprovação

Aprovação formal, datada, do responsável científico (Samara Ruiz) e do revisor, registrada no PR ou documento que a introduza e referenciando o commit deste texto. Esta proposta não contém, e não deve conter, linha `APROVACAO-DECISAO`. Enquanto não for aprovada, a ordem do protocolo vigente prevalece, e a Etapa 6 do benchmark S6 permanece bloqueada.
