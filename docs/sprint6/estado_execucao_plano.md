# Estado da execução do PLANO_EXECUCAO_AUTONOMA_BENCHMARK_S6

**Atualizado em:** 03/11/2026, 10h40 (America/Sao_Paulo) · **Branch:** `feat/s6-benchmark-executor` · **Natureza:** registro operacional. **Não é adendo, não aprova nada, não contém registro `APROVACAO-DECISAO` e não altera a configuração.**

## Resultado

**BLOQUEADO na Etapa 6.** As Etapas 1 a 5 estão concluídas até onde podem ir sem as aprovações. As Etapas 6, 7 e 8 não foram executadas: dependem de portões humanos que não existem. Nenhum treino definitivo, métrica externa ou figura de resultado foi produzido.

## Matriz de portões (verificada no Git e nas APIs reais às 10h32)

| Portão | Evidência exigida | Estado | Fonte |
|---|---|---|---|
| Emenda 1 (S4) integrada | PR #10 no `main` | **liberado** | merge `a0dcae9` |
| Registros `APROVACAO-DECISAO` de P1, P2, P3 (S4) | linhas na coluna 0 do protocolo + `config.DECISOES` com os commits | **ausente** | nenhuma linha em `main`; Etapas 1–3 da S4 → `BloqueioDecisao` |
| Etapas 1 e 2 oficiais da S4 | manifestos definitivos | **sem código oficial, não executadas** | `src/incremental` (só exploratórias) |
| Etapa 3 oficial da S4 | manifesto definitivo, elegível, concluído, versionado | **não executada** (código pronto na branch `feat/s4-etapa3-validacao`) | — |
| Adendo de reordenação da S4 | aprovação formal prévia | **proposto, não aprovado** | `adendo_reordenacao_s4_proposta.md` |
| `S4_E3_REGISTRADA` | registro + manifesto | **ausente** | `verificar_s4_e3` → `RegistroAusente` |
| D2 (variável da faixa) | confirmação documental da responsável pelo Exp. 26.0.1 | **pendente** (há só a declaração do usuário, abaixo) | P9 §17 |
| D1–D12 e P9 | ratificação e registro `P9` com `valor_sha256` da configuração | **ausente** (PR #11 Draft, head `9b172f2`) | `verificar_p9` → `RegistroAusente` |
| Configuração canônica | SHA-256 `1cd08306786067c1e29229a93a3215e087b0db21da35bc05c595bd5c8a2117d5` | **conferida** com o JSON real do PR #11 | `--modo configuracao --conferir` → código 0 |

## Declarações do usuário (registradas como declarações, não como aprovação)

Em 03/11/2026, o usuário da sessão (conta `RafaellLobo`, revisor designado da Emenda 1) declarou, no plano:

1. **Concorda com D1–D12** do pré-registro P9. É concordância individual: **não** é ratificação do grupo nem registro `APROVACAO-DECISAO`.
2. **A variável da faixa no Exp. 26.0.1 foi `original_shape_Maximum3DDiameter`, com `12 <= D < 30` mm.** É declaração do usuário. A P9 exige confirmação documental da **responsável pelo experimento**; ela **não** foi obtida e **D2 permanece pendente**. A evidência no DoD do Exp. 26.0.1 ainda precisa ser localizada e citada.
3. **Concorda em propor** um adendo para realizar a Etapa 3 da S4 antes das Etapas 1 e 2, mantendo as três etapas, as análises e as limitações. A proposta está em `adendo_reordenacao_s4_proposta.md`. **Não vale como adendo** até aprovação formal.

## Etapas

| Etapa | Estado | Evidência |
|---|---|---|
| 1 — Governança | **concluída como auditoria; execução definitiva bloqueada** | matriz acima; proposta de adendo |
| 2 — Ambiente e configuração | **concluída** | Python 3.9.13, scikit-learn 1.3.2, numpy 1.23.5, pandas 1.5.3, scipy 1.10.1, matplotlib 3.7.3, XGBoost 2.1.4 (fumaça determinística aprovada); SHA da configuração reproduzido; seis arquivos da S4 fixados idênticos ao `main` |
| 3 — Dados e folds | **concluída em implementação** | contagens definitivas (353/509; faixa 226/187/135/91) só podem ser atestadas depois da liberação; nada de validação ou teste foi lido |
| 4 — Modelos e ajuste interno | **concluída** | grades e componentes conferidos contra a configuração |
| 5 — Coordenador e validação de implementação | **concluída** | 613 testes, OK (1 pulado, condicional ao XGBoost ausente); ensaio `implementacao` com 25 unidades em 124 s, 0 falhas, invariante da LR fixa dentro da tolerância; saídas em `reports/sprint6/benchmark/implementacao/`, sem métrica externa |
| 6 — Treino definitivo | **BLOQUEADA** | portões acima |
| 7 — Métricas, bootstrap, contrastes, seleção | **não executada** (código pronto e testado com dados sintéticos) | depende da Etapa 6 |
| 8 — Figuras, relatório, verificação | **não executada** (código pronto) | depende da Etapa 7 |

## Ações externas que desbloqueiam

1. Aprovar o adendo de reordenação (ou decidir executar as Etapas 1 e 2 antes da 3).
2. Registrar `APROVACAO-DECISAO` de P1, P2 e P3 no protocolo da S4 e atualizar `config.DECISOES`.
3. Implementar e executar o que o protocolo da S4 exige das Etapas 1, 2 e 3 (a Etapa 3 está pronta na branch S4, sem as sensibilidades; as Etapas 1 e 2 oficiais não têm código).
4. Registrar `S4_E3_REGISTRADA` com o SHA-256 canônico do manifesto da Etapa 3.
5. Confirmar D2 documentalmente com a responsável; ratificar D1–D12; registrar `P9` com `valor_sha256 = 1cd08306…17d5` em commit posterior ao do JSON.
6. Só então: integrar S4 e PR #11 ao `main`, rebasear a S6 e rodar `--modo definitivo` e `--modo verificar`.

---

## Atualização de 03/11/2026 — execução definitiva concluída

**Natureza:** registro de estado. Os registros formais `APROVACAO-DECISAO` foram lavrados a pedido do revisor com base na confirmação da equipe **relatada por ele** (sem ata ou assinatura individual no repositório); ver a Seção 19 do pré-registro e o Adendo 1 da S4.

| Etapa | Estado | Evidência |
|---|---|---|
| Adendo de ordem das Etapas 1–3 | formalizado antes de qualquer execução | S4 `c9c2a2b` (`docs/sprint4/adendo_01_ordem_das_etapas.md`) |
| P1–P4 (S4) | registradas; `config.DECISOES` atualizada | S4 `d041987` (registros), `7112548` (config) |
| Etapa 0, item 2 oficial | concluída (definitiva, elegível) | S4 `ac8102f`, `reports/sprint4/etapa0_item2/` |
| Etapa 3 oficial (inclui sensibilidades) | concluída (definitiva, elegível); M3−M0 ΔAUC 0,0306, IC [0,0028; 0,0670] | S4 `25c843c`, `reports/sprint4/etapa3_validacao/` |
| SHA-256 do manifesto da Etapa 3 | `1ad800ea62a85be1a38ddcd6b1c68566e573cf2ac55e8ae73f2770aa21ff5705` | validado por `aprovacoes.validar_manifesto_s4` |
| Configuração canônica P9 | regenerada: `a0d096a5415244f17f40360a2a5a8901bc8af1da5cf8687b891f920e1ee122b3` (único campo alterado: pin de `config.py`) | S6 `e4c95e6`, `001d715`, `451123f` |
| Registros `P9` e `S4_E3_REGISTRADA` | lavrados | S6 `c4cc126`, `c124c81` |
| Benchmark definitivo | concluído: 250 unidades, 0 falhas técnicas pendentes, faixa 226/187/135 conferida | S6 `22d6a9b`, `reports/sprint6/benchmark/` |
| `--modo verificar` | 38 saídas conferem; recálculo idêntico | `C:\PI3\logs_s6\verificar.txt` (fora do Git) |

**Pendências que permanecem:** Etapas 1 e 2 oficiais da S4 (obrigatórias, depois da Etapa 3 e antes da 4); circularidade (§10.3) e máscaras reduzidas (§10.4), em atividade própria; esquema de blocos da permutação da P1; P5 e `REVISAO_PROTOCOLO` (Etapa 4); P8; merges e push (nenhum feito; PR #11 segue Draft; as branches S4 e S6 só existem localmente).
