# Etapa 3 da S4 — implementação técnica e caminho para desbloquear a S6

**Data:** 03/11/2026 · **Branch:** `feat/s4-etapa3-validacao` (a partir do `main` = `c43afe8`, sem os commits da S6) · **Natureza:** registro técnico. Este documento **não** aprova decisões, não registra `P1`, `P2`, `P3` nem `S4_E3_REGISTRADA`, não altera o protocolo e não executa nenhuma etapa com dados oficiais.

---

## 1. Resumo

Foram implementados, só com dados sintéticos, os três componentes técnicos que faltavam à Etapa 3 da S4:

| # | Componente | Módulo | Situação |
|---|---|---|---|
| 1 | Etapa 0, item 2: procedimento ajustado no treino completo, só descritivo | `src/incremental/etapa0.py` (+ `selecao.folds_estabilidade_treino_completo` e `selecao.selecionar_treino_completo`) | implementado e testado; **não executado** |
| 2 | Bootstrap pareado por paciente da S4 (Seção 8.2) | `src/incremental/bootstrap.py` | implementado e testado |
| 3 | Orquestrador `etapa3_validacao`, com checkpoints, rastreabilidade e manifesto definitivo | `src/incremental/validacao.py` | implementado e testado; **não executado** |

A Etapa 3 oficial **continua bloqueada**: as decisões P1, P2 e P3 não estão aprovadas, e a sequência das Etapas 1 e 2 oficiais depende de uma decisão da equipe (Seção 4). Nada foi lido de `data/`; `src/benchmark_s6`, os PRs #10 e #11, o protocolo e a configuração científica não foram alterados.

## 2. O que foi implementado

### 2.1 Etapa 0, item 2 — `etapa0.py`

Protocolo, Seção 7.1, item 2: o procedimento da Seção 5.4 é aplicado **uma vez** aos 417 nódulos de treino, e a lista resultante é registrada **apenas de forma descritiva**, com a diferença em relação à lista declarada.

- `selecao.folds_estabilidade_treino_completo(amostra)`: 5 folds de estabilidade (seed 42) sobre todas as linhas; os folds internos saem de `folds.folds_internos` **sem alteração** (seed `100 + j + 1`). A função fica em `selecao.py` e não em `folds.py` de propósito: a S6 fixa o SHA-256 de `src/incremental/folds.py` (`adaptadores_s4.py`), e alterá-lo mudaria a configuração canônica `1cd08306…17d5` e invalidaria o pré-registro P9. Nenhum dos seis arquivos da S4 fixados pela S6 foi alterado.
- `selecao.selecionar_treino_completo(amostra, X, features)`: mesma seleção da Seção 5.4 (filtros e scaler ajustados em cada treino interno, regra de pelo menos 3 de 5); devolve um `SelecaoFold` com `repeticao = fold_externo = selecao.TREINO_COMPLETO (-1)`.
- `etapa0.executar_etapa0_item2(destino, ...)`: atividade `etapa0_prerequisitos` (não exige decisões; lê só `train`); não ajusta modelo de avaliação nem calcula métrica; grava `relatorio_etapa0_item2.json` e `manifesto_etapa0_item2.json`.
- A comparação usa `config.PAINEL_DECLARADO` como referência **descritiva**. Isso não é a aprovação de P3 e não libera M2, M3 ou M4.

### 2.2 Bootstrap da S4 — `bootstrap.py`

Parâmetros de `config.ESPEC_BOOTSTRAP`, conforme a Seção 8.2:

| Parâmetro | Valor |
|---|---|
| Unidade | `patient_id`; cada paciente sorteado entra com **todos** os seus nódulos |
| Pareamento | a **mesma** amostra de pacientes vale para todos os modelos e contrastes em cada réplica |
| Réplicas / seed | B = 2000 / 42 (`numpy.random.default_rng`; os B sorteios são feitos de uma vez, na ordem) |
| IC | percentil 2,5 e 97,5 |
| Réplicas degeneradas | descartadas **e contadas**, sem novo sorteio |
| Retreino | não |

Calcula AUC (principal) e, para a Seção 10.2, Brier e log-loss, na mesma réplica. **Não reutiliza** o bootstrap da S6 (outro desenho: B = 10.000, Bonferroni, ponderação); um teste confere que o módulo não importa `benchmark_s6`.

### 2.3 Orquestrador da Etapa 3 — `validacao.py`

Atividade `etapa3_validacao` (`config.Etapa.E3`): ajusta M0–M4, com a lista fixa, nos 417 nódulos de `train` e avalia nos 92 nódulos / 62 pacientes de `validation`. **O teste nunca é lido.**

Sequência, sem pular passos:

1. destino livre; autorização das guardas para a atividade e para M0–M4 (exige P1, P2 e P3 com evidência no git, **antes** de ler qualquer dado);
2. sessão de checkpoints local (fora do repositório e de `data/`) e `rastreabilidade.Execucao`, que exige git limpo e ambiente esperado na execução definitiva;
3. carga de `train` + `validation` por `dados.carregar`; contagens, classes e disjunção de pacientes conferidas;
4. invariante do M1 (Seção 6.2): M1 reajustado no treino tem de reproduzir `baseline_metrics.json`, só em parâmetros e **antes** de qualquer predição;
5. uma unidade (checkpoint) por modelo; unidade concluída é reaproveitada na retomada, e falha não conta como concluída;
6. consolidação sempre a partir dos payloads gravados: AUC por modelo, Δ de M3−M0 (primário), M4−M1 (secundário), M1−M0 e M2−M0 (descritivos), IC do bootstrap, classificação da Seção 9 e métricas descritivas da Seção 10.2 (Δ Brier, Δ log-loss, calibração, recall e especificidade em 0,5);
7. relatório agregado (sem identificadores, predições individuais, p-valor ou significância) e manifesto definitivo.

**Fora desta implementação, e declarado no próprio relatório:** sensibilidade com spline (Seção 10.1), circularidade (10.3), máscaras reduzidas e faixas de diâmetro (10.4) e as sensibilidades de P3 (lista reproduzida) e P7 (`C = 0,1`). Nenhuma delas muda a leitura da Seção 9. A comparação com a CV aninhada (Etapa 2) e com o teste (Etapa 5), exigida pela Seção 9, também fica fora do relatório da Etapa 3.

## 3. Testes

Só dados sintéticos, repositórios git temporários e aprovações **fictícias** de P1–P3 nesses repositórios; as linhas de `test` têm valores inválidos (se fossem lidas, a carga falharia).

| Arquivo | Testes | O que cobre |
|---|---:|---|
| `tests/incremental/test_bootstrap.py` | 15 | referência independente (pandas + `default_rng`), pareamento, independência dos modelos pedidos, determinismo, réplicas degeneradas, validação de entradas, ausência de p-valor |
| `tests/incremental/test_etapa0.py` | 18 | folds do treino completo, seleção contra referência independente, comparação de listas, manifesto, só `train`, determinismo, recusa de árvore suja |
| `tests/incremental/test_validacao.py` | 24 | isolamento (só `train` e `validation`), AUC e métricas contra ajuste independente, bootstrap, contrastes, bloqueios de aprovação sem leitura nem gravação, invariante do M1, retomada e interrupção, determinismo, manifesto definitivo aceito pelo validador estrito da S6 |

**Resultados (rodados no estado final do código, com `PYTHONHASHSEED=0`):**

- `python -B -m unittest discover -s tests/incremental`: **485 testes, OK** (428 anteriores + 57 novos; 780 s).
- `python -B -m unittest discover -s tests/benchmark_s6` sobre esta branch: **379 testes, OK, 1 pulado** (teste condicional à ausência do XGBoost, que está instalado). Isso confirma que a S6 continua válida contra esta S4, e que `sha256_configuracao()` da S6 segue `1cd08306786067c1e29229a93a3215e087b0db21da35bc05c595bd5c8a2117d5`.
- Uma primeira rodada da S4 (485 OK) foi feita antes de eu mover a função de `folds.py` para `selecao.py`; a rodada acima é a do estado final.

## 4. Conferência da sequência oficial das Etapas 0–3

Protocolo, Seção 7: "as etapas são sequenciais. Cada uma só começa depois que a anterior tiver sido concluída e documentada". As guardas da S4 **não** exigem manifestos de etapas anteriores; exigem só as decisões (`P1`, `P2`, `P3` para a Etapa 3).

| Etapa | Situação verificada | Evidência | Depende de |
|---|---|---|---|
| 0, item 1 — reprodução fiel | **concluída** e auditada; a lista reproduzida diverge da declarada (6 de 8) | `reports/sprint4/feature_selection/` (commit `0ef4039`) | decisão P3 sobre seguir com a lista declarada |
| 0, item 2 — procedimento ajustado no treino completo | **código pronto, não executado** | `etapa0.py` | execução oficial (git limpo) |
| 0, item 3 — invariantes | **concluída** (23 de 23) | `reports/sprint4/reference_models/validacao_modelos_referencia.json` | — |
| 0, item 4 — isolamento | garantido por código | `dados.py`, `ETAPA_DE_LEITURA` | — |
| **1 — piloto oficial** | **não executada e sem código oficial**. Só existe o piloto **exploratório**, que as guardas e a rastreabilidade proíbem de ser definitivo | `piloto.py`; `config.ATIVIDADES_SOMENTE_EXPLORATORIAS` | P1 e P2 aprovadas; implementação do controle por permutação (P1) e da sensibilidade com spline (P2); execução |
| **2 — CV aninhada completa oficial** | **não executada e sem código oficial**. Só existe a CV repetida **exploratória** (também só exploratória); a CV com a lista fixa exige P3 | `cv_repetida.py` | as mesmas de Etapa 1 e P3; execução |
| 3 — validação oficial | **código pronto, não executado** | `validacao.py` | P1, P2, P3 aprovadas; item 2 da Etapa 0; decisão sobre a sequência (abaixo) |

Os experimentos exploratórios (piloto e CV repetida) **não substituem** as Etapas 1 e 2: a rastreabilidade os recusa como definitivos, não cobrem permutação nem spline e rodaram antes de P1–P3.

**Achado:** pelo texto do protocolo, a Etapa 3 só começa depois de as Etapas 1 e 2 terem sido concluídas e documentadas; hoje essas duas etapas ainda dependem de **implementação e de execução oficial**. Executar a Etapa 3 antes delas inverte a ordem prevista. O protocolo não indica dependência de dados entre as etapas (a Etapa 3 usa a lista fixa, não o resultado das Etapas 1 e 2), mas a ordem é uma regra do protocolo, e mudá-la é decisão da equipe, não técnica.

## 5. Pré-requisitos para executar a Etapa 3 oficialmente

| # | Pré-requisito | Tipo | Estado |
|---|---|---|---|
| 1 | Aprovação da Emenda 1 (PR #10) e integração dela no protocolo da S4 | decisão humana | pendente (Draft) |
| 2 | Linhas `APROVACAO-DECISAO` de `P1`, `P2` (com `EspecSpline`) e `P3` (com o painel) em `docs/sprint4/protocolo_valor_incremental.md`, introduzidas pelo commit da aprovação | decisão humana | pendente |
| 3 | `config.DECISOES` com P1, P2 e P3 `APROVADA`, o hash completo do commit e o `valor` (**mudança em `src/incremental/config.py`, depois da aprovação; altera `hash_configuracao` e as constantes de hash dos testes**) | desenvolvimento | depende do item 2 |
| 4 | Decisão sobre a sequência: executar e documentar as Etapas 1 e 2 oficiais antes **ou** registrar, antes de rodar, uma decisão datada de que a Etapa 3 pode preceder (ver Seção 6) | decisão humana | pendente |
| 5 | Execução oficial do item 2 da Etapa 0 | execução | pendente |
| 6 | Árvore git limpa, código e ambiente congelados (Python 3.9.13, scikit-learn 1.3.2, numpy 1.23.5, pandas 1.5.3, scipy 1.10.1) | execução | — |
| 7 | Destino das saídas **dentro do repositório**, em `reports/sprint4/` (a guarda da S6 só aceita manifesto lá) | execução | — |

## 6. Caminho mínimo para desbloquear a S6

A S6 exige apenas o manifesto da Etapa 3 (`etapa3_validacao`, definitivo, elegível, `concluida`, versionado em `reports/sprint4/`) e o registro `S4_E3_REGISTRADA` com o SHA-256 canônico desse manifesto. A ordem mínima é:

1. **Revisão e integração desta branch** (código S4 da Etapa 0, item 2, do bootstrap e da Etapa 3), em PR próprio.
2. **PR #10:** o revisor aprova a Emenda 1 com data; o PR é integrado.
3. **Registro de P1, P2 e P3** no protocolo, em commit de aprovação; em seguida, commit com a atualização de `config.DECISOES` e das constantes de hash dos testes.
4. **Decisão da equipe sobre a sequência das Etapas 1 e 2** (item 4 da Seção 5):
   - **Opção A (literal):** implementar e executar as Etapas 1 e 2 oficiais antes da Etapa 3. É o maior bloco de trabalho: controle por permutação, spline, M0s/M3vs e M2–M4 na CV.
   - **Opção B:** registrar, em adendo datado e antes de executar, que a Etapa 3 pode preceder as Etapas 1 e 2 porque usa a lista fixa e não depende dos resultados delas. Isso altera a ordem do protocolo e **só a equipe pode decidir**; não foi feito.
5. **Etapa 0, item 2, oficial** (git limpo):
   `python -c "from src.incremental import etapa0; etapa0.executar_etapa0_item2('reports/sprint4/etapa0_item2')"` (diretório criado antes e versionado depois).
6. **Etapa 3 oficial** (git limpo, depois dos itens 2 a 5):
   `python -m src.incremental.validacao --checkpoints <pasta local fora do repositório> --destino reports/sprint4/etapa3_validacao --confirmo-etapa3-oficial`
7. **Versionar** `relatorio_etapa3_validacao.json` e `manifesto_etapa3_validacao.json` em commit; conferir com `aprovacoes.validar_manifesto_s4` e calcular `aprovacoes.sha256_manifesto_s4`.
8. **Registrar `S4_E3_REGISTRADA`** na Seção 19 do pré-registro P9 (decisão humana) e, em paralelo, concluir a F8 da S6 (P9, D2, D1–D12).
9. Só então a S6: rebase sobre o `main` com a S4, F9 e F10.

## 7. Cuidados

- **Não executar** `validacao.py` nem `etapa0.py` com dados oficiais antes de os itens 1 a 4 da Seção 5 estarem resolvidos. Sem P1–P3 as guardas recusam a Etapa 3 e nada é lido; a Etapa 0 não exige decisões e roda com os dados oficiais, por isso só deve ser executada de forma deliberada.
- **Checkpoints:** usar pasta nova fora do repositório e de `data/`; qualquer mudança de código ou ambiente muda a identidade e invalida os checkpoints anteriores.
- **Mudar `config.DECISOES` depois da aprovação** altera `hash_configuracao`; as constantes dos testes (`HASH_CONFIG_INCREMENTO_1` e equivalentes) devem ser atualizadas no mesmo commit, e o PR deve dizer isso.
- **Resultados sintéticos dos testes não têm significado científico.**
