# Relatório de pendências da liberação científica (F8) — benchmark S6

**Data:** 03/11/2026 · **Branch:** `feat/s6-benchmark-executor` (8 commits locais sobre `main` = `c43afe8`) · **Natureza:** levantamento de evidências para decisão da equipe. Este relatório **não** aprova nada, não registra aprovação e não altera o protocolo.

**Escopo do levantamento:** estado da Etapa 3 definitiva da S4, bloqueios dos PRs #10 e #11, coerência da opção técnica `--manifesto-s4` e conferência do estado técnico local. Nada foi executado: nem a Etapa 3, nem o benchmark definitivo.

---

## 1. Resumo para decisão

O pipeline técnico do benchmark S6 (F0–F7) está pronto e testado. O benchmark definitivo continua bloqueado por três motivos independentes, todos de governança ou da S4:

| # | Bloqueio | Natureza | Contornável? |
|---|---|---|---|
| B1 | Emenda 1 (PR #10) e pré-registro P9 (PR #11) em Draft, sem revisão e sem aprovação registrada | Decisão humana | Não |
| B2 | A Etapa 3 definitiva da S4 **não foi executada** e **não existe código que a execute** | Desenvolvimento (S4) + decisão humana | Não |
| B3 | Os registros `P9` e `S4_E3_REGISTRADA` não existem e só podem entrar depois de B1 e B2 | Decisão humana | Não |

**O que depende de desenvolvimento** (nenhum item é do código S6 já pronto):

1. Orquestrador da Etapa 3 da S4 (`etapa3_validacao`), com bootstrap pareado por paciente (protocolo S4, Seção 8.2) e manifesto definitivo. Hoje `src/incremental/` só tem os orquestradores do piloto e da CV repetida, ambos exploratórios.
2. Atualização de `src/incremental/config.py` para marcar P1, P2 e P3 como aprovadas, com o hash do commit de revisão (e o painel de P3). As guardas da S4 exigem isso para a Etapa 3.
3. Integração do código S6 (os 8 commits locais) por PR próprio, depois das fases acima, e só então a execução da F9.

**O que depende de decisão humana** está na Seção 7.

---

## 2. Conferência do estado técnico local

| Item | Resultado | Evidência |
|---|---|---|
| Commits locais sobre `main` | 8, na ordem F0 → F6 (`ea81910` … `8a947ea`) | `git log main..HEAD` |
| Árvore de trabalho | Limpa antes deste relatório | `git status --short` vazio |
| `src/incremental` | Sem diferença contra `main` | `git diff main --stat -- src/incremental` vazio |
| Suíte completa | 613 testes, `OK (skipped=1)`, 895 s | `C:\PI3\logs_s6\suite_completa_F6_b.txt` (log local, fora do Git) |
| Primeira rodada da suíte | 12 falhas e 3 erros, causa já corrigida (patch global de grades reduzidas nos meus testes novos) | `C:\PI3\logs_s6\suite_completa_F6.txt` |
| Hash da configuração canônica | `1cd08306786067c1e29229a93a3215e087b0db21da35bc05c595bd5c8a2117d5`, inalterado | `docs/sprint6/plano_mestre_execucao_s6.md` |

### Causa do único teste pulado

É `test_ausencia_no_ambiente_atual_e_explicita`, em [test_modelos.py:511](../../tests/benchmark_s6/test_modelos.py#L511), marcado com `@unittest.skipIf(XGB_DISPONIVEL, ...)`.

- O teste verifica que a **ausência** do XGBoost produz o erro explícito `XGBoostIndisponivel`.
- Desde a F1, o XGBoost 2.1.4 está instalado na `.venv`, então a condição de ausência não existe e o teste se pula com a mensagem *"XGBoost 2.1.4 instalado: a ausência não se aplica"*.
- O par dele, `test_fumaca_xgboost_real` (ajuste, clone, `GridSearchCV` e duas execuções idênticas), roda e passa.
- Reconferido hoje: `Ran 2 tests … OK (skipped=1)`.

Não é teste de comportamento desativado por defeito. É um teste condicional ao ambiente, e o ramo coberto (XGBoost ausente) não se aplica neste ambiente.

---

## 3. Etapa 3 definitiva da S4 — levantamento

### 3.1 Estado atual: não executada

- **Protocolo e estado do cartão:** `docs/sprint4/estado_cartao4.md`, tabela *Situação das etapas*: "3 — Validação oficial: **não executada**; bloqueada por P1, P2, P3". A Etapa 0 consta como **parcial** (falta o item 2 da Seção 7.1).
- **Artefatos:** os únicos manifestos em `reports/sprint4/` são `piloto_exploratorio` e `cv_repetida_exploratoria`. Nenhum arquivo em `reports/` ou `docs/sprint4/` contém `etapa3_validacao`. Só o plano mestre do S6 menciona a atividade.
- **Manifesto válido:** não existe. Consequentemente não existe candidato a `S4_E3_REGISTRADA`.

### 3.2 O que a Etapa 3 exige (protocolo S4, Seções 7.1 e 7.3)

| Pré-requisito | Situação |
|---|---|
| Reprodução fiel do procedimento original (Seção 7.1, item 1) | Concluída e auditada (`reports/sprint4/feature_selection/`). A lista reproduzida diverge da declarada (6 de 8 coincidem). Pela regra da Seção 7.1, a lista declarada permanece e a reproduzida entra como sensibilidade. A decisão de seguir adiante é "registrada pelo grupo antes da Etapa 3" (P3). |
| Etapa 0, item 2 (procedimento ajustado no treino completo, descritivo) | **Pendente**, segundo `estado_cartao4.md`. O item não exige decisão. |
| Invariantes estruturais e invariante do M1 (item 3) | Concluídos (`validacao_modelos_referencia.json`: 23 de 23). |
| Decisões P1, P2 e P3 aprovadas | **Pendentes** (Seção 3.4). |
| Lista fixa e painel de M2–M4 (P3) | Pendente. `config.painel_fixo()` levanta `BloqueioDecisao` enquanto P3 estiver pendente. |
| Dados lidos | Treino → validação (417 → 92 nódulos / 62 pacientes). Teste nunca. `config.PARTICOES_POR_ETAPA[E3] = (train, validation)`. |
| Saída | IC 95% por bootstrap pareado por paciente para M3 − M0 (primário) e M4 − M1 (secundário). |

### 3.3 Código e comandos existentes

| Componente | Existe? | Observação |
|---|---|---|
| Configuração, modelos M0–M4, folds, dados, checkpoints, rastreabilidade, guardas | Sim (`src/incremental/`) | Infraestrutura geral, 428 testes na S4 (`estado_cartao4.md`, Seção 1.1). |
| Leitura de treino + validação para `etapa3_validacao` | Sim | `dados.ETAPA_DE_LEITURA` mapeia as etapas e a Etapa 3 está em `cfg.Etapa.E3`. |
| Manifesto definitivo (`Execucao` com `Finalidade.DEFINITIVA`) | Sim, como API | Só pode ser gerado por uma atividade que a use. Nenhuma atividade definitiva existe hoje. |
| **Orquestrador da Etapa 3** | **Não** | Os únicos orquestradores são `piloto.py` (exploratório) e `cv_repetida.py` (exploratório). `scripts/sprint4/` tem apenas a reprodução de Juan e a validação dos modelos de referência. |
| **Bootstrap pareado da S4 (Seção 8.2)** | **Não** | `config.ESPEC_BOOTSTRAP` guarda os parâmetros, mas não há função que os use. O bootstrap implementado é o do S6 (`src/benchmark_s6/bootstrap.py`), para outro desenho. |
| Comando para executar a Etapa 3 | **Não** | — |

**Conclusão:** a Etapa 3 definitiva exige **desenvolvimento novo no domínio da S4**, e não só uma execução. Isso fica fora do que a restrição "não alterar `src/incremental`" do S6 permite. Precisa de decisão sobre onde e por quem será feito (Seção 7, item H6).

### 3.4 Aprovações necessárias para a Etapa 3

As guardas da S4 derivam as decisões exigidas por etapa de `config.DECISOES_POR_ETAPA`. Para `etapa3_validacao`, são **P1, P2 e P3** (a exigência é cumulativa a partir de cada prazo).

Cada decisão só conta se:

1. houver uma linha `APROVACAO-DECISAO codigo=Pn status=aprovada data=AAAA-MM-DD valor_sha256=<64 hex>` na coluna 0 de `docs/sprint4/protocolo_valor_incremental.md`, fora de bloco de código, introduzida pelo commit da aprovação, com data ≤ data do commit;
2. esse commit (40 hex) estiver gravado na `Decisao` correspondente de `src/incremental/config.py` (status `APROVADA`, com `valor` quando houver, por exemplo o painel de P3);
3. o commit for ancestral do `HEAD`, e o registro estiver idêntico e único no `HEAD`.

Estado atual: **nenhuma linha `APROVACAO-DECISAO` existe no protocolo** e as seis decisões estão `PENDENTE` em `config.py`. A Emenda 1 (revisão 2/3) descreve que a responsável tomou P1, P2, P3, P5, P6, P7 e P8 em 31/10/2026, **mas o registro formal exigido pelas guardas ainda não foi feito**, e a aprovação do revisor está pendente (Seção 4).

Observação de nomenclatura: a **P9 da S4** (hiperparâmetros em anexo datado) foi **transferida** para o protocolo da S6 pela revisão 3 (E11/E12). Não confundir com o registro `P9` do pré-registro S6 (SHA da configuração canônica).

---

## 4. PR #10 — Emenda 1 ao protocolo da S4 (Draft)

| Campo | Valor |
|---|---|
| Estado | `OPEN`, **Draft**, `MERGEABLE`, 0 revisões formais |
| HEAD | `573bc1729f7dec33c73675ccc9c48c97d5e50c5a` (igual ao `origin`) |
| Arquivos | `docs/sprint4/emenda_01_auditoria_teste.md`, `docs/sprint4/protocolo_valor_incremental.md` |
| Atividade | 3 comentários: revisão técnica do revisor (31/10), resposta da responsável (revisão 2) e pacote do patch |

**Condições de congelamento** (Emenda, seção *Congelamento*):

1. emenda aprovada pelo revisor, **com data**, e a ausência de resposta do grupo registrada (E11);
2. responsável e revisor registrados no card (feito: Samara / Rafael);
3. commit que contém a emenda aprovada identificado e seu hash registrado.

**Pendências observadas:**

- A aprovação datada do revisor **não existe**. O comentário de 31/10 é uma revisão técnica com seis pontos, todos reconhecidos como procedentes na revisão 2; ele própria diz "a proposta permanece em revisão". O merge sozinho **não** é aprovação.
- O texto da Emenda diz que o congelamento depende só da confirmação com data do revisor.
- O PR descreve o patch de revisão 2, e a revisão 3 (02/11) está no HEAD `573bc17`, que o P9 cita. Convém que a aprovação identifique **qual commit** está aprovando.
- Duas verificações da E4 (rodadas fora do ambiente oficial) viraram pendência com prazo "antes do relatório final"; não condicionam o congelamento.
- A E7 mantém o uso da validação pelo benchmark S6 **bloqueado mesmo após o congelamento**, até o registro da Etapa 3.

---

## 5. PR #11 — Pré-registro P9 do benchmark S6 (Draft)

| Campo | Valor |
|---|---|
| Estado | `OPEN`, **Draft**, 0 revisões, 0 comentários |
| HEAD | `9b172f27f637db97075866471ab05848a3b89e23` (igual ao `origin`) |
| Arquivos | `docs/sprint6/preregistro_p9_benchmark.md`, `docs/sprint6/config_benchmark_p9.json` |
| Situação declarada no texto | "PROPOSTA — não aprovada" · "Registros de aprovação: nenhum" · "Ratificação do grupo: solicitada" |

**Pendências do próprio PR (checklist no corpo):**

| Item | Estado | O que falta |
|---|---|---|
| **D2** — variável da faixa | Aberta | A responsável confirma, no DoD do Exp. 26.0.1, que a faixa foi definida por `original_shape_Maximum3DDiameter` ∈ [12, 30) mm. Nenhum documento versionado registra a variável. Se a variável mudar, a configuração e o SHA são gerados de novo. |
| **D1–D12** (Seção 17) | Todas propostas | Ratificação formal. A confirmação da responsável e do revisor "não equivale à aprovação formal do grupo". |
| Configuração canônica | Cumprida | Arquivo versionado (`74c494a`) e SHA anotado na Seção 19 (`9b172f2`), **antes** de qualquer registro. |
| Dependência do PR #10 | Aberta | Até o merge dele, a Emenda é citada pelo commit `573bc17`; depois entram links relativos (critério de aceite 1). |
| Registro `APROVACAO-DECISAO codigo=P9` | Inexistente | Entra na Seção 19, em commit **posterior** ao da configuração, com `valor_sha256 = 1cd08306…17d5`. |
| Registro `S4_E3_REGISTRADA` | Inexistente | Depende do manifesto definitivo da Etapa 3 (Seção 3). |

**Evidências verificadas hoje com a API pública de `aprovacoes.py` (leitura do Git, sem efeitos):**

| Chamada | Resultado | Saída |
|---|---|---|
| `verificar_p9()` | `RegistroAusente`: o pré-registro não existe no HEAD | código 2 |
| `verificar_s4_e3(None)` | `RegistroAusente`: manifesto não informado | código 2 |
| `verificar_s4_e3("reports/sprint4/…")` | `RegistroAusente`: o pré-registro não existe no HEAD | código 2 |

**Consequência prática:** os registros precisam estar num documento que exista no HEAD da execução. Hoje o pré-registro só existe na branch do PR #11. A execução definitiva só será possível numa branch cujo histórico contenha, nesta ordem, o pré-registro com a configuração (PR #11), o registro `P9` e o registro `S4_E3_REGISTRADA`, todos ancestrais do HEAD.

**Achado a corrigir na documentação (não é do código):** o PR #11 e o P9 dizem que o código gerador `db4f476` "existe só localmente, em `feat/s6-benchmark-configuracao`", mas o PR #12 dessa branch foi **integrado** ao `main`. A Seção 19 vale como registro histórico e só precisa de adendo se a equipe quiser corrigir.

---

## 6. Coerência da opção `--manifesto-s4`

**Veredito: coerente com a P9 e com as guardas; a Seção 14.3 do pré-registro omite a opção na sinopse do comando. Não é divergência de método, é lacuna de redação.**

| Verificação | Resultado |
|---|---|
| A P9 exige `S4_E3_REGISTRADA` no modo `definitivo` e `verificar` (Seção 14.3, passo 3; critério 6) | Sim |
| A guarda precisa do caminho do manifesto | Sim. `aprovacoes.verificar_s4_e3(None)` levanta `RegistroAusente`. O caminho não vem da configuração canônica (que não tem campo para ele). |
| A opção contraria algo da P9? | Não. A sinopse da Seção 14.3 traz `--checkpoints`, `--destino` e `[--processos N]` e não lista `--manifesto-s4`. Mas o passo 3 não funciona sem o caminho. |
| A opção cria brecha nas guardas? | Não. O caminho só é aceito se for relativo e normalizado, em `reports/sprint4/` e terminar em `.json`. Nada em `data/`. A leitura é por `git cat-file blob`, nunca pela cópia de trabalho. O manifesto é validado por inteiro (formato, atividade `etapa3_validacao`, definitiva, elegível, `concluida`, hashes internos recalculados, sem dados individuais) e seu SHA-256 canônico tem de igualar o `valor_sha256` do registro. |
| Modos em que vale | `definitivo` e `verificar`. Nos modos `configuracao` e `implementacao` o coordenador recusa a opção com código 64. |
| Testes | `test_coordenador.py` cobre a recusa nos outros modos, o conjunto de opções e o bloqueio com código 2 no definitivo sem aprovação real. |

**Ação sugerida para aprovação (sem alterar o protocolo agora):** se a equipe quiser a sinopse da Seção 14.3 idêntica ao comando real, registrar um **adendo datado** (P9, Seção 15/19) acrescentando `--manifesto-s4 CAMINHO` (obrigatório em `definitivo` e `verificar`). Esse adendo não muda método, só documenta uma entrada que o próprio passo 3 já pressupõe. Se a equipe entender que a omissão é irrelevante, nada muda no código.

---

## 7. Pendências e ações para aprovação da equipe

### 7.1 Decisão humana

| ID | Ação | Quem decide | Depende de | Evidência |
|---|---|---|---|---|
| H1 | Aprovar a Emenda 1 com data, registrar a ausência de resposta do grupo e identificar o commit aprovado | Revisor | — | PR #10; Emenda, seção *Congelamento* |
| H2 | Confirmar D2: a variável da faixa no Exp. 26.0.1 é `original_shape_Maximum3DDiameter` ∈ [12, 30) mm | Responsável | — | P9, Seção 17 (D2) |
| H3 | Ratificar D1–D12. Se alguma mudar, a configuração e o SHA são regerados e anotados antes do registro | Grupo (ou responsável e revisor, declarando a limitação de governança) | H1, H2 | P9, Seção 17; PR #11 |
| H4 | Registrar `APROVACAO-DECISAO codigo=P9` com `valor_sha256 = 1cd08306…17d5` na Seção 19, em commit posterior ao da configuração | Responsável e revisor | H1, H2, H3 | P9, Seção 14.4 e 19 |
| H5 | Registrar as aprovações P1, P2 e P3 da S4 no protocolo (linhas `APROVACAO-DECISAO`), incluindo o painel de P3 e a decisão de seguir adiante com a lista declarada | Responsável e revisor | H1 | Protocolo S4, Seções 7.1 e 13; `guardas.py` |
| H6 | Decidir **onde e por quem** será desenvolvida a Etapa 3 da S4 (Seção 3.3) e autorizar a alteração de `src/incremental` | Equipe | — | Seção 3 deste relatório |
| H7 | Autorizar a execução da Etapa 3 definitiva e do item 2 da Etapa 0 | Responsável e revisor | H5, D6 abaixo | Protocolo S4, Seção 7.1 |
| H8 | Registrar `APROVACAO-DECISAO codigo=S4_E3_REGISTRADA` com o SHA-256 canônico do manifesto da Etapa 3, versionado | Responsável e revisor | D3 e H7 | P9, Seção 19; `aprovacoes.py` |
| H9 | (Opcional) Decidir o adendo da Seção 6 sobre `--manifesto-s4` | Revisor | — | Seção 6 |
| H10 | Autorizar o PR do código S6 (8 commits locais) e a ordem de integração | Equipe | — | Seção 1 |

### 7.2 Desenvolvimento

| ID | Ação | Escopo | Depende de |
|---|---|---|---|
| D1 | Etapa 0, item 2: procedimento ajustado uma vez no treino completo, registro descritivo da lista | S4 (`src/incremental`) | H6 |
| D2 | Bootstrap pareado por paciente da S4 (Seção 8.2), com os contrastes M3 − M0 e M4 − M1 | S4 | H6 |
| D3 | Orquestrador da Etapa 3 (`etapa3_validacao`): ajuste de M0–M4 nos 417 nódulos, avaliação nos 92 da validação, manifesto definitivo | S4 | H5, H6 |
| D4 | Atualizar `config.DECISOES` para P1, P2 e P3 `APROVADA` com o commit e o painel | S4 | H5 |
| D5 | Rebase da branch S6 sobre o `main` com as mudanças da S4, mantendo `git diff main -- src/incremental` vazio na S6 | S6 | D1–D4 integrados |
| D6 | Regerar o ambiente de execução: código e ambiente congelados, Git limpo, pasta de checkpoints nova | S6 | H4, H8 |
| D7 | Executar a F9 (250 unidades) e a F10 (`--modo verificar`) | S6 | Todas as anteriores |

D7 só ocorre depois que `verificar_p9()` e `verificar_s4_e3(<manifesto>)` autorizarem por evidência real.

### 7.3 Sequência mínima

```
H1 → H2 → H3                      (PR #10 e PR #11 aprovados e integrados)
H6 → D1, D2, D3                    (código da Etapa 3 da S4, PR próprio)
H5 → D4                            (P1–P3 registradas no protocolo e na config)
H7 → Etapa 3 definitiva → manifesto versionado
H8                                  (registro S4_E3_REGISTRADA)
H4                                  (registro P9)
D5 → D6 → D7                       (F9 e F10)
```

---

## 8. Riscos e ressalvas

1. **Independência da validação.** O benchmark S6 lê os pacientes de validação; a Emenda (E7) exige que a Etapa 3 esteja registrada antes. Executar o S6 antes disso invalida a regra de precedência.
2. **Mudança em `src/incremental` e critério S6.** O critério 7 do pré-registro (`git diff` de `src/incremental` vazio) é contra o `main`. As mudanças da S4 têm de entrar no `main` primeiro; senão a F10 reprova.
3. **Congelamento do código S6.** Qualquer mudança de código depois dos checkpoints muda a identidade e invalida os checkpoints. O ensaio da F7 foi em `C:\PI3\checkpoints_s6`; para o definitivo, usar pasta nova.
4. **Orçamento de tempo.** Projeção: 0,3–0,6 h com 4 processos (teto de alarme 3 h). Incertezas: escala 509/417 assumida linear, memória não medida, só 5 unidades de RF amostradas.
5. **Governança.** A ratificação do grupo não veio. Os documentos declaram isso; a banca deve ser informada nesses termos (Emenda, *Congelamento*).
6. **Dependência instalada.** O XGBoost 2.1.4 foi instalado na `.venv` do projeto com `--no-deps`, e fixado em `environment.yml`. O estado anterior está em `C:\PI3\logs_s6\pip_freeze_antes_xgboost.txt` (local).

---

## 9. Referências

- Plano mestre: [plano_mestre_execucao_s6.md](plano_mestre_execucao_s6.md), fase F8
- Pré-registro P9 (branch do PR #11): `docs/sprint6/preregistro_p9_benchmark.md`, Seções 14.3, 14.4, 17, 18 e 19
- Emenda 1 (commit `573bc17`): `docs/sprint4/emenda_01_auditoria_teste.md`, E7, E11, E12 e *Congelamento*
- Protocolo S4: [protocolo_valor_incremental.md](../sprint4/protocolo_valor_incremental.md), Seções 7.1, 7.3, 8 e 13
- Estado do cartão: [estado_cartao4.md](../sprint4/estado_cartao4.md), tabela *Situação das etapas*
- Guardas: [aprovacoes.py](../../src/benchmark_s6/aprovacoes.py) (S6), [guardas.py](../../src/incremental/guardas.py) e [config.py](../../src/incremental/config.py) (S4)
- Coordenador: [coordenador.py](../../src/benchmark_s6/coordenador.py)
- Diário local (fora do Git): `C:\PI3\logs_s6\progresso.json`
