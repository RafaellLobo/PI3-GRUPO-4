# Diagnóstico operacional — bloqueios da Etapa 6 do benchmark S6

**Data:** 03/11/2026 · **Base:** `estado_execucao_plano.md`, `main` = `a0dcae9`, S4 = `feat/s4-etapa3-validacao` (`793999b`), S6 = `feat/s6-benchmark-executor`, PR #11 em `9b172f2` (Draft).
**Natureza:** diagnóstico. **Não é aprovação, adendo nem registro.** Os blocos de código abaixo são **modelos com campos a preencher**; nenhum deles vale como registro (as guardas ignoram blocos cercados e exigem a linha na coluna 0).

---

## 1. Achado que muda a ordem das ações

A configuração canônica da P9 (`1cd08306…17d5`) **fixa o SHA-256 de seis arquivos da S4**, entre eles `src/incremental/config.py` (`cedb0f55…`) e `guardas.py`. Para registrar P1, P2 e P3 como aprovadas, as guardas da S4 exigem que `config.DECISOES` traga o commit de aprovação. **Isso altera `config.py` e, portanto, o SHA da configuração da P9.**

Consequência: **todas as edições nos seis arquivos fixados da S4 têm de ser feitas antes** de regerar a configuração da S6 e antes da ratificação e do registro da P9. Como a P9 ainda não foi aprovada, a Seção 19 do próprio pré-registro prevê a regeneração do arquivo e do SHA, anotados antes do registro. Não é um adendo; é a regeneração controlada prevista. O `SHA256_ESPERADO` dos testes e os pins de `adaptadores_s4.py` mudam junto. **Quem ratificar a P9 deve ratificar o novo SHA**, não `1cd08306…`.

## 2. Requisitos exatos de cada registro

### 2.1 P1, P2 e P3 da S4 (liberam a Etapa 3)
- **Documento:** `docs/sprint4/protocolo_valor_incremental.md` (`config.PROTOCOLO`).
- **Formato:** uma linha por decisão, na coluna 0, fora de bloco cercado e de comentário HTML, introduzida pelo commit da aprovação (ausente nos pais), com data ≤ data do commit e única no `HEAD`.
- **`valor_sha256`** (`guardas.hash_valor_decisao`), calculados hoje a partir dos valores que a Emenda 1 (E11) decidiu:

| Decisão | Conteúdo aprovado | `valor_sha256` |
|---|---|---|
| P1 | nenhum (`null`) | `74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b` |
| P2 | `EspecSpline` atual (`SPLINE_PROPOSTO`: diâmetro, 4 nós, grau 3, quantis, extrapolação linear, sem viés) | `99afebabfa1d0edb66e83ff553eeaa7fdfbd1fceb98ed971067287d25f5ebfc3` |
| P3 | `PAINEL_DECLARADO` (as 8 features da Seção 5.1, nessa ordem) | `2241b2632297445bcb93d3db43e708318d8fefe229bb5b0194a7ecf35d759ae8` |

Modelo (preencher data real; **não** copiar sem a decisão humana):
```
APROVACAO-DECISAO codigo=P1 status=aprovada data=AAAA-MM-DD valor_sha256=<hash da tabela>
```
- **Depois do commit de aprovação:** `src/incremental/config.py` — `DECISOES["P1"|"P2"|"P3"]` com `status=APROVADA`, `commit_revisao_protocolo=<hash completo de 40 hex>` e `valor` (P2: `SPLINE_PROPOSTO`; P3: `PAINEL_DECLARADO`; P1: nenhum). Só P1–P3 são exigidas pela Etapa 3.
- **Efeito colateral nos testes S4:** mudam `hash_configuracao` e as constantes `HASH_CONFIG_INCREMENTO_1` (`test_guardas.py`), `HASH_CONFIG_DA_REFERENCIA` e `SHA256_RELATORIO_SINTETICO` (`test_piloto.py`), além de testes que assumem todas as decisões pendentes (`TestEstadoReal`).

### 2.2 `S4_E3_REGISTRADA`
- **Documento:** `docs/sprint6/preregistro_p9_benchmark.md`, Seção 19 (hoje só no PR #11).
- **Pré-condição:** manifesto `etapa3_validacao` definitivo, elegível, `concluida`, **versionado** em `reports/sprint4/*.json`, com todas as saídas versionadas no `HEAD`, e o `HEAD` da execução ancestral do commit do registro.
- **`valor_sha256`:** `aprovacoes.sha256_manifesto_s4(manifesto)` (SHA-256 canônico do manifesto inteiro). Só pode ser calculado depois da Etapa 3 oficial.
```
APROVACAO-DECISAO codigo=S4_E3_REGISTRADA status=aprovada data=AAAA-MM-DD valor_sha256=<sha256_manifesto_s4>
```
- O manifesto é passado ao coordenador por `--manifesto-s4 reports/sprint4/<arquivo>.json`.

### 2.3 P9
- **Documento:** o mesmo, Seção 19. **Valor:** SHA-256 da configuração canônica **vigente após a Seção 1** (hoje `1cd08306…17d5`, que vai mudar).
- **Ordem exigida (`aprovacoes._verificar_p9`):** o JSON `docs/sprint6/config_benchmark_p9.json` e a anotação do SHA na Seção 19 já existem em **todos os pais** do commit de registro e no `HEAD`; o registro entra em commit **posterior**, na coluna 0, com data ≤ a do commit.
```
APROVACAO-DECISAO codigo=P9 status=aprovada data=AAAA-MM-DD valor_sha256=<SHA vigente da configuração>
```
- **Antes dela:** D2 confirmada documentalmente pela responsável pelo Exp. 26.0.1; D1–D12 ratificadas; PR #11 atualizado com a referência ao PR #10 já integrado.

## 3. Pendências humanas × técnicas

### 3.1 Humanas (não posso concedê-las nem simulá-las)
| ID | Decisão | Quem |
|---|---|---|
| H1 | Aprovar o adendo de reordenação (`adendo_reordenacao_s4_proposta.md`) **ou** decidir executar as Etapas 1 e 2 antes da 3 | responsável + revisor |
| H2 | Commit com as três linhas P1, P2 e P3 no protocolo da S4 | responsável + revisor |
| H3 | Autorizar a execução oficial do item 2 da Etapa 0 e da Etapa 3 | responsável + revisor |
| H4 | Registro `S4_E3_REGISTRADA` | responsável + revisor |
| H5 | Confirmação documental de D2 (variável da faixa) pela responsável do Exp. 26.0.1 | responsável do experimento |
| H6 | Ratificação de D1–D12 sobre o SHA vigente e registro `P9` | autoridades previstas na P9 |
| H7 | Autorização de merge dos PRs (S4, PR #11, S6) | revisor |
| H8 | Esquema de permutação da P1 ("por paciente, em blocos"): tamanho dos blocos e tratamento dos folds sob rótulos permutados (a E11 só diz isso) — necessário apenas para a Etapa 1 oficial | responsável |

### 3.2 Técnicas
| ID | Tarefa | Depende de | Arquivos |
|---|---|---|---|
| T1 | Atualizar `config.DECISOES` e as constantes dos testes S4 | H2 | `src/incremental/config.py`, `tests/incremental/test_guardas.py`, `test_piloto.py` |
| T2 | Regerar a configuração da S6 (`--modo configuracao --gerar`), atualizar pins e `SHA256_ESPERADO`, atualizar JSON e anotação do PR #11 | T1 | `src/benchmark_s6/adaptadores_s4.py`, `tests/benchmark_s6/test_config.py`, `test_aprovacoes.py`, `docs/sprint6/config_benchmark_p9.json`, `preregistro_p9_benchmark.md` §19 |
| T3 | Executar o item 2 da Etapa 0 e a Etapa 3 oficiais; versionar relatório e manifesto | T1, H1, H3 | `reports/sprint4/etapa0_item2/`, `reports/sprint4/etapa3_validacao/` |
| T4 | Calcular `sha256_manifesto_s4` e conferir com `aprovacoes.validar_manifesto_s4` | T3 | — |
| T5 | Rebase da S6 sobre o `main` com a S4; `git diff main -- src/incremental` vazio | H7 | — |
| T6 | `--modo definitivo` (250 unidades) e `--modo verificar` | tudo acima | `reports/sprint6/benchmark/` |

## 4. O que se pode implementar ou testar agora, sem aprovação adicional
- **T1 e T2 em rascunho local**, sem registrar nada: depois do H2 ficam mecânicas. O que se pode preparar já é o procedimento e os testes, mas **não** a marcação de `APROVADA`, que exige um commit de aprovação real.
- **Execução do item 2 da Etapa 0:** as guardas já a autorizam (não exige decisões; lê só `train`). Falta apenas a sua autorização (H3).
- **Nada em `src/benchmark_s6` ou nas guardas precisa mudar** para o caminho acima, além de pins e constantes de hash (T2).

## 5. O que está expressamente bloqueado
- **Registrar** qualquer linha `APROVACAO-DECISAO`, marcar decisões como aprovadas ou criar manifesto "definitivo" da Etapa 3 sem as aprovações.
- **Etapas 1 e 2 oficiais da S4:** sem código oficial e sem H8 (e, na Opção A, são o caminho crítico: controle por permutação K = 100, cerca de 2 a 9 h de computação).
- **Sensibilidades da Etapa 3** (spline M0s/M3s, lista reproduzida, `C = 0,1`): as guardas só autorizam M0–M4 e derivam a composição de M2–M4 do painel aprovado em P3; uma lista diferente ou um modelo novo seria recusado. Implementá-las exige alterar as guardas ou definir uma atividade própria, e isso é decisão metodológica/de governança, não técnica. O protocolo as prevê para a Etapa 3 e para a Etapa 5; **sem decisão, a Etapa 3 oficial sairia incompleta em relação ao protocolo.**
- **Treino definitivo da S6, leitura de `validation` pela S6 e qualquer leitura de `test`.**

## 6. Menor sequência para concluir a Etapa 3 e liberar a S6 (Opção B)

1. **H1** — aprovar o adendo.
2. **H2** — commit com P1, P2, P3 no protocolo da S4.
3. **T1** — `config.DECISOES` + constantes de teste (PR da S4).
4. **H8/decisão sobre as sensibilidades** — definir se entram na Etapa 3 e como (Seção 5).
5. **H3 → T3** — Etapa 0 item 2 e Etapa 3 oficiais (git limpo); versionar os artefatos agregados.
6. **T2** — regerar a configuração da S6 sobre o `config.py` final da S4; atualizar o PR #11.
7. **H5, H6** — confirmar D2, ratificar D1–D12 sobre o novo SHA, e só então o registro `P9`.
8. **T4 → H4** — `S4_E3_REGISTRADA`.
9. **H7 → T5** — integrar S4, PR #11 e S6; **T6**.

As Etapas 1 e 2 oficiais ficam depois da Etapa 3 e antes da Etapa 4, e **não** estão no caminho crítico da S6 se o adendo for aprovado. Na Opção A, elas entram entre os passos 3 e 5.

**Ordem de compute estimada:** Etapa 0 item 2 e Etapa 3 oficiais são rápidas (minutos); a S6 definitiva projeta 0,3 a 0,6 h com 4 processos (F7). A restrição real é humana e de desenvolvimento das Etapas 1 e 2.
