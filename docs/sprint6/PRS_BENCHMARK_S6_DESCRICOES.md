# Descrições preparadas dos PRs — S4 (Etapa 3), P9 (#11) e executor S6

**Natureza:** texto de apoio à publicação. **Nenhum PR foi aberto ou atualizado, nada foi enviado ao remoto e nenhum merge foi feito.** Os blocos "Descrição" podem ser colados nos PRs depois da autorização; os links só existem depois da publicação. Status de cada PR conforme o remoto em 03/11/2026: PR #10 (Emenda 1) **integrado** no `main` (`a0dcae9`); PR #11 (P9) **aberto, rascunho**, na ponta `9b172f2`; PRs #12–#19 (módulos do S6) **integrados**; **não existe** PR para a S4 (Etapa 3) nem para o executor.

## 1. Matriz de dependências (conferida com `git merge-base` e `git diff`)

| Branch | Base real | Commits à frente do `main` (`a0dcae9`) | Estado |
|---|---|---|---|
| `feat/s4-etapa3-validacao` (`25c843c`) | `main` (inclui o PR #10) | 12 | só local |
| `docs/s6-p9-preregistro-benchmark` (PR #11, `9b172f2`) | `7f3e1c2` (anterior ao PR #10) | 3 | remoto, rascunho |
| `feat/s6-benchmark-executor` (`22d6a9b`) | contém o `main`, a S4 e o PR #11 por merges locais | 37 | só local |
| `docs/s6-estado-final-benchmark` (`0cc22e0`) | `22d6a9b` | +1 | só local |
| `docs/s6-guia-revisao-benchmark` | `0cc22e0` | +2 (guia e descrições) | só local |

- O executor **contém** os commits da S4 e do PR #11 (`git merge-base --is-ancestor` confirmado para os dois). Os registros `P9` e `S4_E3_REGISTRADA`, o JSON regenerado da configuração e os artefatos congelados só existem na linha do executor.
- O PR #11, tal como está no remoto, traz o JSON e a Seção 19 **antigos** (SHA `1cd08306…`). A versão final está na linha do executor. Atualizar o PR #11 isoladamente não é possível sem o manifesto da S4 (exigido pelo registro `S4_E3_REGISTRADA`).
- **Restrição de verificabilidade:** o `--modo verificar` recusa mudanças fora de `reports/sprint6/benchmark/` depois da execução e exige o manifesto e o HEAD registrados. Foi aprovado em `22d6a9b`. Reescrever histórico (rebase, squash) ou integrar de forma que a execução (`c124c81`) deixe de ser ancestral invalida a cadeia de rastreabilidade e as guardas; um merge commit que não traga arquivos novos pode preservá-la, mas isso deve ser **conferido** (Seção 6).

## 2. Ordem proposta (sujeita à decisão da equipe)

1. **PR da S4** (`feat/s4-etapa3-validacao` → `main`): é independente do restante.
2. **PR do executor S6** (`feat/s6-benchmark-executor` → `main`), depois do passo 1: leva o código do benchmark, as 250 unidades, os registros do P9 e os artefatos. Como a linha contém os commits do PR #11, o GitHub o marcará como integrado quando o executor entrar no `main`; a equipe decide se o fecha antes (com remissão) ou o deixa seguir.
3. **PR de documentação** (`docs/s6-guia-revisao-benchmark` → `main`), por último: guia de revisão, descrições e estado final. Ele muda o HEAD, e por desenho o `verificar` não passará nesse HEAD; a evidência é `22d6a9b`.

Cada passo é uma autorização separada. **Push não autoriza merge.**

## 3. Descrição — PR da S4 (`feat/s4-etapa3-validacao`)

**Título sugerido:** `feat(s4): Etapa 3 oficial, sensibilidades, registros P1–P4 e adendo de ordem das etapas`

**Objetivo.** Implementar e executar a Etapa 3 (validação oficial, lista fixa, M0–M4) com as três sensibilidades, formalizar antes a ordem das etapas e registrar P1–P4.

**Escopo fechado (12 commits, 25 arquivos).**
- Código: `src/incremental/` (`bootstrap.py`, `etapa0.py`, `selecao.py`, `sensibilidades.py`, `validacao.py`, `config.DECISOES`); guardas, fábrica de modelos e checkpoints **não alterados**.
- Governança: [Adendo 1](../sprint4/adendo_01_ordem_das_etapas.md) (`c9c2a2b`, anterior a qualquer execução); registros `APROVACAO-DECISAO` de P1–P4 no protocolo (`d041987`); `config.DECISOES` (`7112548`).
- Evidências: `reports/sprint4/etapa0_item2/` (`ac8102f`) e `reports/sprint4/etapa3_validacao/` (`25c843c`).
- Testes: `tests/incremental/`, com os testes de bloqueio reescritos sobre o mapa sem registros.

**Resultados (agregados).** Etapa 3: 417 nódulos de treino → 92 nódulos / 62 pacientes de validação; M3 − M0, ΔAUC 0,0306, IC95% [0,0028; 0,0670] (leitura L > 0 da Seção 9); M4 − M1 0,0384; M2 − M0 0,0297; M1 − M0 −0,0078 (inconclusivo). Spline M3s − M0s 0,0235, IC [−0,0090; 0,0609]. Manifesto: SHA-256 canônico `1ad800ea62a85be1a38ddcd6b1c68566e573cf2ac55e8ae73f2770aa21ff5705`.

**Testes.** Suíte da S4: 508 aprovados, 0 falhas, 0 ignorados (execução de 03/11/2026, log local fora do Git).

**Governança.** Os registros foram lavrados a pedido do revisor com base na confirmação da equipe **relatada por ele**; o repositório não tem ata nem assinatura individual. P4 foi registrada como aprovada com valor 0,9762 (representação da decisão encerrada). Mudar `config.py` alterou `hash_configuracao` e o pin da S6 (tratado no PR do executor).

**Limitações e exclusões.** Esta Etapa 3 **não encerra a S4**: as Etapas 1 e 2 oficiais (obrigatórias, depois da 3), a circularidade (§10.3) e as máscaras reduzidas (§10.4) **não foram executadas**. Sem comparação com a CV aninhada, a leitura da Seção 9 fica incompleta. Sem dados individuais no repositório; o teste não foi lido.

**Dependências.** Nenhuma além do `main`. **Itens a revisar:** adendo; registros P1–P4; P4 como "aprovada 0,9762"; relatório e manifesto da Etapa 3.

**Checklist do DoD do card.** Parcial: a S4 é insumo do benchmark; ver o item do "diâmetro isolado" no guia (Seção 9).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 4. Descrição — PR #11 (P9), proposta de atualização

**Título atual:** `docs(s6): pré-registro P9 do benchmark de classificadores — proposta`

**Situação.** Rascunho no remoto, na ponta `9b172f2`, com a configuração **antiga** (`1cd08306…17d5`). A versão final do pré-registro e do JSON está na linha do executor (`001d715`, `451123f`, `c4cc126`, `c124c81`).

**Texto a acrescentar à descrição, se a equipe optar por mantê-lo.** O pré-registro final tem a configuração regenerada (`a0d096a5415244f17f40360a2a5a8901bc8af1da5cf8687b891f920e1ee122b3`): o **único** campo alterado é o pin do SHA-256 de `src/incremental/config.py`, consequência do registro de P1–P4 na S4; nenhum parâmetro mudou. A Seção 19 traz as confirmações da equipe (relatadas pelo revisor), a documentação das diferenças de limites de D2 ([12, 30] × [12, 30)) e os registros `P9` e `S4_E3_REGISTRADA`. **Rascunho não é aprovação de merge**, e os registros não substituem a revisão humana.

**Dependências.** Os registros dependem do manifesto da S4 (PR da S4) e do pin atualizado (PR do executor). **A decisão pendente é se o PR #11 é atualizado, fechado com remissão ou deixado para ser marcado como integrado pelo executor.**

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 5. Descrição — PR do executor S6 (`feat/s6-benchmark-executor`)

**Título sugerido:** `feat(s6): benchmark definitivo de classificadores — executor, 250 unidades, relatórios e figuras F1–F8 (P9)`

**Objetivo.** Entregar o executor do P9 (coordenador, consolidação, seleção, figuras, manifesto), a execução definitiva e os artefatos congelados.

**Escopo.** Além do que vem da S4 e do PR #11 (ver Seção 1): `src/benchmark_s6/` completo, `tests/benchmark_s6/`, pin de `config.py` atualizado (`e4c95e6`), configuração regenerada (`001d715`), registros `P9` (`c4cc126`) e `S4_E3_REGISTRADA` (`c124c81`) e `reports/sprint6/benchmark/` (`22d6a9b`, 39 arquivos mais o diretório `implementacao/` anterior).

**Resultados (agregados, descritivos e condicionais).** 353 pacientes / 509 nódulos; 250 unidades; 0 falhas técnicas pendentes; faixa F com 226 nódulos / 187 pacientes / 135 positivos / 91 negativos. AUC geral: LR fixa 0,9442; LR com ajuste 0,9449; SVM 0,9424; RF 0,9476; XGBoost 0,9438. AUC na faixa: 0,7903; 0,7907; 0,7890; 0,7968; 0,7844. Nenhum contraste contra a LR com ajuste tem intervalo com Bonferroni longe de zero, e nenhum desafiante cumpre C1–C4; a regra recomenda a **LR com ajuste**.

**Evidências e verificação.** Relatório `reports/sprint6/benchmark/resumo_benchmark.md`; configuração `a0d096a5415244f17f40360a2a5a8901bc8af1da5cf8687b891f920e1ee122b3`; `--modo verificar` aprovado em `22d6a9b` (38 saídas conferem, recálculo idêntico). Suíte da S6: 613 aprovados, 1 ignorado.

**Governança.** Registros lavrados a pedido do revisor com base na confirmação da equipe relatada por ele. A confirmação de D1–D12 foi sobre os parâmetros; o SHA da configuração mudou depois, só pelo pin. Evidências de D2 (`piso_26_0_1.py`, `26_0_1_resumo.json`) **não estão no repositório** e não foram inspecionadas.

**Limitações.** IC condicional; dependência entre dobras; otimismo da seleção; Bonferroni como convenção; alvo radiológico; bordas das grades; reutilização da validação da Etapa 3 da S4. Itens editoriais F4, F5 e F6 (layout), sem alterar dados.

**Dependências.** PR da S4 (antes). **Exclusões intencionais:** dados individuais, checkpoints, tabela-fonte, teste histórico, novos experimentos.

**Itens a revisar.** Aprovação do P9 como suprimento do requisito "diâmetro isolado" do card (decisão humana pendente); layout de F4/F5/F6; aceitação do SHA novo.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 6. Descrição — PR de documentação (`docs/s6-guia-revisao-benchmark`)

**Título sugerido:** `docs(s6): guia de revisão do benchmark, descrições dos PRs e estado final`

**Escopo.** [`GUIA_REVISAO_BENCHMARK_S6.md`](GUIA_REVISAO_BENCHMARK_S6.md) (F1–F8 por caminhos relativos, com leitura científica), este arquivo, o plano de publicação e o registro de estado. **Não altera** nada em `reports/sprint6/benchmark/`, nem código, nem registros. **Atenção:** por desenho, o `--modo verificar` falha em HEADs posteriores à execução; a evidência de integridade é `22d6a9b`. Depende do PR do executor.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 7. Comandos para quem for publicar (**não executados**)

```
# 1. Conferir o estado local
git -C C:\PI3\PI3-s6 branch -vv
# 2. Publicar as branches, sem force, na ordem escolhida (autorização de push)
git push -u origin feat/s4-etapa3-validacao
gh pr create --base main --head feat/s4-etapa3-validacao --title "..." --body-file <arquivo com a Seção 3>
# 3. Só depois da revisão e da autorização de merge (separada do push), integrar a S4;
#    em seguida, o executor e, por último, a documentação.
```
