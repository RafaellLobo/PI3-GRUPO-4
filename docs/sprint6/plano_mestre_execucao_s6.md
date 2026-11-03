# S6 — Plano mestre de implementação e execução do benchmark P9

**Natureza:** roteiro operacional de engenharia, não adendo nem aprovação científica.  
**Referências prioritárias:** `docs/sprint6/preregistro_p9_benchmark.md` (PR #11), `docs/sprint6/config_benchmark_p9.json`, Emenda 1 da S4 (PR #10).  
**Objetivo:** terminar o software, verificar o pipeline com dados sintéticos, efetuar o ensaio técnico permitido e, **após aprovação**, executar as 250 unidades do benchmark definitivo e gerar todos os artefatos da P9.  
**Local planejado no repositório:** `docs/sprint6/plano_mestre_execucao_s6.md`.

> Este plano **não substitui** a P9, a configuração canônica, a Emenda 1 nem os registros de aprovação. Em conflito, prevalece o protocolo científico aprovado. O agente pode implementar e verificar, mas não pode conceder aprovação em nome da equipe ou forçar a passagem dos controles.

## 0. Ponto de partida — confirmar localmente

| Recurso | Estado reportado em 03/11/2026 |
|---|---|
| `main` | merge PR #19: `c43afe8f6b354fd33871270e80ec4e98a364cf87` |
| Branch de trabalho | `feat/s6-benchmark-executor` |
| Trabalho local | novos `executor.py`, `test_executor.py` e alteração em `unidades.py`, **sem commit** |
| Testes | 397 coletados; 394 passaram e 3 pulados na `.venv` (XGBoost ausente) |
| XGBoost | 2.1.4 validado em ambiente isolado; fixação no ambiente final pendente |
| SHA canônico P9 | `1cd08306786067c1e29229a93a3215e087b0db21da35bc05c595bd5c8a2117d5` |
| PRs científicos | #10 e #11 em Draft, não aprovados |
| Etapa 3 definitiva da S4 | **não há evidência apresentada de conclusão**; verificar, não presumir |
| Checkpoints | módulo integrado ao main; caminho sugerido `C:\PI3\checkpoints_s6` |
| Coordenador no main | somente `--modo configuracao` implementado |

**Primeira ação:** `git status --short`, `git branch --show-current`, `git log -1 --oneline`, inspeção do diff local. **Não executar** `git reset --hard`, `git clean` ou qualquer ação que descarte arquivos não commitados.

## 1. Contrato científico e técnico imutável

### Dados, validação e modelos

- `implementacao`: **somente `train`**, 291 pacientes / 417 nódulos; r=0, 5 dobras externas, 5 configurações, **25 unidades**. Permite treinamento técnico e tuning, mas **nenhuma métrica externa** de desempenho, nenhum acesso a valores de `validation` ou do teste.
- `definitivo`: **`train + validation`**, 353 pacientes / 509 nódulos, 10 repetições × 5 dobras × 5 configurações = **250 unidades**. A partição histórica `test` fica fora da modelagem; apenas identificadores podem ser usados nas verificações estruturais autorizadas.
- Três características, nesta ordem: `original_shape_MeshVolume`, `original_shape_Maximum3DDiameter`, `original_shape_Sphericity`; `float64` finitas.
- Folds `StratifiedGroupKFold` agrupados por `patient_id`; sementes externas `42+r`, internas `1000+10r+k`. Mesmas dobras para todos os modelos; sem vazamento entre treino e avaliação.
- Desfechos **G** (AUC geral por dobra) e **F** (AUC por dobra no subconjunto `12 <= Maximum3DDiameter < 30` mm). O recorte F é **somente da avaliação**; os estimadores treinam com todas as observações do treino da dobra. Conferir no definitivo: F=226 nódulos / 187 pacientes / 135 malignos / 91 benignos. A variável exata da faixa ainda depende da confirmação documental D2.

| Código | Papel | Grade fixada pela P9 | Fits aproximados por unidade, incluindo refit |
|---|---|---|---:|
| `lr_fixa` | referência | sem tuning, C=1 | 1 |
| `lr_tuning` | comparador | 13 valores de C | 66 |
| `svm` | desafiante | 6 C × 5 gamma | 151 |
| `rf` | desafiante | 5 min_samples_leaf × 3 max_features, 500 árvores | 76 |
| `xgb` | desafiante | 4 depth × 2 estimators × 2 learning rates × 2 child weights | 161 |
| **Total** | 5 configurações de 4 algoritmos | | **455 por dobra; 22.750 fits no definitivo** |

**Não modificar** grades, repetições, sementes, árvores, n_jobs internos=1, número de dobras, 10.000 réplicas de bootstrap, regra de seleção ou método de pontuação para ganhar tempo. O SVM usa `decision_function`; LR/RF/XGB usam `predict_proba[:,1]`.

### Inferência exigida

- Estados **FT, AVISO, DD, RI, NE, RD, IQ**, sem imputar AUC para dobras degeneradas. Validade por desfecho: pelo menos 5 nódulos positivos e 5 negativos *distintos* por avaliação; repetição com ≥4/5 dobras válidas; desfecho estimável com ≥8/10 repetições.
- Consolidação pareada por repetição/dobra: θ̂ por modelo/desfecho e Δ̂ contra `lr_tuning`, mais sensibilidade às partições.
- Bootstrap **por paciente**, 10.000 réplicas, `default_rng(42)`, predições fixas sem refit. IC descritivo de 95% para cada θ̂; Bonferroni **somente** para os seis contrastes da família RF/XGB/SVM × G/F (percentis 0,41667 e 99,58333). Descartes >1% → IQ sem IC no desfecho, sem alterar elegibilidade.
- Regra de seleção P9 §10: C1 ΔG ≥ 0,02; C2 vantagem em ≥80% das repetições válidas; C3 F estimável e ΔF ≥ 0; C4 sem FT pendente, G estimável e sem aviso de não convergência do desafiante; desempate por simplicidade. Sem elegível → `lr_tuning`. Declarar otimismo por seleção e limitações científicas.
- Métricas descritivas **somente G**: probabilísticas para LR/RF/XGB; SVM apenas sensibilidade/especificidade com limiar nativo, não comparável. Não produzir p-valores nem afirmações clínicas.

## 2. Política de execução e persistência

1. Trabalhar por **fases**, com verificações e commits **locais** incrementais. Não fazer push, merge, excluir histórico ou alterar PR #10/#11 sem ordem explícita.
2. Guardar este plano **versionado** em `docs/sprint6/`, mas **não editá-lo durante o treinamento**. Alterações no Git durante a execução podem invalidar a rastreabilidade e as verificações.
3. Manter **diário local fora do Git** (ex.: `C:\PI3\logs_s6\progresso.json`), atualizado a cada fase com: fase, estado, HEAD, arquivos, testes, problemas, tempo, artefatos e próximo passo. O diário não contém dados por paciente/nódulo.
4. Checkpoints em diretório validado, **fora do repositório e de `data/`**, não versionável (ex.: `C:\PI3\checkpoints_s6`). Nunca salvar `patient_id`, `nodulo_id`, rótulos ou pontuações individuais em `reports/`, PRs ou logs públicos.
5. Ao retomar uma sessão do Claude Code: ler o diário e verificar os fatos em `git status`, commits, arquivos, hashes e testes; **não** confiar apenas no status escrito.
6. Identidade de checkpoints inclui configuração, código, dados, ambiente e unidade. Mudança do código/ambiente após uma sessão definitiva começar → **não reutilizar** os checkpoints antigos. Não corrigir `.py` enquanto as 250 unidades rodam.
7. Um erro FT pendente ou integridade divergente significa execução **incompleta**, nunca conclusão parcial. Uma falha de aprovação impede o definitivo, mesmo que o software esteja pronto.
8. Testes **direcionados** durante desenvolvimento; suíte completa no marco final. Não fazer campanhas extensas de mutação sem defeito crítico concreto.
9. O plano não autoriza atividades que dependem de decisão científica humana: ratificar decisões, registrar aprovação, emitir adendo ou contornar guardas.

## 3. Etapas de implementação — sequência de execução

### F0 — Estabilizar executor e preservar trabalho local

**Módulos:** `executor.py`, `unidades.py`, `test_executor.py`, ponto de entrada `__main__.py`.

- [ ] Revisar o executor local e manter as alterações já testadas. Confirmar `ProcessPoolExecutor` com `spawn` no Windows, inicialização de contexto **uma vez por worker**, tarefas com hashes de X e da amostra, e resultados conferidos pelo processo principal.
- [ ] Garantir que só o processo principal abra/grave a `SessaoCheckpoints`; worker nunca grava.
- [ ] Validar determinismo bit a bit e equivalência para 1 e 4 processos, ordem canônica, falha de worker, retomada, checkpoint corrompido e hashes divergentes com dados sintéticos.
- [ ] **Corrigir o risco concreto do ponto de entrada:** no `main` examinado, `src/benchmark_s6/__main__.py` executa `sys.exit(main())` sem `if __name__ == "__main__"`. A guarda precisa ser verificada/corrigida antes de usar `spawn` na CLI.
- [ ] Rodar testes dirigidos, revisar diff, fazer **commit local** dos arquivos necessários e registrar no diário.

**Aceite:** execução paralela sintética reproduzível, sem escrita concorrente, sem inicialização recursiva do coordenador.

### F1 — Fixar ambiente oficial e XGBoost

**Módulos:** `environment.yml` e verificações pertinentes; **commit próprio** para dependência.

- [ ] Confirmar Python 3.9.13, sklearn 1.3.2, numpy 1.23.5, pandas 1.5.3, scipy 1.10.1, matplotlib 3.7.3 e **XGBoost 2.1.4** (P9 §13).
- [ ] Fixar a dependência da P9 sem atualizar desnecessariamente outras bibliotecas nem alterar o JSON canônico.
- [ ] Validar o XGBoost 2.1.4 no **mesmo ambiente** que executará o definitivo: DLL/import, fit, clone, GridSearchCV e determinismo real, sem skip.
- [ ] Registrar versões, dependências efetivas e hashes no diário; interromper se o ambiente não puder ser reproduzido.

**Aceite:** ambiente oficial íntegro e os testes reais de XGBoost passam.

### F2 — Métricas por dobra e consolidação

**Módulos:** `metricas.py`, `consolidacao.py`, testes.

- [ ] Consumir checkpoints individuais **somente localmente** e vincular predições a grupos e faixa F usando dados da mesma sessão.
- [ ] Calcular AUC por dobra externa para G e F; contagens de pacientes/nódulos/positivos/negativos; estados DD/RI/NE e proibição de imputação.
- [ ] Consolidar θ̂, contrastes Δ̂ pareados, médias por repetição, mínimos/máximos e frequência de Δ>0.
- [ ] Calcular métricas descritivas só em G; não aplicar probabilidade ao SVM nem comparar limiares nativos como equivalentes.
- [ ] Criar testes de referência independentes para dobras inválidas, duplicações, desfechos não estimáveis, pareamento e isolamento.
- [ ] Garantir por desenho e testes que `implementacao` **não chama** métricas externas.

**Aceite:** cálculos alinhados à P9 §§6–8, sem dados individuais nas saídas agregadas.

### F3 — Bootstrap e incerteza

**Módulos:** `bootstrap.py`, testes.

- [ ] Implementar 10.000 réplicas **por paciente**, gerador fixado, predições fixas, pesos multiplicativos, sem refit.
- [ ] Para cada réplica reavaliar dobras e desfechos segundo as regras de nódulos **distintos**, RI/NE e descarte RD.
- [ ] Gerar IC 95% descritivo para θ̂ e contrastes descritivos; Bonferroni nos **seis** contrastes da família, não nas AUC individuais.
- [ ] IQ quando mais de 1% das réplicas de um desfecho forem descartadas; IC não relatado, seleção não alterada.
- [ ] Testar pesos, percentis, determinismo, limites, descarte e ausência de reamostragem de nódulos no lugar de pacientes.

**Aceite:** resultados reproduzíveis e verificados com cálculo sintético independente.

### F4 — Seleção do algoritmo

**Módulos:** `selecao.py`, testes.

- [ ] Implementar exatamente os critérios C1–C4, referência `lr_tuning`, δ=0,02, 80% e desempate de 0,005 da P9 §10.
- [ ] Não usar IC ou taxa IQ como filtro de elegibilidade. Sem desafiante elegível → `lr_tuning`.
- [ ] Produzir justificativa por critério e declarar explicitamente otimismo por seleção.
- [ ] Testar fronteiras (igual a 0,02, 80%, ΔF=0, aviso de convergência, FT, desempate).

**Aceite:** seleção inteiramente recalculável a partir do relatório JSON.

### F5 — Figuras, relatórios, privacidade e manifesto

**Módulos:** `figuras.py`, `relatorios.py`, `privacidade.py`, `manifesto.py`, testes.

- [ ] Gerar **F1–F8** rigorosamente conforme P9 §11, cada uma em PNG 300 dpi, SVG e com CSV agregado. Paleta Okabe–Ito, sem pontos individuais. F1: IC 95% descritivo da AUC; F2: dois intervalos dos contrastes; F5: rotular faixa como **dispersão entre dobras, não IC**; F6: sem SVM.
- [ ] Gerar os artefatos obrigatórios P9 §§12 e 14.5, inclusive `config_benchmark.json`, `plano_folds.json`, `denominadores.csv`, `auc_por_fold.csv`, `auc_por_repeticao.csv`, `contrastes.csv`, `bootstrap_resumo.json`, `bootstrap_delta.csv`, `hiperparametros_escolhidos.csv`, `metricas_descritivas.csv`, `selecao_algoritmo.json`, F1–F8 e seus CSVs, `relatorio_benchmark.json`, `resumo_benchmark.md`, `relatorio_execucao.json`, `manifesto_benchmark.json`.
- [ ] Manifesto com SHA-256 de entradas, configuração, código, ambiente e saídas; relatório JSON versionado suficiente para reproduzir seleção.
- [ ] Validar **todos os arquivos versionáveis**: sem ids individuais, valores por nódulo, caminhos privados, `LIDC-IDRI-\d{4}` e expressões vetadas pela P9 §8.3.
- [ ] No modo `implementacao`, produzir **somente** relatório de execução, invariante LR fixa e manifesto exploratório, sem AUC externa, contrastes ou gráficos de desempenho.
- [ ] Testar completude das figuras, formato dos CSVs, privacidade, vocabulário e consistência de hashes.

**Aceite:** artefatos completos; nenhum dado individual sai do diretório local.

### F6 — Coordenador e integração de ponta a ponta

**Módulos:** `coordenador.py`, `__main__.py` e testes de integração/CLI.

- [ ] Implementar `--modo {configuracao|implementacao|definitivo|verificar} --checkpoints DIR_LOCAL --destino reports/sprint6/benchmark [--processos N]`.
- [ ] Códigos de saída P9: 0 sucesso, 2 bloqueio de aprovação, 3 falha técnica, 4 integridade, 5 privacidade/vocabulário, 64 erro de uso.
- [ ] No definitivo, exigir aprovação P9 e `S4_E3_REGISTRADA`, `oficial=True`, Git limpo, configuração canônica, ambiente válido, 353/509 e denominadores F antes de qualquer treinamento.
- [ ] No `implementacao`, apenas `train`, 25 unidades, invariante LR fixa e **sem qualquer métrica externa**, inclusive nos relatórios.
- [ ] No definitivo, concluir as 250 unidades antes de agregar, selecionar e publicar relatório final; FT pendente → `incompleta`.
- [ ] No `verificar`, reler/recalcular artefatos e SHA-256 sem treinar; compatibilizar verificação com a publicação posterior dos agregados versionados.
- [ ] Provar com testes sintéticos: CLI Windows e `spawn`, 1×4 processos, retomada, aprovação ausente, teste isolado, dado inválido, erro de privacidade, corrupção e hashes divergentes.
- [ ] Executar suíte completa **uma vez no código estabilizado**; documentar skips/falhas; fazer commits locais e garantir Git limpo.

**Aceite:** pipeline **tecnicamente completo**, validado ponta a ponta, pronto para execução segundo a autorização científica. Não iniciar definitivo nesta fase.

### F7 — Ensaio técnico permitido e projeção de runtime

**Pré-requisitos:** F0–F6 aprovadas.

- [ ] Executar somente `--modo implementacao` com `train`, 25 unidades, grades oficiais completas e XGBoost real; não calcular/registrar métricas externas.
- [ ] Usar checkpoints locais compatíveis com o HEAD estável; validar invariante da LR fixa.
- [ ] Medir tempo por modelo, inicialização, retomada, memória/processos e produzir projeção fundamentada para o definitivo.
- [ ] **P9 §15:** se projeção superar **3 horas**, não iniciar o definitivo sem **adendo científico aprovado previamente**. O teto é um alarme, não autorização para diminuir grades.
- [ ] Registrar tempos e observações no diário local; qualquer correção de código invalida os checkpoints de identidade anterior.

**Aceite:** tempo projetado com incertezas e condições de execução documentadas, sem resultado de desempenho externo.

### F8 — Regularização científica (dependência externa, em paralelo)

**O Claude só pode conferir e preparar documentação para aprovação, não aprovar sozinho.**

- [ ] **PR #10:** revisar e aprovar a Emenda S4 segundo a seção *Congelamento*, incluindo aprovação datada do revisor, situação de resposta do grupo, responsáveis e commit identificado. O merge sozinho **não** é aprovação.
- [ ] **PR #11:** confirmar documentalmente **D2** (variável da faixa conforme DoD Exp. 26.0.1), ratificar **D1–D12**, registrar decisões e congelar protocolo antes da avaliação definitiva.
- [ ] Confirmar SHA canônico vigente, anteriormente anotado; se configuração mudar, exigir novo SHA, adendo e nova aprovação.
- [ ] Registrar a aprovação **real** `P9` com SHA canônico em commit posterior ao commit da configuração já versionada/anotada, conforme P9 §19.
- [ ] Concluir efetivamente a **Etapa 3 definitiva da S4**, obter seu manifesto `etapa3_validacao` **concluído, elegível e versionado**, sem acesso indevido ao teste.
- [ ] Registrar `S4_E3_REGISTRADA` com o SHA-256 do **manifesto definitivo da S4** e verificar sua ancestralidade e integridade no Git.
- [ ] Usar a API pública de `aprovacoes.py` para confirmar que ambos os registros são válidos e não revogados; não falsificar datas, registros ou hashes.

**Aceite:** guardas de execução definitiva **autorizam por evidência real**. Enquanto F8 estiver incompleta, o benchmark definitivo **permanece bloqueado**.

### F9 — Treinamento científico definitivo (somente após F0–F8)

**Pré-voo obrigatório:** código e ambiente congelados; Git limpo; aprovação P9 e S4_E3 válidas; XGBoost 2.1.4; dados `oficial=True`; 353 pacientes/509 nódulos; faixa 226/187/135/91; projeção ≤3h ou adendo válido; armazenamento privado adequado; número conservador de processos.

**Comando-alvo, apenas quando liberado:**

```powershell
python -m src.benchmark_s6 --modo definitivo --checkpoints C:\PI3\checkpoints_s6 --destino reports/sprint6/benchmark --processos 4
```

`--processos 4` é exemplo, **não** requisito fixo. Escolher conforme RAM/CPU; não assumir aceleração linear. Manter o computador ativo e o ambiente inalterado.

- [ ] Executar **250/250 unidades** nas grades e sementes congeladas, sem usar `test` na modelagem.
- [ ] Persistir checkpoints por unidade; resultados individuais nunca são versionados. Retomar apenas unidades da **mesma identidade**.
- [ ] FT pendente, corrupção, ambiente alterado ou aprovação inválida → parar, registrar `incompleta`, **não** gerar seleção/relatório científico final.
- [ ] Após sucesso das 250 unidades, calcular G/F, consolidação, bootstrap B=10.000, seleção e todos os artefatos da P9.

**Aceite:** 250 unidades concluídas, zero FT pendente, relatórios e manifesto elegíveis; não declarar sucesso antes da fase F10.

### F10 — Verificação final e entrega

- [ ] Validar relatórios, estados DD/RI/NE/RD/IQ, denominadores, ICs, seleção C1–C4 e limites científicos.
- [ ] Examinar **todos** os CSV, JSON, Markdown, PNG/SVG versionáveis quanto a dados individuais e vocabulário; recusar publicação se falhar.
- [ ] Commitar **somente agregados permitidos** em `reports/sprint6/benchmark`, depois de conferir privacidade; preservar checkpoints e predições individuais fora do Git.
- [ ] Garantir árvore limpa e executar `--modo verificar`; exigir hashes reproduzidos e código 0.
- [ ] Registrar no diário local: HEAD de execução, commits de artefatos, tempo total/por modelo, processos, unidades, falhas/retomadas, localização das figuras e relatórios.
- [ ] Preparar resumo para a equipe sem afirmar validação clínica ou desempenho no teste. Não fazer push ou merge automaticamente.

**Aceite:** artefatos agregados completos, versão rastreável e verificação independente aprovada.

## 4. Dependências e pontos de decisão

```text
F0 executor ── F1 ambiente ── F2 métricas ── F3 bootstrap
                                   │               │
                                   └───────────────┴── F4 seleção
                                                         │
                                                         ▼
                                           F5 relatórios/figuras/privacidade
                                                         │
                                                         ▼
                                              F6 coordenador/testes
                                                         │
                                                         ▼
                                           F7 ensaio técnico/orçamento
                                                         │
          F8 aprovações científicas reais ────────────────┤
                                                         ▼
                                              F9 definitivo 250/250
                                                         │
                                                         ▼
                                                F10 verificação
```

F8 pode ocorrer **em paralelo** às fases técnicas. **F9 depende tanto de F7 quanto de F8**.

| Controle | Passagem permitida | Interromper |
|---|---|---|
| Engenharia | F0–F6, testes e integração aprovados | módulos faltando/teste crítico falho |
| Dependências | versões da P9, XGB real funcional | ambiente divergente |
| Dados | fontes e hashes oficiais, test isolado | acesso indevido/contagens diferentes |
| Governança | P9 e S4_E3_REGISTRADA válidos e Etapa 3 concluída | apenas merge ou aprovação ausente |
| Custo | projeção até 3h ou adendo aprovado | projeção >3h sem adendo |
| Estado Git/checkpoints | HEAD estável, árvore limpa e identidade íntegra | código alterado/checkpoint incompatível |
| Privacidade | relatórios agregados passam na varredura | IDs ou scores individuais no Git |

## 5. Diário de progresso local — exemplo

Salvar de forma atômica fora do Git, em `C:\PI3\logs_s6\progresso.json`. Este **exemplo não representa execução concluída**:

```json
{
  "formato": "s6-pipeline-progresso-v1",
  "fase": "F0",
  "estado": "PENDENTE",
  "head": null,
  "testes": {"aprovados": null, "pulados": null, "falhas": null},
  "fases_concluidas": [],
  "bloqueios": [],
  "proxima_acao": "Inspecionar branch e alterações locais do executor"
}
```

Para retomar, **validar** o diário contra Git, testes e manifestos; não confiar em contagens/estados declarados sem evidência. O diário não substitui checkpoint nem manifesto.

## 6. Evidências para apresentação à equipe

Ao concluir, disponibilizar visão geral com: (a) arquitetura dados → folds → tuning → unidades → checkpoints → métricas → bootstrap → seleção → relatórios; (b) objetivo e metodologia do benchmark com quatro algoritmos/cinco configurações; (c) isolamento entre pacientes e do teste; (d) aprovações e hashes; (e) 250 unidades, G/F, ICs, limites estatísticos e otimismo por seleção; (f) figuras F1–F8, relatório e comando de verificação.

**Definições de pronto:** `pipeline tecnicamente completo` após F6; `autorizado para execução definitiva` apenas após F7 e F8; `benchmark concluído` somente após F9 e F10. Não confundir esses três marcos.
