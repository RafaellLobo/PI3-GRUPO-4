# Plano de execução autônoma — Benchmark Sprint 6 (P9)

**Projeto:** PI3-GRUPO-4 — Radiômica e Scoring de Risco (LIDC-IDRI)  
**Data de elaboração:** 03/11/2026  
**Finalidade:** instrução operacional para o Claude Code implementar, executar, verificar e relatar **exatamente** o benchmark aprovado, sem ampliar o escopo.  
**Card:** [S6 P0 — Benchmark LR, Random Forest, XGBoost e SVM](https://trello.com/c/06w7Zbh9)  
**Referência normativa primária:** [`docs/sprint6/preregistro_p9_benchmark.md` (PR #11)](https://github.com/RafaellLobo/PI3-GRUPO-4/pull/11); configuração [`docs/sprint6/config_benchmark_p9.json`](https://github.com/RafaellLobo/PI3-GRUPO-4/blob/docs/s6-p9-preregistro-benchmark/docs/sprint6/config_benchmark_p9.json).  
**Referência normativa relacionada:** Emenda 1 da S4, integrada pelo PR #10; observar os documentos reais do repositório e sua versão vigente.  
**Natureza:** plano subordinado ao P9 e às aprovações formais. **Não é adendo, não aprova protocolo e não altera configuração.**

---

## 0. Contrato de autonomia e precedência

### 0.1 Instrução principal

Execute as oito etapas deste documento **na ordem**. Trabalhe autonomamente nas operações técnicas autorizadas, sem pedir autorização para cada arquivo, teste, commit local, instalação em ambiente isolado ou ajuste de implementação **que preserve o protocolo**. Ao terminar uma etapa, valide seus critérios de aceite, registre a evidência e prossiga à seguinte apenas se o portão estiver satisfeito. Não encerre após entregar somente código: o objetivo final é produzir **resultados efetivos, métricas e figuras F1–F8**, desde que todos os portões metodológicos estejam satisfeitos.

Se um portão não puder ser atendido com evidência real, **pare naquele ponto**, registre `BLOQUEADA` e a causa objetiva, preserve o trabalho válido já concluído e informe a ação externa necessária. **Nunca simule aprovação, conclusão, resultado, execução ou integração.** Não substitua a execução definitiva por uma execução reduzida. Não use dados de teste para resolver bloqueios.

### 0.2 Ordem de autoridade

1. Registros formais válidos de aprovação e adendos já versionados no Git.
2. P9 na versão formalmente aprovada, incluindo sua configuração canônica e SHA-256.
3. Emenda 1 e protocolo da S4 vigentes, incluindo as condições para `S4_E3_REGISTRADA`.
4. Este plano operacional.

Em caso de conflito, **não escolha arbitrariamente uma versão**: registre o conflito e bloqueie a operação afetada até regularização. Nunca modifique os parâmetros canônicos para adaptar o projeto ao plano.

### 0.3 Decisões fornecidas pelo usuário — a documentar, não presumir como aprovação coletiva

- O usuário concorda com as decisões **D1–D12** do pré-registro P9.
- A variável utilizada no Experimento 26.0.1 para a faixa foi **`original_shape_Maximum3DDiameter`**, com **`12 <= D < 30` mm**. Registrar a declaração com autoria/data e rastrear a evidência no DoD. A declaração do usuário não deve ser atribuída artificialmente à responsável original do experimento; se o protocolo exigir confirmação documental dela, este requisito permanece pendente até comprovação.
- O usuário concorda em **propor** um adendo permitindo realizar a Etapa 3 da S4 antes das Etapas 1 e 2, **preservando a obrigatoriedade das três etapas, análises e limitações**. Isso não autoriza alterar a ordem sem o adendo formalmente aprovado.
- A concordância individual **não substitui ratificação ou aprovação formal do grupo**. Não redigir registros afirmando aprovação de pessoas que não a concederam.

### 0.4 Escopo fechado

**Dentro do escopo:** implementação do pacote `src/benchmark_s6/`, testes em `tests/benchmark_s6/`, governança documental exigida, execução do P9, artefatos agregados, relatórios, figuras e manifesto, exatamente nas seções 3–19 do pré-registro.

**Fora do escopo:** incluir outras features radiômicas; novas famílias de classificadores; alterar target; usar o teste histórico para comparar/selecionar; introduzir calibração do SVM; mudar número de folds, grades, seeds, bootstrap, critérios C1–C4, política de IC, limiares, interpretações clínicas; editar `src/incremental`; refazer a S4 sem autorização; iniciar experimentos exploratórios; publicar informações por paciente ou nódulo; fazer merge, publicar PR ou atualizar Trello como se estivesse concluído sem evidências.

**Correções admissíveis:** erros de sintaxe, importação, compatibilidade de API sob versões fixadas, testes e implementação, **sem alterar decisões científicas ou hashes canônicos**. Mudança do protocolo exige adendo **prévio e aprovado**. Não recorrer a parâmetros menores ou execução parcial para cumprir prazo.

### 0.5 Estado conhecido ao elaborar o plano (revalidar antes de agir)

- PR #10: **merged** no `main`.
- PR #11: **open/draft**, pré-registro P9 ainda **proposto**, sem aprovação formal identificada no texto consultado.
- Hash canônico indicado no P9: `1cd08306786067c1e29229a93a3215e087b0db21da35bc05c595bd5c8a2117d5`. É o hash do **conteúdo JSON canônico**, não necessariamente dos bytes do arquivo no disco.
- `S4_E3_REGISTRADA`: **não presumir que existe**. Conferir registro, validade, manifesto e commit antes do modo `definitivo`.
- Branch do gerador de configuração citada no P9: `feat/s6-benchmark-configuracao`; commit `db4f476e91ee0893900f2ac19c5caa58ddc5a370`. Verificar existência e compatibilidade; não pressupor que já está integrada.

### 0.6 Relato e progresso

Manter registro operacional em **arquivo de acompanhamento local do plano** (por exemplo `reports/sprint6/benchmark/estado_execucao_plano.md`, respeitando privacidade e restrições de versionamento), com: etapa, status `PENDENTE|EM_EXECUCAO|CONCLUIDA|BLOQUEADA`, commit/versão, comandos, saídas, testes e impedimentos. Esta nota de acompanhamento **não modifica a configuração experimental**. Resumir o progresso ao usuário após cada etapa. Não divulgar predições individuais nem resultados externos do modo `implementacao`.

---

## 1. Etapa 1 — Governança, conformidade e liberações

**Objetivo:** assegurar que a execução só ocorra sob protocolo congelado e autorizações verdadeiras.

**Entradas:** PRs #10 e #11, Emenda 1 da S4 vigente, P9, `config_benchmark_p9.json`, histórico de commits, declaração D1–D12 e D2 do usuário.

**Fazer:**

1. Inspecionar `git status`, branches, PRs, documentos e commits relevantes; identificar o estado atual, sem assumir que o retrato de 03/11 continua válido.
2. Confrontar o P9 com a Emenda 1 já integrada; identificar referências desatualizadas e conflitos apenas documentais. Não alterar o desenho metodológico.
3. Registrar formalmente a concordância do usuário com D1–D12 e a informação sobre `Maximum3DDiameter`; **distinguir declaração do usuário de confirmação exigida da responsável pelo Experimento 26.0.1**.
4. Preparar ou conferir **adendo de reordenação** da S4: motivo, etapas preservadas, dependências, análises que continuam obrigatórias, limitações do reordenamento e data anterior às ações afetadas. Propor, **não aprovar em nome de outros**.
5. Conferir a configuração e o SHA-256; qualquer mudança anterior à aprovação requer regeneração controlada e documentação. Se não houver mudança, conservar exatamente o hash registrado.
6. Verificar requisitos e evidências de aprovação **P9** e **`S4_E3_REGISTRADA`** nos registros do Git. **Não inventar registros.** Se houver interface e permissão para colher aprovação, solicitar à parte autorizada; ausência impede a passagem ao modo definitivo.
7. Verificar se o pré-registro se encontra formalmente aprovado; PR em draft ou concordância individual não bastam.

**Esperado como retorno:** matriz curta de pendências e confirmações com fonte/commit; documentação atualizada sem extrapolar as declarações; hash validado; identificadores dos portões efetivamente liberados e daqueles pendentes.

**Critério de aceite da etapa:** auditoria documental concluída, divergências explicitadas, estado real de aprovações e dependências registrado. **A etapa documental pode terminar com bloqueio à execução definitiva**, mas nenhum passo posterior poderá ignorá-lo.

**Regra de avanço:** etapas 2–5 podem avançar em modo de implementação com as restrições do P9; etapa 6 exige **P9 aprovada, `S4_E3_REGISTRADA` válido e adendo formalmente aprovado se a ordem da S4 tiver sido alterada**. Não executar nem registrar uma Etapa 3 fictícia para liberar o benchmark.

---

## 2. Etapa 2 — Ambiente, configuração canônica e isolamento

**Objetivo:** preparar um ambiente reproduzível e conferir a identidade do experimento.

**Entradas:** seções 5, 13 e 14 do P9, configuração canônica e APIs públicas da S4.

**Fazer:**

1. Conferir e fixar **Python 3.9.13; scikit-learn 1.3.2; numpy 1.23.5; pandas 1.5.3; scipy 1.10.1; matplotlib 3.7.3; XGBoost 2.1.4**, conforme P9. Respeitar outras dependências efetivamente exigidas pelo código e pelo protocolo, sem alterar as versões fixadas.
2. Testar o XGBoost 2.1.4 com **dados sintéticos**: importação, fit, clone, `GridSearchCV`, repetição determinística bit a bit, carregamento de DLL em Windows quando aplicável. Bloquear mudança de versão sem adendo.
3. Implementar/conferir `src/benchmark_s6/config.py` e modo `configuracao`, reaproveitando apenas as APIs públicas expressamente autorizadas.
4. Conferir o JSON canônico versionado e SHA-256 `1cd08306786067c1e29229a93a3215e087b0db21da35bc05c595bd5c8a2117d5` **somente se a versão aprovada ainda corresponder a esse valor**. Não trocar hash automaticamente.
5. Implementar validações de ambiente, configuração, git limpo, identidade de código e entradas; verificar o teste de hash dos arquivos S4 usados pelo adaptador.
6. Respeitar a separação de `DIR_LOCAL` (checkpoints privados, fora de git e `data/`) e `reports/sprint6/benchmark/` (somente resultados agregados permitidos).

**Esperado como retorno:** ambiente verificável, teste de fumaça aprovado, modo `configuracao` funcional, hash reproduzido e evidências de isolamento.

**Critério de aceite:** smoke test, verificador de configuração e testes de integridade passam; nenhuma alteração não aprovada na configuração.

**Se bloqueada:** indicar especificamente pacote/versão, conflito de hash ou recurso ausente; não atualizar versão nem JSON por conveniência.

---

## 3. Etapa 3 — Dados, população e folds sem vazamento

**Objetivo:** preparar a coorte elegível, sem ler alvos/features do teste, garantindo validação agrupada por paciente.

**Entradas:** `data/modeling_table_sprint3.csv`, `reports/sprint3/baseline/patient_split.csv`, hashes oficiais na S4 e seções 3–4 do P9.

**Fazer:**

1. Implementar `dados.py`, `folds.py` e `adaptadores_s4.py`; este último é **o único arquivo** a importar a S4, só com símbolos públicos listados na seção 14.2 do P9.
2. Aplicar carga em duas fases: conferência estrutural autorizada de identificadores/partições e, depois, leitura de alvo/features **somente** das linhas elegíveis. Qualquer linha do teste na matriz de treinamento aborta.
3. Usar `target_binary` e **somente** `original_shape_MeshVolume`, `original_shape_Maximum3DDiameter`, `original_shape_Sphericity`, nessa ordem, `float64`; abortar em `NaN/Inf`.
4. No modo `implementacao`, restringir a `train`: **291 pacientes/417 nódulos**, uma repetição (`r=0`). **Não ler valores de validação**.
5. No modo `definitivo`, somente após as aprovações, usar `train + validation`: **353 pacientes/509 nódulos**, **258 da classe 0/251 da classe 1**, ordenação estável por `nodulo_id`.
6. No modo definitivo, conferir antes de qualquer ajuste a faixa `12 <= original_shape_Maximum3DDiameter < 30` mm: **226 nódulos, 187 pacientes, 135 positivos/91 negativos**. Divergência aborta; nunca ajustar contagem à força.
7. Construir folds externos `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42+r)` para `r=0..9`; internos `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1000+10*r+k)`; gerar as mesmas divisões para **todos os modelos**.
8. Garantir ausência de sobreposição de pacientes nas dobras e ajuste interno, presença de ambas as classes, cobertura por repetição e fingerprints idênticos. Tratar warnings do divisor como falhas, segundo P9.

**Esperado como retorno:** módulos de dados/folds testados; contagens e fingerprint agregados permitidos; erros explícitos para vazamento, inconsciência de população ou contagens divergentes.

**Critério de aceite:** testes de separação por paciente, alvo, features, ordenação, isolamento do teste e fingerprints passam. Contagens definitivas **só podem ser atestadas após liberação da execução definitiva**.

---

## 4. Etapa 4 — Modelos e ajuste interno conforme P9

**Objetivo:** implementar somente os cinco pipelines fixados, com busca interna idêntica ao pré-registro.

**Entradas:** seção 5 do P9 e JSON canônico aprovado.

**Fazer:**

1. Implementar `modelos.py` e `tuning.py` com:
   - `lr_fixa`: `StandardScaler -> LogisticRegression(C=1, penalty='l2', solver='lbfgs', class_weight=None, max_iter=1000, random_state=42)`; **referência histórica**.
   - `lr_tuning`: mesmo pipeline, com `C` em `numpy.logspace(-3, 3, 13)`.
   - `svm`: `StandardScaler -> SVC(kernel='rbf', probability=False, class_weight=None, random_state=42)`; grade `C=(0.01,0.1,1,10,100,1000)` × `gamma=(0.01,0.03,0.1,0.3,1)`; sem calibração.
   - `rf`: `RandomForestClassifier(n_estimators=500, criterion='gini', bootstrap=True, class_weight=None, n_jobs=1, random_state=42)`; `min_samples_leaf=(20,10,5,3,1)` × `max_features=(1,2,3)`.
   - `xgb`: `XGBClassifier(objective='binary:logistic', tree_method='hist', subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0, n_jobs=1, random_state=42)`; `max_depth=(1,2,3,4)` × `n_estimators=(100,300)` × `learning_rate=(0.03,0.1)` × `min_child_weight=(5,1)`.
2. Usar `GridSearchCV(scoring='roc_auc', cv=folds_internos_prevalidados, n_jobs=1, error_score='raise', return_train_score=False)` com grade explícita e `refit` determinístico.
3. Escolher maior AUC média interna; empates até `1e-9` pela **primeira configuração na ordem de simplicidade canônica**. Registrar escolha e ocorrência na borda da grade.
4. Padronizar apenas LR e SVM, dentro de cada pipeline treinado com a dobra apropriada. Não transformar RF/XGB por conveniência nem aplicar normalização global.
5. Escores externos: `predict_proba[:,1]` em LR/RF/XGB; `decision_function` no SVM. Não ativar `probability=True` no SVM.
6. Criar testes sintéticos de grids, ordem, desempate, clonagem, determinismo e ausência de vazamento de transformação.

**Esperado como retorno:** fábricas e tuning testados; parâmetros e grades iguais ao JSON canônico; SVM sem métricas probabilísticas.

**Critério de aceite:** testes reproduzem todos os componentes/grades canônicos sem adicionar hiperparâmetros, features ou outro modelo.

---

## 5. Etapa 5 — Coordenador, checkpoints e validação de implementação

**Objetivo:** construir o executor completo e validá-lo **sem produzir métricas externas no modo de implementação**.

**Entradas:** seção 6 e seções 14–15, 18 do P9; componentes das etapas 2–4.

**Fazer:**

1. Implementar os módulos previstos na seção 14.1 do P9: `aprovacoes.py`, `unidades.py`, `metricas.py`, `bootstrap.py`, `consolidacao.py`, `selecao.py`, `figuras.py`, `relatorios.py`, `privacidade.py`, `manifesto.py`, `coordenador.py`, `__main__.py`, além dos já criados. **Esta lista é a arquitetura prevista, não autorização para inventar novos subsistemas.**
2. Implementar comando canônico: `python -m src.benchmark_s6 --modo {configuracao|implementacao|definitivo|verificar} --checkpoints DIR_LOCAL --destino reports/sprint6/benchmark [--processos N]`.
3. Suportar checkpoints atômicos por `(repeticao, fold, modelo)`, metadados de hashes, retomada na mesma identidade, invalidação total se o hash de código mudar, execução paralela **entre unidades**, `n_jobs=1` dentro dos modelos e BLAS em uma thread.
4. Implementar estados `FT`, aviso, `DD`, `RI`, `NE`, `RD`, `IQ` exatamente segundo seção 6; nenhuma AUC imputada. Falhas técnicas pendentes tornam o experimento incompleto.
5. No `implementacao`, somente `train`, `r=0`; tuning interno é permitido, mas seus escores não são publicados/versionados. Predições externas são usadas **apenas para validar formato e finitude**, sem calcular, armazenar ou divulgar métricas externas de desempenho.
6. Executar invariante da LR fixa em relação ao baseline oficial, conforme coeficientes/intercepto/scaler e tolerâncias da seção 14.3.
7. Testar códigos de saída **0, 2, 3, 4, 5, 64**, guardas e modo `verificar`; `argparse` inválido deve retornar 64, não 2.
8. Rodar testes unitários e de integração, inclusive determinismo bit a bit, 1 versus 4 processos, retomada, proibição de acesso ao teste, validação bloqueada na implementação, privacidade e git limpo.
9. Conferir orçamento de execução **sem reduzir protocolo**: o teto de três horas é alarme, não condição para alterar parâmetro. Se previsão ultrapassar o teto, exigir adendo aprovado **antes** do definitivo.

**Esperado como retorno:** implementação e testes aprovados; `reports/sprint6/benchmark/implementacao/relatorio_execucao.json` e os registros permitidos; resumo de integridade **sem AUC externa**.

**Critério de aceite:** modo implementação completo e validado; `python -B -m unittest discover -s tests` sem falhas; nenhuma métrica externa criada nesse modo.

---

## 6. Etapa 6 — Treinamento definitivo, sem acesso ao teste

**Objetivo:** executar o benchmark integral fixado pelo P9.

**PORTÃO OBRIGATÓRIO:** antes do primeiro ajuste definitivo, verificar **todos**: P9 formalmente aprovada e hash canônico coincidente; `S4_E3_REGISTRADA` válido apontando para manifesto da Etapa 3 **realmente concluída e elegível**; adendo prévio aprovado caso se tenha reordenado S4; ambiente conforme seção 13; repositório limpo; hashes de dados/código/configuração corretos; XGBoost smoke test aprovado. Se qualquer ponto faltar, **BLOQUEAR ETAPA 6**. Não interpretar concordância do usuário como esses registros.

**Fazer quando liberada:**

1. Executar o modo `definitivo` na coorte completa `train+validation` de 353 pacientes/509 nódulos, sem ler alvo/features de teste.
2. Conferir previamente os denominadores da faixa `12 <= D < 30`; abortar se diferentes de 226/187/135.
3. Treinar os **cinco modelos** em **10 repetições × 5 folds externos**, cada um com o tuning interno previsto quando aplicável: **250 unidades `(r,k,modelo)`**. LR fixa não passa por busca.
4. Capturar escore externo de cada unidade apenas no modo definitivo, checkpoints locais privados, tempos, warnings, hiperparâmetros e fingerprints.
5. Retomar falhas técnicas com identidade idêntica quando permitido; se necessário corrigir código, invalidar todos os checkpoints e recomeçar, sem reutilizar resultados incompatíveis.
6. Assegurar que dobras, classes, critérios e denominadores usados por todos os modelos sejam os mesmos e que não haja FT pendente.
7. Não reportar resultados parciais como finais; se houver falha/bloqueio, registrar status `incompleta`.

**Esperado como retorno:** 250 unidades completas e reproduzíveis; predições privadas locais preservadas; logs de tempo, warnings e hiperparâmetros; base íntegra para métricas finais.

**Critério de aceite:** 250 unidades válidas quanto à integridade técnica, sem FT pendente, conforme protocolo. Dobras degeneradas para um desfecho (`DD`) seguem as regras normativas, sem fabricar AUC.

---

## 7. Etapa 7 — Métricas, bootstrap, contrastes e seleção

**Objetivo:** consolidar os resultados sem reinterpretar o protocolo em função das AUC obtidas.

**Entradas:** resultados definitivos completos, seções 6–10 do P9.

**Fazer:**

1. Calcular AUC-ROC **dentro de cada dobra externa**, para `G` (todos os nódulos elegíveis da dobra) e `F` (somente `12 <= D < 30` na avaliação da dobra); não cortar a população usada no treinamento por faixa.
2. Em cada dobra/desfecho exigir **pelo menos 5 nódulos positivos distintos e 5 negativos distintos**; invalidar a AUC quando degenerada. Exigir pelo menos **4/5 folds válidos** por repetição e **8/10 repetições válidas** por desfecho; caso contrário rotular `RI`/`NE`, sem imputação.
3. Estimar `theta` como média das AUC médias das dobras válidas nas repetições válidas; diferenças pareadas `Delta(modelo)=theta(modelo)-theta(lr_tuning)`, mantendo a correspondência de folds/repetições e descrevendo sensibilidade às partições.
4. Fazer **bootstrap por paciente**, com reposição dos 353 pacientes, **B=10.000**, `numpy.random.default_rng(42)`, predições **fixas**, sem reajuste; aplicar ponderações por multiplicidade do paciente na dobra e validade por nódulos distintos conforme P9. Registrar `RD`; se mais de 1% de réplicas descartadas, marcar `IQ` e não relatar aquele IC.
5. Reportar IC **percentil 95% descritivo** para cada AUC de modelo; para a família de **seis contrastes** (`RF`, `XGB`, `SVM` versus `lr_tuning`, em `G` e `F`), reportar IC Bonferroni com percentis **0,41667 e 99,58333**, e também IC descritivo 95% em anexo/resultado. Reportar contrastes exploratórios somente como descritivos.
6. Calcular métricas adicionais **somente em G**: Brier, log-loss, calibração e sensibilidade/especificidade com limiar 0,5 para modelos probabilísticos; SVM só sensibilidade/especificidade pela regra `decision_function >= 0`, rotulada não comparável.
7. Aplicar seleção congelada: comparador padrão **LR com ajuste**. Para desafiante elegível exigir **C1** `Delta(G)>=0.02`, **C2** `Delta(G)>0` em pelo menos 80% das repetições válidas, **C3** `F` estimável e `Delta(F)>=0`, **C4** integridade completa/G estimável/sem aviso de não convergência do desafiante. Em múltiplos elegíveis, maior `Delta(G)`; diferença <0,005 usa ordem de simplicidade `LR < SVM < RF < XGB`. Nenhum elegível → LR ajustada.
8. Identificar otimismo de seleção do vencedor e incerteza condicional; não afirmar ganho clínico ou prova de superioridade. **Não calcular p-valor**, nem usar vocabulário vetado pelo P9.

**Esperado como retorno:** `denominadores.csv`, `auc_por_fold.csv`, `auc_por_repeticao.csv`, `contrastes.csv`, `bootstrap_resumo.json`, `bootstrap_delta.csv`, `hiperparametros_escolhidos.csv`, `metricas_descritivas.csv`, `selecao_algoritmo.json` e consolidação rastreável; campos/estados completos.

**Critério de aceite:** seis contrastes correspondem à família registrada; sem métricas inventadas, sem AUC imputada; seleção recalculável e independente do teste.

---

## 8. Etapa 8 — Gráficos, relatório científico, verificação e entrega

**Objetivo:** produzir a entrega integral do card, com resultados reais e interpretação metodologicamente limitada.

**Entradas:** artefatos finais da etapa 7, seções 11–12, 14.5 e 18 do P9.

**Fazer:**

1. Gerar **F1–F8**, cada uma em **PNG 300 dpi + SVG**, acompanhada por **CSV agregado** em `figuras/dados/`. Sem identificador ou ponto por nódulo. Paleta Okabe–Ito, eixos comparáveis e legendas com base, desfecho e natureza do intervalo.
2. **F1:** AUC dos cinco modelos em G/F, pontos por repetição, média e IC 95% descritivo; referência da LR fixa. **Não usar Bonferroni sobre AUC individual.**
3. **F2:** floresta dos seis contrastes, IC Bonferroni grosso, IC 95% fino, amplitude entre repetições; linhas em `0` e `delta=0,02`.
4. **F3:** distribuição de Delta por fold para cada contraste/desfecho, com contagens de folds degenerados.
5. **F4:** frequências das configurações de hiperparâmetros selecionadas e destaque dos valores na borda da grade.
6. **F5:** ROC média vertical por modelo para G, grade de 101 pontos de FPR, com percentis de dispersão entre as 50 curvas e legenda **“dispersão entre dobras — não é intervalo de confiança”**.
7. **F6:** calibração por decis para modelos probabilísticos, predições externas de `r=0` em G; fundir decis com menos de 10 nódulos conforme P9.
8. **F7:** denominadores de F por dobra: nódulos, positivos, negativos e marcação de validade.
9. **F8:** histograma dos Deltas bootstrap de cada contraste da família, IC e taxa de descarte.
10. Produzir `relatorio_benchmark.json` com esquema versionado e números suficientes para recalcular a seleção; `resumo_benchmark.md` contendo **todos os 15 itens da seção 12 do P9**; `relatorio_execucao.json`; `manifesto_benchmark.json` com hashes da entrada, código, configuração e saídas; `config_benchmark.json` como cópia da canônica.
11. Incluir tabela principal por modelo e desfecho, AUC e IC, contrastes, sensibilidade à partição, configuração escolhida, critérios C1–C4, população e denominadores, falhas e avisos, discussão de viés de seleção, limitações dos intervalos e declaração explícita de **não uso do teste**.
12. Rodar `python -m src.benchmark_s6 --modo verificar` para reproduzir fingerprints e hashes e executar `python -B -m unittest discover -s tests`. Verificar determinismo bit a bit, 1 versus 4 processos, integridade/privacidade, ausência de `LIDC-IDRI-\d{4}`, vocabulário permitido e ausência de edições em `src/incremental`.
13. Conferir o **DoD do card**; organizar links reais para commits/PR/relatórios. **Não marcar Trello como concluído sem resultados reais e DoD integral**; se houver revisão humana exigida, indicar a evidência pendente.
14. Apresentar **resumo final inteligível** com: população, modelos, AUC G/F e IC, principais Deltas e ICs, algoritmo recomendado pela regra, alertas, acesso às figuras F1–F8, diretório de artefatos, testes e hashes. Se não houver elegível, relatar LR com ajuste sem fabricar vencedor alternativo.

**Esperado como retorno:** todas as tabelas e figuras F1–F8, relatório científico, JSON reproduzível, manifesto, testes bem-sucedidos e links válidos às evidências. Se algum desfecho for `NE`, reportar como não estimável, sem preencher números fictícios.

**Critério de aceite:** checklist integral da **seção 18 (33 itens)** do P9 validado ou cada exceção explicitamente identificada como bloqueio; `modo verificar` e suíte de testes aprovados; entrega científica final verdadeira.

---

## 9. Matriz global de portões e condições de parada

| Situação | Pode avançar? | Ação obrigatória |
|---|---|---|
| P9 ainda sem aprovação formal | **Somente etapas 1–5**, sem métricas externas | Solicitar aprovação, não fabricar registro |
| Adendo de reordenação da S4 ainda não aprovado | **Não alterar ordem da S4** | Preparar e submeter adendo; manter bloqueio |
| `S4_E3_REGISTRADA` ausente/inválido | **Não entrar na etapa 6** | Concluir/validar E3 de verdade e registrar manifesto elegível |
| Ambiente, hash, versão ou entradas divergentes | **Não treinar definitivamente** | Corrigir só implementação compatível; adendo para mudanças científicas |
| XGBoost smoke test falha | **Não substituir versão por conta própria** | Reportar evidência e exigir adendo para mudança |
| Execução exige mais que teto planejado | **Não reduzir grades/folds** | Alarme e necessidade de adendo prévio, conforme P9 |
| Falha técnica em alguma unidade | **Não consolidar final** | Retomar unidade idêntica ou reexecutar tudo se código mudou |
| Desfecho `NE` ou IC `IQ` | Pode produzir relatório **com ressalvas normativas** | Marcar não estimável/IC não relatado; nunca imputar |
| Resultado contraria expectativa | **Não mudar critérios** | Aplicar C1–C4 como estão |
| Todos os portões/DoD passam | **Concluir** | Entregar métricas, oito figuras, relatórios e rastreabilidade |

## 10. Inventário da entrega — verificar antes de declarar conclusão

### Código e testes

- [ ] Pacote completo `src/benchmark_s6/` conforme seção 14.1 do P9.
- [ ] Testes `tests/benchmark_s6/` e suíte global passando.
- [ ] Sem alteração de `src/incremental`.
- [ ] Ambiente, configuração, hashes e aprovações validados.
- [ ] Checkpoints locais privados com identidade verificável.

### Resultados agregados versionáveis

- [ ] `reports/sprint6/benchmark/config_benchmark.json`
- [ ] `reports/sprint6/benchmark/plano_folds.json`
- [ ] `reports/sprint6/benchmark/denominadores.csv`
- [ ] `reports/sprint6/benchmark/auc_por_fold.csv`
- [ ] `reports/sprint6/benchmark/auc_por_repeticao.csv`
- [ ] `reports/sprint6/benchmark/contrastes.csv`
- [ ] `reports/sprint6/benchmark/bootstrap_resumo.json`
- [ ] `reports/sprint6/benchmark/bootstrap_delta.csv`
- [ ] `reports/sprint6/benchmark/hiperparametros_escolhidos.csv`
- [ ] `reports/sprint6/benchmark/metricas_descritivas.csv`
- [ ] `reports/sprint6/benchmark/selecao_algoritmo.json`
- [ ] `reports/sprint6/benchmark/relatorio_benchmark.json`
- [ ] `reports/sprint6/benchmark/resumo_benchmark.md`
- [ ] `reports/sprint6/benchmark/relatorio_execucao.json`
- [ ] `reports/sprint6/benchmark/manifesto_benchmark.json`
- [ ] Figuras `F1`–`F8` em PNG/SVG, com CSV agregado correspondente.

### Conformidade final

- [ ] P9 e `S4_E3_REGISTRADA` autenticados em commits antecedentes ao definitivo.
- [ ] Conformidade com o adendo aprovado, se houver inversão da ordem da S4.
- [ ] 250 unidades sem FT pendente; regras DD/RI/NE/RD/IQ respeitadas.
- [ ] AUC geral e faixa [12,30) relatadas corretamente, com ICs aplicáveis.
- [ ] Seis contrastes previstos; intervalos e Bonferroni corretamente rotulados.
- [ ] Algoritmo selecionado por C1–C4, sem avaliar o teste histórico.
- [ ] Varredura de privacidade, vocabulário e reprodutibilidade passa.
- [ ] Todos os 33 critérios da seção 18 do P9 verificados.
- [ ] Links reais para resultados e evidências; revisão humana identificada quando exigida.

## 11. Formato obrigatório do retorno final do Claude Code

Responder, na ordem:

1. **Status:** `CONCLUIDO`, `INCOMPLETO` ou `BLOQUEADO`, indicando etapa e motivo preciso.
2. **Rastreabilidade:** branch, HEAD, P9 e adendos aprovados, SHA-256 canônico, manifesto E3, hashes de dados/código/saídas.
3. **Resumo da coorte:** contagens gerais, por classe e faixa; confirmação de não acesso ao teste.
4. **Tabela científica:** modelos (LR fixa, LR ajustada, RF, XGB, SVM), AUC G, IC G, AUC F, IC F; não estimável quando aplicável.
5. **Contrastes:** seis Deltas, IC Bonferroni e IC 95% descritivo, sensibilidade às partições.
6. **Escolha:** cumprimento de C1–C4 para cada desafiante, vencedor e ressalva sobre otimismo de seleção.
7. **Gráficos:** links reais para F1–F8 (PNG e SVG) e dados agregados.
8. **Qualidade:** testes, comando `verificar`, contagem de FT/DD/RI/NE/RD/IQ e warnings, integridade e privacidade.
9. **Limitações:** as fixadas nas seções 8.3, 9.2, 10.3 e 16 do P9; nenhuma alegação clínica.
10. **Pendências reais**, se existirem, sem declarar conclusão do DoD.

**Não exibir resultados de exemplo como se fossem observados. Não relatar estimativa, AUC, figura ou hash que não tenha sido efetivamente produzido e validado.**

---

## 12. Comando inicial sugerido para Claude Code

> Leia integralmente `PLANO_EXECUCAO_AUTONOMA_BENCHMARK_S6.md`, o pré-registro P9 e sua configuração canônica. Consulte apenas as skills pertinentes à tarefa atual. Siga autonomamente as oito etapas, em ordem, sem ampliar escopo e registrando evidências e critérios de aceite. Não solicite confirmação para decisões puramente técnicas já cobertas pelo plano. Respeite estritamente os portões de aprovação: nunca invente aprovações, nunca acesse o teste para seleção e nunca altere parâmetros congelados para acelerar. Se houver bloqueio externo, interrompa a fase afetada e reporte a pendência com evidência. Se todos os portões forem satisfeitos, prossiga até a Etapa 8 e apresente resultados reais, AUC/IC/contrastes, seleção, figuras F1–F8 e relatório final.
