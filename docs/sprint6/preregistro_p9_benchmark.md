# Pré-registro P9 — Benchmark de classificadores da Sprint 6

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Natureza:** protocolo complementar, comparativo-descritivo, **fora do escopo** da Seção 14 do protocolo da Sprint 4 (S4)
**Base normativa:** Emenda 1 ao protocolo da S4, revisão 3 — `docs/sprint4/emenda_01_auditoria_teste.md` no commit `573bc17` (PR #10, ainda não integrado ao `main`), Seções E3, E5, E6, E7, E10, E11 (P6 e P9) e E12
**Protocolo relacionado:** [`docs/sprint4/protocolo_valor_incremental.md`](../sprint4/protocolo_valor_incremental.md) — a versão do `main` ainda não traz as anotações da Emenda 1, que estão no commit `573bc17`
**Responsável:** Samara Ruiz · **Revisor e redator desta especificação:** Rafael Lazarin Lobo de Almeida
**Versão:** 1.0 — 02/11/2026
**Situação:** **PROPOSTA — não aprovada.** Nenhuma execução definitiva está autorizada. A confirmação da responsável e do revisor não equivale à aprovação formal do grupo. A decisão D2 depende de confirmação documental da responsável (Seção 17).

---

## 0. Resumo

Compara quatro classificadores — regressão logística com ajuste de `C` (LR-T), Random Forest (RF), XGBoost (XGB) e SVM de kernel RBF — com as **três características do baseline**, em **353 pacientes / 509 nódulos** (partições de treino e validação), por **validação cruzada aninhada 10 × 5 agrupada por paciente**, com as **mesmas dobras** para todos os modelos. A regressão logística fixa do baseline (LR-F, `C = 1`) é relatada como **referência histórica**. O conjunto de teste **não é lido**. A execução definitiva só ocorre **depois** de a Etapa 3 da S4 estar concluída e registrada. Os resultados são **descritivos e condicionais**: o estudo não é confirmatório, e nenhum resultado altera a S4.

## 1. Escopo e relação com a S4

1. O benchmark responde a uma pergunta só: *com as três características do baseline, nesta base, algum algoritmo mais flexível discrimina melhor do que a regressão logística com ajuste interno?*
2. Não avalia o valor incremental da radiômica, que é objeto da S4, e não usa as oito características radiômicas. Qualquer análise com radiômicas exige adendo datado e aprovado antes da execução (Emenda 1, E7).
3. Nenhum resultado deste benchmark altera modelo, característica, critério ou etapa da S4 (Emenda 1, E7, item 4).
4. Os números deste benchmark não se misturam com os da S4: as bases e as partições são diferentes (Emenda 1, E3, item 5).

## 2. Decisões já registradas na Emenda 1 (revisão 3, commit `573bc17`)

| Item | Conteúdo | Fonte |
|---|---|---|
| Base | 353 pacientes de treino + validação; nenhuma validação cruzada sobre os 416 | E11, P6 |
| Características | `original_shape_MeshVolume`, `original_shape_Maximum3DDiameter`, `original_shape_Sphericity` | Registro da revisão 3, ponto 3 |
| Algoritmos | Regressão logística, Random Forest, XGBoost e SVM | Registro da revisão 3, ponto 2 |
| Dobras | Validação cruzada agrupada por paciente; dobras idênticas para todos | Registro da revisão 3, ponto 3 |
| Teste | Não usado | E7 |
| Ordem | Os dois protocolos congelados antes dos experimentos definitivos; a validação só entra depois do registro da Etapa 3 da S4 | E7; registro da revisão 3, ponto 8 |
| Conteúdo deste pré-registro | Contrastes, intervalos de confiança, multiplicidade e regra de seleção fixados antes de executar | E11, P9 |
| Dados | Nada com `patient_id` ou valor por nódulo no repositório | E10 |

## 3. Dados, população e isolamento

| Item | Especificação |
|---|---|
| Fonte | `data/modeling_table_sprint3.csv` e `reports/sprint3/baseline/patient_split.csv`, com os SHA-256 fixados em `src/incremental/dados.py` (`TABELA_OFICIAL`, `SPLIT_OFICIAL`) |
| Linhas usadas | Partições `train` (291 pacientes / 417 nódulos) e `validation` (62 / 92) = **353 / 509**, sendo 258 de classe 0 e 251 de classe 1, ordenadas por `nodulo_id` (ordem estável) |
| Alvo | `target_binary` (regra oficial; Seção 3 do protocolo da S4) |
| Características | As três da Seção 2, nessa ordem, `float64`, sem `NaN` nem `Inf` (qualquer ocorrência aborta). Nenhuma transformação além da padronização dentro do pipeline de LR e SVM |
| Teste | **Exclusão integral do alvo e das características.** Os identificadores das linhas de teste só passam pela verificação estrutural pública da S4 (`dados.verificar_estrutura`), que lê identificadores e partição para conferir contagens. O carregador deste benchmark pula as linhas de teste antes de ler qualquer valor, e uma linha de teste na matriz aborta a execução |
| Validação | Lida **somente** no modo definitivo, depois do registro `S4_E3_REGISTRADA` (Seção 19) |

## 4. Partições

| Nível | Especificação |
|---|---|
| Externo | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + r)`, r = 0…9; estratificado por `target_binary` e agrupado por `patient_id` |
| Interno (ajuste) | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1000 + 10r + k)` sobre o treino externo k |
| Validação de cada divisão | Treino e avaliação não vazios e disjuntos; nenhum paciente nos dois lados; as duas classes nos dois lados; cada linha avaliada exatamente uma vez por repetição; nenhuma linha nem paciente da avaliação externa em qualquer divisão interna. Avisos do divisor são tratados como erro |
| Compartilhamento | A atribuição de dobras externas e internas é gerada uma vez e é **idêntica** para os cinco modelos; só o fingerprint (SHA-256 da composição) é versionado |

## 5. Modelos, grades e ajuste

### 5.1 Especificação

| Código | Papel | Componentes fixos | Grade, na ordem de simplicidade | Configurações |
|---|---|---|---|--:|
| `lr_fixa` | Referência histórica, só descritiva | `StandardScaler → LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", class_weight=None, max_iter=1000, random_state=42)` | — | 1 |
| `lr_tuning` | **Comparador** | Igual à LR fixa, menos `C` | `C` ∈ `numpy.logspace(-3, 3, 13)`, crescente | 13 |
| `svm` | Desafiante | `StandardScaler → SVC(kernel="rbf", probability=False, class_weight=None, random_state=42)`; **sem calibração** | `C` ∈ (0,01; 0,1; 1; 10; 100; 1000) × `gamma` ∈ (0,01; 0,03; 0,1; 0,3; 1), `C` antes de `gamma` | 30 |
| `rf` | Desafiante | `RandomForestClassifier(n_estimators=500, criterion="gini", bootstrap=True, class_weight=None, n_jobs=1, random_state=42)` | `min_samples_leaf` ∈ (20, 10, 5, 3, 1) × `max_features` ∈ (1, 2, 3) | 15 |
| `xgb` | Desafiante | `XGBClassifier(objective="binary:logistic", tree_method="hist", subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0, n_jobs=1, random_state=42)`, sem parada antecipada | `max_depth` ∈ (1, 2, 3, 4) × `n_estimators` ∈ (100, 300) × `learning_rate` ∈ (0,03; 0,1) × `min_child_weight` ∈ (5, 1) | 32 |

### 5.2 Ajuste interno

- `GridSearchCV(estimator, param_grid=<lista explícita na ordem acima>, scoring="roc_auc", cv=<lista de divisões internas validadas>, refit=<função de escolha>, n_jobs=1, error_score="raise", return_train_score=False)`.
- O critério é a média da AUC das cinco dobras internas. No SVM, a AUC usa `decision_function`.
- **Escolha:** a maior média. Diferenças de até 1e-9 contam como empate, e o empate vai para a configuração que aparece primeiro na ordem de simplicidade.
- Reajuste no treino externo inteiro com a configuração escolhida.
- Para cada dobra externa, registram-se a configuração escolhida e se ela está na **borda da grade**. A grade só muda por adendo (Seção 15).

## 6. Unidades de execução e estados

Uma **unidade** é o par `(r, k, modelo)`: 10 × 5 × 5 = **250 unidades**.

| Estado | Definição | Tratamento |
|---|---|---|
| **FT — falha técnica** | Exceção no ajuste ou na predição; pontuação com `NaN` ou `Inf`; pontuação interna inválida; divisão que viola a Seção 4; hash de entrada, código ou configuração divergente; invariante violado | A unidade não produz resultado. A causa é registrada. Repetir a execução com a mesma especificação e o mesmo código é permitido (o resultado é determinístico). Corrigir o código invalida **todos** os checkpoints (a identidade inclui o hash do código), e a execução definitiva recomeça inteira. Mudar a especificação exige adendo. Execução com FT pendente não gera relatório final (status `incompleta`) |
| **Aviso** | Aviso capturado sem falha, por exemplo `ConvergenceWarning` | Registrado por unidade; entra no critério C4 (Seção 10) |
| **DD — dobra degenerada** | Para um desfecho, a avaliação da dobra tem menos de 5 nódulos distintos positivos ou negativos (Seção 7.3) | Não é falha. A AUC daquela dobra e desfecho não é calculada nem imputada. Como as linhas são as mesmas para todos os modelos, a degeneração também é a mesma, e o pareamento se mantém |
| **RI — repetição inválida** | Para um desfecho, menos de 4 das 5 dobras são válidas | A repetição sai da estimativa daquele desfecho e é relatada |
| **NE — desfecho não estimável** | Menos de 8 das 10 repetições válidas | O desfecho aparece como "não estimável", só com os valores por dobra; os contrastes daquele desfecho não são relatados |
| **RD — réplica descartada** | Uma réplica do bootstrap em que o estimador de um desfecho não fica definido pelas mesmas regras | Contada por desfecho (Seção 8.2) |
| **IQ — IC de qualidade insuficiente** | Mais de 1% de réplicas descartadas num desfecho | O IC daquele desfecho não é relatado. **Não afeta a elegibilidade do algoritmo** |

## 7. Desfechos e métricas

### 7.1 Pontuações

LR fixa, LR com ajuste, RF e XGBoost usam `predict_proba[:, 1]`. O SVM usa `decision_function`.

### 7.2 Desfechos obrigatórios

- **G — AUC geral:** AUC-ROC calculada **dentro de cada dobra externa**, sobre todos os nódulos avaliados nela.
- **F — AUC na faixa de 12 a 30 mm:** AUC-ROC dentro de cada dobra externa, só sobre os nódulos avaliados com **12 ≤ D < 30 mm**, em que D = `original_shape_Maximum3DDiameter`.
  - Os modelos são treinados e ajustados com **todos** os nódulos; a faixa só recorta a avaliação.
  - **Contagem esperada na base de 353:** 226 nódulos, 187 pacientes, 135 malignos (91 benignos) — Emenda 1, E1 (commit `573bc17`).
  - O intervalo é o registrado na E1 ("12 a <30 mm"). A **variável** ainda depende de confirmação documental da responsável no DoD do Exp. 26.0.1 (D2, Seção 17), porque nenhum documento versionado a registra.

### 7.3 Regra de validade por dobra

A AUC de uma dobra num desfecho é calculada se a avaliação tiver **pelo menos 5 nódulos distintos positivos e 5 negativos**. Para cada dobra e desfecho, registram-se nódulos, pacientes, positivos, negativos e a validade.

### 7.4 Métricas descritivas

Não entram em contrastes nem na seleção.

| Métrica | LR fixa | LR com ajuste | RF | XGB | SVM |
|---|:-:|:-:|:-:|:-:|:-:|
| Brier; log-loss (probabilidade cortada em [1e-15, 1 − 1e-15]); intercepto e inclinação de calibração; por dobra, desfecho G | ✔ | ✔ | ✔ | ✔ | — |
| Sensibilidade e especificidade com p ≥ 0,5 | ✔ | ✔ | ✔ | ✔ | — |
| Sensibilidade e especificidade com `decision_function` ≥ 0, rotuladas "regra nativa, não comparável" | — | — | — | — | ✔ |

## 8. Estimação e incerteza

### 8.1 Estimadores

Para o modelo m e o desfecho d:

- θ̂(m, d) = média, entre as repetições válidas, da média das AUC das dobras válidas.
- Δ̂(a, d) = θ̂(a, d) − θ̂(lr_tuning, d), calculado com pareamento por dobra.
- **Sensibilidade à partição:** Δ̂ de cada repetição, com média, mínimo, máximo e número de repetições com Δ > 0.

### 8.2 Bootstrap de avaliação

- Reamostragem de **pacientes** com reposição sobre os 353; **B = 10.000**; gerador `numpy.random.default_rng(42)`; predições fixas, sem reajuste.
- Em cada réplica, para cada repetição e dobra, a AUC é recalculada sobre as linhas reamostradas daquela dobra, com peso igual ao número de vezes que o paciente foi sorteado, e com a regra da Seção 7.3 aplicada a nódulos distintos. Depois vêm θ\* e Δ\* pelas regras da Seção 8.1.
- IC por **percentis**:
  - **θ̂ de cada modelo:** IC de 95% (percentis 2,5 e 97,5), **descritivo e sem correção de multiplicidade**;
  - **contrastes da família principal:** IC com Bonferroni (Seção 9.1);
  - **contrastes descritivos:** IC de 95%.
- Uma réplica em que um desfecho não fica definido é descartada (RD) e contada. Com mais de 1% de descartes, o IC daquele desfecho não é relatado (IQ).

### 8.3 Como ler o intervalo

O intervalo é **descritivo e condicional**: descreve a variabilidade da amostra de pacientes avaliados, **dados** as partições, as configurações escolhidas e os modelos já ajustados.

| Não captura | Consequência |
|---|---|
| Variabilidade do treinamento: não há reajuste nas réplicas | O intervalo tende a ser estreito demais |
| Variabilidade da escolha de hiperparâmetros | Idem |
| Dependência entre dobras e repetições, cujos treinos se sobrepõem | A partição só aparece como mínimo e máximo |
| A validação cruzada estima o desempenho esperado do **procedimento**, não de um modelo final | Não há modelo final validado |
| Os pacientes não são amostra aleatória de uma população clínica; o alvo é suspeita radiológica | Não se generaliza para a prática clínica |
| Reuso dos pacientes de validação depois da Etapa 3 da S4, e as escolhas analíticas | Declarado como limitação |

**Vocabulário controlado do relatório.**

- *Vetados:* "significativo", "significância", "superior", "comprovado", "demonstrado", "estatisticamente", "p-valor".
- *Admitidos:* "diferença estimada", "compatível com", "não detectada nesta base", "intervalo descritivo".
- Não se calcula p-valor.

## 9. Contrastes e multiplicidade

### 9.1 Família

| Família principal (6) | Descritivos, fora da família |
|---|---|
| Δ(RF), Δ(XGB) e Δ(SVM) contra a LR com ajuste, cada um nos desfechos G e F | LR com ajuste contra LR fixa; pares entre RF, XGB e SVM; métricas da Seção 7.4 |

Na família principal, o IC usa **Bonferroni**: α = 0,05/6, percentis **0,41667 e 99,58333**. O IC de 95% sem correção vai em anexo. Os contrastes descritivos levam IC de 95%, rotulados. O ajuste de Bonferroni vale **só** para os seis contrastes; as AUC de cada modelo (θ̂) levam IC de 95% descritivo (Seção 8.2).

### 9.2 Limitações do ajuste de Bonferroni

1. Ele pressupõe que os intervalos tenham a cobertura declarada. Os intervalos daqui são condicionais e tendem a ser estreitos (Seção 8.3), então a correção **não garante** controle do erro de família. É uma convenção de largura, não uma garantia.
2. É conservador com contrastes correlacionados: os seis compartilham o comparador, e o desfecho F é um subconjunto do G.
3. Não cobre outras escolhas analíticas (grades, desfechos, regras de validade), nem os contrastes descritivos.
4. Exige caudas extremas no bootstrap: com B = 10.000, cada cauda de 0,417% se apoia em cerca de 42 réplicas.

## 10. Regra de seleção do algoritmo

### 10.1 Regra

Por padrão, o algoritmo recomendado é a **LR com ajuste**. Um desafiante a ∈ {RF, XGB, SVM} só é **elegível** se cumprir os quatro critérios:

| Critério | Condição |
|---|---|
| C1 — magnitude | Δ̂(a, G) ≥ **δ = 0,02** |
| C2 — consistência | Δ(a, G) > 0 em pelo menos 80% das repetições válidas |
| C3 — sem piora na faixa | F estimável e Δ̂(a, F) ≥ 0 |
| C4 — integridade técnica | As 250 unidades sem falha técnica pendente; G estimável; nenhum aviso de não convergência nas unidades do desafiante |

- Se houver mais de um elegível, vence o de maior Δ̂(G). Com diferença menor que 0,005 entre os dois primeiros, vence o mais simples, pela ordem LR < SVM < RF < XGB (número de hiperparâmetros ajustados — 1, 2, 2, 4 — e, no empate entre SVM e RF, o ajuste determinístico antes do estocástico).
- **Sem elegível, fica a LR com ajuste.**
- O IC e a taxa de descarte do bootstrap são relatados ao lado, mas **não são critério**.

### 10.2 Natureza de δ

δ = 0,02 é um **critério operacional** de decisão, igual à referência de magnitude da Seção 9 da S4. **Não tem interpretação clínica** e não define relevância clínica. Com AUC entre 0,94 e 0,98, o resultado mais provável é que nenhum desafiante seja elegível. Isso é um desfecho previsto e legítimo.

### 10.3 Otimismo da seleção

O algoritmo escolhido sai dos **mesmos** resultados de validação cruzada usados para estimá-lo. A estimativa do vencedor, θ̂, é **otimista** — a maldição do vencedor —, e nenhuma camada de validação cruzada externa à seleção nem conjunto independente a corrige. Por isso:

- o relatório apresenta as estimativas de **todos** os modelos, não só a do escolhido;
- o desempenho do escolhido é rotulado "otimista por seleção";
- a escolha vale só como recomendação para estas três características nesta base, **sem validação** no teste ou externa;
- uma correção formal do viés (por exemplo, Tibshirani e Tibshirani, 2009) não é aplicada, e isso fica registrado como limitação.

## 11. Figuras obrigatórias

Regras gerais:

- só dados agregados, sem identificadores nem pontos por nódulo;
- `matplotlib` do ambiente fixado;
- PNG a 300 dpi e SVG, cada figura com o CSV agregado que a gerou;
- cor fixa por modelo, paleta acessível a daltônicos (Okabe–Ito);
- eixos de AUC comuns entre painéis;
- legenda com a base (353 / 509), o desfecho e o tipo de intervalo ou de faixa.

| Código | Conteúdo |
|---|---|
| F1 | θ̂ por modelo nos desfechos G e F: pontos das 10 repetições, média e **IC de 95% descritivo** do bootstrap de θ\*, **sem correção de multiplicidade**, com uma linha de referência na LR fixa. Os intervalos com Bonferroni **não** são aplicados às AUC individuais; eles aparecem só nos contrastes, em F2 |
| F2 | Gráfico de floresta dos seis contrastes da família: Δ̂, IC com Bonferroni (traço grosso), IC de 95% (traço fino), faixa entre a menor e a maior repetição, linhas verticais em 0 e em δ = 0,02 |
| F3 | Distribuição de Δ por dobra (até 50 valores por contraste e desfecho), em pontos com caixa; dobras degeneradas indicadas em contagem |
| F4 | Frequência das configurações escolhidas por modelo (50 dobras), com as bordas da grade destacadas |
| F5 | Curva ROC média vertical por modelo no desfecho G (média das 50 curvas por dobra em uma grade fixa de 101 pontos de 1 − especificidade), com a faixa entre os percentis 2,5 e 97,5 das 50 curvas, rotulada na legenda como **"dispersão entre dobras — não é intervalo de confiança"** |
| F6 | Calibração por decis da probabilidade (LR fixa, LR com ajuste, RF, XGB; desfecho G, predições fora da dobra de r = 0). Decis com menos de 10 nódulos são fundidos com o vizinho |
| F7 | Denominadores no desfecho F por dobra: nódulos, positivos e negativos, com a validade marcada |
| F8 | Histograma de Δ\* do bootstrap por contraste da família, com os percentis do IC e a taxa de descarte |

## 12. Relatórios obrigatórios

| Artefato | Conteúdo mínimo |
|---|---|
| `resumo_benchmark.md` (relatório científico) | 1. Identificação: commit, SHA-256 da configuração canônica, das entradas e das saídas, ambiente. 2. Desvios em relação a este pré-registro (adendos), ou "nenhum". 3. População e denominadores (G e F), com as contagens de DD, RI, NE, RD e IQ. 4. Tabela principal: θ̂ por modelo e desfecho, com IC de 95% descritivo. 5. Contrastes da família com os dois intervalos. 6. Sensibilidade à partição. 7. LR com ajuste contra LR fixa. 8. Hiperparâmetros escolhidos e bordas da grade. 9. Métricas descritivas (sem as de probabilidade do SVM). 10. Regra de seleção com C1 a C4 por desafiante e o resultado. 11. Otimismo da seleção (Seção 10.3). 12. A tabela "o que o intervalo não captura" (Seção 8.3). 13. Falhas técnicas e avisos. 14. Declaração de não uso do teste e de independência em relação à S4. 15. Lista de figuras |
| `relatorio_benchmark.json` | Todos os números do resumo, em esquema versionado (`versao_esquema`), suficientes para recalcular a regra de seleção |
| `relatorio_execucao.json` | Tempos por unidade e por modelo, número de processos, falhas, avisos e retomadas |
| Verificação automática | O resumo passa na checagem de vocabulário (Seção 8.3) e na varredura `LIDC-IDRI-\d{4}` |

## 13. Ambiente e dependências

| Pacote | Versão |
|---|---|
| Python | 3.9.13 |
| scikit-learn | 1.3.2 |
| numpy | 1.23.5 |
| pandas | 1.5.3 |
| scipy | 1.10.1 |
| matplotlib | 3.7.3 |
| **xgboost** | **2.1.4** |

O XGBoost 2.1.4 é a última versão do ramo 2.x; o ramo 3.0+ exige Python ≥ 3.10. No PyPI ele tem `requires_python >= 3.8`, wheel `py3-none-win_amd64` não retirada, e depende de `numpy` e `scipy`. No conda-forge, `py-xgboost 2.1.4` tem builds para CPU com `python >= 3.9`.

**Entrada condicionada a um teste de fumaça** em ambiente isolado, com dados sintéticos: importação; ajuste; `clone`; `GridSearchCV`; duas execuções bit a bit idênticas; carregamento da DLL sem bloqueio do Controle de Aplicativo do Windows. A fixação (`xgboost==2.1.4` no pip e `py-xgboost=2.1.4`, build CPU, no conda) entra em commit próprio na branch da S6. Se o teste falhar, qualquer troca de versão exige adendo.

O ambiente é verificado pelo próprio benchmark, sem `rastreabilidade.ambiente()` da S4, cuja lista não inclui o XGBoost.

## 14. Pipeline coordenador e artefatos

### 14.1 Pacote

`src/benchmark_s6/`, com testes em `tests/benchmark_s6/`. `src/incremental` **não é modificado**.

| Módulo | Função |
|---|---|
| `adaptadores_s4.py` | **Único** arquivo que importa a S4 (lista na Seção 14.2) |
| `config.py` | Especificação congelada, geração da configuração canônica (Seção 14.4) e seu SHA-256 |
| `aprovacoes.py` | Verifica os registros da Seção 19 no git |
| `dados.py` | Carga em duas fases (identificadores, depois linhas autorizadas) |
| `folds.py` | Plano de dobras e validação (Seção 4) |
| `modelos.py` | Fábrica dos cinco modelos (Seção 5.1) |
| `tuning.py` | `GridSearchCV` e a função de escolha |
| `unidades.py` | Execução de `(r, k, modelo)` com checkpoints atômicos |
| `metricas.py` | AUC por dobra, validade e métricas descritivas |
| `bootstrap.py` | Seção 8.2 |
| `consolidacao.py` | Seções 8.1 e 9 |
| `selecao.py` | Seção 10 |
| `figuras.py` | Seção 11 |
| `relatorios.py` | Seção 12 |
| `privacidade.py` | Varredura por dados individuais |
| `manifesto.py` | Manifesto com SHA-256 de entradas, código, configuração e saídas |
| `coordenador.py`, `__main__.py` | Comando coordenador |

### 14.2 Reuso da S4: só APIs públicas

| Importado de `src/incremental` | Uso |
|---|---|
| `folds.Amostra`, `folds.criar_amostra` | Amostra alinhada na ordem canônica |
| `checkpoints.fingerprint_folds` | Fingerprint das dobras |
| `dados.TABELA_OFICIAL`, `dados.SPLIT_OFICIAL`, `dados.ArquivoVerificado`, `dados.fonte_oficial`, `dados.verificar_estrutura` | Fonte fixada e verificação estrutural sem alvo |
| `rastreabilidade.json_canonico`, `rastreabilidade.sha256_canonico`, `rastreabilidade.calcular_sha256`, `rastreabilidade.estado_git` | Hash canônico e estado do git |
| `guardas.formatar_registro`, `guardas.hash_valor_decisao` | Formato dos registros de aprovação |
| `config.FEATURES_M1` | As três características |

**Nenhuma dependência privada.** Funções equivalentes que, na S4, estão presas à configuração da S4 ou são privadas — validação de dobras, carga autorizada, verificação de aprovações, sessão de checkpoints, manifesto, varredura de dados individuais e bootstrap — são **reimplementadas**, com a função equivalente da S4 citada no docstring e testes espelhados. Um teste fixa o SHA-256 dos arquivos da S4 importados; se algum mudar, o teste falha e exige revisão, nunca adaptação automática.

### 14.3 Comando coordenador

```
python -m src.benchmark_s6 --modo {configuracao|implementacao|definitivo|verificar} --checkpoints DIR_LOCAL --destino reports/sprint6/benchmark [--processos N]
```

O modo `configuracao` só gera ou confere a configuração canônica (Seção 14.4); não lê dados nem ajusta modelos. Os demais modos executam os passos abaixo.

| Passo | `implementacao` | `definitivo` | `verificar` |
|---|:-:|:-:|:-:|
| 1. Ambiente (Seção 13) | ✔ | ✔, divergência aborta | ✔ |
| 2. Configuração: a gerada pelo código tem de ser idêntica à canônica versionada | ✔ | ✔, e com o SHA-256 aprovado em P9; divergência aborta | ✔ |
| 3. Aprovações: P9 e `S4_E3_REGISTRADA` (Seção 19) | — | ✔, ausência aborta | ✔ |
| 4. Árvore git limpa, HEAD registrado | registra | exige | exige |
| 5. Verificação estrutural (contagens, sem alvo) | ✔ | ✔ | — |
| 6. Carga | só `train` | `train` + `validation` | — |
| 7. Conferência da faixa (226/187/135) | — | ✔, divergência aborta antes de qualquer ajuste | — |
| 8. Plano de dobras, validação e fingerprints | ✔ (base 291/417, r = 0) | ✔ (base 353/509, r = 0…9) | recalcula e compara |
| 9. Unidades com checkpoints e paralelismo entre unidades | ✔ (ver nota) | ✔ | — |
| 10. Invariante da LR fixa no treino oficial (coeficientes, intercepto e scaler contra `baseline_metrics.json`; tolerâncias rtol 1e-6 / atol 1e-8 nos coeficientes e rtol 1e-10 / atol 1e-12 no scaler, as de `scripts/sprint4/validate_reference_models.py`), sem métrica | ✔ | — | — |
| 11. Consolidação e estados (Seções 6 e 8.1) | **nenhuma métrica externa** (ver nota) | ✔ | recalcula |
| 12. Bootstrap (Seção 8.2) | — | ✔ | recalcula |
| 13. Seleção (Seção 10) | — | ✔ | recalcula |
| 14. Figuras e relatórios (Seções 11 e 12) | só `relatorio_execucao.json` | ✔ | recalcula e compara SHA-256 |
| 15. Varredura de privacidade e vocabulário; manifesto final | ✔ | ✔ | ✔ |

**Nota sobre o modo `implementacao`.** O ajuste interno usa a AUC das dobras internas para escolher os hiperparâmetros, como no modo definitivo; essas pontuações internas ficam só no diretório local e **não são relatadas nem versionadas**. As predições sobre as dobras externas são geradas apenas para conferir formato e finitude e ficam só no diretório local. **Nenhuma métrica de desempenho externa** — AUC, Brier, log-loss, calibração, sensibilidade, especificidade ou contraste — é calculada, gravada ou divulgada neste modo.

**Códigos de saída:** 0 concluído · 2 bloqueio de aprovação · 3 falha técnica · 4 integridade (hash, configuração, ambiente ou git) · 5 privacidade ou vocabulário.

**Paralelismo:** entre unidades, por processos; dentro do estimador, `n_jobs=1`; BLAS limitado a uma thread por `threadpoolctl`. O resultado não depende de `--processos`.

### 14.4 Configuração canônica

- **Arquivo:** `docs/sprint6/config_benchmark_p9.json`, gerado pelo modo `configuracao` a partir de `src/benchmark_s6/config.py`, gravado na forma canônica de `rastreabilidade.json_canonico` (chaves ordenadas, UTF-8, sem espaços, sem `NaN`).
- **Conteúdo mínimo:** base e partições; características; SHA-256 fixados das entradas; definição da faixa; dobras e sementes; modelos, componentes fixos, grades e ordem de simplicidade; regra de escolha do ajuste interno; regras de validade (Seção 7.3) e estados (Seção 6); B, gerador e percentis; família de contrastes; critérios C1 a C4, δ, margem de desempate e ordem de simplicidade dos algoritmos; ambiente (Seção 13); lista e SHA-256 dos arquivos da S4 importados.
- **SHA-256 da configuração:** `guardas.hash_valor_decisao` do conteúdo do arquivo, que coincide com `rastreabilidade.sha256_canonico` para JSON sem tipos especiais.
- **Ordem obrigatória antes do registro de aprovação P9:**
  1. implementar `config.py` e o modo `configuracao`;
  2. gerar o arquivo e versioná-lo num commit;
  3. calcular o SHA-256 e anotá-lo na Seção 19 deste documento;
  4. só então registrar a aprovação P9, com esse valor em `valor_sha256`, num commit posterior aos passos 2 e 3.
- Qualquer mudança na configuração gera novo arquivo, novo SHA-256, adendo e nova aprovação.

### 14.5 Artefatos

| Onde | Artefato | Versionado |
|---|---|:-:|
| `docs/sprint6/` | Este pré-registro; `config_benchmark_p9.json` (configuração canônica); adendos | sim |
| `reports/sprint6/benchmark/` | `config_benchmark.json` (cópia da canônica usada na execução, com o mesmo SHA-256); `plano_folds.json` (fingerprints e contagens agregadas por dobra); `denominadores.csv`; `auc_por_fold.csv`; `auc_por_repeticao.csv`; `contrastes.csv`; `bootstrap_resumo.json`; `bootstrap_delta.csv` (Δ\* das réplicas, só agregados); `hiperparametros_escolhidos.csv`; `metricas_descritivas.csv`; `selecao_algoritmo.json`; `figuras/F1…F8.{png,svg}`; `figuras/dados/F1…F8.csv`; `relatorio_benchmark.json`; `resumo_benchmark.md`; `relatorio_execucao.json`; `manifesto_benchmark.json` | sim |
| `reports/sprint6/benchmark/implementacao/` | `relatorio_execucao.json`, invariante da LR fixa, manifesto exploratório — sem métrica externa de desempenho | sim |
| `DIR_LOCAL` (fora do git e de `data/`) | Checkpoints por unidade; predições fora da dobra com `nodulo_id`; atribuição de dobras com `patient_id`; resultados detalhados do ajuste interno | **não** |

## 15. Orçamento e alterações

- **Estimativa:** cerca de 22.750 ajustes, entre 45 e 80 min em série; o bootstrap com predições fixas leva minutos.
- **O teto de 3 h é só um alarme.** Não há caminho de código que altere grade, número de árvores, dobras, repetições, B ou sementes por causa do tempo. Isso é verificado por teste.
- Se o modo `implementacao` projetar que o teto será estourado, a execução definitiva **não começa** até haver adendo aprovado.
- Execução definitiva interrompida é retomada pelos checkpoints, com a mesma especificação. Resultados parciais nunca são relatados.
- **Adendo:** documento datado e numerado (A1, A2…), com motivo, o que muda, o que se sabia no momento (sem resultado de desempenho deste benchmark) e registro de aprovação, num commit anterior à execução afetada.

## 16. Riscos e limitações

1. Teto de AUC (0,94–0,98): ganhos pequenos, e é provável que a LR com ajuste fique como recomendada.
2. O desfecho F tem cerca de 18 negativos por dobra; a AUC por dobra é instável, e são possíveis DD e RD.
3. A faixa é recortada por um preditor (o diâmetro): F mede a discriminação residual dentro do tamanho, não independência dele.
4. Intervalos condicionais e estreitos; Bonferroni sem garantia de erro de família (Seções 8.3 e 9.2).
5. Otimismo da seleção (Seção 10.3).
6. Reuso dos pacientes de validação depois da Etapa 3 da S4.
7. `MeshVolume` assimétrico, sem transformação.
8. Hiperparâmetros na borda da grade.
9. XGBoost: compatibilidade e possível bloqueio da DLL; determinismo atrelado a versão e CPU.
10. Dependência do cronograma da S4: sem a Etapa 3 registrada, não há execução definitiva.
11. Governança: decisões tomadas pela responsável e pelo revisor, sem resposta do grupo.

## 17. Matriz de decisões D1–D12

Todas: **propostas, pendentes de aprovação** (Seção 19). D2 depende, além disso, de confirmação documental da responsável.

| # | Decisão | Resolução | Observação | Justificativa |
|---|---|---|---|---|
| D1 | Reuso da S4 | Só APIs públicas, num único adaptador; nenhuma dependência privada; reimplementação documentada; teste de hash dos arquivos importados | — | As APIs públicas de alto nível da S4 estão presas à configuração dela; usar símbolos privados criaria acoplamento frágil |
| D2 | Faixa | `original_shape_Maximum3DDiameter` ∈ [12, 30) mm; conferência 226/187/135 no modo definitivo, antes de qualquer ajuste | **Pendente de confirmação documental da responsável** | O intervalo está registrado na E1. A variável usada no Exp. 26.0.1 não está em nenhum documento versionado (o `docs/indice_experimentos.md` registra os Exp. 18–26 como não localizados) |
| D3 | Validade | ≥ 5 positivos e ≥ 5 negativos distintos por dobra; ≥ 4 de 5 dobras; ≥ 8 de 10 repetições; descarte > 1% → IQ, sem efeito na elegibilidade | — | Separa a qualidade do intervalo da decisão |
| D4 | Multiplicidade | Família de 6 contrastes com Bonferroni; B = 10.000; IC de 95% descritivo para cada θ̂ | B = 10.000 em vez de 2000 | Caudas de 0,417%: cerca de 42 réplicas por cauda, contra cerca de 8 com B = 2000 |
| D5 | Seleção | C1 a C4; δ = 0,02 operacional; 80%; Δ na faixa ≥ 0; desempate 0,005; LR < SVM < RF < XGB | Taxa de descarte fora do C4 | Multicritério, sem depender de significância |
| D6 | Grades e números fixos | Como na Seção 5.1; RF com 500 árvores | — | Nenhum motivo técnico para mudar |
| D7 | Sensibilidade e especificidade do SVM | Incluídas pela regra nativa, rotuladas e só descritivas | — | Completa o quadro sem exigir probabilidade |
| D8 | Estratificação | Só pela classe | — | Coerente com a S4; os denominadores do desfecho F são tratados pela Seção 7.3 |
| D9 | XGBoost | 2.1.4, condicionado ao teste de fumaça | — | Última versão 2.x; compatível com Python 3.9 pelos metadados |
| D10 | Calibração do SVM | Fora; só por adendo | — | Evita o ajuste extra e a comparação assimétrica |
| D11 | Identificadores do teste | Só na verificação estrutural pública da S4; alvo e características nunca lidos | — | Necessário para conferir as invariantes de contagem |
| D12 | Sementes | Externas `42 + r`; internas `1000 + 10r + k`; modelos `random_state=42`; bootstrap `default_rng(42)` | — | Sem colisão entre níveis |

## 18. Critérios de aceite

**A. Pré-registro e governança**

1. Este documento está versionado na branch da S6. Até o merge do PR #10, a Emenda 1 é referida pelo commit `573bc17`; depois dele, os links relativos para a Emenda são acrescentados e validados.
2. D2 foi confirmada pela responsável, com a fonte documental citada.
3. A configuração canônica (Seção 14.4) foi gerada pelo código, versionada e teve o SHA-256 anotado na Seção 19 **antes** do registro de aprovação P9.
4. Há registro de aprovação da P9, com esse SHA-256, num commit anterior a qualquer execução definitiva.
5. Qualquer mudança passa por adendo aprovado (Seção 15).

**B. Isolamento**

6. Testes unitários comprovam a abortagem diante de linha de teste, de validação sem `S4_E3_REGISTRADA`, de configuração divergente da canônica e de árvore git suja no modo definitivo.
7. `git diff` em `src/incremental` sai vazio; o teste de hash dos arquivos importados passa; `adaptadores_s4.py` não importa nenhum símbolo com `_`.
8. O modo `implementacao` não calcula, não grava e não divulga nenhuma métrica externa de desempenho, comprovado por teste sobre os artefatos e o caminho de código.

**C. Dobras e ajuste**

9. São 10 × 5 dobras externas válidas, e o fingerprint é idêntico para os cinco modelos.
10. Nenhuma linha nem paciente da avaliação externa aparece no ajuste interno.
11. O desempate segue a ordem declarada, comprovado por teste com empate sintético.
12. A grade é exatamente a da Seção 5.1, comprovado por teste contra a configuração canônica.

**D. Métricas e estados**

13. A AUC é calculada por dobra para os cinco modelos, nos desfechos G e F.
14. Denominadores e validade aparecem para todas as dobras.
15. Nenhuma AUC é imputada.
16. Os estados da Seção 6 são distinguidos nos artefatos.
17. O SVM não tem métricas de probabilidade.
18. A contagem da faixa confere.

**E. Inferência, seleção e figuras**

19. O bootstrap é por percentis, com B = 10.000, e RD e IQ são aplicados.
20. Os seis contrastes trazem os dois intervalos; as AUC individuais trazem só o IC de 95% descritivo.
21. O relatório contém as Seções 8.3, 9.2 e 10.3.
22. A regra de seleção é recalculável a partir de `relatorio_benchmark.json`, com C1 a C4 explícitos.
23. Existem as figuras F1 a F8 com seus CSV; F1 não usa intervalos de Bonferroni; a faixa de F5 está rotulada como dispersão entre dobras.

**F. Reprodutibilidade**

24. Reexecutar uma unidade dá predições idênticas bit a bit.
25. O resultado é idêntico com 1 e com 4 processos.
26. O invariante da LR fixa passa no modo `implementacao`.
27. O modo `verificar` reproduz os SHA-256 das saídas versionadas.

**G. Ambiente, privacidade e qualidade**

28. O teste de fumaça do XGBoost passa.
29. O manifesto definitivo é elegível, com o ambiente da Seção 13.
30. Os artefatos versionados passam na varredura de dados individuais e de `LIDC-IDRI-\d{4}`.
31. O resumo passa na checagem de vocabulário.
32. `python -B -m unittest discover -s tests` passa.
33. Nenhum caminho altera parâmetros por tempo, comprovado por teste.

## 19. Registro de aprovação e adendos

**Configuração canônica:** `docs/sprint6/config_benchmark_p9.json` — **ainda não gerada**.
**SHA-256 da configuração canônica:** **a definir** (Seção 14.4).

O benchmark definitivo exige **dois** registros neste documento, cada um numa linha própria começando na coluna 0, fora de bloco de código, introduzida no commit da aprovação e com data igual ou anterior à desse commit (mesma regra das guardas da S4):

- **P9** — `valor_sha256` = SHA-256 da configuração canônica, anotado acima **antes** do registro;
- **S4_E3_REGISTRADA** — `valor_sha256` = SHA-256 do manifesto definitivo, elegível e concluído da Etapa 3 da S4.

Formato (exemplo, que não conta como registro):

```
APROVACAO-DECISAO codigo=P9 status=aprovada data=AAAA-MM-DD valor_sha256=<64 hex>
APROVACAO-DECISAO codigo=S4_E3_REGISTRADA status=aprovada data=AAAA-MM-DD valor_sha256=<64 hex>
```

**Registros de aprovação:** nenhum.

**Ratificação do grupo:** solicitada; a ausência de resposta é registrada aqui, com data, e não equivale a aprovação.

**Adendos:** nenhum.

## Referências

A conferir antes da citação no relatório final.

- BATES, S.; HASTIE, T.; TIBSHIRANI, R. Cross-validation: what does it estimate and how well does it do it? *Journal of the American Statistical Association*, 2024.
- CAWLEY, G. C.; TALBOT, N. L. C. On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, v. 11, 2010.
- TIBSHIRANI, R. J.; TIBSHIRANI, R. A bias correction for the minimum error rate in cross-validation. *The Annals of Applied Statistics*, v. 3, n. 2, 2009.
- VARMA, S.; SIMON, R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, v. 7, 2006.
- PROJETO INTEGRADOR III — GRUPO 4. Protocolo de Valor Incremental da Radiômica — Sprint 4, e Emenda 1, revisão 3 (commit `573bc17`). 2026.
