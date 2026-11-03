# Benchmark de classificadores da Sprint 6 — resumo

Relatório gerado pelo pipeline `src/benchmark_s6` (pré-registro P9). Texto descritivo e condicional: não afirma validação clínica nem desempenho em teste.

## 1. Identificação

- Modo: `definitivo`; dados oficiais: sim; unidades: 250.
- Commit da execução: `c124c81d912ffda5c80c9cd1fa164189d9859f8b` (branch `feat/s6-benchmark-executor`).
- SHA-256 da configuração canônica: `a0d096a5415244f17f40360a2a5a8901bc8af1da5cf8687b891f920e1ee122b3`.
- SHA-256 da entrada `split`: `ed120c2b5c0f308d41c11cfc32dae0e897a179c71d6ea803588720ba1ba4f65f`.
- SHA-256 da entrada `tabela`: `5e92e7e860ad8edd327570c929cd263cd780be98902e57017551ad0080da60ab`.
- Ambiente: joblib 1.5.3, numpy 1.23.5, pandas 1.5.3, python 3.9.13, scikit-learn 1.3.2, scipy 1.10.1, sistema Windows, threadpoolctl 3.7.0, xgboost 2.1.4.
- SHA-256 das saídas (LF) dos demais artefatos; o do próprio resumo e o do manifesto ficam em `manifesto_benchmark.json`:
  - `auc_por_fold.csv`: `449a1689e26b3d83442a839b2069c85950e09128d200ce3d7f3a9a55449e5cd6`
  - `auc_por_repeticao.csv`: `b53c35f60410514871e42767d1c7ff50134a13c2e24c9a3c3af402d8946744f1`
  - `bootstrap_delta.csv`: `b9869d12c610d54c5e33141b7818f4a7f29474b05b10bed94a5f3c9f8dcd671f`
  - `bootstrap_resumo.json`: `f54c0176b3786e1eb65362d1f19d0ec912ed323858ddd87ac47d4eec2b85a4b1`
  - `config_benchmark.json`: `f6176d2ac29eb1958ca73de7b394483b75541d269fc30b5828581fb211883c8a`
  - `contrastes.csv`: `18eafac731e1e5998730fde2df1885d1e167ee14f721badcc605786a4075fd42`
  - `denominadores.csv`: `4ccbb6ed71196c83b375bc72ce366f2b4a616fe0f36997f508295efd24d4f4eb`
  - `figuras/F1.png`: `f166e29e55815152a09b320623a3972fd03d1822919f5cabc8b3e7d8c6b26a95`
  - `figuras/F1.svg`: `a7dda41c4412c2e4f5f20192876cbe195b8640b9defeb52581d8a11204ecae1e`
  - `figuras/F2.png`: `e860888ade3e57dd93869b3e8f819df425620352cbca70e87761b8c5a12dcf67`
  - `figuras/F2.svg`: `70e519b28d51869181ac7e6707e965f36d7101cf4d84373e48a9a44624437bba`
  - `figuras/F3.png`: `aada0d1165d7323c8bc53541c69f47a3a0911497df11b0b22170e4fd94604eae`
  - `figuras/F3.svg`: `e5ace8b273b3e9d12767963d2c973cc6dd21ec4ce8b9a2bf9c6524fbce810af8`
  - `figuras/F4.png`: `03967d3a466efe6b7ee1fc2acd359e5788d9ed6d1d8538d1f104862d4ff1beb7`
  - `figuras/F4.svg`: `35ff501ba0595ed2fd61d3108aa7ff064affd167ba6bb3686700ad32509cbc2d`
  - `figuras/F5.png`: `23fd0eef5cc8d61d0d477d0c780d0ea4e7787d38a25adf9e0754bc392b1b9dd8`
  - `figuras/F5.svg`: `89079cc968d9c7d8ab8d7d0f9821d71981214e223c8b26bb35fb904635557605`
  - `figuras/F6.png`: `42defada1b485a59cfed2669af6fb981b5754f22a9afcd9471ba1f341231ac11`
  - `figuras/F6.svg`: `e1beded3f25782e781eabd8b1e54940878de9f72c0b3754b28f3b888f029f76a`
  - `figuras/F7.png`: `d6c615233d6d2c14fe1f204fb6a8d9a1bc470b427a1e722d8de8b125fdb590fa`
  - `figuras/F7.svg`: `d975467ad9eb3a70b26a3f584e3d3b7afc3f0e10e09e78d1ccc366c413716290`
  - `figuras/F8.png`: `33c36bc328f1de7f552c944eea847cc40b08e05646e565e07f4de065b853145f`
  - `figuras/F8.svg`: `ff8135a96ef745d40fe8573f6ca53a8fdf87dbb8db01dbfb9759f0e428926214`
  - `figuras/dados/F1.csv`: `efdbecfa166a73e8e8f07ed62b3a4e2a478301ac4c324ea12b2e8bc573ccac36`
  - `figuras/dados/F2.csv`: `b6b839296e289eb6d671f3cd4fd27b7a9d36bc35b71361849529d16999cf954b`
  - `figuras/dados/F3.csv`: `3b04697cd1924f7d0161453855e11b9e8340b41fdb45973eff2ce2bb5d776f6c`
  - `figuras/dados/F4.csv`: `b66f6ff6dd1c23996c7ee29771c6ae2f5faebb371237e51d64fc566f66c44b7f`
  - `figuras/dados/F5.csv`: `072e08a63f22b218ca1d49e89e328c4c4ebb770eecf8a44c370a9f32a89ceed1`
  - `figuras/dados/F6.csv`: `6469bdd5cdbbe281b9bb22a892924f7e90f9331c2f88d49d3830be862e798746`
  - `figuras/dados/F7.csv`: `125de73734fde4ca2a687a3204f61193a0fa4659a2d3702a1fd718480e0ff5ed`
  - `figuras/dados/F8.csv`: `2c482c447d987588ff081074f4ba21d4e7a9fd7bebe7682c2615b24c98ac75b1`
  - `hiperparametros_escolhidos.csv`: `d2fca17c933b4ee829012139e93caa403f661be8f2e2f70fb76bb25d92d27ac1`
  - `metricas_descritivas.csv`: `82fd4bb699008b242b520ac747f08cb4065ce52954adc52c58dbd6bfda844474`
  - `plano_folds.json`: `6b6671633babcc83a1dc3cfd68a5c26e97f8167fb3764c28ec1667c53c0006d2`
  - `relatorio_benchmark.json`: `d5ee08b0fe24a8fb3e1925aade0650613e4e65e73084b23b0f9e41aa65e73081`
  - `selecao_algoritmo.json`: `b45603f09be118719ade7ac71fa42e682e5ab411610ed3f3f339d6ce71896f31`

## 2. Desvios em relação ao pré-registro

Adendos: nenhum. Desvios: nenhum.

## 3. População e denominadores

Base: 353 pacientes e 509 nódulos (258 benignos e 251 malignos). Faixa do desfecho F (12 ≤ D < 30 mm): 226 nódulos, 187 pacientes, 135 malignos e 91 benignos.

| Desfecho | Dobras | DD | Repetições | RI | Repetições válidas | NE | RD | IQ |
|---|--:|--:|--:|--:|--:|:-:|--:|:-:|
| G | 50 | 0 | 10 | 0 | 10 | não | 0 | não |
| F | 50 | 0 | 10 | 0 | 10 | não | 0 | não |

Os denominadores por dobra estão em `denominadores.csv`. Falhas técnicas pendentes: 0.

## 4. AUC por modelo

θ̂ é a média, entre as repetições válidas, da média das AUC das dobras válidas. O intervalo é o de 95% descritivo do bootstrap por paciente, sem correção de multiplicidade.

| Modelo | Desfecho | θ̂ | Intervalo descritivo de 95% | Rótulo |
|---|---|--:|---|---|
| LR fixa | G | 0.9442 | [0.9246; 0.9624] |  |
| LR com ajuste | G | 0.9449 | [0.9254; 0.9628] | otimista por seleção |
| SVM | G | 0.9424 | [0.9224; 0.9605] |  |
| RF | G | 0.9476 | [0.9288; 0.9646] |  |
| XGBoost | G | 0.9438 | [0.9242; 0.9619] |  |
| LR fixa | F | 0.7903 | [0.7359; 0.8460] |  |
| LR com ajuste | F | 0.7907 | [0.7364; 0.8464] | otimista por seleção |
| SVM | F | 0.7890 | [0.7357; 0.8437] |  |
| RF | F | 0.7968 | [0.7459; 0.8501] |  |
| XGBoost | F | 0.7844 | [0.7329; 0.8389] |  |

## 5. Contrastes da família principal

Diferença estimada contra a LR com ajuste. Dois intervalos: Bonferroni (seis contrastes, percentis 0.41667 e 99.58333) e 95% sem correção.

| Contraste | Desfecho | Diferença estimada | Bonferroni | 95% sem correção |
|---|---|--:|---|---|
| RF − LR com ajuste | G | 0.0027 | [-0.0043; 0.0098] | [-0.0024; 0.0078] |
| RF − LR com ajuste | F | 0.0061 | [-0.0250; 0.0368] | [-0.0168; 0.0291] |
| XGBoost − LR com ajuste | G | -0.0011 | [-0.0075; 0.0050] | [-0.0057; 0.0034] |
| XGBoost − LR com ajuste | F | -0.0063 | [-0.0358; 0.0221] | [-0.0276; 0.0152] |
| SVM − LR com ajuste | G | -0.0025 | [-0.0080; 0.0013] | [-0.0064; 0.0005] |
| SVM − LR com ajuste | F | -0.0017 | [-0.0213; 0.0129] | [-0.0157; 0.0094] |

## 6. Sensibilidade à partição

Diferença estimada de cada repetição: média, mínimo, máximo e número de repetições com diferença positiva.

| Contraste | Desfecho | Média | Mínimo | Máximo | Repetições com Δ > 0 |
|---|---|--:|--:|--:|--:|
| RF − LR com ajuste | G | 0.0027 | 0.0001 | 0.0044 | 10 de 10 |
| RF − LR com ajuste | F | 0.0061 | -0.0016 | 0.0209 | 8 de 10 |
| XGBoost − LR com ajuste | G | -0.0011 | -0.0060 | 0.0012 | 6 de 10 |
| XGBoost − LR com ajuste | F | -0.0063 | -0.0230 | 0.0102 | 3 de 10 |
| SVM − LR com ajuste | G | -0.0025 | -0.0060 | 0.0011 | 1 de 10 |
| SVM − LR com ajuste | F | -0.0017 | -0.0141 | 0.0101 | 4 de 10 |
| LR com ajuste − LR fixa | G | 0.0007 | -0.0007 | 0.0018 | 8 de 10 |
| LR com ajuste − LR fixa | F | 0.0004 | -0.0059 | 0.0042 | 6 de 10 |
| RF − XGBoost | G | 0.0038 | 0.0016 | 0.0073 | 10 de 10 |
| RF − XGBoost | F | 0.0124 | 0.0044 | 0.0280 | 10 de 10 |
| RF − SVM | G | 0.0051 | 0.0014 | 0.0085 | 10 de 10 |
| RF − SVM | F | 0.0078 | 0.0040 | 0.0142 | 10 de 10 |
| XGBoost − SVM | G | 0.0014 | -0.0047 | 0.0064 | 5 de 10 |
| XGBoost − SVM | F | -0.0046 | -0.0237 | 0.0076 | 3 de 10 |

## 7. LR com ajuste contra LR fixa

- Desfecho G: diferença estimada 0.0007, intervalo descritivo de 95% [-0.0009; 0.0022] (contraste descritivo, fora da família).
- Desfecho F: diferença estimada 0.0004, intervalo descritivo de 95% [-0.0053; 0.0060] (contraste descritivo, fora da família).

## 8. Hiperparâmetros escolhidos e bordas da grade

Frequência das configurações escolhidas nas 50 dobras (detalhe em `hiperparametros_escolhidos.csv`).

| Modelo | Dobras com escolha na borda da grade | Configuração mais frequente (n) |
|---|--:|---|
| LR fixa | 0 | (sem ajuste) (50) |
| LR com ajuste | 5 | C=3.1622776601683795 (13) |
| SVM | 38 | C=10.0;gamma=0.01 (13) |
| RF | 50 | max_features=1;min_samples_leaf=20 (30) |
| XGBoost | 50 | learning_rate=0.03;max_depth=2;min_child_weight=1;n_estimators=100 (20) |

## 9. Métricas descritivas (somente desfecho G)

Média, mínimo e máximo entre as dobras. Não entram em contrastes nem na seleção. O SVM não tem métricas de probabilidade: só sensibilidade e especificidade pela regra nativa (`decision_function` ≥ 0), que não é comparável.

| Modelo | Métrica | Dobras | Média | Mínimo | Máximo | Observação |
|---|---|--:|--:|--:|--:|---|
| LR fixa | brier | 50 | 0.0976 | 0.0550 | 0.1370 |  |
| LR fixa | log_loss | 50 | 0.3211 | 0.1954 | 0.4701 |  |
| LR fixa | intercepto_calibracao | 50 | 0.1065 | -1.0068 | 1.6007 |  |
| LR fixa | inclinacao_calibracao | 50 | 1.1999 | 0.6164 | 2.3292 |  |
| LR fixa | sensibilidade_p_0_5 | 50 | 0.8449 | 0.6327 | 0.9286 |  |
| LR fixa | especificidade_p_0_5 | 50 | 0.9053 | 0.8222 | 0.9804 |  |
| LR com ajuste | brier | 50 | 0.0967 | 0.0544 | 0.1358 |  |
| LR com ajuste | log_loss | 50 | 0.3202 | 0.1889 | 0.4812 |  |
| LR com ajuste | intercepto_calibracao | 50 | 0.0684 | -0.9819 | 1.5586 |  |
| LR com ajuste | inclinacao_calibracao | 50 | 1.1132 | 0.5663 | 2.2104 |  |
| LR com ajuste | sensibilidade_p_0_5 | 50 | 0.8491 | 0.6531 | 0.9231 |  |
| LR com ajuste | especificidade_p_0_5 | 50 | 0.8998 | 0.7955 | 0.9804 |  |
| SVM | sensibilidade_regra_nativa | 50 | 0.8535 | 0.6531 | 0.9362 | regra nativa, não comparável |
| SVM | especificidade_regra_nativa | 50 | 0.8855 | 0.7727 | 0.9796 | regra nativa, não comparável |
| RF | brier | 50 | 0.0931 | 0.0527 | 0.1367 |  |
| RF | log_loss | 50 | 0.2984 | 0.1912 | 0.4439 |  |
| RF | intercepto_calibracao | 50 | -0.0196 | -0.8961 | 0.9982 |  |
| RF | inclinacao_calibracao | 50 | 1.1601 | 0.5089 | 2.1411 |  |
| RF | sensibilidade_p_0_5 | 50 | 0.8790 | 0.6735 | 0.9615 |  |
| RF | especificidade_p_0_5 | 50 | 0.8553 | 0.7273 | 0.9512 |  |
| XGBoost | brier | 50 | 0.0969 | 0.0549 | 0.1414 |  |
| XGBoost | log_loss | 50 | 0.3189 | 0.2066 | 0.5009 |  |
| XGBoost | intercepto_calibracao | 50 | -0.0033 | -0.7659 | 1.0013 |  |
| XGBoost | inclinacao_calibracao | 50 | 1.1609 | 0.5835 | 1.9772 |  |
| XGBoost | sensibilidade_p_0_5 | 50 | 0.8744 | 0.6939 | 0.9434 |  |
| XGBoost | especificidade_p_0_5 | 50 | 0.8570 | 0.7500 | 0.9512 |  |

## 10. Regra de seleção

Padrão: LR com ajuste. Critérios: C1 (diferença estimada em G ≥ 0,02), C2 (diferença positiva em pelo menos 80% das repetições válidas), C3 (F estimável e diferença em F ≥ 0) e C4 (sem falha técnica pendente, G estimável e sem aviso de não convergência). O intervalo e a taxa de descarte não são critério.

| Desafiante | C1 | C2 | C3 | C4 | Elegível |
|---|:-:|:-:|:-:|:-:|:-:|
| RF | não | sim | sim | sim | não |
| SVM | não | não | não | sim | não |
| XGBoost | não | não | não | sim | não |

**Algoritmo recomendado: LR com ajuste.**

- rf: não elegível; critério(s) não cumprido(s): C1.
- svm: não elegível; critério(s) não cumprido(s): C1, C2, C3.
- xgb: não elegível; critério(s) não cumprido(s): C1, C2, C3.
- Nenhum desafiante cumpriu C1 a C4; fica o padrão da regra, lr_tuning.

## 11. Otimismo da seleção

O algoritmo escolhido sai dos mesmos resultados de validação cruzada usados para estimá-lo; por isso a sua estimativa de AUC é otimista por seleção, e nenhuma validação cruzada externa à seleção nem conjunto independente a corrige. A escolha vale como recomendação para estas três características nesta base, sem validação no teste ou externa. Uma correção formal do viés não foi aplicada, e isso fica registrado como limitação.

## 12. O que o intervalo não captura e limites do ajuste de Bonferroni

| Não captura | Consequência |
|---|---|
| Variabilidade do treinamento: não há reajuste nas réplicas | O intervalo tende a ser estreito demais. |
| Variabilidade da escolha de hiperparâmetros | O intervalo tende a ser estreito demais. |
| Dependência entre dobras e repetições, cujos treinos se sobrepõem | A partição só aparece como mínimo e máximo entre repetições. |
| A validação cruzada estima o desempenho esperado do procedimento, não de um modelo final | Não há modelo final validado. |
| Os pacientes não são amostra aleatória de uma população clínica; o alvo é suspeita radiológica | Não se generaliza para a prática clínica. |
| Reuso dos pacientes de validação depois da Etapa 3 da S4, e as escolhas analíticas | Declarado como limitação. |

- O ajuste pressupõe que os intervalos tenham a cobertura declarada; os daqui são condicionais e tendem a ser estreitos, então a correção NÃO garante controle do erro de família: é uma convenção de largura.
- É conservador com contrastes correlacionados: os seis compartilham o comparador, e o desfecho F é um subconjunto do G.
- Não cobre outras escolhas analíticas (grades, desfechos, regras de validade), nem os contrastes descritivos.
- Exige caudas extremas no bootstrap: com B = 10.000, cada cauda de 0,417% se apoia em cerca de 42 réplicas.

## 13. Falhas técnicas e avisos

Falhas técnicas pendentes: 0.

| Modelo | Unidades | Unidades com aviso | Avisos de convergência |
|---|--:|--:|--:|
| LR fixa | 50 | 0 | 0 |
| LR com ajuste | 50 | 0 | 0 |
| SVM | 50 | 0 | 0 |
| RF | 50 | 0 | 0 |
| XGBoost | 50 | 0 | 0 |

## 14. Teste e independência em relação à S4

A partição histórica de teste não foi usada na modelagem, no ajuste, na validação cruzada, nas métricas, no bootstrap nem na seleção; seus identificadores só entram na verificação estrutural pública da S4. Este benchmark é independente das métricas da S4: não reaproveita nenhum resultado dela.

## 15. Figuras

- `figuras/F1.png`
- `figuras/F1.svg`
- `figuras/F2.png`
- `figuras/F2.svg`
- `figuras/F3.png`
- `figuras/F3.svg`
- `figuras/F4.png`
- `figuras/F4.svg`
- `figuras/F5.png`
- `figuras/F5.svg`
- `figuras/F6.png`
- `figuras/F6.svg`
- `figuras/F7.png`
- `figuras/F7.svg`
- `figuras/F8.png`
- `figuras/F8.svg`
