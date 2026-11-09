# S6 [P0] - Parte A: quadro comparativo da validacao cruzada por paciente

Rotulos: o alvo e suspeicao radiologica (classe 1 e classe 0 pela mediana das notas), nao malignidade confirmada. Os intervalos sao descritivos (bootstrap de pacientes, condicional aos modelos ajustados). A LR com ajuste e otimista por selecao. Nenhum resultado e validacao independente.

Populacao: G = 509 nodulos, 353 pacientes (251 classe 1 e 258 classe 0); F (12 a 30 mm) = 226 nodulos, 187 pacientes (135 classe 1 e 91 classe 0). Contagens somadas sobre as dobras de uma repeticao e identicas nas 10 repeticoes.

## Quadro 1 - AUC por modelo e desfecho

| Modelo | Desfecho | AUC | IC 95% descritivo | Rotulo |
|---|---|--:|---|---|
| LR fixa | G | 0,9442 | [0,9246; 0,9624] |  |
| LR com ajuste | G | 0,9449 | [0,9254; 0,9628] | otimista por selecao |
| SVM | G | 0,9424 | [0,9224; 0,9605] |  |
| RF | G | 0,9476 | [0,9288; 0,9646] |  |
| XGBoost | G | 0,9438 | [0,9242; 0,9619] |  |
| LR fixa | F | 0,7903 | [0,7359; 0,8460] |  |
| LR com ajuste | F | 0,7907 | [0,7364; 0,8464] | otimista por selecao |
| SVM | F | 0,7890 | [0,7357; 0,8437] |  |
| RF | F | 0,7968 | [0,7459; 0,8501] |  |
| XGBoost | F | 0,7844 | [0,7329; 0,8389] |  |

## Quadro 2 - Contrastes da familia principal (contra a LR com ajuste)

| Contraste | Desfecho | Diferenca estimada | IC Bonferroni (6 contrastes) | IC 95% sem correcao |
|---|---|--:|---|---|
| RF - LR com ajuste | G | 0,0027 | [-0,0043; 0,0098] | [-0,0024; 0,0078] |
| RF - LR com ajuste | F | 0,0061 | [-0,0250; 0,0368] | [-0,0168; 0,0291] |
| XGBoost - LR com ajuste | G | -0,0011 | [-0,0075; 0,0050] | [-0,0057; 0,0034] |
| XGBoost - LR com ajuste | F | -0,0063 | [-0,0358; 0,0221] | [-0,0276; 0,0152] |
| SVM - LR com ajuste | G | -0,0025 | [-0,0080; 0,0013] | [-0,0064; 0,0005] |
| SVM - LR com ajuste | F | -0,0017 | [-0,0213; 0,0129] | [-0,0157; 0,0094] |

## Quadro 3 - Sensibilidade a particao (todos os contrastes)

| Contraste | Desfecho | Grupo | Media | Minimo | Maximo | Repeticoes com diferenca > 0 |
|---|---|---|--:|--:|--:|--:|
| RF - LR com ajuste | G | familia | 0,0027 | 0,0001 | 0,0044 | 10 de 10 |
| RF - LR com ajuste | F | familia | 0,0061 | -0,0016 | 0,0209 | 8 de 10 |
| XGBoost - LR com ajuste | G | familia | -0,0011 | -0,0060 | 0,0012 | 6 de 10 |
| XGBoost - LR com ajuste | F | familia | -0,0063 | -0,0230 | 0,0102 | 3 de 10 |
| SVM - LR com ajuste | G | familia | -0,0025 | -0,0060 | 0,0011 | 1 de 10 |
| SVM - LR com ajuste | F | familia | -0,0017 | -0,0141 | 0,0101 | 4 de 10 |
| LR com ajuste - LR fixa | G | descritivo | 0,0007 | -0,0007 | 0,0018 | 8 de 10 |
| LR com ajuste - LR fixa | F | descritivo | 0,0004 | -0,0059 | 0,0042 | 6 de 10 |
| RF - XGBoost | G | descritivo | 0,0038 | 0,0016 | 0,0073 | 10 de 10 |
| RF - XGBoost | F | descritivo | 0,0124 | 0,0044 | 0,0280 | 10 de 10 |
| RF - SVM | G | descritivo | 0,0051 | 0,0014 | 0,0085 | 10 de 10 |
| RF - SVM | F | descritivo | 0,0078 | 0,0040 | 0,0142 | 10 de 10 |
| XGBoost - SVM | G | descritivo | 0,0014 | -0,0047 | 0,0064 | 5 de 10 |
| XGBoost - SVM | F | descritivo | -0,0046 | -0,0237 | 0,0076 | 3 de 10 |

## Regra de selecao do P9 (reaplicada a partir de selecao_algoritmo.json)

Algoritmo recomendado: LR com ajuste.

| Desafiante | C1 (magnitude) | C2 (consistencia) | C3 (sem piora em F) | C4 (integridade) |
|---|---|---|---|---|
| RF | nao cumpre | cumpre | cumpre | cumpre |
| XGBoost | nao cumpre | nao cumpre | nao cumpre | cumpre |
| SVM | nao cumpre | nao cumpre | nao cumpre | cumpre |

## Leitura permitida

Descritiva e condicional: diferenca estimada, compativel com, nao detectada nesta base. Vetados: significativo, superior, comprovado, demonstrado, estatisticamente, p-valor. O bootstrap nao reajusta os modelos nem captura a variabilidade da escolha de hiperparametros; o limite de Bonferroni vale para qualquer dependencia entre os contrastes, desde que cada intervalo marginal tenha a cobertura nominal; como o bootstrap e aproximado, essa cobertura e apenas aproximada. Intervalos largos nao provam equivalencia.
