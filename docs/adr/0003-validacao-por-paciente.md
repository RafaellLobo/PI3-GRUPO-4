# 0003 - Validar por paciente e preservar o protocolo P9

Status: Aceito

Data: 2026-11-23

Registro retrospectivo das regras implementadas e pré-registradas. Não modifica aprovações, coorte, split, sementes ou decisões científicas.

## Contexto

Um paciente pode ter vários nódulos. Misturar nódulos do mesmo paciente entre treino e avaliação compromete a separação entre grupos. Comparar classificadores com divisões, características ou critérios escolhidos depois de observar resultados também compromete a interpretação. O teste histórico já foi consultado e precisa conservar esse estado declarado.

## Decisão

Preservar o split oficial por paciente e, no P9, a validação cruzada aninhada com `StratifiedGroupKFold`: dez repetições de cinco dobras externas e cinco internas, usando as mesmas divisões por paciente para todos os modelos. Manter as três características do baseline, ordenação canônica, sementes externas `42 + r`, internas `1000 + 10*r + k` e fingerprints das divisões. Ajuste interno usa somente o treino externo; o P9 utiliza treino+validação e exclui o teste histórico de suas avaliações.

Aplicar C1–C4 do pré-registro para seleção, com LR ajustada como padrão. Apresentar G e F como conjunto geral e subconjunto de tamanho; intervalos descritivos não demonstram equivalência ou superioridade clínica. A pergunta de valor incremental da S4 permanece separada deste benchmark.

## Consequências

**Benefícios:** reduz vazamento por paciente; permite comparação pareada sob as mesmas divisões; critérios prévios e fingerprints tornam a revisão e a retomada mais rastreáveis.

**Custos e limites:** a execução envolve 250 unidades modelo/repetição/dobra e ajuste interno, com maior custo que um único split. Modelos ajustados e réplicas têm limites de dependência e seleção; sementes não garantem identidade entre bibliotecas/plataformas. Excluir o teste do P9 não apaga seus usos anteriores nem fornece validação externa.

**Rastreabilidade:** [pré-registro](../sprint6/preregistro_p9_benchmark.md), [configuração](../sprint6/config_benchmark_p9.json), [dobras](../../src/benchmark_s6/folds.py), [seleção](../../src/benchmark_s6/selecao.py), [emenda](../sprint4/emenda_01_auditoria_teste.md), [RF-03/RF-05/RF-07, RNF-01 e RD-01–RD-05](../requisitos.md).
