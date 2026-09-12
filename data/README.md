# `data/` — dados locais

Este é o único arquivo versionado desta pasta. **Todo o conteúdo real de `data/` é local e não versionado**: imagens DICOM, tabelas radiômicas e qualquer CSV derivado delas ficam fora do Git, conforme a política do repositório para dados médicos.

## `base_radiomica_oficial_422.csv`

Entrada oficial da Sprint 3, lida por [`scripts/build_modeling_table.py`](../scripts/build_modeling_table.py).

Foi produzida pela **atividade de extração radiômica completa da Sprint 3**, e contém **958 linhas de nódulos** e **107 features radiômicas**, antes da reconstrução e filtragem do target.

O artefato original está armazenado no Drive do projeto:

```
08_extracao_completa_oficial_422/CSV E JSON/base_radiomica_oficial_422.csv
```

Para reproduzir a Sprint 3, baixe esse arquivo do Drive e coloque-o em `data/` com o mesmo nome.

## `modeling_table_sprint3.csv`

Arquivo **derivado**, também local e não versionado. É produzido por [`scripts/build_modeling_table.py`](../scripts/build_modeling_table.py) a partir do CSV acima, e não precisa ser baixado — basta executar o script.

## Regra

**Nenhum CSV médico ou radiômico deve ser commitado.** Isso vale tanto para as tabelas de entrada quanto para as derivadas.
