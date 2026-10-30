# `data/` — dados locais

Esta pasta versiona apenas este README e o placeholder `.gitkeep`. **Os dados em `data/` são locais, não versionados e não acompanham um clone do repositório**: imagens DICOM, tabelas radiômicas e CSVs derivados ficam fora do Git.

| Arquivo | Situação no checkout inspecionado | Origem ou produtor documentado |
|---|---|---|
| `base_radiomica_oficial_422.csv` | Cópia local encontrada, não versionada. | Entrada externa com referência no Drive do projeto; produtor original não versionado. |
| `modeling_table_sprint3.csv` | Arquivo local encontrado, não versionado. | Derivado pelo script de construção da tabela de modelagem. |
| `vies_malignidade_mascara_pequena.csv` | Arquivo local encontrado, não versionado. | Produtor original e procedimento de obtenção ainda não comprovados. |

Encontrar uma cópia local não comprova a disponibilidade atual de sua fonte externa. Os caminhos abaixo são referências documentadas; o Drive não foi consultado nesta atualização.

## `base_radiomica_oficial_422.csv`

Entrada oficial da Sprint 3, lida por [`scripts/build_modeling_table.py`](../scripts/build_modeling_table.py).

Foi produzida pela **atividade de extração radiômica completa da Sprint 3**, e contém **958 linhas de nódulos** e **107 features radiômicas**, antes da reconstrução e filtragem do target.

O caminho abaixo é uma referência documental ao artefato original no Drive do projeto; sua disponibilidade atual não foi verificada:

```
08_extracao_completa_oficial_422/CSV E JSON/base_radiomica_oficial_422.csv
```

Para reproduzir a Sprint 3, baixe esse arquivo do Drive e coloque-o em `data/` com o mesmo nome.

O notebook original da extração completa não está versionado, conforme a [convenção de identificação dos nódulos](../docs/sprint3/convencao_identificacao_nodulos.md). Esta base também é consumida pela reprodução da seleção da Sprint 4, descrita abaixo.

## `modeling_table_sprint3.csv`

Arquivo **derivado**, também local e não versionado. É produzido por [`scripts/build_modeling_table.py`](../scripts/build_modeling_table.py) a partir do CSV acima, e não precisa ser baixado — basta executar o script.

O produtor reconstrói o alvo a partir do banco de anotações embutido no `pylidc`, sem ler DICOM nem reextrair as features. Sua execução substitui a tabela se ela já existir; consulte o [guia de reprodução](../docs/guia_reproducao.md) antes de reexecutá-lo.

É entrada do split e do baseline oficial da Sprint 3, além das verificações e atividades exploratórias da Sprint 4. Os consumidores, resultados e manifestos estão relacionados no [índice de experimentos](../docs/indice_experimentos.md). O relatório da seleção também registra seu uso nas checagens de consistência com o treino canônico.

## `vies_malignidade_mascara_pequena.csv`

Entrada de alvo da reprodução fiel da seleção de features da Sprint 4. O consumidor direto versionado identificado é [`scripts/sprint4/reproduce_juan_feature_selection.py`](../scripts/sprint4/reproduce_juan_feature_selection.py): lê as colunas `nodulo_id` e `malignidade_mediana`, associa o alvo à base radiômica e aplica a [política oficial de binarização](../docs/protocolo_coorte_target_sprint2.md). A seleção usa somente o treino oficial, conforme a [auditoria da reprodução](../reports/sprint4/feature_selection/auditoria_reproducao_juan.md). O alvo representa suspeição radiológica, não malignidade histológica confirmada.

**Pendência de proveniência:** o script produtor original, a origem primária e um procedimento verificável de obtenção deste CSV não foram localizados no checkout. A [comparação com o pacote de Juan](../reports/sprint4/feature_selection/comparacao_pacote_juan.md), Seção 4, registra uma cópia de entrada idêntica à usada na auditoria, mas não estabelece seu produtor original nem comprova os arquivos usados na execução histórica do Colab.

O [relatório versionado da reprodução](../reports/sprint4/feature_selection/reproducao_juan/relatorio_reproducao.json) registra o arquivo no objeto `sha256_entradas["data/vies_malignidade_mascara_pequena.csv"]`, com os campos `sha256` e `sha256_lf_normalizado`. O SHA-256 bruto registrado é:

```text
67f3fd2c3a55de75a58c40e7380480a5df0f5747ac017ec2b1f6ec091c3c6478
```

Esse hash identifica a entrada registrada naquela execução; não comprova sua origem. Ele foi consultado no relatório, sem nova leitura integral ou recálculo do hash do dataset nesta atualização. Antes de reconstruir esta entrada a partir de outra fonte, é necessário consolidar sua proveniência e preservar o vínculo com os artefatos históricos.

## Regra

**Novas tabelas médicas ou radiômicas com dados individuais devem permanecer fora do Git.** Isso vale tanto para as entradas quanto para as tabelas derivadas desta pasta, protegidas pela [.gitignore](../.gitignore).

Os artefatos históricos já versionados em [`reports/`](../reports/), incluindo tabelas da Sprint 2, devem ser preservados. Não devem ser excluídos, regravados ou substituídos como parte de atualizações documentais; sua existência não autoriza o versionamento de novas tabelas individuais. Preserve também as regras de [.gitattributes](../.gitattributes) associadas aos hashes registrados.
