# Convenção de Identificação de Nódulos — Sprint 3

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Card relacionado:** S3 — Canonização e validação da identificação de nódulos
**Escopo do documento:** documentação canônica de como um nódulo é identificado no projeto
**Documentos relacionados:** [`docs/sprint2/pipeline_selecao_mascaras.md`](../sprint2/pipeline_selecao_mascaras.md) (registro histórico da Sprint 2, **não alterado por esta atividade**) · [`docs/sprint3/baseline_regressao_logistica.md`](baseline_regressao_logistica.md)
**Status:** consolidado

---

## 1. Objetivo e o que esta atividade não é

Registrar, em um único lugar, **como um nódulo é identificado** no projeto: quais formatos textuais existem, o que cada um significa, e qual é a identidade canônica.

> **Nenhuma convenção foi alterada e nenhum identificador foi renomeado.** Esta atividade é de formalização: dá *formatter* e *parser* explícitos a formatos que já existiam, e nomeia uma chave que os scripts já usavam sem nome.

**Nenhum artefato científico sofreu alteração de conteúdo.** Os artefatos derivados — `data/modeling_table_sprint3.csv`, `reports/sprint3/baseline/patient_split.csv`, `confusion_matrix.csv` e `baseline_metrics.json` — **foram reexecutados** como prova de não regressão, e os quatro voltaram **byte a byte idênticos**. A reexecução é o método de verificação, não uma mudança de conteúdo: a identidade dos hashes é resultado medido, e não consequência de os arquivos não terem sido escritos.

Fora do escopo: alterar `selection.nodule_id`, alterar os artefatos versionados, reextrair a base radiômica, criar uma terceira string de identificação.

## 2. Regra central

> **Os identificadores textuais são representação, não chave científica.**

A identidade de um nódulo é a tupla **`(scan_id, original_nodule_idx)`**. As strings existem para ler e escrever os artefatos que já as carregam, e para servir de rótulo legível — nunca para carregar semântica que o código precise reconstruir por *parsing* ad-hoc.

Consequências práticas:

- todo código novo que precise identificar um nódulo usa a tupla, não a string;
- toda leitura de uma string passa por um *parser* nomeado de `src/radiomics/ids.py` — nunca por `split`, `rsplit` ou fatiamento manual;
- um ID textual nunca deve ser reconstruído a partir de dados de outra proveniência para servir de chave de junção.

## 3. Formato histórico da Sprint 2 — piloto

```
{patient_id}_N{nodule_idx:02d}          exemplo: LIDC-IDRI-0006_N03
```

`Nxx` é o **`nodule_idx` original** devolvido por `Scan.cluster_annotations()`, contando também os clusters descartados, e **nunca reindexado após os filtros**. Em `LIDC-IDRI-0006`, `N00`–`N02` são descartados e o nódulo elegível continua sendo `N03`.

| Item | Onde |
|---|---|
| Produtor | [`selection.nodule_id`](../../src/radiomics/selection.py) — **congelada** |
| Parser | `ids.decompor_id_piloto` |
| Padrão | `ids.PADRAO_ID_PILOTO` — `^(LIDC-IDRI-\d{4})_N(\d+)$` |

**Esta função está congelada.** Ela é a chave dos artefatos versionados em `reports/sprint2/`, e a validação da Sprint 2 compara o resultado dela, string a string, contra esses CSVs. Alterá-la — inclusive para acrescentar `scan_id` — quebraria a equivalência científica que o [documento da Sprint 2](../sprint2/pipeline_selecao_mascaras.md) atesta.

Observação: este formato também nomeia clusters **descartados** ([`selection.py`](../../src/radiomics/selection.py), lista `descartes`). Descartados não têm posição em `selecionados`, o que é uma das razões pelas quais este formato e o da Seção 4 não podem ser fundidos.

## 4. Formato histórico da extração da Sprint 3

```
{patient_id}_N{selected_position:02d}_scan{scan_id}    exemplo: LIDC-IDRI-0132_N01_scan7
```

Presente em `data/base_radiomica_oficial_422.csv` (958 linhas) e propagado para `data/modeling_table_sprint3.csv`.

| Item | Onde |
|---|---|
| Produtor original | Notebook de extração da Sprint 3 — **não versionado** |
| Produtor versionado | `ids.formatar_id_extracao` |
| Parser | `ids.decompor_id_extracao` |
| Padrão | `ids.PADRAO_ID_EXTRACAO` — `^(LIDC-IDRI-\d{4})_N(\d+)_scan(\d+)$` |

`ids.formatar_id_extracao` foi criada para que o formato tenha um produtor versionado e testável: até então, a única descrição dele no repositório era prosa. Ela **descreve** o formato da base existente, e é validada contra as 958 linhas já persistidas; não reextrai nem reescreve a base.

O `scan_id` corresponde a `pylidc.Scan.id`, verificado coerente com `patient_id` em 958/958 linhas.

## 5. A diferença semântica entre os dois `Nxx`

Os dois formatos compartilham o prefixo `LIDC-IDRI-XXXX_Ndd`, **com significados diferentes**:

| | Sprint 2 (piloto) | Sprint 3 (extração) |
|---|---|---|
| O que `Nxx` conta | índice original do clustering | posição na lista filtrada `selecionados` |
| Conta descartados? | **sim** | **não** |
| Estável sob filtragem? | sim | não |
| Carrega scan? | não | sim |

O notebook de extração da Sprint 3 gerou a posição a partir de:

```python
selecionados, _ = selection.selecionar_clusters(scan, clusters=clusters)
for idx, (_, anns) in enumerate(selecionados):
```

O `enumerate` reindexa o que sobrou depois do filtro. Por isso as duas numerações divergem sempre que há descarte antes da posição em questão: **448 das 958 linhas (46,8%)**.

O sentido do deslocamento importa e é sempre o mesmo — a filtragem só pode remover clusters antes da posição, nunca acrescentar:

- `original_nodule_idx − selected_position`: de **1 a 15** nas 448 linhas divergentes;
- equivalentemente, `selected_position − original_nodule_idx`: de **−15 a −1**.

Ou seja, o índice original é sempre **maior ou igual** à posição filtrada, e a diferença é exatamente o número de clusters descartados antes dela.

### 5.1 O risco, e o alcance exato de cada proteção

Um ID no formato da Sprint 3 construído com o índice *original* no lugar da posição seria **sintaticamente indistinguível** do formato correto e semanticamente errado. Como `build_modeling_table.py` usa a posição para indexar `selecionados[pos]`, a linha seria associada a **outro cluster** — e as invariantes de contagem (958 mapeadas, 607/416, 310/297) **não pegariam o erro**, porque nenhuma delas verifica *qual* cluster foi associado.

**O que a rejeição mútua dos parsers protege.** Os dois padrões são ancorados em `^` e `$`, e os *parsers* se recusam mutuamente: `decompor_id_piloto` levanta `ValueError` diante de um ID de extração, e vice-versa. Isso impede a confusão **entre os formatos S2 e S3** — ler um ID sem scan como se tivesse scan, ou tratar um `Nxx` de piloto como posição filtrada. A checagem está em [`scripts/validate_nodule_ids.py`](../../scripts/validate_nodule_ids.py).

**O que ela NÃO protege.** A rejeição mútua é uma garantia **sintática**. Ela não detecta — e não tem como detectar — um ID S3 bem formado cujo campo `Nxx` tenha sido preenchido com `original_nodule_idx` em vez de `selected_position`: os dois produzem a mesma string `LIDC-IDRI-XXXX_Ndd_scanN`, e nenhum parser consegue distinguir um do outro olhando só para o texto.

Essa parte da correção é **semântica**, e se apoia em três coisas, nenhuma delas um parser:

1. **O contrato explícito** — `ids.formatar_id_extracao` documenta que o segundo argumento é a posição em `selecionados`, e `docs` e docstrings repetem a distinção;
2. **A proveniência da base** — as 958 linhas de `data/base_radiomica_oficial_422.csv` são artefato congelado, cuja numeração já foi caracterizada (Seção 5): 448 linhas divergentes, deslocamento sempre no sentido `original ≥ posição`;
3. **A reconstrução end-to-end em [`scripts/build_modeling_table.py`](../../scripts/build_modeling_table.py)** — que recupera `original_nodule_idx` do `pylidc` para as 958 linhas, exige unicidade da chave canônica sobre elas e falha se qualquer posição cair fora de `selecionados`. É esta etapa, e não o parser, que liga a string ao cluster real.

Uma inversão de argumentos numa chamada futura a `formatar_id_extracao` continuaria, portanto, sendo um erro possível. O que o repositório garante é que ele não passaria silenciosamente pela reconstrução do alvo.

## 6. Identidade canônica

```
CHAVE_CANONICA = ("scan_id", "original_nodule_idx")
```

| Componente | Origem | Por que |
|---|---|---|
| `scan_id` | `pylidc.Scan.id` | Remove a ambiguidade de pacientes com mais de uma série reconstruída |
| `original_nodule_idx` | posição em `cluster_annotations()` | Estável sob filtragem: não é reindexado |

Ambos são colunas de `data/modeling_table_sprint3.csv`, e a unicidade do par é invariante verificada antes de qualquer escrita em [`scripts/build_modeling_table.py`](../../scripts/build_modeling_table.py).

### 6.1 Por que não `(patient_id, original_nodule_idx)`

Hoje esse par também é único na base modelável — mas apenas porque **nenhum paciente dela contribui com mais de um `scan_id`**. É acidente de cobertura, não propriedade do domínio: no universo completo do `pylidc`, `(patient_id, nodule_index)` já colide em **21 chaves, 8 pacientes** (`scripts/output/cohort_diagnostic_raw.csv`), seis deles presentes na base oficial.

Adotá-la como chave criaria um teste que passa hoje e proíbe, amanhã, exatamente o caso multi-série que a chave canônica existe para suportar. O número de pacientes multi-scan é, por isso, **reportado e não exigido** pelo validador.

### 6.2 Por que não uma terceira string canônica

Foi considerada e **descartada nesta atividade**. Os dois formatos existentes já cobrem os artefatos, e a identidade canônica é uma tupla — introduzir uma terceira representação textual acrescentaria uma forma a mais para confundir, sem consumidor concreto que a justificasse.

## 7. Artefatos históricos congelados

Nenhum destes tem o conteúdo alterado por esta atividade, nem deve ter por atividades futuras de refatoração de identificadores. Os derivados da Sprint 3 foram reexecutados para prova de não regressão e voltaram byte a byte idênticos (Seção 1):

| Artefato | Formato | Razão do congelamento |
|---|---|---|
| `reports/sprint2/features/piloto_piloto_v2_consenso50.{csv,parquet}` | Sprint 2 | Testemunha da validação de `src/radiomics` |
| `reports/sprint2/features/piloto_piloto_v3_consenso50.csv` | Sprint 2 | **Não tem colunas `scan_id`/`nodule_idx`** — o `nodule_id` é a única identidade |
| `reports/sprint2/validacao/descartes_piloto_v2_consenso50.csv` | Sprint 2 | Testemunha dos descartes e da ordem |
| `reports/sprint2/historico/v1/*`, `historico/v3_oficial/*` | Sprint 2 | Histórico versionado deliberadamente |
| `reports/sprint2/validacao/*.json`, `reports/sprint2/comparacao/*.json` | — | Carregam `sha256` das tabelas: qualquer edição invalida a proveniência |
| `reports/sprint3/baseline/*` | — | Teste congelado ([baseline, Seção 10.6](baseline_regressao_logistica.md)) |
| `notebooks/**` (inclusive `historico/`) | Sprint 2 | Registro do que efetivamente rodou |
| `data/base_radiomica_oficial_422.csv` | Sprint 3 | Entrada oficial imutável; origem no Drive do projeto |
| `scripts/output/cohort_diagnostic_raw.csv` | `(patient_id, nodule_index)` | Diagnóstico histórico, já documentado como impróprio como fonte de alvo |

`docs/sprint2/pipeline_selecao_mascaras.md` também permanece **intacto**, como registro histórico da Sprint 2.

## 8. Onde a convenção vive no código

| Arquivo | Papel |
|---|---|
| [`src/radiomics/ids.py`](../../src/radiomics/ids.py) | Padrões, *formatter*, *parsers* e `CHAVE_CANONICA` |
| [`src/radiomics/selection.py`](../../src/radiomics/selection.py) | `nodule_id` — produtor congelado do formato da Sprint 2 |
| [`scripts/build_modeling_table.py`](../../scripts/build_modeling_table.py) | Consome o formato da Sprint 3; valida a unicidade da chave canônica |
| [`scripts/validate_selection_masks.py`](../../scripts/validate_selection_masks.py) | Valida `src/radiomics` contra os artefatos da Sprint 2 |
| [`scripts/validate_nodule_ids.py`](../../scripts/validate_nodule_ids.py) | Valida a convenção descrita neste documento |

## 9. Validação

A convenção é validada em **duas camadas**, com alcances deliberadamente diferentes.

### 9.1 Validação estrutural e offline

```bash
python scripts/validate_nodule_ids.py
```

Sem `pytest` e sem dependência nova. **Não exige `pylidc` nem DICOM**, e não cria nem modifica nenhum arquivo.

Confere: ida e volta dos dois formatos, incluindo a fronteira de largura do `:02d`; rejeição mútua dos *parsers* e recusa de entradas malformadas; conformidade dos sete artefatos versionados da Sprint 2, com `nodule_id == selection.nodule_id(patient_id, nodule_idx)` linha a linha nos que carregam as colunas; conformidade das 958 linhas da base oficial, incluindo `formatar_id_extracao(...) == nodulo_id` para todas elas e unicidade de `(scan_id, posição)` sobre as 958; e unicidade da chave canônica nas **607** linhas modeláveis, que são as únicas em que `original_nodule_idx` está persistido.

### 9.2 Validação semântica, com recuperação dos índices originais

```bash
python scripts/build_modeling_table.py
```

A unicidade **direta** de `(scan_id, original_nodule_idx)` sobre os **958 registros mapeados** é provada aqui, e não na camada offline: `data/base_radiomica_oficial_422.csv` não carrega a coluna `original_nodule_idx`, que precisa ser recuperada do banco de anotações do `pylidc` scan a scan. É também esta etapa que liga cada ID ao cluster real e que detecta posições fora de `selecionados` (Seção 5.1).

Por isso as duas camadas não são redundantes, e nenhuma substitui a outra.

---

_Documento canônico da convenção de identificação. Qualquer mudança de formato, de produtor ou de chave canônica deve ser registrada em nova revisão, mantendo o histórico das decisões anteriores._
