# Pipeline de Seleção, Segmentação e Máscaras — Sprint 2

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Atividade:** S2 — Pipeline de seleção, segmentação e máscaras
**Documentos relacionados:** [`docs/sprint2/piloto_radiomico.md`](piloto_radiomico.md) (documento canônico do piloto) · [`docs/protocolo_coorte_target_sprint2.md`](../protocolo_coorte_target_sprint2.md) (política de coorte e alvo)

---

## 1. Objetivo

Extrair para código reutilizável, em `src/radiomics/`, a lógica de **seleção de nódulos** e **consolidação de máscaras** já validada no piloto radiômico da Sprint 2, **preservando exatamente** o comportamento científico que produziu os artefatos versionados em `reports/sprint2/`.

A entrega **não reprojeta** a lógica. O código é uma transposição da célula 20 de `notebooks/sprint2/PI3_G4_sprint2_v2_extração_piloto.ipynb` (função `extrair_scan` e auxiliares), reorganizada em funções nomeadas e documentada, sem alteração de regra, de ordem de avaliação ou de tolerância numérica.

## 2. O que esta atividade é — e o que ela não é

> **Não foi criado nenhum modelo de segmentação automática.** Não há rede neural, treinamento, inferência ou algoritmo de segmentação próprio nesta entrega.

A segmentação utilizada é a que **já existe no LIDC-IDRI**: os contornos manuais desenhados pelos radiologistas, disponíveis como `annotations` através do `pylidc`. O que o código faz é **consolidar** essas anotações existentes em uma única região de interesse por consenso, e recortá-la do volume. Nenhum voxel é segmentado por nós.

## 3. Escopo

**Dentro do escopo — os sete comportamentos extraídos:**

1. obter os clusters de annotations de um `Scan` via `pylidc`;
2. selecionar nódulos com pelo menos 3 annotations;
3. aplicar o critério de diâmetro médio mínimo de 3,0 mm;
4. gerar a máscara de consenso com `clevel = 0.5`;
5. obter a bounding box correspondente;
6. expandir a bbox em 4 voxels de contexto respeitando os limites do volume;
7. produzir CT ROI e máscara ROI com exatamente a mesma shape e alinhamento espacial.

**Fora do escopo** (permanece no notebook, sem alteração): PyRadiomics e a extração de features; `binCount`/`binWidth`; variável-alvo e `rotular()`; salvaguardas de intensidade (`intensidade_nao_finita`, `intensidade_fora_de_faixa_hu`); conversão para SimpleITK e `SetSpacing`; split treino/validação/teste; baseline; processamento dos 422 pacientes; CLI; configuração em YAML; classes e abstrações adicionais.

Nada foi alterado em notebooks, artefatos CSV/JSON/Parquet, no protocolo de coorte ou nos scripts anteriores.

## 4. O que cada módulo faz

### 4.1 `src/radiomics/selection.py`

Depende apenas de `numpy` e do banco de anotações do `pylidc`; **não requer DICOM**.

- **obtém os clusters via `pylidc`** — `scan.cluster_annotations()`, chamado **sem argumentos** (defaults `metric='min'`, `tol=None`, `factor=0.9`, `min_tol=1e-1`). Alterar esses parâmetros reagruparia as annotations e renomearia todos os nódulos do scan;
- **mantém o índice original** — o `nodule_idx` é a posição do cluster na lista devolvida pelo clustering, contando também os descartados, e **nunca é reindexado após os filtros**. Em `LIDC-IDRI-0006`, `N00`–`N02` são descartados e o nódulo elegível continua sendo `N03`;
- **exige ≥ 3 annotations** — `len(anns) < min_anotadores` → descarte `leitores_insuficientes`. Comparação estritamente `<`: exatamente 3 annotations passa;
- **aplica diâmetro médio ≥ 3,0 mm** — média aritmética de `a.diameter` sobre **todas** as annotations do cluster; abaixo do limite → descarte `diametro_abaixo_do_minimo`. Diâmetro exatamente 3,0 mm passa.

A **ordem** dos dois critérios é parte do contrato: annotations primeiro, diâmetro depois. Um cluster que falha nos dois sai como `leitores_insuficientes`.

| Função | Assinatura |
|---|---|
| `nodule_id` | `(patient_id, nodule_idx) -> str` — formato `{patient_id}_N{idx:02d}` |
| `diametro_medio_mm` | `(anns) -> float` |
| `motivo_descarte_cluster` | `(anns, min_anotadores=3, diametro_min_mm=3.0) -> str \| None` |
| `selecionar_clusters` | `(scan, min_anotadores=3, diametro_min_mm=3.0, clusters=None) -> (selecionados, descartes)` |

`clusters` existe apenas para reaproveitar um `cluster_annotations()` já executado — nunca para alterar os parâmetros do clustering. Quando é `None`, o `pylidc` é chamado sem argumentos.

### 4.2 `src/radiomics/masks.py`

Depende de `numpy` e de `pylidc.utils.consensus`.

- **cria a máscara `consenso50` com `clevel = 0.5`** — os contornos dos radiologistas do cluster são consolidados em uma região única: um voxel entra na máscara quando ≥ 50% das annotations o incluem;
- **obtém `cmask` e `cbbox`** — máscara booleana e tupla de 3 `slice` indexável diretamente no volume, ambas devolvidas pelo `consensus`. O argumento `pad` do `consensus` **não é usado**: a margem é aplicada depois, com semântica de recorte própria;
- **expande a bbox em 4 voxels respeitando o volume** — `start = max(0, start - margem)` e `stop = min(dimensão, stop + margem)`, com clipping nas bordas. Uma tupla **nova** é devolvida: a `cbbox` original permanece intacta, porque o offset de realocação da máscara é calculado a partir dela;
- **produz ROI CT e máscara ROI com mesma shape e alinhamento espacial** — a CT é recortada pela bbox expandida; a máscara, que cobre apenas a bbox original, é reposicionada pelo offset entre as duas caixas. Os eixos não são transpostos e nada é convertido para SimpleITK. Invariantes: `sub.shape == mask_exp.shape`, `mask_exp.dtype == cmask.dtype` (`bool`) e `mask_exp.sum() == cmask.sum()`.

| Função | Assinatura |
|---|---|
| `montar_mascara` | `(anns, modo='consenso50') -> (cmask, cbbox)` |
| `expandir_bbox` | `(cbbox, vol_shape, margem=4) -> tuple[slice, ...]` |
| `recortar_roi` | `(vol, cmask, cbbox, margem=4) -> (sub, mask_exp)` |

Nenhum dos dois módulos importa PyRadiomics, SimpleITK ou pandas.

## 5. Parâmetros preservados

Valores idênticos aos de `PARAMS` na célula 8 do notebook e aos registrados em `reports/sprint2/config/config_piloto_v2.json`:

| Parâmetro | Valor | Onde vive no código |
|---|---|---|
| Mínimo de annotations por nódulo | `3` | `selection.MIN_ANOTADORES_PADRAO` |
| Diâmetro médio mínimo | `3.0` mm | `selection.DIAMETRO_MIN_MM_PADRAO` |
| Estratégia de máscara | `consenso50` → `clevel = 0.5` | `masks.CLEVEL` |
| Margem de contexto | `4` voxels | `masks.MARGEM_CONTEXTO_VOXELS_PADRAO` |

Vocabulário de descarte preservado literalmente: `leitores_insuficientes`, `diametro_abaixo_do_minimo`.

## 6. Validação executada

### 6.1 Validação automatizada

```bash
python scripts/validate_selection_masks.py
```

Sem `pytest` e sem dependência nova — apenas `assert`, `csv` e `numpy`. Nenhum arquivo é criado ou modificado pela execução.

A referência é o par de artefatos versionados que o notebook produziu:

- `reports/sprint2/features/piloto_piloto_v2_consenso50.csv`
- `reports/sprint2/validacao/descartes_piloto_v2_consenso50.csv`

Isso é possível porque `n_anotadores`, `diametro_medio_mm` e `voxels_mascara` derivam apenas do banco de anotações do `pylidc` — nenhuma imagem DICOM é necessária para reproduzi-los. Por isso **não existe uma segunda cópia da lógica do notebook** no repositório: o CSV é a testemunha.

Pacientes validados, escolhidos por cobertura de caso e não por ordem:

| Paciente | Caso que cobre |
|---|---|
| `LIDC-IDRI-0001` | nódulo único, nenhum descarte; único com DICOM presente em `data/` |
| `LIDC-IDRI-0003` | descarte no **primeiro** índice, três elegíveis depois |
| `LIDC-IDRI-0005` | descarte no **último** índice |
| `LIDC-IDRI-0006` | `N00`–`N02` descartados e `N03` elegível — prova a não-reindexação |
| `LIDC-IDRI-0008` | `N00` com **exatamente 3** annotations — prova a fronteira `<` do critério |

Checagens sem DICOM: `nodule_id` e ordem; pares `(nodule_id, motivo)` de descarte e ordem; `nodule_idx` não reindexado; `n_anotadores` (igualdade exata); `diametro_medio_mm` (erro relativo ≤ 1e-9); `voxels_mascara` (igualdade exata de inteiro); `dtype` booleano da máscara; bbox válida (3 slices contíguos, dentro do volume, shape coerente com a máscara); expansão de 4 voxels com clipping correto e bbox original intacta.

**Resultado:**

| Métrica | Valor |
|---|---|
| Pacientes | 5 |
| Checagens | **226** |
| Skips | **0** |
| Determinismo | **3 repetições, resultado idêntico** |

A saída termina em `VALIDAÇÃO OK — 226 checagens, 0 skip(s)`.

### 6.2 Validação real com DICOM

Executada sobre o volume real de `LIDC-IDRI-0001`, nódulo `N00`:

| Item | Valor |
|---|---|
| Volume original | `(512, 512, 133)` |
| bbox original | `[(340, 392), (297, 341), (86, 95)]` |
| bbox expandida (margem 4) | `[(336, 396), (293, 345), (82, 99)]` |
| ROI (CT e máscara) | `(60, 52, 17)` |
| Voxels totais da máscara | **5.428** |

Os 5.428 voxels são exatamente o `voxels_mascara` registrado no CSV do piloto. As dimensões da ROI conferem com a expansão sem clipping em nenhum eixo: `52+8 = 60`, `44+8 = 52`, `9+8 = 17`.

Quando o volume não está disponível localmente, esse bloco é registrado como **SKIP informativo** e a validação não falha.

### 6.3 Inspeção visual

```bash
python scripts/visualize_selection_mask.py
```

Ferramenta **opcional** de conferência visual, não faz parte da validação automatizada. Abre uma janela matplotlib com três painéis — CT ROI, máscara e sobreposição semitransparente — na slice de maior área de máscara. Por padrão nada é gravado em disco (existe um `--salvar CAMINHO` opcional).

Resultado da inspeção de `LIDC-IDRI-0001_N00`:

| Item | Valor |
|---|---|
| Slice local escolhida | **7** |
| Slice correspondente no volume original | **z = 89** |
| Voxels da máscara nesse corte | **924** |

**A máscara aparece visualmente alinhada à estrutura correspondente na CT:** a sobreposição cobre a massa do nódulo com as bordas acompanhando o contorno, sem deslocamento em nenhum eixo — confirmação visual do alinhamento que a validação automatizada já afirmava numericamente.

## 7. Limitações

1. **Cobertura de pacientes.** A validação cobre 5 pacientes (9 nódulos elegíveis, 5 descartes) dos 25 do piloto. Os demais 20 pacientes do CSV não foram verificados nesta entrega; os 422 da coorte modelável estão fora do escopo.
2. **ROI validada em um único paciente.** Apenas `LIDC-IDRI-0001` tem DICOM no repositório, então os itens 6 e 7 foram exercitados sobre um volume real uma única vez.
3. **Nenhum caso de borda real exercitado.** Nenhuma bbox dos nódulos validados encosta nos limites do volume, então o clipping de `expandir_bbox` foi conferido pela fórmula, não por um caso em que ele efetivamente corta.
4. **Nenhum descarte por diâmetro, máscara vazia ou modo `leitorN`.** Nos artefatos do piloto os 30 descartes são todos `leitores_insuficientes`; esses caminhos existem no código e vêm do notebook, mas não têm testemunha empírica.
5. **A divergência entre notebook e `scripts/explore_cohort_criteria.py` permanece.** O notebook conta `len(anns)`; o script de diagnóstico conta apenas annotations com `characteristics` preenchidos. Na prática os números coincidem, mas as duas implementações não são a mesma regra. Preservamos a do notebook.
6. **A célula 33 do notebook não foi ajustada.** Ela verifica, por `inspect.getsource(extrair_scan)`, que a margem de contexto está em vigor, testando strings literais no corpo da função. Se o notebook passar a importar de `src/`, essa célula precisará ser reescrita — caso contrário levantará `RuntimeError`.
7. **As cópias da lógica nos outros notebooks continuam vivas.** As células 16 de `PI3_G4_comparacao_binCount_binWidth.ipynb` e de `historico/PI3_G4_piloto_v3_binWidth_validacao.ipynb` repetem as mesmas funções. Substituí-las por imports é trabalho de outro card.

## 8. Divergência registrada e não resolvida — critério de diâmetro

Registro explícito, para decisão metodológica futura:

1. **O notebook aplica `diametro_min_mm = 3.0` como filtro efetivo**, com motivo de descarte próprio (`diametro_abaixo_do_minimo`). O [protocolo de coorte](../protocolo_coorte_target_sprint2.md), Seção 4, tem redação diferente: "não aplicado como filtro", mantido apenas como *assert de sanidade*, sob o argumento de que 99,9% dos clusters já satisfazem o limite por construção do `pylidc`.
2. **Esta atividade preserva o comportamento do notebook**, porque foi ele que gerou os artefatos científicos validados em `reports/sprint2/`. Trocar o filtro por um *assert* agora seria uma mudança de regra científica disfarçada de refatoração, e poderia alterar resultados sobre a coorte completa sem que ninguém tivesse decidido isso.
3. **A divergência NÃO é resolvida aqui.** A revisão fica para decisão metodológica futura, a ser registrada em nova revisão do protocolo. Nos 25 pacientes do piloto a diferença é indistinguível: nenhum nódulo foi descartado por diâmetro, e o menor diâmetro médio entre os 44 extraídos é 5,70 mm.

---

_Documento da microentrega de extração. A referência canônica do piloto continua sendo [`docs/sprint2/piloto_radiomico.md`](piloto_radiomico.md)._
