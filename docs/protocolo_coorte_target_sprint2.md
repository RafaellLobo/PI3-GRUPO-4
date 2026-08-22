# Protocolo de Definição de Coorte e Variável-Alvo — Sprint 2

**Projeto:** PI3-GRUPO-4 — Radiômica de Nódulos Pulmonares (LIDC-IDRI)
**Card relacionado:** S2 — Definição da coorte e variável-alvo
**Responsável pela execução:** Rafael
**Status:** Aguardando revisão por outro integrante
**Data:** 22/08/2026

---

## 1. Objetivo

Definir, de forma reprodutível e metodologicamente defensável, os critérios de elegibilidade de nódulos/pacientes do LIDC-IDRI e a regra de consolidação da variável `malignancy` em uma variável-alvo binária (`target_binary`), incluindo a estratégia de divisão treino/validação/teste que previne data leakage entre pacientes.

Este protocolo é o entregável formal exigido pelo Definition of Done da Sprint 2 e é dependência direta das atividades:

- S2 — Pipeline de seleção, segmentação e máscaras
- S2 — Extração radiômica piloto com PyRadiomics

## 2. Aviso metodológico obrigatório

A variável `malignancy` do LIDC-IDRI **não representa confirmação anatomopatológica de câncer**. Trata-se de uma avaliação subjetiva de probabilidade/aparência de malignidade, atribuída por radiologistas em escala de 1 a 5. Este protocolo, e qualquer modelo derivado dele, produz um **score de risco baseado em características radiológicas disponíveis no dataset** — não um diagnóstico. O LIDC-IDRI não fornece classificação Lung-RADS nativa.

## 3. Fonte de validação empírica

Todas as decisões abaixo foram validadas contra os dados reais do dataset via script diagnóstico (`scripts/explore_cohort_criteria.py`), que processou os **1018 scans** disponíveis no banco de anotações (SQLite) do `pylidc`, resultando em **2651 nódulos (clusters de annotations)** analisados. Resultado consolidado em `scripts/output/cohort_diagnostic_raw.csv`.

## 4. Critérios de elegibilidade da coorte

| Critério                                               | Regra                        | Justificativa                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------------ | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nº mínimo de annotations com características completas | `≥ 3` annotations por nódulo | Garante consolidação a partir de múltiplos observadores, evitando que o alvo reflita a opinião de um único radiologista. Empiricamente, isolado, este critério já define os elegíveis: 1392/2651 nódulos (52,5%).                                                                                                                                                                                    |
| Diâmetro mínimo                                        | Não aplicado como filtro     | Validado que 99,9% dos clusters já possuem diâmetro médio ≥ 3mm — é uma propriedade estrutural de como o `pylidc` materializa `unblindedReadNodule` com `characteristics`, não um critério que corte casos adicionais. Mantido apenas como **assert de sanidade** no pipeline de segmentação: qualquer nódulo elegível com diâmetro < 3mm deve gerar alerta, pois indicaria inconsistência upstream. |

**Correção de decisão intermediária:** o corte de `≥3 annotations` foi originalmente justificado como forma de estabilizar a mediana de malignancy. A validação empírica mostrou que isso é apenas parcialmente verdadeiro — mesmo com `≥3`, uma fração relevante dos elegíveis ainda produz mediana fracionária (ver Seção 5). A justificativa correta e mantida é: **robustez de consenso multi-observador**, não eliminação de indefinição de alvo (que é tratada separadamente).

## 5. Regra de consolidação da variável-alvo

### 5.1 Cálculo

Para cada nódulo elegível, calcula-se a **mediana** dos valores de `malignancy` das annotations do cluster.

### 5.2 Binarização

```
target_binary = 0   se malignancy_median ≤ 2
target_binary = 1   se malignancy_median ≥ 4
target_binary = NaN se malignancy_median ∈ {2.5, 3.0, 3.5}
```

### 5.3 Motivo da exclusão em três categorias (não apenas "nota 3")

A mediana de um conjunto par de valores (comum quando o nódulo tem 4 annotations, situação frequente no LIDC) é a média dos dois valores centrais, o que produz frações (`2.5`, `3.5`) sempre que os dois radiologistas centrais discordam. Isso é um efeito estrutural do método, não um erro de leitura. Por isso, o campo de rastreabilidade não usa um booleano único, mas uma categoria explícita:

```python
exclusion_reason = np.select(
    [malignancy_median == 3.0, malignancy_median.isin([2.5, 3.5])],
    ["consensus_indeterminate", "fractional_median_even_raters"],
    default="included"
)
```

Essa distinção é obrigatória para a defesa acadêmica do trabalho: as duas causas de exclusão têm naturezas diferentes (indecisão real de consenso vs. efeito de contagem par de observadores) e precisam ser reportadas separadamente na Sprint 3.

### 5.4 Impacto empírico da regra (base de 1392 elegíveis)

| Categoria                                                            | N                               | %     |
| -------------------------------------------------------------------- | ------------------------------- | ----- |
| Classe 0 (baixo risco)                                               | 315                             | 22,6% |
| Classe 1 (alto risco)                                                | 301                             | 21,6% |
| Excluído — mediana = 3.0 (`consensus_indeterminate`)                 | 513                             | 36,9% |
| Excluído — mediana fracionária par (`fractional_median_even_raters`) | 263                             | 18,9% |
| **Base modelável final**                                             | **616 nódulos / 422 pacientes** | —     |

Balanceamento de classe na base modelável: **51,1% / 48,9%** — considerado saudável, sem necessidade de técnicas de rebalanceamento nesta fase.

## 6. Estratégia de divisão treino/validação/teste

### 6.1 Regra

Divisão realizada por `patient_id`, nunca por `nodule_id`, para impedir que nódulos do mesmo paciente apareçam em partições diferentes (data leakage anatômico/fisiológico).

- **Proporção:** 70% treino / 15% validação / 15% teste
- **Método:** `GroupShuffleSplit` (ou `StratifiedGroupKFold` para validação cruzada), `groups=patient_id`, `y=target_binary`
- **Seed fixa:** `random_state=42`, para reprodutibilidade

### 6.2 Validação empírica do split

```
Treino    : n=431 (70,0%) | classe0=228 | classe1=203 | pacientes=295
Validação : n=90  (14,6%) | classe0=39  | classe1=51  | pacientes=63
Teste     : n=95  (15,4%) | classe0=48  | classe1=47  | pacientes=64
Vazamento de paciente entre partições: 0
```

**Limitação documentada:** apenas 42 dos 422 pacientes possuem nódulos de ambas as classes. Isso limita a capacidade de estratificação perfeita por grupo e gera variância entre folds (observada em `StratifiedGroupKFold`, n_splits=5: classe 1 variando entre 46 e 70 exemplos por fold de teste). Este ponto não compromete a Sprint 2 (cujo objetivo é validar o pipeline), mas deve ser considerado ao reportar métricas finais na Sprint 3 — intervalos de confiança devem refletir essa variância.

### 6.3 Transformações aprendidas

Qualquer normalização, imputação ou seleção de features deve ser ajustada exclusivamente nos dados de treino de cada fold/partição, nunca no dataset completo antes do split — previne vazamento de informação estatística entre partições.

## 7. Infraestrutura de dados

- O banco de anotações do `pylidc` (SQLite embutido) cobre os 1018 pacientes independentemente da presença local do volume DICOM, permitindo que este diagnóstico seja válido para a coorte inteira sem necessidade de download prévio.
- Para a extração radiômica real (Sprint 2, atividades subsequentes), apenas os **422 pacientes da base modelável** têm download de imagem priorizado — não a coleção completa (1018 pacientes), evitando uso desnecessário de armazenamento e tempo de download (~55GB reais vs. ~120GB+ da coleção completa).
- Manifesto de download filtrado gerado em `scripts/generate_download_manifest.py`, produzindo `scripts/output/lidc_manifest_422.tcia` (428 séries CT, 6 pacientes com múltiplas reconstruções mantidas para decisão posterior na etapa de segmentação).
- Dados armazenados em disco externo (fora do repositório), com `pylidc.conf` local apontando para o caminho correspondente. Nenhum arquivo DICOM é versionado no GitHub.

## 8. Riscos e decisões que permanecem em aberto para Sprints futuras

1. **Unidade de análise da extração radiômica:** decidir se a extração de features ocorre por cluster consolidado (uma máscara/ROI de consenso) ou por annotation individual (gerando múltiplas linhas de features por nódulo). Esta decisão pertence à atividade de segmentação/máscaras, mas deve respeitar a mesma regra de split por paciente definida aqui, para não introduzir leakage indireto.
2. **Nódulos excluídos por indefinição de alvo** (776 no total) permanecem na tabela piloto com `exclusion_reason` preenchido, não são fisicamente descartados — preservando a possibilidade de revisão metodológica futura sem necessidade de reprocessar a extração radiômica.
3. **Reprodutibilidade de ambiente:** `environment.yml` deve ser atualizado com `pandas=1.5.3` (dependência usada no diagnóstico, ainda não commitada no arquivo de ambiente).

## 9. Reprodutibilidade

- Script de diagnóstico: `scripts/explore_cohort_criteria.py`
- Script de geração de manifesto: `scripts/generate_download_manifest.py`
- Dataset de diagnóstico: `scripts/output/cohort_diagnostic_raw.csv` (não versionado — dado derivado, local)
- Seed fixa: `42` em todas as operações estocásticas (split de amostra)
- Ambiente: Python 3.9, conda env `pi3-radiomics`

---

_Este documento consolida as decisões metodológicas da Sprint 2 relativas à definição de coorte e variável-alvo. Qualquer alteração nas regras acima deve ser registrada em nova revisão deste arquivo, mantendo o histórico de decisões anteriores._
