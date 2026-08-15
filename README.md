# 🫁 Score de Risco de Malignidade em Nódulos Pulmonares via Radiômica

## 📋 Resumo

Este trabalho propõe uma **arquitetura de processamento em lote (batch)** para o cálculo de um score de risco de malignidade em nódulos pulmonares a partir de imagens de **Tomografia Computadorizada (TC)**, utilizando técnicas de **radiômica**.

A partir do dataset **LIDC-IDRI** ([ARMATO III et al., 2011](#-referências)), características de **forma**, **intensidade** e **textura** são extraídas por meio da biblioteca **PyRadiomics** ([VAN GRIETHUYSEN et al., 2017](#-referências)) e utilizadas para treinar modelos de Machine Learning, comparados a um **baseline convencional** baseado em atributos geométricos simples (volume, diâmetro e esfericidade) e **Regressão Logística**.

## 🏗️ Arquitetura

A solução foi estruturada em **três pipelines independentes**, garantindo **modularidade** e **reprodutibilidade** do processo:

| Pipeline | Responsabilidade |
|---|---|
| **1. Preparação dos Dados** | Segmentação, normalização e extração de atributos radiômicos |
| **2. Treinamento e Avaliação** | Ajuste dos modelos (baseline vs. radiômica completa) e validação estatística |
| **3. Inferência em Lote** | Aplicação do modelo versionado sobre novos exames em escala |

## ⚙️ MLOps e Conformidade Regulatória

Além da modelagem, o trabalho discute a **viabilidade da solução sob a ótica de MLOps** ([SCULLEY et al., 2015](#-referências)), abordando:

- **Versionamento** de dados, features e modelos
- **Rastreabilidade** do pipeline ponta a ponta
- **Monitoramento de drift** em produção
- **Conformidade regulatória** com a **RDC nº 657/2022 (ANVISA)**

## 📈 Resultados Esperados

Os resultados esperados indicam que a **abordagem radiômica completa** pode superar o desempenho do **baseline tradicional**, oferecendo suporte mais robusto à **estratificação de risco** de nódulos pulmonares.

## 📚 Referências

- ARMATO III, S. G. et al. **The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)**: a completed reference database of lung nodules on CT scans. *Medical Physics*, 2011.
- VAN GRIETHUYSEN, J. J. M. et al. **Computational Radiomics System to Decode the Radiographic Phenotype**. *Cancer Research*, 2017.
- SCULLEY, D. et al. **Hidden Technical Debt in Machine Learning Systems**. *NeurIPS*, 2015.
