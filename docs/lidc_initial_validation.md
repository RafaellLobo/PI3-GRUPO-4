# Validação inicial do LIDC-IDRI

## 1. Objetivo

Esta atividade teve como objetivo validar tecnicamente:

- o acesso a uma amostra local do dataset LIDC-IDRI;
- a estrutura dos arquivos DICOM da série de Tomografia Computadorizada (CT);
- a estrutura das anotações XML;
- a leitura dos metadados e dos pixels com `pydicom`;
- a correspondência entre DICOM, XML e o `Scan` do `pylidc`;
- o carregamento do volume CT completo;
- a identificação inicial de um nódulo e de suas annotations.

O escopo desta validação foi estritamente técnico e não incluiu interpretação clínica, classificação Lung-RADS ou extração de características radiômicas.

## 2. Dataset e amostra utilizada

| Campo | Valor |
|---|---|
| Dataset | LIDC-IDRI |
| Paciente | LIDC-IDRI-0001 |
| Quantidade de DICOMs | 133 |
| StudyInstanceUID | `1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178` |
| SeriesInstanceUID | `1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192` |

## 3. Estrutura observada

A organização conceitual observada na amostra foi:

```text
Collection
└── Patient
    └── Study
        └── CT Series
            └── arquivos DICOM

XML de anotações
└── referencia Study, Series e imagens pelos respectivos UIDs
```

O `StudyInstanceUID` identifica o estudo, enquanto o `SeriesInstanceUID` identifica a série de imagens dentro desse estudo. Esses identificadores permitem relacionar os arquivos DICOM, o XML de anotações e o objeto `Scan` representado pelo `pylidc`, sem depender de nomes locais de diretórios ou manifestos.

## 4. Validação DICOM

O script [`scripts/inspect_dicom.py`](../scripts/inspect_dicom.py) localizou os arquivos recursivamente, leu o primeiro DICOM com `pydicom` e validou o acesso ao `pixel_array`.

Resultados observados:

- PatientID: `LIDC-IDRI-0001`
- Modality: `CT`
- quantidade de DICOMs: 133
- SliceThickness: 2.5 mm
- PixelSpacing: `[0.703125, 0.703125]`
- Rows: 512
- Columns: 512
- `pixel_array shape`: `(512, 512)`
- dtype: `int16`
- intervalo de valores observado no primeiro DICOM: -1024 a 4095

## 5. Estrutura das anotações XML

O script [`scripts/inspect_lidc_xml.py`](../scripts/inspect_lidc_xml.py) analisou o XML com tratamento de namespace pelo nome local das tags.

Estrutura contabilizada:

- XMLs encontrados: 1
- `readingSession`: 4
- `unblindedReadNodule`: 13
- `smallNodule`: 0
- `nonNodule`: 13

### Relação entre unblindedReadNodule e pylidc

A inspeção estrutural do XML e a consulta ao `pylidc` apresentaram as seguintes contagens:

| Tipo | Quantidade |
|---|---:|
| `unblindedReadNodule` total | 13 |
| com `characteristics` | 4 |
| sem `characteristics` | 9 |
| com ROI | 13 |
| com `edgeMap` | 13 |
| `scan.annotations` | 4 |
| clusters encontrados | 1 |

Todos os 13 elementos `unblindedReadNodule` possuem pelo menos um ROI e um `edgeMap`. Os nove elementos sem `characteristics` possuem exatamente um ROI e um `edgeMap` cada.

O `scan.annotations` retornou quatro objetos `Annotation`. Portanto, a quantidade retornada pelo `pylidc` coincide numericamente com os quatro elementos `unblindedReadNodule` que possuem `characteristics`. O método `cluster_annotations()` agrupou essas quatro annotations em um único cluster.

Esta etapa não validou um mapeamento individual 1:1 entre cada `noduleID` do XML e cada objeto `Annotation` do `pylidc`. A coincidência observada é exclusivamente numérica e estrutural, sem interpretação clínica.

### Primeiro unblindedReadNodule com characteristics

- noduleID: `Nodule 001`
- subtlety: 5
- internalStructure: 1
- calcification: 6
- sphericity: 3
- margin: 3
- lobulation: 3
- spiculation: 4
- texture: 5
- malignancy: 5
- ROIs: 8
- edgeMap/pontos: 948
- primeiro imageZposition: -125.000000

O campo `malignancy` registra uma avaliação presente nas anotações do LIDC-IDRI. Esse valor não deve ser interpretado isoladamente como diagnóstico clínico definitivo nem como uma categoria Lung-RADS.

## 6. Validação com pylidc

A consulta ao banco interno do `pylidc`, filtrada simultaneamente pelo `StudyInstanceUID` e pelo `SeriesInstanceUID`, retornou exatamente um `Scan` correspondente.

Resultados do Scan e do volume:

- patient_id: `LIDC-IDRI-0001`
- slice_thickness: 2.5
- pixel_spacing: 0.703125
- annotations associadas ao Scan consultado: 4
- shape do volume CT: `(512, 512, 133)`
- dtype: `int16`
- intervalo observado no volume: -2048 a 3071
- clusters de nódulos: 1
- annotations no primeiro cluster: 4

Observou-se uma coincidência numérica entre os quatro elementos XML com `characteristics` e os quatro objetos em `scan.annotations`. Não foi estabelecida correspondência individual entre os `noduleID` do XML e os objetos `Annotation` do `pylidc`.

### Annotations do primeiro cluster

#### Annotation 1

- malignancy: 5
- spiculation: 4
- texture: 5
- contours: 8

#### Annotation 2

- malignancy: 5
- spiculation: 5
- texture: 5
- contours: 7

#### Annotation 3

- malignancy: 5
- spiculation: 3
- texture: 5
- contours: 8

#### Annotation 4

- malignancy: 4
- spiculation: 5
- texture: 4
- contours: 9

O método `cluster_annotations()` agrupa annotations que o `pylidc` considera correspondentes ao mesmo nódulo físico. Esse agrupamento representa a associação entre marcações dos leitores e não constitui interpretação clínica adicional.

## 7. Configuração local do pylidc

Para localizar os DICOMs, o `pylidc` precisa de uma configuração local que aponte para a raiz que contém os diretórios de pacientes no padrão `LIDC-IDRI-xxxx`.

Exemplo genérico:

```ini
[dicom]
path = <CAMINHO_LOCAL_PARA_LIDC-IDRI>
warn = True
```

Essa configuração depende da máquina do desenvolvedor, deve permanecer local e não deve ser versionada no Git.

## 8. Scripts utilizados

- [`scripts/inspect_dicom.py`](../scripts/inspect_dicom.py): localiza DICOMs, exibe metadados da primeira imagem e valida o acesso aos pixels.
- [`scripts/inspect_lidc_xml.py`](../scripts/inspect_lidc_xml.py): inspeciona a estrutura do XML e resume as anotações do primeiro nódulo com characteristics.
- [`scripts/validate_environment.py`](../scripts/validate_environment.py): importa as dependências principais e exibe suas versões para validar o ambiente.

## 9. Conclusão

A validação inicial confirmou que:

- o acesso à amostra do LIDC-IDRI está funcional;
- a série CT foi baixada e lida com sucesso;
- os DICOMs e o XML foram relacionados pelos UIDs de estudo e série;
- o `pylidc` encontrou exatamente um Scan correspondente;
- o volume CT completo foi carregado;
- um cluster de nódulo com quatro annotations foi identificado;
- a amostra está pronta para as próximas etapas de processamento e radiômica.
