# PI3 — Grupo 4: Radiomics e Scoring de Risco

## Sobre o projeto

Projeto acadêmico dedicado ao estudo de radiômica aplicada a imagens de Tomografia Computadorizada (TC) de nódulos pulmonares.

A Sprint 2 contém um **pipeline piloto de extração e validação radiômica** sobre o LIDC-IDRI: consolidação de máscaras por consenso, extração de atributos com PyRadiomics e validação da coorte e da variável-alvo. A documentação técnica canônica está em [`docs/sprint2/piloto_radiomico.md`](docs/sprint2/piloto_radiomico.md).

## Objetivo

Investigar a extração de características quantitativas de nódulos pulmonares e sua relação com categorias de risco, incluindo o Lung-RADS.

## Tecnologias principais

- Python 3.9
- NumPy, SciPy, Matplotlib e scikit-learn
- pydicom, pylidc, SimpleITK e PyRadiomics
- SQLAlchemy
- Conda e pip para gerenciamento das dependências

O ambiente utiliza Python 3.9 e foi validado no Windows com Python 3.9.23.

## Estrutura do repositório

```text
.
├── data/                  # Dados locais (DICOM); não versionado
├── docs/                  # Documentação técnica
│   └── sprint2/           # Documento canônico do piloto radiômico
├── notebooks/
│   └── sprint2/           # Notebooks de extração, comparação e validação
├── reports/
│   └── sprint2/           # Configs, validações, features e artefatos históricos
├── scripts/               # Scripts de diagnóstico e validação de ambiente
├── src/                   # Código de biblioteca do projeto
├── .gitattributes         # EOL determinístico para artefatos científicos
├── .gitignore
├── environment.yml        # Especificação do ambiente Conda
└── README.md
```

## Como começar

### Pré-requisitos

- Git
- Miniconda ou Anaconda com Conda
- Windows, ambiente validado atualmente

Não é necessário instalar o Python 3.9 globalmente, pois ele é criado pelo `environment.yml`.

### Clonar o repositório

```bash
git clone https://github.com/RafaellLobo/PI3-GRUPO-4.git
cd PI3-GRUPO-4
```

### Criar o ambiente

Na raiz do repositório clonado, execute:

```bash
conda --no-plugins env create --solver classic -f environment.yml
```

No ambiente Windows testado, `--no-plugins` foi necessário para evitar a interferência do plugin de Terms of Service do Conda sobre canais globais não utilizados. As dependências Conda do projeto são obtidas do `conda-forge`, conforme definido em `environment.yml`.

### Ativar o ambiente

```bash
conda activate pi3-radiomics
```

### Validar o ambiente

```bash
python scripts/validate_environment.py
```

O script importa as principais dependências e exibe as versões instaladas, retornando erro caso algum import falhe.

## Dataset

O dataset LIDC-IDRI não está incluído no repositório. Dados locais, arquivos DICOM e demais imagens médicas não devem ser versionados; o diretório `data/` está reservado para esse conteúdo local.

## Equipe

PI3 — Grupo 4.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
