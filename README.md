# PI3 Grupo 4 — Radiômica e suspeição radiológica de nódulos pulmonares

## 1. Nome e descrição

Este projeto estuda medidas extraídas de tomografias de nódulos pulmonares para investigar a suspeição atribuída por radiologistas. O código ajuda pesquisadores, professores e estudantes a organizar as análises, comparar métodos e conferir de onde vêm os resultados.

O Projeto Integrador III do Grupo 4 utiliza a base pública LIDC-IDRI e um pipeline científico em Python. As análises também investigam a influência do tamanho dos nódulos e das escolhas de pré-processamento.

O alvo binário deriva da mediana dos escores de 1 a 5: mediana ≤ 2 recebe classe 0; mediana ≥ 4 recebe classe 1; medianas intermediárias ficam fora da modelagem binária. **Não é diagnóstico de câncer confirmado por histopatologia nem ferramenta clínica.** O LIDC-IDRI não fornece categorias Lung-RADS nativas.

A pergunta de valor incremental da radiômica é tratada pelo protocolo da Sprint 4. O benchmark P9, da Sprint 6, responde a outra pergunta: com volume, diâmetro máximo 3D e esfericidade fixos, trocar o classificador melhora a discriminação? Não misturar resultados de bases ou protocolos diferentes. O teste histórico já foi consultado em etapas anteriores; não é um conjunto globalmente intocado nem validação externa.

Licença do código: [MIT](LICENSE). A licença do código não substitui as condições de acesso e uso das fontes de dados.

## 2. Funcionalidades principais

- Seleção de nódulos e construção de máscaras por consenso dos anotadores, com código em `src/radiomics/` e notebooks históricos em `notebooks/sprint2/`.
- Extração radiômica nos notebooks, construção da tabela de modelagem e definição do alvo, com dependência de entradas externas documentadas.
- Split por paciente e baseline de regressão logística na Sprint 3.
- Infraestrutura de validação agrupada, seleção e análise incremental em `src/incremental/`.
- Benchmark P9 por linha de comando em `src/benchmark_s6/`: LR fixa, LR com ajuste, SVM, Random Forest e XGBoost; dobras externas/internas, métricas, bootstrap, seleção predefinida e manifestos.
- Conferência de configuração sem dados privados; testes sintéticos e inspeção de resultados agregados já versionados.

O projeto é executado por scripts, módulos Python e notebooks. Este repositório não fornece interface web, API de atendimento ou serviço de inferência hospedado.

| Pasta | Conteúdo |
|---|---|
| `src/radiomics/` | Seleção, máscaras e identificação de nódulos |
| `src/incremental/` | Infraestrutura de avaliação da Sprint 4 |
| `src/benchmark_s6/` | Coordenador e componentes do benchmark P9 |
| `scripts/` | Ambiente, tabelas, split, baseline e validações |
| `notebooks/` | Extração e análises históricas |
| `docs/` | Requisitos, decisões, protocolos e guias |
| `reports/` | Artefatos e resultados das etapas, com exceções históricas documentadas |
| `tests/` | Testes automatizados, incluindo entradas sintéticas |
| `data/` | Entradas locais; dados não acompanham o clone |

Os resultados existentes do P9 estão em [resumo_benchmark.md](reports/sprint6/benchmark/resumo_benchmark.md). A execução registrada tem 509 nódulos/353 pacientes em treino+validação; o recorte F de 12 ≤ diâmetro < 30 mm é subconjunto de G. A LR com ajuste foi mantida pela regra predefinida, sem demonstrar superioridade clínica ou equivalência dos modelos.

## 3. Tecnologias usadas

| Tecnologia | Uso e versão declarada |
|---|---|
| Python | `3.9.13` no ambiente declarativo, alinhado à guarda do P9 |
| NumPy / pandas / SciPy | Dados e estatística; `1.23.5` / `1.5.3` / `1.10.1` |
| scikit-learn / XGBoost | Modelagem e validação; `1.3.2` / `2.1.4` |
| PyRadiomics / SimpleITK | Características de imagem; `3.1.0` / `2.5.5` no YAML |
| pydicom / pylidc | DICOM e anotações LIDC-IDRI; `2.4.4` / `0.2.3` |
| SQLAlchemy | Dependência do acesso às anotações pelo pylidc; `1.4.54` |
| Matplotlib | Figuras; `3.7.3` |
| Jupyter/IPython | Notebooks; `ipykernel` declarado sem versão fixada |
| Setuptools | `setuptools<81`, para manter o `pkg_resources` exigido pelo pylidc 0.2.3 |
| Git e Conda/pip | Versionamento e preparação do ambiente |

O arquivo [environment.yml](environment.yml) é a referência de instalação. Não há lockfile completo de dependências transitivas. O Colab histórico usou outras versões, inclusive PyRadiomics de desenvolvimento: o YAML não reproduz automaticamente todos os experimentos históricos. Consulte o [guia de reprodução](docs/guia_reproducao.md) e o [pré-registro P9](docs/sprint6/preregistro_p9_benchmark.md) antes de reexecutá-los.

## 4. Como instalar e executar

**Sem os dados externos, execute somente os passos 4.2–4.4:** instalação, conferência da configuração e testes sintéticos. A seção 4.5 envolve as etapas científicas e exige entradas autorizadas, ambiente compatível e aprovações do protocolo.

### 4.1 Pré-requisitos

- Git e Miniconda/Anaconda disponíveis no terminal.
- Windows é a plataforma das execuções locais documentadas; outros sistemas não foram validados nesta entrega.
- Para os passos 4.2–4.4, não são necessários DICOMs, Drive, credenciais, tabelas individuais ou checkpoints.

Abra o Anaconda Prompt ou um terminal no qual `conda` funcione. Execute os comandos a partir da raiz do projeto.

### 4.2 Clonar e instalar

```text
git clone https://github.com/RafaellLobo/PI3-GRUPO-4.git
cd PI3-GRUPO-4
conda env create -f environment.yml
conda activate pi3-radiomics
python scripts/validate_environment.py
```

O YAML fixa Python 3.9.13 para atender à guarda do P9 e seleciona OpenBLAS para evitar carregar Intel OpenMP e LLVM OpenMP no mesmo processo, conforme a [orientação do threadpoolctl](https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md).

O YAML inclui `setuptools<81` porque o pylidc 0.2.3 importa `pkg_resources`. Essa dependência legada ainda pode emitir aviso de depreciação; o limite de versão mantém o import disponível. A [documentação do Setuptools](https://setuptools.pypa.io/en/stable/deprecated/pkg_resources.html) descreve a descontinuação dessa API.

A saída esperada do último comando termina em `Ambiente validado com sucesso.`. Ele verifica imports e mostra versões; não certifica equivalência com o ambiente de cada experimento. Se o ambiente `pi3-radiomics` já existir, ative-o e confira as versões antes de decidir sobre qualquer atualização.

Se houver interferência de plugins do Conda, o procedimento alternativo já documentado pelo projeto é `conda --no-plugins env create --solver classic -f environment.yml`. Não desative controles do Windows para contornar falhas de importação. O histórico de problemas de ambiente está no [guia](docs/guia_reproducao.md).

### 4.3 Executar a conferência do P9 sem dados privados

```text
python -m src.benchmark_s6 --help
python -m src.benchmark_s6 --modo configuracao --conferir docs/sprint6/config_benchmark_p9.json
```

O primeiro comando lista os modos e argumentos. O segundo executa uma rotina real do coordenador: confere a configuração canônica e as referências de código da S4, sem treinamento nem carga da base de pacientes. Deve encerrar com código 0 e apresentar:

```text
conferido: docs/sprint6/config_benchmark_p9.json
sha256_configuracao=a0d096a5415244f17f40360a2a5a8901bc8af1da5cf8687b891f920e1ee122b3
```

O separador do caminho pode variar entre sistemas. Não alterar a configuração para obter esse hash. Se houver divergência, conferir a versão do código e o protocolo.

### 4.4 Executar testes rápidos com dados sintéticos

```text
python -m unittest discover -s tests/benchmark_s6 -p test_selecao.py
```

A saída deve terminar em `OK`. No checkout validado nesta entrega, foram 22 testes. Eles verificam a regra de seleção com entradas sintéticas; não treinam modelos nem leem a tabela de pacientes. O framework `unittest` faz parte do Python, sem instalação adicional de pytest.

Para inspecionar os resultados científicos já produzidos, abra [o resumo P9](reports/sprint6/benchmark/resumo_benchmark.md), [a seleção](reports/sprint6/benchmark/selecao_algoritmo.json) e [a figura F1](reports/sprint6/benchmark/figuras/F1.png). A existência desses arquivos e o sucesso dos testes não significam que o benchmark foi reexecutado.

### 4.5 Executar etapas científicas com dados autorizados

Um clone contém código e resultados agregados, mas **não contém todas as entradas necessárias para extração e treinamento**. Antes dessas rotinas:

1. Obtenha o LIDC-IDRI e as anotações conforme a [documentação dos dados](data/README.md). Para a Sprint 3, a entrada documentada é `data/base_radiomica_oficial_422.csv`; a tabela derivada é `data/modeling_table_sprint3.csv`. A extração completa histórica não tem produtor integralmente versionado neste checkout.
2. Confira origem, hashes e partições contra o protocolo da etapa. A tabela de modelagem completa tem 607 nódulos/416 pacientes; o P9 restringe a carga a treino+validação, com 509/353. Não substituir a tabela oficial por uma seleção manual de linhas.
3. Preserve `reports/sprint3/baseline/patient_split.csv`; não refaça o split para melhorar resultados. DICOMs exigem configuração local do pylidc, conforme [validação inicial](docs/lidc_initial_validation.md).
4. Para o P9 definitivo, confira todas as versões exigidas, inclusive Python 3.9.13, os registros de aprovação `P9` e `S4_E3_REGISTRADA`, o manifesto definitivo da S4 e a árvore Git limpa. O YAML fixa o patch 3.9.13; confira também as demais dependências e os registros da execução histórica. Dados indisponíveis ou divergências de ambiente são bloqueios reais, não autorizações para alterar o protocolo.
5. Use diretório novo para os artefatos e checkpoints fora do repositório. O destino padrão já tem resultados congelados; não o apague nem use o baseline apenas para testar a instalação, pois ele treina e sobrescreve relatórios.

Após essas condições e autorização da equipe, a interface existente é:

```text
python -m src.benchmark_s6 --modo definitivo --checkpoints CAMINHO_LOCAL_FORA_DO_REPOSITORIO --destino reports/sprint6/NOVA_EXECUCAO --processos 1 --manifesto-s4 reports/sprint4/etapa3_validacao/manifesto_etapa3_validacao.json
```

`CAMINHO_LOCAL_FORA_DO_REPOSITORIO` e `NOVA_EXECUCAO` são marcadores que precisam ser substituídos por destinos novos apropriados. Esse comando ajusta modelos e pode ser demorado; não integra a verificação rápida. Os modos `implementacao` e `verificar` também exigem dados/checkpoints e controles próprios; `verificar` não é uma simples leitura de CSVs. Procedimentos e pré-condições completos: [pré-registro P9](docs/sprint6/preregistro_p9_benchmark.md) e [guia de revisão](docs/sprint6/GUIA_REVISAO_BENCHMARK_S6.md).

**Validação em 24/11/2026:** um ambiente Conda novo foi criado de `environment.yml`, em diretório separado, com Python 3.9.13 e setuptools 80.9.0. Passaram o validador de ambiente, a ajuda da CLI, a conferência da configuração canônica, os 22 testes de seleção e a suíte de 613 testes de `benchmark_s6` (`OK`, com 1 teste ignorado). O aviso de depreciação de `pkg_resources` permanece, mas o import do pylidc funciona. Não foram executadas extração nem a reprodução científica do benchmark; os testes usam dados sintéticos. O patch de Python está alinhado ao P9; a reprodução científica integral continua condicionada às entradas e aprovações da seção 4.5.

## 5. Equipe e papéis

Papéis do encerramento do PI3; a revisão por pares não transfere a autoria dos blocos.

| Integrante | Responsabilidade |
|---|---|
| Rafael Lazarin Lobo de Almeida | Benchmark P9, rastreabilidade/reprodução e reunião dos slides entregues pelos autores |
| Samara Ruiz Silva | Síntese científica, tabela de resultados e integração/padronização do relatório final |
| Leonardo Secco Silva | Literatura, indeterminados, associação com tamanho e revisão da integração do relatório |
| Juan Carlos Guatierri Alves | Explicabilidade SHAP, síntese das características e revisão do bloco de Rafael |
| Victor Hugo Stein Blauth | Síntese documental Lung-RADS e apoio à conferência visual e de links |

## 6. Requisitos e decisões técnicas

- [Requisitos funcionais, não funcionais e de domínio](docs/requisitos.md).
- [ADR 0001 — Pipeline científico modular em Python](docs/adr/0001-pipeline-cientifico-python.md).
- [ADR 0002 — Artefatos em arquivos e separação dos dados individuais](docs/adr/0002-artefatos-arquivos-dados-locais.md).
- [ADR 0003 — Validação agrupada por paciente e protocolo P9](docs/adr/0003-validacao-por-paciente.md).

As datas dos ADRs são as datas deste registro retrospectivo das decisões implementadas, não datas presumidas de sua adoção original. Eles não alteram protocolos ou aprovam novas execuções.

Referências técnicas: [índice de experimentos](docs/indice_experimentos.md), [guia de reprodução das etapas históricas](docs/guia_reproducao.md), [política de coorte/alvo](docs/protocolo_coorte_target_sprint2.md), [piloto S2](docs/sprint2/piloto_radiomico.md), [baseline S3](docs/sprint3/baseline_regressao_logistica.md), [protocolo S4](docs/sprint4/protocolo_valor_incremental.md), [emenda sobre o teste histórico](docs/sprint4/emenda_01_auditoria_teste.md) e [pré-registro P9](docs/sprint6/preregistro_p9_benchmark.md). Documentos históricos preservam o estado da época; conferir o protocolo e os artefatos da etapa pretendida.
