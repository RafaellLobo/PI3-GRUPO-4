**FACULDADE DONADUZZI**

**CURSO DE INTELIGÊNCIA ARTIFICIAL**

**PROJETO INTEGRADOR III — GRUPO 4**

**PROTOCOLO DE EXECUÇÃO DO CARTÃO S6 \[P0\] — AVALIAÇÃO COMPARATIVA DOS MODELOS E TESTE DESCRITIVO**

**Versão 3, de 06/11/2026 — registrada antes de qualquer execução**

> Responsável: Samara Ruiz (atribuição no Trello a confirmar).  
> Revisor: Rafael Lazarin Lobo de Almeida (definido pela responsável em 05/11/2026).  
> Liberação: 04/11/2026. Prazo do cartão: 06/11/2026.  
> Estado do repositório consultado em 05/11/2026: ramo main, commit 1530be5, sem commits posteriores a 03/11/2026.

Toledo-PR  
04 de novembro de 2026

**1 FINALIDADE E LEITURA CORRETA DESTE DOCUMENTO**

O DoD do cartão exige “protocolo fixado antes da execução”. Este documento é esse protocolo: define o que será feito, em que ordem, com que fontes e com que regras de leitura, antes de qualquer número ser produzido ou o teste ser aberto. Nada aqui foi executado. Os números citados vêm dos arquivos já versionados do benchmark da Sprint 6 e foram conferidos contra o repositório em 04/11/2026; o que não pôde ser conferido está sinalizado como tal.

Regra de ordem, decidida pela responsável em 06/11/2026: a Parte A começa depois da abertura do pull request deste protocolo (o commit fixa a data e a hora do registro), sem aguardar a aprovação do revisor, porque ela só reorganiza resultados já versionados e não estima modelo nenhum. A revisão do Rafael ocorre em paralelo. Se ela pedir alteração, a Parte A é refeita e a mudança é registrada por adendo. A execução antes da aprovação fica declarada neste documento. A Parte B não segue essa regra: só começa com o protocolo aprovado e todas as pré-condições da Seção 5 comprovadas.

O cartão tem duas partes de natureza diferente, e o protocolo as separa. A Parte A (consolidação da validação cruzada) usa resultados que já existem e não exige nova execução de modelos. A Parte B (tabela descritiva do teste histórico) exige código ainda não existente no repositório e depende de etapas anteriores da Sprint 4. Por isso, a Parte A pode ser concluída dentro do prazo do cartão; a Parte B só pode ser executada quando as pré-condições da Seção 5 estiverem registradas.

**2 PREMISSAS E BASE NORMATIVA**

A premissa do cartão é que o conjunto oficial de teste foi acessado em estudos anteriores e, portanto, não é evidência independente de generalização. Isso já está registrado na Emenda 1 (revisão 3, aprovada pelo revisor em 03/11/2026), Seção E1, que classifica ao menos seis análises como de uso direto do teste (tipo A) e dez como de validação cruzada sobre a base inteira, que inclui pacientes de teste (tipo B), e na Seção E3, que torna o teste apenas descritivo. Este protocolo não altera nenhuma dessas decisões.

As fontes normativas são: (a) Emenda 1 ao protocolo de valor incremental, revisão 3 (Seções E1, E3, E5, E7); (b) protocolo de valor incremental da Sprint 4 (Seções 4.1, 7.4 e 7.5); (c) pré-registro P9 do benchmark da Sprint 6 (Seções 8 e 9); (d) relatório do benchmark, em reports/sprint6/benchmark/resumo_benchmark.md.

**3 PARTE A — CONSOLIDAÇÃO DA VALIDAÇÃO CRUZADA POR PACIENTE**

**3.1 O que já existe e será usado**

O benchmark P9 foi executado em modo definitivo, sobre 353 pacientes e 509 nódulos (258 benignos e 251 malignos), por validação cruzada aninhada 10 × 5 agrupada por paciente, com as mesmas dobras para os quatro algoritmos e 250 unidades de execução, sem falhas técnicas pendentes. A faixa de 12 a 30 mm (desfecho F) tem 226 nódulos, 187 pacientes, 135 malignos e 91 benignos. Esses valores coincidem com os do cartão. O desfecho geral é chamado de G.

Os intervalos foram calculados por bootstrap de pacientes (B = 10.000, semente 42), condicional aos modelos já ajustados. Os contrastes da família principal (RF, XGBoost e SVM contra a regressão logística com ajuste, em G e F, seis contrastes) usam correção de Bonferroni (percentis 0,41667 e 99,58333), e os demais contrastes são descritivos, com IC de 95% sem correção. Essa é a regra “conforme protocolo” citada no cartão (pré-registro P9, Seção 9.1; Emenda 1, E5).

**3.2 O que será feito**

Será montado um quadro comparativo único, com as AUC de cada modelo em G e F e seus intervalos descritivos, os seis contrastes da família com os dois intervalos (Bonferroni e 95% sem correção), os contrastes descritivos rotulados como tais, e a sensibilidade à partição (mínimo e máximo por repetição). O quadro será gerado por script a partir de contrastes.csv, bootstrap_resumo.json e denominadores.csv, e conferido linha a linha contra o resumo_benchmark.md. Não haverá novo ajuste de modelo, nova reamostragem nem novo contraste fora da lista do P9.

Quadro 1 — AUC por modelo e desfecho, já produzidas pelo benchmark (a ser reproduzido pelo script de consolidação).

| **Modelo**                     | **AUC em G (IC 95% descritivo)** | **AUC em F (IC 95% descritivo)** |
|--------------------------------|----------------------------------|----------------------------------|
| LR fixa (referência histórica) | 0,9442 \[0,9246; 0,9624\]        | 0,7903 \[0,7359; 0,8460\]        |
| LR com ajuste                  | 0,9449 \[0,9254; 0,9628\]        | 0,7907 \[0,7364; 0,8464\]        |
| SVM                            | 0,9424 \[0,9224; 0,9605\]        | 0,7890 \[0,7357; 0,8437\]        |
| Random Forest                  | 0,9476 \[0,9288; 0,9646\]        | 0,7968 \[0,7459; 0,8501\]        |
| XGBoost                        | 0,9438 \[0,9242; 0,9619\]        | 0,7844 \[0,7329; 0,8389\]        |

Fonte: reports/sprint6/benchmark/resumo_benchmark.md, Seção 4. A LR com ajuste é rotulada “otimista por seleção”.

Quadro 2 — Contrastes pareados da família principal.

| **Contraste contra a LR com ajuste** | **Desfecho** | **Diferença estimada** | **IC com Bonferroni** | **Repetições com Δ \> 0** |
|--------------------------------------|--------------|------------------------|-----------------------|---------------------------|
| RF                                   | G            | +0,0027                | \[−0,0043; 0,0098\]   | 10 de 10                  |
| RF                                   | F            | +0,0061                | \[−0,0250; 0,0368\]   | 8 de 10                   |
| XGBoost                              | G            | −0,0011                | \[−0,0075; 0,0050\]   | 6 de 10                   |
| XGBoost                              | F            | −0,0063                | \[−0,0358; 0,0221\]   | 3 de 10                   |
| SVM                                  | G            | −0,0025                | \[−0,0080; 0,0013\]   | 1 de 10                   |
| SVM                                  | F            | −0,0017                | \[−0,0213; 0,0129\]   | 4 de 10                   |

Fonte: reports/sprint6/benchmark/resumo_benchmark.md, Seções 5 e 6.

**3.3 Regras de leitura**

Os seis intervalos com Bonferroni contêm o zero, e nenhuma diferença estimada se aproxima de δ = 0,02 (critério C1 do pré-registro). Nenhum desafiante cumpriu C1 a C4, e a regra de seleção recomenda a regressão logística com ajuste. A leitura permitida é descritiva e condicional: “diferença estimada”, “compatível com”, “não detectada nesta base”. Ficam vetados “significativo”, “superior”, “comprovado”, “demonstrado”, “estatisticamente” e “p-valor”, conforme o P9, Seção 8.3. O bootstrap não reajusta os modelos nem captura a variabilidade da escolha de hiperparâmetros; os intervalos tendem a ser estreitos demais, e o Bonferroni é convenção de largura, não garantia de controle do erro de família, pois os contrastes compartilham o comparador. Intervalos largos não provam equivalência entre algoritmos.

**4 O QUE ESTE PROTOCOLO NÃO CONSOLIDA**

Os resultados da Etapa 3 da Sprint 4 (modelos M0 a M4 na validação oficial, com oito características fixas) e os do benchmark da Sprint 6 têm bases, partições e desenhos diferentes e não são misturados (Emenda 1, E3, item 5). O quadro do cartão cobre o benchmark. A Etapa 3 será citada em seção separada, se o grupo quiser, sem comparação numérica direta com o benchmark. O diâmetro isolado, conforme o cartão, fica em atividade separada.

**5 PARTE B — TABELA DESCRITIVA DO TESTE HISTÓRICO**

**5.1 Fato verificado que condiciona esta parte**

O benchmark da Sprint 6 nunca lê o teste: o carregador pula as linhas de teste antes de ler qualquer valor e aborta a execução se uma linha de teste aparecer (pré-registro P9, Seção 5 e invariantes). O cartão confirma que a avaliação do teste não foi realizada. O protocolo da Sprint 4 prevê a avaliação do teste na Etapa 5, mas, na busca feita no ramo main, não localizei módulo executável da Etapa 5 em src/incremental (há apenas referências a ela na configuração e na validação). Portanto, a tabela descritiva do teste exige código novo, que precisa ser escrito, revisado e versionado antes de qualquer leitura de teste.

**5.2 Pré-condições, todas obrigatórias antes de abrir o teste**

Quadro 3 — Pré-condições para a avaliação descritiva do teste.

| **\#** | **Pré-condição**                                                                                                      | **Fonte**                 | **Situação verificada**                                                                                                                                    |
|--------|-----------------------------------------------------------------------------------------------------------------------|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1      | Registros formais de P1, P2 e P3 nas guardas da S4                                                                    | Adendo 1, Seção 4         | Não verificado neste levantamento                                                                                                                          |
| 2      | Etapa 3 da S4 concluída e registrada                                                                                  | Protocolo 7.3; Emenda E7  | Relatório existe em reports/sprint4/etapa3_validacao; o benchmark em modo definitivo só roda com o registro S4_E3_REGISTRADA (P9), e rodou                 |
| 3      | Etapas 0 (item 2), 1 e 2 oficiais concluídas                                                                          | Adendo 1, Seção 3, item 2 | Declaradas concluídas pela responsável em 05/11/2026. Não localizei seus relatórios no main em 05/11/2026. \[A INFORMAR: caminho e commit dos relatórios\] |
| 4      | Etapa 4: congelamento em commit identificado (protocolo, código das Etapas 0 a 5, relatórios das Etapas 0 a 3)        | Protocolo 7.4             | Declarada concluída pela responsável em 05/11/2026. Não localizei o commit no repositório. \[A INFORMAR: hash do commit de congelamento\]                  |
| 5      | Código da Etapa 5 escrito, testado com dados sintéticos, revisado por outro integrante e incluído no commit congelado | Protocolo 7.4 e 7.5       | Não localizei módulo executável no main em 05/11/2026. \[A INFORMAR: caminho do código e revisor\]                                                         |
| 6      | Ambiente oficial definido (Python 3.9.13 ou 3.9.23, com critério de escolha da E4)                                    | Emenda E4                 | Parcial: a Emenda lista o que refazer no ambiente oficial                                                                                                  |

Fonte: elaborado pela autora (2026), com base nos documentos do repositório citados.

Se qualquer pré-condição não estiver atendida, o teste não é aberto e o cartão é entregue apenas com a Parte A e com o registro explícito de que a Parte B ficou condicionada. Esse é o resultado correto, e não uma falha do cartão: abrir o teste antes do congelamento invalida a regra “nenhuma escolha pós-teste”.

**5.3 Decisão que precisa ser tomada antes de escrever o código da Etapa 5**

O cartão pede o desempenho “dos modelos congelados” no split oficial, sem dizer quais. Há duas leituras, e a escolha precisa ser registrada pelo grupo, e não por quem executa:

Quadro 4 — Opções para o conteúdo da tabela descritiva do teste.

| **Opção**                       | **O que seria avaliado no teste**                                                                                 | **Consequência**                                                                                                                          |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| 1 (prevista no protocolo da S4) | M0 a M4 e as sensibilidades da Seção 7.5, ajustados no treino oficial (417 nódulos), com os contrastes da Seção 8 | Já pré-especificada e aprovada; não exige alterar o P9. Não avalia os quatro classificadores do benchmark                                 |
| 2                               | Os quatro classificadores do benchmark, reajustados nos 353 pacientes de treino e validação                       | O P9 declara que o teste nunca é lido pelo benchmark; exigiria adendo ao pré-registro e novo procedimento de ajuste final, que não existe |

Fonte: elaborado pela autora (2026), com base no protocolo da S4 (7.5) e no pré-registro P9 (Seções 5 e 8).

Decisão registrada: a responsável escolheu a opção 1 em 05/11/2026, sujeita ao aceite do revisor. Ela está em conformidade com o que já foi aprovado e não altera o P9. Valem as regras da Seção 5.4.

**5.4 Regras de execução e de relato do teste**

> — O teste é avaliado uma única vez. Reexecução só por falha técnica, documentada com a causa, e nenhum resultado dele altera modelo, característica, limiar, faixa, parâmetro ou critério de parada (Protocolo 7.5; Emenda E3, item 4).
>
> — Os modelos já estão congelados; não há reajuste com dados de teste nem de validação (Protocolo 7.5).
>
> — Toda tabela do teste leva a declaração: “conjunto já avaliado em análises anteriores do projeto; não constitui evidência independente” (Emenda E3, item 3).
>
> — Os números do teste não são misturados aos da validação cruzada nem aos do benchmark (Emenda E3, item 5). O quadro do teste fica separado, com título próprio.
>
> — Discordância de sinal entre validação e teste é declarada como limitação e não altera a leitura da Etapa 3 (Protocolo, Seção 9).
>
> — O hash do commit congelado é gravado no artefato da Etapa 5 (Protocolo 7.4).

**6 PARTE C — REGISTROS EXIGIDOS PELO CARTÃO**

O cartão exige registrar população, denominadores, versões, seeds e as diferenças metodológicas entre a validação cruzada e o teste. O quadro abaixo é o molde; as células do teste só serão preenchidas depois da execução da Etapa 5.

Quadro 5 — Registro comparativo entre validação cruzada e teste.

| **Item**             | **Validação cruzada (benchmark P9)**                                                            | **Teste histórico (Etapa 5)**                              |
|----------------------|-------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| População            | 353 pacientes, 509 nódulos; faixa F: 187 pacientes, 226 nódulos                                 | 98 nódulos, 63 pacientes (Protocolo, Seção 11, item 12)    |
| Partição             | Treino e validação; dobras agrupadas por paciente, 10 repetições × 5 dobras                     | Split oficial; ajuste no treino oficial                    |
| Estimador            | Média das AUC por dobra e repetição; IC por bootstrap de pacientes condicional                  | AUC única no teste; IC conforme Seção 8 do protocolo da S4 |
| Seeds e versões      | Semente 42; Python 3.9.13, scikit-learn 1.3.2, xgboost 2.1.4; SHA-256 da configuração a0d096a5… | A preencher: commit congelado, ambiente oficial, sementes  |
| Uso anterior do dado | Validação reutilizada depois da Etapa 3 da S4 (limitação declarada)                             | Usado em ao menos seis análises de tipo A (Emenda E1)      |
| Leitura permitida    | Descritiva e condicional                                                                        | Apenas descritiva, sem leitura confirmatória               |

Fonte: elaborado pela autora (2026), com base em resumo_benchmark.md, protocolo da S4 e Emenda 1.

**7 SUSPEIÇÃO RADIOLÓGICA E MALIGNIDADE CONFIRMADA**

O cartão exige documentar essa diferença. O desfecho do projeto é derivado das avaliações radiológicas do LIDC-IDRI e expressa suspeição de malignidade, e não diagnóstico histopatológico; o P9 já declara, entre as limitações, que o alvo é radiológico e que os resultados não se generalizam para a prática clínica. O texto final do cartão deve repetir esse ponto e remeter ao protocolo da coorte (docs/protocolo_coorte_target_sprint2.md) para a definição operacional do alvo. Não consegui, neste levantamento, ler a definição operacional completa do alvo nesse arquivo; ela deve ser conferida antes de ser citada. Não se atribui ao alvo nenhuma propriedade além da que o protocolo da coorte declara.

**8 CRITÉRIOS DE ACEITE (DoD DO CARTÃO)**

Quadro 6 — Correspondência entre o DoD e este protocolo.

| **Critério do DoD**                    | **Como será atendido**                                                                                                        | **Evidência**                         |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| Protocolo fixado antes da execução     | Este documento, revisado, registrado em commit antes de qualquer script                                                       | Commit e comentário de aprovação      |
| Comparação pareada correta             | Quadros 1 e 2 reproduzidos por script a partir dos CSV do benchmark, sem nova estimação                                       | Script e log com SHA-256 das entradas |
| Relato explícito do histórico do teste | Seções 2 e 5.4 e Quadro 5, com a declaração obrigatória                                                                       | Texto do relatório                    |
| Nenhuma escolha pós-teste              | Pré-condições do Quadro 3; teste aberto uma vez, após congelamento                                                            | Hash do commit congelado no artefato  |
| Revisão de outro integrante            | Revisor registrado (Rafael); revisão em paralelo à Parte A; aprovação obrigatória antes da Parte B e do fechamento definitivo | Comentário no PR ou cartão            |
| Evidências no GitHub e no Drive        | Script, logs e relatório versionados; DoD no Drive                                                                            | Links no cartão                       |

Fonte: elaborado pela autora (2026), com base no cartão P0 da Sprint 7.

**9 CRONOGRAMA E RISCOS**

Parte A: protocolo registrado em 06/11 (abertura do PR); script, quadro e relatório da Parte A em 06/11, em paralelo à revisão. Parte B: só começa depois que as pré-condições do Quadro 3 estiverem comprovadas por caminho e commit no repositório. A responsável declarou as Etapas 1, 2 e 4 concluídas em 05/11/2026; como o main não mostra essas evidências, os campos \[A INFORMAR\] do Quadro 3 precisam ser preenchidos antes do registro. Sem eles, a Parte B não é executada e o cartão sai só com a Parte A.

Riscos de questionamento: (a) abrir o teste antes do congelamento; (b) tabela do teste lida como validação independente; (c) misturar números da Etapa 3, do benchmark e do teste; (d) usar linguagem inferencial vetada pelo P9; (e) reportar a LR com ajuste sem o rótulo de otimismo por seleção; (f) o código da Etapa 5 não ter sido revisado por outro integrante.

**10 LIMITAÇÕES DESTE LEVANTAMENTO**

Este protocolo foi redigido a partir da leitura dos documentos e dos resultados versionados no ramo main (commit 1530be5), reconsultado em 05/11/2026 sem novidades. Não executei scripts do repositório nem a suíte de testes, não li o código da Etapa 3 além da busca por referências à Etapa 5, e não verifiquei se os registros formais de P1 a P3 estão no repositório. O estado de Trello e Drive não foi consultado. Cada uma dessas verificações deve ser feita antes do registro do protocolo.

**REFERÊNCIAS**

PI3-GRUPO-4. Emenda 1 ao protocolo de valor incremental: auditoria do teste e congelamento (revisão 3). Repositório RafaellLobo/PI3-GRUPO-4, docs/sprint4/emenda_01_auditoria_teste.md, commit 573bc17, 2026.

PI3-GRUPO-4. Protocolo de valor incremental da radiômica. Repositório RafaellLobo/PI3-GRUPO-4, docs/sprint4/protocolo_valor_incremental.md, 2026.

PI3-GRUPO-4. Pré-registro P9 do benchmark de classificadores. Repositório RafaellLobo/PI3-GRUPO-4, docs/sprint6/preregistro_p9_benchmark.md, 2026.

PI3-GRUPO-4. Resumo do benchmark de classificadores da Sprint 6. Repositório RafaellLobo/PI3-GRUPO-4, reports/sprint6/benchmark/resumo_benchmark.md, commit 1530be5, 2026.

PI3-GRUPO-4. Adendo 1 ao protocolo de valor incremental: ordem das Etapas 1 a 3. Repositório RafaellLobo/PI3-GRUPO-4, docs/sprint4/adendo_01_ordem_das_etapas.md, 2026.

*“Todos os modelos estão errados, mas alguns são úteis.” (George E. P. Box)*
