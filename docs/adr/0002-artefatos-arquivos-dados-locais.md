# 0002 - Armazenar artefatos em arquivos e separar os dados individuais

Status: Aceito

Data: 2026-11-23

Registro retrospectivo do armazenamento já adotado. Não aprova migração, remoção de históricos ou nova divulgação de dados.

## Contexto

A equipe precisa relacionar código, configuração, partições e resultados experimentais. Um clone deve permitir inspecionar a documentação e os resultados agregados, mas não deve carregar novas tabelas individuais, DICOMs ou predições por paciente. Também é necessário preservar resultados históricos e conferir integridade das execuções.

## Decisão

Manter tabelas agregadas em CSV, configurações/relatórios/manifestos em JSON e figuras nos formatos previstos no protocolo. Versionar código, documentação e artefatos agregados permitidos em Git; manter entradas em `data/` e checkpoints/predições individuais em armazenamento local autorizado, fora do Git. Usar hashes SHA-256 e manifestos para rastreabilidade e recusar sobrescrita de destinos congelados no coordenador.

Não introduzir servidor PostgreSQL ou outro banco de aplicação: esse componente não existe na arquitetura atual. O uso de SQLAlchemy pelo pylidc para anotações é uma dependência distinta do armazenamento dos resultados. Preservar as exceções históricas de arquivos individuais descritas na emenda E10/E11; este ADR não autoriza excluir esses arquivos nem generalizar a exceção.

## Consequências

**Benefícios:** artefatos legíveis e comparáveis em revisão, sem operação de servidor; separação entre evidência pública/agregada e dados locais; hashes ajudam a detectar alterações.

**Custos e limites:** a equipe precisa obter e organizar entradas externas; o clone não reproduz sozinho toda a cadeia; arquivos não fornecem transações e consultas de um banco de aplicação. Exceções históricas exigem contexto e não desaparecem por retirar arquivos de uma revisão atual. `.gitignore` auxilia, mas não substitui a varredura nem a conferência antes de publicar.

**Rastreabilidade:** [dados](../../data/README.md), [.gitignore](../../.gitignore), [privacidade](../../src/benchmark_s6/privacidade.py), [manifesto](../../src/benchmark_s6/manifesto.py), [emenda](../sprint4/emenda_01_auditoria_teste.md), [RF-08 e RNF-02/RNF-03/RNF-04](../requisitos.md).
