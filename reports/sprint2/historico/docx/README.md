# Documentos históricos — não usar como referência metodológica

Os `.docx` neste diretório registram **estados anteriores** às correções dos
Bloqueadores 1 e 2 da Sprint 2. Contêm números de target e interpretações da
comparação de discretização que foram **superados**:

- **Bloqueador 1** — a regra de consolidação do alvo a partir da mediana de
  `malignancy` foi corrigida; as contagens de alvo nestes documentos precedem
  essa correção.
- **Bloqueador 2** — a comparação `binCount = 64` × `binWidth = 25` HU registrada
  aqui é a versão **inválida**, contaminada por diferença de geometria de recorte
  entre os dois lados.

São preservados apenas por **valor de rastreabilidade**, para reconstruir o
histórico das decisões do grupo.

> **Fonte canônica atual:** [`docs/sprint2/piloto_radiomico.md`](../../../../docs/sprint2/piloto_radiomico.md)
