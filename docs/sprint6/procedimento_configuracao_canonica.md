# Procedimento — configuração canônica do benchmark S6 (P9)

**Objeto:** como a configuração canônica do benchmark é gerada pelo código, versionada no PR #11 e ligada ao registro de aprovação P9.
**Base:** pré-registro P9, Seções 14.4, 15 e 19 (`docs/sprint6/preregistro_p9_benchmark.md`, PR #11, commit `31931d0`).
**Situação:** procedimento proposto. Nenhuma aprovação foi registrada.

---

## 1. O que é a configuração canônica

- **Fonte:** `src/benchmark_s6/config.py` (`CONFIGURACAO`), com todos os parâmetros pré-registrados, validados na construção.
- **Arquivo:** `docs/sprint6/config_benchmark_p9.json`, que é o JSON de `rastreabilidade.json_canonico` (chaves ordenadas, UTF-8, sem espaços, sem `NaN`) seguido de `\n`.
- **SHA-256 da configuração:** `guardas.hash_valor_decisao` do conteúdo, igual a `rastreabilidade.sha256_canonico`. É o `valor_sha256` do registro P9.
  - **Não** é o SHA-256 dos bytes brutos do arquivo, que têm o `\n` final e podem ter CRLF no checkout do Windows.
  - Valores derivados (grade de `C` materializada, percentis de Bonferroni, número de ajustes) não entram no JSON; só as especificações que os geram.
- **Valor atual, gerado pelo código desta branch:** `1cd08306786067c1e29229a93a3215e087b0db21da35bc05c595bd5c8a2117d5`.

O valor muda se mudar qualquer um destes itens:
- um parâmetro;
- a versão do pré-registro (`versao_preregistro`, hoje `1.0`);
- os SHA-256 fixados dos arquivos da S4 (`adaptadores_s4.ARQUIVOS_S4`);
- os pins das entradas.

## 2. Comandos

A partir da raiz do repositório, com a árvore limpa:

```
python -m src.benchmark_s6 --modo configuracao
python -m src.benchmark_s6 --modo configuracao --gerar ARQUIVO
python -m src.benchmark_s6 --modo configuracao --conferir ARQUIVO
```

- O primeiro só imprime o SHA-256.
- `--gerar` grava o arquivo se ele não existir. Se existir, só confere, e **nunca sobrescreve**.
- `--conferir` exige que o arquivo esteja na forma canônica e tenha o mesmo SHA-256 da configuração do código.
- Os três comandos verificam antes os SHA-256 dos arquivos da S4. Código de saída 4 indica divergência.
- Erro de argumentos (modo inexistente ou ainda não implementado, `--gerar` com `--conferir`) sai com o código 64, e não com o 2 do `argparse`: a P9 (Seção 14.3) reserva o 2 ao bloqueio de aprovação.

O modo `configuracao` não lê dados, não ajusta modelos e não consulta aprovações.

## 3. Incorporação ao PR #11, antes da aprovação P9

A ordem segue a Seção 14.4 da P9.

1. **Validar e commitar** a branch de implementação (`feat/s6-benchmark-configuracao`). Chame de `C_cfg` o commit resultante.
2. **Gerar** o arquivo nesse commit, com a árvore limpa, fora do repositório:
   `python -m src.benchmark_s6 --modo configuracao --gerar <dir_temporario>/config_benchmark_p9.json`
3. **Versionar no PR #11:** na branch `docs/s6-p9-preregistro-benchmark`, copiar o arquivo para `docs/sprint6/config_benchmark_p9.json`, sem editar, e anotar na Seção 19 do pré-registro:
   - o caminho do arquivo;
   - o SHA-256 da configuração;
   - o commit `C_cfg` do código que o gerou.

   Commitar só esses dois arquivos.
4. **Conferir:** a partir de `C_cfg`, extrair o arquivo do PR #11 (`git show docs/s6-p9-preregistro-benchmark:docs/sprint6/config_benchmark_p9.json`) para um arquivo temporário e rodar `--conferir` sobre ele. Esperado: código 0 e o mesmo SHA-256.
5. **Só depois de:**
   - a confirmação documental de D2 pela responsável;
   - a ratificação de D1 a D12;
   - a aprovação do PR #10,

   o registro `APROVACAO-DECISAO codigo=P9 … valor_sha256=<SHA-256 da configuração>` entra na Seção 19, em coluna 0, num commit **posterior** ao do passo 3. Este procedimento não cria esse registro.

## 4. Mudanças

- **Antes da aprovação:** qualquer mudança de parâmetro, inclusive uma redação de D2 que altere a variável da faixa ou uma nova versão do pré-registro, exige atualizar `config.py` e o teste que fixa o SHA-256 e repetir os passos 1 a 4. O arquivo antigo é substituído por um commit explícito no PR #11, nunca pelo `--gerar`.
- **Depois da aprovação:** só por adendo aprovado (P9, Seção 15), com nova configuração, novo SHA-256 e novo registro.
- **Arquivos da S4 alterados:** a verificação falha (código 4). Atualizar `ARQUIVOS_S4` exige revisão humana do impacto e conta como mudança de configuração.
