"""Sprint 4 — verificação de treinamento dos modelos de referência M0 e M1.

AVISO SOBRE DADOS MÉDICOS
-------------------------
Lê ``data/modeling_table_sprint3.csv`` (dado derivado, não versionado),
``reports/sprint3/baseline/patient_split.csv`` e
``reports/sprint3/baseline/baseline_metrics.json``. Grava somente um JSON
agregado em ``reports/sprint4/reference_models/`` — sem identificadores de
paciente ou de nódulo e sem valores de feature por nódulo.

ESCOPO
------
Etapa 0 do protocolo ``docs/sprint4/protocolo_valor_incremental.md``
(Seções 6.1, 6.2 e 7.1, itens 3 e 4):

* ajusta M0 (diâmetro) e M1 (baseline oficial) no treino oficial, com o mesmo
  pipeline e os mesmos hiperparâmetros da Sprint 3;
* compara os parâmetros ajustados de M1 aos de ``baseline_metrics.json``
  (coeficientes, intercepto, média e escala do ``StandardScaler``), com
  tolerâncias explícitas;
* NÃO calcula nenhuma métrica de desempenho e NÃO ajusta M2, M3 ou M4.

ISOLAMENTO
----------
A tabela de modelagem é lida em duas passagens:

1. somente as colunas de identificação (``nodulo_id``, ``patient_id``) de
   todas as linhas, para as contagens estruturais;
2. target e features com ``skiprows``, que descarta no parser toda linha que
   não seja de treino. Os bytes dessas linhas passam pelo leitor de arquivo,
   mas seus campos não são convertidos nem chegam a nenhum DataFrame.

Por isso, a composição de CLASSES é conferida apenas no treino; a de validação
e teste fica para as Etapas 3 e 5, que são as únicas autorizadas a ler essas
linhas.

As convenções (features, hiperparâmetros, caminhos e pipeline) são importadas
de ``scripts/run_baseline.py`` — fonte única do baseline. Importar o módulo não
executa ``main()`` e, portanto, não reescreve os relatórios da Sprint 3.

Execução:
    python scripts/sprint4/validate_reference_models.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
for _path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import sklearn  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import run_baseline as s3  # noqa: E402 - convenções do baseline oficial
from src.radiomics.ids import decompor_id_extracao  # noqa: E402


# --------------------------------------------------------------------------- #
# Contrato
# --------------------------------------------------------------------------- #

BASELINE_JSON: Path = s3.OUTPUT_DIR / "baseline_metrics.json"
OUTPUT_DIR: Path = REPO_ROOT / "reports" / "sprint4" / "reference_models"
OUTPUT_JSON: Path = OUTPUT_DIR / "validacao_modelos_referencia.json"

DIAMETER_FEATURE: str = "original_shape_Maximum3DDiameter"

#: Listas explícitas por nome (protocolo, Seção 5.5) — nunca derivadas por prefixo.
MODELOS: dict[str, tuple[str, ...]] = {
    "M0": (DIAMETER_FEATURE,),
    "M1": s3.BASELINE_FEATURES,
}

#: Composição esperada (protocolo, Seção 4). Classes só para o treino (ver ISOLAMENTO).
EXPECTED_PARTITIONS: dict[str, dict[str, int]] = {
    s3.SPLIT_TRAIN: {"pacientes": 291, "nodulos": 417},
    s3.SPLIT_VALIDATION: {"pacientes": 62, "nodulos": 92},
    s3.SPLIT_TEST: {"pacientes": 63, "nodulos": 98},
}
EXPECTED_TRAIN_CLASSES: dict[int, int] = {0: 209, 1: 208}

#: Tolerâncias da reprodução do M1, aplicadas como ``np.isclose(obtido, ref, rtol, atol)``.
#:
#: - StandardScaler: média e desvio são aritmética fechada sobre os mesmos 417
#:   valores; só cabe diferença de arredondamento de ponto flutuante.
#: - Coeficientes e intercepto: saem de uma otimização iterativa (lbfgs,
#:   ``tol=1e-4``). No mesmo ambiente o resultado deve ser idêntico; a folga
#:   cobre variações de BLAS/plataforma sem aceitar um ajuste diferente — uma
#:   mudança de features, linhas ou hiperparâmetros produz diferenças de ordem
#:   1e-2 ou maiores.
TOLERANCIAS: dict[str, dict[str, float]] = {
    "scaler_mean": {"rtol": 1e-10, "atol": 1e-12},
    "scaler_scale": {"rtol": 1e-10, "atol": 1e-12},
    "coeficientes": {"rtol": 1e-6, "atol": 1e-8},
    "intercepto": {"rtol": 1e-6, "atol": 1e-8},
}

SEPARATOR: str = s3.SEPARATOR


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #


def sha256(caminho: Path) -> str:
    """Hash SHA-256 do arquivo, para rastrear exatamente o que foi lido."""
    h = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def relativo(caminho: Path) -> str:
    return str(caminho.relative_to(REPO_ROOT)).replace("\\", "/")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )


def estado_git() -> dict[str, Any]:
    """Estado do repositório no momento da execução (protocolo, Seção 12).

    Somente leitura. ``status_porcelain`` lista caminhos de arquivos do
    repositório — nunca dados. Se o git não estiver disponível, o fato é
    registrado e a execução continua.
    """
    try:
        head = _git("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"disponivel": False, "erro": head.stderr.strip()}
        branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        status = [linha for linha in _git("status", "--porcelain").stdout.splitlines() if linha]
        script = relativo(Path(__file__).resolve())
        versionado = _git("ls-files", "--error-unmatch", "--", script).returncode == 0
        # None quando o script não está no HEAD: "sem alterações" não se aplica.
        alterado = (
            _git("diff", "--quiet", "HEAD", "--", script).returncode != 0 if versionado else None
        )
    except OSError as exc:
        return {"disponivel": False, "erro": str(exc)}
    return {
        "disponivel": True,
        "commit_head": head.stdout.strip(),
        "branch": branch,
        "arvore_limpa": not status,
        "status_porcelain": status,
        "script": script,
        "script_versionado_no_head": versionado,
        "script_com_alteracoes_nao_commitadas": alterado,
        "observacao": (
            "o resultado só é rastreável a um commit quando o script está versionado e "
            "sem alterações; caso contrário, commit_head indica apenas a base da execução"
        ),
    }


class Checagens:
    """Acumula invariantes; a primeira violação interrompe a execução."""

    def __init__(self) -> None:
        self.registro: list[dict[str, Any]] = []

    def ok(self, rotulo: str, obtido: Any, esperado: Any) -> None:
        s3._exigir(
            obtido == esperado,
            f"INVARIANTE VIOLADA — {rotulo}: esperado {esperado}, obtido {obtido}. "
            "Nada foi ajustado nem escrito.",
        )
        self.registro.append({"checagem": rotulo, "valor": obtido})
        print(f"  [OK] {rotulo:<62} {obtido}")


# --------------------------------------------------------------------------- #
# Carga isolada
# --------------------------------------------------------------------------- #


def carregar_treino(chk: Checagens, leitura: dict[str, Any]) -> pd.DataFrame:
    """Valida a estrutura e devolve somente as linhas de treino.

    Target e features de validação/teste não são convertidos pelo parser nem
    chegam a nenhum DataFrame (ver ISOLAMENTO). ``leitura`` recebe o registro
    da verificação de alinhamento da leitura isolada.
    """
    for caminho in (s3.MODELING_CSV, s3.SPLIT_CSV, BASELINE_JSON):
        if not caminho.exists():
            raise FileNotFoundError(f"{caminho} não encontrado.")

    split = pd.read_csv(s3.SPLIT_CSV)
    chk.ok("split: colunas", sorted(split.columns), sorted([s3.PATIENT_COLUMN, "split"]))
    chk.ok("split: linhas == pacientes únicos", len(split), int(split[s3.PATIENT_COLUMN].nunique()))
    chk.ok("split: pacientes", len(split), 416)
    chk.ok("split: partições", sorted(split["split"].unique()), sorted(s3.SPLIT_ORDER))

    # Identificação de todas as linhas — sem target, sem features.
    ids = pd.read_csv(s3.MODELING_CSV, usecols=[s3.NODULE_COLUMN, s3.PATIENT_COLUMN])
    chk.ok("tabela: nódulos", len(ids), s3.EXPECTED_NODULES)
    chk.ok("tabela: pacientes únicos", int(ids[s3.PATIENT_COLUMN].nunique()), s3.EXPECTED_PATIENTS)
    chk.ok("tabela: nodulo_id duplicados", int(ids[s3.NODULE_COLUMN].duplicated().sum()), 0)
    incoerentes = sum(
        decompor_id_extracao(n)[0] != p
        for n, p in zip(ids[s3.NODULE_COLUMN], ids[s3.PATIENT_COLUMN])
    )
    chk.ok("tabela: nodulo_id incoerentes com patient_id", incoerentes, 0)

    # Ordem física das linhas de dados, preservada antes do merge.
    ids_arquivo = ids.copy()
    ids = ids.merge(split, on=s3.PATIENT_COLUMN, how="left", validate="many_to_one")
    chk.ok("tabela: nódulos sem partição", int(ids["split"].isna().sum()), 0)
    chk.ok(
        "split: pacientes sem nódulo na tabela",
        len(set(split[s3.PATIENT_COLUMN]) - set(ids[s3.PATIENT_COLUMN])),
        0,
    )

    por_particao = {
        nome: {
            "pacientes": int(ids.loc[ids["split"] == nome, s3.PATIENT_COLUMN].nunique()),
            "nodulos": int((ids["split"] == nome).sum()),
        }
        for nome in s3.SPLIT_ORDER
    }
    for nome in s3.SPLIT_ORDER:
        chk.ok(f"composição {nome} (pacientes, nódulos)", por_particao[nome], EXPECTED_PARTITIONS[nome])

    pacientes = {
        nome: set(split.loc[split["split"] == nome, s3.PATIENT_COLUMN]) for nome in s3.SPLIT_ORDER
    }
    compartilhados = sum(
        len(pacientes[a] & pacientes[b])
        for a, b in (
            (s3.SPLIT_TRAIN, s3.SPLIT_VALIDATION),
            (s3.SPLIT_TRAIN, s3.SPLIT_TEST),
            (s3.SPLIT_VALIDATION, s3.SPLIT_TEST),
        )
    )
    chk.ok("pacientes compartilhados entre partições", compartilhados, 0)

    # Somente treino: target e features. ``skiprows`` recebe o número da linha
    # FÍSICA do arquivo: a linha 0 é o cabeçalho (mantida) e a linha k >= 1 é o
    # registro de dados de posição k - 1 na leitura de identificadores acima.
    # A correspondência pressupõe um registro por linha física; ela é conferida
    # logo abaixo pelos nodulo_id efetivamente lidos.
    colunas = [s3.NODULE_COLUMN, s3.PATIENT_COLUMN, s3.TARGET_COLUMN, *s3.BASELINE_FEATURES]
    pacientes_treino = pacientes[s3.SPLIT_TRAIN]
    mascara_treino = ids_arquivo[s3.PATIENT_COLUMN].isin(pacientes_treino).to_numpy()
    posicoes_treino = frozenset(np.flatnonzero(mascara_treino).tolist())
    esperados = ids_arquivo.loc[mascara_treino, s3.NODULE_COLUMN].tolist()

    treino = pd.read_csv(
        s3.MODELING_CSV,
        usecols=colunas,
        skiprows=lambda linha: linha != 0 and (linha - 1) not in posicoes_treino,
    )
    s3._exigir(
        treino[s3.NODULE_COLUMN].tolist() == esperados,
        "INVARIANTE VIOLADA — leitura isolada desalinhada: os nodulo_id lidos com "
        "skiprows não coincidem, em conteúdo e ordem, com as linhas de treino "
        "esperadas. Nada foi ajustado nem escrito.",
    )
    leitura.update({
        "metodo": "pandas.read_csv(skiprows=callable) sobre a linha física do arquivo",
        "linhas_de_dados_no_arquivo": int(len(ids_arquivo)),
        "linhas_mantidas_pelo_parser": int(len(treino)),
        "linhas_descartadas_pelo_parser": int(len(ids_arquivo) - len(treino)),
        "nodulo_id_lidos_coincidem_em_conteudo_e_ordem": True,
        "colunas_lidas": colunas,
    })

    # Mesma ordenação do run_baseline.carregar (nodulo_id, mergesort), para que
    # o ajuste veja as linhas na mesma ordem.
    treino = treino.sort_values(s3.NODULE_COLUMN, kind="mergesort").reset_index(drop=True)

    chk.ok("treino: nódulos lidos com target/features", len(treino), 417)
    chk.ok("treino: pacientes", int(treino[s3.PATIENT_COLUMN].nunique()), 291)
    chk.ok(
        "treino: todos os pacientes pertencem ao treino oficial",
        bool(set(treino[s3.PATIENT_COLUMN]) <= pacientes_treino),
        True,
    )
    classes = {int(k): int(v) for k, v in treino[s3.TARGET_COLUMN].value_counts().sort_index().items()}
    chk.ok("treino: classes {0, 1}", classes, EXPECTED_TRAIN_CLASSES)

    for nome, features in MODELOS.items():
        chk.ok(f"{nome}: original_nodule_idx presente entre as features",
               "original_nodule_idx" in features, False)
    for coluna in s3.BASELINE_FEATURES:
        valores = pd.to_numeric(treino[coluna], errors="coerce").to_numpy(dtype=float)
        chk.ok(f"treino: valores não finitos em {coluna}", int((~np.isfinite(valores)).sum()), 0)

    return treino


# --------------------------------------------------------------------------- #
# Ajuste
# --------------------------------------------------------------------------- #


def ajustar(nome: str, treino: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    """Ajusta o pipeline do baseline no treino; nenhuma predição é feita."""
    features = list(MODELOS[nome])
    X = treino[features].to_numpy(dtype=float)
    y = treino[s3.TARGET_COLUMN].to_numpy(dtype=int)

    modelo = s3.construir_pipeline()
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        modelo.fit(X, y)
    return modelo, [f"{w.category.__name__}: {w.message}" for w in capturados]


def parametros(nome: str, modelo: Pipeline, avisos: list[str], n_treino: int) -> dict[str, Any]:
    """Parâmetros ajustados — somente quantidades aprendidas no treino."""
    features = MODELOS[nome]
    scaler = modelo.named_steps["scaler"]
    logreg = modelo.named_steps["logreg"]
    n_vistos = int(np.atleast_1d(scaler.n_samples_seen_)[0])
    n_iter = int(np.atleast_1d(logreg.n_iter_)[0])
    max_iter = int(logreg.get_params()["max_iter"])

    s3._exigir(n_vistos == n_treino, f"{nome}: scaler viu {n_vistos} amostras, treino tem {n_treino}.")
    return {
        "features": list(features),
        "parametros_logreg": s3.LOGREG_PARAMS,
        "solver": logreg.get_params()["solver"],
        "coeficientes": {f: float(c) for f, c in zip(features, logreg.coef_[0])},
        "intercepto": float(logreg.intercept_[0]),
        "scaler_mean": {f: float(v) for f, v in zip(features, scaler.mean_)},
        "scaler_scale": {f: float(v) for f, v in zip(features, scaler.scale_)},
        "scaler_n_samples_seen": n_vistos,
        "n_iteracoes": n_iter,
        "convergiu": n_iter < max_iter and not any("ConvergenceWarning" in a for a in avisos),
        "warnings_sklearn": avisos,
    }


# --------------------------------------------------------------------------- #
# Reprodução do M1
# --------------------------------------------------------------------------- #


def comparar_m1(m1: dict[str, Any], referencia: dict[str, Any]) -> dict[str, Any]:
    """Compara os parâmetros do M1 aos de baseline_metrics.json, termo a termo."""
    ref = referencia["coeficientes"]
    pares = {
        "coeficientes": (m1["coeficientes"], ref["por_feature"]),
        "intercepto": ({"intercepto": m1["intercepto"]}, {"intercepto": ref["intercepto"]}),
        "scaler_mean": (m1["scaler_mean"], ref["scaler_mean"]),
        "scaler_scale": (m1["scaler_scale"], ref["scaler_scale"]),
    }

    grupos: dict[str, Any] = {}
    for grupo, (obtido, esperado) in pares.items():
        s3._exigir(set(obtido) == set(esperado), f"M1 {grupo}: chaves divergem de baseline_metrics.json.")
        tol = TOLERANCIAS[grupo]
        termos = {}
        for chave in esperado:
            o, e = float(obtido[chave]), float(esperado[chave])
            termos[chave] = {
                "obtido": o,
                "referencia": e,
                "dif_abs": abs(o - e),
                "dif_rel": abs(o - e) / abs(e) if e != 0 else None,
                "identico_bit_a_bit": o == e,
                "dentro_da_tolerancia": bool(np.isclose(o, e, rtol=tol["rtol"], atol=tol["atol"])),
            }
        grupos[grupo] = {
            "tolerancia": tol,
            "termos": termos,
            "todos_dentro_da_tolerancia": all(t["dentro_da_tolerancia"] for t in termos.values()),
            "todos_identicos_bit_a_bit": all(t["identico_bit_a_bit"] for t in termos.values()),
            "max_dif_abs": max(t["dif_abs"] for t in termos.values()),
        }

    contrato = {
        "features": (m1["features"], referencia["features"]),
        "parametros_logreg": (m1["parametros_logreg"], referencia["parametros_logreg"]),
        "versao_sklearn": (sklearn.__version__, referencia["versao_sklearn"]),
        "n_iteracoes": (m1["n_iteracoes"], ref["n_iteracoes"]),
        "scaler_n_samples_seen": (
            m1["scaler_n_samples_seen"],
            referencia["prevencao_vazamento"]["scaler_n_samples_seen"],
        ),
    }
    igualdades = {k: {"obtido": o, "referencia": e, "igual": o == e} for k, (o, e) in contrato.items()}

    reproduzido = all(g["todos_dentro_da_tolerancia"] for g in grupos.values()) and all(
        v["igual"] for v in igualdades.values()
    )
    return {
        "reproduzido": reproduzido,
        "identico_bit_a_bit": all(g["todos_identicos_bit_a_bit"] for g in grupos.values()),
        "parametros_numericos": grupos,
        "contrato": igualdades,
    }


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #


def main() -> int:
    """Valida a estrutura, ajusta M0/M1 no treino e confere a reprodução do M1."""
    s3._exigir(
        s3.OUTPUT_DIR not in OUTPUT_JSON.parents,
        "Saída apontando para reports/sprint3 — proibido.",
    )
    hashes_antes = {relativo(p): sha256(p) for p in (s3.MODELING_CSV, s3.SPLIT_CSV, BASELINE_JSON)}
    git = estado_git()  # antes de qualquer escrita

    print(f"{SEPARATOR}\nINVARIANTES ESTRUTURAIS (antes de qualquer ajuste)\n{SEPARATOR}")
    chk = Checagens()
    leitura: dict[str, Any] = {}
    try:
        treino = carregar_treino(chk, leitura)
    except (AssertionError, FileNotFoundError, ValueError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    referencia = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))

    print(f"\n{SEPARATOR}\nAJUSTE NO TREINO OFICIAL (sem predição, sem métrica)\n{SEPARATOR}")
    modelos: dict[str, dict[str, Any]] = {}
    for nome in MODELOS:
        modelo, avisos = ajustar(nome, treino)
        modelos[nome] = parametros(nome, modelo, avisos, len(treino))
        p = modelos[nome]
        print(f"\n  {nome}  features={len(p['features'])}  n_iter={p['n_iteracoes']}  "
              f"convergiu={p['convergiu']}  warnings={len(p['warnings_sklearn'])}")
        for f, c in p["coeficientes"].items():
            print(f"    coef {f:<40} {c:>+.12f}")
        print(f"    intercepto {'':<36} {p['intercepto']:>+.12f}")

    # Coerência interna: o scaler de M0 vê a mesma coluna de diâmetro que o de M1.
    coerencia = {
        grupo: modelos["M0"][grupo][DIAMETER_FEATURE] == modelos["M1"][grupo][DIAMETER_FEATURE]
        for grupo in ("scaler_mean", "scaler_scale")
    }

    reproducao = comparar_m1(modelos["M1"], referencia)

    print(f"\n{SEPARATOR}\nREPRODUÇÃO DO M1 vs baseline_metrics.json\n{SEPARATOR}")
    for grupo, g in reproducao["parametros_numericos"].items():
        print(f"  {grupo:<14} rtol={g['tolerancia']['rtol']:<7g} atol={g['tolerancia']['atol']:<7g} "
              f"max|dif|={g['max_dif_abs']:.3e}  dentro={g['todos_dentro_da_tolerancia']}  "
              f"bit-a-bit={g['todos_identicos_bit_a_bit']}")
    for chave, v in reproducao["contrato"].items():
        print(f"  {chave:<22} igual={v['igual']}  ({v['obtido']} vs {v['referencia']})")
    print(f"  M0/M1 scaler do diâmetro idêntico: {coerencia}")

    hashes_depois = {relativo(p): sha256(p) for p in (s3.MODELING_CSV, s3.SPLIT_CSV, BASELINE_JSON)}
    entradas_intactas = hashes_antes == hashes_depois

    status = "reproduzido" if (reproducao["reproduzido"] and all(coerencia.values())
                               and entradas_intactas) else "divergente"
    resultado: dict[str, Any] = {
        "etapa": "Sprint 4 — Cartão 4 — Etapa 0: verificação dos modelos de referência M0 e M1",
        "protocolo": "docs/sprint4/protocolo_valor_incremental.md (Seções 6.1, 6.2, 7.1)",
        "status": status,
        "escopo": {
            "metricas_de_desempenho_calculadas": False,
            "predicoes_realizadas": False,
            "modelos_ajustados": list(MODELOS),
            "linhas_usadas_em_ajuste": "somente treino oficial",
            "validacao_teste_lidos": (
                "somente nodulo_id e patient_id, para contagens estruturais; na leitura de "
                "target e features, essas linhas são descartadas pelo parser (skiprows) e "
                "seus campos não são convertidos"
            ),
            "composicao_de_classes_validacao_teste": "não verificada nesta etapa (isolamento, Seção 7.1)",
        },
        "git": git,
        "leitura_isolada": leitura,
        "ambiente": {
            "python": sys.version.split()[0],
            "plataforma": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "seed": s3.RANDOM_STATE,
        "sha256_entradas": hashes_antes,
        "entradas_intactas_apos_execucao": entradas_intactas,
        "invariantes_estruturais": chk.registro,
        "modelos": modelos,
        "coerencia_scaler_diametro_M0_M1": coerencia,
        "reproducao_M1": reproducao,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    if git.get("disponivel"):
        print(f"\n  git: HEAD {git['commit_head']} ({git['branch']}) | árvore limpa: {git['arvore_limpa']} "
              f"| script versionado: {git['script_versionado_no_head']}")
    else:
        print(f"\n  git: indisponível ({git.get('erro')})")
    print(f"\n{SEPARATOR}\nSTATUS: {status.upper()}  |  entradas intactas: {entradas_intactas}")
    print(f"Relatório: {relativo(OUTPUT_JSON)}\n{SEPARATOR}")
    return 0 if status == "reproduzido" else 1


if __name__ == "__main__":
    raise SystemExit(main())
