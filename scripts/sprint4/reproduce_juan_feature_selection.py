"""Etapa 0 — reprodução fiel da seleção de features do notebook de Juan.

Versão versionável do script de reprodução auditado em
``reports/sprint4/feature_selection/auditoria_reproducao_juan.md`` (Seção 15,
script original com SHA-256 ``066fee8d…``). O núcleo da seleção (blocos 1–5)
replica o código do notebook "S4 — Revisão e execução da seleção de features
radiômicas" sem alterar algoritmo, filtros, seeds, parâmetros, ordem das
features nem critério de estabilidade.

Diferenças em relação ao notebook, nenhuma delas computacional:

* caminhos apontam para os arquivos do projeto;
* warnings são REGISTRADOS em vez de suprimidos;
* ``n_jobs`` é parâmetro (padrão 1: capta os warnings emitidos nos ajustes).

Diferenças em relação ao script original auditado, apenas de portabilidade e
rastreabilidade:

* ``argparse``; a raiz do repositório é derivada do próprio arquivo;
* o notebook é opcional e serve só para registrar seu hash;
* o JSON não contém caminhos absolutos; as chaves de arquivo são relativas ao
  repositório, em formato POSIX; registra estado do git e hash deste script;
* o diagnóstico de convergência, antes um script separado, roda DEPOIS da
  seleção, com captura de warnings própria. Ele refaz a mesma partição e os
  mesmos filtros, confere que são idênticos aos da seleção e não altera a
  sequência de operações nem os resultados da seleção.

LIMITAÇÃO METODOLÓGICA PRESERVADA DE PROPÓSITO
----------------------------------------------
Como no notebook, os filtros de variância e de correlação são ajustados no
treino inteiro de cada fold de estabilidade, antes e fora da ``GridSearchCV``
interna (auditoria, Seção 2.4). Esta reprodução fiel mantém esse desenho; ele
NÃO deve ser reutilizado na CV aninhada corrigida (protocolo, Seção 5.4).

ISOLAMENTO
----------
Como no notebook, a base e o alvo são carregados inteiros; as linhas de
validação e de teste servem apenas à verificação de disjunção de pacientes e
são descartadas em seguida. Somente os 417 nódulos do treino oficial entram em
filtros, LASSO, Random Forest e diagnóstico. Nenhum modelo M0–M4 é ajustado e
nenhuma métrica de validação ou de teste é calculada.

Uso (a partir da raiz do repositório):

    python scripts/sprint4/reproduce_juan_feature_selection.py \\
        [--out-dir DIR] [--notebook NOTEBOOK.ipynb] [--n-jobs 1] [--sobrescrever]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.radiomics.ids import decompor_id_extracao  # noqa: E402

# Script original auditado (auditoria, Seção 15).
SCRIPT_ORIGINAL_SHA256 = "066fee8da95dc6b0f9d41b485c036d6d64ed42c2f4a072bb7cc9dc67c859c66c"
DIAGNOSTICO_ORIGINAL_SHA256 = "fa83f5b56a154aeabb9e2d1e32b05288973960b239eb06f4747fc5318b258b4d"

# ----------------------------------------------------------------- constantes
# idênticas ao notebook (célula 1)
RANDOM_STATE = 42
N_FOLDS = 5
RF_TOP_K = 15
LIMIAR_ESTABILIDADE = 3

BASE_PATH = REPO_ROOT / "data/base_radiomica_oficial_422.csv"
TARGET_PATH = REPO_ROOT / "data/vies_malignidade_mascara_pequena.csv"
SPLIT_PATH = REPO_ROOT / "reports/sprint3/baseline/patient_split.csv"
CANON_PATH = REPO_ROOT / "data/modeling_table_sprint3.csv"
OUT_PADRAO = REPO_ROOT / "reports/sprint4/feature_selection/reproducao_juan"

SAIDAS = (
    "frequencia_selecao_features.csv",
    "resumo_folds_selecao.csv",
    "features_selecionadas_lasso.csv",
    "relatorio_reproducao.json",
)

DECLARADAS = [
    "original_glcm_Imc2",
    "original_shape_Sphericity",
    "original_shape_SurfaceVolumeRatio",
    "original_firstorder_90Percentile",
    "original_firstorder_RootMeanSquared",
    "original_shape_Elongation",
    "original_firstorder_Skewness",
    "original_ngtdm_Coarseness",
]

# Features cuja divergência é analisada na auditoria (Seção 7.1).
DIVERGENTES = [
    "original_firstorder_Skewness",
    "original_ngtdm_Coarseness",
    "original_glcm_Imc1",
    "original_glcm_ClusterShade",
]


# --------------------------------------------------------------------- apoio
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def sha256_lf(p: Path) -> str:
    """SHA-256 após normalizar somente CRLF -> LF (comparação entre sistemas)."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def exigir(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"INVARIANTE VIOLADA: {msg}")


def relativo(p: Path) -> str:
    """Caminho relativo ao repositório, em POSIX; nunca absoluto."""
    p = p.resolve()
    if p == REPO_ROOT or REPO_ROOT in p.parents:
        return p.relative_to(REPO_ROOT).as_posix()
    return "<externo ao repositório>"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )


def estado_git() -> dict[str, Any]:
    """Estado do repositório no momento da execução (protocolo, Seção 12). Somente leitura."""
    try:
        head = _git("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"disponivel": False, "erro": head.stderr.strip()}
        branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        status = [linha for linha in _git("status", "--porcelain").stdout.splitlines() if linha]
        script = relativo(Path(__file__))
        versionado = _git("ls-files", "--error-unmatch", "--", script).returncode == 0
        # None quando o script não está no HEAD: "sem alterações" não se aplica.
        alterado = _git("diff", "--quiet", "HEAD", "--", script).returncode != 0 if versionado else None
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
            "o resultado só é rastreável a um commit quando o script está versionado e sem "
            "alterações; caso contrário, commit_head indica apenas a base da execução, e o "
            "script é identificado por sha256_script"
        ),
    }


def argumentos() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_PADRAO,
                    help="diretório das saídas (padrão: %(default)s)")
    ap.add_argument("--notebook", type=Path, default=None,
                    help="notebook auditado, usado apenas para registrar o SHA-256 (opcional)")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="n_jobs da GridSearchCV e do Random Forest; o resultado não depende dele (padrão: 1)")
    ap.add_argument("--sobrescrever", action="store_true",
                    help="permite substituir saídas já existentes no diretório")
    return ap.parse_args()


# ------------------------------------------------------- diagnóstico (após a seleção)
def diagnostico_convergencia(X, y, groups, f_reduzidas_por_fold, resumo_folds, detalhes_folds) -> dict:
    """Replica ``diagnostico_convergencia.py`` sobre a mesma partição e os mesmos filtros.

    Roda depois da seleção e não altera nenhum resultado dela. Confere que os
    folds e as features pós-filtro são idênticos aos usados na seleção e que o
    refit com o melhor C reproduz os coeficientes selecionados.
    """
    grade = np.logspace(-4, 2, 13)
    melhores = {r["fold"]: r["melhor_C_lasso"] for r in resumo_folds}

    def pipe(c):
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000,
                                                    random_state=RANDOM_STATE, C=c))])

    nao_convergidos, refit, n_ajustes, consistencia = [], [], 0, []
    outer = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, _) in enumerate(outer.split(X, y, groups), start=1):
        Xt, yt, gt = X.iloc[tr], y.iloc[tr], groups.iloc[tr]
        vt = VarianceThreshold(0.0).fit(Xt)
        fv = Xt.columns[vt.get_support()].tolist()
        corr = Xt[fv].corr().abs()
        up = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        red = [c for c in up.columns if (up[c] > 0.90).any()]
        fr = [f for f in fv if f not in red]

        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=100 + fold)
        for k, (itr, _) in enumerate(inner.split(Xt[fr], yt, gt), start=1):
            for c in grade:
                n_ajustes += 1
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    pipe(c).fit(Xt[fr].iloc[itr], yt.iloc[itr])
                if any(issubclass(x.category, ConvergenceWarning) for x in w):
                    nao_convergidos.append({"fold_externo": fold, "fold_interno": k, "C": float(c)})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = pipe(melhores[fold]).fit(Xt[fr], yt)
        coef = p.named_steps["clf"].coef_.ravel()
        coef_nz = {f: float(v) for f, v in zip(fr, coef) if abs(v) > 1e-12}
        refit.append({"fold": fold, "C": melhores[fold],
                      "n_iter": int(np.max(p.named_steps["clf"].n_iter_)),
                      "warning": any(issubclass(x.category, ConvergenceWarning) for x in w)})
        consistencia.append({
            "fold": fold,
            "n_treino_nodulos_igual": len(Xt) == resumo_folds[fold - 1]["n_treino_nodulos"],
            "features_pos_filtros_iguais": fr == f_reduzidas_por_fold[fold - 1],
            "coeficientes_refit_iguais_aos_da_selecao": coef_nz == detalhes_folds[fold - 1]["coeficientes_lasso"],
        })

    # Partição e filtros precisam coincidir para o diagnóstico valer; a igualdade
    # dos coeficientes do refit é registrada, não exigida.
    exigir(all(c["n_treino_nodulos_igual"] and c["features_pos_filtros_iguais"] for c in consistencia),
           "o diagnóstico não reproduz a partição ou os filtros da seleção")

    tabela = {d["fold"]: {f: d["coeficientes_lasso"].get(f) for f in DIVERGENTES} for d in detalhes_folds}
    todos = pd.Series([v for d in detalhes_folds for v in d["coeficientes_lasso"].values()])
    return {
        "escopo": "somente treino; mesma partição e mesmos filtros da seleção; executado após a seleção",
        "derivado_de_sha256": DIAGNOSTICO_ORIGINAL_SHA256,
        "ajustes_na_grade": n_ajustes,
        "ajustes_nao_convergidos": nao_convergidos,
        "refit_com_melhor_C": refit,
        "consistencia_com_a_selecao": consistencia,
        "coeficientes_l1_features_divergentes": {
            "observacao": "espaço padronizado; null = não selecionada no fold",
            "por_fold": tabela,
        },
        "abs_coef_todas_as_selecoes": {
            "min": float(todos.abs().min()),
            "mediana": float(todos.abs().median()),
            "max": float(todos.abs().max()),
        },
    }


# ----------------------------------------------------------------------- main
def main() -> int:
    args = argumentos()
    OUT = args.out_dir.resolve()
    N_JOBS = args.n_jobs
    NOTEBOOK = args.notebook

    exigir(REPO_ROOT / "reports/sprint3" not in [OUT, *OUT.parents], "saída em reports/sprint3 é proibida")
    exigir(REPO_ROOT / "data" not in [OUT, *OUT.parents], "saída em data/ é proibida")
    existentes = [n for n in SAIDAS if (OUT / n).exists()]
    exigir(not existentes or args.sobrescrever,
           f"saídas já existem em {relativo(OUT)}: {existentes}; use --sobrescrever para substituí-las")
    for p in (BASE_PATH, TARGET_PATH, SPLIT_PATH, CANON_PATH):
        exigir(p.exists(), f"entrada ausente: {relativo(p)}")
    if NOTEBOOK is not None:
        exigir(NOTEBOOK.exists(), "notebook informado não existe")

    script_path = Path(__file__).resolve()
    relatorio: dict = {
        "etapa": "Sprint 4 — Cartão 4 — Etapa 0: reprodução fiel da seleção de features de Juan",
        "documentacao": "reports/sprint4/feature_selection/auditoria_reproducao_juan.md",
        "script": relativo(script_path),
        "sha256_script": sha256(script_path),
        "derivado_do_script_auditado_sha256": SCRIPT_ORIGINAL_SHA256,
        "n_jobs": N_JOBS,
        "parametros": {
            "random_state": RANDOM_STATE, "n_folds": N_FOLDS, "rf_top_k": RF_TOP_K,
            "limiar_estabilidade": LIMIAR_ESTABILIDADE, "limiar_correlacao": 0.90,
            "grade_C": "logspace(-4, 2, 13)", "cv_interna": "StratifiedGroupKFold(4, shuffle=True, random_state=100 + fold)",
            "lasso": "LogisticRegression(penalty='l1', solver='liblinear', max_iter=5000, random_state=42)",
            "rf": "RandomForestClassifier(n_estimators=500, random_state=42 + fold, class_weight='balanced_subsample')",
            "criterio_selecao_no_fold": "|coef| > 1e-12",
        },
        "isolamento": {
            "linhas_em_filtros_lasso_rf_diagnostico": "somente os 417 nódulos do treino oficial",
            "validacao_teste": "usadas apenas na verificação de disjunção de pacientes e descartadas em seguida",
            "modelos_M0_M4": "não ajustados",
            "metricas_validacao_teste": "não calculadas",
        },
        "saida": {"diretorio": relativo(OUT)},
        "git": estado_git(),
    }
    avisos_registrados: list[dict] = []

    relatorio["ambiente"] = {
        "python": sys.version.split()[0],
        "plataforma": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }
    entradas = (BASE_PATH, TARGET_PATH, SPLIT_PATH, CANON_PATH)
    relatorio["sha256_entradas"] = {
        relativo(p): {"sha256": sha256(p), "sha256_lf_normalizado": sha256_lf(p)} for p in entradas
    }
    relatorio["notebook_auditado"] = (
        {"arquivo": NOTEBOOK.name, "sha256": sha256(NOTEBOOK), "uso": "somente registro de hash"}
        if NOTEBOOK is not None else None
    )

    t0 = time.time()
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")

        # ============================================================ bloco 1
        # célula 3 do notebook — carregamento e base modelável
        df_base = pd.read_csv(BASE_PATH)
        df_target = pd.read_csv(TARGET_PATH)[["nodulo_id", "malignidade_mediana"]]

        df = df_base.merge(df_target, on="nodulo_id", how="left", validate="one_to_one")
        assert df["malignidade_mediana"].isna().sum() == 0, "Há nódulos sem target após o merge."

        df["malignidade_binaria"] = np.where(
            df["malignidade_mediana"] <= 2, 0,
            np.where(df["malignidade_mediana"] >= 4, 1, np.nan),
        )
        df_modelavel = df.dropna(subset=["malignidade_binaria"]).copy()
        df_modelavel["malignidade_binaria"] = df_modelavel["malignidade_binaria"].astype(int)

        FEATURES = [c for c in df_modelavel.columns if c.startswith("original_")]

        assert len(df_modelavel) == 607
        assert df_modelavel["patient_id"].nunique() == 416
        assert len(FEATURES) == 107

        # ============================================================ bloco 2
        # célula 5 do notebook — split oficial e isolamento
        split_df = pd.read_csv(SPLIT_PATH)
        assert set(["patient_id", "split"]).issubset(split_df.columns)
        assert split_df["patient_id"].nunique() == 416
        assert len(split_df) == 416
        assert set(split_df["split"].unique()) == {"train", "validation", "test"}

        df_modelavel = df_modelavel.merge(
            split_df[["patient_id", "split"]], on="patient_id", how="left", validate="many_to_one"
        )
        assert df_modelavel["split"].isna().sum() == 0

        df_selecao = df_modelavel[df_modelavel["split"] == "train"].copy()
        df_validacao = df_modelavel[df_modelavel["split"] == "validation"].copy()
        df_teste = df_modelavel[df_modelavel["split"] == "test"].copy()

        assert set(df_selecao["patient_id"]).isdisjoint(set(df_validacao["patient_id"]))
        assert set(df_selecao["patient_id"]).isdisjoint(set(df_teste["patient_id"]))
        assert set(df_validacao["patient_id"]).isdisjoint(set(df_teste["patient_id"]))
        assert df_selecao["patient_id"].nunique() == 291
        assert df_validacao["patient_id"].nunique() == 62
        assert df_teste["patient_id"].nunique() == 63

        # Nada de validação/teste é usado daqui em diante.
        del df_validacao, df_teste

        # ============================================================ checagens prévias
        # (adicionais ao notebook; somente leitura, somente treino)
        canon = pd.read_csv(CANON_PATH)
        train_pids = set(split_df.loc[split_df["split"] == "train", "patient_id"])
        canon_tr = canon[canon["patient_id"].isin(train_pids)].copy()

        chk: dict = {}
        # IDs
        for nome, frame in (("base", df_base), ("vies", pd.read_csv(TARGET_PATH)), ("canonica", canon)):
            pids = [decompor_id_extracao(x)[0] for x in frame["nodulo_id"]]  # ValueError aborta
            chk[f"ids_{nome}"] = {
                "linhas": int(len(frame)),
                "duplicados": int(frame["nodulo_id"].duplicated().sum()),
                "formato_S3_ok": True,
            }
            if "patient_id" in frame.columns:
                chk[f"ids_{nome}"]["patient_id_coerente"] = int(
                    (pd.Series(pids, index=frame.index) == frame["patient_id"]).sum()
                )
        exigir(chk["ids_base"]["duplicados"] == 0 and chk["ids_vies"]["duplicados"] == 0, "IDs duplicados")
        exigir(chk["ids_base"]["patient_id_coerente"] == len(df_base), "patient_id incoerente na base")
        exigir(set(df_base["nodulo_id"]) == set(pd.read_csv(TARGET_PATH)["nodulo_id"]),
               "conjuntos de nodulo_id da base e do arquivo de alvo diferem")

        # Linhas de seleção = treino canônico
        exigir(len(df_selecao) == 417, "seleção != 417 nódulos")
        exigir(set(df_selecao["nodulo_id"]) == set(canon_tr["nodulo_id"]), "nódulos de seleção != treino canônico")
        m = df_selecao[["nodulo_id", "malignidade_binaria"]].merge(
            canon_tr[["nodulo_id", "target_binary"]], on="nodulo_id", validate="one_to_one"
        )
        chk["target_identico_ao_canonico"] = int((m["malignidade_binaria"] == m["target_binary"]).sum())
        exigir(chk["target_identico_ao_canonico"] == 417, "target diverge do canônico")
        chk["classes_selecao"] = {str(k): int(v) for k, v in df_selecao["malignidade_binaria"].value_counts().sort_index().items()}
        exigir(chk["classes_selecao"] == {"0": 209, "1": 208}, "composição de classes inesperada")

        # Features: 107, ordem, valores
        canon_feats = [c for c in canon.columns if c.startswith("original_") and c != "original_nodule_idx"]
        chk["n_features_candidatas"] = len(FEATURES)
        chk["original_nodule_idx_em_FEATURES"] = "original_nodule_idx" in FEATURES
        chk["ordem_107_igual_canonica_sem_nodule_idx"] = FEATURES == canon_feats
        exigir(not chk["original_nodule_idx_em_FEATURES"], "original_nodule_idx entre as candidatas")
        exigir(chk["ordem_107_igual_canonica_sem_nodule_idx"], "ordem das 107 candidatas difere")
        a = df_selecao.set_index("nodulo_id")[FEATURES].sort_index()
        b = canon_tr.set_index("nodulo_id")[FEATURES].sort_index()
        chk["valores_identicos_treino"] = bool((a.values == b.values).all())
        exigir(chk["valores_identicos_treino"], "valores das features divergem da tabela canônica")
        Xchk = df_selecao[FEATURES].to_numpy(dtype=float)
        chk["valores_nao_finitos_treino"] = int((~np.isfinite(Xchk)).sum())
        exigir(chk["valores_nao_finitos_treino"] == 0, "valores não finitos no treino")
        chk["declaradas_presentes"] = all(f in FEATURES for f in DECLARADAS)
        chk["posicao_das_declaradas_no_pool"] = {f: FEATURES.index(f) for f in DECLARADAS}
        relatorio["checagens_previas"] = chk

        # ============================================================ bloco 3
        # célula 7 do notebook — seleção estável dentro do treino
        X = df_selecao[FEATURES]
        y = df_selecao["malignidade_binaria"]
        groups = df_selecao["patient_id"]

        outer_cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        lasso_por_fold = []
        rf_por_fold = []
        pos_filtros_por_fold = []
        resumo_folds = []
        detalhes_folds = []
        f_reduzidas_por_fold = []  # registro para o diagnóstico (não altera nada)

        for fold, (train_idx, _) in enumerate(outer_cv.split(X, y, groups), start=1):
            X_train = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()
            g_train = groups.iloc[train_idx].copy()

            # 1) remove apenas features constantes.
            vt = VarianceThreshold(threshold=0.0)
            vt.fit(X_train)
            f_variancia = X_train.columns[vt.get_support()].tolist()

            # 2) redundância por correlação, aprendida somente no treino deste fold.
            corr = X_train[f_variancia].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            redundantes = [col for col in upper.columns if (upper[col] > 0.90).any()]
            f_reduzidas = [f for f in f_variancia if f not in redundantes]
            pos_filtros_por_fold.append(set(f_reduzidas))

            # 3) LASSO: tuning interno também agrupado por paciente.
            inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=100 + fold)
            lasso_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000,
                                           random_state=RANDOM_STATE)),
            ])
            busca = GridSearchCV(
                estimator=lasso_pipe,
                param_grid={"clf__C": np.logspace(-4, 2, 13)},
                cv=inner_cv,
                scoring="roc_auc",
                n_jobs=N_JOBS,
                refit=True,
            )
            busca.fit(X_train[f_reduzidas], y_train, groups=g_train)

            coef = busca.best_estimator_.named_steps["clf"].coef_.ravel()
            selecionadas_lasso = set(np.array(f_reduzidas)[np.abs(coef) > 1e-12])
            lasso_por_fold.append(selecionadas_lasso)

            # 4) Random Forest como checagem independente.
            rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE + fold,
                                        n_jobs=N_JOBS, class_weight="balanced_subsample")
            rf.fit(X_train[f_reduzidas], y_train)
            importancia = pd.Series(rf.feature_importances_, index=f_reduzidas).sort_values(ascending=False)
            selecionadas_rf = set(importancia.head(RF_TOP_K).index)
            rf_por_fold.append(selecionadas_rf)

            resumo_folds.append({
                "fold": fold,
                "n_treino_nodulos": len(X_train),
                "n_treino_pacientes": g_train.nunique(),
                "n_features_entrada": len(FEATURES),
                "n_pos_variancia": len(f_variancia),
                "n_pos_correlacao": len(f_reduzidas),
                "melhor_C_lasso": busca.best_params_["clf__C"],
                "n_lasso_selecionadas": len(selecionadas_lasso),
                "n_rf_top15": len(selecionadas_rf),
            })

            # registro adicional (não altera nada)
            grade = list(np.logspace(-4, 2, 13))
            best_c = busca.best_params_["clf__C"]
            f_reduzidas_por_fold.append(list(f_reduzidas))
            detalhes_folds.append({
                "fold": fold,
                "pacientes_treino_subconjunto_do_treino_oficial": bool(set(g_train) <= train_pids),
                "classes_treino": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
                "removidas_variancia": sorted(set(FEATURES) - set(f_variancia)),
                "n_redundantes": len(redundantes),
                "melhor_C_no_limite_da_grade": bool(np.isclose(best_c, grade[0]) or np.isclose(best_c, grade[-1])),
                "best_score_cv_interna_treino": float(busca.best_score_),
                "declaradas_sobreviventes_aos_filtros": sorted(set(DECLARADAS) & set(f_reduzidas)),
                "declaradas_eliminadas_pelos_filtros": sorted(set(DECLARADAS) - set(f_reduzidas)),
                "lasso_selecionadas": sorted(selecionadas_lasso),
                "coeficientes_lasso": {f: float(c) for f, c in zip(f_reduzidas, coef) if abs(c) > 1e-12},
                "rf_top15": list(importancia.head(RF_TOP_K).index),
            })

        # ============================================================ bloco 4
        # célula 9 do notebook — frequência e estabilidade
        freq_lasso = pd.Series(0, index=FEATURES, dtype=int)
        freq_rf = pd.Series(0, index=FEATURES, dtype=int)
        freq_pos = pd.Series(0, index=FEATURES, dtype=int)
        for conjunto in lasso_por_fold:
            for f in conjunto:
                freq_lasso[f] += 1
        for conjunto in rf_por_fold:
            for f in conjunto:
                freq_rf[f] += 1
        for conjunto in pos_filtros_por_fold:
            for f in conjunto:
                freq_pos[f] += 1

        tabela_estabilidade = pd.DataFrame({
            "freq_pos_filtros": freq_pos,
            "freq_lasso": freq_lasso,
            "freq_rf_top15": freq_rf,
        })
        tabela_estabilidade["pct_lasso"] = tabela_estabilidade["freq_lasso"] / N_FOLDS
        tabela_estabilidade["pct_rf_top15"] = tabela_estabilidade["freq_rf_top15"] / N_FOLDS
        tabela_estabilidade["estavel_lasso_3de5"] = tabela_estabilidade["freq_lasso"] >= LIMIAR_ESTABILIDADE
        tabela_estabilidade["estavel_rf_3de5"] = tabela_estabilidade["freq_rf_top15"] >= LIMIAR_ESTABILIDADE
        tabela_estabilidade = tabela_estabilidade.sort_values(
            ["freq_lasso", "freq_rf_top15", "freq_pos_filtros"], ascending=False
        )

        # ============================================================ bloco 5
        # célula 11 do notebook — conjunto final
        features_finais_lasso = tabela_estabilidade.query("freq_lasso >= @LIMIAR_ESTABILIDADE").index.tolist()
        features_finais_rf = tabela_estabilidade.query("freq_rf_top15 >= @LIMIAR_ESTABILIDADE").index.tolist()
        intersecao = sorted(set(features_finais_lasso) & set(features_finais_rf))
        uniao = set(features_finais_lasso) | set(features_finais_rf)
        jaccard = len(intersecao) / len(uniao) if uniao else 0.0

    relatorio["tempo_s"] = round(time.time() - t0, 1)
    for w in capturados:
        avisos_registrados.append({"categoria": w.category.__name__, "mensagem": str(w.message)[:300],
                                   "arquivo": Path(w.filename).name, "linha": w.lineno})
    relatorio["warnings"] = {
        "escopo": "seleção (blocos 1–5); o diagnóstico registra os seus à parte",
        "total": len(avisos_registrados),
        "por_categoria": pd.Series([a["categoria"] for a in avisos_registrados], dtype=object).value_counts().to_dict(),
        "exemplos_unicos": list({(a["categoria"], a["mensagem"]): a for a in avisos_registrados}.values())[:10],
        "observacao": ("com n_jobs != 1, warnings emitidos dentro dos workers do joblib não chegam ao "
                       "processo principal; a execução com n_jobs=1 capta todos"),
    }

    declaradas = set(DECLARADAS)
    reproduzidas = set(features_finais_lasso)
    relatorio["resumo_folds"] = resumo_folds
    relatorio["detalhes_folds"] = detalhes_folds
    relatorio["observacao_best_score"] = (
        "best_score_cv_interna_treino é a AUC média da CV interna usada para escolher C, calculada "
        "somente com linhas do treino oficial; não é métrica de validação nem de teste"
    )
    relatorio["resultado"] = {
        "features_finais_lasso": features_finais_lasso,
        "n_features_finais_lasso": len(features_finais_lasso),
        "features_finais_rf": features_finais_rf,
        "n_features_finais_rf": len(features_finais_rf),
        "intersecao_lasso_rf": intersecao,
        "jaccard_lasso_rf": jaccard,
        "frequencia_lasso_nao_nula": {f: int(v) for f, v in freq_lasso[freq_lasso > 0].sort_values(ascending=False).items()},
    }
    relatorio["comparacao_declaradas"] = {
        "conjunto_identico": reproduzidas == declaradas,
        "so_reproduzidas": sorted(reproduzidas - declaradas),
        "so_declaradas": sorted(declaradas - reproduzidas),
        "ordem_identica": features_finais_lasso == DECLARADAS,
        "frequencia_lasso_das_declaradas": {f: int(freq_lasso[f]) for f in DECLARADAS},
        "frequencia_pos_filtros_das_declaradas": {f: int(freq_pos[f]) for f in DECLARADAS},
    }
    relatorio["afirmacoes_do_notebook"] = {
        "n_pos_correlacao_por_fold (declarado 35–39)": [r["n_pos_correlacao"] for r in resumo_folds],
        "n_lasso_por_fold (declarado 7–13)": [r["n_lasso_selecionadas"] for r in resumo_folds],
        "n_rf_estaveis (declarado 16)": len(features_finais_rf),
        "intersecao_lasso_rf (declarado 5)": len(intersecao),
        "jaccard (declarado 0,263)": round(jaccard, 3),
    }

    # Diagnóstico de convergência: depois da seleção, com captura própria.
    t1 = time.time()
    with warnings.catch_warnings(record=True) as capt_diag:
        warnings.simplefilter("always")
        diag = diagnostico_convergencia(X, y, groups, f_reduzidas_por_fold, resumo_folds, detalhes_folds)
    diag["tempo_s"] = round(time.time() - t1, 1)
    diag["warnings_total"] = len(capt_diag)
    relatorio["diagnostico_convergencia"] = diag

    OUT.mkdir(parents=True, exist_ok=True)
    tabela_estabilidade.reset_index(names="feature").to_csv(OUT / "frequencia_selecao_features.csv", index=False)
    pd.DataFrame(resumo_folds).to_csv(OUT / "resumo_folds_selecao.csv", index=False)
    pd.DataFrame({"feature": features_finais_lasso, "criterio": "LASSO >= 3/5 folds"}).to_csv(
        OUT / "features_selecionadas_lasso.csv", index=False)

    relatorio["sha256_saidas"] = {
        n: {"sha256": sha256(OUT / n), "sha256_lf_normalizado": sha256_lf(OUT / n)} for n in SAIDAS[:3]
    }
    relatorio["entradas_intactas_apos_execucao"] = all(
        sha256(p) == relatorio["sha256_entradas"][relativo(p)]["sha256"] for p in entradas
    )
    (OUT / "relatorio_reproducao.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print(json.dumps({k: relatorio[k] for k in ("ambiente", "tempo_s", "resumo_folds", "resultado",
                                                 "comparacao_declaradas", "afirmacoes_do_notebook",
                                                 "sha256_saidas")},
                     indent=1, ensure_ascii=False, default=float))
    print("warnings da seleção:", relatorio["warnings"]["total"], relatorio["warnings"]["por_categoria"])
    print("diagnóstico: não convergidos", diag["ajustes_nao_convergidos"], "| warnings", diag["warnings_total"])
    print("entradas intactas:", relatorio["entradas_intactas_apos_execucao"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
