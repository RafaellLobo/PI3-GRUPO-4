#!/usr/bin/env python3
"""S6 [P0] - Parte A: consolida a comparacao dos modelos (validacao cruzada por paciente).

Nao ajusta modelo, nao reamostra e nao le o conjunto de teste. Apenas reorganiza
resultados ja versionados em reports/sprint6/benchmark e confere o resultado contra
o resumo_benchmark.md. Protocolo: PR #23 (docs/sprint6/protocolo_s6_p0_avaliacao_comparativa_teste.md).
"""
import argparse, csv, hashlib, json, platform, re, sys
from datetime import datetime, timezone
from pathlib import Path

ROTULO = {"lr_fixa": "LR fixa", "lr_tuning": "LR com ajuste", "svm": "SVM", "rf": "RF", "xgb": "XGBoost"}
ORDEM = ["lr_fixa", "lr_tuning", "svm", "rf", "xgb"]
ENTRADAS = ["auc_por_repeticao.csv", "contrastes.csv", "bootstrap_resumo.json", "denominadores.csv",
            "selecao_algoritmo.json", "config_benchmark.json", "resumo_benchmark.md"]
TOL = 5e-5  # conferencia ao resumo, que traz 4 casas decimais


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def le_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bool_(s):
    return str(s).strip().lower() == "true"


def f4(x):
    return f"{x:.4f}".replace(".", ",")


def parse_tabelas_resumo(texto):
    """Devolve {secao: [linhas de tabela como listas de celulas]} para as secoes 4, 5 e 6."""
    secao, saida = None, {}
    for lin in texto.splitlines():
        m = re.match(r"^##\s+(\d+)\.", lin)
        if m:
            secao = int(m.group(1))
            continue
        if secao in (4, 5, 6) and lin.startswith("|") and not re.match(r"^\|[-:| ]+\|$", lin):
            cel = [c.strip() for c in lin.strip().strip("|").split("|")]
            if cel[0] in ("Modelo", "Contraste"):
                continue
            saida.setdefault(secao, []).append(cel)
    return saida


def intervalo(txt):
    n = re.findall(r"-?\d+\.\d+", txt)
    return float(n[0]), float(n[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo-commit", default="nao informado")
    ap.add_argument("--protocolo", default="PR #23")
    a = ap.parse_args()
    bd, od = Path(a.benchmark_dir), Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)

    # 0. entradas e hashes
    faltam = [e for e in ENTRADAS if not (bd / e).exists()]
    if faltam:
        sys.exit(f"ERRO: entradas ausentes: {faltam}")
    hashes = {e: sha256(bd / e) for e in ENTRADAS}

    # 1. AUC por modelo e desfecho (media entre repeticoes validas) e IC descritivo
    auc_rep = le_csv(bd / "auc_por_repeticao.csv")
    boot = json.load(open(bd / "bootstrap_resumo.json", encoding="utf-8"))
    sel = json.load(open(bd / "selecao_algoritmo.json", encoding="utf-8"))
    theta = {(t["modelo"], t["desfecho"]): t for t in boot["intervalos_theta"]}
    quadro, checagens = [], []
    for des in ("G", "F"):
        for m in ORDEM:
            v = [float(r["auc"]) for r in auc_rep if r["modelo"] == m and r["desfecho"] == des and bool_(r["valida"])]
            t = theta[(m, des)]
            media = sum(v) / len(v)
            checagens.append(("AUC recalculada = theta do bootstrap", f"{ROTULO[m]} {des}", abs(media - t["theta"]) < 1e-9))
            quadro.append({"modelo": ROTULO[m], "codigo": m, "desfecho": des, "n_repeticoes_validas": len(v),
                           "auc_media": media, "ic95_descritivo_baixo": t["limite_baixo"],
                           "ic95_descritivo_alto": t["limite_alto"],
                           "rotulo": "otimista por selecao" if m == "lr_tuning" else ""})

    # 2. contrastes
    con = le_csv(bd / "contrastes.csv")
    def nome(r):
        return f"{ROTULO[r['a']]} - {ROTULO[r['b']]}"
    familia, sens = [], []
    for r in con:
        base = {"contraste": nome(r), "desfecho": r["desfecho"], "grupo": r["grupo"],
                "delta": float(r["delta"]), "minimo_repeticoes": float(r["minimo_repeticoes"]),
                "maximo_repeticoes": float(r["maximo_repeticoes"]),
                "repeticoes_positivas": int(r["n_repeticoes_positivas"]), "repeticoes_validas": int(r["n_repeticoes_validas"])}
        sens.append(base)
        if r["grupo"] == "familia":
            familia.append(dict(base, bonf_baixo=float(r["bonferroni_baixo"]), bonf_alto=float(r["bonferroni_alto"]),
                                ic95_baixo=float(r["ic95_baixo"]), ic95_alto=float(r["ic95_alto"])))

    # 3. denominadores (contagens por repeticao 0, conferidas nas demais)
    den = [r for r in le_csv(bd / "denominadores.csv") if bool_(r["valida"])]
    denom = {}
    for des in ("G", "F"):
        por_rep = {}
        for r in den:
            if r["desfecho"] == des:
                d = por_rep.setdefault(r["repeticao"], [0, 0, 0, 0])
                for i, k in enumerate(("nodulos", "pacientes", "positivos", "negativos")):
                    d[i] += int(r[k])
        conj = {tuple(v) for v in por_rep.values()}
        checagens.append(("denominadores identicos em todas as repeticoes", des, len(conj) == 1))
        n, p, pos, neg = next(iter(conj))
        denom[des] = {"nodulos": n, "pacientes": p, "positivos": pos, "negativos": neg, "repeticoes": len(por_rep)}

    # 4. conferencia contra o resumo_benchmark.md (secoes 4, 5, 6)
    tab = parse_tabelas_resumo((bd / "resumo_benchmark.md").read_text(encoding="utf-8"))
    inv = {v: k for k, v in ROTULO.items()}
    for c in tab.get(4, []):
        mod, des, th, ic = c[0], c[1], float(c[2]), intervalo(c[3])
        q = next(x for x in quadro if x["modelo"] == mod and x["desfecho"] == des)
        ok = (abs(q["auc_media"] - th) < TOL and abs(q["ic95_descritivo_baixo"] - ic[0]) < TOL
              and abs(q["ic95_descritivo_alto"] - ic[1]) < TOL)
        checagens.append(("quadro de AUC = resumo, Secao 4", f"{mod} {des}", ok))
    def chave(txt):
        a_, b_ = [s.strip() for s in txt.replace("−", "-").split(" - ")]
        return f"{a_} - {b_}"
    for c in tab.get(5, []):
        f = next(x for x in familia if x["contraste"] == chave(c[0]) and x["desfecho"] == c[1])
        bo, ic = intervalo(c[3].replace("−", "-")), intervalo(c[4].replace("−", "-"))
        ok = (abs(f["delta"] - float(c[2])) < TOL and abs(f["bonf_baixo"] - bo[0]) < TOL and abs(f["bonf_alto"] - bo[1]) < TOL
              and abs(f["ic95_baixo"] - ic[0]) < TOL and abs(f["ic95_alto"] - ic[1]) < TOL)
        checagens.append(("contrastes da familia = resumo, Secao 5", f"{c[0]} {c[1]}", ok))
    for c in tab.get(6, []):
        s = next(x for x in sens if x["contraste"] == chave(c[0]) and x["desfecho"] == c[1])
        pos = re.match(r"(\d+) de (\d+)", c[5])
        ok = (abs(s["delta"] - float(c[2])) < TOL and abs(s["minimo_repeticoes"] - float(c[3])) < TOL
              and abs(s["maximo_repeticoes"] - float(c[4])) < TOL and s["repeticoes_positivas"] == int(pos.group(1))
              and s["repeticoes_validas"] == int(pos.group(2)))
        checagens.append(("sensibilidade = resumo, Secao 6", f"{c[0]} {c[1]}", ok))
    falhas = [c for c in checagens if not c[2]]

    # 5. saidas
    def grava_csv(nome_, linhas):
        with open(od / nome_, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            w.writeheader()
            w.writerows(linhas)
    grava_csv("quadro_auc_por_modelo.csv", quadro)
    grava_csv("contrastes_familia.csv", familia)
    grava_csv("sensibilidade_particao.csv", sens)
    grava_csv("conferencia_vs_resumo.csv", [{"verificacao": c[0], "item": c[1], "resultado": "OK" if c[2] else "FALHA"} for c in checagens])

    L = ["# S6 [P0] - Parte A: quadro comparativo da validacao cruzada por paciente", "",
         "Rotulos: o alvo e suspeicao radiologica (classe 1 e classe 0 pela mediana das notas), nao malignidade confirmada. "
         "Os intervalos sao descritivos (bootstrap de pacientes, condicional aos modelos ajustados). "
         "A LR com ajuste e otimista por selecao. Nenhum resultado e validacao independente.", "",
         f"Populacao: G = {denom['G']['nodulos']} nodulos, {denom['G']['pacientes']} pacientes "
         f"({denom['G']['positivos']} classe 1 e {denom['G']['negativos']} classe 0); "
         f"F (12 a 30 mm) = {denom['F']['nodulos']} nodulos, {denom['F']['pacientes']} pacientes "
         f"({denom['F']['positivos']} classe 1 e {denom['F']['negativos']} classe 0). "
         "Contagens somadas sobre as dobras de uma repeticao e identicas nas 10 repeticoes.", "",
         "## Quadro 1 - AUC por modelo e desfecho", "",
         "| Modelo | Desfecho | AUC | IC 95% descritivo | Rotulo |", "|---|---|--:|---|---|"]
    for q in quadro:
        L.append(f"| {q['modelo']} | {q['desfecho']} | {f4(q['auc_media'])} | [{f4(q['ic95_descritivo_baixo'])}; {f4(q['ic95_descritivo_alto'])}] | {q['rotulo']} |")
    L += ["", "## Quadro 2 - Contrastes da familia principal (contra a LR com ajuste)", "",
          "| Contraste | Desfecho | Diferenca estimada | IC Bonferroni (6 contrastes) | IC 95% sem correcao |", "|---|---|--:|---|---|"]
    for f in familia:
        L.append(f"| {f['contraste']} | {f['desfecho']} | {f4(f['delta'])} | [{f4(f['bonf_baixo'])}; {f4(f['bonf_alto'])}] | [{f4(f['ic95_baixo'])}; {f4(f['ic95_alto'])}] |")
    L += ["", "## Quadro 3 - Sensibilidade a particao (todos os contrastes)", "",
          "| Contraste | Desfecho | Grupo | Media | Minimo | Maximo | Repeticoes com diferenca > 0 |", "|---|---|---|--:|--:|--:|--:|"]
    for s in sens:
        L.append(f"| {s['contraste']} | {s['desfecho']} | {s['grupo']} | {f4(s['delta'])} | {f4(s['minimo_repeticoes'])} | {f4(s['maximo_repeticoes'])} | {s['repeticoes_positivas']} de {s['repeticoes_validas']} |")
    cr = sel["criterios"]
    L += ["", "## Regra de selecao do P9 (reaplicada a partir de selecao_algoritmo.json)", "",
          f"Algoritmo recomendado: {ROTULO.get(sel['algoritmo_recomendado'], sel['algoritmo_recomendado'])}.", "",
          "| Desafiante | C1 (magnitude) | C2 (consistencia) | C3 (sem piora em F) | C4 (integridade) |", "|---|---|---|---|---|"]
    for m in ("rf", "xgb", "svm"):
        c = cr[m]
        L.append(f"| {ROTULO[m]} | " + " | ".join("cumpre" if c[k]["cumprido"] else "nao cumpre" for k in ("C1", "C2", "C3", "C4")) + " |")
    L += ["", "## Leitura permitida", "",
          "Descritiva e condicional: diferenca estimada, compativel com, nao detectada nesta base. "
          "Vetados: significativo, superior, comprovado, demonstrado, estatisticamente, p-valor. "
          "O bootstrap nao reajusta os modelos nem captura a variabilidade da escolha de hiperparametros; "
          "o limite de Bonferroni vale para qualquer dependencia entre os contrastes, desde que cada intervalo marginal tenha a cobertura nominal; como o bootstrap e aproximado, essa cobertura e apenas aproximada. "
          "Intervalos largos nao provam equivalencia."]
    (od / "quadro_comparativo_cv.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    log = {"script": Path(__file__).name, "executado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "protocolo": a.protocolo, "commit_dos_dados": a.repo_commit, "python": platform.python_version(),
           "entradas_sha256": hashes, "conferencias": {"total": len(checagens), "ok": len(checagens) - len(falhas), "falhas": len(falhas)},
           "le_teste": False, "ajusta_modelos": False}
    (od / "log_execucao.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"conferencias: {len(checagens) - len(falhas)} OK, {len(falhas)} falhas")
    for c in falhas:
        print("FALHA:", c[0], c[1])
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
