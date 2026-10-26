"""Supplementary analyses of the AWSM design (stored runs only):
  T7_LIMIT_STUDY.md        Limit in {10, 15, 25, 100} on the tuning seeds 2001-2005 (campaign t7limit) against the v1 runs
                           and NSGA-II of campaign t7 (same scenario and seeds)
  AWSM_ABLATION.md         2x2 ablation on the six clamped-free main scenarios (seeds 1001-1030):
                           v1b (Limit 100, 2307 FE per run, archive gaps, no refinement), A2 (Limit 10, 2307 FE per run),
                           A1 (Limit 100, v2 budget structure with refinement), v2 (Limit 10, v2 budget structure)
Usage: python -I -X utf8 analyze_ablation.py [--only t7limit,ablation]"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_campaign as AC  # noqa: E402
import metrics as M  # noqa: E402
import stat_tests as S  # noqa: E402

RES = AC.RESULTS
OUT = AC.OUT


def load_runs(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        out.append({"front": z["front"][:, :2], "meta": m, "alg": m["alg"], "seed": m["seed"]})
    return out


def t7limit():
    groups = {}
    for lim in (10, 15, 25, 100):
        groups[f"v2, Limit = {lim}"] = load_runs(f"{RES}/t7limit/ALSIC_CF_mu0.1_limit{lim}/*.npz")
    groups["v1 (archive gaps, Limit = 100, 1500 FE per weight)"] = [r for r in load_runs(f"{RES}/t7/ALSIC_CF_mu0.1/awsm_B30000_*.npz")]
    groups["NSGA-II"] = [r for r in load_runs(f"{RES}/t7/ALSIC_CF_mu0.1/nsga2_B30000_*.npz")]
    allp = np.vstack([r["front"] for v in groups.values() for r in v])
    ref = (1.1 * allp[:, 0].max(), 0.9 * allp[:, 1].min())
    L = ["# T7_LIMIT_STUDY (AWSM v2: deneme limiti duyarlılığı)\n",
         f"*Üretici: `03_Code_Execution/analyze_ablation.py`; senaryo ALSIC_CF_mu0.1; ayar tohumları 2001–2005 (kampanya tohumlarından ayrı); B = 30 000. "
         f"HV tüm gruplar için ortak ham referansla r = ({ref[0]:.5g}, {ref[1]:.5g}). n = 5 koşu: aralıklar örtüşür, fark kanıtı değildir; "
         "yalnız Lévy aşamasının etkinliği ve HV'nin kötüleşmediği gösterilir.*\n",
         "| Grup | n | HV medyan | HV min | HV maks | FE medyan | Lévy (öncü) olayı medyan | inceltme ağırlığı medyan |\n|---|---|---|---|---|---|---|---|"]
    for k, v in groups.items():
        hv = np.array([M.hypervolume(r["front"], ref) for r in v])
        ex = [r["meta"].get("extra") or {} for r in v]
        L.append(f"| {k} | {len(v)} | {np.median(hv):.4g} | {hv.min():.4g} | {hv.max():.4g} | {np.median([r['meta']['nfe'] for r in v]):.0f} | "
                 f"{np.median([e.get('n_scout') or 0 for e in ex]):.0f} | {np.median([e.get('n_refined') or 0 for e in ex]):.0f} |")
    open(os.path.join(OUT, "T7_LIMIT_STUDY.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


def ablation():
    scen = sorted(d for d in os.listdir(f"{RES}/mainA1") if os.path.isdir(f"{RES}/mainA1/{d}"))
    variants = {"v1b (L100, 2307 FE, arşiv)": "mainB", "A2 (L10, 2307 FE)": "mainA2", "A1 (L100, v2 bütçe)": "mainA1", "v2 (L10, v2 bütçe)": "mainV2"}
    L = ["# AWSM_ABLATION (2×2: Limit × bütçe yapısı; Al/SiC ve sentetik, C–F, 30 tohum)\n",
         "*Üretici: `03_Code_Execution/analyze_ablation.py`. Her senaryoda tüm AWSM varyantları, NSGA-II ve MOEA/D ortak P*-normalizasyonuyla (HV birincil tanım r=(1,1; −0,1)) karşılaştırılır; "
         "değerler: varyantın HV medyanının NSGA-II HV medyanına oranı. Lévy olayı ve inceltme ağırlığı kayıtlı meta'dan okunur.*\n",
         "| Senaryo | " + " | ".join(variants) + " | MOEA/D |\n|---|" + "---|" * (len(variants) + 1)]
    ratios = {k: [] for k in variants}
    rows_scout = {k: [] for k in variants}
    for s in scen:
        runs = []
        for lab, camp in variants.items():
            for r in AC.load_scenario(camp, s):
                if r["alg"] == "awsm":
                    r = dict(r)
                    r["alg"] = lab
                    runs.append(r)
        for r in AC.load_scenario("main", s):
            if r["alg"] == "nsga2":
                runs.append(r)
        for r in AC.load_scenario("mainM", s):
            if r["alg"] == "moead":
                runs.append(r)
        rows, inf = AC.scenario_metrics(runs)
        med = {}
        for alg in list(variants) + ["nsga2", "moead"]:
            v = [x["HV"] for x in rows if x["alg"] == alg]
            med[alg] = float(np.median(v)) if v else float("nan")
        for lab in variants:
            ratios[lab].append(med[lab] / med["nsga2"])
        L.append(f"| {s} | " + " | ".join(f"{med[lab] / med['nsga2']:.3f}" for lab in variants) + f" | {med['moead'] / med['nsga2']:.3f} |")
    L.append("| **ortalama oran** | " + " | ".join(f"{np.mean(ratios[lab]):.3f}" for lab in variants) + " | |")
    L.append("\n## Lévy olayı ve inceltme (kayıtlı meta, medyan)\n\n| Varyant | FE medyan | Lévy olayı | inceltme ağırlığı |\n|---|---|---|---|")
    for lab, camp in variants.items():
        nf, sc, rf = [], [], []
        for s in scen:
            for r in AC.load_scenario(camp, s):
                if r["alg"] == "awsm":
                    ex = r["meta"].get("extra") or {}
                    nf.append(r["meta"]["nfe"])
                    sc.append(ex.get("n_scout") or 0)
                    rf.append(ex.get("n_refined") or 0)
        L.append(f"| {lab} | {np.median(nf):.0f} | {np.median(sc):.0f} | {np.median(rf):.0f} |")
    # paired comparison of the factor effects over the six scenarios (Wilcoxon, exact)
    L.append("\n## Etki karşılaştırması (altı senaryoda oranlar, Wilcoxon işaretli sıra, tam, iki yönlü; n=6'da en küçük p = 0,031)\n\n| Karşılaştırma | ortalama oran farkı | p |\n|---|---|---|")
    keys = list(variants)
    for a_, b_, name in ((2, 0, "A1 − v1b (bütçe yapısı, Limit=100)"), (3, 1, "v2 − A2 (bütçe yapısı, Limit=10)"),
                         (1, 0, "A2 − v1b (Limit 100→10, 2307 FE)"), (3, 2, "v2 − A1 (Limit 100→10, v2 bütçe)")):
        x, y = np.array(ratios[keys[a_]]), np.array(ratios[keys[b_]])
        from scipy import stats
        p = stats.wilcoxon(x, y, method="exact").pvalue
        L.append(f"| {name} | {np.mean(x - y):+.4f} | {p:.3g} |")
    open(os.path.join(OUT, "AWSM_ABLATION.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="t7limit,ablation")
    a = ap.parse_args()
    if "t7limit" in a.only:
        t7limit()
    if "ablation" in a.only:
        ablation()
    print("done")
