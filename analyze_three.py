"""Three-algorithm analysis from stored runs (no optimisation is repeated).
AWSM-ABC-Levy (<base>V2 = corrected AWSM, default; <base>B = campaign v1b with fe_per_run = B//13), NSGA-II (<base>),
MOEA/D (<base>M). Primary inferential family (K14, 2026-10-26): the 12 main scenarios (C-F and C-P).

Usage: python -I -X utf8 analyze_three.py [--awsm V2|B]
Writes results/three_algorithms_<awsm>/metrics.csv, ranks.json and 04_Results/CAMPAIGN_three_algorithms_<awsm>_METRICS.md
Primary HV: P*-normalised, r = (1.1, -0.1); sensitivity HV_raw (raw plane) and HV_r10 (normalised, r = (1, 0))."""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_campaign as AC  # noqa: E402
import stat_tests as S  # noqa: E402

ALGS = ("awsm", "nsga2", "moead")
NAMES = {"awsm": "AWSM-ABC-Levy", "nsga2": "NSGA-II", "moead": "MOEA/D"}
METRICS = (("HV", True), ("IGD", False), ("SP", False))
SENS = (("HV_raw", True), ("HV_r10", True))
GROUPS = ("main", "restricted05", "kappa2")
Q05_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}   # Demsar (2006), Table 5(a), p. 12


def collect(base, awsm_suffix):
    src = {"awsm": awsm_suffix, "nsga2": "", "moead": "M"}
    scen = sorted(d for d in os.listdir(os.path.join(AC.RESULTS, base + "M")) if os.path.isdir(os.path.join(AC.RESULTS, base + "M", d)))
    out = {}
    for s in scen:
        runs = []
        for alg in ALGS:
            runs += [r for r in AC.load_scenario(base + src[alg], s) if r["alg"] == alg]
        rows, inf = AC.scenario_metrics(runs)
        for r in rows:
            r["scenario"] = s
            r["group"] = base
        out[s] = (rows, inf)
    return out


def medians(data, m):
    med = {}
    for s, (rows, _) in data.items():
        for alg in ALGS:
            v = np.array([r[m] for r in rows if r["alg"] == alg], float)
            v = v[np.isfinite(v)]
            if len(v):
                med[(s, alg)] = float(np.median(v))
    return med


def friedman_block(data, m, hb):
    main_s = [s for s, (rows, _) in data.items() if rows[0]["group"] == "main"]
    med = medians(data, m)
    blocks = np.array([[med[(s, a)] for a in ALGS] for s in main_s])
    q, p = S.friedman(blocks)
    ranks = np.apply_along_axis(stats.rankdata, 1, -blocks if hb else blocks)   # rank 1 = best
    return main_s, blocks, ranks, q, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--awsm", default="V2", choices=["V2", "B"])
    a = ap.parse_args()
    data = {}
    for base in GROUPS:
        if os.path.isdir(os.path.join(AC.RESULTS, base + "M")) and os.path.isdir(os.path.join(AC.RESULTS, base + a.awsm)):
            data.update(collect(base, a.awsm))
    all_rows = [r for rows, _ in data.values() for r in rows]
    tag = f"three_algorithms_{a.awsm}"
    outdir = os.path.join(AC.RESULTS, tag)
    os.makedirs(outdir, exist_ok=True)
    fields = ["group", "scenario", "alg", "seed", "B", "HV", "HV_raw", "HV_r10", "IGD", "SP", "front_size", "nfe", "wall_s"]
    with open(os.path.join(outdir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r[k] for k in fields} for r in all_rows])
    pairs = list(itertools.combinations(ALGS, 2))
    awsm_txt = ("düzeltilmiş AWSM (v2: boşluk ağırlık-en-iyisinde, çapa 3000 FE, Limit=10)" if a.awsm == "V2"
                else "ilk AWSM (v1b: fe_per_run = B//13, 29 991 FE, arşiv boşluk ölçütü, Limit=100)")
    L = [f"# CAMPAIGN_three_algorithms_{a.awsm}_METRICS\n",
         f"*Üretici: `03_Code_Execution/analyze_three.py --awsm {a.awsm}`; girdi: kayıtlı koşular. AWSM = {awsm_txt}; NSGA-II ve MOEA/D = 30 000 FE. "
         "Birincil çıkarım ailesi: 12 ana senaryo (C–F ve C–P). HV (birincil): P* aralığına normalize, r=(1,1; −0,1); duyarlılık: HV_raw, HV_r10. "
         "Metrik tanımları `analyze_campaign.py` başlığında.*\n"]
    ranks_out = {}
    for m, hb in METRICS + SENS:
        L.append(f"## {m} ({'yüksek iyi' if hb else 'düşük iyi'})" + (" — duyarlılık" if (m, hb) in SENS else "") + "\n")
        if (m, hb) not in SENS:
            L.append("| Grup | Senaryo | Alg. | n | Min | Medyan | Ort. | Maks | Std |\n|---|---|---|---|---|---|---|---|---|")
            for s, (rows, inf) in data.items():
                for alg in ALGS:
                    v = np.array([r[m] for r in rows if r["alg"] == alg], float)
                    v = v[np.isfinite(v)]
                    if len(v) == 0:
                        continue
                    d = S.describe(v)
                    L.append(f"| {rows[0]['group']} | {s} | {NAMES[alg]} | {len(v)} | {d['min']:.5g} | {d['median']:.5g} | {d['mean']:.5g} | {d['max']:.5g} | {d['std']:.3g} |")
            recs = []
            for s, (rows, inf) in data.items():
                for x_, y_ in pairs:
                    x = np.array([r[m] for r in rows if r["alg"] == x_], float)
                    y = np.array([r[m] for r in rows if r["alg"] == y_], float)
                    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
                    if len(x) < 2 or len(y) < 2:
                        continue
                    u, p = S.mann_whitney(x, y)
                    recs.append((s, x_, y_, p, S.a12(x, y)))
            adj = np.empty(len(recs))
            grp = [data[r[0]][0][0]["group"] for r in recs]
            for g_ in set(grp):
                ix = [k for k, gg in enumerate(grp) if gg == g_]
                adj[ix] = S.holm([recs[k][3] for k in ix])
            L.append(f"\n### {m}: ikili Mann–Whitney (iki yönlü); Holm ailesi grup başına (birincil: 12 ana senaryo × 3 çift = 36; restricted05 ve kappa2 ayrı ikincil aileler). "
                     f"A12 > 0.5: ilk algoritmanın metrik değeri daha büyük ({'iyi' if hb else 'kötü'}).\n")
            L.append("| Senaryo | Çift (1 vs 2) | p | p (Holm) | A12 (1 > 2) |\n|---|---|---|---|---|")
            for (s, x_, y_, p, a12), pa in zip(recs, adj):
                L.append(f"| {s} | {NAMES[x_]} vs {NAMES[y_]} | {p:.3g} | {pa:.3g} | {a12:.3f} |")
        main_s, blocks, ranks, q, p = friedman_block(data, m, hb)
        N, k = blocks.shape
        cd = Q05_NEMENYI[k] * np.sqrt(k * (k + 1) / (6.0 * N))
        mr = ranks.mean(axis=0)
        ranks_out[m] = {"mean_rank": {NAMES[al]: float(mr[i]) for i, al in enumerate(ALGS)}, "chi2": q, "p": p, "CD_0.05": float(cd), "N": N}
        L.append(f"\n**{m}: Friedman** (blok = {N} ana senaryo, k = {k}): χ² = {q:.4g}, p = {p:.3g}; ortalama sıra (1 = en iyi): "
                 + ", ".join(f"{NAMES[al]} {mr[i]:.2f}" for i, al in enumerate(ALGS))
                 + f". Nemenyi kritik fark (α=0,05; q={Q05_NEMENYI[k]} Demšar 2006 Tablo 5a): CD = {cd:.3f}.")
        post = []
        for x_, y_ in pairs:
            ia, ib = ALGS.index(x_), ALGS.index(y_)
            res = stats.wilcoxon(blocks[:, ia], blocks[:, ib], alternative="two-sided", method="exact")
            post.append((x_, y_, float(res.pvalue), float(abs(mr[ia] - mr[ib]))))
        padj = S.holm([p_[2] for p_ in post])
        L.append("\n| Post-hoc çift | Wilcoxon işaretli sıra (tam) p | p (Holm, 3 çift) | ortalama sıra farkı | Nemenyi: fark > CD |\n|---|---|---|---|---|")
        for (x_, y_, p_, dr), pa in zip(post, padj):
            L.append(f"| {NAMES[x_]} vs {NAMES[y_]} | {p_:.3g} | {pa:.3g} | {dr:.2f} | {'evet' if dr > cd else 'hayır'} |")
        L.append("")
    path = os.path.join(AC.OUT, f"CAMPAIGN_{tag}_METRICS.md")
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    json.dump(ranks_out, open(os.path.join(outdir, "ranks.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(path)


if __name__ == "__main__":
    main()
