"""Density-ratio block of the synthetic pair (C-F): rho_R/rho_L in {0.5, 1 (main campaign), 2.0}.
Algorithms available in this block: AWSM (corrected, campaign densityV2 / mainV2) and NSGA-II (density / main); MOEA/D was not run here.
Writes 04_Results/DENSITY_BLOCK.md and a figure-free summary; every number is read from stored runs.
Usage: python -I -X utf8 analyze_density.py"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_campaign as AC  # noqa: E402
import post_optimal as PO  # noqa: E402

RES = AC.RESULTS
OUT = os.path.join(AC.OUT, "DENSITY_BLOCK.md")


def front_rows(camp, scen, alg):
    parts, meta = [], None
    for f in sorted(glob.glob(os.path.join(RES, camp, scen, "*.npz"))):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        if m["alg"] != alg or len(z["front"]) == 0:
            continue
        meta = meta or m
        fr = z["front"]
        parts.append(np.column_stack([fr[:, 0], fr[:, 1], fr[:, 3:21], np.zeros(len(fr)), np.full(len(fr), m["seed"])]))
    return (np.vstack(parts) if parts else None), meta


def main():
    L = ["# DENSITY_BLOCK (sentetik çift, C–F: ρ_R/ρ_L ∈ {0.5, 1, 2.0})\n",
         "*Üretici: `03_Code_Execution/analyze_density.py`; kayıtlı koşular. ρ_R/ρ_L = 1 satırları ana kampanyadandır (`mainV2`, `main`); 0.5 ve 2.0 `densityV2` ve `density` kampanyalarındandır. "
         "Bu blokta MOEA/D koşulmadı. P* = iki algoritmanın birleşik cephesi; diz noktası ve V̄ (ortalama V_r) bu birleşik cepheden okunur. f1bar = m/m0, m0 = ρ_L b0 h0 L (ρ_L = 1 sabit), "
         "bu nedenle f1bar ve HV farklı ρ_R/ρ_L değerleri arasında doğrudan karşılaştırılmaz; λ ve V̄ karşılaştırılabilir.*\n",
         "| l/h0 | ρ_R/ρ_L | |P*| | diz noktası f1bar | diz noktası λ | diz noktası V̄ | HV oranı AWSM/NSGA-II (ortak P* normalizasyonu) |\n|---|---|---|---|---|---|---|"]
    for mu in (0.0, 0.1, 0.2):
        for rho in (0.5, 1.0, 2.0):
            if rho == 1.0:
                s = f"SYN_CF_mu{mu}"
                ca, cn = "mainV2", "main"
            else:
                s = f"SYN_CF_mu{mu}_rho{rho}"
                ca, cn = "densityV2", "density"
            ra, meta = front_rows(ca, s, "awsm")
            rn, _ = front_rows(cn, s, "nsga2")
            allrows = np.vstack([ra, rn])
            P = PO.pareto_union(allrows)
            knee = P[dict(PO.select_points(P))["knee"]]
            m = PO.make_model(meta, N=12)
            Vbar = float(np.sum(m.prof.w * (m.prof.B @ knee[2 + 12:2 + 18])))
            runs = []
            for alg, camp in (("awsm", ca), ("nsga2", cn)):
                for r in AC.load_scenario(camp, s):
                    if r["alg"] == alg:
                        runs.append(r)
            rows, _ = AC.scenario_metrics(runs)
            hv = {a: np.median([x["HV"] for x in rows if x["alg"] == a]) for a in ("awsm", "nsga2")}
            L.append(f"| {mu} | {rho} | {len(P)} | {knee[0]:.4g} | {knee[1]:.4g} | {Vbar:.4f} | {hv['awsm'] / hv['nsga2']:.3f} |")
    open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
