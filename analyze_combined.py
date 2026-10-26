"""Combine stored runs from different campaigns without repeating any optimisation:
AWSM runs from <base>B (option b, fe_per_run = B//13) with NSGA-II runs from <base> (original).
Usage: python -I -X utf8 analyze_combined.py --base main
Writes results/<base>B/metrics_combined.csv and 04_Results/CAMPAIGN_<base>_awsmB_METRICS.md"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_campaign as AC  # noqa: E402
import stat_tests as S  # noqa: E402


def run(base):
    awsm_c = base + "B"
    scen = sorted(d for d in os.listdir(os.path.join(AC.RESULTS, awsm_c)) if os.path.isdir(os.path.join(AC.RESULTS, awsm_c, d)))
    all_rows, comp, info = [], {}, {}
    for s in scen:
        runs = [r for r in AC.load_scenario(awsm_c, s) if r["alg"] == "awsm"]
        runs += [r for r in AC.load_scenario(base, s) if r["alg"] == "nsga2"]
        if not runs:
            continue
        rows, inf = AC.scenario_metrics(runs)
        for r in rows:
            r["scenario"] = s
        all_rows += rows
        info[s] = inf
        for m, hb in (("HV", True), ("IGD", False), ("SP", False)):
            c = AC.compare(rows, m, hb)
            if c:
                comp[(s, m)] = c
    for m in ("HV", "IGD", "SP"):
        keys = [k for k in comp if k[1] == m]
        adj = S.holm([comp[k]["p"] for k in keys])
        for k, pa in zip(keys, adj):
            comp[k]["p_holm"] = float(pa)
    fried = {}
    for m in ("HV", "IGD", "SP"):
        blocks = [[comp[(s, m)]["desc_awsm"]["median"], comp[(s, m)]["desc_nsga2"]["median"]] for s in info if (s, m) in comp]
        if len(blocks) >= 2:
            fried[m] = S.friedman(np.array(blocks))
    with open(os.path.join(AC.RESULTS, awsm_c, "metrics_combined.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "alg", "seed", "B", "HV", "IGD", "SP", "front_size", "nfe", "wall_s"])
        w.writeheader()
        w.writerows(all_rows)
    name = f"{base}_awsmB"
    path = AC.write_report(name, all_rows, comp, fried, info)
    return path, all_rows, comp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    path, rows, comp = run(a.base)
    print(path)
    nfe = [r["nfe"] for r in rows if r["alg"] == "awsm"]
    print("AWSM FE used: median", np.median(nfe), "min", min(nfe), "max", max(nfe))
