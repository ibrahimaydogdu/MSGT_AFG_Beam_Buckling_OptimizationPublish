"""Campaign runner: one task = one independent run (scenario x algorithm x seed).

Every run is stored separately and permanently in results/<campaign>/<scenario>/<alg>_<seed>.npz
(front with designs, per-evaluation objectives, run record) so that later changes to metrics,
reference points, normalisation or plots never require repeating the optimisation. A task whose
file already exists with the same configuration hash is skipped (use --force to redo).

Usage:  python -I -X utf8 run_campaign.py --campaign main --workers 7
        campaigns: t7 (tuning, seeds 2001-2005), main, restricted05, kappa2, density;
        mainB, restricted05B, kappa2B, densityB = AWSM-only reruns with fe_per_run = B//13 (original results kept)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results")

SEEDS = list(range(1001, 1031))
T7_SEEDS = list(range(2001, 2006))
AWSM_V2_LIMIT = 10   # fixed after the T7 limit study (Limit in {10, 15, 25, 100}; Round 27)
SRC = ["bspline.py", "materials.py", "msgt_ritz.py", "section_model.py", "objectives.py", "optimizers.py", "awsm.py"]


def code_hash():
    h = hashlib.sha256()
    for f in SRC:
        h.update(open(os.path.join(HERE, f), "rb").read())
    return h.hexdigest()[:12]


def cfg_hash(task):
    keys = ["pair", "rho_ratio", "mu", "bc", "kappa_min", "h_min", "alg", "seed", "B", "fe_per_run", "N", "nq"]
    d = {k: task[k] for k in keys}
    if task.get("limit", 100) != 100:
        d["limit"] = task["limit"]
    if task.get("anchor_fe") is not None:
        d["anchor_fe"] = task["anchor_fe"]
    if task.get("gap_mode", "archive") != "archive":
        d["gap_mode"] = task["gap_mode"]
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def scen_id(t):
    s = f"{t['pair']}_{t['bc']}_mu{t['mu']}"
    if t["pair"] == "SYN" and t["rho_ratio"] != 1.0:
        s += f"_rho{t['rho_ratio']}"
    if t["h_min"] != 0.2:
        s += f"_hmin{t['h_min']}"
    if t["kappa_min"] != 1.0:
        s += f"_kappa{t['kappa_min']}"
    if t["campaign"] == "t7limit":
        s += f"_limit{t['limit']}"
    return s


def make_task(campaign, pair, mu, bc, alg, seed, rho_ratio=1.0, kappa_min=1.0, h_min=0.2, B=30000,
              fe_per_run=1500, N=12, nq=24, limit=100, anchor_fe=None, gap_mode="archive"):
    t = dict(campaign=campaign, pair=pair, rho_ratio=rho_ratio, mu=mu, bc=bc, kappa_min=kappa_min,
             h_min=h_min, alg=alg, seed=seed, B=B, fe_per_run=fe_per_run, N=N, nq=nq,
             limit=limit, anchor_fe=anchor_fe, gap_mode=gap_mode)
    t["scenario"] = scen_id(t)
    t["cfg_hash"] = cfg_hash(t)
    return t


def build_tasks(campaign):
    if campaign.endswith("M") and campaign[:-1] in ("main", "restricted05", "kappa2", "density"):
        # third algorithm (MOEA/D-Tchebycheff, 2026-10-26): same scenarios and seeds as the base campaign
        base = campaign[:-1]
        out = []
        for t in build_tasks(base):
            if t["alg"] == "nsga2":
                out.append(make_task(campaign, t["pair"], t["mu"], t["bc"], "moead", t["seed"], rho_ratio=t["rho_ratio"],
                                     kappa_min=t["kappa_min"], h_min=t["h_min"], B=t["B"], fe_per_run=t["fe_per_run"],
                                     N=t["N"], nq=t["nq"]))
        return out
    if campaign in ("mainA1", "mainA2"):
        # AWSM ablation (2x2 with v1b and v2), C-F scenarios of the main campaign only:
        #   A1: v2 with Limit = 100            (isolates the Levy scout phase inside the v2 budget structure)
        #   A2: weights/anchors 2307 FE each, Limit = 10, weight-best gaps (no budget left for refinement)
        #       (isolates the Limit inside the v1b budget structure)
        out = []
        for t in build_tasks("main"):
            if t["alg"] == "awsm" and t["bc"] == "CF":
                if campaign == "mainA1":
                    out.append(make_task(campaign, t["pair"], t["mu"], t["bc"], "awsm", t["seed"], B=t["B"], fe_per_run=1500,
                                         N=t["N"], nq=t["nq"], limit=100, anchor_fe=3000, gap_mode="weight_best"))
                else:
                    out.append(make_task(campaign, t["pair"], t["mu"], t["bc"], "awsm", t["seed"], B=t["B"], fe_per_run=t["B"] // 13,
                                         N=t["N"], nq=t["nq"], limit=10, anchor_fe=t["B"] // 13, gap_mode="weight_best"))
        return out
    if campaign.endswith("V2") and campaign[:-2] in ("main", "restricted05", "kappa2", "density"):
        # corrected AWSM (v2, 2026-10-26): gap on per-weight best designs, anchors 3000 FE, Limit from the T7 limit study
        base = campaign[:-2]
        out = []
        for t in build_tasks(base):
            if t["alg"] == "awsm":
                out.append(make_task(campaign, t["pair"], t["mu"], t["bc"], "awsm", t["seed"], rho_ratio=t["rho_ratio"],
                                     kappa_min=t["kappa_min"], h_min=t["h_min"], B=t["B"], fe_per_run=1500,
                                     N=t["N"], nq=t["nq"], limit=AWSM_V2_LIMIT, anchor_fe=3000, gap_mode="weight_best"))
        return out
    if campaign.endswith("B") and campaign != "B":
        # option (b), 2026-10-26: AWSM only, fe_per_run = B // 13 so that 2 anchors + 11 weights use the
        # whole budget. NSGA-II runs are NOT repeated: they are read from the original campaign.
        base = campaign[:-1]
        out = []
        for t in build_tasks(base):
            if t["alg"] == "awsm":
                out.append(make_task(campaign, t["pair"], t["mu"], t["bc"], "awsm", t["seed"], rho_ratio=t["rho_ratio"],
                                     kappa_min=t["kappa_min"], h_min=t["h_min"], B=t["B"], fe_per_run=t["B"] // 13,
                                     N=t["N"], nq=t["nq"]))
        return out
    tasks = []
    if campaign == "t7":
        for B in (15000, 30000, 60000):
            for alg in ("awsm", "nsga2"):
                for seed in T7_SEEDS:
                    tasks.append(make_task("t7", "ALSIC", 0.1, "CF", alg, seed, B=B, fe_per_run=B // 20))
    elif campaign == "t7limit":   # Limit study for AWSM v2 (tuning seeds, separate from the campaign seeds)
        for lim in (10, 15, 25, 100):
            for seed in T7_SEEDS:
                tasks.append(make_task("t7limit", "ALSIC", 0.1, "CF", "awsm", seed, limit=lim, anchor_fe=3000, gap_mode="weight_best"))
    elif campaign == "main":
        for pair in ("SYN", "ALSIC"):
            for bc in ("CF", "CP"):
                for mu in (0.0, 0.1, 0.2):
                    for alg in ("awsm", "nsga2"):
                        for seed in SEEDS:
                            tasks.append(make_task("main", pair, mu, bc, alg, seed))
    elif campaign == "restricted05":
        for pair in ("SYN", "ALSIC"):
            for alg in ("awsm", "nsga2"):
                for seed in SEEDS:
                    tasks.append(make_task("restricted05", pair, 0.5, "CF", alg, seed, h_min=0.5))
    elif campaign == "kappa2":
        for alg in ("awsm", "nsga2"):
            for seed in SEEDS:
                tasks.append(make_task("kappa2", "ALSIC", 0.2, "CF", alg, seed, kappa_min=2.0, h_min=0.4))
    elif campaign == "density":   # parametric density ratio of the synthetic pair (extra, labelled)
        for rho in (0.5, 2.0):
            for mu in (0.0, 0.1, 0.2):
                for alg in ("awsm", "nsga2"):
                    for seed in SEEDS:
                        tasks.append(make_task("density", "SYN", mu, "CF", alg, seed, rho_ratio=rho))
    else:
        raise SystemExit("unknown campaign")
    return tasks


def build_problem(t):
    from materials import AL_SIC, MaterialPair, Phase
    from section_model import ColumnModel
    from objectives import DesignProblem
    if t["pair"] == "ALSIC":
        pair, h0, l_mode = AL_SIC, 1.0e-6, "const"
    else:
        # synthetic pair: E_L = 2 E_R, nu = 0.3, rho_L = 1, rho_R = rho_ratio (dimensionless units, h0 = 1)
        pair = MaterialPair(Phase("syn-L", 2.0, 0.3, 1.0), Phase("syn-R", 1.0, 0.3, t["rho_ratio"]), "synthetic")
        h0, l_mode = 1.0, "mixture"
    model = ColumnModel(pair, mu=t["mu"], h0=h0, bc=t["bc"], bc_type="BC1", N=t["N"], nq=t["nq"],
                        l_mode=l_mode, h_min=t["h_min"])
    return DesignProblem(model, kappa_min=t["kappa_min"], record=False)


def out_path(t):
    d = os.path.join(RESULTS, t["campaign"], t["scenario"])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{t['alg']}_B{t['B']}_{t['seed']}.npz" if t["campaign"] == "t7" else f"{t['alg']}_{t['seed']}.npz")


def run_task(t, force=False):
    from awsm import awsm_search, feasible_front
    from optimizers import nsga2
    path = out_path(t)
    if os.path.exists(path) and not force:
        try:
            with np.load(path, allow_pickle=False) as z:
                if json.loads(str(z["meta"]))["cfg_hash"] == t["cfg_hash"]:
                    return {"path": path, "skipped": True}
        except Exception:
            pass
    problem = build_problem(t)
    t0 = time.time()
    extra = {}
    if t["alg"] == "awsm":
        out = awsm_search(problem, t["seed"], B=t["B"], fe_per_run=t["fe_per_run"], limit=t.get("limit", 100),
                          anchor_fe=t.get("anchor_fe"), gap_mode=t.get("gap_mode", "archive"))
        arch = out["archive"]
        rows = arch[:, :-1] if len(arch) else np.empty((0, 21))
        tags = arch[:, -1] if len(arch) else np.empty(0)
        nfe = out["fe"]
        extra = {"runs": out["runs"], "anchors": out.get("anchors"), "failed": out.get("failed"),
                 "n_scout": out.get("n_scout"), "n_refined": out.get("n_refined")}
    elif t["alg"] == "moead":
        from optimizers import moead
        problem.reset()
        arch = []
        moead(problem, t["seed"], t["B"], archive=arch)
        rows = np.array(arch)
        tags = np.full(len(rows), np.nan)
        nfe = problem.nfe
    else:
        problem.reset()
        arch = []
        nsga2(problem, t["seed"], t["B"], archive=arch)
        rows = np.array(arch)
        tags = np.full(len(rows), np.nan)
        nfe = problem.nfe
    wall = time.time() - t0
    full = np.column_stack([rows, tags]) if len(rows) else np.empty((0, 22))
    front = feasible_front(full) if len(full) else np.empty((0, 22))
    meta = dict(t, nfe=int(nfe), wall_s=wall, code_hash=code_hash(), n_feasible=int(np.sum(rows[:, 2] == 0)) if len(rows) else 0,
                numpy=np.__version__, **{"extra": extra})
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, front=front, evals=rows[:, :3].astype(np.float64) if len(rows) else np.empty((0, 3)),
                        meta=json.dumps(meta, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)))
    os.replace(tmp, path)
    return {"path": path, "skipped": False, "wall": wall, "nfe": nfe, "front": len(front)}


def _worker(t):
    try:
        return run_task(t)
    except Exception:
        return {"path": out_path(t), "error": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    tasks = build_tasks(a.campaign)
    logdir = os.path.join(RESULTS, a.campaign)
    os.makedirs(logdir, exist_ok=True)
    log = open(os.path.join(logdir, "progress.log"), "a", encoding="utf-8")
    print(f"{a.campaign}: {len(tasks)} tasks, {a.workers} workers, code {code_hash()}", file=log, flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            done += 1
            t = futs[f]
            msg = f"[{done}/{len(tasks)}] {t['scenario']} {t['alg']} {t['seed']}: " + (
                "ERROR\n" + r["error"] if "error" in r else ("skipped" if r.get("skipped") else
                                                              f"nfe={r['nfe']} front={r['front']} {r['wall']:.1f}s"))
            print(msg, file=log, flush=True)
    print(f"finished {done} tasks in {time.time() - t0:.0f} s", file=log, flush=True)


if __name__ == "__main__":
    main()
