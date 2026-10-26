"""Adaptive weight refinement (AWSM-type), Section 3.2, Eqs. (75)-(78), with a total
evaluation cap B (Eq. 88): anchors + all sub-problems share the budget.

Design choices NOT fixed by the manuscript (flagged, tunable, to be settled in the T7 study):
  * fe_per_run : evaluations given to every anchor run and every weight sub-problem (default 1500);
  * refined weights are warm-started from the best design of the parent weight with the smaller
    w1 (initial weights are warm-started sequentially, as in Section 3.2 step 6);
  * when the remaining budget cannot fund all new weights of an iteration, weights belonging to the
    largest gaps are solved first and the loop stops when the budget is exhausted.
"""
import numpy as np

from optimizers import abc_levy
from objectives import deb_better

W_INIT = np.round(np.linspace(0.0, 1.0, 11), 4)  # Eq. (76)
D_TH = 0.15
N_ITER = 3
MIN_RUN_FE = 100


def nondominated(F):
    """Indices of non-dominated rows of F (both columns minimised)."""
    n = len(F)
    keep = np.ones(n, dtype=bool)
    order = np.lexsort((F[:, 1], F[:, 0]))
    best_f2 = np.inf
    for idx in order:
        if F[idx, 1] < best_f2:
            best_f2 = F[idx, 1]
        else:
            keep[idx] = False
    return np.nonzero(keep)[0]


def feasible_front(rows):
    """rows: array [f1, lam, cv, u..., w]; returns the non-dominated feasible set sorted by f1."""
    rows = np.asarray(rows)
    ok = rows[:, 2] == 0.0
    ok &= np.isfinite(rows[:, 0])
    rows = rows[ok]
    if len(rows) == 0:
        return rows
    F = np.column_stack([rows[:, 0], -rows[:, 1]])
    idx = nondominated(F)
    out = rows[idx]
    return out[np.argsort(out[:, 0])]


def awsm_search(problem, seed, B=30000, fe_per_run=1500, n_food=50, limit=100,
                d_th=D_TH, n_iter=N_ITER, w_init=W_INIT, anchor_fe=None, gap_mode="archive"):
    """Run the whole AWSM-type search once (one independent run, one seed).

    anchor_fe : evaluations of each of the two anchor runs (default = fe_per_run).
    gap_mode  : 'archive'     gaps measured on the non-dominated set of ALL evaluations (campaign v1, Round 20-25)
                'weight_best' gaps measured on the non-dominated set of the per-weight best designs, in the
                              anchor-normalised space of Eqs. (75)-(77) (manuscript Section 3.2 step 3; v2).
    Returns dict(front, archive, fe, runs, anchors, n_scout, n_refined)."""
    rng = np.random.default_rng(seed)
    problem.reset()
    rows = []   # [f1, lam, cv, u..., w1_tag]
    runs = []
    used = 0
    n_scout = 0
    a_fe = fe_per_run if anchor_fe is None else anchor_fe

    def budget_left():
        return B - used

    def solve(scalar, w_tag, init, label, fe_cap):
        nonlocal used, n_scout
        fe = min(fe_cap, budget_left())
        arch = []
        res = abc_levy(problem, scalar, rng, fe, n_food=n_food, limit=limit, init=init, archive=arch)
        used += res["fe"]
        n_scout += res.get("n_scout", 0)
        for r in arch:
            rows.append(np.concatenate([r, [w_tag]]))
        runs.append({"label": label, "w1": w_tag, "fe": res["fe"], "best_f1": res["best_f1"],
                     "best_lam": res["best_lam"], "best_cv": res["best_cv"], "n_scout": res.get("n_scout", 0)})
        return res

    # anchors (Section 3.2): mass-only and buckling-only runs
    a1 = solve(lambda f1, lam: f1, 1.0, None, "anchor_mass", a_fe)
    a2 = solve(lambda f1, lam: -lam, 0.0, None, "anchor_lambda", a_fe)
    if a1["best_cv"] > 0 or a2["best_cv"] > 0:
        return {"front": feasible_front(rows) if rows else np.empty((0, 0)), "archive": np.array(rows),
                "fe": used, "runs": runs, "failed": "anchor infeasible", "n_scout": n_scout, "n_refined": 0}
    f1_min, lam_at_f1min = a1["best_f1"], a1["best_lam"]       # f2^min
    lam_max, f1_at_lammax = a2["best_lam"], a2["best_f1"]      # f1^max
    s1 = max(f1_at_lammax - f1_min, 1e-12)
    s2 = max(lam_max - lam_at_f1min, 1e-12)

    def make_scalar(w1):
        w2 = 1.0 - w1
        return lambda f1, lam: w1 * (f1 - f1_min) / s1 + w2 * (lam_max - lam) / s2   # Eq. (75)

    solved = {}    # w1 -> best u
    wbest = {}     # w1 -> (f1, lam) of the weight's best design (feasible only)
    prev_best = None
    for w1 in w_init:
        if budget_left() < MIN_RUN_FE:
            break
        res = solve(make_scalar(float(w1)), float(w1), None if prev_best is None else [prev_best], f"w={w1}", fe_per_run)
        prev_best = res["best_u"]
        solved[float(w1)] = res["best_u"]
        if res["best_cv"] == 0.0:
            wbest[float(w1)] = (res["best_f1"], res["best_lam"])

    n_refined = 0
    for it in range(n_iter):
        if budget_left() < MIN_RUN_FE:
            break
        if gap_mode == "weight_best":
            ws = sorted(wbest)
            pts = np.array([[wbest[w][0], wbest[w][1], w] for w in ws])
            if len(pts) < 2:
                break
            keep = nondominated(np.column_stack([pts[:, 0], -pts[:, 1]]))
            front = pts[keep]
            front = front[np.argsort(front[:, 0])]
            F1 = (front[:, 0] - f1_min) / s1                       # Eq. (77) normalisation (anchor based)
            F2 = (lam_max - front[:, 1]) / s2
            tags = front[:, 2]
        else:
            front = feasible_front(rows)
            if len(front) < 2:
                break
            F1 = (front[:, 0] - front[:, 0].min()) / max(np.ptp(front[:, 0]), 1e-12)
            F2 = (front[:, 1].max() - front[:, 1]) / max(np.ptp(front[:, 1]), 1e-12)
            tags = front[:, -1]
        d = np.hypot(np.diff(F1), np.diff(F2))
        cand = []
        for k in range(len(d)):
            if d[k] > d_th and tags[k] != tags[k + 1]:
                cand.append((d[k], round(0.5 * (tags[k] + tags[k + 1]), 4), float(min(tags[k], tags[k + 1]))))
        if not cand:
            break
        cand.sort(key=lambda t: -t[0])
        seen = set(solved)
        for _, w_new, parent in cand:
            if w_new in seen or budget_left() < MIN_RUN_FE:
                continue
            seen.add(w_new)
            init = [solved[parent]] if parent in solved else None
            res = solve(make_scalar(w_new), w_new, init, f"refine{it + 1}_w={w_new}", fe_per_run)
            solved[w_new] = res["best_u"]
            n_refined += 1
            if res["best_cv"] == 0.0:
                wbest[w_new] = (res["best_f1"], res["best_lam"])
    return {"front": feasible_front(rows), "archive": np.array(rows), "fe": used, "runs": runs, "n_scout": n_scout,
            "n_refined": n_refined,
            "anchors": {"f1_min": f1_min, "lam_at_f1_min": lam_at_f1min, "lam_max": lam_max, "f1_at_lam_max": f1_at_lammax}}
