"""Post-optimal analyses on stored optimisation runs (no optimisation is repeated).

Analyses (all direct evaluations with the validated solver; every result is written to results/post/):
  select  : representative designs per scenario (P* = union of the three algorithms' fronts; arc-length
            quantiles {0, .25, .5, .75, 1} of the normalised front plus the knee point)
  A  bc2  : BC1-optimised designs re-evaluated with BC2 (N = 15) and N-convergence
  B  sweep: l/h0 sweep {0, .05, .1, .15, .2, .3, .5, 1} for BC1 and BC2 (> 0.2 or h/l < 1: 'trend only')
  C  cross: CT-optimal designs evaluated with MSGT and MSGT-optimal designs evaluated with CT
  D  abl  : l(x) = l [1 + (r-1) V_r], r in {2, 0.5}, against constant l and constant equal-mean l
  E  bench: uniform (b = h) and linearly tapered benchmark columns (families of direct evaluations)
  F  prof : profiles b, h, V_r, D1/EI, h/l, eps_s and active constraints (representative + per-run knees)
  G  sect : equal-area circle and ellipse for the optimised rectangular profiles (S1, Critic Round 12)
Usage: python -I -X utf8 post_optimal.py [--only select,A,B,...] [--out DIR]
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_campaign as RC  # noqa: E402  (sets single-thread BLAS)
from materials import AL_SIC, MaterialPair, Phase  # noqa: E402
from msgt_ritz import SolverFailure, buckling_load, D_coefficients  # noqa: E402
from section_model import ColumnModel  # noqa: E402
import metrics as M  # noqa: E402

RESULTS = RC.RESULTS
POST = os.path.join(RESULTS, "post")
BASES = ("main", "restricted05", "kappa2")
SRC = {"awsm": "V2", "nsga2": "", "moead": "M"}   # corrected AWSM (v2); "B" = campaign v1b
MU_SWEEP = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
QUANT = (0.0, 0.25, 0.5, 0.75, 1.0)


# ------------------------------------------------------------------ loading
def load_union(base):
    """scenario -> dict(meta, rows) where rows = array [f1, lam, u(18), alg_idx, seed] (feasible fronts)."""
    out = {}
    algs = ("awsm", "nsga2", "moead")
    d0 = os.path.join(RESULTS, base + "M")
    if not os.path.isdir(d0):
        return out
    for s in sorted(os.listdir(d0)):
        parts, meta = [], None
        for ai, alg in enumerate(algs):
            for f in sorted(glob.glob(os.path.join(RESULTS, base + SRC[alg], s, "*.npz"))):
                z = np.load(f, allow_pickle=False)
                m = json.loads(str(z["meta"]))
                if m["alg"] != alg:
                    continue
                fr = z["front"]
                if len(fr) == 0:
                    continue
                meta = meta or m
                parts.append(np.column_stack([fr[:, 0], fr[:, 1], fr[:, 3:21], np.full(len(fr), ai), np.full(len(fr), m["seed"])]))
        if parts:
            out[s] = {"meta": meta, "rows": np.vstack(parts), "base": base}
    return out


def pareto_union(rows):
    """Non-dominated subset (f1 min, lam max) of the union, sorted by f1."""
    idx = np.lexsort((-rows[:, 1], rows[:, 0]))
    keep, best = [], -np.inf
    for i in idx:
        if rows[i, 1] > best:
            keep.append(i)
            best = rows[i, 1]
    return rows[keep]


def select_points(P):
    """Representative designs: arc-length quantiles of the normalised P* and the knee (min distance to ideal)."""
    f1, lam = P[:, 0], P[:, 1]
    x = (f1 - f1.min()) / max(np.ptp(f1), 1e-12)
    y = (lam - lam.min()) / max(np.ptp(lam), 1e-12)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    s /= s[-1] if s[-1] > 0 else 1.0
    sel = []
    for q in QUANT:
        sel.append((f"q{q:.2f}", int(np.argmin(np.abs(s - q)))))
    sel.append(("knee", int(np.argmin(np.hypot(x, 1.0 - y)))))
    return sel


# ------------------------------------------------------------------ model helpers
def make_model(meta, **over):
    t = dict(meta)
    t.update(over)
    pair, h0, l_mode = (AL_SIC, 1.0e-6, "const") if t["pair"] == "ALSIC" else (
        MaterialPair(Phase("syn-L", 2.0, 0.3, 1.0), Phase("syn-R", 1.0, 0.3, t["rho_ratio"]), "synthetic"), 1.0, "mixture")
    l_mode = over.get("l_mode", l_mode)
    return ColumnModel(pair, mu=t["mu"], h0=h0, bc=t["bc"], bc_type=t.get("bc_type", "BC1"), N=t["N"], nq=t["nq"],
                       l_mode=l_mode, l_ratio=over.get("l_ratio", 1.0), h_min=t["h_min"])


def to_X(model, u):
    n = model.n
    return np.concatenate([u[:n] * model.b0, u[n:2 * n] * model.h0, u[2 * n:]])


def lam_of(model, u):
    try:
        return model.objectives(to_X(model, u))[1]
    except (SolverFailure, ValueError):
        return float("nan")


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------ select
def run_select(out):
    sel_all, rows = {}, []
    for base in BASES:
        for s, d in load_union(base).items():
            P = pareto_union(d["rows"])
            pts = select_points(P)
            sel_all[s] = {"meta": d["meta"], "base": base, "P": P, "pts": pts}
            for name, i in pts:
                rows.append([base, s, name, i, len(P), P[i, 0], P[i, 1], int(P[i, 20]), int(P[i, 21])] + list(P[i, 2:20]))
    write_csv(os.path.join(out, "select", "representative_designs.csv"),
              ["base", "scenario", "point", "index_in_Pstar", "Pstar_size", "f1bar", "lam", "alg(0=AWSM,1=NSGA-II,2=MOEAD)", "seed"] + [f"u{k}" for k in range(18)], rows)
    np.savez_compressed(os.path.join(out, "select", "Pstar.npz"), **{s: v["P"] for s, v in sel_all.items()})
    return sel_all


# ------------------------------------------------------------------ A: BC2
def run_bc2(sel, out):
    rows = []
    for s, v in sel.items():
        for name, i in v["pts"]:
            u = v["P"][i, 2:20]
            vals = {}
            for bct in ("BC1", "BC2"):
                for N in (12, 15, 18):
                    vals[(bct, N)] = lam_of(make_model(v["meta"], bc_type=bct, N=N), u)
            rows.append([s, name, v["P"][i, 0], vals[("BC1", 12)], vals[("BC1", 15)], vals[("BC1", 18)],
                         vals[("BC2", 12)], vals[("BC2", 15)], vals[("BC2", 18)], vals[("BC2", 15)] / vals[("BC1", 15)]])
    write_csv(os.path.join(out, "A_bc2", "bc2_reevaluation.csv"),
              ["scenario", "point", "f1bar", "lam_BC1_N12", "lam_BC1_N15", "lam_BC1_N18", "lam_BC2_N12", "lam_BC2_N15", "lam_BC2_N18", "BC2/BC1_N15"], rows)


# ------------------------------------------------------------------ B: parametric sweep
def run_sweep(sel, out):
    rows = []
    for s, v in sel.items():
        if v["meta"]["mu"] not in (0.1, 0.2) or v["base"] != "main":
            continue
        for name, i in v["pts"]:
            u = v["P"][i, 2:20]
            for bct in ("BC1", "BC2"):
                base_ct = lam_of(make_model(v["meta"], bc_type=bct, N=15, mu=0.0), u)
                for mu in MU_SWEEP:
                    m = make_model(v["meta"], bc_type=bct, N=15, mu=mu)
                    lam = lam_of(m, u)
                    hl = float(np.min(m.fields(to_X(m, u))["h"] / max(mu * m.h0, 1e-300))) if mu > 0 else float("inf")
                    flag = "trend only (beyond validity)" if (mu > 0.2 or hl < 1.0) else "within validity"
                    rows.append([s, name, bct, mu, lam, 100.0 * (lam - base_ct) / base_ct, hl, flag])
    write_csv(os.path.join(out, "B_sweep", "mu_sweep.csv"),
              ["scenario", "point", "BC", "l_over_h0", "lam", "dP_cr_percent_vs_CT", "min_h_over_l", "validity"], rows)


# ------------------------------------------------------------------ C: CT <-> MSGT cross evaluation
def run_cross(sel, out):
    rows = []
    groups = {}
    for s, v in sel.items():
        if v["base"] != "main":
            continue
        k = (v["meta"]["pair"], v["meta"]["bc"])
        groups.setdefault(k, {})[v["meta"]["mu"]] = (s, v)
    for k, g in groups.items():
        if 0.0 not in g:
            continue
        s0, v0 = g[0.0]
        for mu in (0.1, 0.2):
            if mu not in g:
                continue
            s1, v1 = g[mu]
            m_ct, m_ms = make_model(v1["meta"], mu=0.0), make_model(v1["meta"])
            ref = (1.1 * max(v0["P"][:, 0].max(), v1["P"][:, 0].max()), 0.9 * min(v0["P"][:, 1].min(), v1["P"][:, 1].min()))
            # CT-optimal designs evaluated with MSGT, MSGT-optimal designs evaluated with CT
            ct_in_ms = np.array([[p[0], lam_of(m_ms, p[2:20])] for p in v0["P"]])
            ms_in_ct = np.array([[p[0], lam_of(m_ct, p[2:20])] for p in v1["P"]])
            ct_in_ms = ct_in_ms[np.isfinite(ct_in_ms[:, 1])]
            rows.append([k[0], k[1], mu, "HV(MSGT-optimal, MSGT)", M.hypervolume(v1["P"][:, :2], ref)])
            rows.append([k[0], k[1], mu, "HV(CT-optimal designs evaluated with MSGT)", M.hypervolume(ct_in_ms, ref)])
            rows.append([k[0], k[1], mu, "HV(CT-optimal, CT)", M.hypervolume(v0["P"][:, :2], ref)])
            rows.append([k[0], k[1], mu, "HV(MSGT-optimal designs evaluated with CT)", M.hypervolume(ms_in_ct, ref)])
            # mass-matched lambda comparison at the representative MSGT points (nearest CT-optimal design, |dm|/m <= 1 %)
            for name, i in v1["pts"]:
                m1, l1 = v1["P"][i, 0], v1["P"][i, 1]
                j = int(np.argmin(np.abs(ct_in_ms[:, 0] - m1)))
                if abs(ct_in_ms[j, 0] - m1) / m1 <= 0.01:
                    rows.append([k[0], k[1], mu, f"lam ratio CT-optimal/MSGT-optimal at {name} (mass-matched)", ct_in_ms[j, 1] / l1])
                else:
                    rows.append([k[0], k[1], mu, f"lam ratio CT-optimal/MSGT-optimal at {name} (mass-matched)", float("nan")])
    write_csv(os.path.join(out, "C_cross", "ct_msgt_cross.csv"), ["pair", "BC", "l_over_h0", "quantity", "value"], rows)


# ------------------------------------------------------------------ D: l(x) ablation
def run_ablation(sel, out):
    rows = []
    for s, v in sel.items():
        meta = v["meta"]
        if meta["pair"] != "ALSIC" or v["base"] != "main" or meta["mu"] == 0.0:
            continue
        for name, i in v["pts"]:
            u = v["P"][i, 2:20]
            m0 = make_model(meta, N=15)
            lam_c = lam_of(m0, u)
            for r in (2.0, 0.5):
                mp = make_model(meta, N=15, l_mode="phase", l_ratio=r)
                lam_p = lam_of(mp, u)
                lbar = float(np.sum(mp.prof.w * mp.length_scale(mp.prof.B @ u[12:]))) / mp.h0       # mean l / h0
                lam_e = lam_of(make_model(meta, N=15, mu=lbar), u)
                rows.append([s, name, r, lam_c, lam_p, lam_e, lbar, 100 * (lam_p - lam_c) / lam_c, 100 * (lam_p - lam_e) / lam_e])
    write_csv(os.path.join(out, "D_ablation", "l_of_x_ablation.csv"),
              ["scenario", "point", "r (artificial ratio)", "lam_const_l", "lam_phase_l(x)", "lam_const_equal_mean_l",
               "mean_l_over_h0", "dlam_percent_vs_const_l", "dlam_percent_vs_equal_mean_l"], rows)


# ------------------------------------------------------------------ E: benchmark columns
def bench_designs():
    g = np.array([0.0, 1 / 9, 1 / 3, 2 / 3, 8 / 9, 1.0])   # Greville abscissae: linear profiles are reproduced exactly
    out = []
    for s in np.linspace(0.2, 1.5, 27):
        for V in (0.0, 0.25, 0.5, 0.75, 1.0):
            out.append(("uniform", s, 0.0, V, np.concatenate([np.full(6, s), np.full(6, s), np.full(6, V)])))
    for a in (0.2, 0.4, 0.6):
        for s in np.linspace(0.2 / (1 - a), 1.5, 27):
            if s > 1.5:
                continue
            for V in (0.0, 0.25, 0.5, 0.75, 1.0):
                prof = s * (1.0 - a * g)
                out.append(("linear taper", s, a, V, np.concatenate([prof, prof, np.full(6, V)])))
    return out


def run_bench(sel, out):
    rows, summ = [], []
    designs = bench_designs()
    for s, v in sel.items():
        meta = v["meta"]
        m = make_model(meta, N=12)
        from objectives import DesignProblem
        p = DesignProblem(m, kappa_min=meta["kappa_min"])
        pts = []
        for fam, sc, a, V, u in designs:
            u = np.clip(u, p.lb, p.ub)
            f1, lam, cv = p.evaluate(u)
            if cv == 0.0 and np.isfinite(f1):
                pts.append([f1, lam, fam, sc, a, V])
        F = np.array([[q[0], q[1]] for q in pts])
        nd = M.non_dominated(F)
        ref = (1.1 * max(v["P"][:, 0].max(), F[:, 0].max()), 0.9 * min(v["P"][:, 1].min(), F[:, 1].min()))
        hv_b, hv_o = M.hypervolume(nd, ref), M.hypervolume(v["P"][:, :2], ref)
        summ.append([s, len(pts), len(nd), hv_b, hv_o, hv_o / hv_b if hv_b > 0 else float("nan")])
        for name, i in v["pts"]:
            m1, l1 = v["P"][i, 0], v["P"][i, 1]
            j = int(np.argmin(np.abs(nd[:, 0] - m1)))
            ok = abs(nd[j, 0] - m1) / m1 <= 0.02
            rows.append([s, name, m1, l1, nd[j, 0] if ok else float("nan"), nd[j, 1] if ok else float("nan"), l1 / nd[j, 1] if ok else float("nan")])
    write_csv(os.path.join(out, "E_bench", "benchmark_summary.csv"),
              ["scenario", "n_feasible_benchmarks", "n_nondominated_benchmarks", "HV_benchmark", "HV_optimised_Pstar", "HV_ratio"], summ)
    write_csv(os.path.join(out, "E_bench", "benchmark_mass_matched.csv"),
              ["scenario", "point", "f1bar_opt", "lam_opt", "f1bar_bench(|dm|<=2%)", "lam_bench", "lam_opt/lam_bench"], rows)


# ------------------------------------------------------------------ F: profiles
def profile_row(m, u):
    X = to_X(m, u)
    f = m.fields(X)
    xi = m.prof.xi
    D1, _ = D_coefficients(f["E"], f["G"], f["A"], f["I"], f["l"], f["l"], f["l"])
    hl = f["h"] / np.maximum(f["l"], 1e-300) if np.all(f["l"] > 0) else np.full_like(xi, np.inf)
    return xi, f["b"] / m.b0, f["h"] / m.h0, f["Vr"], D1 / (f["E"] * f["I"]), hl


def active_constraints(m, u, tol=1e-6):
    n = m.n
    b, h, V = u[:n], u[n:2 * n], u[2 * n:]
    vbar = float(np.sum(m.prof.w * (m.prof.B @ V)))
    return {"b_eq_h": int(np.sum(np.abs(b - h) < tol)), "h_at_min": int(np.sum(h < m.h_min + tol)), "h_at_max": int(np.sum(h > 1.5 - tol)),
            "b_at_max": int(np.sum(b > 1.5 - tol)), "V_at_0": int(np.sum(V < tol)), "V_at_1": int(np.sum(V > 1 - tol)),
            "Vbar_at_0.1": int(abs(vbar - 0.10) < 1e-4), "Vbar_at_0.9": int(abs(vbar - 0.90) < 1e-4), "Vbar": vbar}


def run_profiles(sel, out):
    rows, store = [], {}
    for s, v in sel.items():
        m = make_model(v["meta"])
        for name, i in v["pts"]:
            u = v["P"][i, 2:20]
            xi, b, h, V, d1, hl = profile_row(m, u)
            store[f"{s}|{name}"] = np.vstack([xi, b, h, V, d1, hl])
            ac = active_constraints(m, u)
            rows.append([s, name, v["P"][i, 0], v["P"][i, 1], m.screening(to_X(m, u)), float(np.min(hl)), float(np.max(d1))] + list(ac.values()))
    write_csv(os.path.join(out, "F_profiles", "representative_profiles_summary.csv"),
              ["scenario", "point", "f1bar", "lam", "eps_s_max", "min_h_over_l", "max_D1_over_EI", "n_b_eq_h", "n_h_at_min", "n_h_at_1.5",
               "n_b_at_1.5", "n_V_at_0", "n_V_at_1", "Vbar_at_0.1", "Vbar_at_0.9", "Vbar"], rows)
    np.savez_compressed(os.path.join(out, "F_profiles", "representative_profiles.npz"), **store)
    # per-run knee designs (robustness across the 30 seeds and the three algorithms)
    krow, kstore = [], {}
    for base in BASES:
        for s, d in load_union(base).items():
            m = make_model(d["meta"])
            for ai, alg in enumerate(("awsm", "nsga2", "moead")):
                for f in sorted(glob.glob(os.path.join(RESULTS, base + SRC[alg], s, "*.npz"))):
                    z = np.load(f, allow_pickle=False)
                    mm = json.loads(str(z["meta"]))
                    fr = z["front"]
                    if mm["alg"] != alg or len(fr) == 0:
                        continue
                    P = pareto_union(np.column_stack([fr[:, 0], fr[:, 1], fr[:, 3:21], np.zeros(len(fr)), np.zeros(len(fr))]))
                    knee = dict(select_points(P))["knee"]
                    u = P[knee, 2:20]
                    xi, b, h, V, d1, hl = profile_row(m, u)
                    kstore[f"{s}|{alg}|{mm['seed']}"] = np.vstack([xi, b, h, V, d1, hl])
                    krow.append([s, alg, mm["seed"], P[knee, 0], P[knee, 1]])
    write_csv(os.path.join(out, "F_profiles", "per_run_knee_designs.csv"), ["scenario", "alg", "seed", "f1bar", "lam"], krow)
    np.savez_compressed(os.path.join(out, "F_profiles", "per_run_knee_profiles.npz"), **kstore)


# ------------------------------------------------------------------ G: equal-area circle / ellipse
def run_sections(sel, out):
    rows = []
    for s, v in sel.items():
        meta = v["meta"]
        m = make_model(meta, N=15)
        for name, i in v["pts"]:
            u = v["P"][i, 2:20]
            f = m.fields(to_X(m, u))
            A, I, b, h = f["A"], f["I"], f["b"], f["h"]
            I_circ = A**2 / (4.0 * np.pi)                                    # circle with the same A(x)
            h_e = np.sqrt(4.0 * A / (np.pi * (b / h)))                       # ellipse, axis ratio b/h kept, same A
            I_ell = A * h_e**2 / 16.0
            vals = {}
            for lab, Iv in (("rect", I), ("circle", I_circ), ("ellipse", I_ell)):
                l = f["l"]
                vals[lab] = buckling_load(m.rb, f["E"], f["G"], A, Iv, l, l, l, m.L)
            lam = lambda P: P * m.L**2 / (m.pair.right.E * m.I0)       # noqa: E731
            rows.append([s, name, v["P"][i, 0], lam(vals["rect"]), lam(vals["circle"]), lam(vals["ellipse"]),
                         vals["circle"] / vals["rect"], vals["ellipse"] / vals["rect"]])
    write_csv(os.path.join(out, "G_sections", "equal_area_sections.csv"),
              ["scenario", "point", "f1bar (equal for all sections)", "lam_rect", "lam_circle", "lam_ellipse(b/h ratio kept)", "circle/rect", "ellipse/rect"], rows)


# ------------------------------------------------------------------ S: sensitivity to the number of representative points
def run_sensitivity(sel, out):
    """Profile descriptors at 5 and 9 arc-length quantiles (+ knee): shows that the morphology trend along the
    front does not depend on the number of representative points (Supplementary Materials)."""
    rows = []
    for s, v in sel.items():
        P = v["P"]
        f1, lam = P[:, 0], P[:, 1]
        x = (f1 - f1.min()) / max(np.ptp(f1), 1e-12)
        y = (lam - lam.min()) / max(np.ptp(lam), 1e-12)
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        arc /= arc[-1] if arc[-1] > 0 else 1.0
        m = make_model(v["meta"])
        for nq in (5, 9):
            for q in np.linspace(0.0, 1.0, nq):
                i = int(np.argmin(np.abs(arc - q)))
                u = P[i, 2:20]
                xi, b, h, V, d1, hl = profile_row(m, u)
                rows.append([s, nq, round(float(q), 3), P[i, 0], P[i, 1], b[0], b[-1], h[0], h[-1], float(np.mean(V)), V[0], V[-1], float(np.mean(h / b))])
    write_csv(os.path.join(out, "S_sensitivity", "n_points_sensitivity.csv"),
              ["scenario", "n_points", "arc_quantile", "f1bar", "lam", "b_root", "b_tip", "h_root", "h_tip", "V_mean", "V_root", "V_tip", "mean_h_over_b"], rows)


# ------------------------------------------------------------------ H: measured convexity of the empirical front
def lower_hull(pts):
    """Lower convex hull (Andrew's monotone chain) of points (x, y), both minimised, sorted by x."""
    pts = sorted(map(tuple, pts))
    hull = []
    for p in pts:
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= 0:
            hull.pop()
        hull.append(p)
    return np.array(hull)


def run_convexity(sel, out):
    """AGENTS rule 2: convexity is MEASURED, never assumed. For the union front P* and for each algorithm's pooled
    front: fraction of points lying more than 0.01 above the lower convex hull in normalised (f1bar, 1 - lam) space."""
    rows = []
    for s, v in sel.items():
        P = v["P"]
        lo = (P[:, 0].min(), P[:, 1].min())
        rg = (max(np.ptp(P[:, 0]), 1e-12), max(np.ptp(P[:, 1]), 1e-12))
        allr = load_union(v["base"])[s]["rows"]
        for lab, R in [("P* (union)", P)] + [(n, pareto_union(allr[allr[:, 20] == k])) for k, n in enumerate(("AWSM", "NSGA-II", "MOEA/D"))]:
            if len(R) < 3:
                continue
            x = (R[:, 0] - lo[0]) / rg[0]
            y = 1.0 - (R[:, 1] - lo[1]) / rg[1]
            H = lower_hull(np.column_stack([x, y]))
            yh = np.interp(x, H[:, 0], H[:, 1])
            gap = y - yh
            rows.append([s, lab, len(R), len(H), float(np.mean(gap > 0.01)), float(np.median(gap)), float(np.max(gap))])
    write_csv(os.path.join(out, "H_convexity", "front_convexity_measure.csv"),
              ["scenario", "front", "n_points", "hull_vertices", "fraction_above_hull_0.01", "median_gap", "max_gap"], rows)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="select,A,B,C,D,E,F,G,S,H")
    ap.add_argument("--out", default=POST)
    a = ap.parse_args()
    todo = set(a.only.split(","))
    sel = run_select(a.out)
    print("selected designs for", len(sel), "scenarios")
    for key, fn in (("A", run_bc2), ("B", run_sweep), ("C", run_cross), ("D", run_ablation), ("E", run_bench), ("F", run_profiles), ("G", run_sections), ("S", run_sensitivity), ("H", run_convexity)):
        if key in todo:
            fn(sel, a.out)
            print("done", key)


if __name__ == "__main__":
    main()
