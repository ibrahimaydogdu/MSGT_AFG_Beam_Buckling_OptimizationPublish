"""Results figures for Section 4 from stored data (no optimisation, no re-evaluation of the campaign):
  Fig_pareto_CF / Fig_pareto_CP   Pareto fronts per algorithm (non-dominated union over 30 runs) and the benchmark columns
  Fig_size_effect                 P* for l/h0 = 0, 0.1, 0.2 (CT vs MSGT) with the knee design of each front
  Fig_profiles_CF / _CP           knee-design profiles b(x)/b0, h(x)/h0, V_r(x): median and inter-quartile band over runs
  Fig_sweep                       l/h0 sweep of the representative designs (BC1/BC2); beyond 0.2 = trend only
Colours: Okabe-Ito for algorithms (+ marker shapes); a single-hue sequence for the ordered parameter l/h0.
Usage: python -I -X utf8 make_figures_results.py [--out DIR]"""
import argparse
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import post_optimal as PO  # noqa: E402
import metrics as M  # noqa: E402
from objectives import DesignProblem  # noqa: E402

RES = os.path.join(HERE, "results")
POST = os.path.join(RES, "post")
FIG = (os.path.normpath(os.path.join(HERE, "..", "04_Results", "figures"))
       if os.path.exists(os.path.join(HERE, "..", "04_Results"))
       else os.path.join(HERE, "figures"))
ALGS = ("awsm", "nsga2", "moead")
NAMES = {"awsm": "AWSM-ABC-Levy", "nsga2": "NSGA-II", "moead": "MOEA/D"}
COL = {"awsm": "#0072B2", "nsga2": "#D55E00", "moead": "#009E73"}
MRK = {"awsm": "o", "nsga2": "s", "moead": "^"}
MU_COL = {0.0: "#9ecae1", 0.1: "#4292c6", 0.2: "#08306b"}       # one hue, light -> dark with increasing l/h0
PAIRS = (("ALSIC", "Al/SiC"), ("SYN", "Synthetic pair"))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8.5, "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "pdf.fonttype": 42})


def save(fig, name, out):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def sid(pair, bc, mu):
    return f"{pair}_{bc}_mu{mu}"


def load_all():
    U = {}
    union_cache = os.path.join(POST, "select", "union_main.npz")
    if os.path.exists(union_cache):
        import ast
        z = np.load(union_cache)
        scenarios = [k[:-5] for k in z.files if k.endswith("_rows")]
        for s in scenarios:
            meta_val = str(z[s + "_meta"])
            try:
                meta_dict = json.loads(meta_val)
            except Exception:
                meta_dict = ast.literal_eval(meta_val)
            U[s] = {"meta": meta_dict, "rows": z[s + "_rows"], "base": "main"}
    else:
        for base in ("main",):
            for s, d in PO.load_union(base).items():
                U[s] = d
    P = np.load(os.path.join(POST, "select", "Pstar.npz"))
    return U, P


def alg_front(rows, k):
    r = rows[rows[:, 20] == k]
    return PO.pareto_union(r) if len(r) else r


def bench_front(meta):
    m = PO.make_model(meta, N=12)
    p = DesignProblem(m, kappa_min=meta["kappa_min"])
    pts = []
    for fam, sc, a, V, u in PO.bench_designs():
        f1, lam, cv = p.evaluate(np.clip(u, p.lb, p.ub))
        if cv == 0.0 and np.isfinite(f1):
            pts.append([f1, lam])
    return M.non_dominated(np.array(pts))


BCLAB = {"CF": "C–F", "CP": "C–P"}
PARETO_MUS = (0.0, 0.1)      # CT baseline and the main-text value; l/h0 = 0.2 is shown in Fig_size_effect


def fig_pareto(bc, U, out):
    fig, axs = plt.subplots(2, 2, figsize=(5.9, 5.4))
    for i, (pair, plab) in enumerate(PAIRS):
        for j, mu in enumerate(PARETO_MUS):
            ax = axs[i, j]
            d = U[sid(pair, bc, mu)]
            for k, al in enumerate(ALGS):
                f = alg_front(d["rows"], k)
                ax.plot(f[:, 0], f[:, 1], MRK[al], ms=2.6, color=COL[al], alpha=0.75, mew=0, label=NAMES[al], rasterized=True, zorder=3 + k)
            b = bench_front(d["meta"])
            ax.plot(b[:, 0], b[:, 1], color="#222222", lw=1.1, ls="--", label="Uniform / linearly tapered columns", zorder=2)
            ax.set_yscale("log")
            ax.set_title(f"{plab}, {BCLAB[bc]}, $l/h_0$ = {mu}", fontsize=9)
            ax.grid(True, lw=0.3, color="#dddddd")
            ax.tick_params(labelsize=8.5)
            if i == 1:
                ax.set_xlabel(r"Mass ratio $\bar f_1 = m/m_0$", fontsize=9)
            if j == 0:
                ax.set_ylabel(r"Buckling parameter $\lambda$", fontsize=9)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5, markerscale=3.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, f"Fig_pareto_{bc}", out)


def knee_point(P):
    sel = dict(PO.select_points(P))
    return P[sel["knee"]]


def fig_size_effect(U, P, out):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for i, (pair, plab) in enumerate(PAIRS):
        for j, bc in enumerate(("CF", "CP")):
            ax = axs[i, j]
            for mu in (0.0, 0.1, 0.2):
                p = P[sid(pair, bc, mu)]
                ax.plot(p[:, 0], p[:, 1], ".", ms=1.4, color=MU_COL[mu], alpha=0.8, mew=0, rasterized=True, label=f"$l/h_0$ = {mu}" + (" (CT)" if mu == 0 else ""))
                k = knee_point(p)
                ax.plot([k[0]], [k[1]], "D", ms=4.5, color=MU_COL[mu], mec="black", mew=0.7)
            ax.set_yscale("log")
            ax.set_title(f"{plab}, {'clamped-free' if bc == 'CF' else 'clamped-pinned'}", fontsize=8)
            ax.grid(True, lw=0.3, color="#dddddd")
            if i == 1:
                ax.set_xlabel(r"Mass ratio $\bar f_1$")
            if j == 0:
                ax.set_ylabel(r"Buckling parameter $\lambda$")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5, markerscale=5, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, -0.045, "Pooled non-dominated sets P* of the three algorithms; diamonds = knee designs.", ha="center", fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "Fig_size_effect", out)


PR_COL = {0.0: "#0072B2", 0.1: "#D55E00", 0.2: "#009E73"}          # Okabe-Ito blue, vermillion, bluish green
PR_LS = {0.0: "-", 0.1: "--", 0.2: (0, (1, 1.2))}


def fig_profiles(bc, out):
    z = np.load(os.path.join(POST, "F_profiles", "per_run_knee_profiles.npz"))
    fig, axs = plt.subplots(2, 3, figsize=(5.9, 4.6), sharex=True)
    titles = (r"$b(x)/b_0$", r"$h(x)/h_0$", r"$V_r(x)$")
    for i, (pair, plab) in enumerate(PAIRS):
        for mu in (0.0, 0.1, 0.2):
            keys = [k for k in z.files if k.startswith(sid(pair, bc, mu) + "|")]
            if not keys:
                continue
            arr = np.array([z[k] for k in keys])           # runs x 6 x nodes
            xi = arr[0, 0]
            for j, row in enumerate((1, 2, 3)):
                ax = axs[i, j]
                q1, md, q3 = np.percentile(arr[:, row, :], [25, 50, 75], axis=0)
                ax.plot(xi, q1, color=PR_COL[mu], lw=0.5, alpha=0.55, ls=PR_LS[mu])
                ax.plot(xi, q3, color=PR_COL[mu], lw=0.5, alpha=0.55, ls=PR_LS[mu])
                ax.plot(xi, md, color=PR_COL[mu], lw=1.6, ls=PR_LS[mu], label=f"$l/h_0$ = {mu}" + (" (CT)" if mu == 0 else ""))
                if i == 0:
                    ax.set_title(titles[j], fontsize=9)
                if i == 1:
                    ax.set_xlabel(r"$x/L$", fontsize=9)
        axs[i, 0].set_ylabel(plab, fontsize=9)
    for ax in axs.ravel():
        ax.grid(True, lw=0.3, color="#dddddd")
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=8.5)
    for i in range(2):
        axs[i, 2].set_ylim(-0.03, 1.03)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, f"Fig_profiles_{bc}", out)


def fig_beam_schematic(out, mu=0.1, bc="CF"):
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    z = np.load(os.path.join(POST, "F_profiles", "representative_profiles.npz"))
    cmap = LinearSegmentedColormap.from_list("phase", ["#56B4E9", "#D55E00"])
    fig = plt.figure(figsize=(5.9, 6.0))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 0.6, 1, 1], hspace=0.12)
    axs = [fig.add_subplot(gs[0])]
    for r in (1, 3, 4):
        axs.append(fig.add_subplot(gs[r], sharex=axs[0]))
    ends = {"ALSIC": ("Al", "SiC"), "SYN": ("left phase", "right phase")}
    for p, (pair, plab) in enumerate(PAIRS):
        a = z[f"{sid(pair, bc, mu)}|knee"]
        xi, b, h, V = a[0], a[1], a[2], a[3]
        xs = np.linspace(0, 1, 400)
        b, h, V = (np.interp(xs, xi, q) for q in (b, h, V))
        for q, (wd, nm, lab) in enumerate(((h, "side", r"side view, $h(x)/h_0$"), (b, "top", r"top view, $b(x)/b_0$"))):
            ax = axs[2 * p + q]
            quads = [[(xs[k], -wd[k] / 2), (xs[k + 1], -wd[k + 1] / 2), (xs[k + 1], wd[k + 1] / 2), (xs[k], wd[k] / 2)] for k in range(len(xs) - 1)]
            pc = PolyCollection(quads, array=0.5 * (V[:-1] + V[1:]), cmap=cmap, clim=(0, 1), edgecolors="face", linewidths=0.3)
            ax.add_collection(pc)
            ax.plot(xs, wd / 2, color="black", lw=1.0)
            ax.plot(xs, -wd / 2, color="black", lw=1.0)
            ax.plot([0, 1], [0.5, 0.5], color="#555555", lw=0.6, ls=(0, (4, 3)))
            ax.plot([0, 1], [-0.5, -0.5], color="#555555", lw=0.6, ls=(0, (4, 3)))
            ax.plot([0, 0], [-wd[0] / 2, wd[0] / 2], color="black", lw=1.0)
            ax.plot([1, 1], [-wd[-1] / 2, wd[-1] / 2], color="black", lw=1.0)
            if bc == "CF":                       # clamped (left) support: wall with hatching
                ax.add_patch(Rectangle((-0.045, -0.9), 0.045, 1.8, facecolor="white", edgecolor="black", hatch="////", lw=0.8))
            ax.set_xlim(-0.07, 1.04)
            ax.set_ylim(-0.95, 0.95)
            ax.set_yticks([])
            for sp in ("left", "right", "top", "bottom"):
                ax.spines[sp].set_visible(False)
            ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8.5, labelpad=6)
            if q == 0:
                ax.set_title(f"{plab}, {BCLAB[bc]}, $l/h_0$ = {mu}", fontsize=9, loc="left", pad=18)
            ax.text(1.015, 0.0, "free end" if q == 0 and p == 0 else "", fontsize=8.5, va="center", ha="left")
            ax.text(-0.025, 0.0, "", fontsize=1)
    axs[0].text(-0.025, 1.0, "clamped", fontsize=8.5, ha="center", va="bottom", transform=axs[0].get_xaxis_transform())
    axs[3].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axs[3].tick_params(axis="x", labelsize=8.5)
    axs[3].spines["bottom"].set_visible(True)
    axs[3].set_xlabel(r"$x/L$", fontsize=9)
    for ax in axs[:3]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap), ax=axs, orientation="horizontal", fraction=0.035, pad=0.14, aspect=40)
    cb.set_label(r"Local volume fraction $V_r(x)$  (Al/SiC: 1 = SiC; synthetic pair: 1 = softer right-end phase)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8.5)
    fig.text(0.5, 0.0, r"Knee designs (median over runs); dashed lines: $b_0$, $h_0$. Transverse scale exaggerated relative to $L$ (not to scale).",
             ha="center", fontsize=8.5, color="#333333")
    save(fig, "Fig_beam_schematic", out)


def envelope(front, grid):
    """lambda envelope of a front: largest lambda among points with f1 <= m, on the mass grid (nan before the first point)."""
    f = front[np.argsort(front[:, 0])]
    cm = np.maximum.accumulate(f[:, 1])
    idx = np.searchsorted(f[:, 0], grid, side="right") - 1
    out = np.where(idx >= 0, cm[np.clip(idx, 0, None)], np.nan)
    return out


def fig_front_ratio(U, P, out, mu=0.1):
    """Median over runs of lambda_envelope(run) / lambda_envelope(P*) versus mass: where each algorithm is close to the pooled front."""
    import glob
    import json
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey=True)
    src = {"awsm": "mainV2", "nsga2": "main", "moead": "mainM"}
    for i, (pair, plab) in enumerate(PAIRS):
        for j, bc in enumerate(("CF", "CP")):
            ax = axs[i, j]
            s = sid(pair, bc, mu)
            Ps = P[s]
            grid = np.exp(np.linspace(np.log(Ps[:, 0].min() * 1.0001), np.log(Ps[:, 0].max()), 120))
            envP = envelope(Ps[:, :2], grid)
            rows_s = U[s]["rows"]
            for ai, al in enumerate(ALGS):
                al_rows = rows_s[rows_s[:, 20] == ai]
                seeds = np.unique(al_rows[:, 21])
                rat = []
                for seed in seeds:
                    fr_seed = al_rows[al_rows[:, 21] == seed]
                    if len(fr_seed) > 0:
                        p_seed = PO.pareto_union(fr_seed[:, :2])
                        rat.append(envelope(p_seed, grid) / envP)
                if rat:
                    rat = np.array(rat)
                    ax.plot(grid / Ps[:, 0].max(), np.nanmedian(rat, axis=0), color=COL[al], lw=1.4, label=NAMES[al])
            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{plab}, {'clamped-free' if bc == 'CF' else 'clamped-pinned'}, $l/h_0$ = {mu}", fontsize=8)
            ax.grid(True, lw=0.3, color="#dddddd")
            if i == 1:
                ax.set_xlabel(r"Mass ratio relative to the heaviest design of P*")
            if j == 0:
                ax.set_ylabel(r"$\lambda$ envelope of a run / $\lambda$ envelope of P*")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "Fig_front_ratio", out)


def fig_sweep(out):
    rows = list(csv.DictReader(open(os.path.join(POST, "B_sweep", "mu_sweep.csv"), encoding="utf-8")))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, bct in zip(axs, ("BC1", "BC2")):
        for (pair, plab), ls in zip(PAIRS, ("-", "--")):
            for mu0, col in ((0.1, "#4292c6"), (0.2, "#08306b")):
                for bc in ("CF",):
                    s = sid(pair, bc, mu0)
                    pts = [r for r in rows if r["scenario"] == s and r["point"] == "knee" and r["BC"] == bct and float(r["l_over_h0"]) > 0]
                    pts.sort(key=lambda r: float(r["l_over_h0"]))
                    ax.plot([float(r["l_over_h0"]) for r in pts], [float(r["dP_cr_percent_vs_CT"]) for r in pts], ls, color=col, lw=1.3,
                            marker="o", ms=3, label=f"{plab}, designed at $l/h_0$ = {mu0}")
        ax.axvspan(0.2, 1.05, color="#eeeeee", zorder=0)
        ax.text(0.22, 0.97, "beyond the validity range:\ntrend only", transform=ax.get_xaxis_transform(), fontsize=6.5, va="top", color="#555555")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$l/h_0$ of the evaluation")
        ax.set_title(f"{bct}, clamped-free knee designs", fontsize=8)
        ax.grid(True, lw=0.3, color="#dddddd")
    axs[0].set_ylabel(r"$\Delta P_{cr}$ relative to CT [%]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "Fig_sweep", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=FIG)
    a = ap.parse_args()
    U, P = load_all()
    for bc in ("CF", "CP"):
        fig_pareto(bc, U, a.out)
        fig_profiles(bc, a.out)
    fig_beam_schematic(a.out)
    fig_size_effect(U, P, a.out)
    fig_front_ratio(U, P, a.out)
    fig_sweep(a.out)
    print("figures saved successfully")


if __name__ == "__main__":
    main()
