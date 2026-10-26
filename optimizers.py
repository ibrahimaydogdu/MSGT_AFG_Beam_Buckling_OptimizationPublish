"""ABC-Levy (Section 3.3, Eqs. 79-84) and NSGA-II (Section 3.4) with exact FE accounting.

Both optimisers work on any problem object exposing: dim, lb, ub, evaluate(u)->(f1, lam, cv),
nfe, n_constraints, constraints_vec(u) (see objectives.DesignProblem).
Constraint handling: Deb's feasibility-first rule (Eq. 74) in both algorithms.
"""
import numpy as np
from math import gamma, pi, sin

from objectives import deb_better

PENALTY_K = 1.0e2  # K in Eq. (81); user decision 2026-10-25 (text still shows 10^4; see FINAL_REVISION_NOTES.md)
ALPHA0 = 0.01      # Eq. (82)
BETA = 1.5


def levy_step(rng, dim, beta=BETA):
    """Mantegna's algorithm, Eqs. (83)-(84)."""
    sigma_u = (gamma(1 + beta) * sin(pi * beta / 2)
               / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0.0, sigma_u, dim)
    v = rng.normal(0.0, 1.0, dim)
    return u / np.abs(v) ** (1 / beta)


def _fit(F, cv, K):
    """Eq. (81) mapped to a positive roulette fitness (standard ABC treatment for F < 0)."""
    if not np.isfinite(F) or not np.isfinite(cv):
        return 0.0
    Fp = F + (K * cv if cv > 0 else 0.0)
    return 1.0 / (1.0 + Fp) if Fp >= 0 else 1.0 + abs(Fp)


def abc_levy(problem, scalar_fn, rng, max_fe, n_food=50, limit=100, K=PENALTY_K,
             init=None, archive=None):
    """Single-objective ABC-Levy for F = scalar_fn(f1, lam).

    n_food = N_pop = number of food sources = employed bees = onlooker bees (each source is served
    by one employed and, in the onlooker phase, by roulette-selected onlookers; user convention 2026-10-26).
    max_fe : evaluations allowed for THIS call (counted exactly, never exceeded).
    init   : optional list of design vectors injected into the initial population (warm start).
    archive: optional list receiving [f1, lam, cv, u...] for every evaluation.
    Returns dict(best_u, best_f1, best_lam, best_cv, best_F, fe, history)."""
    dim, lb, ub = problem.dim, problem.lb, problem.ub
    fe = 0
    X = problem.sample(rng, n_food) if hasattr(problem, "sample") else rng.uniform(lb, ub, (n_food, dim))
    if init is not None:
        for t, u0 in enumerate(init[:n_food]):
            X[t] = np.clip(u0, lb, ub)
    F = np.empty(n_food)
    CV = np.empty(n_food)
    OBJ = np.empty((n_food, 2))
    trial = np.zeros(n_food)

    def ev(u):
        nonlocal fe
        f1, lam, cv = problem.evaluate(u)
        fe += 1
        if archive is not None:
            archive.append(np.concatenate([[f1, lam, cv], u]))
        Fv = scalar_fn(f1, lam) if np.isfinite(f1) else np.inf
        return Fv, cv, f1, lam

    for i in range(n_food):
        if fe >= max_fe:
            break
        F[i], CV[i], OBJ[i, 0], OBJ[i, 1] = ev(X[i])
    n_done = min(n_food, fe)
    if n_done < n_food:  # budget smaller than the colony
        X, F, CV, OBJ, trial = X[:n_done], F[:n_done], CV[:n_done], OBJ[:n_done], trial[:n_done]
        n_food = n_done
    history = []
    n_scout = 0

    def best_index():
        b = 0
        for i in range(1, n_food):
            if deb_better(F[i], CV[i], F[b], CV[b]):
                b = i
        return b

    def try_move(i):
        k = rng.integers(n_food - 1)
        k = k + 1 if k >= i else k
        j = rng.integers(dim)
        v = X[i].copy()
        v[j] = np.clip(X[i, j] + rng.uniform(-1.0, 1.0) * (X[i, j] - X[k, j]), lb[j], ub[j])  # Eq. (79)
        Fv, cvv, f1v, lamv = ev(v)
        if deb_better(Fv, cvv, F[i], CV[i]):
            X[i], F[i], CV[i], OBJ[i] = v, Fv, cvv, (f1v, lamv)
            trial[i] = 0
        else:
            trial[i] += 1

    while fe < max_fe and n_food >= 2:
        for i in range(n_food):  # employed bee phase
            if fe >= max_fe:
                break
            try_move(i)
        fit = np.array([_fit(F[i], CV[i], K) for i in range(n_food)])  # Eqs. (80)-(81)
        if fit.sum() <= 0:
            p = np.full(n_food, 1.0 / n_food)
        else:
            p = fit / fit.sum()
        for _ in range(n_food):  # onlooker bee phase
            if fe >= max_fe:
                break
            i = rng.choice(n_food, p=p)
            try_move(i)
        b = best_index()  # scout bee phase (Levy flight relative to the best solution)
        for i in range(n_food):
            if fe >= max_fe:
                break
            if trial[i] > limit:
                v = np.clip(X[i] + ALPHA0 * levy_step(rng, dim) * (X[i] - X[b]), lb, ub)  # Eq. (82)
                F[i], CV[i], OBJ[i, 0], OBJ[i, 1] = ev(v)
                X[i] = v
                trial[i] = 0
                n_scout += 1
        b = best_index()
        history.append((fe, float(F[b]), float(CV[b])))
    b = best_index()
    return {"best_u": X[b].copy(), "best_f1": float(OBJ[b, 0]), "best_lam": float(OBJ[b, 1]),
            "best_cv": float(CV[b]), "best_F": float(F[b]), "fe": fe, "history": history, "n_scout": n_scout}


def nsga2(problem, seed, max_fe, pop_size=50, pc=0.90, eta_c=15, eta_m=20, archive=None):
    """NSGA-II via pymoo with Deb's constrained domination. Offspring count = pop_size and
    duplicate elimination off, so the number of evaluations is exactly pop_size*(1+g)."""
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.core.sampling import Sampling
    from pymoo.optimize import minimize

    class _P(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=problem.dim, n_obj=2, n_ieq_constr=problem.n_constraints,
                             xl=problem.lb, xu=problem.ub)

        def _evaluate(self, x, out, *args, **kwargs):
            f1, lam, cv = problem.evaluate(x)
            if archive is not None:
                archive.append(np.concatenate([[f1, lam, cv], x]))
            if np.isfinite(cv):
                out["F"] = [f1, -lam]
                out["G"] = problem.constraints_vec(x)
            else:  # solver failure: infinite violation
                out["F"] = [1.0e12, 1.0e12]
                out["G"] = np.full(problem.n_constraints, 1.0e12)

    class _Init(Sampling):
        def _do(self, _problem, n_samples, *args, random_state=None, **kwargs):
            rs = random_state if random_state is not None else np.random.default_rng(0)
            if hasattr(problem, "sample"):
                return problem.sample(rs, n_samples)       # same sampler as ABC-Levy
            return rs.uniform(problem.lb, problem.ub, (n_samples, problem.dim))

    algo = NSGA2(pop_size=pop_size, n_offsprings=pop_size, sampling=_Init(),
                 crossover=SBX(prob=pc, eta=eta_c), mutation=PM(prob_var=1.0 / problem.dim, eta=eta_m),
                 eliminate_duplicates=False)
    res = minimize(_P(), algo, ("n_eval", max_fe), seed=seed, verbose=False)
    return res


# --------------------------------------------------------------------------------------------
# MOEA/D with Tchebycheff decomposition (Section 3.5, Eqs. 85-87). Own implementation because the
# pymoo version does not support constraints. Same SBX/PM operators as NSGA-II; Deb rule in the update.
def _sbx_child(rng, p1, p2, lb, ub, pc, eta):
    """One child of bounded simulated binary crossover (Deb and Agrawal); with probability pc the pair is
    crossed and each variable is crossed with probability 0.5."""
    y = p1.copy()
    if rng.random() > pc:
        return y
    for k in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[k] - p2[k]) < 1e-14:
            continue
        x1, x2 = min(p1[k], p2[k]), max(p1[k], p2[k])
        rnd = rng.random()
        beta = 1.0 + 2.0 * (x1 - lb[k]) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        bq = (rnd * alpha) ** (1.0 / (eta + 1.0)) if rnd <= 1.0 / alpha else (1.0 / (2.0 - rnd * alpha)) ** (1.0 / (eta + 1.0))
        c1 = 0.5 * ((x1 + x2) - bq * (x2 - x1))
        beta = 1.0 + 2.0 * (ub[k] - x2) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        bq = (rnd * alpha) ** (1.0 / (eta + 1.0)) if rnd <= 1.0 / alpha else (1.0 / (2.0 - rnd * alpha)) ** (1.0 / (eta + 1.0))
        c2 = 0.5 * ((x1 + x2) + bq * (x2 - x1))
        c = c1 if rng.random() < 0.5 else c2
        y[k] = min(max(c, lb[k]), ub[k])
    return y


def _poly_mutation(rng, y, lb, ub, pm, eta):
    y = y.copy()
    for k in range(len(y)):
        if rng.random() < pm:
            d1 = (y[k] - lb[k]) / (ub[k] - lb[k])
            d2 = (ub[k] - y[k]) / (ub[k] - lb[k])
            u = rng.random()
            if u < 0.5:
                dq = (2.0 * u + (1.0 - 2.0 * u) * (1.0 - d1) ** (eta + 1.0)) ** (1.0 / (eta + 1.0)) - 1.0
            else:
                dq = 1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - d2) ** (eta + 1.0)) ** (1.0 / (eta + 1.0))
            y[k] = min(max(y[k] + dq * (ub[k] - lb[k]), lb[k]), ub[k])
    return y


def moead(problem, seed, max_fe, N=50, T=10, delta=0.9, nr=2, pc=0.90, eta_c=15, eta_m=20, archive=None):
    """MOEA/D-Tchebycheff, exactly max_fe evaluations. Returns dict(X, f1, lam, cv, fe)."""
    rng = np.random.default_rng(seed)
    dim, lb, ub = problem.dim, problem.lb, problem.ub
    pm = 1.0 / dim
    w = np.column_stack([np.arange(N) / (N - 1.0), 1.0 - np.arange(N) / (N - 1.0)])
    w = np.maximum(w, 1e-6)                                   # Eq. (85), zero components replaced
    dist = np.linalg.norm(w[:, None, :] - w[None, :, :], axis=2)
    B = np.argsort(dist, axis=1)[:, :T]
    fe = 0
    X = problem.sample(rng, N) if hasattr(problem, "sample") else rng.uniform(lb, ub, (N, dim))
    F1, LAM, CV = np.empty(N), np.empty(N), np.empty(N)
    z1, z2 = np.inf, -np.inf                                  # ideal point (feasible designs only)

    def ev(u):
        nonlocal fe, z1, z2
        f1, lam, cv = problem.evaluate(u)
        fe += 1
        if archive is not None:
            archive.append(np.concatenate([[f1, lam, cv], u]))
        if cv == 0.0:
            z1, z2 = min(z1, f1), max(z2, lam)
        return f1, lam, cv

    for i in range(N):
        F1[i], LAM[i], CV[i] = ev(X[i]) if fe < max_fe else (np.inf, 0.0, np.inf)

    def norm_spans():
        feas = CV == 0.0
        if not feas.any() or not np.isfinite(z1):
            return 1.0, 1.0
        return max(F1[feas].max() - z1, 1e-12), max(z2 - LAM[feas].min(), 1e-12)

    def g_te(f1, lam, wi, s1, s2):
        if not np.isfinite(f1):
            return np.inf
        return max(wi[0] * (f1 - z1) / s1, wi[1] * (z2 - lam) / s2)   # Eq. (86)

    while fe < max_fe:
        s1, s2 = norm_spans()
        for i in range(N):
            if fe >= max_fe:
                break
            from_nb = rng.random() < delta
            pool = B[i] if from_nb else np.arange(N)
            a, b = rng.choice(pool, 2, replace=False)
            y = _poly_mutation(rng, _sbx_child(rng, X[a], X[b], lb, ub, pc, eta_c), lb, ub, pm, eta_m)
            f1, lam, cv = ev(y)
            c = 0
            for j in rng.permutation(pool):
                if c >= nr:
                    break
                if cv == 0.0 and CV[j] > 0.0:
                    rep = True
                elif cv > 0.0 and CV[j] > 0.0:
                    rep = cv < CV[j]
                elif cv == 0.0 and CV[j] == 0.0:
                    rep = g_te(f1, lam, w[j], s1, s2) <= g_te(F1[j], LAM[j], w[j], s1, s2)   # Eq. (87)
                else:
                    rep = False
                if rep:
                    X[j], F1[j], LAM[j], CV[j] = y, f1, lam, cv
                    c += 1
    return {"X": X, "f1": F1, "lam": LAM, "cv": CV, "fe": fe}
