"""Unit tests for the optimisation layer (DEVELOPER_PROMPT, stage 2): ABC-Levy, NSGA-II,
AWSM-type search, metrics and statistics on analytic problems with known answers, plus
integration checks on the real column model. Run: python -I -m pytest test_optimizers.py -q"""
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from optimizers import abc_levy, nsga2, levy_step, _fit  # noqa: E402
from objectives import DesignProblem, deb_better  # noqa: E402
from awsm import awsm_search, nondominated, feasible_front  # noqa: E402
import metrics as M  # noqa: E402
import stat_tests as S  # noqa: E402


class Zdt1Like:
    """min f1 = x0, max lam = -f2 with ZDT1 f2 = g (1 - sqrt(x0/g)), g = 1 + 9 mean(x1..);
    convex front f2 = 1 - sqrt(f1) at x1.. = 0. Constraint: x1 >= 0.0 (inactive), cv = 0."""

    def __init__(self, dim=6):
        self.dim, self.lb, self.ub = dim, np.zeros(dim), np.ones(dim)
        self.nfe, self.n_constraints = 0, 1

    def reset(self):
        self.nfe = 0

    def constraints_vec(self, u):
        return np.array([-1.0])

    def evaluate(self, u):
        self.nfe += 1
        g = 1.0 + 9.0 * np.mean(u[1:])
        f2 = g * (1.0 - np.sqrt(u[0] / g))
        return float(u[0]), float(-f2), 0.0


def test_levy_sigma_value():
    from math import gamma, pi, sin
    b = 1.5
    s = (gamma(1 + b) * sin(pi * b / 2) / (gamma((1 + b) / 2) * b * 2 ** ((b - 1) / 2))) ** (1 / b)
    assert abs(s - 0.6966) < 5e-4          # Eq. (84), beta = 1.5 (standard value 0.6966)
    rng = np.random.default_rng(0)
    st = levy_step(rng, 1000)
    assert np.all(np.isfinite(st))


def test_deb_truth_table():
    assert deb_better(5.0, 0.0, 1.0, 0.2)        # feasible beats infeasible even if worse F
    assert not deb_better(1.0, 0.2, 5.0, 0.0)
    assert deb_better(5.0, 0.1, 1.0, 0.2)        # smaller CV wins among infeasible
    assert deb_better(1.0, 0.0, 2.0, 0.0)        # F decides among feasible
    assert not deb_better(2.0, 0.0, 2.0, 0.0)
    assert not deb_better(1.0, np.inf, 2.0, np.inf)


def test_fit_mapping():
    assert _fit(0.0, 0.0, 100.0) == 1.0
    assert _fit(1.0, 0.0, 100.0) == 0.5
    assert _fit(1.0, 0.01, 100.0) == pytest.approx(1.0 / (1 + 1.0 + 100.0 * 0.01))   # Eq. (81)
    assert _fit(-2.0, 0.0, 100.0) == 3.0 and _fit(np.inf, np.inf, 100.0) == 0.0


def test_abc_sphere_and_fe_accounting():
    class Sphere(Zdt1Like):
        def evaluate(self, u):
            self.nfe += 1
            return float(np.sum((u - 0.3) ** 2)), 0.0, 0.0
    p = Sphere(6)
    rng = np.random.default_rng(1001)
    res = abc_levy(p, lambda f1, lam: f1, rng, 6000, n_food=25, limit=100)
    assert res["fe"] == 6000 == p.nfe           # exact budget
    assert res["best_f1"] < 1e-3
    res2 = abc_levy(Sphere(6), lambda f1, lam: f1, np.random.default_rng(1001), 6000, n_food=25)
    assert res2["best_f1"] == res["best_f1"]    # same seed, same result


def test_abc_constraint_feasibility_first():
    class Constrained(Zdt1Like):
        def evaluate(self, u):
            self.nfe += 1
            return float(np.sum(u)), 0.0, float(max(0.0, 2.0 - np.sum(u)))   # feasible iff sum(u) >= 2
    p = Constrained(6)
    res = abc_levy(p, lambda f1, lam: f1, np.random.default_rng(7), 4000)
    assert res["best_cv"] == 0.0 and 2.0 <= res["best_f1"] < 2.2


def test_awsm_zdt_front_and_budget():
    p = Zdt1Like(6)
    out = awsm_search(p, seed=1001, B=9000, fe_per_run=500)
    assert out["fe"] <= 9000 and p.nfe == out["fe"]
    F = out["front"]
    assert len(F) >= 8
    f1, lam = F[:, 0], F[:, 1]
    # no point can beat the analytic front f2 = 1 - sqrt(f1) (lam = -f2 <= sqrt(f1) - 1)
    assert np.all(lam <= np.sqrt(f1) - 1.0 + 1e-9)
    # and the front is reasonably close to it
    assert np.median((np.sqrt(f1) - 1.0) - lam) < 0.1
    out2 = awsm_search(Zdt1Like(6), seed=1001, B=9000, fe_per_run=500)
    assert np.array_equal(out["front"], out2["front"])      # reproducible


def test_nsga2_exact_fe_and_reproducible():
    p = Zdt1Like(6)
    arch = []
    res = nsga2(p, seed=1001, max_fe=2000, archive=arch)
    assert p.nfe == 2000 == len(arch)
    p2 = Zdt1Like(6)
    res2 = nsga2(p2, seed=1001, max_fe=2000)
    assert np.array_equal(res.F, res2.F)
    F = res.F
    # F[:,1] = f2 = -lam >= 1 - sqrt(f1)
    assert np.all(F[:, 1] >= 1.0 - np.sqrt(F[:, 0]) - 1e-9)


def test_metrics_known_values():
    P = np.array([[1.0, 2.0], [2.0, 3.0]])   # trade-off pair: f1bar minimised, lam maximised
    assert M.hypervolume(P, (3.0, 0.0)) == pytest.approx(5.0)   # 1*2 + 1*3
    assert M.hypervolume(np.array([[1.0, 3.0], [2.0, 2.0]]), (3.0, 0.0)) == pytest.approx(6.0)   # second point dominated
    assert M.hypervolume(np.vstack([P, [[2.5, 1.0]]]), (3.0, 0.0)) == pytest.approx(5.0)   # dominated point ignored
    assert M.igd(P, P) == 0.0
    assert M.igd(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])) == pytest.approx(5.0)
    U = np.column_stack([np.linspace(0, 1, 6), np.linspace(0, 1, 6)])
    assert M.spacing(U) == pytest.approx(0.0, abs=1e-12)
    assert M.spacing(np.array([[0.0, 0.0], [1.0, 0.0], [1.1, 0.0]])) > 0
    nd = M.non_dominated(np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 2.0], [0.5, 0.5]]))
    assert len(nd) == 3   # (0.5,0.5), (1,1), (2,3); (3,2) is dominated by (2,3)


def test_nondominated_helper():
    F = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 4.0], [4.0, 1.0], [2.5, 3.5]])
    assert sorted(nondominated(F).tolist()) == [0, 1, 3]


def test_statistics():
    assert S.a12([4, 5, 6], [1, 2, 3]) == 1.0 and S.a12([1, 2, 3], [4, 5, 6]) == 0.0
    assert S.a12([1, 2, 3], [1, 2, 3]) == 0.5
    adj = S.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    u, p = S.mann_whitney([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert p < 0.05
    st, p = S.friedman(np.array([[1, 2, 3], [1, 3, 2], [1, 2, 3], [1, 2, 3]]))
    assert 0 <= p <= 1
    # k = 2: reduces to a sign-type test; 6 blocks all favouring treatment 2 -> Q = 6, p = chi2.sf(6, 1)
    q2, p2 = S.friedman(np.array([[1, 2], [1, 3], [2, 3], [1, 5], [0, 1], [2, 4]], dtype=float))
    assert q2 == pytest.approx(6.0) and p2 == pytest.approx(0.014306, abs=1e-5)
    d = S.describe(np.arange(1.0, 6.0))
    assert d["median"] == 3.0 and d["min"] == 1.0 and d["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))


def _model(mu=0.1, bc="CF"):
    from section_model import ColumnModel
    from materials import AL_SIC
    return ColumnModel(AL_SIC, mu=mu, bc=bc, N=10, nq=16)


def test_design_problem_real_model():
    p = DesignProblem(_model(0.1), record=True)
    u = np.concatenate([np.full(6, 1.0), np.full(6, 1.0), np.full(6, 0.5)])   # b = h = h0, Vr = 0.5
    f1, lam, cv = p.evaluate(u)
    assert np.isfinite(f1) and lam > 0 and cv == 0.0 and p.nfe == 1 and len(p.archive) == 1
    # weak-axis violation: h > b
    u2 = u.copy()
    u2[6:12] = 1.4
    u2[:6] = 0.6
    assert p.evaluate(u2)[2] > 0
    # kappa violation: h/l < 1 needs l/h0 > h/h0 -> mu = 0.5 with h = 0.2 h0
    p3 = DesignProblem(_model(0.5))
    u3 = np.concatenate([np.full(6, 0.4), np.full(6, 0.2), np.full(6, 0.5)])
    assert p3.evaluate(u3)[2] > 0
    # solver breakdown path is infeasible with infinite violation
    class Broken(DesignProblem):
        pass
    pb = Broken(_model(0.1))
    pb.model.rb.d1 = pb.model.rb.d1 * np.nan
    f = pb.evaluate(u)
    assert f[2] == np.inf and f[0] == np.inf


def test_real_model_short_runs_and_timing():
    m = _model(0.1)
    p = DesignProblem(m)
    t0 = time.time()
    out = awsm_search(p, seed=1001, B=12000, fe_per_run=1500)
    dt = time.time() - t0
    assert out["fe"] <= 12000 and out.get("failed") is None and len(out["front"]) >= 2
    assert np.all(out["front"][:, 2] == 0.0)             # only feasible points on the front
    print(f"\nreal-model AWSM short run: {out['fe']} FE in {dt:.1f} s ({1e3 * dt / out['fe']:.2f} ms/FE)")
    p2 = DesignProblem(_model(0.1))
    res = nsga2(p2, seed=1001, max_fe=500)
    assert p2.nfe == 500


def test_moead_exact_fe_reproducible_and_front():
    from optimizers import moead
    p = Zdt1Like(6)
    arch = []
    r = moead(p, seed=1001, max_fe=4000, archive=arch)
    assert p.nfe == 4000 == len(arch) == r["fe"]
    p2 = Zdt1Like(6)
    r2 = moead(p2, seed=1001, max_fe=4000)
    assert np.array_equal(r["X"], r2["X"])                                   # reproducible
    f1, lam = np.array(arch)[:, 0], np.array(arch)[:, 1]
    assert np.all(lam <= np.sqrt(f1) - 1.0 + 1e-9)                           # cannot beat the analytic front
    assert np.median((np.sqrt(r["f1"]) - 1.0) - r["lam"]) < 0.1              # population close to the front
    assert np.ptp(r["f1"]) > 0.5                                             # spread along the front


def test_moead_deb_rule_prefers_feasible():
    from optimizers import moead

    class Constrained(Zdt1Like):
        def evaluate(self, u):
            self.nfe += 1
            g = 1.0 + 9.0 * np.mean(u[1:])
            cv = float(max(0.0, 0.3 - u[1]))                                   # feasible iff x1 >= 0.3
            return float(u[0]), float(-(g * (1 - np.sqrt(u[0] / g)))), cv
    p = Constrained(6)
    r = moead(p, seed=3, max_fe=6000)
    assert np.mean(r["cv"] == 0.0) > 0.9                                      # population ends feasible
