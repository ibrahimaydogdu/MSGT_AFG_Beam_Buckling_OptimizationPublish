"""Optimisation problem wrapper (Section 3.1): normalised design vector, objectives,
constraint violation (Eq. 74), evaluation counting and archive.

u = [P^b/b0 (6), P^h/h0 (6), P^V (6)]  with bounds [0.2,1.5]^12 x [0,1]^6 (Table 3, g1-g24).
evaluate(u) -> (f1bar, lam, cv): f1bar = m/m0 is MINIMISED, lam = P_cr L^2/(E_R I_0) is MAXIMISED.
A solver failure returns (inf, 0, inf): infeasible with infinite violation (Section 3.1).
"""
import numpy as np

from msgt_ritz import SolverFailure
from section_model import ColumnModel


class DesignProblem:
    def __init__(self, model: ColumnModel, kappa_min=1.0, record=False):
        self.model = model
        self.kappa_min = kappa_min
        n = model.n
        self.dim = 3 * n
        self.lb = np.array([0.2] * n + [model.h_min] * n + [0.0] * n)
        self.ub = np.array([1.5] * (2 * n) + [1.0] * n)
        self.nfe = 0
        self.record = record
        self.archive = []  # rows: [f1bar, lam, cv, u...]
        self.n_constraints = len(self.constraints_vec(0.5 * (self.lb + self.ub)))
        self._vbar_w = model.prof.B.T @ model.prof.w   # V_bar = P^V . _vbar_w

    def to_dimensional(self, u):
        u = np.asarray(u, dtype=float)
        n = self.model.n
        return np.concatenate([u[:n] * self.model.b0, u[n:2 * n] * self.model.h0, u[2 * n:]])

    def constraints_vec(self, u):
        return self.model.constraints(self.to_dimensional(u), self.kappa_min)

    def sample(self, rng, n):
        """Feasibility-biased initial designs (used by BOTH algorithms): b = max(b, h) enforces the
        weak-axis rows, V_r resampled until 0.10 <= V_bar <= 0.90; box bounds hold by construction.
        rng: numpy Generator or the numpy.random module."""
        m = self.model.n
        h = rng.uniform(self.lb[m:2 * m], self.ub[m:2 * m], (n, m))
        b = np.maximum(rng.uniform(self.lb[:m], self.ub[:m], (n, m)), h)
        V = rng.uniform(0.0, 1.0, (n, m))
        for _ in range(50):
            vb = V @ self._vbar_w
            bad = (vb < 0.10) | (vb > 0.90)
            if not bad.any():
                break
            V[bad] = rng.uniform(0.0, 1.0, (int(bad.sum()), m))
        return np.hstack([b, h, V])

    def reset(self):
        self.nfe = 0
        self.archive = []

    def evaluate(self, u):
        """One function evaluation (counted)."""
        self.nfe += 1
        try:
            X = self.to_dimensional(u)
            f1, lam = self.model.objectives(X)
            cv = float(np.sum(np.maximum(0.0, self.model.constraints(X, self.kappa_min))))
            if not (np.isfinite(f1) and np.isfinite(lam) and np.isfinite(cv)):
                raise SolverFailure("non-finite result")
        except (SolverFailure, FloatingPointError):
            f1, lam, cv = np.inf, 0.0, np.inf
        if self.record:
            self.archive.append(np.concatenate([[f1, lam, cv], u]))
        return float(f1), float(lam), float(cv)


def deb_better(f_a, cv_a, f_b, cv_b):
    """Deb's feasibility-first rule for a scalar objective to be minimised:
    True if candidate a is strictly preferred to b (Section 3.1)."""
    if cv_a == 0.0 and cv_b > 0.0:
        return True
    if cv_a > 0.0 and cv_b == 0.0:
        return False
    if cv_a > 0.0 and cv_b > 0.0:
        return cv_a < cv_b
    return f_a < f_b
