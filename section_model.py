"""Design vector -> profiles, mass, P_cr, constraints (Sections 2.4, 2.6, 3.1).

X = [P^b_1..6, P^h_1..6, P^V_1..6]  (Eq. 46), amplitudes dimensional for b, h [m],
V_r dimensionless. Reference geometry: h0, b0 = h0, L = zeta*h0 (zeta = 20).
"""
import numpy as np

from bspline import BSplineProfile
from materials import MaterialPair, AL_SIC
from msgt_ritz import RitzBasis, buckling_load, D_coefficients, SolverFailure

L_OVER_H0 = 20.0


class ColumnModel:
    def __init__(self, pair: MaterialPair = AL_SIC, mu=0.0, h0=1.0e-6, bc="CF",
                 bc_type="BC1", N=12, nq=24, n_cp=6, p=3, basis="legendre",
                 l_mode="const", l_ratio=1.0, h_min=0.2):
        """mu = l/h0 (constant length scale, l0=l1=l2=l). l_mode:
        'const'  : l = mu*h0 in both phases and everywhere (Al/SiC main campaign);
        'mixture': l(x) = l_L + (l_R - l_L) V_r with l_L = mu*h0, l_R = l_L/2 (synthetic pair);
        'phase'  : l(x) = l [1 + (r-1) V_r], r = l_ratio (phase-dependent ablation)."""
        self.pair, self.mu, self.h0 = pair, mu, h0
        self.b0 = h0
        self.L = L_OVER_H0 * h0
        self.bc, self.bc_type, self.N = bc, bc_type, N
        self.l_mode, self.l_ratio = l_mode, l_ratio
        self.h_min = h_min          # lower bound of P^h / h0 (Table 3: 0.2; 0.5 and 0.4 in the extra campaigns)
        self.prof = BSplineProfile(n_cp, p, nq)
        self.n = n_cp
        self.rb = RitzBasis(self.prof.xi, self.prof.w, N, bc, bc_type, basis)
        self.I0 = self.b0 * self.h0**3 / 12.0

    # ---- profiles -----------------------------------------------------
    def split(self, X):
        X = np.asarray(X, dtype=float)
        n = self.n
        return X[:n], X[n:2 * n], X[2 * n:3 * n]

    def length_scale(self, Vr):
        l = self.mu * self.h0
        if self.l_mode == "const":
            return l * np.ones_like(Vr)
        if self.l_mode == "mixture":
            return l + (0.5 * l - l) * Vr
        if self.l_mode == "phase":
            return l * (1.0 + (self.l_ratio - 1.0) * Vr)
        raise ValueError("unknown l_mode")

    def fields(self, X):
        Pb, Ph, PV = self.split(X)
        b, h, Vr = self.prof(Pb), self.prof(Ph), self.prof(PV)
        E, G, rho = self.pair.mix(Vr)
        return {"b": b, "h": h, "Vr": Vr, "A": b * h, "I": b * h**3 / 12.0,
                "E": E, "G": G, "rho": rho, "l": self.length_scale(Vr)}

    # ---- objectives ---------------------------------------------------
    def mass(self, X):
        f = self.fields(X)
        if f["rho"] is None:
            raise ValueError("mass undefined: densities of the pair are not specified")
        return self.L * float(np.sum(self.prof.w * f["rho"] * f["A"]))        # Eq. (48)

    def m0(self):
        return self.pair.left.rho * self.b0 * self.h0 * self.L                # Section 2.6

    def pcr(self, X, return_info=False):
        f = self.fields(X)
        l = f["l"]
        return buckling_load(self.rb, f["E"], f["G"], f["A"], f["I"], l, l, l, self.L,
                             return_info=return_info)

    def lam(self, P):
        return P * self.L**2 / (self.pair.right.E * self.I0)                   # Eq. (65)

    def objectives(self, X):
        """(f1bar, lambda) or raises SolverFailure."""
        P = self.pcr(X)
        return self.mass(X) / self.m0(), self.lam(P)

    # ---- constraints (Table 3) -----------------------------------------
    def constraints(self, X, kappa_min=1.0):
        """g_j(X) <= 0, all dimensionless. Table 3 constraints g1-g27 with the two-sided box bounds written
        out as separate rows (vector length 45: 12+12 box rows, 6 weak-axis, 12 V_r box, 2 V_bar, 1 kappa)."""
        Pb, Ph, PV = self.split(X)
        g = [0.2 - Pb / self.b0, Pb / self.b0 - 1.5,        # width bounds
             self.h_min - Ph / self.h0, Ph / self.h0 - 1.5,  # height bounds
             (Ph - Pb) / self.h0,                           # weak axis: P^h <= P^b (dimensionless, b0 = h0)
             0.0 - PV, PV - 1.0]                            # volume fraction bounds
        Vbar = float(np.sum(self.prof.w * (self.prof.B @ PV)))
        g.append(np.array([0.10 - Vbar, Vbar - 0.90]))
        f = self.fields(X)
        with np.errstate(divide="ignore"):
            ratio = f["h"] / f["l"] if np.all(f["l"] > 0) else np.array([np.inf])
        g.append(np.array([kappa_min - float(np.min(ratio))]))
        return np.concatenate([np.atleast_1d(v) for v in g])

    def violation(self, X, kappa_min=1.0):
        """CV of Eq. (74)."""
        return float(np.sum(np.maximum(0.0, self.constraints(X, kappa_min))))

    def screening(self, X):
        """eps_s = 0.6415 (D1/EI) (h/L)^2, maximum over the span (Section 3.1)."""
        f = self.fields(X)
        D1, _ = D_coefficients(f["E"], f["G"], f["A"], f["I"], f["l"], f["l"], f["l"])
        return float(np.max(0.6415 * D1 / (f["E"] * f["I"]) * (f["h"] / self.L) ** 2))
