"""Rayleigh-Ritz MSGT buckling solver (Section 2.5, Eqs. 49-64).

Generalised eigenproblem (K - P K_G) a = 0 with
  K_ij  = 1/L^3 int D1 phi_i'' phi_j'' dxi + 1/L^5 int D2 phi_i''' phi_j''' dxi   (58)
  KG_ij = 1/L   int phi_i' phi_j' dxi                                              (59)
  D1 = E I + G A (2 l0^2 + 8/15 l1^2 + l2^2),  D2 = G I (2 l0^2 + 4/5 l1^2)
Trial-function space (Eqs. 60-63), written as xi^m (1-xi)^c q(xi), deg q <= N-1:
  C-F : m = 2 (BC1) | 3 (BC2), c = 0      C-P : m = 2 | 3, c = 1
Two bases span exactly the same space and therefore give the same Ritz
eigenvalues in exact arithmetic:
  basis="monomial"  phi_i = xi^(i+1) [(1-xi)] (BC1) / xi^(i+2) [(1-xi)] (BC2): Eqs. (60)-(63)
  basis="legendre"  phi_i = xi^m (1-xi)^c P~_(i-1)(xi), shifted Legendre P~: well
                    conditioned for N > 10 (Section 2.5, [12, 37]).
"""
import numpy as np
from numpy.polynomial import legendre as _leg
from scipy.linalg import eigh, LinAlgError
from math import comb


class SolverFailure(Exception):
    """Numerical failure (non-positive eigenvalue, singular matrix, NaN/Inf).
    The optimiser treats it as an infeasible design (Section 3.1)."""


def _monomial_terms(i, bc, bc_type):
    a = i + 1 if bc_type == "BC1" else i + 2
    if bc == "CF":
        return [(1.0, a)]
    if bc == "CP":
        return [(1.0, a), (-1.0, a + 1)]
    raise ValueError("bc must be 'CF' or 'CP'")


def _mono_derivs(terms, xi, nder):
    out = []
    for n in range(nder + 1):
        s = np.zeros_like(xi)
        for c, a in terms:
            if a - n < 0:
                continue
            ff = 1.0
            for t in range(n):
                ff *= (a - t)
            s += c * ff * xi ** (a - n)
        out.append(s)
    return out


def _legendre_derivs(k, xi, nder):
    """d^r/dxi^r of shifted Legendre P~_k(xi) = P_k(2 xi - 1), r = 0..nder."""
    t = 2.0 * xi - 1.0
    c = np.zeros(k + 1)
    c[k] = 1.0
    out = []
    for r in range(nder + 1):
        cr = _leg.legder(c, r) if r > 0 else c
        out.append((2.0 ** r) * _leg.legval(t, cr) if cr.size else np.zeros_like(xi))
    return out


def _prefactor_derivs(bc, bc_type, xi, nder):
    """f(xi) = xi^m (1-xi)^c and its derivatives (exact, via expanded terms)."""
    m = 2 if bc_type == "BC1" else 3
    if bc == "CF":
        terms = [(1.0, m)]
    else:
        terms = [(1.0, m), (-1.0, m + 1)]
    return _mono_derivs(terms, xi, nder)


def trial_functions(xi, N, bc="CF", bc_type="BC1", basis="legendre"):
    """phi, phi', phi'', phi''' arrays of shape (N, len(xi))."""
    xi = np.asarray(xi, dtype=float)
    out = [np.empty((N, len(xi))) for _ in range(4)]
    if basis == "monomial":
        for i in range(1, N + 1):
            d = _mono_derivs(_monomial_terms(i, bc, bc_type), xi, 3)
            for k in range(4):
                out[k][i - 1] = d[k]
    elif basis == "legendre":
        fd = _prefactor_derivs(bc, bc_type, xi, 3)
        for i in range(1, N + 1):
            qd = _legendre_derivs(i - 1, xi, 3)
            for n in range(4):
                s = np.zeros_like(xi)
                for k in range(n + 1):
                    s += comb(n, k) * fd[k] * qd[n - k]
                out[n][i - 1] = s
    else:
        raise ValueError("basis must be 'monomial' or 'legendre'")
    return out


class RitzBasis:
    """Trial functions precomputed at fixed quadrature nodes (reused for every
    design evaluation in an optimisation run)."""

    def __init__(self, xi, w, N=10, bc="CF", bc_type="BC1", basis="legendre"):
        self.xi = np.asarray(xi, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.N, self.bc, self.bc_type, self.basis = N, bc, bc_type, basis
        _, self.d1, self.d2, self.d3 = trial_functions(self.xi, N, bc, bc_type, basis)


def D_coefficients(E, G, A, I, l0, l1, l2):
    """Eqs. (32)-(34): D1 and D2 at nodes."""
    D1 = E * I + G * A * (2.0 * l0**2 + (8.0 / 15.0) * l1**2 + l2**2)
    D2 = G * I * (2.0 * l0**2 + 0.8 * l1**2)
    return D1, D2


def assemble(rb, D1, D2, L):
    """K and KG (Eqs. 58-59) from a RitzBasis and node values of D1, D2."""
    K = (rb.d2 * (rb.w * D1)) @ rb.d2.T / L**3 + (rb.d3 * (rb.w * D2)) @ rb.d3.T / L**5
    KG = (rb.d1 * rb.w) @ rb.d1.T / L
    return 0.5 * (K + K.T), 0.5 * (KG + KG.T)


def buckling_load(rb, E, G, A, I, l0, l1, l2, L, return_info=False):
    """Smallest positive eigenvalue P_cr of Eq. (55) (Eq. 64).

    rb : RitzBasis; E, G, A, I, l0, l1, l2: arrays at rb.xi. Raises SolverFailure."""
    try:
        D1, D2 = D_coefficients(E, G, A, I, l0, l1, l2)
        if not (np.all(np.isfinite(D1)) and np.all(np.isfinite(D2))):
            raise SolverFailure("non-finite stiffness coefficients")
        K, KG = assemble(rb, D1, D2, L)
        s = 1.0 / np.sqrt(np.diag(KG))  # diagonal scaling: same eigenvalues, better conditioning
        Ks = K * s[:, None] * s[None, :]
        Gs = KG * s[:, None] * s[None, :]
        lam = eigh(Ks, Gs, eigvals_only=True)
        pos = lam[lam > 0.0]
        if pos.size == 0 or not np.isfinite(pos[0]):
            raise SolverFailure("no positive finite eigenvalue")
        P = float(pos[0])
        if not return_info:
            return P
        gev = np.linalg.eigvalsh(Gs)
        info = {
            "cond_KG_scaled": float(gev[-1] / gev[0]) if gev[0] > 0 else float("inf"),
            "min_eig_KG_scaled": float(gev[0]),
            "sym_err_K": float(np.max(np.abs(K - K.T)) / max(np.max(np.abs(K)), 1e-300)),
            "eigenvalues": lam,
        }
        return P, info
    except (LinAlgError, ValueError, FloatingPointError) as exc:
        raise SolverFailure(str(exc)) from exc
