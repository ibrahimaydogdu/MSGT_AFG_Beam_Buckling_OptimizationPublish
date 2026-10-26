"""Univariate B-Spline utilities (Eqs. 38-47 of the manuscript).

Open (clamped) uniform knot vector, Cox-de Boor recursion (Eqs. 43-44) and
span-wise Gauss-Legendre quadrature nodes (Section 2.5). Control point
abscissae are fixed (Eq. 45); only the amplitudes are design variables.
"""
import numpy as np


def open_uniform_knots(n, p):
    """Eq. (41)-(42): p+1 repeated end knots, uniformly spaced interior knots.
    n control points, degree p -> n+p+1 knots (n=6, p=3 -> 10 knots)."""
    if n < p + 1:
        raise ValueError("n must be >= p+1")
    n_int = n - p - 1
    interior = np.linspace(0.0, 1.0, n_int + 2)[1:-1] if n_int > 0 else np.array([])
    return np.concatenate((np.zeros(p + 1), interior, np.ones(p + 1)))


def basis_matrix(xi, knots, p):
    """Cox-de Boor basis matrix B[i, k] = N_{k,p}(xi_i) (Eqs. 43-44, 0/0 = 0).
    The right end point xi = 1 belongs to the last non-empty knot span."""
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    n = len(knots) - p - 1
    nk = len(knots)
    # degree 0
    B = np.zeros((len(xi), nk - 1))
    for k in range(nk - 1):
        if knots[k] < knots[k + 1]:
            B[:, k] = (xi >= knots[k]) & (xi < knots[k + 1])
    last = np.max(np.nonzero(knots[:-1] < knots[1:])[0])
    B[xi == knots[-1], last] = 1.0
    for d in range(1, p + 1):
        Bn = np.zeros((len(xi), nk - 1 - d))
        for k in range(nk - 1 - d):
            den1 = knots[k + d] - knots[k]
            den2 = knots[k + d + 1] - knots[k + 1]
            if den1 > 0:
                Bn[:, k] += (xi - knots[k]) / den1 * B[:, k]
            if den2 > 0:
                Bn[:, k] += (knots[k + d + 1] - xi) / den2 * B[:, k + 1]
        B = Bn
    return B[:, :n]


def span_quadrature(knots, nq):
    """Gauss-Legendre nodes/weights on xi in [0,1], independently on each
    non-empty knot span (Section 2.5)."""
    u = np.unique(knots)
    g, w = np.polynomial.legendre.leggauss(nq)
    xs, ws = [], []
    for a, b in zip(u[:-1], u[1:]):
        xs.append(0.5 * (b - a) * g + 0.5 * (b + a))
        ws.append(0.5 * (b - a) * w)
    return np.concatenate(xs), np.concatenate(ws)


class BSplineProfile:
    """Precomputed basis matrix at quadrature nodes for fast evaluation of
    profile(xi_q) = B @ P for any control amplitude vector P."""

    def __init__(self, n=6, p=3, nq=20):
        self.n, self.p = n, p
        self.knots = open_uniform_knots(n, p)
        self.xi, self.w = span_quadrature(self.knots, nq)
        self.B = basis_matrix(self.xi, self.knots, p)

    def __call__(self, P):
        return self.B @ np.asarray(P, dtype=float)

    def evaluate(self, P, xi):
        return basis_matrix(xi, self.knots, self.p) @ np.asarray(P, dtype=float)
