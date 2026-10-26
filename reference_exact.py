"""Independent reference solver for V4 (VALIDATION_SPEC): the symbolic Ritz flow of
00_Reference/cantilevermsgt.m and proppedmsgt.m re-implemented with exact rational
arithmetic (sympy) and 60-digit eigenvalues (mpmath). It shares no code with
msgt_ritz.py. MATLAB itself is not used (not available here)."""
import sympy as sp
import mpmath as mp


def _poly(expr, x):
    return sp.Poly(sp.expand(expr), x)


def min_dimensionless_load(EL, ER, nu, h1, b1, L, lL, lR, e, f, n, N, bc, dps=60):
    """Smallest positive k = P L^2/(E_R I0) of cantilevermsgt.m (bc='CF') or
    proppedmsgt.m (bc='CP', basis x^(i+1)(L-x)); all inputs rational numbers."""
    mp.mp.dps = dps
    x = sp.Symbol("x")
    R = sp.Rational
    EL, ER, nu, h1, b1, L, lL, lR, e, f = [R(v) for v in (EL, ER, nu, h1, b1, L, lL, lR, e, f)]
    GL, GR = EL / (2 * (1 + nu)), ER / (2 * (1 + nu))
    t = (x / L) ** n
    E = EL + t * (ER - EL)
    G = GL + t * (GR - GL)
    l0 = lL + t * (lR - lL)
    b = b1 * (1 - e * x / L)
    h = h1 * (1 - f * x / L)
    I, A = b * h**3 / 12, b * h
    c1 = _poly(E * I + 2 * G * A * l0**2 + R(8, 15) * G * A * l0**2 + G * A * l0**2, x)
    c2 = _poly(I * (2 * G * l0**2 + R(4, 5) * G * l0**2), x)
    d1, d2, d3 = [], [], []
    for i in range(1, N + 1):
        w = x ** (i + 1) if bc == "CF" else x ** (i + 1) * (L - x)
        d1.append(_poly(sp.diff(w, x), x))
        d2.append(_poly(sp.diff(w, x, 2), x))
        d3.append(_poly(sp.diff(w, x, 3), x))

    def integ(p):
        return p.integrate().eval(L) - p.integrate().eval(0)

    K = mp.matrix(N, N)
    KG = mp.matrix(N, N)
    for i in range(N):
        for j in range(i, N):
            k_ij = integ(c1 * d2[i] * d2[j]) + integ(c2 * d3[i] * d3[j])
            g_ij = integ(d1[i] * d1[j])
            K[i, j] = K[j, i] = mp.mpf(k_ij.p) / mp.mpf(k_ij.q)
            KG[i, j] = KG[j, i] = mp.mpf(g_ij.p) / mp.mpf(g_ij.q)
    Rm = mp.cholesky(KG)
    Ri = mp.inverse(Rm)
    M = Ri * K * Ri.T
    M = (M + M.T) / 2
    ev = sorted(mp.eigsy(M, eigvals_only=True))
    P = [v for v in ev if v > 0][0]
    I0 = b1 * h1**3 / 12
    return P * mp.mpf(L.p) ** 2 / mp.mpf(L.q) ** 2 / (mp.mpf((ER * I0).p) / mp.mpf((ER * I0).q))
