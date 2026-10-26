"""Tests for ColumnModel quantities not covered by the V1-V7 gate (Critic Round 09, K-D):
lam, m0, objectives, constraints, l-modes, and non-uniform B-spline P_cr against an
independent profile evaluation (scipy BSpline, finer quadrature)."""
import os
import sys

import numpy as np
import pytest
from scipy.interpolate import BSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from section_model import ColumnModel  # noqa: E402
from materials import AL_SIC, SYNTHETIC  # noqa: E402
from msgt_ritz import RitzBasis, buckling_load  # noqa: E402
from objectives import DesignProblem  # noqa: E402
from bspline import open_uniform_knots  # noqa: E402


def design(m, rng):
    Pb = rng.uniform(0.6, 1.4, 6)
    Ph = np.minimum(rng.uniform(0.3, 1.0, 6), Pb)
    PV = rng.uniform(0.2, 0.8, 6)
    return np.concatenate([Pb * m.b0, Ph * m.h0, PV])


def test_pcr_nonuniform_vs_independent_profiles():
    m = ColumnModel(AL_SIC, mu=0.15, bc="CF", bc_type="BC1", N=12, nq=24)
    X = design(m, np.random.default_rng(3))
    kn = open_uniform_knots(6, 3)
    g, w = np.polynomial.legendre.leggauss(64)
    u = np.unique(kn)
    xi = np.concatenate([0.5 * (b - a) * g + 0.5 * (b + a) for a, b in zip(u[:-1], u[1:])])
    wq = np.concatenate([0.5 * (b - a) * w for a, b in zip(u[:-1], u[1:])])
    b = BSpline(kn, X[:6], 3)(xi)
    h = BSpline(kn, X[6:12], 3)(xi)
    V = BSpline(kn, X[12:], 3)(xi)
    E = AL_SIC.left.E + (AL_SIC.right.E - AL_SIC.left.E) * V
    G = AL_SIC.left.G + (AL_SIC.right.G - AL_SIC.left.G) * V
    l = np.full_like(xi, 0.15 * m.h0)
    rb = RitzBasis(xi, wq, 12, "CF", "BC1")
    P_ind = buckling_load(rb, E, G, b * h, b * h**3 / 12, l, l, l, m.L)
    assert m.pcr(X) == pytest.approx(P_ind, rel=1e-9)


def test_objectives_scaling_identities():
    m = ColumnModel(AL_SIC, mu=0.1, bc="CF", N=10)
    X = np.concatenate([np.full(6, m.b0), np.full(6, m.h0), np.full(6, 0.0)])   # uniform b0 x h0, pure Al
    f1, lam = m.objectives(X)
    assert f1 == pytest.approx(1.0, rel=1e-12)          # m = rho_Al b0 h0 L = m0
    P = m.pcr(X)
    assert lam == pytest.approx(P * m.L**2 / (AL_SIC.right.E * m.I0), rel=1e-12)   # Eq. (65), E_R = E_SiC
    # classical square Al column: lam = (E_Al/E_SiC) * 2.4674 (Euler cantilever, mu = 0)
    m0 = ColumnModel(AL_SIC, mu=0.0, bc="CF", N=10)
    _, lam0 = m0.objectives(X)
    assert lam0 == pytest.approx(AL_SIC.left.E / AL_SIC.right.E * np.pi**2 / 4, rel=1e-6)


def test_constraints_rows():
    m = ColumnModel(AL_SIC, mu=0.1, bc="CF", N=8)
    ok = np.concatenate([np.full(6, m.b0), np.full(6, m.h0), np.full(6, 0.5)])
    assert m.violation(ok) == 0.0
    bad_w = ok.copy(); bad_w[:6] = 1.6 * m.b0                       # width above 1.5 b0
    assert m.violation(bad_w) == pytest.approx(6 * 0.1, rel=1e-9)
    bad_axis = ok.copy(); bad_axis[6:12] = 1.2 * m.h0               # h > b by 20 % of h0
    assert m.violation(bad_axis) == pytest.approx(6 * 0.2, rel=1e-9)   # dimensionless, not ~1e-6
    bad_v = ok.copy(); bad_v[12:] = 0.95
    assert m.violation(bad_v) == pytest.approx(0.05, rel=1e-9)      # V_bar = 0.95 > 0.90
    low_v = ok.copy(); low_v[12:] = 0.04
    assert m.violation(low_v) == pytest.approx(0.06, rel=1e-9)      # V_bar = 0.04 < 0.10
    m5 = ColumnModel(AL_SIC, mu=0.5, bc="CF", N=8)
    thin = np.concatenate([np.full(6, 0.4 * m5.b0), np.full(6, 0.2 * m5.h0), np.full(6, 0.5)])
    assert m5.violation(thin) == pytest.approx(1.0 - 0.2 / 0.5, rel=1e-9)   # kappa_min - h/l


def test_length_scale_modes():
    V = np.array([0.0, 0.5, 1.0])
    mc = ColumnModel(AL_SIC, mu=0.1, N=6)
    assert np.allclose(mc.length_scale(V), 0.1 * mc.h0)
    mm = ColumnModel(SYNTHETIC, mu=0.2, N=6, l_mode="mixture")
    assert np.allclose(mm.length_scale(V) / mm.h0, [0.2, 0.15, 0.1])        # l_L = 2 l_R
    mp = ColumnModel(AL_SIC, mu=0.1, N=6, l_mode="phase", l_ratio=2.0)
    assert np.allclose(mp.length_scale(V) / mp.h0, [0.1, 0.15, 0.2])        # l[1 + (r-1) V_r]


def test_synthetic_pair_mass_undefined_fails_loudly():
    m = ColumnModel(SYNTHETIC, mu=0.1, N=6)
    X = np.concatenate([np.full(6, m.b0), np.full(6, m.h0), np.full(6, 0.5)])
    with pytest.raises(ValueError):
        m.mass(X)
    p = DesignProblem(m)
    with pytest.raises(ValueError):
        p.evaluate(np.concatenate([np.ones(12), np.full(6, 0.5)]))   # must not be silently infeasible


def test_feasibility_biased_sampler():
    from objectives import DesignProblem
    for mu, hmin, kappa in ((0.1, 0.2, 1.0), (0.2, 0.2, 1.0), (0.5, 0.5, 1.0), (0.2, 0.4, 2.0)):
        p = DesignProblem(ColumnModel(AL_SIC, mu=mu, bc="CF", N=6, h_min=hmin), kappa_min=kappa)
        U = p.sample(np.random.default_rng(0), 500)
        assert np.all(U >= p.lb - 1e-12) and np.all(U <= p.ub + 1e-12)
        cv = [float(np.sum(np.maximum(0, p.constraints_vec(u)))) for u in U]
        assert max(cv) == 0.0                       # all samples feasible (kappa holds since h >= h_min >= kappa*l)
    # the same sampler is reachable through numpy.random (pymoo path uses a Generator too)
    U2 = p.sample(np.random.default_rng(1), 5)
    assert U2.shape == (5, 18)
