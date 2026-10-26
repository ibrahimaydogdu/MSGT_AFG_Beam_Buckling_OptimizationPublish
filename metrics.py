"""Multi-objective metrics (Section 3.6, Eqs. 87-89). Points are given as (f1bar, lam):
f1bar minimised, lam maximised. Internally converted to minimisation (f1bar, -lam)."""
import numpy as np


def _to_min(P):
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    return np.column_stack([P[:, 0], -P[:, 1]])


def non_dominated(P):
    """Non-dominated subset of P ((f1bar, lam) rows), sorted by f1bar."""
    M = _to_min(P)
    if len(M) == 0:
        return M
    order = np.lexsort((M[:, 1], M[:, 0]))
    keep, best = [], np.inf
    for i in order:
        if M[i, 1] < best:
            keep.append(i)
            best = M[i, 1]
    out = np.asarray(P, dtype=float).reshape(-1, 2)[keep]
    return out


def hypervolume(P, ref):
    """Eq. (87): area dominated by P up to the anti-optimal reference ref = (r1, r2),
    r1 >= all f1bar, r2 <= all lam."""
    A = non_dominated(P)
    A = A[(A[:, 0] < ref[0]) & (A[:, 1] > ref[1])]
    if len(A) == 0:
        return 0.0
    A = A[np.argsort(A[:, 0])]
    hv, prev_lam = 0.0, ref[1]
    # sweep from the largest f1bar downwards: strips between consecutive f1bar values
    f1 = A[:, 0]
    lam = A[:, 1]
    for k in range(len(A)):
        right = f1[k + 1] if k + 1 < len(A) else ref[0]
        hv += (right - f1[k]) * (lam[k] - ref[1])
    return float(hv)


def igd(A, Pstar):
    """Eq. (88) in (f1bar, lam) space (callers normalise both objectives beforehand)."""
    A = np.asarray(A, dtype=float).reshape(-1, 2)
    Ps = np.asarray(Pstar, dtype=float).reshape(-1, 2)
    d = np.sqrt(((Ps[:, None, :] - A[None, :, :]) ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())


def spacing(A):
    """Eq. (89), L1 nearest-neighbour distances."""
    A = np.asarray(A, dtype=float).reshape(-1, 2)
    n = len(A)
    if n < 2:
        return float("nan")
    d = np.abs(A[:, None, :] - A[None, :, :]).sum(axis=2)
    np.fill_diagonal(d, np.inf)
    di = d.min(axis=1)
    return float(np.sqrt(((di - di.mean()) ** 2).sum() / (n - 1)))


def normalise(P, f1_lo, f1_hi, lam_lo, lam_hi):
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    return np.column_stack([(P[:, 0] - f1_lo) / (f1_hi - f1_lo), (P[:, 1] - lam_lo) / (lam_hi - lam_lo)])
