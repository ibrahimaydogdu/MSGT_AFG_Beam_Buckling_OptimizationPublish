"""Statistical protocol (Section 3.7): descriptive statistics, two-sided Mann-Whitney U,
Holm step-down, Vargha-Delaney A12 (Eq. 90), Friedman.
(File name avoids shadowing the standard library module `statistics`.)"""
import numpy as np
from scipy import stats


def describe(x):
    x = np.asarray(x, dtype=float)
    return {"min": float(x.min()), "median": float(np.median(x)), "mean": float(x.mean()),
            "max": float(x.max()), "std": float(x.std(ddof=1))}


def a12(x, y):
    """Eq. (90): A12 = (R1 - n1 (n1+1)/2) / (n1 n2), R1 = rank sum of x in the pooled sample
    (average ranks for ties). A12 > 0.5: x tends to be larger."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.rankdata(np.concatenate([x, y]))
    n1, n2 = len(x), len(y)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def mann_whitney(x, y):
    res = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return float(res.statistic), float(res.pvalue)


def holm(pvals):
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def friedman(blocks):
    """blocks: array (n_blocks, k_treatments); returns (statistic, p-value)."""
    b = np.asarray(blocks, dtype=float)
    n, k = b.shape
    if k >= 3:
        res = stats.friedmanchisquare(*[b[:, j] for j in range(k)])
        return float(res.statistic), float(res.pvalue)
    # k = 2: scipy requires k >= 3; same statistic computed directly (average ranks, tie correction)
    r = np.apply_along_axis(stats.rankdata, 1, b)
    Rj = r.sum(axis=0)
    ties = 0.0
    for row in b:
        _, c = np.unique(row, return_counts=True)
        ties += float(np.sum(c**3 - c))
    num = 12.0 / (n * k * (k + 1)) * np.sum(Rj**2) - 3.0 * n * (k + 1)
    q = num / (1.0 - ties / (n * k * (k**2 - 1)))
    return float(q), float(stats.chi2.sf(q, k - 1))
