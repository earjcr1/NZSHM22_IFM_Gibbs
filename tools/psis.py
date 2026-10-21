"""Pareto-smoothed importance sampling (Vehtari, Gelman and Gabry 2017; as in arviz's _psislw and _gpdfit)."""
import numpy as np


def gpdfit(x):
    """Zhang-Stephens empirical Bayes fit of a generalized Pareto distribution to sorted positive exceedances x."""
    prior_bs, prior_k = 3, 10
    n = len(x)
    m = 30 + int(n ** 0.5)
    b = 1 - np.sqrt(m / (np.arange(1, m + 1, dtype=float) - 0.5))
    b /= prior_bs * x[int(n / 4 + 0.5) - 1]
    b += 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    L = n * (np.log(-(b / k)) - k - 1)
    w = 1 / np.exp(L - L[:, None]).sum(axis=1)
    keep = w >= 10 * np.finfo(float).eps
    w, b = w[keep] / w[keep].sum(), b[keep]
    bp = np.sum(b * w)
    kp = np.log1p(-bp * x).mean()
    sigma = -kp / bp
    kp = (n * kp + prior_k * 0.5) / (n + prior_k)
    return kp, sigma


def gpinv(p, k, sigma):
    if abs(k) < np.finfo(float).eps:
        return -np.log1p(-p) * sigma
    return np.expm1(-k * np.log1p(-p)) / k * sigma


def psislw(lw):
    """Smoothed, normalized log weights and the Pareto k-hat, from raw log weights lw (one per draw)."""
    x = np.asarray(lw, float) - np.max(lw)
    S = len(x)
    M = int(np.ceil(min(0.2 * S, 3 * np.sqrt(S))))
    order = np.argsort(x)
    cutoff = max(x[order[-M - 1]], np.log(np.finfo(float).tiny))
    tail = np.flatnonzero(x > cutoff)
    k = np.inf
    if len(tail) > 4:
        ts = tail[np.argsort(x[tail])]
        ex = np.exp(x[ts]) - np.exp(cutoff)
        k, sigma = gpdfit(ex)
        if np.isfinite(k):
            p = np.arange(0.5, len(ts)) / len(ts)
            x[ts] = np.log(gpinv(p, k, sigma) + np.exp(cutoff))
            x[x > 0] = 0
    x -= np.log(np.sum(np.exp(x - x.max()))) + x.max()
    return x, k
