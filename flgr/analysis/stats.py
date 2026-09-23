"""Statistical toolkit used for every number in the paper.

* Confidence intervals: BCa bootstrap (10 000 resamples) of the mean over seeds.
* Paired comparisons (same seed = same model and data): exact sign-flip permutation test on the mean
  paired difference (all 2^n sign patterns for n <= 16), plus paired t-test and Wilcoxon signed-rank as
  robustness checks; effect size Cohen's d_z with a bootstrap CI; Holm-Bonferroni correction within a family.
* Equivalence: two one-sided tests (TOST) with a pre-specified margin.
* Rank agreement: Spearman's rho with a Fisher-z interval; cross-seed stability as the mean pairwise
  Spearman with a bootstrap CI over seed pairs.
* Trends (dose-response): linear mixed-effects model  y ~ x + (1 | seed)  (statsmodels MixedLM).
* Concentration: Lorenz curve and Gini coefficient of positive harm.
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy import stats


def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def mean_ci(x, n_boot: int = 10000, level: float = 0.95, seed: int = 0):
    """Mean and BCa bootstrap CI (falls back to percentile when BCa is degenerate)."""
    x = _clean(x)
    if len(x) == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0, sd=np.nan)
    if len(x) < 3 or np.allclose(x, x[0]):
        return dict(mean=float(x.mean()), lo=float(x.min()), hi=float(x.max()), n=len(x), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0)
    rng = np.random.default_rng(seed)
    try:
        r = stats.bootstrap((x,), np.mean, n_resamples=n_boot, confidence_level=level, method="BCa", random_state=rng)
        lo, hi = r.confidence_interval
    except Exception:
        bs = rng.choice(x, (n_boot, len(x))).mean(1)
        lo, hi = np.quantile(bs, [(1 - level) / 2, (1 + level) / 2])
    return dict(mean=float(x.mean()), lo=float(lo), hi=float(hi), n=len(x), sd=float(x.std(ddof=1)))


def signflip_test(d) -> float:
    """Exact two-sided sign-flip permutation p-value for H0: E[d] = 0 (paired differences d)."""
    d = _clean(d)
    n = len(d)
    if n == 0 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    if n <= 16:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    else:
        signs = np.random.default_rng(0).choice([-1, 1], (200000, n))
    null = np.abs((signs * d).mean(1))
    return float((null >= obs - 1e-12).mean())


def cohen_dz(d, n_boot: int = 5000, seed: int = 0):
    d = _clean(d)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return dict(dz=np.nan, lo=np.nan, hi=np.nan)
    dz = d.mean() / d.std(ddof=1)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        s = rng.choice(d, len(d))
        sd = s.std(ddof=1)
        if sd > 0:
            bs.append(s.mean() / sd)
    lo, hi = np.quantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan)
    return dict(dz=float(dz), lo=float(lo), hi=float(hi))


def paired(a, b) -> dict:
    """Full paired comparison of a vs b (arrays aligned by seed)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    out = dict(n=len(d), diff=mean_ci(d), p_perm=signflip_test(d))
    out["p_t"] = float(stats.ttest_rel(a, b).pvalue) if len(d) > 1 and d.std() > 0 else np.nan
    try:
        out["p_w"] = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        out["p_w"] = np.nan
    out["dz"] = cohen_dz(d)
    out["wins"] = int((d > 0).sum())
    return out


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(((k, v) for k, v in pvals.items() if np.isfinite(v)), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    for k, v in pvals.items():
        out.setdefault(k, np.nan)
    return out


def tost(a, b, margin: float) -> dict:
    """Two one-sided paired t-tests: equivalence if |E[a-b]| < margin (p = max of the two)."""
    d = _clean(np.asarray(a, float) - np.asarray(b, float))
    n = len(d)
    if n < 2:
        return dict(p=np.nan, equivalent=False)
    se = d.std(ddof=1) / np.sqrt(n)
    p_lo = 1 - stats.t.cdf((d.mean() + margin) / se, n - 1)
    p_hi = stats.t.cdf((d.mean() - margin) / se, n - 1)
    p = float(max(p_lo, p_hi))
    return dict(p=p, equivalent=p < 0.05, margin=margin, diff=float(d.mean()))


def spearman_ci(x, y, level: float = 0.95) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r = stats.spearmanr(x, y).correlation
    n = len(x)
    if n < 4 or not np.isfinite(r):
        return dict(rho=float(r), lo=np.nan, hi=np.nan, n=n, p=np.nan)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    h = stats.norm.ppf((1 + level) / 2) * 1.06 / np.sqrt(n - 3)     # Fieller et al. variance for Spearman
    p = float(stats.spearmanr(x, y).pvalue)
    return dict(rho=float(r), lo=float(np.tanh(z - h)), hi=float(np.tanh(z + h)), n=n, p=p)


def pairwise_rank_stability(mats: Sequence[np.ndarray], n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean pairwise Spearman correlation between score vectors of different seeds, with a bootstrap
    CI obtained by resampling seeds (not pairs, which are dependent)."""
    V = [np.asarray(v, float) for v in mats]
    k = len(V)
    R = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i + 1, k):
            R[i, j] = R[j, i] = stats.spearmanr(V[i], V[j]).correlation
    iu = np.triu_indices(k, 1)
    mean = float(np.nanmean(R[iu]))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        s = rng.choice(k, k)
        vals = [R[a, b] for a, b in itertools.combinations(s, 2) if a != b]
        if vals:
            bs.append(np.nanmean(vals))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return dict(mean=mean, lo=float(lo), hi=float(hi), matrix=R)


def mixed_trend(x, y, groups) -> dict:
    """y ~ x + (1 | group) by REML; returns slope, 95% CI and p-value."""
    import pandas as pd
    import statsmodels.formula.api as smf
    df = pd.DataFrame(dict(x=np.asarray(x, float), y=np.asarray(y, float), g=np.asarray(groups)))
    df = df[np.isfinite(df.x) & np.isfinite(df.y)]
    try:
        m = smf.mixedlm("y ~ x", df, groups=df["g"]).fit(reml=True)
        ci = m.conf_int().loc["x"].values
        return dict(slope=float(m.params["x"]), lo=float(ci[0]), hi=float(ci[1]), p=float(m.pvalues["x"]), n=len(df))
    except Exception as e:  # singular fits fall back to OLS with cluster-robust errors
        import statsmodels.api as sm
        X = sm.add_constant(df.x.values)
        m = sm.OLS(df.y.values, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.g)[0]})
        ci = m.conf_int()[1]
        return dict(slope=float(m.params[1]), lo=float(ci[0]), hi=float(ci[1]), p=float(m.pvalues[1]), n=len(df), note=str(e))


def lorenz(harm) -> tuple:
    """Cumulative share of positive harm carried by the top-x fraction of samples, and the Gini index."""
    h = np.clip(np.asarray(harm, float), 0, None)
    h = np.sort(h)[::-1]
    tot = h.sum()
    if tot <= 0:
        return np.linspace(0, 1, len(h) + 1), np.linspace(0, 1, len(h) + 1), 0.0
    cum = np.concatenate([[0], np.cumsum(h) / tot])
    frac = np.linspace(0, 1, len(h) + 1)
    asc = np.sort(h)
    n = len(asc)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(asc) / (n * asc.sum()))
    return frac, cum, gini


def fmt_ci(r: dict, pct: bool = False, digits: int = 1) -> str:
    if r is None or not np.isfinite(r.get("mean", np.nan)):
        return "--"
    f = 100.0 if pct else 1.0
    return f"{f * r['mean']:.{digits}f} [{f * r['lo']:.{digits}f}, {f * r['hi']:.{digits}f}]"


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    if p < 1e-4:
        return "\\ensuremath{<10^{-4}}"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"
