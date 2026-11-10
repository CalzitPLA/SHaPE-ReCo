#!/usr/bin/env python3
"""
The statistics of the reconstruction protocol (docs/thesis/chapters/12_datasets_and_protocol.tex, "The reconstruction
protocol"; scripts/sci_test.py): paired comparisons of two arms on the same parts.

    p = mcnemar_exact(breaks, rescues, "greater")      # B better than A: more rescues than breaks
    d = paired_bootstrap_delta(good_a, good_b)         # the change of the good rate, 95 % interval
    w = wilcoxon_paired(iou_a, iou_b)                  # the paired IoU (or seconds) shift
    q = holm({"cell": p, ...})                         # the per-class p values, family-wise corrected
    lo, hi = wilson(k, n)
    n = mcnemar_sample_size(p_disc, delta)             # pairs needed (Connor 1987)
    f = flip_rate(good_a, good_a2)                     # the noise floor: discordance of a replicate (A against A')

Conventions: a pair is (A, B) on one part; `breaks` = A good and B not, `rescues` = A not good and B good. Every
function takes plain sequences and returns plain numbers or dicts (JSON-able); none of them looks at a part twice.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95):
    """The Wilson score interval of k successes in n trials ((0, 1) for n = 0)."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def discordant(good_a: Sequence[bool], good_b: Sequence[bool]) -> Dict[str, int]:
    """The 2x2 table of paired outcomes: both, breaks (A only), rescues (B only), neither."""
    a, b = np.asarray(good_a, bool), np.asarray(good_b, bool)
    if a.shape != b.shape:
        raise ValueError("the two arms must be paired (equal length)")
    return dict(both=int(np.sum(a & b)), breaks=int(np.sum(a & ~b)), rescues=int(np.sum(~a & b)),
                neither=int(np.sum(~a & ~b)), n=int(a.size))


def mcnemar_exact(breaks: int, rescues: int, alternative: str = "two-sided") -> float:
    """The exact McNemar test: the rescues among the discordant pairs against Binomial(breaks + rescues, 1/2).
    alternative "greater": B better than A (more rescues); "less": B worse; "two-sided". 1.0 without discordant pairs."""
    from scipy.stats import binomtest
    n = int(breaks) + int(rescues)
    if n == 0:
        return 1.0
    return float(binomtest(int(rescues), n, 0.5, alternative=alternative).pvalue)


def paired_bootstrap_delta(a: Sequence[float], b: Sequence[float], stat: Callable = np.mean, n: int = 10_000,
                           seed: int = 0, level: float = 0.95) -> Dict[str, float]:
    """stat(b) - stat(a) over the pairs and its percentile interval from `n` resamples of the pairs (the paired bootstrap
    of scripts/validate_known.py, generalised to any statistic)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("the two arms must be paired and non-empty")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n, a.size))
    deltas = np.array([stat(b[i]) - stat(a[i]) for i in idx])
    lo, hi = np.quantile(deltas, [(1 - level) / 2, 1 - (1 - level) / 2])
    return dict(delta=float(stat(b) - stat(a)), lo=float(lo), hi=float(hi), level=level, resamples=n)


def wilcoxon_paired(a: Sequence[float], b: Sequence[float], alternative: str = "two-sided") -> Dict[str, float]:
    """The Wilcoxon signed-rank test on b - a (pairs with a missing value dropped; zero differences dropped, "wilcox").
    dict(n, median_diff, statistic, p); p = 1.0 when every difference is zero."""
    from scipy.stats import wilcoxon
    x, y = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = y[ok] - x[ok]
    out = dict(n=int(ok.sum()), median_diff=float(np.median(d)) if d.size else float("nan"))
    if not np.any(d != 0):
        return dict(out, statistic=0.0, p=1.0)
    r = wilcoxon(d, zero_method="wilcox", alternative=alternative)
    return dict(out, statistic=float(r.statistic), p=float(r.pvalue))


def holm(pvalues: Mapping[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p values (step-down, monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def mcnemar_sample_size(p_disc: float, delta: float, alpha: float = 0.05, power: float = 0.8,
                        two_sided: bool = True) -> int:
    """Pairs needed to detect a change `delta` of the good rate (rescue share - break share) when a share `p_disc` of
    the pairs is discordant (Connor 1987): n = (z_a sqrt(p_disc) + z_b sqrt(p_disc - delta^2))^2 / delta^2."""
    from scipy.stats import norm
    if not 0 < abs(delta) <= p_disc <= 1:
        raise ValueError("need 0 < |delta| <= p_disc <= 1")
    za = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
    zb = norm.ppf(power)
    return int(math.ceil((za * math.sqrt(p_disc) + zb * math.sqrt(p_disc - delta * delta)) ** 2 / (delta * delta)))


def flip_rate(good_a: Sequence[bool], good_a2: Sequence[bool]) -> Dict[str, float]:
    """The noise floor: how often a part's good/not-good outcome flips between two runs of the same arm (identical code
    and settings). dict(n, flips, rate, lo, hi) with the Wilson interval of the rate."""
    t = discordant(good_a, good_a2)
    flips = t["breaks"] + t["rescues"]
    lo, hi = wilson(flips, t["n"])
    return dict(n=t["n"], flips=flips, rate=flips / t["n"] if t["n"] else 0.0, lo=lo, hi=hi)


def net_change_floor(rate: float, n: int, z: float = Z95) -> float:
    """The |rescues - breaks| that replicate noise alone reaches with probability 1 - level: with a share `rate` of
    the pairs flipping at random, the net change has mean 0 and standard deviation sqrt(rate n)."""
    return z * math.sqrt(max(rate, 0.0) * n)


def noninferior(delta_lo: float, margin: float) -> bool:
    """Non-inferiority: the lower end of the interval of B - A lies above -margin."""
    return delta_lo > -abs(margin)


def summary_good(good: Sequence[bool]) -> Dict[str, float]:
    k, n = int(np.sum(np.asarray(good, bool))), len(good)
    lo, hi = wilson(k, n)
    return dict(k=k, n=n, rate=k / n if n else 0.0, lo=lo, hi=hi)


def median_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None
