"""Statistical helpers shared by Exp 0 v2 and Exp 2 (pre-registered protocol).

- bootstrap_ci: percentile CI over resampled means of binary/score vectors
  (10k resamples, seed 42 — fixed by the 2026-07-15 pre-registration).
- mcnemar_exact: paired comparison of two binary outcome vectors on the SAME
  items; exact two-sided binomial test on the discordant pairs (no normal
  approximation, no scipy dependency).
"""

from __future__ import annotations

import math
import random

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values: list[float], n_resamples: int = N_BOOTSTRAP,
                 seed: int = BOOTSTRAP_SEED,
                 alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for the mean of `values`."""
    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)])
                   for _ in range(n_resamples))
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return {"mean": mean(values), "ci_low": lo, "ci_high": hi, "n": n}


def bootstrap_ci_paired_diff(a: list[float], b: list[float],
                             n_resamples: int = N_BOOTSTRAP,
                             seed: int = BOOTSTRAP_SEED,
                             alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for mean(a_i - b_i) on paired items."""
    if len(a) != len(b):
        raise ValueError(f"paired vectors differ in length: {len(a)} vs {len(b)}")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    out = bootstrap_ci(diffs, n_resamples, seed, alpha)
    out["mean_diff"] = out.pop("mean")
    return out


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired binary outcomes.

    a/b are per-item correctness on the SAME items. Discordant pairs:
    n01 = a wrong & b right, n10 = a right & b wrong. Under H0 the
    discordants are Binomial(n01+n10, 0.5).
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("paired vectors differ in length")
    n10 = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if x and not y)
    n01 = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if y and not x)
    n = n10 + n01
    if n == 0:
        return {"n10": 0, "n01": 0, "n_discordant": 0, "p_value": 1.0}
    k = min(n10, n01)
    # two-sided exact binomial: 2 * P(X <= k), capped at 1
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return {"n10": n10, "n01": n01, "n_discordant": n, "p_value": p}
