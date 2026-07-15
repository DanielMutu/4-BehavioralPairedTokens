"""Stats helpers: deterministic bootstrap, exact McNemar sanity values."""

from __future__ import annotations

import pytest

from src.stats import bootstrap_ci, bootstrap_ci_paired_diff, mcnemar_exact


class TestBootstrap:
    def test_deterministic_with_seed(self):
        vals = [0, 1, 1, 0, 1, 1, 1, 0, 1, 0] * 10
        a = bootstrap_ci(vals, n_resamples=500)
        b = bootstrap_ci(vals, n_resamples=500)
        assert a == b
        assert a["ci_low"] <= a["mean"] <= a["ci_high"]

    def test_degenerate_inputs(self):
        assert bootstrap_ci([])["n"] == 0
        c = bootstrap_ci([1.0] * 20, n_resamples=200)
        assert c["ci_low"] == c["ci_high"] == c["mean"] == 1.0

    def test_paired_diff(self):
        a = [1.0, 1.0, 0.0, 1.0] * 25
        b = [0.0, 1.0, 0.0, 0.0] * 25
        d = bootstrap_ci_paired_diff(a, b, n_resamples=500)
        assert d["mean_diff"] == pytest.approx(0.5)
        assert d["ci_low"] > 0  # clearly positive difference

    def test_paired_length_mismatch(self):
        with pytest.raises(ValueError):
            bootstrap_ci_paired_diff([1.0], [1.0, 0.0])


class TestMcNemar:
    def test_no_discordant_pairs(self):
        r = mcnemar_exact([True, False], [True, False])
        assert r["p_value"] == 1.0 and r["n_discordant"] == 0

    def test_balanced_discordants_not_significant(self):
        a = [True, False] * 10
        b = [False, True] * 10
        r = mcnemar_exact(a, b)
        assert r["n10"] == r["n01"] == 10
        assert r["p_value"] > 0.9

    def test_lopsided_discordants_significant(self):
        # a right / b wrong on 15 items, reverse on 1 -> p well below 0.05
        a = [True] * 15 + [False] * 1 + [True, False] * 5
        b = [False] * 15 + [True] * 1 + [True, False] * 5
        r = mcnemar_exact(a, b)
        assert r["n10"] == 15 and r["n01"] == 1
        assert r["p_value"] < 0.01

    def test_exact_value_small_case(self):
        # n10=5, n01=0 -> p = 2 * (1/2^5) = 0.0625
        a = [True] * 5
        b = [False] * 5
        r = mcnemar_exact(a, b)
        assert r["p_value"] == pytest.approx(0.0625)
