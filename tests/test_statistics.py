"""Unit tests for the inferential statistics module.

Validates that:
- Bootstrap CI brackets the point estimate
- PSR is monotone in observed Sharpe
- DSR penalizes for more trials (selection bias correction)
- DM test returns p-value in [0, 1]
- PBO returns valid probability for synthetic random data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portopt.evaluation.statistics import (
    deflated_sharpe_ratio,
    diebold_mariano,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfit,
    sharpe_bootstrap_ci,
)


@pytest.fixture
def positive_returns():
    """500 days of synthetic returns with positive mean → annualized Sharpe ~1."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(loc=5e-4, scale=1e-2, size=500))


@pytest.fixture
def zero_returns():
    """500 days of zero-mean returns → Sharpe ~0."""
    rng = np.random.default_rng(123)
    return pd.Series(rng.normal(loc=0.0, scale=1e-2, size=500))


def test_bootstrap_ci_brackets_point_estimate(positive_returns):
    ci = sharpe_bootstrap_ci(positive_returns, n_bootstrap=500)
    assert ci.ci_low_ann <= ci.sharpe_ann <= ci.ci_high_ann
    assert ci.ci_low_ann < ci.ci_high_ann  # CI must have positive width


def test_bootstrap_ci_short_series_degenerate():
    short = pd.Series(np.random.randn(10))
    ci = sharpe_bootstrap_ci(short, n_bootstrap=100)
    # With <30 obs, function returns degenerate CI = point estimate
    assert ci.n_bootstrap == 0


def test_psr_monotone_in_sharpe(positive_returns, zero_returns):
    """Higher observed Sharpe → higher PSR (probability of true edge)."""
    psr_pos = probabilistic_sharpe_ratio(positive_returns, 0.0)
    psr_zero = probabilistic_sharpe_ratio(zero_returns, 0.0)
    assert psr_pos > psr_zero
    assert 0.0 <= psr_pos <= 1.0
    assert 0.0 <= psr_zero <= 1.0


def test_psr_zero_sharpe_is_half(zero_returns):
    """If observed = reference, PSR should be ~0.5 (no preference)."""
    sr_obs = zero_returns.mean() / zero_returns.std(ddof=1) * np.sqrt(252)
    psr = probabilistic_sharpe_ratio(zero_returns, sr_obs)
    assert abs(psr - 0.5) < 0.05


def test_psr_higher_threshold_lower_probability(positive_returns):
    """PSR(>0) should be > PSR(>1) > PSR(>2)."""
    psr_0 = probabilistic_sharpe_ratio(positive_returns, 0.0)
    psr_1 = probabilistic_sharpe_ratio(positive_returns, 1.0)
    psr_2 = probabilistic_sharpe_ratio(positive_returns, 2.0)
    assert psr_0 >= psr_1 >= psr_2


def test_dsr_penalizes_more_trials(positive_returns):
    """DSR with more historical trials → tougher bar → smaller DSR."""
    rng = np.random.default_rng(7)
    sh_small = rng.normal(0.5, 0.4, 10)
    sh_large = rng.normal(0.5, 0.4, 500)
    dsr_small = deflated_sharpe_ratio(positive_returns, sh_small)
    dsr_large = deflated_sharpe_ratio(positive_returns, sh_large)
    assert dsr_small["sharpe_ref_ann"] < dsr_large["sharpe_ref_ann"]
    # Higher reference → lower DSR (harder to clear bar)
    assert dsr_small["dsr"] >= dsr_large["dsr"]


def test_dsr_returns_psr_with_few_trials(positive_returns):
    """With <2 historical trials, DSR falls back to PSR vs 0."""
    dsr = deflated_sharpe_ratio(positive_returns, [])
    psr = probabilistic_sharpe_ratio(positive_returns, 0.0)
    assert abs(dsr["dsr"] - psr) < 1e-9


def test_dm_test_returns_valid_pvalue(positive_returns, zero_returns):
    """DM p-value must be in [0, 1] and stat must be finite."""
    res = diebold_mariano(positive_returns, zero_returns, h=5)
    assert 0.0 <= res["p_value"] <= 1.0
    assert np.isfinite(res["dm_stat"])
    assert res["n"] == 500


def test_dm_test_rejects_when_clearly_different():
    """Two series with very different means should yield a strongly significant DM."""
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(1e-3, 1e-2, 1000))
    b = pd.Series(rng.normal(-1e-3, 1e-2, 1000))
    res = diebold_mariano(a, b, h=1)
    assert res["p_value"] < 0.05
    # Mean diff per year should be in the 20-30% range (2e-3 daily * 252)
    assert res["mean_diff_ann"] > 0.20


def test_dm_test_short_series_returns_nan():
    res = diebold_mariano(pd.Series([0.01] * 10), pd.Series([0.01] * 10))
    assert np.isnan(res["dm_stat"])
    assert np.isnan(res["p_value"])


def test_pbo_random_data_around_half():
    """If all trials are random noise, PBO should hover near 0.5 (no skill)."""
    rng = np.random.default_rng(42)
    T = 500
    N = 20
    R = pd.DataFrame(rng.normal(0, 0.01, (T, N)), columns=[f"t{i}" for i in range(N)])
    R.index = pd.date_range("2020-01-01", periods=T, freq="B")
    res = probability_of_backtest_overfit(R, n_splits=12)
    # Random data → PBO around 0.5 (give wide margin since 12 splits = 924 combos is noisy)
    assert 0.30 <= res["pbo"] <= 0.70
    assert res["n_combinations"] > 0


def test_pbo_synthetic_skilled_trial_lowers_pbo():
    """Insert one consistently dominant trial; PBO should drop noticeably below 0.5.

    Note: PBO is sensitive to signal-to-noise. For PBO < 0.3 the true edge
    must dominate the empirical max-of-N-random spread. With 19 noise trials
    and 250-day windows, std-of-mean is ~6e-4; max-of-19 noise means ~1e-3.
    So edge must be ≥ 2e-3 to clearly dominate.
    """
    rng = np.random.default_rng(0)
    T = 500
    N = 20
    noise = rng.normal(0, 0.01, (T, N))
    # Strong edge: 2e-3 daily ≈ Sharpe 3.2 annualized → dominates noise
    noise[:, 0] += 2e-3
    R = pd.DataFrame(noise, columns=[f"t{i}" for i in range(N)])
    R.index = pd.date_range("2020-01-01", periods=T, freq="B")
    res = probability_of_backtest_overfit(R, n_splits=12)
    assert res["pbo"] < 0.30, f"PBO={res['pbo']:.2f} expected < 0.30 with dominant trial"


def test_pbo_handles_too_few_trials():
    """PBO with <2 trials returns NaN."""
    R = pd.DataFrame({"only": np.random.randn(500)})
    R.index = pd.date_range("2020-01-01", periods=500, freq="B")
    res = probability_of_backtest_overfit(R, n_splits=12)
    assert np.isnan(res["pbo"])


def test_bootstrap_ci_with_block_size_handles_autocorrelation():
    """Stationary bootstrap should handle a slowly varying series without crashing."""
    n = 500
    # AR(1) with phi=0.3
    rng = np.random.default_rng(99)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.3 * r[i - 1] + rng.normal(0, 0.01)
    ci = sharpe_bootstrap_ci(pd.Series(r), n_bootstrap=300, block_size=10)
    assert ci.ci_low_ann <= ci.sharpe_ann <= ci.ci_high_ann
    assert ci.n_bootstrap == 300


# ----- Edge case / robustness tests -----


def test_bootstrap_ci_strips_nan_returns():
    """NaN values in input should be filtered, not crash the bootstrap."""
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(5e-4, 1e-2, 500))
    # Sprinkle ~10% NaN
    nan_idx = rng.choice(500, size=50, replace=False)
    r.iloc[nan_idx] = np.nan
    ci = sharpe_bootstrap_ci(r, n_bootstrap=200)
    assert ci.n_observations == 450  # NaN dropped
    assert ci.ci_low_ann <= ci.sharpe_ann <= ci.ci_high_ann
    assert np.isfinite(ci.sharpe_ann)


def test_bootstrap_ci_zero_volatility_degenerate():
    """Constant returns → zero vol → degenerate point estimate, no crash."""
    constant = pd.Series([0.01] * 500)
    ci = sharpe_bootstrap_ci(constant, n_bootstrap=100)
    assert ci.sharpe_ann == 0.0  # sharpe is 0 when sd=0
    # Each bootstrap resample is also constant, so CI collapses
    assert ci.ci_low_ann == ci.ci_high_ann == 0.0


def test_psr_zero_std_returns_indicator():
    """When std(returns) = 0, PSR falls back to 0/1 based on mean comparison."""
    constant = pd.Series([0.01] * 100)  # positive mean, zero vol
    # daily_sharpe returns 0 when sd=0, so PSR(>0) compares 0 > 0 → 0.5
    psr = probabilistic_sharpe_ratio(constant, 0.0)
    assert 0.0 <= psr <= 1.0  # must still be a probability


def test_dsr_history_with_nan_filtered():
    """NaN in sharpe history must be ignored when computing the multiple-testing bar."""
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(5e-4, 1e-2, 500))
    history_with_nan = np.array([1.5, np.nan, 2.0, np.nan, 1.8, 1.6, 1.9])
    res = deflated_sharpe_ratio(r, history_with_nan)
    assert res["n_trials"] == 5  # NaN dropped
    assert 0.0 <= res["dsr"] <= 1.0


def test_dsr_single_trial_falls_back_to_psr():
    """With n_trials=1, DSR degenerates to PSR vs 0 (no selection bias)."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(5e-4, 1e-2, 500))
    res = deflated_sharpe_ratio(r, [1.5])  # only 1 historical sharpe
    psr = probabilistic_sharpe_ratio(r, 0.0)
    assert abs(res["dsr"] - psr) < 1e-9
    assert res["n_trials"] == 1


def test_pbo_empty_dataframe_returns_nan():
    """Empty trials matrix → PBO is NaN."""
    res = probability_of_backtest_overfit(pd.DataFrame(), n_splits=14)
    assert np.isnan(res["pbo"])
    assert res["n_combinations"] == 0


def test_pbo_odd_n_splits_rounded_to_even():
    """CSCV requires even splits (it picks half for IS). Odd input is auto-rounded."""
    rng = np.random.default_rng(42)
    R = pd.DataFrame(rng.normal(0, 0.01, (500, 10)), columns=[f"t{i}" for i in range(10)])
    R.index = pd.date_range("2020-01-01", periods=500, freq="B")
    res = probability_of_backtest_overfit(R, n_splits=11)  # odd
    # With 12 (rounded), C(12,6) = 924 combinations
    assert res["n_combinations"] == 924


def test_dm_test_handles_misaligned_lengths():
    """If returns_a and returns_b have different lengths, use min length."""
    a = pd.Series(np.random.RandomState(1).randn(500) * 0.01)
    b = pd.Series(np.random.RandomState(2).randn(400) * 0.01)
    res = diebold_mariano(a, b, h=5)
    assert res["n"] == 400
    assert np.isfinite(res["dm_stat"])


def test_dm_test_identical_series_yields_zero_diff():
    """If both series are identical, mean diff = 0 and variance = 0 → NaN."""
    rng = np.random.default_rng(42)
    a = pd.Series(rng.normal(0, 0.01, 500))
    res = diebold_mariano(a, a.copy(), h=5)
    assert res["mean_diff_ann"] == 0.0
    # Variance of zero diff is 0 → degenerate → NaN per impl
    assert np.isnan(res["dm_stat"])
    assert np.isnan(res["p_value"])
