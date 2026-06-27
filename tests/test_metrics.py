from __future__ import annotations

import numpy as np
import pandas as pd

from portopt.evaluation.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="B"))


def test_total_return_compounds_correctly():
    r = _series([0.10, -0.05, 0.10])  # 1.10 * 0.95 * 1.10 = 1.1495
    assert np.isclose(total_return(r), 0.1495)


def test_equity_curve_starts_above_one():
    r = _series([0.01, 0.01, 0.01])
    eq = equity_curve(r)
    assert np.isclose(eq.iloc[0], 1.01)
    assert np.isclose(eq.iloc[-1], 1.01**3)


def test_max_drawdown_negative_for_decline():
    # Up 50%, down 50% → end at 75% of peak; drawdown = -25%
    r = _series([0.50, -1 / 3])
    assert np.isclose(max_drawdown(r), -1 / 3, atol=1e-9)


def test_sharpe_zero_volatility_is_zero():
    r = _series([0.0, 0.0, 0.0, 0.0])
    assert sharpe_ratio(r) == 0.0


def test_sharpe_positive_when_returns_positive():
    np.random.seed(0)
    r = pd.Series(
        np.random.normal(0.001, 0.005, 1000),
        index=pd.date_range("2020-01-01", periods=1000, freq="B"),
    )
    sr = sharpe_ratio(r)
    assert sr > 0


def test_annualized_volatility_scales():
    np.random.seed(0)
    daily = pd.Series(
        np.random.normal(0, 0.01, 252),
        index=pd.date_range("2020-01-01", periods=252, freq="B"),
    )
    av = annualized_volatility(daily)
    # std of daily ≈ 0.01 → annualized ≈ 0.01 * sqrt(252) ≈ 0.158
    assert 0.13 < av < 0.18


def test_sortino_uses_downside_only():
    # All-positive returns: sortino should be 0 (no downside)
    r = _series([0.01, 0.02, 0.005])
    assert sortino_ratio(r) == 0.0


def test_calmar_zero_drawdown_is_zero():
    r = _series([0.01, 0.01, 0.01])  # monotonically up
    assert calmar_ratio(r) == 0.0


def test_annualized_return_consistent_with_total():
    r = _series([0.001] * 252)  # 252 days of 0.1% → ~1 year
    ann = annualized_return(r)
    tot = total_return(r)
    # for 1 year of data, annualized = total
    assert np.isclose(ann, tot, rtol=1e-6)
