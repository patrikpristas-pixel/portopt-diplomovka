"""CRITICAL test: enforce that strategies cannot peek at the future.

The backtest engine must call `strategy.get_weights(asof_date, history)` such that
`history.index[-1] < asof_date`. If this contract is broken, look-ahead corrupts
all results — Sharpe ratios become unrealistically high.

We use a "guard" strategy that asserts the contract on every call. If the engine
ever passes a history including or beyond `asof_date`, the assertion fires and
the test fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portopt.backtest.engine import BacktestConfig, run_backtest
from portopt.evaluation.metrics import sharpe_ratio
from portopt.strategies.base import Strategy


class GuardStrategy(Strategy):
    name = "guard"

    def __init__(self) -> None:
        self.calls: list[tuple[pd.Timestamp, pd.Timestamp | None]] = []

    def get_weights(self, asof_date: pd.Timestamp, history: pd.DataFrame) -> np.ndarray:
        last_seen = history.index[-1] if len(history) > 0 else None
        self.calls.append((asof_date, last_seen))
        if last_seen is not None and last_seen >= asof_date:
            raise AssertionError(
                f"LOOK-AHEAD: strategy at {asof_date} saw history ending at {last_seen}"
            )
        n = history.shape[1]
        return np.ones(n) / n if n > 0 else np.array([])


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    np.random.seed(42)
    n_days, n_assets = 800, 4
    dates = pd.date_range("2020-01-06", periods=n_days, freq="B")
    data = np.random.normal(0, 0.01, size=(n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_strategy_only_sees_past(synthetic_returns: pd.DataFrame) -> None:
    """Engine must never pass data on or after asof_date to the strategy."""
    guard = GuardStrategy()
    run_backtest(guard, synthetic_returns, BacktestConfig(cost_bps=0.0))
    # If the assertion in GuardStrategy fired, run_backtest would have raised.
    assert len(guard.calls) > 0, "guard was never called — no rebalance dates?"
    for asof, last in guard.calls:
        if last is not None:
            assert last < asof, f"violation: {last} >= {asof}"


def test_random_data_yields_low_sharpe(synthetic_returns: pd.DataFrame) -> None:
    """Sanity check: equal-weight on i.i.d. zero-mean noise should give Sharpe ≈ 0.

    A look-ahead bug would let strategy 'see' high-return days and produce abs(Sharpe) > 5.
    """
    from portopt.strategies.equal_weight import EqualWeight

    result = run_backtest(EqualWeight(), synthetic_returns, BacktestConfig(cost_bps=0.0))
    sr = sharpe_ratio(result.portfolio_returns)
    assert abs(sr) < 1.0, f"unrealistic Sharpe={sr:.2f} on random data — possible look-ahead"


def test_guard_strategy_catches_known_violation():
    """Meta-test: confirm GuardStrategy actually fires when handed bad data."""
    guard = GuardStrategy()
    bad_history = pd.DataFrame(
        [[0.01, 0.02]],
        index=[pd.Timestamp("2020-02-01")],
        columns=["A", "B"],
    )
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        guard.get_weights(pd.Timestamp("2020-01-01"), bad_history)
