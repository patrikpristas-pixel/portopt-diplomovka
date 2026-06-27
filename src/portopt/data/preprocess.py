from __future__ import annotations

import numpy as np
import pandas as pd


def forward_fill_limited(prices: pd.DataFrame, max_days: int) -> pd.DataFrame:
    """Forward-fill up to `max_days` consecutive NaN; longer gaps remain NaN."""
    return prices.ffill(limit=max_days)


def align_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Trim to the first date where all tickers have at least one observation.

    This avoids the strategy seeing leading NaN columns. If individual tickers
    have isolated gaps later, those are handled by forward-fill / NaN policy.
    """
    first_valid = prices.apply(lambda s: s.first_valid_index())
    empty = first_valid[first_valid.isna()].index.tolist()
    if empty:
        raise ValueError(f"tickers with no data at all: {empty}")
    common_start = first_valid.max()
    return prices.loc[common_start:]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: log(P_t / P_{t-1}). First row is NaN by construction."""
    return np.log(prices / prices.shift(1))


def coverage_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker first/last valid date and count of observations."""
    rows = []
    for col in prices.columns:
        s = prices[col]
        rows.append(
            {
                "ticker": col,
                "first_date": s.first_valid_index(),
                "last_date": s.last_valid_index(),
                "n_obs": int(s.notna().sum()),
                "n_missing": int(s.isna().sum()),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")
