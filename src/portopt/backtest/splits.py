from __future__ import annotations

import pandas as pd


def monthly_rebalance_dates(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """Last available trading day of each calendar month from the given index."""
    s = pd.Series(index=index, data=index)
    last_per_month = s.resample("ME").last().dropna()
    return set(pd.DatetimeIndex(last_per_month.values))
