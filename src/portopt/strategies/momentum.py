from __future__ import annotations

import math

import numpy as np
import pandas as pd

from portopt.strategies.base import Strategy


class MomentumPortfolio(Strategy):
    """Cross-sectional momentum: equal weight the strongest trailing assets."""

    def __init__(self, lookback: int = 126, max_weight: float = 0.25) -> None:
        self.lookback = int(lookback)
        self.max_weight = float(max_weight)
        self.name = "momentum"

    def get_weights(self, asof_date: pd.Timestamp, history: pd.DataFrame) -> np.ndarray:
        n = history.shape[1]
        if n == 0:
            return np.array([])
        if len(history) < self.lookback:
            return np.ones(n) / n

        recent = history.tail(self.lookback)
        simple = np.expm1(recent).dropna(how="any")
        if len(simple) < self.lookback // 2:
            return np.ones(n) / n

        trailing = (1.0 + simple).prod(axis=0) - 1.0
        order = trailing.sort_values(ascending=False).index.tolist()
        min_assets_for_cap = max(1, int(math.ceil(1.0 / max(self.max_weight, 1e-9))))
        k = min(n, max(5, min_assets_for_cap))
        selected = order[:k]

        w = pd.Series(0.0, index=history.columns, dtype=float)
        w.loc[selected] = 1.0 / len(selected)
        return w.to_numpy()
