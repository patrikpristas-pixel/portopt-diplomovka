from __future__ import annotations

import numpy as np
import pandas as pd

from portopt.strategies.base import Strategy


class EqualWeight(Strategy):
    """1/N benchmark: assigns equal weight to each asset."""

    name = "equal_weight"

    def get_weights(self, asof_date: pd.Timestamp, history: pd.DataFrame) -> np.ndarray:
        n = history.shape[1]
        if n == 0:
            return np.array([])
        return np.ones(n) / n
