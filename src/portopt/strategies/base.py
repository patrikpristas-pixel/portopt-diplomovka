from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Abstract base for portfolio strategies.

    Contract:
      - At rebalance date `asof_date`, strategy receives `history` = log returns
        DataFrame strictly preceding asof_date (last index < asof_date).
      - Returns weights as np.ndarray of shape (n_assets,), summing to 1, long-only.
      - Column order in `history` matches the order of returned weights.
    """

    name: str = "strategy"

    @abstractmethod
    def get_weights(self, asof_date: pd.Timestamp, history: pd.DataFrame) -> np.ndarray: ...
