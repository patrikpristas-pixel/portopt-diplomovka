from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from pypfopt import EfficientFrontier
from sklearn.covariance import LedoitWolf

from portopt.strategies.base import Strategy


class Markowitz(Strategy):
    """Mean-variance optimization using PyPortfolioOpt + Ledoit-Wolf shrinkage on cov.

    Inputs to optimizer derived from log returns history:
      - mu (annualized arithmetic mean of simple returns, 252-day lookback)
      - Sigma (Ledoit-Wolf shrunk covariance of simple returns, annualized)

    Falls back to equal weight when not enough history.
    """

    def __init__(
        self,
        lookback: int = 252,
        max_weight: float = 0.25,
        objective: str = "max_sharpe",
        risk_free_rate: float = 0.0,
    ) -> None:
        if objective not in ("max_sharpe", "min_volatility"):
            raise ValueError(f"unknown objective: {objective}")
        self.lookback = lookback
        self.max_weight = max_weight
        self.objective = objective
        self.risk_free_rate = risk_free_rate
        self.name = f"markowitz_{objective}"

    def get_weights(self, asof_date: pd.Timestamp, history: pd.DataFrame) -> np.ndarray:
        n = history.shape[1]
        if len(history) < self.lookback:
            return np.ones(n) / n  # warmup → equal weight

        recent = history.tail(self.lookback)
        simple = np.expm1(recent).dropna(how="any")
        if len(simple) < self.lookback // 2:
            return np.ones(n) / n

        mu = simple.mean() * 252
        lw = LedoitWolf().fit(simple.values)
        S = pd.DataFrame(lw.covariance_ * 252, index=recent.columns, columns=recent.columns)

        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0.0, self.max_weight))
            if self.objective == "max_sharpe":
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            else:
                ef.min_volatility()
            cleaned = ef.clean_weights()
        except Exception as e:
            logger.warning(f"Markowitz optimization failed at {asof_date.date()}: {e} — falling back to equal weight")
            return np.ones(n) / n

        return np.array([cleaned[c] for c in history.columns])
