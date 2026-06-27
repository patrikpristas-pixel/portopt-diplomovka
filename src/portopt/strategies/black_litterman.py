from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from pypfopt import EfficientFrontier, black_litterman
from sklearn.covariance import LedoitWolf

from portopt.strategies.base import Strategy


class BlackLitterman(Strategy):
    """Black-Litterman baseline with equal-cap prior and historical-return views."""

    def __init__(
        self,
        lookback: int = 252,
        max_weight: float = 0.25,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.lookback = lookback
        self.max_weight = float(max_weight)
        self.tau = float(tau)
        self.risk_aversion = float(risk_aversion)
        self.risk_free_rate = float(risk_free_rate)
        self.name = "black_litterman"

    def get_weights(self, asof_date: pd.Timestamp, history: pd.DataFrame) -> np.ndarray:
        n = history.shape[1]
        if len(history) < self.lookback:
            return np.ones(n) / n

        recent = history.tail(self.lookback)
        simple = np.expm1(recent).dropna(how="any")
        if len(simple) < self.lookback // 2:
            return np.ones(n) / n

        try:
            lw = LedoitWolf().fit(simple.values)
            cov = pd.DataFrame(
                lw.covariance_ * 252.0,
                index=history.columns,
                columns=history.columns,
            )
            market_caps = pd.Series(1.0, index=history.columns)
            prior = black_litterman.market_implied_prior_returns(
                market_caps,
                self.risk_aversion,
                cov,
            )
            views = (simple.mean() * 252.0).to_dict()
            bl = black_litterman.BlackLittermanModel(
                cov,
                pi=prior,
                absolute_views=views,
                tau=self.tau,
            )
            post_ret = bl.bl_returns()
            post_cov = bl.bl_cov()
            ef = EfficientFrontier(
                post_ret,
                post_cov,
                weight_bounds=(0.0, self.max_weight),
            )
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            cleaned = ef.clean_weights()
        except Exception as e:
            logger.warning(
                f"BlackLitterman failed at {asof_date.date()}: {e} — falling back to equal weight"
            )
            return np.ones(n) / n

        return np.array([cleaned[c] for c in history.columns])
