"""NN portfolio policy strategy.

Loads a trained PortfolioPolicyMLP from a checkpoint, then at each rebalance
date computes features from the visible history and returns the network's
softmax output as portfolio weights.

In walk-forward backtesting the strategy is loaded ONCE for a given test
window; weights are computed at the test_start date and then locked
(buy-and-hold) for the duration of the test window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from portopt.models.dataset import (
    MIN_HISTORY_FOR_FEATURES,
    build_features_at,
)
from portopt.models.return_predictor import PortfolioPolicyMLP
from portopt.strategies.base import Strategy


class NNSoftmaxStrategy(Strategy):
    """Portfolio policy: features → softmax weights, clipped to max_weight."""

    def __init__(
        self,
        checkpoint_path: Path,
        asset_universe: list[str],
        lookback: int = 60,
        max_weight: float = 0.25,
    ) -> None:
        self.lookback = lookback
        self.max_weight = float(max_weight)
        self.name = "nn_softmax"

        ckpt = torch.load(checkpoint_path, weights_only=False)
        ckpt_universe = list(ckpt.get("asset_universe", []))
        if ckpt_universe and ckpt_universe != list(asset_universe):
            logger.warning(
                f"NN checkpoint universe {ckpt_universe} ≠ requested {asset_universe}; "
                f"using requested order (network must have been trained on this order)"
            )
        self.asset_universe = list(asset_universe)

        self.model = PortfolioPolicyMLP(
            n_assets=int(ckpt["n_assets"]),
            input_dim=int(ckpt["input_dim"]),
            hidden=int(ckpt["hidden"]),
            n_layers=int(ckpt.get("n_layers", 2)),
            dropout=float(ckpt.get("dropout", 0.2)),
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
        self.checkpoint_meta = {
            "best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
            "val_sharpe": float(ckpt.get("val_sharpe", float("nan"))),
            "val_total_return": float(ckpt.get("val_total_return", float("nan"))),
            "criterion": str(ckpt.get("criterion", "sharpe")),
        }

    def _apply_max_weight_clip(self, w: np.ndarray) -> np.ndarray:
        """Iteratively cap weights at max_weight and redistribute the excess."""
        w = w.astype(np.float64).copy()
        n = len(w)
        if self.max_weight * n < 1.0 - 1e-9:
            # Can't sum to 1 with cap this tight → just clip and accept
            w = np.minimum(w, self.max_weight)
            s = w.sum()
            return (w / s) if s > 1e-9 else np.ones(n) / n
        for _ in range(50):
            over = w > self.max_weight + 1e-9
            if not over.any():
                break
            excess = (w[over] - self.max_weight).sum()
            w[over] = self.max_weight
            under = ~over & (w < self.max_weight - 1e-9)
            if under.sum() == 0:
                break
            w[under] += excess * (w[under] / w[under].sum())
        s = w.sum()
        return w / s if s > 1e-9 else np.ones(n) / n

    def get_weights(self, asof_date: pd.Timestamp, history: pd.DataFrame) -> np.ndarray:
        cols = list(self.asset_universe)
        n = len(cols)
        if not all(c in history.columns for c in cols):
            missing = [c for c in cols if c not in history.columns]
            raise ValueError(f"history missing required columns: {missing}")
        panel = history[cols]
        t = len(panel)
        if t < max(self.lookback, MIN_HISTORY_FOR_FEATURES):
            return np.ones(n) / n
        feats = build_features_at(panel, t, lookback=self.lookback)
        if feats.size == 0:
            return np.ones(n) / n
        feats_scaled = (feats - self.scaler_mean) / self.scaler_scale
        with torch.no_grad():
            x = torch.from_numpy(feats_scaled.astype(np.float32)).unsqueeze(0)
            w = self.model(x).squeeze(0).numpy()
        w = self._apply_max_weight_clip(w)
        return w
