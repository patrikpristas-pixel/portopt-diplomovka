"""Window-based dataset for portfolio policy training.

Each training sample at time t consists of:
  - features: flat vector describing market state at t-1
      * per-asset: last 60 daily log returns + 5 engineered
        (vol_1m, vol_3m, mom_3m, mom_12m_1m, rank_1m)
  - future_simple_returns: shape (horizon, N) — daily simple returns
    of the N tradable assets over the next `horizon` days. Used by the
    portfolio policy loss to simulate buy-and-hold performance.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

ENGINEERED_PER_ASSET = ("vol_1m", "vol_3m", "mom_3m", "mom_12m_1m", "rank_1m")
MIN_HISTORY_FOR_FEATURES = 252


def feature_dim(n_assets: int, lookback: int = 60) -> int:
    return n_assets * (lookback + len(ENGINEERED_PER_ASSET))


def _engineered_per_asset(panel_vals: np.ndarray, asset_idx: int, t: int) -> np.ndarray:
    """Compute 5 engineered features at position t for one asset (cross-sectional)."""
    col = panel_vals[:, asset_idx]
    vol_1m = float(np.nanstd(col[t - 21 : t]))
    vol_3m = float(np.nanstd(col[t - 63 : t]))
    mom_3m = float(np.nansum(col[t - 63 : t]))
    mom_12m_1m = float(np.nansum(col[t - 252 : t - 21]))
    last_1m_universe = np.nansum(panel_vals[t - 21 : t], axis=0)
    asset_1m = last_1m_universe[asset_idx]
    rank_1m = float((last_1m_universe < asset_1m).mean())
    return np.array([vol_1m, vol_3m, mom_3m, mom_12m_1m, rank_1m], dtype=np.float32)


def build_features_at(
    panel: pd.DataFrame,
    t: int,
    lookback: int = 60,
) -> np.ndarray:
    """Build flat feature vector at integer position t.

    Layout (per asset, then per asset):
      [a0_ret_{t-60}..a0_ret_{t-1}, a0_eng_5,
       a1_ret_{t-60}..a1_ret_{t-1}, a1_eng_5, ...]
    """
    n_assets = panel.shape[1]
    vals = panel.values
    parts = []
    for a in range(n_assets):
        past = vals[t - lookback : t, a]
        if np.isnan(past).any():
            return np.array([])  # caller must skip
        eng = _engineered_per_asset(vals, a, t)
        if np.isnan(eng).any():
            return np.array([])
        parts.append(past.astype(np.float32))
        parts.append(eng)
    return np.concatenate(parts)


@dataclass
class WindowDataset:
    """Samples for training the portfolio policy network."""
    X: np.ndarray              # (n_samples, feature_dim)
    future_simple: np.ndarray  # (n_samples, horizon, n_assets) — simple daily returns
    dates: list[pd.Timestamp]  # asof date of each sample
    asset_universe: list[str]

    def __len__(self) -> int:
        return len(self.X)

    def slice_by_date(self, start: pd.Timestamp, end: pd.Timestamp) -> "WindowDataset":
        mask = np.array([(start <= d) & (d < end) for d in self.dates])
        return WindowDataset(
            X=self.X[mask],
            future_simple=self.future_simple[mask],
            dates=[d for d, m in zip(self.dates, mask) if m],
            asset_universe=self.asset_universe,
        )


def build_window_dataset(
    log_returns: pd.DataFrame,
    asset_universe: list[str],
    lookback: int = 60,
    horizon: int = 252,
    stride: int = 5,
) -> WindowDataset:
    """Sliding-window dataset for portfolio policy.

    log_returns: DataFrame of daily log returns, columns ⊇ asset_universe.
    horizon: future-simulation length in trading days (default 252 = ~1 year)
    stride: step between consecutive samples (saves memory; default 5 = weekly)
    Returns samples where:
      X[i]: features at position t (uses data up to t-1)
      future_simple[i]: simple returns from t to t+horizon-1
    """
    cols = [a for a in asset_universe if a in log_returns.columns]
    if len(cols) < 2:
        raise ValueError(f"asset_universe has <2 valid tickers: {cols}")
    panel = log_returns[cols]
    panel = panel.dropna(how="all")
    n = len(panel)
    min_t = max(lookback, MIN_HISTORY_FOR_FEATURES)
    if n < min_t + horizon + 1:
        raise ValueError(
            f"too few rows: have {n}, need ≥ {min_t + horizon + 1} "
            f"(lookback={lookback}, horizon={horizon})"
        )
    log_vals = panel.values

    X_list, fut_list, date_list = [], [], []
    for t in range(min_t, n - horizon, stride):
        feats = build_features_at(panel, t, lookback=lookback)
        if feats.size == 0:
            continue
        fut_log = log_vals[t : t + horizon]
        if np.isnan(fut_log).any():
            continue
        fut_simple = np.expm1(fut_log).astype(np.float32)
        X_list.append(feats)
        fut_list.append(fut_simple)
        date_list.append(panel.index[t])

    if not X_list:
        raise ValueError(
            f"no valid samples (universe={cols}, horizon={horizon}, "
            f"period {panel.index[0].date()}→{panel.index[-1].date()})"
        )

    return WindowDataset(
        X=np.stack(X_list),
        future_simple=np.stack(fut_list),
        dates=date_list,
        asset_universe=cols,
    )


def build_features_for_inference(
    log_returns: pd.DataFrame,
    asset_universe: list[str],
    asof: pd.Timestamp,
    lookback: int = 60,
) -> np.ndarray:
    """Build a single feature vector for inference at asof date (uses data < asof)."""
    cols = [a for a in asset_universe if a in log_returns.columns]
    panel = log_returns[cols].loc[:asof]
    panel = panel.iloc[:-1] if panel.index[-1] == asof else panel
    n = len(panel)
    if n < max(lookback, MIN_HISTORY_FOR_FEATURES):
        return np.array([])
    return build_features_at(panel, n, lookback=lookback)


# Back-compat shims (legacy callers; should be replaced)
def feature_dim_legacy(lookback: int = 60) -> int:
    """Deprecated: only here so unrelated imports don't crash. Use feature_dim()."""
    return lookback + len(ENGINEERED_PER_ASSET)
