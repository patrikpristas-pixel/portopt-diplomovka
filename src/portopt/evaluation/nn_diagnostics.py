"""Is the NN actually learning, or is it noise?

Diagnostic A: compares a TRAINED checkpoint's output to K randomly-initialized
networks (same architecture, no training, same input). If trained ≠ random by
more than the spread among random nets themselves, the training did real work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from portopt.models.dataset import MIN_HISTORY_FOR_FEATURES, build_features_at
from portopt.models.return_predictor import PortfolioPolicyMLP


@dataclass
class WindowDiagnostic:
    window_idx: int
    test_start: pd.Timestamp
    trained_weights: dict[str, float]   # asset → raw NN softmax output
    random_mean: dict[str, float]
    random_std_mean: float              # avg of per-asset std across random nets
    dist_to_random_mean: float          # L2 distance trained ↔ random mean
    dist_to_uniform: float              # L2 distance trained ↔ 1/N
    avg_dist_among_random: float        # how spread random nets are between each other
    ratio: float                         # dist_to_random_mean / avg_dist_among_random
    learned: bool                        # ratio > 2 → real learning

    @property
    def verdict(self) -> str:
        if self.ratio > 5:
            return "silno učené"
        if self.ratio > 2:
            return "učené"
        if self.ratio > 1.2:
            return "slabo učené"
        return "≈ random (nenaučené)"


def _features_at(
    log_returns: pd.DataFrame,
    universe: list[str],
    asof: pd.Timestamp,
    lookback: int = 60,
) -> np.ndarray | None:
    panel = log_returns[universe].loc[:asof]
    if len(panel) > 0 and panel.index[-1] == asof:
        panel = panel.iloc[:-1]
    n = len(panel)
    if n < max(lookback, MIN_HISTORY_FOR_FEATURES):
        return None
    return build_features_at(panel, n, lookback=lookback)


def diagnose_checkpoint(
    ckpt_path: Path,
    log_returns: pd.DataFrame,
    test_start: pd.Timestamp,
    window_idx: int,
    n_random: int = 20,
) -> WindowDiagnostic | None:
    ckpt = torch.load(ckpt_path, weights_only=False)
    universe = list(ckpt["asset_universe"])
    n_assets = int(ckpt["n_assets"])
    input_dim = int(ckpt["input_dim"])
    hidden = int(ckpt["hidden"])
    n_layers = int(ckpt.get("n_layers", 2))
    dropout = float(ckpt.get("dropout", 0.2))
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)

    feats = _features_at(log_returns, universe, test_start)
    if feats is None or feats.size == 0:
        return None
    feats_scaled = (feats - scaler_mean) / scaler_scale
    x = torch.from_numpy(feats_scaled.astype(np.float32)).unsqueeze(0)

    trained = PortfolioPolicyMLP(
        n_assets=n_assets, input_dim=input_dim, hidden=hidden,
        n_layers=n_layers, dropout=dropout,
    )
    trained.load_state_dict(ckpt["model_state"])
    trained.eval()
    with torch.no_grad():
        w_trained = trained(x).squeeze(0).numpy()

    random_outputs = []
    for seed in range(n_random):
        torch.manual_seed(seed)
        rnd = PortfolioPolicyMLP(
            n_assets=n_assets, input_dim=input_dim, hidden=hidden,
            n_layers=n_layers, dropout=dropout,
        )
        rnd.eval()
        with torch.no_grad():
            random_outputs.append(rnd(x).squeeze(0).numpy())
    random_arr = np.stack(random_outputs)
    random_mean = random_arr.mean(axis=0)
    random_std_mean = float(random_arr.std(axis=0).mean())

    uniform = np.ones(n_assets) / n_assets
    dist_random = float(np.linalg.norm(w_trained - random_mean))
    dist_uniform = float(np.linalg.norm(w_trained - uniform))
    avg_among_random = float(
        np.mean([np.linalg.norm(r - random_mean) for r in random_arr])
    )
    ratio = dist_random / max(avg_among_random, 1e-9)

    return WindowDiagnostic(
        window_idx=window_idx,
        test_start=test_start,
        trained_weights={a: float(v) for a, v in zip(universe, w_trained)},
        random_mean={a: float(v) for a, v in zip(universe, random_mean)},
        random_std_mean=random_std_mean,
        dist_to_random_mean=dist_random,
        dist_to_uniform=dist_uniform,
        avg_dist_among_random=avg_among_random,
        ratio=ratio,
        learned=ratio > 2.0,
    )


_WINDOW_IDX_RE = re.compile(r"window_(\d+)")


def diagnose_trial(
    trial_dir: Path,
    log_returns: pd.DataFrame,
    test_start: pd.Timestamp,
    test_window_months: int = 12,
    n_random: int = 20,
) -> list[WindowDiagnostic]:
    """Run diagnostic on every saved window checkpoint of a trial.

    Checkpoint filenames are `window_<idx>.pt` (legacy) or
    `window_<idx>_<tag>.pt` (new arch with search/holdout split).
    We extract <idx> via regex to be robust to both formats.
    """
    out: list[WindowDiagnostic] = []
    ckpts = sorted((trial_dir / "models").glob("window_*.pt"))
    for ckpt in ckpts:
        m = _WINDOW_IDX_RE.search(ckpt.stem)
        if m is None:
            continue
        win_idx = int(m.group(1))
        win_test_start = test_start + pd.DateOffset(months=win_idx * test_window_months)
        d = diagnose_checkpoint(ckpt, log_returns, win_test_start, win_idx, n_random=n_random)
        if d is not None:
            out.append(d)
    return out


def summarize(diagnostics: list[WindowDiagnostic]) -> dict:
    if not diagnostics:
        return {"n_windows": 0, "n_learned": 0, "median_ratio": 0.0, "learned_rate": 0.0}
    ratios = [d.ratio for d in diagnostics]
    n_learned = sum(1 for d in diagnostics if d.learned)
    return {
        "n_windows": len(diagnostics),
        "n_learned": n_learned,
        "learned_rate": n_learned / len(diagnostics),
        "median_ratio": float(np.median(ratios)),
        "min_ratio": float(np.min(ratios)),
        "max_ratio": float(np.max(ratios)),
    }
