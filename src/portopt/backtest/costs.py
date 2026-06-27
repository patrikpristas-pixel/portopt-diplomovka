from __future__ import annotations

import numpy as np


def apply_costs(prev_w: np.ndarray, new_w: np.ndarray, cost_bps: float) -> float:
    """Linear cost on turnover.

    `cost_bps` is the cost rate per unit of NAV traded. Total cost as a fraction
    of NAV = sum(|Δw|) * cost_bps / 10000.

    A swap of 10% from A to B → |Δw| sum = 0.20 → at 10 bps → cost = 2 bps of NAV.
    """
    turnover = float(np.abs(new_w - prev_w).sum())
    return turnover * cost_bps / 10000.0


def turnover(prev_w: np.ndarray, new_w: np.ndarray) -> float:
    return float(np.abs(new_w - prev_w).sum())
