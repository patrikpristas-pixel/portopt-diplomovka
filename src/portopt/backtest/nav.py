"""NAV (€) computation with optional monthly deposits.

Decoupled from the backtest engine — works on any returns Series.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NavResult:
    nav_eur: pd.Series              # NAV in EUR per day
    cum_deposits_eur: pd.Series     # cumulative deposits in EUR per day
    nav_no_deposit_eur: pd.Series   # NAV that would have been if no deposits (for time-weighted return)


def compute_nav(
    portfolio_returns: pd.Series,
    rebalance_dates: Iterable[pd.Timestamp],
    monthly_deposit_eur: float = 0.0,
    initial_nav_eur: float = 0.0,
    deposit_at_start: bool = True,
) -> NavResult:
    """Build NAV series from daily simple returns + monthly deposits.

    On each rebalance day, deposit `monthly_deposit_eur` BEFORE applying that
    day's market return. If `deposit_at_start` and there's no rebalance on day 0,
    we still drop the initial deposit on day 0.

    `nav_no_deposit_eur` tracks what you'd have with the same daily returns
    but no deposits (initial_nav_eur + monthly_deposit_eur if deposit_at_start
    else initial_nav_eur). Useful for separating "growth from market" from
    "growth from contributions".
    """
    rebal_set = set(pd.DatetimeIndex(rebalance_dates))

    nav_with = []
    nav_without = []
    cum_dep = []
    current_nav = float(initial_nav_eur)
    current_no_dep = float(initial_nav_eur)
    cum_deposit = float(initial_nav_eur)

    for i, (date, r) in enumerate(portfolio_returns.items()):
        if i == 0 and deposit_at_start and date not in rebal_set:
            current_nav += monthly_deposit_eur
            cum_deposit += monthly_deposit_eur
            if current_no_dep == 0.0:
                current_no_dep = monthly_deposit_eur  # bootstrap counterfactual
        if date in rebal_set:
            current_nav += monthly_deposit_eur
            cum_deposit += monthly_deposit_eur
            if current_no_dep == 0.0:
                current_no_dep = monthly_deposit_eur

        r_safe = float(r) if pd.notna(r) else 0.0
        current_nav *= 1.0 + r_safe
        current_no_dep *= 1.0 + r_safe

        nav_with.append(current_nav)
        nav_without.append(current_no_dep)
        cum_dep.append(cum_deposit)

    idx = portfolio_returns.index
    return NavResult(
        nav_eur=pd.Series(nav_with, index=idx, name="nav_eur"),
        cum_deposits_eur=pd.Series(cum_dep, index=idx, name="cum_deposits_eur"),
        nav_no_deposit_eur=pd.Series(nav_without, index=idx, name="nav_no_deposit_eur"),
    )


def static_blend_returns(
    asset_returns: pd.DataFrame,
    weights: dict[str, float],
    rebalance_dates: Iterable[pd.Timestamp],
) -> pd.Series:
    """Daily simple returns of a fixed-weight portfolio with periodic rebalance.

    Between rebalances, weights drift with realized returns (no daily rebalance).
    """
    cols = [c for c in weights if c in asset_returns.columns]
    if not cols:
        raise ValueError(f"none of {list(weights)} in asset_returns columns")
    target = np.array([weights[c] for c in cols], dtype=np.float64)
    target = target / target.sum()

    rets = asset_returns[cols].fillna(0.0).values
    rebal_set = set(pd.DatetimeIndex(rebalance_dates))
    n = rets.shape[0]
    out = np.zeros(n)
    current_w = target.copy()
    for t in range(n):
        date = asset_returns.index[t]
        if date in rebal_set:
            current_w = target.copy()
        port_r = float(current_w @ rets[t])
        out[t] = port_r
        if 1.0 + port_r > 1e-9:
            new_w = current_w * (1.0 + rets[t]) / (1.0 + port_r)
            s = new_w.sum()
            if s > 0:
                current_w = new_w / s
    return pd.Series(out, index=asset_returns.index)
