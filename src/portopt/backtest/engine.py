from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from portopt.backtest.costs import apply_costs, turnover
from portopt.backtest.splits import monthly_rebalance_dates
from portopt.strategies.base import Strategy


@dataclass
class BacktestConfig:
    cost_bps: float = 0.0  # long-term buy-and-hold: no transaction cost simulated
    rebalance_freq: str = "monthly"  # only matters when first rebalance happens
    min_holding_period_days: int = 0
    buy_and_hold: bool = True  # AI picks weights once, holds forever — tax-efficient default
    first_rebalance_after: pd.Timestamp | None = None  # skip rebalances before this date
                                                       # (so strategy has enough warmup history)


@dataclass
class BacktestResult:
    portfolio_returns: pd.Series  # simple returns, indexed by date
    weights: pd.DataFrame  # per-day weights (held weights, NOT only at rebalance)
    turnover_per_rebalance: pd.Series  # |Δw| sum at each rebalance date
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)


def _rebalance_set(index: pd.DatetimeIndex, freq: str) -> set[pd.Timestamp]:
    if freq == "monthly":
        return monthly_rebalance_dates(index)
    raise NotImplementedError(f"rebalance_freq={freq} not supported")


def run_backtest(
    strategy: Strategy,
    log_returns: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """Walk-forward backtest.

    Convention: at rebalance date *t*, strategy sees data strictly before *t*.
    New weights are applied to *t*'s return; transaction cost is debited from
    *t*'s portfolio return.

    `log_returns` should be a DataFrame of daily log returns, dates as index,
    tickers as columns. NaN values are treated as 0% return for that day.
    """
    n_assets = log_returns.shape[1]
    rebalance = _rebalance_set(log_returns.index, config.rebalance_freq)

    current_w = np.zeros(n_assets)
    weights_rows = np.zeros((len(log_returns), n_assets))
    port_rets = np.zeros(len(log_returns))
    turnover_at: dict[pd.Timestamp, float] = {}
    rebalance_dates: list[pd.Timestamp] = []

    log_arr = log_returns.values
    dates = log_returns.index

    # Holding period: after first rebalance, lock weights for `min_holding_period_days`
    first_rebalance_date: pd.Timestamp | None = None
    holding_end: pd.Timestamp | None = None

    for t_idx, date in enumerate(dates):
        cost = 0.0
        # Trigger candidates: monthly rebalance day OR forced initial rebalance
        # (first trading day >= first_rebalance_after, even if not month-end).
        # This ensures the strategy is invested from the very start of its test
        # window — no dead zero-return days waiting for end-of-month.
        forced_initial = (
            config.first_rebalance_after is not None
            and first_rebalance_date is None
            and date >= config.first_rebalance_after
        )
        if date in rebalance or forced_initial:
            do_rebalance = True
            if (
                config.first_rebalance_after is not None
                and date < config.first_rebalance_after
            ):
                do_rebalance = False  # not enough warmup history yet
            elif first_rebalance_date is not None:
                if config.buy_and_hold:
                    do_rebalance = False  # buy-and-hold: only ever first rebalance
                elif holding_end is not None and date < holding_end:
                    do_rebalance = False  # still inside initial holding period
            if do_rebalance:
                history = log_returns.iloc[:t_idx]
                target_w = strategy.get_weights(date, history)
                target_w = np.asarray(target_w, dtype=float)
                if target_w.shape != (n_assets,):
                    raise ValueError(f"strategy returned wrong shape: {target_w.shape}")
                tov = turnover(current_w, target_w)
                cost = apply_costs(current_w, target_w, config.cost_bps)
                turnover_at[date] = tov
                rebalance_dates.append(date)
                current_w = target_w
                if first_rebalance_date is None:
                    first_rebalance_date = date
                    if config.min_holding_period_days > 0:
                        holding_end = date + pd.Timedelta(days=config.min_holding_period_days)

        today_log = log_arr[t_idx]
        today_simple = np.expm1(np.where(np.isnan(today_log), 0.0, today_log))
        port_ret = float(current_w @ today_simple) - cost
        port_rets[t_idx] = port_ret
        weights_rows[t_idx] = current_w

        # Drift weights with realized returns (true buy-and-hold behavior).
        # On next rebalance (if any), `current_w` is reset to the strategy's
        # new target. Without rebalance, drifting continues forever.
        if 1.0 + port_ret > 1e-9 and current_w.sum() > 1e-9:
            new_w = current_w * (1.0 + today_simple)
            s = new_w.sum()
            if s > 1e-9:
                current_w = new_w / s

    portfolio_returns = pd.Series(port_rets, index=dates, name="returns")
    weights = pd.DataFrame(weights_rows, index=dates, columns=log_returns.columns)
    turnover_series = pd.Series(turnover_at, name="turnover").sort_index()

    return BacktestResult(
        portfolio_returns=portfolio_returns,
        weights=weights,
        turnover_per_rebalance=turnover_series,
        rebalance_dates=rebalance_dates,
    )
