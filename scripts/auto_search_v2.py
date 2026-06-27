"""Per-portfolio Optuna TPE auto-search with walk-forward backtesting.

Each TRIAL = one set of NN hyperparameters, evaluated across ALL walk-forward
windows of the scenario. For each window: train NN on data strictly before
window.test_start (no peeking), freeze, simulate buy-and-hold for the window.
Returns from all windows are concatenated; monthly deposits are applied across
the full regime to produce one NAV trajectory.

Trial output (saved under scenarios/<sid>/trial_data/<trial_id>/):
  hyperparams.json         — the chosen Optuna hyperparams
  walkforward.parquet      — per-window: train_start, train_end, test_start,
                              test_end, return, sharpe, max_dd, final_nav_window
  weights.parquet          — AI weights chosen at each window's test_start
  returns.parquet          — concatenated daily simple returns across windows
  nav.parquet              — NAV trajectory with monthly deposits
  models/window_<i>.pt     — per-window NN checkpoint

Baselines are computed once per scenario, also walk-forward.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from loguru import logger

from portopt.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from portopt.backtest.nav import compute_nav
from portopt.backtest.splits import monthly_rebalance_dates
from portopt.evaluation.metrics import summary
from portopt.evaluation.statistics import (
    diebold_mariano,
    probabilistic_sharpe_ratio,
    sharpe_bootstrap_ci,
)
from portopt.models.dataset import build_window_dataset
from portopt.models.train_predictor import TrainConfig, save_checkpoint, train_policy
from portopt.portfolio import (
    DEFAULT_HOLDOUT_YEARS,
    DEFAULT_MIN_TRAIN_YEARS,
    DEFAULT_TEST_WINDOW_MONTHS,
    Portfolio,
    Scenario,
    WalkForwardWindow,
    make_walkforward_windows,
    split_search_holdout,
)
from portopt.strategies.base import Strategy
from portopt.strategies.black_litterman import BlackLitterman
from portopt.strategies.equal_weight import EqualWeight
from portopt.strategies.markowitz import Markowitz
from portopt.strategies.momentum import MomentumPortfolio
from portopt.strategies.nn_softmax import NNSoftmaxStrategy
from portopt.utils.io import DATA_PROCESSED, ensure_dir
from portopt.utils.logging import setup_logger

WARMUP_DAYS = 400  # cushion: 252d cov lookback + 60d feature lookback + buffer


# ----- Control file -----


def read_control(portfolio: Portfolio) -> dict:
    if not portfolio.control_file.exists():
        return {"running": False}
    try:
        return json.loads(portfolio.control_file.read_text(encoding="utf-8"))
    except Exception:
        return {"running": False}


def write_control(portfolio: Portfolio, ctrl: dict) -> None:
    portfolio.ensure_base()
    portfolio.control_file.write_text(json.dumps(ctrl, indent=2, default=str), encoding="utf-8")


# ----- Walk-forward helpers -----


def get_windows(scenario: Scenario, log_returns: pd.DataFrame) -> list[WalkForwardWindow]:
    """LEGACY — returns ONLY search-period windows. Use get_all_windows() for both."""
    search, _holdout, _ = get_all_windows(scenario, log_returns)
    return search


def get_all_windows(
    scenario: Scenario, log_returns: pd.DataFrame
) -> tuple[list[WalkForwardWindow], list[WalkForwardWindow], pd.Timestamp]:
    """Return (search_windows, holdout_windows, holdout_start).

    Optuna sees ONLY search_windows when computing the objective.
    Holdout windows are evaluated per trial but never feed back into HP search.
    This is the TRUE out-of-sample anchor for honest evaluation.
    """
    knobs = scenario.knobs
    date_start = pd.Timestamp(knobs["date_start"])
    date_end = pd.Timestamp(knobs["date_end"])
    data_start = log_returns.index[0]
    # Back-compat: old scenarios used test_window_years; new ones use test_window_months.
    if "test_window_months" in knobs:
        test_window_months = int(knobs["test_window_months"])
    elif "test_window_years" in knobs:
        test_window_months = int(knobs["test_window_years"]) * 12
    else:
        test_window_months = DEFAULT_TEST_WINDOW_MONTHS
    holdout_years = int(knobs.get("holdout_years", DEFAULT_HOLDOUT_YEARS))
    search, holdout, hold_start = split_search_holdout(
        date_start=date_start,
        date_end=date_end,
        data_start=data_start,
        holdout_years=holdout_years,
        test_window_months=test_window_months,
        min_train_years=DEFAULT_MIN_TRAIN_YEARS,
    )
    if not search:
        raise SystemExit(
            f"No SEARCH windows fit. date_start={date_start.date()}, "
            f"holdout_start={hold_start.date()}, need ≥ {DEFAULT_MIN_TRAIN_YEARS}y "
            f"training before search begins."
        )
    return search, holdout, hold_start


def run_strategy_walkforward(
    strategy_factory,
    panel: pd.DataFrame,
    windows: list[WalkForwardWindow],
    max_weight: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run a strategy across all walk-forward windows.

    Returns:
      full_returns: daily simple returns concatenated across windows
      weights_at_window_start: DataFrame indexed by window_idx, columns = assets,
                               value = weight chosen at window.test_start
    """
    all_returns: list[pd.Series] = []
    weights_rows: list[dict] = []
    for w in windows:
        strat = strategy_factory(w)
        bt_cfg = BacktestConfig(
            cost_bps=0.0,
            buy_and_hold=True,
            first_rebalance_after=w.test_start,
        )
        warmup_start = w.test_start - pd.Timedelta(days=WARMUP_DAYS)
        sub_panel = panel.loc[warmup_start : w.test_end].dropna(how="all")
        if len(sub_panel) < 50:
            logger.warning(f"window {w.idx}: sub_panel too short, skipping")
            continue
        try:
            result = run_backtest(strat, sub_panel, bt_cfg)
        except Exception as e:
            logger.error(f"window {w.idx}: backtest failed: {e}")
            continue
        mask = (result.portfolio_returns.index >= w.test_start) & (
            result.portfolio_returns.index <= w.test_end
        )
        win_rets = result.portfolio_returns[mask]
        if len(win_rets) == 0:
            continue
        all_returns.append(win_rets)
        # Capture weights at first test-window day
        w_idx_arr = np.where(result.weights.index >= w.test_start)[0]
        if len(w_idx_arr) > 0:
            row = {"window_idx": w.idx, "test_start": w.test_start}
            for c in result.weights.columns:
                row[c] = float(result.weights.iloc[w_idx_arr[0]][c])
            weights_rows.append(row)

    full_returns = (
        pd.concat(all_returns).sort_index() if all_returns else pd.Series(dtype=float)
    )
    weights_df = pd.DataFrame(weights_rows) if weights_rows else pd.DataFrame()
    return full_returns, weights_df


def compute_window_metrics(
    full_returns: pd.Series,
    windows: list[WalkForwardWindow],
) -> pd.DataFrame:
    """Per-window metrics from concatenated daily returns."""
    if len(full_returns) == 0 or not isinstance(full_returns.index, pd.DatetimeIndex):
        return pd.DataFrame()
    rows = []
    for w in windows:
        m = (full_returns.index >= w.test_start) & (full_returns.index <= w.test_end)
        r = full_returns[m]
        if len(r) < 5:
            continue
        s = summary(r)
        rows.append(
            {
                "window_idx": w.idx,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "return": float(s["total_return"]),
                "sharpe": float(s["sharpe"]),
                "max_drawdown": float(s["max_drawdown"]),
                "ann_return": float(s["ann_return"]),
                "ann_volatility": float(s["ann_volatility"]),
                "n_days": int(s["n_observations"]),
            }
        )
    return pd.DataFrame(rows)


def compute_overall_metrics(
    full_returns: pd.Series,
    nav: pd.Series,
) -> dict:
    s = summary(full_returns)
    s["final_nav_eur"] = float(nav.iloc[-1]) if len(nav) > 0 else 0.0
    s["initial_nav_eur"] = float(nav.iloc[0]) if len(nav) > 0 else 0.0
    s["nav_growth_eur"] = s["final_nav_eur"] - s["initial_nav_eur"]
    return s


# ----- Baselines (cached per scenario) -----


def compute_baselines(scenario: Scenario, log_returns: pd.DataFrame) -> dict[str, dict]:
    """Run all baseline strategies + benchmarks across SEARCH + HOLDOUT windows.

    Each strategy is evaluated over the FULL period (search ∪ holdout) but
    metrics are computed SEPARATELY for each regime so the "won" check uses
    only search-period data (no holdout leakage).

    Saves per-strategy:
      baselines.parquet              — search-period metrics (used by 'won' check)
      baselines_holdout.parquet      — holdout-period metrics (honest OOS)
      baseline_returns.parquet       — daily returns concatenated (full period)
      baseline_nav.parquet           — NAV trajectory (full period)
      baselines_walkforward.parquet  — per-window per-strategy metrics

    Returns dict with nested {strategy: {"search": {...}, "holdout": {...}}}.
    """
    portfolio = scenario.portfolio
    knobs = scenario.knobs
    universe = [t for t in portfolio.tickers if t in log_returns.columns]
    if len(universe) < 2:
        raise SystemExit(f"portfolio {portfolio.name}: not enough tickers in data")

    search_windows, holdout_windows, holdout_start = get_all_windows(scenario, log_returns)
    windows = search_windows + holdout_windows
    test_start_all = windows[0].test_start
    test_end_all = windows[-1].test_end
    monthly_deposit = float(knobs.get("monthly_deposit_eur", 200))
    max_w = float(knobs.get("max_weight", 0.25))

    panel_universe = log_returns[universe]
    log_simple_all = np.expm1(log_returns)

    logger.info(
        f"baselines: search {len(search_windows)} window(s) "
        f"{search_windows[0].test_start.date()}→{search_windows[-1].test_end.date()}, "
        f"holdout {len(holdout_windows)} window(s) "
        f"{holdout_start.date()}→{test_end_all.date() if holdout_windows else 'n/a'}"
    )

    strategy_factories = {
        "equal_weight": lambda w: EqualWeight(),
        "markowitz": lambda w: Markowitz(lookback=252, max_weight=max_w),
        "black_litterman": lambda w: BlackLitterman(lookback=252, max_weight=max_w),
        "momentum": lambda w: MomentumPortfolio(lookback=126, max_weight=max_w),
    }

    returns_dict: dict[str, pd.Series] = {}
    wf_rows_per_strategy: dict[str, pd.DataFrame] = {}

    for name, fac in strategy_factories.items():
        logger.info(f"  baseline {name}…")
        full_ret, _w = run_strategy_walkforward(fac, panel_universe, windows, max_w)
        returns_dict[name] = full_ret
        wf_rows_per_strategy[name] = compute_window_metrics(full_ret, windows)

    # Benchmark — buy-and-hold SPY as the single market control.
    def bench_returns(ticker: str) -> pd.Series:
        return log_simple_all[ticker].loc[test_start_all:test_end_all]

    if "SPY" in log_returns.columns:
        returns_dict["SPY"] = bench_returns("SPY")
        wf_rows_per_strategy["SPY"] = compute_window_metrics(returns_dict["SPY"], windows)

    # Align daily index
    aligned_idx = pd.DatetimeIndex(
        sorted(set().union(*[r.index for r in returns_dict.values()]))
    )
    aligned_idx = aligned_idx[(aligned_idx >= test_start_all) & (aligned_idx <= test_end_all)]
    returns_df = pd.DataFrame(
        {k: v.reindex(aligned_idx).fillna(0.0) for k, v in returns_dict.items()},
        index=aligned_idx,
    )

    rebal_full = monthly_rebalance_dates(log_returns.index)
    rebal_test = pd.DatetimeIndex(
        [d for d in rebal_full if test_start_all <= d <= test_end_all]
    )

    nav_dict = {}
    for name in returns_df.columns:
        nav_res = compute_nav(
            returns_df[name],
            rebalance_dates=rebal_test,
            monthly_deposit_eur=monthly_deposit,
            initial_nav_eur=0.0,
            deposit_at_start=True,
        )
        nav_dict[name] = nav_res.nav_eur
    nav_df = pd.DataFrame(nav_dict, index=aligned_idx)
    nav_df.index.name = "date"
    returns_df.index.name = "date"

    scenario.ensure_dirs()
    returns_df.to_parquet(scenario.baseline_returns_path)
    nav_df.to_parquet(scenario.baseline_nav_path)

    # Split metrics by regime: search-period (Optuna's 'won' check) vs holdout (honest OOS)
    holdout_mask = returns_df.index >= holdout_start
    search_mask = ~holdout_mask
    metrics_search: dict[str, dict] = {}
    metrics_holdout: dict[str, dict] = {}
    for name in returns_df.columns:
        if search_mask.any():
            metrics_search[name] = compute_overall_metrics(
                returns_df[name][search_mask], nav_df[name][search_mask]
            )
        if holdout_mask.any():
            metrics_holdout[name] = compute_overall_metrics(
                returns_df[name][holdout_mask], nav_df[name][holdout_mask]
            )
    pd.DataFrame(metrics_search).T.to_parquet(scenario.baselines_path)
    if metrics_holdout:
        pd.DataFrame(metrics_holdout).T.to_parquet(
            scenario.base_dir / "baselines_holdout.parquet"
        )

    wf_long = []
    for name, df in wf_rows_per_strategy.items():
        d = df.copy()
        d["strategy"] = name
        # Tag each row as search or holdout based on test_start vs holdout_start
        d["regime"] = d["test_start"].apply(
            lambda ts: "holdout" if pd.Timestamp(ts) >= holdout_start else "search"
        )
        wf_long.append(d)
    if wf_long:
        pd.concat(wf_long, ignore_index=True).to_parquet(
            scenario.base_dir / "baselines_walkforward.parquet"
        )

    logger.info(
        f"baselines saved: search → {scenario.baselines_path.name}, "
        f"holdout → baselines_holdout.parquet ({len(metrics_holdout)} strategies)"
    )

    # ----- Per-window regime tagging (bull/bear/sideways via SPY) -----
    # Used in UI to contextualize each window's results — a strategy may shine
    # in bulls but collapse in bears, which is critical for honest evaluation.
    regimes_rows: list[dict] = []
    spy_simple = log_simple_all.get("SPY") if "SPY" in log_returns.columns else None
    for w in windows:
        if spy_simple is not None:
            sub = spy_simple.loc[w.test_start : w.test_end]
            spy_ret = float((1.0 + sub).prod() - 1.0) if len(sub) > 0 else float("nan")
        else:
            spy_ret = float("nan")
        if pd.isna(spy_ret):
            tag = "unknown"
        elif spy_ret > 0.10:
            tag = "bull"
        elif spy_ret < -0.10:
            tag = "bear"
        else:
            tag = "sideways"
        regimes_rows.append({
            "window_idx": int(w.idx),
            "test_start": pd.Timestamp(w.test_start),
            "test_end": pd.Timestamp(w.test_end),
            "regime_phase": "holdout" if pd.Timestamp(w.test_start) >= holdout_start else "search",
            "spy_return": spy_ret,
            "market_regime": tag,
        })
    if regimes_rows:
        pd.DataFrame(regimes_rows).to_parquet(scenario.base_dir / "window_regimes.parquet")
    # Nested dict for clean access
    nested: dict[str, dict] = {}
    for name in set(list(metrics_search.keys()) + list(metrics_holdout.keys())):
        nested[name] = {
            "search": metrics_search.get(name, {}),
            "holdout": metrics_holdout.get(name, {}),
        }
    return nested


def load_baselines(scenario: Scenario):
    """Load baselines from disk.

    Returns: (nested_metrics, returns_df, nav_df) or None.
    nested_metrics: {strategy: {"search": {...}, "holdout": {...}}}
    """
    if not scenario.baselines_path.exists():
        return None
    metrics_df = pd.read_parquet(scenario.baselines_path)
    metrics_search = {row: metrics_df.loc[row].to_dict() for row in metrics_df.index}
    holdout_path = scenario.base_dir / "baselines_holdout.parquet"
    metrics_holdout: dict[str, dict] = {}
    if holdout_path.exists():
        h_df = pd.read_parquet(holdout_path)
        metrics_holdout = {row: h_df.loc[row].to_dict() for row in h_df.index}
    nested = {
        name: {
            "search": metrics_search.get(name, {}),
            "holdout": metrics_holdout.get(name, {}),
        }
        for name in set(list(metrics_search.keys()) + list(metrics_holdout.keys()))
    }
    returns_df = pd.read_parquet(scenario.baseline_returns_path)
    nav_df = pd.read_parquet(scenario.baseline_nav_path)
    returns_df.index.name = "date"
    nav_df.index.name = "date"
    return nested, returns_df, nav_df


# ----- One trial: walk-forward training + backtest -----


def _train_for_window(
    log_returns: pd.DataFrame,
    universe: list[str],
    window: WalkForwardWindow,
    hp: dict,
    criterion: str,
):
    """Train NN policy on data strictly before window.test_start; return TrainResult."""
    panel_train = log_returns[universe].loc[: window.test_start - pd.Timedelta(days=1)]
    # Validation: last 1y of training data (still strictly before test_start)
    val_start = window.test_start - pd.DateOffset(years=1)
    horizon = int(hp["horizon"])
    lookback = int(hp["lookback"])

    ds = build_window_dataset(
        log_returns=panel_train,
        asset_universe=universe,
        lookback=lookback,
        horizon=horizon,
        stride=int(hp.get("stride", 5)),
    )
    train_ds = ds.slice_by_date(panel_train.index[0], val_start)
    val_ds = ds.slice_by_date(val_start, window.test_start)
    if len(train_ds) < 30 or len(val_ds) < 5:
        # Fall back: use all data, no early stopping
        train_ds = ds
        val_ds = ds

    cfg = TrainConfig(
        epochs=int(hp["epochs"]),
        batch_size=int(hp["batch_size"]),
        lr=float(hp["lr"]),
        weight_decay=float(hp.get("weight_decay", 1e-4)),
        hidden=int(hp["hidden"]),
        n_layers=int(hp.get("n_layers", 2)),
        dropout=float(hp["dropout"]),
        early_stop_patience=int(hp.get("early_stop_patience", 8)),
        seed=int(hp["seed"]),
        criterion=criterion,
        entropy_bonus=float(hp.get("entropy_bonus", 0.0)),
    )
    return train_policy(train_ds, val_ds, cfg)


def _train_eval_block(
    log_returns: pd.DataFrame,
    universe: list[str],
    windows: list[WalkForwardWindow],
    hp: dict,
    criterion: str,
    max_w: float,
    models_dir: Path,
    tag: str,
    history_rows: list[dict] | None = None,
) -> tuple[pd.Series, pd.DataFrame, dict[int, dict], float, float]:
    """Train a fresh NN per window in `windows`, then evaluate buy-and-hold.

    Returns (returns_series, weights_df, train_metadata, train_time, backtest_time).
    Checkpoints saved to models_dir/window_<idx>_<tag>.pt.
    If `history_rows` provided (mutable list), per-epoch training metrics are
    appended for later persistence — used for train/val loss curves in UI.
    """
    ckpt_paths: dict[int, Path] = {}
    train_results: dict[int, dict] = {}
    t0 = time.time()
    for w in windows:
        tr = _train_for_window(log_returns, universe, w, hp, criterion)
        ckpt_path = models_dir / f"window_{w.idx:02d}_{tag}.pt"
        save_checkpoint(tr, ckpt_path)
        ckpt_paths[w.idx] = ckpt_path
        train_results[w.idx] = {
            "val_loss": tr.best_val_loss,
            "val_sharpe": tr.val_sharpe,
            "val_total_return": tr.val_total_return,
        }
        # Persist per-epoch training history for overfit diagnostics
        if history_rows is not None:
            for ep in tr.history:
                history_rows.append({
                    "window_idx": int(w.idx),
                    "regime": tag,
                    "epoch": int(ep.epoch),
                    "train_loss": float(ep.train_loss),
                    "val_loss": float(ep.val_loss),
                    "val_sharpe": float(ep.val_sharpe),
                    "val_total_return": float(ep.val_total_return),
                })
    train_time = time.time() - t0

    def ai_factory(w):
        return NNSoftmaxStrategy(
            checkpoint_path=ckpt_paths[w.idx],
            asset_universe=universe,
            lookback=int(hp["lookback"]),
            max_weight=max_w,
        )

    panel = log_returns[universe]
    t0 = time.time()
    rets, weights_df = run_strategy_walkforward(ai_factory, panel, windows, max_w)
    backtest_time = time.time() - t0
    return rets, weights_df, train_results, train_time, backtest_time


def _won_check(
    ai_metrics: dict,
    competitor_metrics: dict[str, dict],
    criterion: str,
) -> tuple[bool, float, float, float, bool]:
    """Win check vs the allowed competitor set for this app.

    Returns (won, objective, max_comp_sharpe, max_comp_nav, won_vs_bench).
    `won` checks only against the supported comparison set:
    Markowitz, Black-Litterman, Momentum, and SPY.
    `won_vs_bench` is the benchmark-only check against SPY.

    `won_vs_all` is exposed in the trial row as an explicit alias of `won`
    (kept for backward compatibility with the UI / legacy parquet schema).
    """
    if not competitor_metrics:
        return False, ai_metrics.get("sharpe", 0.0), 0.0, 0.0, False
    max_comp_sharpe = max((m.get("sharpe", 0.0) for m in competitor_metrics.values()), default=0.0)
    max_comp_nav = max((m.get("final_nav_eur", 0.0) for m in competitor_metrics.values()), default=0.0)
    bench_keys = ("SPY",)
    bench_only = {k: competitor_metrics[k] for k in bench_keys if k in competitor_metrics}
    max_bench_sharpe = max((m.get("sharpe", 0.0) for m in bench_only.values()), default=0.0)
    max_bench_nav = max((m.get("final_nav_eur", 0.0) for m in bench_only.values()), default=0.0)

    if criterion == "sharpe":
        won = ai_metrics["sharpe"] > max_comp_sharpe
        objective = ai_metrics["sharpe"]
        won_vs_bench = ai_metrics["sharpe"] > max_bench_sharpe
    elif criterion == "total_return":
        won = ai_metrics["final_nav_eur"] > max_comp_nav
        objective = ai_metrics["final_nav_eur"]
        won_vs_bench = ai_metrics["final_nav_eur"] > max_bench_nav
    elif criterion == "beat_benchmarks":
        sharpe_gap = ai_metrics["sharpe"] - max_comp_sharpe
        nav_gap_pct = (
            (ai_metrics["final_nav_eur"] / max_comp_nav - 1.0) if max_comp_nav > 0 else 0.0
        )
        won = (sharpe_gap > 0) and (nav_gap_pct > 0)
        objective = sharpe_gap + nav_gap_pct
        won_vs_bench = (
            (ai_metrics["sharpe"] > max_bench_sharpe)
            and (ai_metrics["final_nav_eur"] > max_bench_nav)
        )
    else:
        won = ai_metrics["sharpe"] > max_comp_sharpe
        objective = ai_metrics["sharpe"]
        won_vs_bench = ai_metrics["sharpe"] > max_bench_sharpe
    return bool(won), float(objective), float(max_comp_sharpe), float(max_comp_nav), bool(won_vs_bench)


def run_one_trial(
    scenario: Scenario,
    hp: dict,
    log_returns: pd.DataFrame,
    baseline_metrics: dict[str, dict],
    trial_id: int,
) -> dict:
    """Run ONE trial with TRUE holdout architecture.

    Phase 1 (SEARCH): Train walk-forward on search windows. Optuna sees this.
    Phase 2 (HOLDOUT): Train walk-forward on holdout windows. Stored only.

    The trial row stores BOTH regimes' metrics so:
      - `objective` comes from SEARCH (Optuna's HP-selection signal)
      - `holdout_*` columns are the honest OOS estimate (never feeds back)
    """
    portfolio = scenario.portfolio
    knobs = scenario.knobs
    universe = [t for t in portfolio.tickers if t in log_returns.columns]
    monthly_deposit = float(knobs.get("monthly_deposit_eur", 200))
    max_w = float(knobs.get("max_weight", 0.25))
    criterion = knobs.get("win_criterion", "sharpe")
    search_windows, holdout_windows, holdout_start = get_all_windows(scenario, log_returns)

    trial_dir = scenario.trial_dirs / f"trial_{trial_id:04d}"
    models_dir = trial_dir / "models"
    ensure_dir(models_dir)
    (trial_dir / "hyperparams.json").write_text(
        json.dumps(hp, indent=2, default=str), encoding="utf-8"
    )

    # Collect per-epoch training history across all windows for overfit diagnostics
    training_history_rows: list[dict] = []

    # ---- PHASE 1: SEARCH ----
    search_returns, search_weights, search_train_meta, train_time_s, bt_time_s = (
        _train_eval_block(
            log_returns, universe, search_windows, hp, criterion, max_w,
            models_dir, "search", history_rows=training_history_rows,
        )
    )
    if len(search_returns) == 0:
        raise RuntimeError("trial produced 0 search returns — all windows failed")

    # ---- PHASE 2: HOLDOUT (locked OOS) ----
    holdout_returns: pd.Series = pd.Series(dtype=float)
    holdout_weights: pd.DataFrame = pd.DataFrame()
    holdout_train_meta: dict[int, dict] = {}
    train_time_h = 0.0
    bt_time_h = 0.0
    if holdout_windows:
        holdout_returns, holdout_weights, holdout_train_meta, train_time_h, bt_time_h = (
            _train_eval_block(
                log_returns, universe, holdout_windows, hp, criterion, max_w,
                models_dir, "holdout", history_rows=training_history_rows,
            )
        )

    # Persist training history (one row per epoch per window)
    if training_history_rows:
        pd.DataFrame(training_history_rows).to_parquet(trial_dir / "training_history.parquet")

    # ---- NAV (full period, deposits applied continuously) ----
    all_returns = pd.concat([search_returns, holdout_returns]).sort_index()
    all_returns = all_returns[~all_returns.index.duplicated(keep="last")]
    rebal_full = monthly_rebalance_dates(log_returns.index)
    full_start = all_returns.index[0]
    full_end = all_returns.index[-1]
    rebal_test = pd.DatetimeIndex([d for d in rebal_full if full_start <= d <= full_end])
    nav_res = compute_nav(
        all_returns,
        rebalance_dates=rebal_test,
        monthly_deposit_eur=monthly_deposit,
        initial_nav_eur=0.0,
        deposit_at_start=True,
    )
    nav_full = nav_res.nav_eur
    nav_search = nav_full[nav_full.index < holdout_start]
    nav_holdout = nav_full[nav_full.index >= holdout_start]

    # ---- Per-window metrics for both regimes ----
    wf_search = compute_window_metrics(search_returns, search_windows)
    if len(wf_search) > 0:
        wf_search["regime"] = "search"
        wf_search["val_loss"] = wf_search["window_idx"].map(
            lambda i: search_train_meta.get(int(i), {}).get("val_loss", float("nan"))
        )
        wf_search["val_sharpe"] = wf_search["window_idx"].map(
            lambda i: search_train_meta.get(int(i), {}).get("val_sharpe", float("nan"))
        )
    wf_holdout = compute_window_metrics(holdout_returns, holdout_windows) if holdout_windows else pd.DataFrame()
    if len(wf_holdout) > 0:
        wf_holdout["regime"] = "holdout"
        wf_holdout["val_loss"] = wf_holdout["window_idx"].map(
            lambda i: holdout_train_meta.get(int(i), {}).get("val_loss", float("nan"))
        )
        wf_holdout["val_sharpe"] = wf_holdout["window_idx"].map(
            lambda i: holdout_train_meta.get(int(i), {}).get("val_sharpe", float("nan"))
        )
    wf_df = pd.concat([wf_search, wf_holdout], ignore_index=True) if len(wf_holdout) > 0 else wf_search
    wf_df.to_parquet(trial_dir / "walkforward.parquet")

    all_weights = pd.concat([search_weights, holdout_weights], ignore_index=True) if len(holdout_weights) > 0 else search_weights
    if len(all_weights) > 0:
        all_weights.to_parquet(trial_dir / "weights.parquet")
    all_returns.to_frame("returns").to_parquet(trial_dir / "returns.parquet")
    nav_full.to_frame("nav_eur").to_parquet(trial_dir / "nav.parquet")

    # ---- Overall metrics: separate for search and holdout ----
    search_overall = compute_overall_metrics(search_returns, nav_search)
    holdout_overall = (
        compute_overall_metrics(holdout_returns, nav_holdout)
        if len(holdout_returns) > 0
        else {}
    )

    # ---- Won check: SEARCH metrics vs SEARCH baselines (no holdout leakage) ----
    baseline_search = {
        k: v.get("search", {}) if isinstance(v, dict) and "search" in v else v
        for k, v in baseline_metrics.items()
    }
    won, objective, max_comp_sharpe, max_comp_nav, won_vs_bench = _won_check(
        search_overall, baseline_search, criterion
    )
    # `won_vs_all` is just an explicit alias of `won` for UI backward compat.
    won_vs_all = won

    # ---- Inferential statistics (search + holdout separately) ----
    ci_s = sharpe_bootstrap_ci(search_returns, n_bootstrap=500)
    psr_s = probabilistic_sharpe_ratio(search_returns, 0.0)
    psr_s_vs1 = probabilistic_sharpe_ratio(search_returns, 1.0)

    ci_h_low = float("nan")
    ci_h_high = float("nan")
    psr_h = float("nan")
    psr_h_vs1 = float("nan")
    if len(holdout_returns) >= 30:
        ci_h = sharpe_bootstrap_ci(holdout_returns, n_bootstrap=500)
        ci_h_low, ci_h_high = ci_h.ci_low_ann, ci_h.ci_high_ann
        psr_h = probabilistic_sharpe_ratio(holdout_returns, 0.0)
        psr_h_vs1 = probabilistic_sharpe_ratio(holdout_returns, 1.0)

    # Aggregate per-window stats for trial row
    if len(wf_search) > 0:
        median_sharpe = float(wf_search["sharpe"].median())
        median_return = float(wf_search["return"].median())
        median_max_dd = float(wf_search["max_drawdown"].median())
    else:
        median_sharpe = median_return = median_max_dd = float("nan")
    if len(wf_holdout) > 0:
        median_sharpe_h = float(wf_holdout["sharpe"].median())
        median_return_h = float(wf_holdout["return"].median())
    else:
        median_sharpe_h = float("nan")
        median_return_h = float("nan")

    return {
        "trial": trial_id,
        "timestamp": pd.Timestamp.now(),
        **hp,
        "n_windows": len(search_windows),
        "n_holdout_windows": len(holdout_windows),
        "holdout_start": pd.Timestamp(holdout_start),
        # ----- SEARCH-period (Optuna's signal) -----
        "total_return": float(search_overall["total_return"]),
        "ann_return": float(search_overall["ann_return"]),
        "ann_volatility": float(search_overall["ann_volatility"]),
        "sharpe": float(search_overall["sharpe"]),
        "sortino": float(search_overall["sortino"]),
        "calmar": float(search_overall["calmar"]),
        "max_drawdown": float(search_overall["max_drawdown"]),
        "final_nav_eur": float(search_overall["final_nav_eur"]),
        "nav_growth_eur": float(search_overall["nav_growth_eur"]),
        "median_sharpe": median_sharpe,
        "median_return": median_return,
        "median_max_dd": median_max_dd,
        # ----- SEARCH-period statistics -----
        "sharpe_ci_low": float(ci_s.ci_low_ann),
        "sharpe_ci_high": float(ci_s.ci_high_ann),
        "psr": float(psr_s),
        "psr_vs_sharpe_1": float(psr_s_vs1),
        # ----- HOLDOUT-period (TRUE OOS, never seen by Optuna) -----
        "holdout_sharpe": float(holdout_overall.get("sharpe", float("nan"))),
        "holdout_total_return": float(holdout_overall.get("total_return", float("nan"))),
        "holdout_ann_return": float(holdout_overall.get("ann_return", float("nan"))),
        "holdout_max_drawdown": float(holdout_overall.get("max_drawdown", float("nan"))),
        "holdout_final_nav_eur": float(holdout_overall.get("final_nav_eur", float("nan"))),
        "holdout_median_sharpe": median_sharpe_h,
        "holdout_median_return": median_return_h,
        "holdout_sharpe_ci_low": ci_h_low,
        "holdout_sharpe_ci_high": ci_h_high,
        "holdout_psr": psr_h,
        "holdout_psr_vs_sharpe_1": psr_h_vs1,
        # ----- Honest win check (search-only) -----
        "won": won,
        "won_vs_benchmarks": won_vs_bench,
        "won_vs_all": won_vs_all,
        "max_competitor_sharpe": max_comp_sharpe,
        "max_competitor_nav": max_comp_nav,
        "objective": float(objective),
        "train_time_s": float(train_time_s + train_time_h),
        "backtest_time_s": float(bt_time_s + bt_time_h),
    }


def append_trial_row(scenario: Scenario, row: dict) -> None:
    df = (
        pd.read_parquet(scenario.trials_db)
        if scenario.trials_db.exists()
        else pd.DataFrame()
    )
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    # Safety net against any prior duplicate `trial` ids (race condition writes)
    if "trial" in df.columns and df["trial"].duplicated().any():
        df = df.drop_duplicates(subset="trial", keep="last").reset_index(drop=True)
    df.to_parquet(scenario.trials_db)


def get_next_trial_id(scenario: Scenario) -> int:
    if not scenario.trials_db.exists():
        return 1
    df = pd.read_parquet(scenario.trials_db)
    if len(df) == 0:
        return 1
    # Use max + 1 of existing ids; also de-dup tolerance.
    return int(df["trial"].max() + 1)


# ----- Main search loop -----


def search_loop(portfolio: Portfolio, knobs: dict, n_trials: int | None) -> str:
    scenario = Scenario.from_knobs(portfolio, knobs)
    scenario.ensure_dirs()
    log_returns = pd.read_parquet(DATA_PROCESSED / "log_returns.parquet").dropna(how="all")
    universe = [t for t in portfolio.tickers if t in log_returns.columns]
    if len(universe) < 2:
        raise SystemExit(f"{portfolio.name}: only {len(universe)} tickers in data")
    logger.info(f"portfolio={portfolio.name} scenario={scenario.id} universe={universe}")

    cached = load_baselines(scenario)
    if cached is None:
        baseline_metrics = compute_baselines(scenario, log_returns)
    else:
        baseline_metrics = cached[0]

    storage = optuna.storages.RDBStorage(scenario.optuna_storage)
    study_name = f"{portfolio.name}_{scenario.id}"
    # Probe existing trial count BEFORE creating study so we can seed the sampler
    # uniquely for each restart. Without this, restarting the search loop with
    # the same seed re-asks identical hyperparameters as the very first trial
    # (TPESampler re-initializes its RNG from `seed` on each script load).
    _prior_trials = 0
    try:
        _probe = optuna.load_study(study_name=study_name, storage=storage)
        _prior_trials = len(_probe.trials)
    except (KeyError, ValueError):
        pass
    sampler_seed = 42 + _prior_trials  # advances across restarts
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=sampler_seed,
            n_startup_trials=10,  # random warmup before TPE kicks in
        ),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    logger.info(
        f"optuna study='{study_name}' trials_so_far={len(study.trials)} "
        f"sampler_seed={sampler_seed}"
    )

    # Warm-start from a champion's hyperparameters, if requested.
    # The first study.ask() will return EXACTLY these params; subsequent calls
    # fall back to normal TPE exploration.
    ctrl_for_seed = read_control(portfolio)
    seed_hp = ctrl_for_seed.get("seed_hp") if ctrl_for_seed else None
    if seed_hp:
        seed_label = ctrl_for_seed.get("seed_label", "")
        # Restrict to keys the search space defines (filter out anything else
        # to avoid Optuna parameter-name mismatch).
        ALLOWED_PARAMS = {
            "lookback", "horizon", "stride", "epochs", "lr", "hidden",
            "n_layers", "dropout", "batch_size", "weight_decay",
            "entropy_bonus", "early_stop_patience", "seed",
        }
        filtered = {k: v for k, v in seed_hp.items() if k in ALLOWED_PARAMS}
        try:
            study.enqueue_trial(filtered, skip_if_exists=False)
            logger.info(
                f"enqueued champion as warm-start (label='{seed_label}'): {filtered}"
            )
            # Consume the seed (one-shot) so a restart loop doesn't re-enqueue it.
            ctrl_for_seed["seed_hp"] = None
            ctrl_for_seed["seed_label"] = None
            write_control(portfolio, ctrl_for_seed)
        except Exception as e:
            logger.error(f"failed to enqueue seed_hp: {e} (continuing without it)")

    completed_in_run = 0
    best_obj = max((t.value for t in study.trials if t.value is not None), default=-np.inf)

    while True:
        ctrl = read_control(portfolio)
        if not ctrl.get("running", False):
            logger.info("control flag = false, exiting")
            return "stopped"
        if n_trials is not None and completed_in_run >= n_trials:
            logger.info(f"completed {n_trials} trials in this run")
            return "completed"

        trial = study.ask()
        hp = {
            "lookback": trial.suggest_categorical("lookback", [60]),
            "horizon": trial.suggest_categorical("horizon", [63, 126, 252]),
            "stride": trial.suggest_categorical("stride", [5, 10]),
            "epochs": trial.suggest_int("epochs", 20, 80, step=10),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "hidden": trial.suggest_categorical("hidden", [64, 128, 192, 256]),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.05),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "entropy_bonus": trial.suggest_float("entropy_bonus", 0.0, 0.05, step=0.01),
            "early_stop_patience": trial.suggest_categorical("early_stop_patience", [6, 10]),
            "seed": trial.suggest_int("seed", 0, 10000),
        }
        trial_id = get_next_trial_id(scenario)
        logger.info(f"trial #{trial_id} (optuna #{trial.number}) hp={hp}")

        try:
            row = run_one_trial(scenario, hp, log_returns, baseline_metrics, trial_id)
            append_trial_row(scenario, row)
            study.tell(trial, row["objective"])

            if row["objective"] > best_obj:
                best_obj = row["objective"]
                # Copy best window's checkpoint (use last window's, most relevant)
                models_dir = scenario.trial_dirs / f"trial_{trial_id:04d}" / "models"
                ckpts = sorted(models_dir.glob("window_*.pt"))
                if ckpts:
                    shutil.copy(str(ckpts[-1]), str(scenario.best_checkpoint))
                logger.info(
                    f"  NEW BEST objective={best_obj:.4f}, saved → best_predictor.pt"
                )
            won_str = "WIN" if row["won"] else "loss"
            logger.info(
                f"  [{won_str}] sharpe={row['sharpe']:.3f} "
                f"nav={row['final_nav_eur']:,.0f}€ obj={row['objective']:.4f}"
            )
        except Exception as e:
            logger.error(f"trial #{trial_id} failed: {e}\n{traceback.format_exc()}")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)

        completed_in_run += 1
        ctrl = read_control(portfolio)
        ctrl["n_completed_in_run"] = completed_in_run
        write_control(portfolio, ctrl)


def main() -> None:
    setup_logger()
    for cfg_file in Path("portfolios").glob("*.yaml"):
        portfolio = Portfolio.from_yaml(cfg_file)
        ctrl = read_control(portfolio)
        if ctrl.get("running"):
            knobs = ctrl.get("knobs", {})
            n_trials = ctrl.get("n_trials")
            logger.info(f"starting search: {portfolio.name} n_trials={n_trials}")
            try:
                search_loop(portfolio, knobs, n_trials)
            except Exception:
                logger.error(traceback.format_exc())
                ctrl = read_control(portfolio)
                ctrl["running"] = False
                write_control(portfolio, ctrl)
                return
            ctrl = read_control(portfolio)
            ctrl["running"] = False
            write_control(portfolio, ctrl)
            return
    logger.warning("no portfolio has control.running=true — nothing to do")


if __name__ == "__main__":
    sys.exit(main())
