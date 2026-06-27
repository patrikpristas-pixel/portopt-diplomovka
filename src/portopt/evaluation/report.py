from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from portopt.backtest.engine import BacktestResult
from portopt.evaluation import plots
from portopt.evaluation.metrics import summary
from portopt.utils.io import ensure_dir, write_parquet


def save_run(
    result: BacktestResult,
    out_dir: Path,
    strategy_name: str,
    benchmark_returns: pd.Series | None = None,
    benchmark_name: str = "SPY",
    config_snapshot: dict | None = None,
) -> dict:
    """Persist a single backtest run: weights, returns, metrics, plots, optional tearsheet."""
    ensure_dir(out_dir)

    write_parquet(result.portfolio_returns.to_frame("returns"), out_dir / "returns.parquet")
    write_parquet(result.weights, out_dir / "weights.parquet")
    if len(result.turnover_per_rebalance) > 0:
        write_parquet(
            result.turnover_per_rebalance.to_frame("turnover"),
            out_dir / "turnover.parquet",
        )

    metrics = summary(
        result.portfolio_returns,
        turnover=result.turnover_per_rebalance,
        benchmark_returns=benchmark_returns,
    )
    metrics["strategy"] = strategy_name
    metrics["start_date"] = str(result.portfolio_returns.index[0].date())
    metrics["end_date"] = str(result.portfolio_returns.index[-1].date())
    if benchmark_returns is not None:
        metrics["benchmark"] = benchmark_name

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str, ensure_ascii=False)

    if config_snapshot is not None:
        with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
            for k, v in config_snapshot.items():
                f.write(f"{k}: {v}\n")

    plots.plot_equity_curve(
        result.portfolio_returns,
        title=f"{strategy_name} — equity curve",
        benchmark=benchmark_returns,
        benchmark_name=benchmark_name,
        out=out_dir / "equity_curve.png",
    )
    plots.plot_drawdown(
        result.portfolio_returns,
        title=f"{strategy_name} — drawdown",
        out=out_dir / "drawdown.png",
    )
    plots.plot_weights_heatmap(
        result.weights,
        title=f"{strategy_name} — weights over time",
        out=out_dir / "weights_heatmap.png",
    )

    try:
        import quantstats as qs

        qs.reports.html(
            result.portfolio_returns,
            benchmark=benchmark_returns,
            output=str(out_dir / "tearsheet.html"),
            title=strategy_name,
        )
    except Exception as e:
        logger.warning(f"quantstats tearsheet failed: {e}")

    logger.info(f"saved {strategy_name} → {out_dir}")
    return metrics
