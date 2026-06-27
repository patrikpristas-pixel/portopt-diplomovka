from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from portopt.evaluation.metrics import equity_curve


def plot_equity_curve(
    returns: pd.Series,
    title: str = "Equity Curve",
    benchmark: pd.Series | None = None,
    benchmark_name: str = "benchmark",
    out: Path | None = None,
) -> plt.Figure:
    sns.set_style("whitegrid")
    eq = equity_curve(returns)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eq.index, eq.values, label=returns.name or "strategy", color="C0", linewidth=1.5)
    if benchmark is not None:
        bench_eq = equity_curve(benchmark)
        ax.plot(bench_eq.index, bench_eq.values, label=benchmark_name, color="gray", linewidth=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("growth of $1 (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_drawdown(returns: pd.Series, title: str = "Drawdown", out: Path | None = None) -> plt.Figure:
    sns.set_style("whitegrid")
    eq = equity_curve(returns)
    dd = eq / eq.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.35)
    ax.plot(dd.index, dd.values, color="darkred", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_weights_heatmap(
    weights: pd.DataFrame, title: str = "Weights over time", out: Path | None = None
) -> plt.Figure:
    sns.set_style("white")
    monthly = weights.resample("ME").last()
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(monthly.T, cmap="viridis", cbar_kws={"label": "weight"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("date (month-end)")
    ax.set_ylabel("ticker")
    n_ticks = min(20, monthly.shape[0])
    tick_idx = np.linspace(0, monthly.shape[0] - 1, n_ticks).astype(int)
    ax.set_xticks(tick_idx + 0.5)
    ax.set_xticklabels([monthly.index[i].strftime("%Y-%m") for i in tick_idx], rotation=45, ha="right")
    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
    return fig


def plot_strategy_overlay(
    returns_by_strategy: dict[str, pd.Series],
    title: str = "Strategies — equity curves",
    out: Path | None = None,
) -> plt.Figure:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, r in returns_by_strategy.items():
        eq = equity_curve(r)
        ax.plot(eq.index, eq.values, label=name, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("growth of $1 (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig
