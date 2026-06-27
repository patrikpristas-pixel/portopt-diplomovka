from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float(equity_curve(returns).iloc[-1] - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        return 0.0
    total = total_return(returns)
    return float((1.0 + total) ** (periods_per_year / len(returns)) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, rf_per_year: float = 0.0, periods_per_year: int = 252
) -> float:
    excess_ann = returns.mean() * periods_per_year - rf_per_year
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0:
        return 0.0
    return float(excess_ann / vol)


def sortino_ratio(
    returns: pd.Series, rf_per_year: float = 0.0, periods_per_year: int = 252
) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year))
    if downside_vol == 0:
        return 0.0
    excess_ann = returns.mean() * periods_per_year - rf_per_year
    return float(excess_ann / downside_vol)


def max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    eq = equity_curve(returns)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(annualized_return(returns, periods_per_year) / abs(mdd))


def avg_annual_turnover(turnover: pd.Series, rebalances_per_year: int = 12) -> float:
    if len(turnover) == 0:
        return 0.0
    return float(turnover.mean() * rebalances_per_year)


def alpha_beta_vs_benchmark(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252
) -> tuple[float, float]:
    """OLS regression of strategy returns on benchmark returns. Returns (alpha_annualized, beta)."""
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return 0.0, 0.0
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    cov = np.cov(x, y, ddof=1)
    beta = cov[0, 1] / cov[0, 0] if cov[0, 0] > 0 else 0.0
    alpha_per_period = y.mean() - beta * x.mean()
    return float(alpha_per_period * periods_per_year), float(beta)


def summary(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict:
    out = {
        "total_return": total_return(returns),
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year=periods_per_year),
        "calmar": calmar_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "n_observations": int(len(returns)),
    }
    if turnover is not None:
        out["avg_annual_turnover"] = avg_annual_turnover(turnover)
    if benchmark_returns is not None:
        a, b = alpha_beta_vs_benchmark(returns, benchmark_returns, periods_per_year)
        out["alpha_vs_benchmark"] = a
        out["beta_vs_benchmark"] = b
    return out
