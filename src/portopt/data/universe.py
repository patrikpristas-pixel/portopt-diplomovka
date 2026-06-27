"""Hardcoded asset universe for the thesis.

The optimizer universe (`UNIVERSE`) holds the ~20 tickers we trade.
`BENCHMARKS` is held out for performance comparison only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    ticker: str
    sector: str


LARGE_CAP_STOCKS: tuple[Asset, ...] = (
    Asset("AAPL", "tech"),
    Asset("MSFT", "tech"),
    Asset("GOOGL", "tech"),
    Asset("NVDA", "tech"),
    Asset("META", "tech"),
    Asset("AMZN", "tech"),
    Asset("JPM", "financials"),
    Asset("BAC", "financials"),
    Asset("JNJ", "healthcare"),
    Asset("PFE", "healthcare"),
    Asset("PG", "consumer_staples"),
    Asset("KO", "consumer_staples"),
    Asset("WMT", "consumer_staples"),
)

COMMODITY_RELATED: tuple[Asset, ...] = (
    Asset("XOM", "energy"),
    Asset("CVX", "energy"),
    Asset("NEM", "mining_gold"),
    Asset("FCX", "mining_copper"),
    Asset("AEM", "mining_gold"),
)

BOND_ETFS: tuple[Asset, ...] = (
    Asset("TLT", "bond_long_treasury"),
    Asset("IEF", "bond_mid_treasury"),
    Asset("LQD", "bond_ig_corp"),
    Asset("HYG", "bond_high_yield"),
)

UNIVERSE: tuple[Asset, ...] = LARGE_CAP_STOCKS + COMMODITY_RELATED + BOND_ETFS

BENCHMARKS: tuple[Asset, ...] = (
    Asset("SPY", "benchmark_equity_sp500"),
)

def universe_tickers() -> list[str]:
    return [a.ticker for a in UNIVERSE]


def benchmark_tickers() -> list[str]:
    return [a.ticker for a in BENCHMARKS]


def all_tickers() -> list[str]:
    return universe_tickers() + benchmark_tickers()
