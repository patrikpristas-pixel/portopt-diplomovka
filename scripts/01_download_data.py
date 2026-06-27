"""Download raw daily prices for the universe + benchmarks via yfinance."""

from __future__ import annotations

from omegaconf import OmegaConf

from portopt.data.download import cache_raw, download_prices
from portopt.data.universe import all_tickers
from portopt.utils.io import CONFIGS
from portopt.utils.logging import setup_logger
from loguru import logger


def main() -> None:
    setup_logger()
    cfg = OmegaConf.load(CONFIGS / "data.yaml")
    tickers = all_tickers()

    logger.info(f"downloading {len(tickers)} tickers: {tickers}")
    logger.info(f"date range: {cfg.date_range.start} -> {cfg.date_range.end}")

    prices = download_prices(
        tickers,
        start=cfg.date_range.start,
        end=cfg.date_range.end,
        auto_adjust=cfg.download.auto_adjust,
        retries=cfg.download.retries,
        retry_sleep_s=cfg.download.retry_sleep_s,
    )

    logger.info(f"downloaded shape={prices.shape}, NaN ratio={prices.isna().mean().mean():.3f}")
    cache_raw(prices, name="prices_raw")


if __name__ == "__main__":
    main()
