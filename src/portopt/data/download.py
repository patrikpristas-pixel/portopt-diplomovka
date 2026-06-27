from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from loguru import logger

from portopt.utils.io import DATA_RAW, ensure_dir, write_parquet


def _yf_download(
    tickers: list[str],
    start: str,
    end: str,
    auto_adjust: bool,
    threads: bool,
) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        threads=threads,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError(f"yfinance did not return Close column; got {raw.columns}")
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return prices


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    auto_adjust: bool = True,
    retries: int = 3,
    retry_sleep_s: float = 2.0,
) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers via yfinance.

    Strategy: one batch call (fast); for any all-NaN columns, fall back to
    single-ticker downloads (yfinance is more reliable for single tickers
    and handles transient rate limits).

    Returns a DataFrame indexed by trading date, columns = tickers (input order).
    """
    prices = _yf_download(tickers, start, end, auto_adjust, threads=True)
    prices = prices.reindex(columns=tickers)

    failed = [t for t in tickers if prices[t].isna().all()]
    if failed:
        logger.warning(f"batch missed {len(failed)} ticker(s): {failed} — retrying individually")
        for t in failed:
            for attempt in range(1, retries + 1):
                try:
                    one = _yf_download([t], start, end, auto_adjust, threads=False)
                    if not one[t].isna().all():
                        prices[t] = one[t]
                        logger.info(f"  recovered {t} on attempt {attempt}")
                        break
                except Exception as e:
                    logger.warning(f"  {t} attempt {attempt}/{retries} raised: {e}")
                time.sleep(retry_sleep_s * attempt)
            else:
                logger.error(f"  {t}: gave up after {retries} attempts; column will be all NaN")

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "date"
    return prices


def cache_raw(prices: pd.DataFrame, name: str = "prices_raw") -> Path:
    out = DATA_RAW / f"{name}.parquet"
    ensure_dir(out.parent)
    write_parquet(prices, out)
    logger.info(f"saved raw prices: {out} (shape={prices.shape})")
    return out
