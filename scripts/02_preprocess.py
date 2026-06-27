"""Align raw prices, compute log returns, save processed panel + coverage summary."""

from __future__ import annotations

from omegaconf import OmegaConf

from portopt.data.preprocess import (
    align_panel,
    compute_log_returns,
    coverage_summary,
    forward_fill_limited,
)
from portopt.utils.io import CONFIGS, DATA_PROCESSED, DATA_RAW, ensure_dir, read_parquet, write_parquet
from portopt.utils.logging import setup_logger
from loguru import logger


def main() -> None:
    setup_logger()
    cfg = OmegaConf.load(CONFIGS / "data.yaml")

    raw_path = DATA_RAW / "prices_raw.parquet"
    if not raw_path.exists():
        raise SystemExit(f"raw prices not found: {raw_path}\nrun scripts/01_download_data.py first")

    prices = read_parquet(raw_path)
    logger.info(f"loaded raw prices: shape={prices.shape}")

    aligned = align_panel(prices)
    logger.info(f"aligned: shape={aligned.shape}, start={aligned.index[0]}, end={aligned.index[-1]}")

    filled = forward_fill_limited(aligned, max_days=cfg.preprocess.forward_fill_max_days)
    logger.info(f"forward-filled (max {cfg.preprocess.forward_fill_max_days}d): NaN remaining={int(filled.isna().sum().sum())}")

    log_returns = compute_log_returns(filled)
    coverage = coverage_summary(filled)

    ensure_dir(DATA_PROCESSED)
    write_parquet(filled, DATA_PROCESSED / "prices.parquet")
    write_parquet(log_returns, DATA_PROCESSED / "log_returns.parquet")
    coverage.to_csv(DATA_PROCESSED / "coverage.csv")

    logger.info(f"wrote {DATA_PROCESSED}/prices.parquet")
    logger.info(f"wrote {DATA_PROCESSED}/log_returns.parquet")
    logger.info(f"wrote {DATA_PROCESSED}/coverage.csv")
    logger.info(f"\n{coverage.to_string()}")


if __name__ == "__main__":
    main()
