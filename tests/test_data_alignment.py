from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portopt.data.preprocess import (
    align_panel,
    compute_log_returns,
    coverage_summary,
    forward_fill_limited,
)


def _panel(values: dict[str, list[float | None]], start: str = "2020-01-01") -> pd.DataFrame:
    n = max(len(v) for v in values.values())
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(values, index=idx)


def test_log_returns_match_manual_calculation():
    prices = _panel({"A": [100.0, 110.0, 121.0]})
    lr = compute_log_returns(prices)
    assert pd.isna(lr["A"].iloc[0])
    np.testing.assert_allclose(lr["A"].iloc[1], np.log(110.0 / 100.0))
    np.testing.assert_allclose(lr["A"].iloc[2], np.log(121.0 / 110.0))


def test_log_returns_sum_back_to_log_price_difference():
    prices = _panel({"A": [100.0, 105.0, 102.0, 108.0]})
    lr = compute_log_returns(prices)
    assert np.isclose(lr["A"].iloc[1:].sum(), np.log(108.0 / 100.0))


def test_align_panel_trims_to_latest_first_date():
    prices = _panel(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [None, None, 30.0, 40.0, 50.0],
        }
    )
    aligned = align_panel(prices)
    assert aligned.index[0] == prices.index[2]
    assert len(aligned) == 3
    assert not aligned["B"].iloc[0:].isna().any()


def test_align_panel_raises_on_empty_column():
    prices = _panel({"A": [1.0, 2.0, 3.0], "B": [None, None, None]})
    with pytest.raises(ValueError):
        align_panel(prices)


def test_forward_fill_limited_respects_max_days():
    prices = _panel({"A": [100.0, None, None, None, None, 105.0]})
    filled = forward_fill_limited(prices, max_days=2)
    assert filled["A"].iloc[1] == 100.0
    assert filled["A"].iloc[2] == 100.0
    assert pd.isna(filled["A"].iloc[3])
    assert pd.isna(filled["A"].iloc[4])
    assert filled["A"].iloc[5] == 105.0


def test_coverage_summary_counts_correctly():
    prices = _panel(
        {
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [None, 20.0, 30.0, None],
        }
    )
    cov = coverage_summary(prices)
    assert cov.loc["A", "n_obs"] == 4
    assert cov.loc["A", "n_missing"] == 0
    assert cov.loc["B", "n_obs"] == 2
    assert cov.loc["B", "n_missing"] == 2
    assert cov.loc["B", "first_date"] == prices.index[1]
    assert cov.loc["B", "last_date"] == prices.index[2]
