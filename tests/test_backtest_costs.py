from __future__ import annotations

import numpy as np

from portopt.backtest.costs import apply_costs, turnover


def test_no_change_no_cost():
    w = np.array([0.5, 0.5])
    assert apply_costs(w, w, cost_bps=10.0) == 0.0
    assert turnover(w, w) == 0.0


def test_full_swap_costs():
    prev = np.array([1.0, 0.0])
    new = np.array([0.0, 1.0])
    # turnover = |0-1| + |1-0| = 2.0; at 10 bps → cost = 2.0 * 10/10000 = 0.002 = 20 bps
    assert turnover(prev, new) == 2.0
    assert np.isclose(apply_costs(prev, new, cost_bps=10.0), 0.0020)


def test_partial_rebalance():
    prev = np.array([0.5, 0.5])
    new = np.array([0.6, 0.4])
    # turnover = 0.1 + 0.1 = 0.2; at 10 bps → cost = 0.2 * 10/10000 = 0.00002 = 0.2 bps NAV
    assert np.isclose(turnover(prev, new), 0.2)
    assert np.isclose(apply_costs(prev, new, cost_bps=10.0), 0.0002)


def test_initial_allocation_from_cash():
    prev = np.array([0.0, 0.0, 0.0])
    new = np.array([1 / 3, 1 / 3, 1 / 3])
    # turnover = 1.0; at 10 bps → cost = 0.001 = 10 bps NAV
    assert np.isclose(turnover(prev, new), 1.0)
    assert np.isclose(apply_costs(prev, new, cost_bps=10.0), 0.001)
