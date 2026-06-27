"""Inferential statistics for portfolio backtest evaluation.

Implements the standard tools for honest evaluation of trading strategies:

- **Stationary bootstrap CI for Sharpe** (Politis & Romano, 1994) — handles
  autocorrelation in daily returns; gives a 95% interval around the point
  estimate so "Sharpe = 1.45" can be reported as "1.45 [0.81, 2.10]".
- **Probabilistic Sharpe Ratio (PSR)** — Bailey & López de Prado (2012).
  Probability that the TRUE Sharpe exceeds a reference (default 0).
- **Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado (2014).
  PSR corrected for selection bias from running N independent trials.
  Directly addresses the "I ran 500 Optuna trials" data-dredging problem.
- **Diebold-Mariano test** (1995) — H0: two strategies have equal mean return.
  Newey-West HAC variance with lag h-1.
- **Probability of Backtest Overfit (PBO)** — Bailey, Borwein, López de Prado,
  Zhu (2015). Via Combinatorially Symmetric Cross-Validation (CSCV).
  Fraction of times the in-sample winner underperforms OOS median.

All public functions accept returns (pd.Series or np.ndarray) and return
ANNUALIZED quantities where applicable. Daily-frequency Sharpe is used
internally to keep formulas numerically stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

DAYS_PER_YEAR = 252


# ----- Low-level helpers -----


def _daily_sharpe(r: np.ndarray) -> float:
    """Daily-frequency Sharpe (mean / std). No annualization, no risk-free rate."""
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float(r.mean() / sd)


def _annualize(daily_sr: float) -> float:
    return daily_sr * np.sqrt(DAYS_PER_YEAR)


def _deannualize(ann_sr: float) -> float:
    return ann_sr / np.sqrt(DAYS_PER_YEAR)


def _return_moments(r: np.ndarray) -> tuple[float, float]:
    """(skewness, Pearson's kurtosis). Kurtosis=3 under normality.

    Falls back to (0, 3) — i.e., assume normal — when the sample has effectively
    zero variance. Without this guard, scipy.stats.skew/kurtosis return NaN on
    near-constant data (catastrophic cancellation), and downstream PSR breaks.
    """
    if len(r) < 4:
        return 0.0, 3.0
    sd = float(np.std(r))
    if sd < 1e-12:  # effectively constant → moments undefined
        return 0.0, 3.0
    sk = float(stats.skew(r, bias=False))
    kt = float(stats.kurtosis(r, bias=False) + 3.0)  # convert Fisher → Pearson
    if not np.isfinite(sk):
        sk = 0.0
    if not np.isfinite(kt):
        kt = 3.0
    return sk, kt


# ----- Stationary bootstrap (Politis & Romano, 1994) -----


def _stationary_bootstrap_indices(
    n: int, expected_block_len: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate indices of length n via stationary bootstrap.

    Block lengths follow Geometric(1/expected_block_len). Block starts are
    uniform over [0, n). Indices wrap around (circular).
    """
    idx = np.empty(n, dtype=np.int64)
    p_new = 1.0 / max(expected_block_len, 1.0)
    cur = int(rng.integers(0, n))
    for i in range(n):
        if i > 0 and rng.random() < p_new:
            cur = int(rng.integers(0, n))
        idx[i] = cur
        cur = (cur + 1) % n
    return idx


# ----- Public API -----


@dataclass
class SharpeCI:
    sharpe_ann: float
    ci_low_ann: float
    ci_high_ann: float
    n_observations: int
    n_bootstrap: int
    confidence: float


def sharpe_bootstrap_ci(
    returns: pd.Series | np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    block_size: float = 5.0,
    rng_seed: int = 42,
) -> SharpeCI:
    """95% stationary-bootstrap CI for the ANNUALIZED Sharpe ratio.

    block_size ~ Geometric mean block length (default 5 = ~weekly).
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[~np.isnan(r)]
    n = len(r)
    pt = _annualize(_daily_sharpe(r))
    if n < 30:
        return SharpeCI(pt, pt, pt, n, 0, confidence)
    rng = np.random.default_rng(rng_seed)
    sharpes = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n, block_size, rng)
        sharpes[b] = _annualize(_daily_sharpe(r[idx]))
    alpha = 1.0 - confidence
    return SharpeCI(
        sharpe_ann=pt,
        ci_low_ann=float(np.quantile(sharpes, alpha / 2)),
        ci_high_ann=float(np.quantile(sharpes, 1.0 - alpha / 2)),
        n_observations=n,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    sharpe_ref_ann: float = 0.0,
) -> float:
    """PSR — probability that the TRUE annualized Sharpe exceeds `sharpe_ref_ann`.

    Bailey & López de Prado (2012). Accounts for skewness and excess kurtosis
    of the return distribution (heavy-tailed returns reduce confidence).
    Returns a probability in [0, 1].
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 4:
        return 0.5
    sr_d = _daily_sharpe(r)
    ref_d = _deannualize(sharpe_ref_ann)
    skew, kurt = _return_moments(r)
    # B-LdP variance term: 1 - γ3·SR + (γ4-1)/4 · SR^2
    # Under normality (γ3=0, γ4=3) → 1 + 0.5·SR^2, matches Mertens (2002).
    denom_sq = 1.0 - skew * sr_d + (kurt - 1.0) / 4.0 * sr_d ** 2
    if not np.isfinite(denom_sq) or denom_sq <= 0:
        # Degenerate: collapse to indicator on sign of sharpe gap.
        return float(sr_d > ref_d)
    z = (sr_d - ref_d) * np.sqrt(max(n - 1, 1)) / np.sqrt(denom_sq)
    if not np.isfinite(z):
        return float(sr_d > ref_d)
    return float(stats.norm.cdf(z))


def _expected_max_normal(n_trials: int) -> float:
    """E[max(Z_1, ..., Z_n)] for iid standard normal. Bailey-LdP (2014) approx."""
    if n_trials < 2:
        return 0.0
    euler = 0.5772156649
    return (1.0 - euler) * stats.norm.ppf(1.0 - 1.0 / n_trials) + euler * stats.norm.ppf(
        1.0 - 1.0 / (n_trials * np.e)
    )


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    sharpe_history_ann: np.ndarray | list[float] | pd.Series,
) -> dict:
    """DSR — PSR adjusted for selection bias from running N trials.

    Bailey & López de Prado (2014).  Directly addresses the multiple-testing
    problem in hyperparameter search: after N independent draws, the expected
    MAXIMUM Sharpe under H0 (no skill) is sd(SR_history) * E[max(Z_1..Z_N)].
    DSR is the PSR with this inflated reference.

    Args:
      returns: daily returns of the candidate (typically the BEST trial)
      sharpe_history_ann: annualized Sharpe of EVERY trial (including this one)

    Returns: dict with dsr, sharpe_ref_ann, n_trials, sharpe_history_std
    """
    sh = np.asarray(sharpe_history_ann, dtype=np.float64).ravel()
    sh = sh[~np.isnan(sh)]
    n_trials = int(len(sh))
    if n_trials < 2:
        psr = probabilistic_sharpe_ratio(returns, 0.0)
        return {
            "dsr": psr,
            "sharpe_ref_ann": 0.0,
            "n_trials": n_trials,
            "sharpe_history_std": 0.0,
        }
    sd = float(sh.std(ddof=1))
    e_max = _expected_max_normal(n_trials)
    sharpe_ref_ann = sd * e_max
    dsr = probabilistic_sharpe_ratio(returns, sharpe_ref_ann=sharpe_ref_ann)
    return {
        "dsr": float(dsr),
        "sharpe_ref_ann": float(sharpe_ref_ann),
        "n_trials": n_trials,
        "sharpe_history_std": sd,
    }


def diebold_mariano(
    returns_a: pd.Series | np.ndarray,
    returns_b: pd.Series | np.ndarray,
    h: int = 1,
) -> dict:
    """Diebold-Mariano test: H0 mean(r_a - r_b) = 0, two-sided.

    Uses Newey-West HAC variance with truncation lag h-1 (default h=1 ⇒ no
    autocorrelation correction beyond contemporaneous). For daily portfolio
    returns with weekly cycles, h=5 is reasonable.

    NOTE: this test compares MEAN return, not Sharpe. To compare risk-adjusted
    performance use the PSR/DSR framework instead.
    """
    ra = np.asarray(returns_a, dtype=np.float64).ravel()
    rb = np.asarray(returns_b, dtype=np.float64).ravel()
    n = int(min(len(ra), len(rb)))
    if n < 30:
        return {
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "mean_diff_ann": float("nan"),
            "n": n,
        }
    d = ra[:n] - rb[:n]
    mean_d = float(d.mean())
    lag = max(0, int(h) - 1)
    s = float(np.var(d, ddof=1))
    for k in range(1, lag + 1):
        if k >= n:
            break
        gk = float(np.cov(d[:-k], d[k:], ddof=1)[0, 1])
        weight = 1.0 - k / (lag + 1)
        s += 2.0 * weight * gk
    var_d_mean = s / n
    if var_d_mean <= 0:
        return {
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "mean_diff_ann": float(mean_d * DAYS_PER_YEAR),
            "n": n,
        }
    dm = mean_d / np.sqrt(var_d_mean)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return {
        "dm_stat": float(dm),
        "p_value": float(p),
        "mean_diff_ann": float(mean_d * DAYS_PER_YEAR),
        "n": n,
    }


def probability_of_backtest_overfit(
    trials_returns: pd.DataFrame,
    n_splits: int = 14,
) -> dict:
    """PBO via Combinatorially Symmetric Cross-Validation (CSCV).

    Bailey, Borwein, López de Prado, Zhu (2015).

    Procedure:
      1. Cut the time-axis into S equal chunks (S even).
      2. For each (S choose S/2) split into in-sample/OOS halves:
         - find best trial by IS Sharpe
         - compute its OOS rank percentile
         - logit-transform: log(rank / (1 - rank))
      3. PBO = fraction of splits where logit < 0 (i.e., OOS underperformed
         the OOS median).

    A PBO close to 0.5 means the IS-best is no better than random OOS — clear
    overfitting. PBO << 0.5 means consistent OOS outperformance.

    Args:
      trials_returns: DataFrame (rows = aligned dates, cols = one per trial)
      n_splits: number of partitions (must be even; paper recommends 12-16)

    Returns: {"pbo", "n_combinations", "logits"}
    """
    if n_splits % 2 != 0:
        n_splits += 1
    R = trials_returns.dropna(how="all").copy()
    T, N = R.shape
    if T < n_splits * 4 or N < 2:
        return {"pbo": float("nan"), "n_combinations": 0, "logits": []}
    split_len = T // n_splits
    chunks = [R.iloc[i * split_len : (i + 1) * split_len] for i in range(n_splits)]
    half = n_splits // 2
    logits: list[float] = []
    losses = 0
    total = 0
    for is_idx in combinations(range(n_splits), half):
        oos_idx = [i for i in range(n_splits) if i not in is_idx]
        is_R = pd.concat([chunks[i] for i in is_idx])
        oos_R = pd.concat([chunks[i] for i in oos_idx])
        is_sd = is_R.std(ddof=1).replace(0.0, np.nan)
        oos_sd = oos_R.std(ddof=1).replace(0.0, np.nan)
        is_sh = (is_R.mean() / is_sd).fillna(0.0)
        oos_sh = (oos_R.mean() / oos_sd).fillna(0.0)
        best_is = int(is_sh.values.argmax())
        ranks = oos_sh.rank(method="average")
        # rank percentile of best_is (in [0, 1])
        rank_pct = float((ranks.iloc[best_is] - 1) / max(N - 1, 1))
        # Avoid logit at 0/1
        eps = 0.5 / N
        rank_pct = float(np.clip(rank_pct, eps, 1.0 - eps))
        logit = float(np.log(rank_pct / (1.0 - rank_pct)))
        logits.append(logit)
        if logit < 0.0:
            losses += 1
        total += 1
    pbo = losses / total if total > 0 else float("nan")
    return {"pbo": float(pbo), "n_combinations": int(total), "logits": logits}


def summarize_trial_stats(
    returns: pd.Series,
    sharpe_history_ann: np.ndarray | list[float] | None = None,
    benchmark_returns: pd.Series | None = None,
    n_bootstrap: int = 1000,
) -> dict:
    """Compute ALL stats for one trial in a flat dict.

    Goes into trials.parquet as new columns. Cheap (~50ms for 1000 bootstrap
    resamples on a year of daily data).
    """
    out: dict = {}
    ci = sharpe_bootstrap_ci(returns, n_bootstrap=n_bootstrap)
    out["sharpe_ci_low"] = float(ci.ci_low_ann)
    out["sharpe_ci_high"] = float(ci.ci_high_ann)
    out["psr"] = float(probabilistic_sharpe_ratio(returns, 0.0))
    out["psr_vs_sharpe_1"] = float(probabilistic_sharpe_ratio(returns, 1.0))
    # Return distribution properties
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[~np.isnan(r)]
    skew, kurt = _return_moments(r)
    out["return_skewness"] = float(skew)
    out["return_kurtosis"] = float(kurt)
    if sharpe_history_ann is not None and len(list(sharpe_history_ann)) >= 2:
        d = deflated_sharpe_ratio(returns, sharpe_history_ann)
        out["dsr"] = float(d["dsr"])
        out["dsr_sharpe_ref"] = float(d["sharpe_ref_ann"])
    if benchmark_returns is not None:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) >= 30:
            dm = diebold_mariano(aligned.iloc[:, 0], aligned.iloc[:, 1], h=5)
            out["dm_stat_vs_bench"] = float(dm["dm_stat"])
            out["dm_pvalue_vs_bench"] = float(dm["p_value"])
            out["dm_mean_diff_ann"] = float(dm["mean_diff_ann"])
    return out
