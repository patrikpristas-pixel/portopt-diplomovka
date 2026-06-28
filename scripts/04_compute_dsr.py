"""Compute Deflated Sharpe Ratio (DSR) for the winning trial of each portfolio.

Bailey & Lopez de Prado (2014).

DSR corrects the Probabilistic Sharpe Ratio for selection bias from running N
Optuna trials. The reference Sharpe is inflated by the expected maximum of N
draws under H0 (no skill), estimated from the spread of the trials' search
Sharpe ratios.

This script reports the full step-by-step inputs to the DSR computation:
candidate Sharpe, skewness, kurtosis, standard deviation of the Sharpe history,
the inflated reference Sharpe, the number of trials and the number of days,
separately for the search period and the TRUE OOS holdout.

The winning trial per portfolio is the one with the highest search-period final
NAV (the selection criterion used throughout the thesis).

Deterministic; reads stored per-trial returns. No retraining.

Usage:
    python scripts/04_compute_dsr.py

Output (reports/dsr_results/):
    dsr_summary.csv / .json      (DSR per portfolio)
    dsr_stepwise.csv / .json     (all inputs to the DSR computation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portopt.evaluation.statistics import (  # noqa: E402
    _annualize,
    _daily_sharpe,
    _return_moments,
    deflated_sharpe_ratio,
)

PORTFOLIOS = ["aggressive", "balanced", "conservative", "all_weather"]
LABELS = {
    "aggressive": "Agresivne",
    "balanced": "Vyvazene",
    "conservative": "Konzervativne",
    "all_weather": "All-Weather",
}


def _scenario_dir(portfolio: str) -> Path:
    base = Path("portfolios") / portfolio / "scenarios"
    return next(d for d in base.iterdir() if d.is_dir())


def _stepwise(returns: pd.Series, sharpe_history: list[float]) -> dict:
    r = returns.values
    res = deflated_sharpe_ratio(returns, sharpe_history)
    skew, kurt = _return_moments(r)
    return {
        "n_days": int(len(r)),
        "sharpe_ann": round(_annualize(_daily_sharpe(r)), 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "sigma_sharpe_history": round(res["sharpe_history_std"], 4),
        "sharpe_ref_ann": round(res["sharpe_ref_ann"], 4),
        "n_trials": int(res["n_trials"]),
        "DSR_pct": round(res["dsr"] * 100, 1),
    }


def main() -> None:
    outdir = Path("reports/dsr_results")
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows, step_rows = [], []
    for portfolio in PORTFOLIOS:
        scen = _scenario_dir(portfolio)
        trials = pd.read_parquet(scen / "trials.parquet")
        winner = trials.loc[trials["final_nav_eur"].idxmax()]
        winner_id = int(winner["trial"])
        holdout_start = pd.Timestamp(winner["holdout_start"])
        sharpe_history = trials["sharpe"].dropna().astype(float).tolist()

        returns = pd.read_parquet(
            scen / "trial_data" / f"trial_{winner_id:04d}" / "returns.parquet"
        )["returns"]
        search_ret = returns[returns.index < holdout_start].dropna()
        holdout_ret = returns[returns.index >= holdout_start].dropna()

        s_search = _stepwise(search_ret, sharpe_history)
        s_holdout = _stepwise(holdout_ret, sharpe_history)

        summary_rows.append(
            {
                "portfolio": LABELS[portfolio],
                "winning_trial": winner_id,
                "n_trials": s_search["n_trials"],
                "DSR_search_pct": s_search["DSR_pct"],
                "DSR_holdout_pct": s_holdout["DSR_pct"],
            }
        )
        for regime, st in (("search", s_search), ("holdout", s_holdout)):
            step_rows.append({"portfolio": LABELS[portfolio], "regime": regime, **st})
        print(
            f"{LABELS[portfolio]:14s}  DSR search={s_search['DSR_pct']:5.1f}%  "
            f"holdout={s_holdout['DSR_pct']:5.1f}%"
        )

    pd.DataFrame(summary_rows).to_csv(
        outdir / "dsr_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary_rows).to_json(
        outdir / "dsr_summary.json", orient="records", force_ascii=False, indent=2
    )
    step = pd.DataFrame(step_rows)
    step.to_csv(outdir / "dsr_stepwise.csv", index=False, encoding="utf-8-sig")
    step.to_json(
        outdir / "dsr_stepwise.json", orient="records", force_ascii=False, indent=2
    )
    print(f"\nUlozene do {outdir}/ (dsr_summary + dsr_stepwise)")


if __name__ == "__main__":
    main()
