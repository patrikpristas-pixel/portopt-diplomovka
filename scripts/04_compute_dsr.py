"""Compute Deflated Sharpe Ratio (DSR) for the winning trial of each portfolio.

Bailey & Lopez de Prado (2014).

DSR corrects the Probabilistic Sharpe Ratio for selection bias from running N
Optuna trials. The reference Sharpe is inflated by the expected maximum of N
draws under H0 (no skill), estimated from the spread of the trials' search
Sharpe ratios.

The winning trial per portfolio is the one with the highest search-period final
NAV (the selection criterion used throughout the thesis). DSR is reported
separately for the search period and the TRUE OOS holdout.

Deterministic; reads stored per-trial returns. No retraining.

Usage:
    python scripts/04_compute_dsr.py

Output (reports/dsr_results/):
    dsr_summary.csv / .json
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portopt.evaluation.statistics import deflated_sharpe_ratio  # noqa: E402

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


def main() -> None:
    outdir = Path("reports/dsr_results")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for portfolio in PORTFOLIOS:
        scen = _scenario_dir(portfolio)
        trials = pd.read_parquet(scen / "trials.parquet")

        # Winning trial = highest search-period final NAV (thesis selection).
        winner = trials.loc[trials["final_nav_eur"].idxmax()]
        winner_id = int(winner["trial"])
        holdout_start = pd.Timestamp(winner["holdout_start"])

        # Sharpe history = annualized search Sharpe of EVERY trial (selection set).
        sharpe_history = trials["sharpe"].dropna().astype(float).tolist()

        returns = pd.read_parquet(
            scen / "trial_data" / f"trial_{winner_id:04d}" / "returns.parquet"
        )["returns"]
        search_ret = returns[returns.index < holdout_start].dropna()
        holdout_ret = returns[returns.index >= holdout_start].dropna()

        dsr_search = deflated_sharpe_ratio(search_ret, sharpe_history)["dsr"]
        dsr_holdout = deflated_sharpe_ratio(holdout_ret, sharpe_history)["dsr"]

        rows.append(
            {
                "portfolio": LABELS[portfolio],
                "winning_trial": winner_id,
                "n_trials": len(sharpe_history),
                "DSR_search_pct": round(dsr_search * 100, 1),
                "DSR_holdout_pct": round(dsr_holdout * 100, 1),
            }
        )
        print(
            f"{LABELS[portfolio]:14s}  trial #{winner_id:<4d}  "
            f"DSR search={dsr_search * 100:5.1f}%  holdout={dsr_holdout * 100:5.1f}%"
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "dsr_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_json(
        outdir / "dsr_summary.json", orient="records", force_ascii=False, indent=2
    )
    print(f"\nUlozene do {outdir}/")


if __name__ == "__main__":
    main()
