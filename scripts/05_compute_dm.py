"""Compute Diebold-Mariano test p-values: AI winner vs each baseline.

Diebold & Mariano (1995). Tests H0: the two strategies have equal mean daily
return. The loss differential is the daily return difference; the standard
error uses a Newey-West HAC estimate with truncation lag h-1 (h=5, weekly).

Reported separately for the search period and the TRUE OOS holdout, for the
winning trial (highest search-period final NAV) of each portfolio against all
five reference strategies.

Deterministic; reads stored per-trial and baseline returns. No retraining.

Usage:
    python scripts/05_compute_dm.py

Output (reports/dm_results/):
    dm_pvalues.csv / .json
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portopt.evaluation.statistics import diebold_mariano  # noqa: E402

HOLDOUT_START = pd.Timestamp("2020-01-01")
HAC_H = 5
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


def _pvalue(result: dict) -> float:
    for key in ("p_value", "p", "pvalue"):
        if key in result:
            return float(result[key])
    return float("nan")


def main() -> None:
    outdir = Path("reports/dm_results")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for portfolio in PORTFOLIOS:
        scen = _scenario_dir(portfolio)
        trials = pd.read_parquet(scen / "trials.parquet")
        winner_id = int(trials.loc[trials["final_nav_eur"].idxmax(), "trial"])

        ai = pd.read_parquet(
            scen / "trial_data" / f"trial_{winner_id:04d}" / "returns.parquet"
        )["returns"]
        baselines = pd.read_parquet(scen / "baseline_returns.parquet")

        for name in baselines.columns:
            b = baselines[name]
            common = ai.index.intersection(b.index)
            ai_c, b_c = ai.loc[common], b.loc[common]

            ai_s, b_s = ai_c[ai_c.index < HOLDOUT_START], b_c[b_c.index < HOLDOUT_START]
            ai_h, b_h = ai_c[ai_c.index >= HOLDOUT_START], b_c[b_c.index >= HOLDOUT_START]

            p_search = _pvalue(diebold_mariano(ai_s, b_s, h=HAC_H))
            p_holdout = _pvalue(diebold_mariano(ai_h, b_h, h=HAC_H))

            rows.append(
                {
                    "portfolio": LABELS[portfolio],
                    "comparison": f"AI vs {name}",
                    "p_search": round(p_search, 3),
                    "p_holdout": round(p_holdout, 3),
                    "search_significant_0.05": bool(p_search < 0.05),
                    "holdout_significant_0.05": bool(p_holdout < 0.05),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "dm_pvalues.csv", index=False, encoding="utf-8-sig")
    summary.to_json(
        outdir / "dm_pvalues.json", orient="records", force_ascii=False, indent=2
    )
    # Print the Aggressive block (reported in the thesis table)
    agg = summary[summary["portfolio"] == "Agresivne"]
    print(agg.to_string(index=False))
    print(f"\nUlozene do {outdir}/")


if __name__ == "__main__":
    main()
