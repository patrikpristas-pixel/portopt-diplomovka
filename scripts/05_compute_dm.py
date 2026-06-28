"""Compute Diebold-Mariano test: AI winner vs each baseline.

Diebold & Mariano (1995). Tests H0: the two strategies have equal mean daily
return. The loss differential is the daily return difference; the standard
error uses a Newey-West HAC estimate with truncation lag h-1 (h=5, weekly).

The test is reported as an exploratory pairwise comparison without correction
for multiple testing: AI is compared against five reference strategies, so the
individual p-values should be interpreted with caution. Since all holdout
p-values are far above 0.05, a family-wise correction would not change the
conclusions.

For each comparison the script reports the full statistics: the DM test
statistic, the annualized mean return difference, the number of aligned
observations, and the p-value, separately for the search period and the TRUE
OOS holdout. The winning trial per portfolio is the one with the highest
search-period final NAV.

Deterministic; reads stored per-trial and baseline returns. No retraining.

Usage:
    python scripts/05_compute_dm.py

Output (reports/dm_results/):
    dm_pvalues.csv / .json     (full DM statistics)
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

            s = diebold_mariano(ai_s, b_s, h=HAC_H)
            h = diebold_mariano(ai_h, b_h, h=HAC_H)

            rows.append(
                {
                    "portfolio": LABELS[portfolio],
                    "comparison": f"AI vs {name}",
                    "n_search": s["n"],
                    "mean_diff_ann_search": round(s["mean_diff_ann"], 4),
                    "dm_stat_search": round(s["dm_stat"], 3),
                    "p_search": round(s["p_value"], 3),
                    "n_holdout": h["n"],
                    "mean_diff_ann_holdout": round(h["mean_diff_ann"], 4),
                    "dm_stat_holdout": round(h["dm_stat"], 3),
                    "p_holdout": round(h["p_value"], 3),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "dm_pvalues.csv", index=False, encoding="utf-8-sig")
    summary.to_json(
        outdir / "dm_pvalues.json", orient="records", force_ascii=False, indent=2
    )
    agg = summary[summary["portfolio"] == "Agresivne"]
    print(
        agg[
            ["comparison", "dm_stat_search", "p_search", "dm_stat_holdout", "p_holdout"]
        ].to_string(index=False)
    )
    print(f"\nUlozene do {outdir}/")


if __name__ == "__main__":
    main()
