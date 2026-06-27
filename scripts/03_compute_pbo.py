"""Compute Probability of Backtest Overfitting (PBO) via CSCV.

Bailey, Borwein, Lopez de Prado, Zhu (2015).

Reproduces the PBO values reported in the thesis from the stored per-trial
daily returns (search period 2013-2019). Deterministic — no retraining.

Usage:
    python scripts/03_compute_pbo.py

Outputs (reports/pbo_results/):
    pbo_summary.csv / .json     PBO per portfolio
    <portfolio>_logits.csv      CSCV logits (lambda) for the histogram
    pbo_lambda_histograms.png   2x2 lambda distribution plot
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portopt.evaluation.statistics import probability_of_backtest_overfit  # noqa: E402

HOLDOUT_START = "2020-01-01"
N_SPLITS = 14  # paper recommends 12-16; C(14,7) = 3432 combinations
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


def _build_search_returns_matrix(portfolio: str) -> pd.DataFrame:
    """Matrix: rows = dates (search period), cols = one per trial."""
    tdir = _scenario_dir(portfolio) / "trial_data"
    cols: dict[str, pd.Series] = {}
    for trial in sorted(d for d in tdir.iterdir() if d.is_dir()):
        rp = trial / "returns.parquet"
        if not rp.exists():
            continue
        r = pd.read_parquet(rp)["returns"]
        cols[trial.name] = r[r.index < HOLDOUT_START]
    return pd.DataFrame(cols)


def main() -> None:
    outdir = Path("reports/pbo_results")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    logits_by_port: dict[str, np.ndarray] = {}
    for portfolio in PORTFOLIOS:
        mat = _build_search_returns_matrix(portfolio)
        res = probability_of_backtest_overfit(mat, n_splits=N_SPLITS)
        logits = np.asarray(res["logits"])
        logits_by_port[portfolio] = logits
        pd.DataFrame({"logit": logits}).to_csv(
            outdir / f"{portfolio}_logits.csv", index=False
        )
        rows.append(
            {
                "portfolio": LABELS[portfolio],
                "n_trials": mat.shape[1],
                "n_days_search": mat.shape[0],
                "PBO": round(res["pbo"], 4),
                "PBO_pct": round(res["pbo"] * 100, 1),
                "n_combinations": res["n_combinations"],
                "logit_median": round(float(np.median(logits)), 3),
            }
        )
        print(f"{LABELS[portfolio]:14s}  PBO = {res['pbo'] * 100:5.1f} %")

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "pbo_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_json(
        outdir / "pbo_summary.json", orient="records", force_ascii=False, indent=2
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, portfolio in zip(axes.flat, PORTFOLIOS):
            lg = logits_by_port[portfolio]
            ax.hist(lg, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
            ax.axvline(0, color="#C44E52", linestyle="--", linewidth=1.5)
            ax.set_title(f"{LABELS[portfolio]}  (PBO = {(lg < 0).mean() * 100:.1f} %)")
            ax.set_xlabel("logit lambda")
            ax.set_ylabel("pocet kombinacii")
        fig.suptitle(
            "PBO - distribucia logitov lambda (CSCV, C(14,7)=3432 kombinacii)"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(outdir / "pbo_lambda_histograms.png", dpi=150)
    except ImportError:
        print("matplotlib nie je nainstalovany - histogram preskoceny")

    print(f"\nUlozene do {outdir}/")


if __name__ == "__main__":
    main()
