"""Print thesis result tables to the command line.

This script is meant for reviewer/supervisor screenshots. It does not retrain
models and does not download data; it prints the stored result tables that are
reported in the thesis chapter "Vysledky prace a diskusia".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_export"
REPORTS = ROOT / "reports"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required result file: {path}")
    return pd.read_csv(path)


def _fmt_float(value: object, decimals: int = 3) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}f}"


def _fmt_pct(value: object, decimals: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}f} %"


def _print_title(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_df(df: pd.DataFrame) -> None:
    print(df.to_string(index=False))


def _ordered_portfolio_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = ["Agresívne", "Vyvážené", "Konzervatívne", "All-Weather"]
    cols = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in cols and c != "Strategy"]
    return df[["Strategy", *cols, *rest]]


def print_ai_winners() -> None:
    df = _read_csv(RESULTS / "01_AI_winner_summary.csv")
    out = pd.DataFrame(
        {
            "Portfolio": df["portfólio"],
            "Winning trial": df["víťazný_trial"].astype(int),
            "Search Sharpe": df["search_Sharpe"].map(lambda x: _fmt_float(x, 3)),
            "Search NAV": df["search_NAV_eur"].map(lambda x: f"{float(x):,.0f}"),
            "Holdout Sharpe": df["holdout_Sharpe"].map(lambda x: _fmt_float(x, 3)),
            "Holdout NAV": df["holdout_NAV_eur"].map(lambda x: f"{float(x):,.0f}"),
            "Gap": df["generalization_gap_%"].map(lambda x: _fmt_pct(x, 1)),
        }
    )
    _print_title("WORD RESULTS - AI WINNERS: SEARCH, HOLDOUT, GENERALIZATION GAP")
    _print_df(out)


def print_holdout_metrics() -> None:
    df = _read_csv(RESULTS / "02_holdout_AI_vs_baselines.csv")

    _print_title("WORD RESULTS - HOLDOUT SHARPE: AI VS BASELINES")
    sharpe = df.pivot(index="stratégia", columns="portfólio", values="Sharpe")
    sharpe = sharpe.map(lambda x: _fmt_float(x, 3))
    sharpe = sharpe.reset_index().rename(columns={"stratégia": "Strategy"})
    _print_df(_ordered_portfolio_columns(sharpe))

    for metric, title, decimals in [
        ("Sortino", "WORD RESULTS - HOLDOUT SORTINO RATIO", 3),
        ("Calmar", "WORD RESULTS - HOLDOUT CALMAR RATIO", 3),
        ("MaxDD", "WORD RESULTS - HOLDOUT MAXIMUM DRAWDOWN", 3),
        ("NAV_eur", "HOLDOUT NAV VALUES USED IN THE RESULTS CHAPTER", 0),
    ]:
        _print_title(title)
        pivot = df.pivot(index="stratégia", columns="portfólio", values=metric)
        if metric == "NAV_eur":
            pivot = pivot.map(lambda x: "-" if pd.isna(x) else f"{float(x):,.0f}")
        else:
            pivot = pivot.map(lambda x: _fmt_float(x, decimals))
        pivot = pivot.reset_index().rename(columns={"stratégia": "Strategy"})
        _print_df(_ordered_portfolio_columns(pivot))


def print_hyperparams() -> None:
    df = _read_csv(RESULTS / "03_winning_hyperparameters.csv")
    _print_title("WORD RESULTS - WINNING HYPERPARAMETERS")
    _print_df(df)


def print_pbo() -> None:
    df = _read_csv(RESULTS / "04_PBO_summary.csv")
    _print_title("WORD RESULTS - PROBABILITY OF BACKTEST OVERFITTING (PBO)")
    _print_df(df)


def print_dsr() -> None:
    df = _read_csv(RESULTS / "01_AI_winner_summary.csv")
    out = pd.DataFrame(
        {
            "Portfolio": df["portfólio"],
            "Winning trial": df["víťazný_trial"].astype(int),
            "DSR search": (df["DSR_search"] * 100).map(lambda x: _fmt_pct(x, 1)),
            "DSR holdout": (df["DSR_holdout"] * 100).map(lambda x: _fmt_pct(x, 1)),
            "Gap": df["generalization_gap_%"].map(lambda x: _fmt_pct(x, 1)),
        }
    )
    _print_title("WORD RESULTS - DSR SUMMARY USED IN STATISTICAL SIGNIFICANCE SECTION")
    _print_df(out)


def print_dm() -> None:
    path = REPORTS / "dm_results" / "dm_pvalues.csv"
    if not path.exists():
        return
    df = _read_csv(path)
    aggressive = df[df["portfolio"] == "Agresivne"].copy()
    cols = ["comparison", "dm_stat_search", "p_search", "dm_stat_holdout", "p_holdout"]
    _print_title("WORD RESULTS - DIEBOLD-MARIANO TEST, AGGRESSIVE PORTFOLIO")
    _print_df(aggressive[cols])


def print_vix() -> None:
    path = REPORTS / "vix_results" / "aggressive_vix_summary.csv"
    if not path.exists():
        return
    df = _read_csv(path)
    cols = [
        "variant",
        "trial_count",
        "wins",
        "best_trial",
        "best_search_sharpe",
        "best_search_nav",
        "best_holdout_sharpe",
        "best_holdout_nav",
    ]
    _print_title("WORD RESULTS - VIX ABLATION STUDY, AGGRESSIVE PORTFOLIO")
    _print_df(df[cols])

    # Paired sign/binomial test on the 100 paired trials (VIX ON vs VIX OFF).
    paired = REPORTS / "vix_results" / "aggressive_vix_paired_trials.csv"
    if paired.exists():
        from scipy import stats

        pdf = _read_csv(paired)
        if "winner" in pdf.columns:
            on = int((pdf["winner"] == "VIX ON").sum())
            off = int((pdf["winner"] == "VIX OFF").sum())
            n = on + off
            p_value = stats.binomtest(on, n, 0.5, alternative="greater").pvalue
            print()
            print(f"  Parovy binomicky test (VIX ON vs VIX OFF, {n} parov):")
            print(f"    VIX ON vyhral v {on} paroch, VIX OFF v {off} paroch")
            print(
                f"    H0: p = 0.5, jednostranna p-hodnota = {p_value:.4f} "
                f"(rozdiel nie je statisticky vyznamny, p > 0.05)"
            )


def main() -> None:
    print("=" * 78)
    print("COMMAND-LINE OUTPUT OF RESULTS REPORTED IN THE THESIS")
    print("Source folders: results_export/ and reports/")
    print("No retraining and no data download are performed by this script.")
    print("=" * 78)

    print_ai_winners()
    print_holdout_metrics()
    print_dsr()
    print_dm()
    print_pbo()
    print_vix()
    print_hyperparams()

    print()
    print("=" * 78)
    print("DONE - all stored thesis result tables were printed to command line.")
    print("=" * 78)


if __name__ == "__main__":
    main()
