"""Regenerate thesis figures and result tables from stored experiment outputs.

This script does not retrain the neural network. It reads already generated
Optuna trials, per-trial returns/NAV/weights and baseline outputs from
`portfolios/`, then regenerates the figures/tables used in the thesis chapter
"Vysledky prace a diskusia".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PORTFOLIOS = ["aggressive", "balanced", "conservative", "all_weather"]
LABELS = {
    "aggressive": "Agresivne",
    "balanced": "Vyvazene",
    "conservative": "Konzervativne",
    "all_weather": "All-Weather",
}
OUT_FIG = ROOT / "reports" / "figures"
OUT_EXPORT = ROOT / "results_export"


def _scenario_dir(portfolio: str) -> Path:
    base = ROOT / "portfolios" / portfolio / "scenarios"
    scenarios = [d for d in base.iterdir() if d.is_dir()]
    if not scenarios:
        raise SystemExit(f"No scenario directory found for portfolio={portfolio}")
    # The active thesis scenario contains trials.parquet.
    for scenario in scenarios:
        if (scenario / "trials.parquet").exists():
            return scenario
    raise SystemExit(f"No trials.parquet found for portfolio={portfolio}")


def _winner_trial(scenario: Path) -> tuple[int, pd.Series, pd.DataFrame]:
    trials = pd.read_parquet(scenario / "trials.parquet")
    usable = trials.dropna(subset=["final_nav_eur"])
    if usable.empty:
        raise SystemExit(f"No usable final_nav_eur values in {scenario / 'trials.parquet'}")
    row = usable.loc[usable["final_nav_eur"].idxmax()]
    return int(row["trial"]), row, trials


def _trial_dir(scenario: Path, trial_id: int) -> Path:
    path = scenario / "trial_data" / f"trial_{trial_id:04d}"
    if not path.exists():
        raise SystemExit(f"Missing trial data directory: {path}")
    return path


def _drawdown(nav: pd.Series) -> pd.Series:
    nav = nav.dropna()
    return nav / nav.cummax() - 1.0


def _save_convergence(portfolio: str, trials: pd.DataFrame) -> Path:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    df = trials.dropna(subset=["trial", "final_nav_eur"]).sort_values("trial")
    out = OUT_FIG / f"convergence_{portfolio}.png"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df["trial"], df["final_nav_eur"], color="#9aa7b1", alpha=0.45, linewidth=1, label="trial NAV")
    ax.plot(df["trial"], df["final_nav_eur"].cummax(), color="#0f766e", linewidth=2.2, label="best-so-far NAV")
    ax.set_title(f"Konvergencia Optuna TPE hladania - {LABELS[portfolio]}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Search NAV")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _save_nav(portfolio: str, scenario: Path, trial_id: int) -> Path:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    trial = _trial_dir(scenario, trial_id)
    ai = pd.read_parquet(trial / "nav.parquet").iloc[:, 0].rename("AI Softmax NN")
    baselines = pd.read_parquet(scenario / "baseline_nav.parquet")
    df = pd.concat([ai, baselines], axis=1).dropna(how="all")
    out = OUT_FIG / f"nav_{portfolio}.png"

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for col in df.columns:
        lw = 2.5 if col == "AI Softmax NN" else 1.4
        alpha = 1.0 if col == "AI Softmax NN" else 0.82
        ax.plot(df.index, df[col], linewidth=lw, alpha=alpha, label=col)
    ax.set_title(f"Vyvoj NAV AI modelu a referencnych strategii - {LABELS[portfolio]}")
    ax.set_xlabel("Datum")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _save_drawdown(portfolio: str, scenario: Path, trial_id: int) -> Path:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    trial = _trial_dir(scenario, trial_id)
    ai = pd.read_parquet(trial / "nav.parquet").iloc[:, 0].rename("AI Softmax NN")
    baselines = pd.read_parquet(scenario / "baseline_nav.parquet")
    nav = pd.concat([ai, baselines], axis=1).dropna(how="all")
    dd = nav.apply(_drawdown)
    out = OUT_FIG / f"drawdown_{portfolio}.png"

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for col in dd.columns:
        lw = 2.5 if col == "AI Softmax NN" else 1.4
        alpha = 1.0 if col == "AI Softmax NN" else 0.82
        ax.plot(dd.index, dd[col] * 100, linewidth=lw, alpha=alpha, label=col)
    ax.set_title(f"Drawdown AI modelu a referencnych strategii - {LABELS[portfolio]}")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _save_allocation(portfolio: str, scenario: Path, trial_id: int) -> Path:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    weights = pd.read_parquet(_trial_dir(scenario, trial_id) / "weights.parquet")
    out = OUT_FIG / f"allocation_{portfolio}.png"

    if "test_start" in weights.columns:
        x = pd.to_datetime(weights["test_start"])
        plot_df = weights.drop(columns=[c for c in ["window_idx", "test_start"] if c in weights.columns])
        plot_df.index = x
    else:
        plot_df = weights.copy()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(plot_df.index, [plot_df[c] * 100 for c in plot_df.columns], labels=plot_df.columns, alpha=0.9)
    ax.set_title(f"Alokacia vah AI modelu po walk-forward oknach - {LABELS[portfolio]}")
    ax.set_xlabel("Testovacie okno")
    ax.set_ylabel("Vaha (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _portfolio_outputs(portfolio: str) -> tuple[pd.Series, pd.DataFrame, list[Path]]:
    scenario = _scenario_dir(portfolio)
    trial_id, winner, trials = _winner_trial(scenario)
    paths = [
        _save_convergence(portfolio, trials),
        _save_nav(portfolio, scenario, trial_id),
        _save_drawdown(portfolio, scenario, trial_id),
        _save_allocation(portfolio, scenario, trial_id),
    ]
    holdout = pd.read_parquet(scenario / "baselines_holdout.parquet")
    return winner, holdout, paths


def _save_holdout_sharpe(holdouts: dict[str, pd.DataFrame]) -> Path:
    OUT_EXPORT.mkdir(parents=True, exist_ok=True)
    rows = []
    for portfolio, df in holdouts.items():
        for strategy, row in df.iterrows():
            rows.append(
                {
                    "portfolio": LABELS[portfolio],
                    "strategy": strategy,
                    "sharpe": row.get("sharpe"),
                    "sortino": row.get("sortino"),
                    "calmar": row.get("calmar"),
                    "max_drawdown": row.get("max_drawdown"),
                    "final_nav_eur": row.get("final_nav_eur"),
                }
            )
    out = OUT_EXPORT / "generated_holdout_strategy_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def _save_stat_significance(winners: dict[str, pd.Series]) -> Path:
    OUT_EXPORT.mkdir(parents=True, exist_ok=True)
    rows = []
    for portfolio, row in winners.items():
        search_sharpe = float(row.get("sharpe", float("nan")))
        holdout_sharpe = float(row.get("holdout_sharpe", float("nan")))
        gap = (holdout_sharpe / search_sharpe - 1.0) * 100 if search_sharpe else float("nan")
        rows.append(
            {
                "portfolio": LABELS[portfolio],
                "winning_trial": int(row["trial"]),
                "search_sharpe": search_sharpe,
                "search_ci_low": row.get("sharpe_ci_low"),
                "search_ci_high": row.get("sharpe_ci_high"),
                "holdout_sharpe": holdout_sharpe,
                "holdout_ci_low": row.get("holdout_sharpe_ci_low"),
                "holdout_ci_high": row.get("holdout_sharpe_ci_high"),
                "generalization_gap_pct": gap,
            }
        )
    out = OUT_EXPORT / "generated_statistical_significance_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    print("=" * 78)
    print("GENEROVANIE VYSTUPOV POUZITYCH V KAPITOLE VYSLEDKY")
    print("Zdroj: ulozene Optuna trialy, NAV, vahy a baseline subory v portfolios/")
    print("Poznamka: tento skript netrenuje 810 trialov nanovo.")
    print("=" * 78)

    winners: dict[str, pd.Series] = {}
    holdouts: dict[str, pd.DataFrame] = {}
    all_paths: list[Path] = []

    for portfolio in PORTFOLIOS:
        print(f"\n[{LABELS[portfolio]}] Generujem konvergenciu, NAV, drawdown a alokaciu...")
        winner, holdout, paths = _portfolio_outputs(portfolio)
        winners[portfolio] = winner
        holdouts[portfolio] = holdout
        all_paths.extend(paths)
        for path in paths:
            print(f"  ulozene: {path.relative_to(ROOT)}")

    print("\nGenerujem tabulku anualizovaneho Sharpeho pomeru a dalsich holdout metrik...")
    holdout_path = _save_holdout_sharpe(holdouts)
    print(f"  ulozene: {holdout_path.relative_to(ROOT)}")

    print("\nGenerujem tabulku statistickej vyznamnosti AI modelu...")
    stat_path = _save_stat_significance(winners)
    print(f"  ulozene: {stat_path.relative_to(ROOT)}")

    print("\nDalsie statisticke vystupy generuju samostatne reprodukcne skripty:")
    print("  python scripts\\03_compute_pbo.py  -> reports\\pbo_results\\ + histogramy CSCV")
    print("  python scripts\\04_compute_dsr.py  -> reports\\dsr_results\\")
    print("  python scripts\\05_compute_dm.py   -> reports\\dm_results\\")

    print("\n" + "=" * 78)
    print("HOTOVO - grafy a tabulky z ulozenych pokusov boli vygenerovane.")
    print("=" * 78)


if __name__ == "__main__":
    main()
