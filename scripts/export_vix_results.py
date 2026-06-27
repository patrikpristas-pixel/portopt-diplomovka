from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_SITE = ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

import pandas as pd


PORTFOLIO = "aggressive"
SCENARIOS = {
    "VIX ON": "03dba391",
    "VIX OFF": "891866cb",
}
METRIC_COLUMNS = [
    "trial",
    "objective",
    "won",
    "won_vs_all",
    "won_vs_benchmarks",
    "sharpe",
    "final_nav_eur",
    "max_drawdown",
    "holdout_sharpe",
    "holdout_final_nav_eur",
    "holdout_max_drawdown",
    "psr",
    "holdout_psr",
    "timestamp",
]


def _load_trials(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    trials = pd.read_parquet(path)
    if "trial" in trials.columns:
        trials = (
            trials.sort_values("trial")
            .drop_duplicates(subset="trial", keep="last")
            .reset_index(drop=True)
        )
    return trials


def _best_row(trials: pd.DataFrame) -> pd.Series | None:
    if len(trials) == 0 or "objective" not in trials.columns:
        return None
    usable = trials.dropna(subset=["objective"])
    if len(usable) == 0:
        return None
    return usable.loc[usable["objective"].idxmax()]


def _float_or_none(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def main() -> None:
    base = ROOT / "portfolios" / PORTFOLIO / "scenarios"
    out_dir = ROOT / "reports" / "vix_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for label, scenario_id in SCENARIOS.items():
        scenario_dir = base / scenario_id
        knobs = json.loads((scenario_dir / "knobs.json").read_text(encoding="utf-8"))
        trials = _load_trials(scenario_dir / "trials.parquet")
        frames[label] = trials

        export_cols = [c for c in METRIC_COLUMNS if c in trials.columns]
        if export_cols:
            export_name = f"{PORTFOLIO.lower()}_{label.lower().replace(' ', '_')}_trials.csv"
            trials[export_cols].to_csv(
                out_dir / export_name,
                index=False,
                encoding="utf-8-sig",
            )

        best = _best_row(trials)
        wins = int(trials["won"].fillna(False).astype(bool).sum()) if "won" in trials.columns else 0
        summary_rows.append(
            {
                "variant": label,
                "scenario_id": scenario_id,
                "use_vix": bool(knobs.get("use_vix", False)),
                "trial_count": int(len(trials)),
                "wins": wins,
                "best_trial": int(best["trial"]) if best is not None and "trial" in best else None,
                "best_objective": _float_or_none(best.get("objective")) if best is not None else None,
                "best_search_sharpe": _float_or_none(best.get("sharpe")) if best is not None else None,
                "best_search_nav": _float_or_none(best.get("final_nav_eur")) if best is not None else None,
                "best_holdout_sharpe": _float_or_none(best.get("holdout_sharpe")) if best is not None else None,
                "best_holdout_nav": _float_or_none(best.get("holdout_final_nav_eur")) if best is not None else None,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        out_dir / f"{PORTFOLIO.lower()}_vix_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_df = pd.DataFrame()
    on_df = frames.get("VIX ON", pd.DataFrame())
    off_df = frames.get("VIX OFF", pd.DataFrame())
    if len(on_df) and len(off_df) and "trial" in on_df.columns and "trial" in off_df.columns:
        on_idx = on_df.set_index("trial")
        off_idx = off_df.set_index("trial")
        common = sorted(set(on_idx.index).intersection(set(off_idx.index)))
        paired_rows = []
        for trial_id in common:
            on_row = on_idx.loc[trial_id]
            off_row = off_idx.loc[trial_id]
            on_hold = on_row.get("holdout_final_nav_eur", math.nan)
            off_hold = off_row.get("holdout_final_nav_eur", math.nan)
            on_search = on_row.get("final_nav_eur", math.nan)
            off_search = off_row.get("final_nav_eur", math.nan)

            if pd.notna(on_hold) and pd.notna(off_hold):
                delta = float(on_hold) - float(off_hold)
                basis = "holdout_final_nav_eur"
            elif pd.notna(on_search) and pd.notna(off_search):
                delta = float(on_search) - float(off_search)
                basis = "final_nav_eur"
            else:
                continue

            winner = "VIX ON" if delta > 0 else ("VIX OFF" if delta < 0 else "Tie")
            paired_rows.append(
                {
                    "trial": int(trial_id),
                    "comparison_basis": basis,
                    "vix_on_final_nav_eur": _float_or_none(on_search),
                    "vix_off_final_nav_eur": _float_or_none(off_search),
                    "vix_on_holdout_final_nav_eur": _float_or_none(on_hold),
                    "vix_off_holdout_final_nav_eur": _float_or_none(off_hold),
                    "delta_on_minus_off": delta,
                    "winner": winner,
                }
            )

        paired_df = pd.DataFrame(paired_rows)
        paired_df.to_csv(
            out_dir / f"{PORTFOLIO.lower()}_vix_paired_trials.csv",
            index=False,
            encoding="utf-8-sig",
        )

    lines: list[str] = []
    lines.append("# VIX vysledky pre diplomovu pracu")
    lines.append("")
    lines.append(f"Portfolio: **{PORTFOLIO}**")
    lines.append("")
    lines.append(
        "Tento export je ulozeny mimo aplikacie, aby VIX nemusel ostat v aktivnej logike a nepredlzoval dalsie testovanie."
    )
    lines.append("")
    lines.append("## Scenare")
    for row in summary_rows:
        lines.append(
            f"- {row['variant']}: scenar `{row['scenario_id']}`, pokusy {row['trial_count']}, vyhry {row['wins']}, najlepsi trial #{row['best_trial']}"
        )
    lines.append("")
    lines.append("## Najlepsi vysledok kazdej vetvy")
    for row in summary_rows:
        lines.append(f"### {row['variant']}")
        lines.append(
            f"- Search NAV: {row['best_search_nav']:,.0f} EUR"
            if row["best_search_nav"] is not None
            else "- Search NAV: n/a"
        )
        lines.append(
            f"- Search Sharpe: {row['best_search_sharpe']:.5f}"
            if row["best_search_sharpe"] is not None
            else "- Search Sharpe: n/a"
        )
        lines.append(
            f"- Holdout NAV: {row['best_holdout_nav']:,.0f} EUR"
            if row["best_holdout_nav"] is not None
            else "- Holdout NAV: n/a"
        )
        lines.append(
            f"- Holdout Sharpe: {row['best_holdout_sharpe']:.5f}"
            if row["best_holdout_sharpe"] is not None
            else "- Holdout Sharpe: n/a"
        )
    lines.append("")

    if len(summary_df) == 2 and summary_df["best_holdout_nav"].notna().all():
        on_nav = float(summary_df.loc[summary_df["variant"] == "VIX ON", "best_holdout_nav"].iloc[0])
        off_nav = float(summary_df.loc[summary_df["variant"] == "VIX OFF", "best_holdout_nav"].iloc[0])
        delta = on_nav - off_nav
        lines.append("## Rychly zaver")
        if delta > 0:
            lines.append(f"- VIX ON mal vyssi najlepsi holdout NAV o {delta:,.0f} EUR.")
        elif delta < 0:
            lines.append(f"- VIX OFF mal vyssi najlepsi holdout NAV o {abs(delta):,.0f} EUR.")
        else:
            lines.append("- Najlepsi holdout NAV bol rovnaky.")
        lines.append("")

    if len(paired_df):
        on_wins = int((paired_df["winner"] == "VIX ON").sum())
        off_wins = int((paired_df["winner"] == "VIX OFF").sum())
        ties = int((paired_df["winner"] == "Tie").sum())
        lines.append("## Parove porovnanie rovnakych trial cisel")
        lines.append(f"- VIX ON vyhral v {on_wins} paroch.")
        lines.append(f"- VIX OFF vyhral v {off_wins} paroch.")
        lines.append(f"- Remiza: {ties}.")
        lines.append("")

    lines.append("## Ulozene subory")
    lines.append(f"- `{PORTFOLIO.lower()}_vix_summary.csv`")
    lines.append(f"- `{PORTFOLIO.lower()}_vix_on_trials.csv`")
    lines.append(f"- `{PORTFOLIO.lower()}_vix_off_trials.csv`")
    if len(paired_df):
        lines.append(f"- `{PORTFOLIO.lower()}_vix_paired_trials.csv`")

    (out_dir / f"{PORTFOLIO.lower()}_vix_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(out_dir)


if __name__ == "__main__":
    main()
