"""Portfolio = named asset universe + default knobs.

Each combination of (portfolio + runtime knobs) is a unique SCENARIO with its
own trials, baselines, checkpoint, and Optuna study. This keeps comparison
meaningful — trials with different settings live in different folders.

Walk-forward semantics:
  The (date_start, date_end) range in knobs defines the OVERALL TEST regime.
  Inside this regime we cut N walk-forward windows of `test_window_years` years.
  For each window, AI trains on data strictly before the window's test_start,
  freezes weights, and is evaluated buy-and-hold on the window. Baselines do
  the same. Across windows AI may retrain (with more recent data) but never
  retrains within a single window.

Layout:
  portfolios/<name>.yaml                       ← static config
  portfolios/<name>/control.json               ← per-portfolio (only 1 active search)
  portfolios/<name>/scenarios/<sid>/           ← per-scenario data
      trials.parquet                           ← per-trial aggregated row
      baselines.parquet                        ← per-strategy aggregated metrics
      baseline_returns.parquet                 ← daily returns concatenated across windows
      baseline_nav.parquet                     ← daily NAV (deposits applied) per strategy
      baselines_walkforward.parquet            ← per-window per-strategy metrics
      best_predictor.pt                        ← best AI per criterion (per-window CKPTs in trial dir)
      optuna.db
      knobs.json
      trial_data/<trial_id>/
          hyperparams.json
          walkforward.parquet                  ← per-window metrics for this trial
          weights.parquet                      ← AI weights at each window's test_start
          returns.parquet                      ← AI daily returns concatenated across windows
          nav.parquet                          ← AI daily NAV
          models/window_<i>.pt                 ← per-window NN checkpoint
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

PORTFOLIOS_DIR = Path("portfolios")

# Knobs that affect the comparison data. Different values → different scenario.
# Buy-and-hold is now the only mode (long-term investor, tax-efficient),
# so holding-period and rebalance-freq are no longer scenario-defining.
SCENARIO_KEYS = (
    "date_start",
    "date_end",
    "max_weight",
    "monthly_deposit_eur",
    "win_criterion",
    "test_window_months",  # walk-forward test window length, in MONTHS (4 = quarterly)
    "holdout_years",       # how many years at the END are TRUE out-of-sample
)

DEFAULT_TEST_WINDOW_MONTHS = 12  # annual retraining (1x per year) — tax-efficient holding (>1y avoids short-term capital gains)
DEFAULT_MIN_TRAIN_YEARS = 2
DEFAULT_HOLDOUT_YEARS = 3  # last N years locked away from Optuna (single-shot eval)


@dataclass
class WalkForwardWindow:
    idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def as_dict(self) -> dict:
        return {
            "window_idx": self.idx,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def make_walkforward_windows(
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    data_start: pd.Timestamp,
    test_window_months: int = DEFAULT_TEST_WINDOW_MONTHS,
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
    start_idx: int = 0,
) -> list["WalkForwardWindow"]:
    """Cut the overall (test_start, test_end) regime into N walk-forward windows.

    `test_window_months` is the length of each test window in months. Default 4
    (quarterly retraining). `start_idx` lets holdout windows continue numbering
    after the search windows.
    """
    windows: list[WalkForwardWindow] = []
    cur = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    data_start = pd.Timestamp(data_start)
    idx = start_idx
    while cur <= end:
        win_end = min(
            cur + pd.DateOffset(months=test_window_months) - pd.Timedelta(days=1),
            end,
        )
        train_end = cur - pd.Timedelta(days=1)
        if (train_end - data_start).days < int(min_train_years * 365):
            cur = win_end + pd.Timedelta(days=1)
            continue
        windows.append(
            WalkForwardWindow(
                idx=idx,
                train_start=data_start,
                train_end=train_end,
                test_start=cur,
                test_end=win_end,
            )
        )
        idx += 1
        cur = win_end + pd.Timedelta(days=1)
    return windows


def split_search_holdout(
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    data_start: pd.Timestamp,
    holdout_years: int = DEFAULT_HOLDOUT_YEARS,
    test_window_months: int = DEFAULT_TEST_WINDOW_MONTHS,
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
) -> tuple[list["WalkForwardWindow"], list["WalkForwardWindow"], pd.Timestamp]:
    """Build SEARCH and HOLDOUT walk-forward windows.

    Search windows:  [date_start, holdout_start)  — Optuna sees these
    Holdout windows: [holdout_start, date_end]    — evaluated but never used
                                                    for HP selection

    Returns: (search_windows, holdout_windows, holdout_start)

    If holdout_years <= 0, holdout_windows is empty and search covers all.
    """
    date_start = pd.Timestamp(date_start)
    date_end = pd.Timestamp(date_end)
    if holdout_years <= 0:
        search = make_walkforward_windows(
            date_start, date_end, data_start, test_window_months, min_train_years
        )
        return search, [], date_end + pd.Timedelta(days=1)
    holdout_start = date_end - pd.DateOffset(years=holdout_years) + pd.Timedelta(days=1)
    if holdout_start <= date_start:
        # Holdout swallows search → degenerate. Push holdout to be at least 1y after start.
        holdout_start = date_start + pd.DateOffset(years=1)
    search_end = holdout_start - pd.Timedelta(days=1)
    search = make_walkforward_windows(
        date_start, search_end, data_start, test_window_months, min_train_years
    )
    holdout = make_walkforward_windows(
        holdout_start,
        date_end,
        data_start,
        test_window_months,
        min_train_years,
        start_idx=len(search),
    )
    return search, holdout, holdout_start


def scenario_id(knobs: dict) -> str:
    canonical = {k: knobs.get(k) for k in SCENARIO_KEYS}
    # Backward compatibility: older scenarios included `use_vix=true` in the
    # scenario hash. Keep the same ids so existing experiments stay visible.
    canonical["use_vix"] = bool(knobs.get("use_vix", True))
    s = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:8]


@dataclass
class PortfolioDefaults:
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    max_weight: float
    monthly_deposit_eur: float
    min_holding_period_days: int
    rebalance_freq: str
    win_criterion: str  # "sharpe" | "total_return" | "beat_benchmarks"


@dataclass
class Portfolio:
    name: str
    display_name: str
    description: str
    tickers: list[str]
    defaults: PortfolioDefaults

    @classmethod
    def from_yaml(cls, path: Path) -> "Portfolio":
        cfg = OmegaConf.load(path)
        defs = cfg.defaults
        return cls(
            name=str(cfg.name),
            display_name=str(cfg.display_name),
            description=str(cfg.description),
            tickers=list(cfg.tickers),
            defaults=PortfolioDefaults(
                date_start=pd.Timestamp(str(defs.date_range.start)),
                date_end=pd.Timestamp(str(defs.date_range.end)),
                max_weight=float(defs.max_weight),
                monthly_deposit_eur=float(defs.monthly_deposit_eur),
                min_holding_period_days=int(defs.min_holding_period_days),
                rebalance_freq=str(defs.rebalance_freq),
                win_criterion=str(defs.win_criterion),
            ),
        )

    @classmethod
    def by_name(cls, name: str) -> "Portfolio":
        return cls.from_yaml(PORTFOLIOS_DIR / f"{name}.yaml")

    @classmethod
    def list_all(cls) -> list["Portfolio"]:
        if not PORTFOLIOS_DIR.exists():
            return []
        out = []
        for f in sorted(PORTFOLIOS_DIR.glob("*.yaml")):
            try:
                out.append(cls.from_yaml(f))
            except Exception:
                continue
        return out

    @property
    def base_dir(self) -> Path:
        return PORTFOLIOS_DIR / self.name

    @property
    def control_file(self) -> Path:
        return self.base_dir / "control.json"

    def ensure_base(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Scenario:
    portfolio: Portfolio
    knobs: dict
    id: str  # 8-char hex

    @classmethod
    def from_knobs(cls, portfolio: "Portfolio", knobs: dict) -> "Scenario":
        return cls(portfolio=portfolio, knobs=dict(knobs), id=scenario_id(knobs))

    @property
    def base_dir(self) -> Path:
        return self.portfolio.base_dir / "scenarios" / self.id

    @property
    def trials_db(self) -> Path:
        return self.base_dir / "trials.parquet"

    @property
    def optuna_storage(self) -> str:
        return f"sqlite:///{self.base_dir.as_posix()}/optuna.db"

    @property
    def baselines_path(self) -> Path:
        return self.base_dir / "baselines.parquet"

    @property
    def baseline_returns_path(self) -> Path:
        return self.base_dir / "baseline_returns.parquet"

    @property
    def baseline_nav_path(self) -> Path:
        return self.base_dir / "baseline_nav.parquet"

    @property
    def best_checkpoint(self) -> Path:
        return self.base_dir / "best_predictor.pt"

    @property
    def trial_dirs(self) -> Path:
        return self.base_dir / "trial_data"

    @property
    def knobs_path(self) -> Path:
        return self.base_dir / "knobs.json"

    def ensure_dirs(self) -> None:
        self.portfolio.ensure_base()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.trial_dirs.mkdir(parents=True, exist_ok=True)
        # Save knobs.json so we know what the scenario hash represents
        if not self.knobs_path.exists():
            self.knobs_path.write_text(
                json.dumps(self.knobs, indent=2, default=str), encoding="utf-8"
            )


WIN_CRITERIA = {
    "total_return": (
        "💰 Maximálny dlhodobý výnos",
        "Pre investora ktorý cieli na **najvyššie konečné €** za 10+ rokov a "
        "vie zniesť ostré výkyvy po ceste (vrátane -40% drawdownov v bear-och). "
        "Optuna maximalizuje konečné NAV.",
    ),
    "sharpe": (
        "🛡️ Stabilné portfólio počas života",
        "Pre investora ktorý chce **plynulý rast bez veľkých stresov** počas "
        "celého života. Penalizuje volatilitu a hlboké drawdowny. "
        "Optuna maximalizuje Sharpe = výnos / riziko (štandard finančnej "
        "ekonometrie, základ PSR/DSR testov).",
    ),
    # Hidden legacy criterion: kept for backward compatibility with old scenarios
    # but no longer exposed in the UI dropdown.
    "beat_benchmarks": (
        "🥊 Poraziť benchmarky (legacy)",
        "AI musí prekonať SPY na Sharpe aj výnose.",
    ),
}

# Public UI choices — only these two are shown in the criterion dropdown.
WIN_CRITERIA_PUBLIC = ("total_return", "sharpe")
