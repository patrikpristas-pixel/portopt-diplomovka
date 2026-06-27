"""Champions — pinned trials a user wants to keep for later use.

A "champion" is a snapshot of a trial's metadata + hyperparameters + key metrics
saved to a portfolio-level JSON file. The original trial directory (with NN
checkpoints) stays where it is — the champion just bookmarks it.

Storage: portfolios/<name>/champions.json
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from portopt.portfolio import Portfolio


@dataclass
class Champion:
    id: str                     # auto-generated, e.g. "agg_t0153_20260514"
    portfolio: str              # portfolio name
    scenario_id: str            # 8-char hex
    trial_id: int
    pinned_at: str              # ISO timestamp
    label: str = ""             # user-given label
    notes: str = ""
    knobs: dict = field(default_factory=dict)
    hyperparams: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)  # sharpe, final_nav_eur, etc.

    def to_dict(self) -> dict:
        return asdict(self)


def _champions_path(portfolio_name: str) -> Path:
    return Path("portfolios") / portfolio_name / "champions.json"


def load_champions(portfolio_name: str) -> list[Champion]:
    p = _champions_path(portfolio_name)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for d in data:
        try:
            out.append(Champion(**d))
        except TypeError:
            # Skip rows with unexpected schema
            continue
    return out


def load_all_champions() -> list[Champion]:
    """All champions across all portfolios."""
    out: list[Champion] = []
    for p in Portfolio.list_all():
        out.extend(load_champions(p.name))
    return sorted(out, key=lambda c: c.pinned_at, reverse=True)


def save_champions(portfolio_name: str, champions: list[Champion]) -> None:
    p = _champions_path(portfolio_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([c.to_dict() for c in champions], indent=2, default=str),
        encoding="utf-8",
    )


def _make_id(portfolio_name: str, trial_id: int) -> str:
    short = portfolio_name[:3].lower()
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return f"{short}_t{trial_id:04d}_{ts}"


def pin_trial(
    portfolio_name: str,
    scenario_id: str,
    trial_id: int,
    knobs: dict,
    label: str = "",
    notes: str = "",
) -> Champion:
    """Pin a trial. Reads its hyperparams.json and metrics from trials.parquet."""
    trial_dir = (
        Path("portfolios") / portfolio_name / "scenarios" / scenario_id
        / "trial_data" / f"trial_{trial_id:04d}"
    )
    hp_path = trial_dir / "hyperparams.json"
    hyperparams = {}
    if hp_path.exists():
        try:
            hyperparams = json.loads(hp_path.read_text(encoding="utf-8"))
        except Exception:
            hyperparams = {}

    trials_path = (
        Path("portfolios") / portfolio_name / "scenarios" / scenario_id / "trials.parquet"
    )
    metrics: dict = {}
    if trials_path.exists():
        df = pd.read_parquet(trials_path)
        row = df[df["trial"] == trial_id]
        if len(row) > 0:
            r = row.iloc[-1]
            for k in (
                "sharpe", "median_sharpe", "final_nav_eur", "total_return",
                "max_drawdown", "ann_volatility", "ann_return",
                "objective", "won", "n_windows",
            ):
                if k in r.index:
                    v = r[k]
                    try:
                        metrics[k] = float(v) if not isinstance(v, bool) else bool(v)
                    except (ValueError, TypeError):
                        metrics[k] = v

    champ = Champion(
        id=_make_id(portfolio_name, trial_id),
        portfolio=portfolio_name,
        scenario_id=scenario_id,
        trial_id=trial_id,
        pinned_at=pd.Timestamp.now().isoformat(timespec="seconds"),
        label=label,
        notes=notes,
        knobs=knobs,
        hyperparams=hyperparams,
        metrics=metrics,
    )

    existing = load_champions(portfolio_name)
    # Avoid exact duplicates: same scenario+trial → replace
    filtered = [
        c for c in existing
        if not (c.scenario_id == scenario_id and c.trial_id == trial_id)
    ]
    filtered.append(champ)
    save_champions(portfolio_name, filtered)
    return champ


def unpin(portfolio_name: str, champion_id: str) -> bool:
    existing = load_champions(portfolio_name)
    new = [c for c in existing if c.id != champion_id]
    if len(new) == len(existing):
        return False
    save_champions(portfolio_name, new)
    return True


def is_pinned(portfolio_name: str, scenario_id: str, trial_id: int) -> bool:
    for c in load_champions(portfolio_name):
        if c.scenario_id == scenario_id and c.trial_id == trial_id:
            return True
    return False


def update_label(portfolio_name: str, champion_id: str, label: str, notes: str) -> None:
    existing = load_champions(portfolio_name)
    for c in existing:
        if c.id == champion_id:
            c.label = label
            c.notes = notes
    save_champions(portfolio_name, existing)


def trial_dir_for(champion: Champion) -> Path:
    return (
        Path("portfolios") / champion.portfolio / "scenarios" / champion.scenario_id
        / "trial_data" / f"trial_{champion.trial_id:04d}"
    )


def archive_trial_files(champion: Champion, dest_root: Path | None = None) -> Path:
    """Optional: copy trial files (checkpoints, weights, walkforward) to a
    portfolio-level champions/ subfolder so they survive scenario cleanup.
    """
    src = trial_dir_for(champion)
    if dest_root is None:
        dest_root = Path("portfolios") / champion.portfolio / "champions" / champion.id
    dest_root.mkdir(parents=True, exist_ok=True)
    if src.exists():
        for item in src.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src)
                target = dest_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
    return dest_root
