"""Train the portfolio policy network.

The network takes a flat market-state vector and outputs portfolio weights
(softmax over assets). Training simulates buy-and-hold over a future horizon
using the dataset's `future_simple` returns; loss depends on the win criterion:
  - "sharpe"         → minimize negative annualized Sharpe
  - "total_return"   → minimize negative cumulative log-return
  - "beat_benchmarks"→ same as Sharpe loss (benchmark gap evaluated in selection)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from portopt.models.dataset import WindowDataset
from portopt.models.return_predictor import PortfolioPolicyMLP


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 128
    n_layers: int = 2
    dropout: float = 0.2
    early_stop_patience: int = 8
    seed: int = 42
    criterion: str = "sharpe"  # one of: sharpe | total_return | beat_benchmarks
    entropy_bonus: float = 0.0  # encourages weight diversification when > 0


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_sharpe: float
    val_total_return: float


@dataclass
class TrainResult:
    model: PortfolioPolicyMLP
    scaler: StandardScaler
    history: list[EpochMetrics]
    best_val_loss: float
    val_sharpe: float
    val_total_return: float
    asset_universe: list[str]
    input_dim: int
    criterion: str


def _portfolio_loss(
    weights: torch.Tensor,            # (B, N)
    future_simple: torch.Tensor,      # (B, T, N)
    criterion: str,
    entropy_bonus: float = 0.0,
) -> torch.Tensor:
    # Daily portfolio simple return assuming fixed-weight buy-and-hold over T days
    # (approximation: ignores intra-window weight drift; OK for training signal).
    daily_port = (weights.unsqueeze(1) * future_simple).sum(dim=-1)  # (B, T)

    if criterion == "total_return":
        # Sum of log(1+r); maximizing this maximizes final NAV (deposit-free).
        log_r = torch.log1p(daily_port.clamp(min=-0.99))
        loss = -log_r.sum(dim=-1).mean()
    else:
        # Sharpe loss (also used for beat_benchmarks during training)
        mean = daily_port.mean(dim=-1)
        std = daily_port.std(dim=-1) + 1e-6
        sharpe = mean / std * (252.0**0.5)
        loss = -sharpe.mean()

    if entropy_bonus > 0:
        # Encourage diversification (uniform = max entropy)
        eps = 1e-8
        ent = -(weights * (weights + eps).log()).sum(dim=-1).mean()
        loss = loss - entropy_bonus * ent
    return loss


def _eval_metrics(model, X_val_t, fut_val_t):
    model.eval()
    with torch.no_grad():
        w = model(X_val_t)
        daily = (w.unsqueeze(1) * fut_val_t).sum(dim=-1)
        mean = daily.mean(dim=-1)
        std = daily.std(dim=-1) + 1e-6
        sharpe = (mean / std * (252.0**0.5)).mean().item()
        total_return = (torch.log1p(daily.clamp(min=-0.99)).sum(dim=-1)).mean().item()
    return float(sharpe), float(total_return)


def train_policy(
    train_ds: WindowDataset,
    val_ds: WindowDataset,
    cfg: TrainConfig,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
) -> TrainResult:
    rng = np.random.RandomState(cfg.seed)
    torch.manual_seed(cfg.seed)

    n_assets = len(train_ds.asset_universe)
    input_dim = train_ds.X.shape[1]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_ds.X)
    X_val = scaler.transform(val_ds.X)

    Xt = torch.from_numpy(X_train.astype(np.float32))
    Ft = torch.from_numpy(train_ds.future_simple.astype(np.float32))
    Xv = torch.from_numpy(X_val.astype(np.float32))
    Fv = torch.from_numpy(val_ds.future_simple.astype(np.float32))

    model = PortfolioPolicyMLP(
        n_assets=n_assets,
        input_dim=input_dim,
        hidden=cfg.hidden,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: list[EpochMetrics] = []
    best_val = float("inf")
    best_state: dict | None = None
    patience_left = cfg.early_stop_patience

    n_train = len(Xt)
    for epoch in range(cfg.epochs):
        model.train()
        idx = rng.permutation(n_train)
        train_losses = []
        for s in range(0, n_train, cfg.batch_size):
            b = idx[s : s + cfg.batch_size]
            xb = Xt[b]
            fb = Ft[b]
            w = model(xb)
            loss = _portfolio_loss(w, fb, cfg.criterion, cfg.entropy_bonus)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            wv = model(Xv)
            val_loss = _portfolio_loss(wv, Fv, cfg.criterion, cfg.entropy_bonus).item()
        val_sharpe, val_total = _eval_metrics(model, Xv, Fv)

        m = EpochMetrics(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)),
            val_loss=float(val_loss),
            val_sharpe=val_sharpe,
            val_total_return=val_total,
        )
        history.append(m)
        if on_epoch is not None:
            on_epoch(m)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_sharpe, final_total = _eval_metrics(model, Xv, Fv)

    return TrainResult(
        model=model,
        scaler=scaler,
        history=history,
        best_val_loss=float(best_val),
        val_sharpe=float(final_sharpe),
        val_total_return=float(final_total),
        asset_universe=list(train_ds.asset_universe),
        input_dim=int(input_dim),
        criterion=cfg.criterion,
    )


def save_checkpoint(result: TrainResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": result.model.state_dict(),
            "n_assets": result.model.n_assets,
            "input_dim": result.input_dim,
            "hidden": result.model.hidden,
            "n_layers": result.model.n_layers,
            "dropout": result.model.dropout,
            "scaler_mean": result.scaler.mean_,
            "scaler_scale": result.scaler.scale_,
            "asset_universe": result.asset_universe,
            "best_val_loss": result.best_val_loss,
            "val_sharpe": result.val_sharpe,
            "val_total_return": result.val_total_return,
            "criterion": result.criterion,
        },
        path,
    )


# Back-compat alias
train_predictor = train_policy
