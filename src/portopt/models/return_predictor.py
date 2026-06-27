"""Portfolio policy network.

Outputs portfolio weights DIRECTLY via softmax over the asset universe.
Long-only by construction; weights sum to 1.

This is a complete replacement of the older NN-Markowitz pipeline where the
network predicted expected returns that were fed to Markowitz optimization.
Now the network IS the portfolio allocator.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PortfolioPolicyMLP(nn.Module):
    """MLP that outputs portfolio weights via softmax.

    Input: flat market-state vector (per-asset features)
    Output: weights of shape (n_assets,) summing to 1, all ≥ 0.
    """

    def __init__(
        self,
        n_assets: int,
        input_dim: int,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_assets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Back-compat alias: any old imports of `ReturnPredictorMLP` get the new policy.
# They'll fail at use-site (different signature), but at least imports don't crash.
ReturnPredictorMLP = PortfolioPolicyMLP
