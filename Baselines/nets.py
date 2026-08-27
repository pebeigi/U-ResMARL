"""Network building blocks shared by the learned baselines."""

from __future__ import annotations

import Baselines._paths  # noqa: F401

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for the learned baselines. Install with: pip install torch"
    ) from exc


class RunningNorm(nn.Module):
    """Welford running mean/variance whitening of an input vector."""

    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.clip = float(clip)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        batch = batch.reshape(-1, batch.shape[-1])
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(batch.shape[0]))

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total)
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta.pow(2) * self.count * batch_count / total
        )

        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normed = (obs - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(normed, -self.clip, self.clip)


def mlp(input_dim: int, hidden_dim: int, output_dim: int, final_tanh: bool = False) -> nn.Sequential:
    layers = [
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    ]
    if final_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)
