"""Return normalization shared by residual and direct categorical PPO."""

import torch
from torch import nn


class ValueNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("value_mean", torch.zeros(()))
        self.register_buffer("value_var", torch.ones(()))
        self.register_buffer("value_count", torch.tensor(1e-4))

    def value_std(self):
        return torch.sqrt(self.value_var + 1e-8)

    def denormalize_value(self, value):
        return value * self.value_std() + self.value_mean

    def normalize_return(self, returns):
        return (returns - self.value_mean) / self.value_std()

    @torch.no_grad()
    def update_value_stats(self, returns):
        count = returns.new_tensor(float(returns.numel()))
        delta = returns.mean() - self.value_mean
        total = self.value_count + count
        var = (self.value_var * self.value_count + returns.var(unbiased=False) * count
               + delta.square() * self.value_count * count / total) / total
        self.value_mean.add_(delta * count / total)
        self.value_var.copy_(var)
        self.value_count.copy_(total)
