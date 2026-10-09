import torch
from torch import nn


def flow_matching_loss(model: nn.Module, x1: torch.Tensor) -> torch.Tensor:
    """Flow matching loss for a batch of real data x1 of shape (B, 2)."""
    # 1. Noise: one Gaussian point per data point
    x0 = torch.randn_like(x1)

    # 2. A random time in [0, 1] for each point
    t = torch.rand(x1.shape[0], device=x1.device)  # (B,)

    # 3. Point on the straight line between noise and data
    t_col = t.view(-1, 1)  # (B,) -> (B, 1) so it multiplies each row
    xt = (1 - t_col) * x0 + t_col * x1

    # 4. Target velocity along that line
    target = x1 - x0

    # 5. Mean squared error between prediction and target
    pred = model(xt, t)
    return ((pred - target) ** 2).mean()