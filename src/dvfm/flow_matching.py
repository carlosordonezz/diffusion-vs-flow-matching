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


@torch.no_grad()
def sample(
    model: nn.Module, n: int, num_steps: int = 100, return_trajectory: bool = False
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Generate n points by following the learned velocity from noise (t=0) to data (t=1).

    If return_trajectory is True, also return the list of intermediate states
    (num_steps + 1 tensors of shape (n, 2), from pure noise to the final samples).
    """
    x = torch.randn(n, 2)  # start from pure noise
    dt = 1.0 / num_steps
    trajectory = [x.clone()]

    for i in range(num_steps):
        t = torch.full((n,), i * dt)  # same time for every point
        x = x + dt * model(x, t)      # one Euler step
        trajectory.append(x.clone())

    if return_trajectory:
        return x, trajectory
    return x