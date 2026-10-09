import torch
from torch import nn


def expand_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a per-sample tensor of shape (B,) so it broadcasts against x of shape (B, ...).

    (B,) -> (B, 1) for 2D points, (B, 1, 1, 1) for images.
    """
    return t.view(-1, *([1] * (x.dim() - 1)))


def flow_matching_loss(model: nn.Module, x1: torch.Tensor) -> torch.Tensor:
    """Flow matching loss for a batch of real data x1 of any shape (B, ...)."""
    x0 = torch.randn_like(x1)                       # noise, same shape and device as x1
    t = torch.rand(x1.shape[0], device=x1.device)   # one time per sample, shape (B,)

    t_b = expand_like(t, x1)
    xt = (1 - t_b) * x0 + t_b * x1                  # point on the straight line
    target = x1 - x0                                # velocity along the line

    pred = model(xt, t)
    return ((pred - target) ** 2).mean()


@torch.no_grad()
def sample(
    model: nn.Module,
    n: int,
    num_steps: int = 100,
    return_trajectory: bool = False,
    shape: tuple[int, ...] = (2,),
    device: torch.device | str = "cpu",
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Generate n samples of the given shape by following the learned velocity (Euler).

    The trajectory is stored on the CPU to save GPU memory.
    """
    x = torch.randn(n, *shape, device=device)  # start from pure noise
    dt = 1.0 / num_steps
    trajectory = [x.cpu()]

    for i in range(num_steps):
        t = torch.full((n,), i * dt, device=device)
        x = x + dt * model(x, t)
        trajectory.append(x.cpu())

    if return_trajectory:
        return x, trajectory
    return x