import math

import torch


def sample_moons(n: int, noise: float = 0.05, seed: int | None = None) -> torch.Tensor:
    """Sample n points from the two-moons distribution.

    Returns a tensor of shape (n, 2).
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    n_outer = n // 2
    n_inner = n - n_outer

    # Outer moon: (cos θ, sin θ), θ in [0, π]
    theta_outer = torch.rand(n_outer, generator=g) * math.pi
    outer = torch.stack([torch.cos(theta_outer), torch.sin(theta_outer)], dim=1)

    # Inner moon: (1 - cos θ, 0.5 - sin θ), θ in [0, π]
    theta_inner = torch.rand(n_inner, generator=g) * math.pi
    inner = torch.stack([1 - torch.cos(theta_inner), 0.5 - torch.sin(theta_inner)], dim=1)

    # Join both moons and add Gaussian noise
    x = torch.cat([outer, inner])
    x = x + noise * torch.randn(n, 2, generator=g)

    # Center around (0, 0) and scale to roughly match N(0, 1)
    x = (x - torch.tensor([0.5, 0.25])) * 1.5

    return x