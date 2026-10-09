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



def sample_spirals(n: int, noise: float = 0.05, seed: int | None = None) -> torch.Tensor:
    """Sample n points from two interleaved spirals. Returns (n, 2)."""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    # Angle along the spiral (sqrt makes the density uniform along its length)
    theta = torch.sqrt(torch.rand(n, generator=g)) * 3 * math.pi
    radius = theta / (3 * math.pi) * 2  # radius grows with the angle, up to 2

    x = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)

    # Flip half of the points to get the second spiral
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    x = x * sign.view(-1, 1)

    return x + noise * torch.randn(n, 2, generator=g)



def sample_checkerboard(n: int, seed: int | None = None) -> torch.Tensor:
    """Sample n points uniformly from the dark squares of a 4x4 checkerboard in [-2, 2]^2."""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    x1 = torch.rand(n, generator=g) * 4 - 2  # uniform in [-2, 2]
    x2 = torch.rand(n, generator=g) - 2 * torch.randint(0, 2, (n,), generator=g)
    x2 = x2 + (torch.floor(x1) % 2)  # shift every other column -> checkerboard

    return torch.stack([x1, x2], dim=1)