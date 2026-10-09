import torch
from torch import nn
import math


class MLP(nn.Module):
    """Time-conditioned MLP.

    Takes points x of shape (B, 2) and times t of shape (B,),
    and returns a tensor of shape (B, 2).
    """

    def __init__(self, data_dim: int = 2, hidden_dim: int = 256, num_layers: int = 4) -> None:
        super().__init__()

        # Input: data_dim coordinates + 1 time value
        layers = [nn.Linear(data_dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        # Output: same shape as the data
        layers.append(nn.Linear(hidden_dim, data_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)             # (B,)   -> (B, 1)
        h = torch.cat([x, t], dim=1)  # (B, 2) + (B, 1) -> (B, 3)
        return self.net(h)



class SinusoidalTimeEmbedding(nn.Module):
    """Turns a scalar time t in [0, 1] into a vector of sines and cosines at many frequencies."""

    def __init__(self, dim: int = 64, max_freq: float = 1000.0) -> None:
        super().__init__()
        half = dim // 2
        # Frequencies spaced geometrically from 1 to max_freq
        freqs = torch.exp(torch.linspace(0, math.log(max_freq), half))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t.view(-1, 1) * self.freqs.view(1, -1)                # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


class TimeEmbeddingMLP(nn.Module):
    """Same MLP as before, but t enters as a 64-dim sinusoidal embedding instead of a single number."""

    def __init__(
        self, data_dim: int = 2, hidden_dim: int = 256, num_layers: int = 4, time_dim: int = 64
    ) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        layers = [nn.Linear(data_dim + time_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, self.time_embedding(t)], dim=1)  # (B, 2 + 64)
        return self.net(h)