import torch
from torch import nn


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