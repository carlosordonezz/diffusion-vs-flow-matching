from collections.abc import Callable

import torch
from torch import nn

from dvfm.flow_matching import flow_matching_loss

LossFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]


def train_model(
    model: nn.Module,
    data: torch.Tensor,
    loss_fn: LossFn,
    num_steps: int = 5000,
    batch_size: int = 512,
    lr: float = 1e-3,
    log_every: int = 1000,
    cosine_schedule: bool = False,
) -> list[float]:
    """Generic training loop. `loss_fn(model, batch)` must return a scalar loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps) if cosine_schedule else None
    )
    losses = []

    model.train()
    for step in range(num_steps):
        idx = torch.randint(0, len(data), (batch_size,))
        loss = loss_fn(model, data[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")

    model.eval()
    return losses


def train_flow_matching(model: nn.Module, data: torch.Tensor, **kwargs) -> list[float]:
    """Shortcut so the earlier notebooks keep working."""
    return train_model(model, data, flow_matching_loss, **kwargs)