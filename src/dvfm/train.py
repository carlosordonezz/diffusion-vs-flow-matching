import torch
from torch import nn

from dvfm.flow_matching import flow_matching_loss


def train_flow_matching(
    model: nn.Module,
    data: torch.Tensor,
    num_steps: int = 5000,
    batch_size: int = 512,
    lr: float = 1e-3,
    log_every: int = 1000,
) -> list[float]:
    """Train a model with flow matching on data of shape (N, 2). Returns the loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    model.train()
    for step in range(num_steps):
        idx = torch.randint(0, len(data), (batch_size,))
        loss = flow_matching_loss(model, data[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")

    model.eval()
    return losses