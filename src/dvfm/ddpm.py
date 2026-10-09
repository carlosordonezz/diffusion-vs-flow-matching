import math

import torch
from torch import nn

from dvfm.flow_matching import expand_like


class DDPMSchedule:
    """Linear noise schedule from Ho et al. (2020).

    Convention: t = 0 is clean data, t = num_timesteps - 1 is (almost) pure noise.
    """

    def __init__(
        self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02
    ) -> None:
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)  # noise added per step
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # product of alphas up to t


def q_sample(
    schedule: DDPMSchedule, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    """Noisy version of x0 at integer timesteps t (shape (B,)), in one shot.

    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    """
    alpha_bar = schedule.alpha_bars.to(x0.device)[t]  # move the schedule to x0's device
    alpha_bar = expand_like(alpha_bar, x0)
    return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise


def ddpm_loss(model: nn.Module, schedule: DDPMSchedule, x0: torch.Tensor) -> torch.Tensor:
    """Simplified DDPM loss: the network predicts the noise that was added to x0."""
    t = torch.randint(0, schedule.num_timesteps, (x0.shape[0],), device=x0.device)
    noise = torch.randn_like(x0)
    xt = q_sample(schedule, x0, t, noise)

    t_input = t.float() / schedule.num_timesteps  # scale to [0, 1) for the network
    pred = model(xt, t_input)
    return ((pred - noise) ** 2).mean()


@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    schedule: DDPMSchedule,
    n: int,
    return_trajectory: bool = False,
    shape: tuple[int, ...] = (2,),
    device: torch.device | str = "cpu",
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Ancestral sampling (Algorithm 2 in Ho et al.): from pure noise at t=T-1 down to t=0."""
    T = schedule.num_timesteps
    x = torch.randn(n, *shape, device=device)
    trajectory = [x.cpu()]

    for t in reversed(range(T)):  # 999, 998, ..., 0
        t_input = torch.full((n,), t / T, device=device)
        eps_pred = model(x, t_input)

        # Plain Python floats: no device issues
        alpha = schedule.alphas[t].item()
        alpha_bar = schedule.alpha_bars[t].item()
        beta = schedule.betas[t].item()

        mean = (x - beta / math.sqrt(1 - alpha_bar) * eps_pred) / math.sqrt(alpha)
        if t > 0:
            x = mean + math.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean

        trajectory.append(x.cpu())

    if return_trajectory:
        return x, trajectory
    return x


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    schedule: DDPMSchedule,
    n: int,
    num_steps: int = 50,
    return_trajectory: bool = False,
    clip: float = 3.0,
    shape: tuple[int, ...] = (2,),
    device: torch.device | str = "cpu",
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Deterministic DDIM sampling (Song et al., 2021) with a DDPM-trained model.

    Uses only `num_steps` timesteps, evenly spaced between T-1 and 0.
    Use clip=3.0 for the 2D datasets and clip=1.0 for images in [-1, 1].
    """
    T = schedule.num_timesteps
    timesteps = torch.linspace(T - 1, 0, num_steps).long().tolist()

    x = torch.randn(n, *shape, device=device)
    trajectory = [x.cpu()]

    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1  # -1 means "clean data"

        t_input = torch.full((n,), t / T, device=device)
        eps_pred = model(x, t_input)

        alpha_bar = schedule.alpha_bars[t].item()
        alpha_bar_prev = schedule.alpha_bars[t_prev].item() if t_prev >= 0 else 1.0

        # 1. Estimate the clean data from the predicted noise
        x0_pred = (x - math.sqrt(1 - alpha_bar) * eps_pred) / math.sqrt(alpha_bar)
        x0_pred = x0_pred.clamp(-clip, clip)

        # 2. Jump to t_prev using that estimate and the same predicted noise
        x = math.sqrt(alpha_bar_prev) * x0_pred + math.sqrt(1 - alpha_bar_prev) * eps_pred

        trajectory.append(x.cpu())

    if return_trajectory:
        return x, trajectory
    return x