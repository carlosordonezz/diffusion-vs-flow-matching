import torch
from torch import nn

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
    alpha_bar = schedule.alpha_bars[t].view(-1, 1)  # (B,) -> (B, 1)
    return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise


def ddpm_loss(model: nn.Module, schedule: DDPMSchedule, x0: torch.Tensor) -> torch.Tensor:
    """Simplified DDPM loss: the network predicts the noise that was added to x0."""
    # 1. A random integer timestep for each point
    t = torch.randint(0, schedule.num_timesteps, (x0.shape[0],), device=x0.device)

    # 2. Noise and the noisy data
    noise = torch.randn_like(x0)
    xt = q_sample(schedule, x0, t, noise)

    # 3. The network gets t scaled to [0, 1), like in flow matching
    t_input = t.float() / schedule.num_timesteps

    # 4. Mean squared error between predicted and real noise
    pred = model(xt, t_input)
    return ((pred - noise) ** 2).mean()



@torch.no_grad()
def ddpm_sample(
    model: nn.Module, schedule: DDPMSchedule, n: int, return_trajectory: bool = False
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Ancestral sampling (Algorithm 2 in Ho et al.): from pure noise at t=T-1 down to t=0."""
    x = torch.randn(n, 2)  # start from pure noise
    trajectory = [x.clone()]

    for t in reversed(range(schedule.num_timesteps)):  # 999, 998, ..., 0
        t_input = torch.full((n,), t / schedule.num_timesteps)
        eps_pred = model(x, t_input)  # predicted noise

        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        beta = schedule.betas[t]

        # Remove a bit of the predicted noise -> mean of x_{t-1}
        mean = (x - beta / torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha)

        # Add fresh noise, except at the very last step
        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean

        trajectory.append(x.clone())

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
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Deterministic DDIM sampling (Song et al., 2021) with a DDPM-trained model.

    Uses only `num_steps` timesteps, evenly spaced between T-1 and 0.
    """
    T = schedule.num_timesteps
    timesteps = torch.linspace(T - 1, 0, num_steps).long().tolist()  # e.g. [999, 946, ..., 0]

    x = torch.randn(n, 2)  # start from pure noise
    trajectory = [x.clone()]

    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1  # -1 means "clean data"

        t_input = torch.full((n,), t / T)
        eps_pred = model(x, t_input)

        alpha_bar = schedule.alpha_bars[t]
        alpha_bar_prev = schedule.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # 1. Estimate the clean data from the predicted noise
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
        x0_pred = x0_pred.clamp(-clip, clip)

        # 2. Jump to t_prev using that estimate and the same predicted noise (no fresh noise)
        x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * eps_pred

        trajectory.append(x.clone())

    if return_trajectory:
        return x, trajectory
    return x