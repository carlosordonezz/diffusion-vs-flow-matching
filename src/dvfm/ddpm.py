import torch


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