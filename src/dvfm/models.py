import torch
from torch import nn
import math
import torch.nn.functional as F


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



class ResBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection, conditioned on the time embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # 1x1 conv to match channels in the residual connection when they differ
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(temb)[:, :, None, None]  # add time info to every pixel
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Small time-conditioned U-Net for 28x28 images (resolutions 28 -> 14 -> 7 -> 14 -> 28)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 2),
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder: one ResBlock per level, then halve the resolution (except at the last level)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out = base_channels * mult
            self.down_blocks.append(ResBlock(ch, out, time_dim))
            ch = out
            if i < len(channel_mults) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))

        # Bottleneck
        self.mid = ResBlock(ch, ch, time_dim)

        # Decoder: concatenate the skip connection, ResBlock, then double the resolution
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out = base_channels * mult
            self.up_blocks.append(ResBlock(ch + out, out, time_dim))
            ch = out
            if i < len(channel_mults) - 1:
                self.upsamples.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                    )
                )

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_embedding(t)
        h = self.stem(x)

        skips = []
        for i, block in enumerate(self.down_blocks):
            h = block(h, temb)
            skips.append(h)  # save for the decoder
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid(h, temb)

        for i, block in enumerate(self.up_blocks):
            h = block(torch.cat([h, skips.pop()], dim=1), temb)  # skip connection
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        return self.out_conv(F.silu(self.out_norm(h)))