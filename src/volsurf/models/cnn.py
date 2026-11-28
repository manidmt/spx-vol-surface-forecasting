"""Convolutional network over the whole surface (port of notebook 05)."""

from typing import Literal

import torch
from torch import nn

from volsurf.models.torch_base import TorchForecaster


class SurfaceCNN(nn.Module):
    """3D convolutions over (time, tenor, moneyness), mean over time, then a head to the changes.

    Like a Keras subclassed ``Model``: layers are created in ``__init__`` and ``forward`` plays
    the role of ``call``. Input is channels-first, ``(batch, 2, lookback, n_tenors,
    n_moneyness)``; output is ``(batch, n_tenors * n_moneyness)`` scaled changes in grid order.

    ``head="conv"``: a 1x1 convolution, the same weights at every grid point (few parameters).
    ``head="dense"``: a linear layer from the whole feature map, separate weights per point.
    The head starts at zero, so the untrained network forecasts no change (persistence).
    ``in_channels`` is 2 plus the number of surface-memory channels.
    """

    def __init__(
        self,
        n_tenors: int,
        n_moneyness: int,
        channels: int = 16,
        head: Literal["conv", "dense"] = "conv",
        in_channels: int = 2,
    ) -> None:
        super().__init__()
        if head not in ("conv", "dense"):
            raise ValueError(f"head must be 'conv' or 'dense', got {head!r}")
        # nn.Sequential ≈ keras.Sequential; Conv3d(in, out, 3, padding=1) ≈ Conv3D(out, 3, padding="same")
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GELU(),
        )
        self.head_type = head
        if head == "conv":
            self.head = nn.Conv2d(channels, 1, 1)
        else:
            self.head = nn.Linear(channels * n_tenors * n_moneyness, n_tenors * n_moneyness)
        # ≈ kernel_initializer="zeros", bias_initializer="zeros"
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, 2, lookback, n_tenors, n_moneyness)`` to ``(batch, n_tenors * n_moneyness)``."""
        h = self.conv(x).mean(dim=2)  # average over the time axis -> (batch, channels, tenor, moneyness)
        # flatten(1) ≈ keras Flatten: keeps the batch axis, merges the rest
        return self.head(h).flatten(1) if self.head_type == "conv" else self.head(h.flatten(1))


def make_cnn(
    name: str, head: str = "conv", channels: int = 16, memory=(), **trainer_kwargs
) -> TorchForecaster:
    """Surface CNN forecaster; ``trainer_kwargs`` go to ``TorchForecaster`` (lookback, seeds, ...)."""
    in_channels = 2 + len(memory)
    return TorchForecaster(
        name,
        net_factory=lambda n_tenors, n_moneyness: SurfaceCNN(
            n_tenors, n_moneyness, channels, head, in_channels
        ),
        memory=memory,
        **trainer_kwargs,
    )
