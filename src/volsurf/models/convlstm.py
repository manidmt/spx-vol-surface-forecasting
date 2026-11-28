"""Convolutional LSTM over the days of the window, on the tenor x moneyness grid.

An LSTM whose gates are convolutions instead of dense layers, so every grid point keeps its
own hidden state and only talks to its neighbours (Shi et al., 2015; applied to implied
volatility surfaces by Adamski and Ślepaczuk, 2026). The Keras analogue is
``keras.layers.ConvLSTM2D(filters, kernel_size, padding="same")``; PyTorch has no built-in
ConvLSTM layer, so the cell and the loop over time are written out here.
"""

import torch
from torch import nn

from volsurf.models.torch_base import TorchForecaster


class ConvLSTMCell(nn.Module):
    """One time step of a ConvLSTM on a 2D grid.

    A single ``Conv2d`` over the concatenated input and previous hidden state produces the four
    gates at once (input, forget, output, candidate), ≈ Keras' ``ConvLSTM2D`` with its
    ``kernel`` and ``recurrent_kernel``. Differences a Keras user would notice: gate order,
    forget-bias initialisation (Keras ``unit_forget_bias=True``, here none) and a single bias
    vector in the conv (PyTorch's ``nn.LSTM`` has two). Padding ``kernel_size // 2`` keeps the grid
    size (≈ ``padding="same"`` for odd kernels).

    ``forward(x, (h, c))`` takes ``x`` of shape ``(batch, in_channels, height, width)`` and the
    previous state, each ``(batch, hidden_channels, height, width)``, and returns the new
    ``(h, c)``.
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=kernel_size // 2
        )

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the state by one day: ``(h, c)`` at ``t-1`` and ``x`` at ``t`` -> ``(h, c)`` at ``t``."""
        h, c = state
        # dim=1 is the channel axis (channels-first); Keras would concatenate on axis=-1
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)  # split the 4 * hidden channels into the four gates
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class SurfaceConvLSTM(nn.Module):
    """ConvLSTM over the lookback days, then a 1x1 convolution head from the last hidden state.

    Input is channels-first, ``(batch, 2, lookback, n_tenors, n_moneyness)`` (channel 0 the
    daily change, channel 1 the level); output is ``(batch, n_tenors * n_moneyness)`` scaled
    changes in grid order. Equivalent Keras stack, on channels-last input
    ``(batch, lookback, n_tenors, n_moneyness, 2)``::

        ConvLSTM2D(hidden_channels, kernel_size, padding="same", return_sequences=False)
        Conv2D(1, 1, kernel_initializer="zeros", bias_initializer="zeros")
        Flatten()

    The head starts at zero, so the untrained network forecasts no change (persistence).
    ``in_channels`` is 2 plus the number of surface-memory channels.
    """

    def __init__(
        self,
        n_tenors: int,
        n_moneyness: int,
        hidden_channels: int = 16,
        kernel_size: int = 3,
        in_channels: int = 2,
    ) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.head = nn.Conv2d(hidden_channels, 1, 1)
        # ≈ kernel_initializer="zeros", bias_initializer="zeros"
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, 2, lookback, n_tenors, n_moneyness)`` to ``(batch, n_tenors * n_moneyness)``."""
        batch, _, lookback, n_tenors, n_moneyness = x.shape
        # zero initial state, as Keras uses by default; new_zeros keeps x's dtype and device
        h = x.new_zeros(batch, self.cell.hidden_channels, n_tenors, n_moneyness)
        c = torch.zeros_like(h)
        # explicit loop over days ≈ what ConvLSTM2D does internally; x[:, :, t] is day t's grid
        for t in range(lookback):
            h, c = self.cell(x[:, :, t], (h, c))
        # last h only ≈ return_sequences=False; flatten(1) ≈ keras Flatten
        return self.head(h).flatten(1)


def make_convlstm(
    name: str, hidden_channels: int = 16, kernel_size: int = 3, memory=(), **trainer_kwargs
) -> TorchForecaster:
    """Surface ConvLSTM forecaster; ``trainer_kwargs`` go to ``TorchForecaster`` (lookback, seeds, ...)."""
    in_channels = 2 + len(memory)
    return TorchForecaster(
        name,
        net_factory=lambda n_tenors, n_moneyness: SurfaceConvLSTM(
            n_tenors, n_moneyness, hidden_channels, kernel_size, in_channels
        ),
        memory=memory,
        **trainer_kwargs,
    )
