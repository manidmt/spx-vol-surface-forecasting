"""Recurrent network over daily surface windows."""

from typing import Any

import torch
from torch import nn

from volsurf.models.torch_base import TorchForecaster


class SurfaceLSTM(nn.Module):
    """Forecast scaled changes from the final hidden state of a single LSTM layer.

    Each day is flattened in channel-major grid order: the daily changes come first, followed by
    the levels, with moneyness varying fastest within each tenor. The zero-initialised head
    forecasts no change before training, which corresponds to persistence.
    """

    def __init__(self, n_tenors: int, n_moneyness: int, hidden: int = 64) -> None:
        super().__init__()
        n_points = n_tenors * n_moneyness
        # batch_first=True uses Keras LSTM's (batch, time, features) input layout.
        self.lstm = nn.LSTM(2 * n_points, hidden, num_layers=1, batch_first=True)
        # nn.Linear(hidden, n_points) ≈ Dense(n_points, kernel_initializer="zeros", bias_initializer="zeros")
        self.head = nn.Linear(hidden, n_points)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, 2, lookback, n_tenors, n_moneyness)`` to ``(batch, n_tenors * n_moneyness)``."""
        # Like transposing before Keras TimeDistributed(Flatten): keep batch and day separate.
        daily_features = x.permute(0, 2, 1, 3, 4).flatten(start_dim=2)
        # PyTorch returns the sequence and (hidden, cell) states. The last layer's final hidden
        # state matches the output of a single Keras LSTM with return_sequences=False.
        _, (hidden, _) = self.lstm(daily_features)
        return self.head(hidden[-1])


def make_lstm(name: str, hidden: int = 64, **trainer_kwargs: Any) -> TorchForecaster:
    """Build a surface LSTM forecaster, passing training settings to ``TorchForecaster``."""
    return TorchForecaster(
        name,
        net_factory=lambda n_tenors, n_moneyness: SurfaceLSTM(n_tenors, n_moneyness, hidden),
        **trainer_kwargs,
    )
