import pandas as pd
import torch
from leakage import assert_no_leakage, leakage_dates

from volsurf.models import KINDS, build_model
from volsurf.models.lstm import SurfaceLSTM, make_lstm
from volsurf.models.torch_base import TorchForecaster

TEST_START = pd.Timestamp("2020-06-01")


def tiny_lstm() -> TorchForecaster:
    return make_lstm("tiny", seeds=1, epochs=3, patience=2, hidden=8, device="cpu", deterministic=True)


def test_untrained_lstm_is_persistence() -> None:
    net = SurfaceLSTM(7, 9)
    out = net(torch.randn(4, 2, 5, 7, 9))
    assert out.shape == (4, 63)
    assert torch.all(out == 0)


def test_lstm_uses_channel_major_days_and_final_hidden_state() -> None:
    net = SurfaceLSTM(2, 3, hidden=6)
    x = torch.randn(2, 2, 4, 2, 3)
    daily_features = torch.cat([x[:, 0].flatten(2), x[:, 1].flatten(2)], dim=-1)
    with torch.no_grad():
        # An identity head exposes the hidden state, which the zero head would mask.
        net.head.weight.copy_(torch.eye(6))
        _, (hidden, _) = net.lstm(daily_features)
        torch.testing.assert_close(net(x), hidden[-1])


def test_lstm_no_leakage(synthetic_surface: pd.DataFrame) -> None:
    surface = synthetic_surface.loc[:"2020-08-31"]  # three monthly refits
    assert_no_leakage(tiny_lstm, surface, TEST_START, leakage_dates(surface, TEST_START))


def test_build_model_lstm() -> None:
    assert "lstm" in KINDS
    model = build_model("lstm", "x", {"hidden": 8, "seeds": 1, "device": "cpu"})
    assert isinstance(model, TorchForecaster)
    assert model.name == "x"
    assert model.seeds == 1
    assert model.net_factory(7, 9).lstm.hidden_size == 8
