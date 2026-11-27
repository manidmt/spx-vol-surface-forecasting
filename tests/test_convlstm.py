import pandas as pd
import torch
from leakage import assert_no_leakage, leakage_dates

from volsurf.models import build_model
from volsurf.models.convlstm import ConvLSTMCell, SurfaceConvLSTM, make_convlstm
from volsurf.models.torch_base import TorchForecaster

TEST_START = pd.Timestamp("2020-06-01")


def tiny_convlstm():
    return make_convlstm(
        "tiny", seeds=1, epochs=3, patience=2, hidden_channels=4, device="cpu", deterministic=True
    )


def test_cell_output_shapes():
    cell = ConvLSTMCell(in_channels=2, hidden_channels=6)
    x = torch.randn(4, 2, 7, 9)
    zeros = torch.zeros(4, 6, 7, 9)
    h, c = cell(x, (zeros, zeros))
    assert h.shape == (4, 6, 7, 9)
    assert c.shape == (4, 6, 7, 9)


def test_cell_gate_convolution():
    cell = ConvLSTMCell(in_channels=2, hidden_channels=6, kernel_size=5)
    assert cell.conv.in_channels == 8
    assert cell.conv.out_channels == 24
    assert cell.conv.kernel_size == (5, 5)
    assert cell.conv.padding == (2, 2)


def test_untrained_convlstm_is_persistence():
    net = SurfaceConvLSTM(7, 9)
    out = net(torch.randn(4, 2, 5, 7, 9))
    assert out.shape == (4, 63)
    assert torch.all(out == 0)


def test_convlstm_uses_every_day_of_the_window():
    torch.manual_seed(0)
    net = SurfaceConvLSTM(7, 9, hidden_channels=4)
    torch.nn.init.normal_(net.head.weight)  # make the head non-zero so the output depends on h
    x = torch.randn(1, 2, 5, 7, 9)
    base = net(x)
    for day in range(5):
        x2 = x.clone()
        x2[:, :, day] += 1.0
        assert not torch.allclose(net(x2), base), f"day {day} of the window is ignored"


def test_convlstm_no_leakage(synthetic_surface):
    surface = synthetic_surface.loc[:"2020-08-31"]  # three refits
    assert_no_leakage(tiny_convlstm, surface, TEST_START, leakage_dates(surface, TEST_START))


def test_build_model_convlstm():
    model = build_model("convlstm", "x", {"hidden_channels": 4, "kernel_size": 3, "lookback": 5, "seeds": 1})
    assert isinstance(model, TorchForecaster)
    assert model.name == "x"
    assert model.lookback == 5
    assert model.seeds == 1
    net = model.net_factory(7, 9)
    assert isinstance(net, SurfaceConvLSTM)
    assert net.cell.hidden_channels == 4
