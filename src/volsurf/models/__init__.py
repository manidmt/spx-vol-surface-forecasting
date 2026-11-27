"""Model registry: build a forecaster from a config's ``kind``, ``name`` and ``params``."""

from volsurf.models.base import Forecaster

KINDS = ("persistence", "ar1", "cnn", "lstm", "convlstm")


def build_model(kind: str, name: str, params: dict) -> Forecaster:
    """Instantiate the forecaster ``kind`` named ``name`` with ``params`` from a config."""
    if kind == "persistence":
        if params:
            raise ValueError(f"persistence takes no params, got {sorted(params)}")
        from volsurf.models.baselines import Persistence

        return Persistence()
    if kind == "ar1":
        from volsurf.models.baselines import AR1Huber

        model = AR1Huber(**params)
        model.name = name
        return model
    if kind == "cnn":
        from volsurf.models.cnn import make_cnn

        return make_cnn(name, **params)
    if kind == "lstm":
        from volsurf.models.lstm import make_lstm

        return make_lstm(name, **params)
    if kind == "convlstm":
        from volsurf.models.convlstm import make_convlstm

        return make_convlstm(name, **params)
    raise ValueError(f"unknown model kind {kind!r}; valid kinds: {', '.join(KINDS)}")
