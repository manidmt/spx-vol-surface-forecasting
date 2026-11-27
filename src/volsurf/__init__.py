"""Experiment framework for next-day forecasts of the S&P 500 implied-volatility surface.

The package loads the 7 x 9 (tenor x moneyness) surface, runs walk-forward evaluations of
baselines and neural models, and scores them with a common set of metrics.
"""

__version__ = "0.1.0"
