"""Plug-in strategies. Each is a Strategy subclass with declared parameters."""
from .benchmark import BuyAndHold
from .breakout import Breakout
from .dual_momentum import DualMomentum
from .mean_reversion import RSIMeanReversion
from .trend_filter import TrendFilter

__all__ = ["BuyAndHold", "Breakout", "DualMomentum", "RSIMeanReversion", "TrendFilter"]
