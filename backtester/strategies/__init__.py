"""Plug-in strategies. Each is a Strategy subclass with declared parameters."""
from .benchmark import BuyAndHold
from .breakout import Breakout
from .crypto import CrossSectionalMomentum, Sleeves, VolTargetBreakout
from .dual_momentum import DualMomentum
from .insider import cluster_signals, InsiderCluster
from .mean_reversion import RSIMeanReversion
from .spreads import CreditSpreads
from .trend_filter import TrendFilter
from .wheel import Wheel

__all__ = [
    "Breakout",
    "BuyAndHold",
    "CreditSpreads",
    "CrossSectionalMomentum",
    "DualMomentum",
    "InsiderCluster",
    "RSIMeanReversion",
    "Sleeves",
    "TrendFilter",
    "VolTargetBreakout",
    "Wheel",
]
