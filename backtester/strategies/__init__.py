"""Plug-in strategies. Each is a Strategy subclass with declared parameters."""
from .benchmark import BuyAndHold
from .breakout import Breakout
from .crypto import CrossSectionalMomentum, Sleeves, VolTargetBreakout
from .dual_momentum import DualMomentum
from .insider import InsiderCluster, cluster_signals
from .mean_reversion import RSIMeanReversion
from .trend_filter import TrendFilter
from .wheel import Wheel

__all__ = [
    "BuyAndHold",
    "Breakout",
    "CrossSectionalMomentum",
    "Sleeves",
    "VolTargetBreakout",
    "DualMomentum",
    "InsiderCluster",
    "RSIMeanReversion",
    "TrendFilter",
    "Wheel",
]
