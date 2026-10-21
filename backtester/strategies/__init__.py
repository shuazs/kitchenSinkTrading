"""Plug-in strategies. Each is a Strategy subclass with declared parameters."""
from .benchmark import BuyAndHold
from .breakout import Breakout
from .dual_momentum import DualMomentum
from .mean_reversion import RSIMeanReversion
from .sector_rotation import SectorRotation
from .trend_filter import TrendFilter
from .turn_of_month import TurnOfMonth
from .vix_term_structure import VixTermStructure
from .vol_target import VolTarget
from .wheel import Wheel

__all__ = [
    "BuyAndHold",
    "Breakout",
    "DualMomentum",
    "RSIMeanReversion",
    "SectorRotation",
    "TrendFilter",
    "TurnOfMonth",
    "VixTermStructure",
    "VolTarget",
    "Wheel",
]
