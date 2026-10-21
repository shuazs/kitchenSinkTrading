"""Plug-in strategies. Each is a Strategy subclass with declared parameters."""
from .benchmark import BuyAndHold
from .breakout import Breakout
from .crypto import CrossSectionalMomentum, EnsembleTrend, Sleeves, VolTargetBreakout
from .dual_momentum import DualMomentum
from .earnings import EarningsPremium
from .insider import classify_insiders, cluster_signals, InsiderCluster
from .mean_reversion import RSIMeanReversion
from .seasonality import SameMonthSeasonality
from .sector_rotation import SectorRotation
from .spreads import CreditSpreads
from .trend_filter import TrendFilter
from .turn_of_month import TurnOfMonth
from .vix_term_structure import VixTermStructure
from .vol_target import VolTarget
from .wheel import Wheel

__all__ = [
    "Breakout",
    "BuyAndHold",
    "CreditSpreads",
    "CrossSectionalMomentum",
    "DualMomentum",
    "EarningsPremium",
    "EnsembleTrend",
    "InsiderCluster",
    "RSIMeanReversion",
    "SameMonthSeasonality",
    "SectorRotation",
    "Sleeves",
    "TrendFilter",
    "TurnOfMonth",
    "VixTermStructure",
    "VolTarget",
    "VolTargetBreakout",
    "Wheel",
]
