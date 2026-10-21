"""Plug-in strategies. Each is a Strategy subclass with declared parameters."""
from .benchmark import BuyAndHold
from .breakout import Breakout
from .crypto import CrossSectionalMomentum, Sleeves, VolTargetBreakout
from .dual_momentum import DualMomentum
from .insider import cluster_signals, InsiderCluster
from .mean_reversion import RSIMeanReversion
from .sector_rotation import SectorRotation
from .spreads import CreditSpreads
from .trend_filter import TrendFilter
from .trump import EventDip, GovStakeEvents, PostRiskOverlay
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
    "EventDip",
    "GovStakeEvents",
    "InsiderCluster",
    "PostRiskOverlay",
    "RSIMeanReversion",
    "SectorRotation",
    "Sleeves",
    "TrendFilter",
    "TurnOfMonth",
    "VixTermStructure",
    "VolTarget",
    "VolTargetBreakout",
    "Wheel",
]
