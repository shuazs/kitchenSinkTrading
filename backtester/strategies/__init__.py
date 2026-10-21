"""Plug-in strategies. Each is a Strategy subclass with declared parameters."""
from .benchmark import BuyAndHold
from .breakout import Breakout
from .crypto import CrossSectionalMomentum, Sleeves, VolTargetBreakout
from .dual_momentum import DualMomentum
from .events import EventHold
from .hedge_fund_clone import best_ideas, HedgeFundClone
from .insider import cluster_signals, InsiderCluster
from .mean_reversion import RSIMeanReversion
from .multi_asset_trend import MultiAssetTrend
from .pension_rebalance import PensionRebalance
from .risk_parity import RiskParity
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
    "EventHold",
    "HedgeFundClone",
    "InsiderCluster",
    "MultiAssetTrend",
    "PensionRebalance",
    "RiskParity",
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
