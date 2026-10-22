"""Plug-in strategies. Each is a Strategy subclass with declared parameters."""
from .basis import BasisCarry
from .benchmark import BuyAndHold
from .breakout import Breakout
from .carry import FXCarry
from .cot_positioning import CotPositioning
from .crypto import CrossSectionalMomentum, EnsembleTrend, Sleeves, VolTargetBreakout
from .dual_momentum import DualMomentum
from .earnings import EarningsPremium
from .events import EventHold
from .fundamental_allocator import FundamentalAllocator
from .hedge_fund_clone import best_ideas, HedgeFundClone
from .insider import classify_insiders, cluster_signals, InsiderCluster
from .mean_reversion import RSIMeanReversion
from .multi_asset_trend import MultiAssetTrend
from .pension_rebalance import PensionRebalance
from .risk_parity import RiskParity
from .seasonality import SameMonthSeasonality
from .sector_rotation import SectorRotation
from .signal_blend import SignalBlend
from .spreads import CreditSpreads
from .stock_momentum import HedgedMomentum
from .trend_filter import TrendFilter
from .trump import EventDip, GovStakeEvents, PostRiskOverlay
from .turn_of_month import TurnOfMonth
from .vix_term_structure import VixTermStructure
from .vol_target import VolTarget
from .wheel import Wheel

__all__ = [
    "CotPositioning",
    "FundamentalAllocator",
    "BasisCarry",
    "Breakout",
    "BuyAndHold",
    "CreditSpreads",
    "CrossSectionalMomentum",
    "DualMomentum",
    "EarningsPremium",
    "EnsembleTrend",
    "EventDip",
    "EventHold",
    "FXCarry",
    "GovStakeEvents",
    "HedgedMomentum",
    "HedgeFundClone",
    "InsiderCluster",
    "MultiAssetTrend",
    "PensionRebalance",
    "PostRiskOverlay",
    "RiskParity",
    "RSIMeanReversion",
    "SameMonthSeasonality",
    "SectorRotation",
    "SignalBlend",
    "Sleeves",
    "TrendFilter",
    "TurnOfMonth",
    "VixTermStructure",
    "VolTarget",
    "VolTargetBreakout",
    "Wheel",
]
