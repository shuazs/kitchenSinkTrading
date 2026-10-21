"""Pluggable backtesting engine for retail trading strategy research."""
from .core import COSTS, Backtester, CostModel, MarketData, Result, Strategy
from .metrics import drawdown, summary
from .research import sweep, walk_forward

__all__ = [
    "COSTS",
    "Backtester",
    "CostModel",
    "MarketData",
    "Result",
    "Strategy",
    "drawdown",
    "summary",
    "sweep",
    "walk_forward",
]
