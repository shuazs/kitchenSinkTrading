"""Machine-learning signal combination: a weekly feature panel built from the
repo's strategy signals, purged walk-forward folds, and a ladder of models
(equal weight, ridge, gradient boosting, MLP, LSTM) that compete on them.

Needs the optional extra: pip install -e ".[ml]"
"""
from .features import ASSET_FEATURES, FEATURES, MARKET_FEATURES, build_panel
from .splits import Fold, holdout_fold, walk_forward_folds

__all__ = [
    "ASSET_FEATURES",
    "FEATURES",
    "MARKET_FEATURES",
    "Fold",
    "build_panel",
    "holdout_fold",
    "walk_forward_folds",
]
