"""Purged walk-forward folds for a weekly panel.

Each label is the return to the next week's close, so the last training
week's label ends where the test starts. `purge` drops that many weeks before
every test (and before every validation block) so no label overlaps the data
it is scored on. Training only ever looks back, so an embargo after the test
would remove nothing; the purge gap plays that role here.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    name: str
    train: pd.DatetimeIndex  # fit here
    val: pd.DatetimeIndex  # tune and early-stop here
    test: pd.DatetimeIndex  # score here, never seen while fitting

    @property
    def fit_all(self) -> pd.DatetimeIndex:
        """Train plus validation, for models that refit once tuned."""
        return self.train.append(self.val)


def _split_train_val(prior: pd.DatetimeIndex, purge: int, val_frac: float) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    usable = prior[: len(prior) - purge] if purge else prior
    n_val = max(1, int(len(usable) * val_frac))
    val = usable[-n_val:]
    train = usable[: max(0, len(usable) - n_val - purge)]
    return train, val


def walk_forward_folds(
    dates,
    first_test: str,
    test_end: str,
    purge: int = 2,
    val_frac: float = 0.2,
    min_train: int = 52,
) -> list[Fold]:
    """One fold per calendar year from `first_test` to `test_end`, each
    trained on every earlier week (an expanding window)."""
    dates = pd.DatetimeIndex(sorted(set(dates)))
    start, end = pd.Timestamp(first_test), pd.Timestamp(test_end)
    folds = []
    for year in range(start.year, end.year + 1):
        lo = max(start, pd.Timestamp(f"{year}-01-01"))
        hi = min(end, pd.Timestamp(f"{year}-12-31"))
        test = dates[(dates >= lo) & (dates <= hi)]
        if test.empty:
            continue
        train, val = _split_train_val(dates[dates < test[0]], purge, val_frac)
        if len(train) < min_train:
            continue
        folds.append(Fold(str(year), train, val, test))
    return folds


def holdout_fold(dates, holdout_start: str, purge: int = 2, val_frac: float = 0.2) -> Fold:
    """Train on everything before `holdout_start`, test on everything after."""
    dates = pd.DatetimeIndex(sorted(set(dates)))
    h = pd.Timestamp(holdout_start)
    train, val = _split_train_val(dates[dates < h], purge, val_frac)
    return Fold("holdout", train, val, dates[dates >= h])
