"""The model ladder. Every model has the same interface:

    model.fit(panel, fold)          # learns on fold.train, tunes on fold.val
    model.predict(panel, dates)     # -> Series of positions in [-1, 1] on (date, symbol)

Regression models predict the vol-scaled return; every model's score is
divided by its standard deviation on the fitting data, so all of them hand
the same kind of signal to the same sizing rule (SignalBlend clips it to
[-1, 1] for risk-parity books and ranks it for stock picking). The LSTM
outputs a position directly, trained on Sharpe ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES, SIGNAL_SIGNS
from .splits import Fold


class Scaler:
    """z-score with statistics from the fitting rows only, clipped at +-5."""

    def fit(self, X: pd.DataFrame) -> "Scaler":
        self.mu = X.mean()
        self.sd = X.std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return ((X - self.mu) / self.sd).clip(-5, 5).fillna(0.0)


class Model:
    name = "model"
    features = FEATURES

    def _rows(self, panel: pd.DataFrame, dates, labelled: bool = True) -> pd.DataFrame:
        rows = panel[panel.index.get_level_values("date").isin(dates)]
        return rows.dropna(subset=["target"]) if labelled else rows

    def fit(self, panel: pd.DataFrame, fold: Fold) -> "Model":
        raise NotImplementedError

    def score(self, panel: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, panel: pd.DataFrame, dates) -> pd.Series:
        rows = self._rows(panel, dates, labelled=False)
        return pd.Series(self.score(panel, rows) / self.scale_, index=rows.index, name=self.name)

    def _set_scale(self, panel: pd.DataFrame, dates) -> None:
        sd = float(np.std(self.score(panel, self._rows(panel, dates))))
        self.scale_ = sd if sd > 0 else 1.0


class EqualWeight(Model):
    """Mean of tanh(signed z-score) over the per-asset strategy signals.
    No weights are learned; the training data only sets the z-score scale."""

    name = "equal_weight"

    def fit(self, panel, fold):
        cols = list(SIGNAL_SIGNS)
        self.scaler = Scaler().fit(self._rows(panel, fold.fit_all)[cols])
        self.signs = pd.Series(SIGNAL_SIGNS)
        self._set_scale(panel, fold.fit_all)
        return self

    def score(self, panel, rows):
        z = self.scaler.transform(rows[list(SIGNAL_SIGNS)])
        return np.tanh(z * self.signs).mean(axis=1).to_numpy()


class RidgeModel(Model):
    name = "ridge"
    alphas = (1.0, 10.0, 100.0, 1_000.0, 10_000.0)

    def fit(self, panel, fold):
        from sklearn.linear_model import Ridge

        tr, va = self._rows(panel, fold.train), self._rows(panel, fold.val)
        sc = Scaler().fit(tr[self.features])
        errs = {}
        for a in self.alphas:
            m = Ridge(alpha=a).fit(sc.transform(tr[self.features]), tr["target"])
            errs[a] = float(np.mean((m.predict(sc.transform(va[self.features])) - va["target"]) ** 2))
        self.alpha_ = min(errs, key=errs.get)
        allr = self._rows(panel, fold.fit_all)
        self.scaler = Scaler().fit(allr[self.features])
        self.model = Ridge(alpha=self.alpha_).fit(self.scaler.transform(allr[self.features]), allr["target"])
        self._set_scale(panel, fold.fit_all)
        return self

    def score(self, panel, rows):
        return self.model.predict(self.scaler.transform(rows[self.features]))


class GradientBoosting(Model):
    """Shallow LightGBM: leaves chosen on the validation block, trees by early stopping."""

    name = "lightgbm"
    leaves = (4, 8)

    def fit(self, panel, fold):
        import warnings

        import lightgbm as lgb

        warnings.filterwarnings("ignore", module="lightgbm")

        tr, va = self._rows(panel, fold.train), self._rows(panel, fold.val)
        base = dict(learning_rate=0.03, n_estimators=300, min_child_samples=100, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, max_depth=3, verbose=-1, random_state=0)
        best = None
        for nl in self.leaves:
            m = lgb.LGBMRegressor(num_leaves=nl, **base)
            m.fit(tr[self.features], tr["target"], eval_set=[(va[self.features], va["target"])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            err = m.best_score_["valid_0"]["l2"]
            if best is None or err < best[0]:
                best = (err, nl, max(10, m.best_iteration_ or 10))
        _, self.leaves_, self.trees_ = best
        allr = self._rows(panel, fold.fit_all)
        self.model = lgb.LGBMRegressor(num_leaves=self.leaves_, **{**base, "n_estimators": self.trees_})
        self.model.fit(allr[self.features], allr["target"])
        self._set_scale(panel, fold.fit_all)
        return self

    def score(self, panel, rows):
        return self.model.predict(rows[self.features])


def _torch():
    import torch

    torch.set_num_threads(4)
    return torch


class MLP(Model):
    """Two hidden layers (32, 16), dropout, weight decay, early stopping on the
    validation block; the prediction is the average of `seeds` nets."""

    name = "mlp"

    def __init__(self, hidden=(32, 16), seeds: int = 10, epochs: int = 100, patience: int = 10,
                 lr: float = 1e-3, weight_decay: float = 1e-3, dropout: float = 0.1, batch: int = 1024):
        self.hidden, self.seeds, self.epochs, self.patience = hidden, seeds, epochs, patience
        self.lr, self.wd, self.dropout, self.batch = lr, weight_decay, dropout, batch

    def _net(self, d):
        nn = _torch().nn
        layers, prev = [], d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        return nn.Sequential(*layers, nn.Linear(prev, 1))

    def fit(self, panel, fold):
        torch = _torch()
        tr, va = self._rows(panel, fold.train), self._rows(panel, fold.val)
        self.scaler = Scaler().fit(tr[self.features])
        Xt = torch.tensor(self.scaler.transform(tr[self.features]).to_numpy(np.float32))
        yt = torch.tensor(tr["target"].to_numpy(np.float32))
        Xv = torch.tensor(self.scaler.transform(va[self.features]).to_numpy(np.float32))
        yv = torch.tensor(va["target"].to_numpy(np.float32))
        self.nets, self.epochs_used = [], []
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            net = self._net(Xt.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            best, best_state, bad, used = np.inf, None, 0, 0
            for ep in range(self.epochs):
                net.train()
                perm = torch.randperm(len(Xt))
                for i in range(0, len(Xt), self.batch):
                    idx = perm[i : i + self.batch]
                    opt.zero_grad()
                    loss = ((net(Xt[idx]).squeeze(1) - yt[idx]) ** 2).mean()
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(((net(Xv).squeeze(1) - yv) ** 2).mean())
                if v < best - 1e-5:
                    best, bad, used = v, 0, ep + 1
                    best_state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)
            self.epochs_used.append(used)
        self._set_scale(panel, fold.fit_all)
        return self

    def seed_scores(self, rows) -> np.ndarray:
        torch = _torch()
        X = torch.tensor(self.scaler.transform(rows[self.features]).to_numpy(np.float32))
        with torch.no_grad():
            return np.stack([net(X).squeeze(1).numpy() for net in self.nets], axis=1)

    def score(self, panel, rows):
        return self.seed_scores(rows).mean(axis=1)

    def predict_seeds(self, panel, dates) -> pd.DataFrame:
        """One column per seed, each scaled like the ensemble, for the seed-spread check."""
        rows = self._rows(panel, dates, labelled=False)
        return pd.DataFrame(self.seed_scores(rows) / self.scale_, index=rows.index)


class LSTMMomentum(Model):
    """Deep Momentum Network (Lim, Zohren and Roberts 2019) on weekly features.

    An LSTM reads each asset's last `window` weeks of features and outputs a
    position in [-1, 1] through tanh. Training maximises the annualised
    Sharpe ratio of position x vol-scaled return, minus `turnover_penalty`
    times the mean absolute change in position. Early stopping on validation
    Sharpe; the position is the average over `seeds` nets.
    """

    name = "lstm"

    def __init__(self, hidden: int = 16, window: int = 26, seeds: int = 5, epochs: int = 100, patience: int = 10,
                 lr: float = 1e-3, dropout: float = 0.1, batch: int = 64, turnover_penalty: float = 0.1):
        self.hidden, self.window, self.seeds, self.epochs, self.patience = hidden, window, seeds, epochs, patience
        self.lr, self.dropout, self.batch, self.turnover_penalty = lr, dropout, batch, turnover_penalty

    def _module(self, d):
        torch = _torch()
        nn = torch.nn
        hidden, dropout = self.hidden, self.dropout

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.drop = nn.Dropout(dropout)
                self.lstm = nn.LSTM(d, hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                h, _ = self.lstm(self.drop(x))
                return torch.tanh(self.head(h)).squeeze(-1)

        return Net()

    def _windows(self, panel, dates, stride=None):
        """(X windows, y windows, last-row index) per symbol over `dates`.
        With stride=None every row gets its own trailing window (for scoring)."""
        L = self.window
        Z = self.scaler.transform(panel[self.features]).to_numpy(np.float32)
        y = panel["target"].fillna(0.0).to_numpy(np.float32)
        sym = panel.index.get_level_values("symbol")
        dt = panel.index.get_level_values("date")
        want = dt.isin(dates)
        Xs, Ys, idx = [], [], []
        for s in pd.unique(sym):
            pos = np.flatnonzero(sym == s)
            pos = pos[np.argsort(dt[pos])]
            # zero rows in front so early weeks still get a full window
            zs = np.vstack([np.zeros((L - 1, Z.shape[1]), np.float32), Z[pos]])
            ys = np.concatenate([np.zeros(L - 1, np.float32), y[pos]])
            ends = np.flatnonzero(want[pos])
            if stride:
                ends = ends[::-1][::stride][::-1]
            for e in ends:
                Xs.append(zs[e : e + L])
                Ys.append(ys[e : e + L])
                idx.append(pos[e])
        if not Xs:
            return np.zeros((0, L, Z.shape[1]), np.float32), np.zeros((0, L), np.float32), np.array([], int)
        return np.stack(Xs), np.stack(Ys), np.array(idx)

    def _loss(self, pos, y):
        torch = _torch()
        cap = pos * y
        sharpe = cap.mean() / (cap.std() + 1e-6) * np.sqrt(52)
        turn = (pos[:, 1:] - pos[:, :-1]).abs().mean()
        return -sharpe + self.turnover_penalty * turn

    def fit(self, panel, fold):
        torch = _torch()
        labelled = panel.dropna(subset=["target"])
        self.scaler = Scaler().fit(self._rows(panel, fold.train)[self.features])
        Xw, Yw, _ = self._windows(labelled, fold.train, stride=self.window // 2)
        Xv, Yv, _ = self._windows(labelled, fold.val)
        Xw, Yw, Xv, Yv = map(torch.tensor, (Xw, Yw, Xv, Yv))
        self.nets = []
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            net = self._module(Xw.shape[2])
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            best, best_state, bad = -np.inf, None, 0
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(len(Xw))
                for i in range(0, len(Xw), self.batch):
                    b = perm[i : i + self.batch]
                    opt.zero_grad()
                    self._loss(net(Xw[b]), Yw[b]).backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                net.eval()
                with torch.no_grad():
                    cap = net(Xv)[:, -1] * Yv[:, -1]
                    v = float(cap.mean() / (cap.std() + 1e-6))
                if v > best + 1e-4:
                    best, bad = v, 0
                    best_state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)
        # A Sharpe loss does not care about scale, so put positions on the
        # same footing as the other models: unit standard deviation in fitting.
        self.scale_ = 1.0
        sd = float(self.predict_seeds(panel, fold.fit_all).mean(axis=1).std())
        self.scale_ = sd if sd > 0 else 1.0
        return self

    def predict_seeds(self, panel, dates) -> pd.DataFrame:
        torch = _torch()
        X, _, idx = self._windows(panel, dates)
        with torch.no_grad():
            out = np.stack([net(torch.tensor(X))[:, -1].numpy() for net in self.nets], axis=1)
        return pd.DataFrame(out / self.scale_, index=panel.index[idx]).sort_index()

    def predict(self, panel, dates):
        return self.predict_seeds(panel, dates).mean(axis=1).rename(self.name)


LADDER = {
    "equal_weight": EqualWeight,
    "ridge": RidgeModel,
    "lightgbm": GradientBoosting,
    "mlp": MLP,
    "lstm": LSTMMomentum,
}
