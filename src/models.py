"""
Two complementary classifiers, both **imbalance-aware**.

* :func:`train_xgboost` — gradient-boosted trees (XGBoost). For the binary task we
  set ``scale_pos_weight`` to up-weight the rare alert class; for multiclass we
  pass per-sample ``balanced`` weights.
* :class:`TorchMLP` — a small PyTorch multilayer perceptron wrapped in an
  sklearn-style ``fit``/``predict``/``predict_proba`` API, with internal feature
  standardisation and a **class-weighted** cross-entropy loss.

Heavy deps (``xgboost``, ``torch``) are imported lazily inside the functions so
this module can be imported — and the test suite can run with a scikit-learn
stand-in — even when they are not installed.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Windows OpenMP load-order workaround -- MUST run before scikit-learn/xgboost.
# PyTorch, scikit-learn and XGBoost each bundle their own OpenMP runtime. On
# Windows, importing scikit-learn or XGBoost *first* can leave PyTorch unable to
# initialise ``c10.dll`` (OSError WinError 1114). Importing torch first lets them
# coexist, so we preload it here at the very top of the module, before any other
# numerical import. train.py imports this module before it imports evaluate.py
# (which pulls scikit-learn), so the global order is always torch-first.
# Best-effort + wrapped: the module stays importable without torch, and the test
# suite (which never imports this module) uses a scikit-learn stand-in.
# --------------------------------------------------------------------------- #
try:  # noqa: SIM105
    import torch  # noqa: F401
except Exception:  # noqa: BLE001
    pass

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

import config


def _as_float_array(X) -> np.ndarray:
    return X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #
def train_xgboost(Xtr, ytr):
    """Fit an imbalance-aware :class:`xgboost.XGBClassifier`."""
    from xgboost import XGBClassifier

    ytr = np.asarray(ytr)
    classes = np.unique(ytr)
    n_classes = len(classes)

    params = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=config.RANDOM_STATE,
        n_jobs=0,
        eval_metric="logloss" if n_classes <= 2 else "mlogloss",
        tree_method="hist",
    )

    sample_weight = None
    if n_classes <= 2:
        # scale_pos_weight = (#negatives / #positives) up-weights the rare alerts.
        n_pos = int((ytr == 1).sum())
        n_neg = int((ytr == 0).sum())
        params["scale_pos_weight"] = (n_neg / n_pos) if n_pos > 0 else 1.0
    else:
        # Balanced per-sample weights for the multiclass case.
        cw = compute_class_weight("balanced", classes=classes, y=ytr)
        weight_map = dict(zip(classes, cw))
        sample_weight = np.array([weight_map[v] for v in ytr], dtype=np.float32)

    model = XGBClassifier(**params)
    model.fit(_as_float_array(Xtr), ytr, sample_weight=sample_weight)
    return model


# --------------------------------------------------------------------------- #
# PyTorch MLP (sklearn-style wrapper)
# --------------------------------------------------------------------------- #
class TorchMLP:
    """
    A small fully-connected network with an sklearn-like interface.

    Pipeline: ``StandardScaler`` -> Linear/ReLU stack -> softmax head, trained with
    a class-weighted ``CrossEntropyLoss`` so rare bleaching alerts are not ignored.
    """

    def __init__(self, hidden=None, epochs=None, lr=None, random_state=None):
        self.hidden = tuple(hidden or config.TORCH_HIDDEN)
        self.epochs = int(epochs or config.TORCH_EPOCHS)
        self.lr = float(lr or config.TORCH_LR)
        self.random_state = config.RANDOM_STATE if random_state is None else random_state
        self.classes_: np.ndarray | None = None
        self._model = None
        self._scaler = None

    def _build(self, n_features: int, n_classes: int):
        import torch.nn as nn

        layers: list = []
        prev = n_features
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = _as_float_array(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        # Map labels to 0..K-1 contiguous indices for CrossEntropyLoss.
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self._class_to_idx[v] for v in y], dtype=np.int64)

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X).astype(np.float32)

        # Class-weighted loss -> rare alerts contribute more to the gradient.
        cw = compute_class_weight("balanced", classes=self.classes_, y=y)
        weight = torch.tensor(cw, dtype=torch.float32)

        self._model = self._build(Xs.shape[1], n_classes)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-4)

        Xt = torch.from_numpy(Xs)
        yt = torch.from_numpy(y_idx)

        self._model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self._model(Xt)
            loss = criterion(logits, yt)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch

        self._model.eval()
        Xs = self._scaler.transform(_as_float_array(X)).astype(np.float32)
        with torch.no_grad():
            logits = self._model(torch.from_numpy(Xs))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X) -> np.ndarray:
        idx = self.predict_proba(X).argmax(axis=1)
        return self.classes_[idx]


def train_torch_mlp(Xtr, ytr) -> TorchMLP:
    """Fit and return a :class:`TorchMLP`."""
    return TorchMLP().fit(Xtr, ytr)
