"""
Imbalance-aware evaluation.

Because bleaching alerts are rare, **accuracy is misleading** (a model that always
predicts "no alert" scores high while being useless). We therefore lead with:

* **macro-F1** and macro precision/recall — every class counts equally,
* a full **per-class** precision/recall/F1 report,
* the **confusion matrix**, and
* for the binary task, the **recall and precision on actual alerts** — the
  operationally meaningful numbers (did we catch the bleaching events?).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import config


def evaluate(y_true, y_pred, target_mode: str | None = None) -> dict:
    """Return a JSON-serialisable dict of imbalance-aware metrics."""
    target_mode = config.TARGET_MODE if target_mode is None else target_mode
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics: dict = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "n_test": int(len(y_true)),
        "support_positive": int((y_true >= config.BAA_ALERT_THRESHOLD).sum())
        if target_mode == "multiclass"
        else int((y_true == 1).sum()),
    }

    # Per-class report (precision/recall/F1/support keyed by class label).
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    metrics["per_class"] = report

    if target_mode == "binary":
        # Alert == positive class (1). Report its recall/precision explicitly.
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], average=None, zero_division=0
        )
        metrics["alert_recall"] = float(r[0])
        metrics["alert_precision"] = float(p[0])
        metrics["alert_f1"] = float(f1[0])
    else:
        metrics["weighted_f1"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
    return metrics


def confusion_df(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows=true, cols=predicted)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(np.unique(y_true)).union(np.unique(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    index = [f"true_{l}" for l in labels]
    cols = [f"pred_{l}" for l in labels]
    return pd.DataFrame(cm, index=index, columns=cols)


def text_report(y_true, y_pred, target_mode: str | None = None) -> str:
    """A human-readable per-class report plus headline macro-F1."""
    target_mode = config.TARGET_MODE if target_mode is None else target_mode
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if target_mode == "binary":
        names = ["no-alert", "ALERT"]
        target_names = [names[i] for i in sorted(np.unique(np.concatenate([y_true, y_pred])))] \
            if set(np.unique(np.concatenate([y_true, y_pred]))) <= {0, 1} else None
    else:
        target_names = None

    rep = classification_report(y_true, y_pred, zero_division=0, target_names=target_names)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return f"{rep}\n  macro-F1 = {macro_f1:.3f}\n"
