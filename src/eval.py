from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, Union, Dict, Optional

from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def proba(model, X) -> np.ndarray:
    """Return positive-class probabilities/scores in [0,1] for any sklearn-like model."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(X).astype(float)


def _k_to_int(k: Union[int, float], n: int) -> int:
    """Accept k as count or fraction; clamp to [1, n]."""
    if isinstance(k, float):
        k = int(np.ceil(k * n))
    return max(1, min(int(k), n))

def precision_recall_at_k(
    y_true: Iterable[int],
    y_score: Iterable[float],
    k: Union[int, float] = 0.1,
) -> Tuple[float, float, float]:
    """Precision@k, Recall@k, and the score threshold used at k."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    kk = _k_to_int(k, n)
    order = np.argsort(-y_score)
    top_idx = order[:kk]
    thr = y_score[order[kk-1]]
    tp = y_true[top_idx].sum()
    precision = tp / kk
    recall = tp / max(1, y_true.sum())
    return float(precision), float(recall), float(thr)


def evaluate_at_ks(
    y_true: Iterable[int],
    y_score: Iterable[float],
    ks: Iterable[Union[int, float]] = (0.01, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Table with precision@k / recall@k across several cutoffs."""
    rows = []
    for k in ks:
        p, r, thr = precision_recall_at_k(y_true, y_score, k)
        rows.append({"k": k, "precision@k": p, "recall@k": r, "threshold": thr})
    return pd.DataFrame(rows)

def plot_lift_gain(y_true, y_score, bins: int = 10, ax=None):
    """Cumulative gain + lift in one compact chart."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = y_true.sum()

    parts = np.array_split(y_sorted, bins)
    cum_pos = np.cumsum([p.sum() for p in parts])
    perc_pop = np.linspace(1/bins, 1, bins)
    gain = cum_pos / max(1, n_pos)
    lift = gain / perc_pop

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(perc_pop, gain, marker="o", label="Cumulative gain")
    ax.plot(perc_pop, perc_pop, "--", label="Baseline")
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Fraction of positives captured")
    ax2 = ax.twinx()
    ax2.plot(perc_pop, lift, color="tab:red", marker="s", label="Lift")
    ax2.set_ylabel("Lift")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax

def expected_calibration_error(
    y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 10
) -> float:
    """Equal-width ECE; lower is better."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        w = m.mean()
        ece += w * abs(acc - conf)
    return float(ece)


def calibration_report(y_true, y_prob, n_bins: int = 10, title: Optional[str] = None):
    """Print Brier & ECE and draw a reliability plot."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    print(f"Brier score: {brier:.4f}  |  ECE@{n_bins}: {ece:.4f}")

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()
    return {"brier": brier, "ece": ece}

def eval_model(model, X_tr, y_tr, X_te, y_te, name="model") -> Dict[str, float]:
    """Fit -> predict -> return ROC-AUC & PR-AUC (use proba/decision_function)."""
    model.fit(X_tr, y_tr)
    p = proba(model, X_te)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_te, p),
        "pr_auc": average_precision_score(y_te, p),
    }


def cv_roc_auc(model, X, y, n_splits: int = 5, seed: int = 42) -> float:
    """Quick mean ROC-AUC via stratified CV."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return float(cross_val_score(model, X, y, scoring="roc_auc", cv=cv).mean())

__all__ = [
    "proba",
    "precision_recall_at_k",
    "evaluate_at_ks",
    "plot_lift_gain",
    "expected_calibration_error",
    "calibration_report",
    "eval_model",
    "cv_roc_auc",
]