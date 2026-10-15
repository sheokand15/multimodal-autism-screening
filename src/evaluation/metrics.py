"""Dependency-light classification metrics."""
from __future__ import annotations
import numpy as np


def classification_metrics(y_true, y_pred, classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_i = {label: i for i, label in enumerate(classes)}
    for truth, pred in zip(y_true, y_pred):
        cm[class_to_i[truth], class_to_i[pred]] += 1
    per_class = {}
    for i, label in enumerate(classes):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp); fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {"precision": precision, "recall": recall, "f1": f1, "support": int(cm[i, :].sum())}
    return {"n": int(len(y_true)), "accuracy": float(np.mean(y_true == y_pred)),
            "balanced_accuracy": float(np.mean([row["recall"] for row in per_class.values()])),
            "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
            "per_class": per_class, "confusion_matrix": cm.tolist(), "class_order": [str(c) for c in classes]}


def accuracy_wilson_ci(y_true, y_pred, z=1.959963984540054):
    """Wilson score interval for test accuracy; describes sampling uncertainty only."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n == 0: return [None, None]
    p = float(np.mean(y_true == y_pred))
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0, center - radius)), float(min(1, center + radius))]


def binary_curves(y_true, positive_scores, positive_label="Yes"):
    """Empirical ROC/precision-recall points and areas for a binary test set."""
    y_true, scores = np.asarray(y_true), np.asarray(positive_scores, dtype=float)
    positive = y_true == positive_label
    order = np.argsort(-scores, kind="mergesort")
    positives, negatives = int(positive.sum()), int((~positive).sum())
    tp = fp = 0
    roc = [[0.0, 0.0]]
    pr = [[0.0, 1.0]]
    old_score = None
    for index in order:
        if old_score is not None and scores[index] != old_score:
            roc.append([fp / negatives if negatives else 0.0, tp / positives if positives else 0.0])
            pr.append([tp / positives if positives else 0.0, tp / (tp + fp) if tp + fp else 1.0])
        if positive[index]: tp += 1
        else: fp += 1
        old_score = scores[index]
    roc.append([1.0, 1.0])
    pr.append([1.0, tp / (tp + fp) if tp + fp else 0.0])
    roc_auc = float(sum((roc[i + 1][0] - roc[i][0]) * (roc[i + 1][1] + roc[i][1]) / 2 for i in range(len(roc) - 1)))
    # Average precision uses recall increments with precision at each threshold.
    average_precision = float(sum((pr[i + 1][0] - pr[i][0]) * pr[i + 1][1] for i in range(len(pr) - 1)))
    return {"roc_auc": roc_auc, "average_precision": average_precision,
            "roc_points_fpr_tpr": roc, "pr_points_recall_precision": pr}
