from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def uar_f1(y_true, y_pred, n_labels):
    """Macro recall (UAR) and macro F1 over the classes present in y_true
    (all four in every real val / test split)."""
    present = sorted(set(np.asarray(y_true).tolist()))
    return {
        "uar": 100 * recall_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "f1": 100 * f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "n": len(y_true),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(n_labels))).tolist(),
    }


def per_language(y_true, y_pred, langs, n_labels):
    groups = defaultdict(list)
    for i, l in enumerate(langs):
        groups[l].append(i)
    out = {l: uar_f1(np.asarray(y_true)[idx], np.asarray(y_pred)[idx], n_labels) for l, idx in sorted(groups.items())}
    out["mean_uar"] = float(np.mean([m["uar"] for m in out.values()]))
    return out
