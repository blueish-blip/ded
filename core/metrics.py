"""Semua metrik dihitung dari prediksi model asli (scikit-learn)."""
import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_curve,
)

import config


def _num(x, nd=6):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _prf(y_true, y_pred, labels, average):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average, zero_division=0
    )
    return {"precision": _num(p), "recall": _num(r), "f1": _num(f)}


def compute_metrics(y_true, y_pred, y_prob, latencies_ms):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        raise ValueError("Tidak ada data untuk dihitung.")

    all_idx = list(range(len(config.LABELS)))
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))  # sama seperti default sklearn

    p_c, r_c, f_c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=all_idx, average=None, zero_division=0
    )
    per_class = [
        {
            "class": config.CLASS_NAMES[i],
            "precision": _num(p_c[i]),
            "recall": _num(r_c[i]),
            "f1": _num(f_c[i]),
            "support": int(sup[i]),
        }
        for i in all_idx
    ]

    cm = confusion_matrix(y_true, y_pred, labels=all_idx)

    roc, aucs = {}, []
    for k in all_idx:
        name = config.CLASS_NAMES[k]
        yt = (y_true == k).astype(int)
        if yt.sum() == 0 or yt.sum() == n:

            roc[name] = None  # AUC tidak terdefinisi bila hanya satu sisi
            continue
        fpr, tpr, _ = roc_curve(yt, y_prob[:, k])
        a = float(auc(fpr, tpr))
        aucs.append(a)
        roc[name] = {
            "fpr": [round(float(v), 5) for v in fpr],
            "tpr": [round(float(v), 5) for v in tpr],
            "auc": _num(a),
        }

    lat = np.asarray(latencies_ms, dtype=float)
    latency = {
        "mean": _num(lat.mean(), 3),
        "median": _num(np.median(lat), 3),
        "p95": _num(np.percentile(lat, 95), 3),
        "min": _num(lat.min(), 3),
        "max": _num(lat.max(), 3),
    }

    return {
        "n": int(n),
        "accuracy": _num(accuracy_score(y_true, y_pred)),
        "macro": _prf(y_true, y_pred, present, "macro"),
        "weighted": _prf(y_true, y_pred, present, "weighted"),
        "mcc": _num(matthews_corrcoef(y_true, y_pred)),
        "kappa": _num(cohen_kappa_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "roc": roc,
        "auc_macro": _num(np.mean(aucs)) if aucs else None,
        "latency_ms": latency,

        "classes_without_support": [config.CLASS_NAMES[i] for i in all_idx if int(sup[i]) == 0],
    }