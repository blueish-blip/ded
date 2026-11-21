"""Grafik matplotlib (confusion matrix, ROC) + ekspor PNG."""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def confusion_matrix_figure(cm, labels, normalize=False):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        data = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
        vmax = 1.0
    else:
        data = cm
        vmax = max(1.0, float(cm.max()))

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label asli")
    ax.set_title("Confusion Matrix" + (" (dinormalisasi per baris)" if normalize else ""))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{int(cm[i, j])}"

            if normalize:
                text += f"\n{data[i, j]:.1%}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if data[i, j] > vmax / 2 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def roc_figure(roc, labels):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    drawn = 0
    for name in labels:
        item = roc.get(name)
        if not item:
            continue
        ax.plot(item["fpr"], item["tpr"], lw=2, label=f"{name} (AUC = {item['auc']:.3f})")
        drawn += 1
    if drawn == 0:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC One-vs-Rest per kelas")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def close(fig):
    plt.close(fig)