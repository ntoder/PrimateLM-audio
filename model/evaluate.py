"""Scoring helpers: accuracy, per-class F1 and confusion-matrix pictures.

Words you'll see:
  * precision - when the model says "gibbon", how often is it right?
  * recall    - of all the real gibbon clips, how many did the model find?
  * F1        - one number that balances precision and recall (1.0 = perfect)
  * macro F1  - the average F1 over classes, so small classes count as much as big ones
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw images to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def scores(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Compute the numbers we report. Only classes present in y_true get a row."""
    present = [l for l in labels if l in set(y_true)]
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=present, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                            "support": int(n[i])} for i, lab in enumerate(present)},
    }


def plot_confusion(y_true, y_pred, labels: list[str], title: str, path: Path) -> None:
    """Save a confusion matrix: rows = true class, columns = predicted class.
    Each row is shown as percentages, with the raw count underneath."""
    rows = [l for l in labels if l in set(y_true)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm[[labels.index(r) for r in rows]]
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2.5, 1.0 * len(rows) + 2))
    ax.imshow(pct, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), [l.replace("_", "\n") for l in labels], fontsize=9)
    ax.set_yticks(range(len(rows)), [l.replace("_", " ") for l in rows], fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=11)
    for i in range(len(rows)):
        for j in range(len(labels)):
            color = "white" if pct[i, j] > 0.6 else "black"
            ax.text(j, i, f"{pct[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown_table(s: dict, label_names: dict[str, str]) -> str:
    """Turn a scores() dict into a Markdown table for the report and the docs."""
    lines = ["| Class | Precision | Recall | F1 | Clips |", "|---|---|---|---|---|"]
    for lab, v in s["per_class"].items():
        lines.append(f"| {label_names.get(lab, lab)} | {v['precision']:.2f} | {v['recall']:.2f} "
                     f"| **{v['f1']:.2f}** | {v['support']} |")
    return "\n".join(lines)


def clip_predictions(window_probs: np.ndarray, clip_ids: np.ndarray, classes: np.ndarray):
    """Combine window predictions into one prediction per clip by averaging
    the probabilities of all windows in that clip."""
    out_ids, out_pred = [], []
    for cid in np.unique(clip_ids):
        mean = window_probs[clip_ids == cid].mean(axis=0)
        out_ids.append(cid)
        out_pred.append(classes[mean.argmax()])
    return np.array(out_ids), np.array(out_pred)
