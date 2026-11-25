"""Drawing helpers shared by the docs generator and the Gradio demo."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from model.config import SAMPLE_RATE


def spectrogram_figure(wave: np.ndarray, detections=None, label_names=None, height: float = 2.6):
    """Draw a spectrogram: time left->right, pitch (frequency) bottom->top,
    loudness as brightness. Optional `detections` are drawn as coloured bars.

    Returns a matplotlib Figure (the caller saves or shows it)."""
    # 512-sample windows (32 ms) give a good balance of time vs pitch detail.
    f, t, S = spectrogram(wave, fs=SAMPLE_RATE, nperseg=512, noverlap=384)
    S_db = 10 * np.log10(S + 1e-10)                       # convert power to decibels
    vmax = np.percentile(S_db, 99.5)

    fig, ax = plt.subplots(figsize=(8, height))
    ax.pcolormesh(t, f / 1000, S_db, shading="auto", cmap="magma", vmin=vmax - 60, vmax=vmax)
    ax.set_ylabel("Pitch (kHz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 8)

    for d in detections or []:
        ax.axvspan(d.start, d.end, ymin=0.92, ymax=1.0, color="#66bb6a", alpha=0.9)
        name = (label_names or {}).get(d.label, d.label).split(" (")[0]
        ax.text(d.start + 0.05, 7.1, name, color="white", fontsize=8, va="top",
                bbox=dict(facecolor="#1b5e20", alpha=0.7, pad=1.5, edgecolor="none"))
    fig.tight_layout()
    return fig
