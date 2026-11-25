"""Turn audio into YAMNet embeddings.

This file is shared by training, the demo and the robot, so audio is always
prepared the same way.

Key ideas for beginners:
  * Audio is just a long list of numbers (the "waveform"). At 16 kHz there are
    16,000 numbers per second.
  * YAMNet is a pretrained neural network. For every ~0.96 s of audio (moving
    0.48 s at a time) it outputs an "embedding": 1,024 numbers that describe
    what that bit of sound is like.
  * Our classifier looks at 2-second WINDOWS. For each window we average the
    YAMNet embeddings that fall inside it, giving one 1,024-number vector.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's chatty log lines

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import HOP_SECONDS, SAMPLE_RATE, WINDOW_SECONDS, YAMNET_DIR  # noqa: E402

YAMNET_WEIGHTS_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
YAMNET_FRAME_HOP = 0.48   # seconds between YAMNet frames (fixed by YAMNet's design)
YAMNET_FRAME_LEN = 0.96   # seconds of audio each YAMNet frame looks at


# ---------------------------------------------------------------------------
# 1. Loading audio
# ---------------------------------------------------------------------------
def to_model_format(wave: np.ndarray, sr: int) -> np.ndarray:
    """Convert any waveform to what YAMNet wants: mono, float32, 16 kHz, range -1..1."""
    wave = np.asarray(wave)
    if np.issubdtype(wave.dtype, np.integer):
        # Microphones and Gradio often give int16 numbers (-32768..32767); scale to -1..1.
        wave = wave / np.iinfo(wave.dtype).max
    wave = wave.astype(np.float32)
    if wave.ndim == 2:                       # stereo (samples, channels) -> mono
        wave = wave.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Resample with a polyphase filter, e.g. 44,100 Hz -> 16,000 Hz.
        g = gcd(sr, SAMPLE_RATE)
        wave = resample_poly(wave, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return wave


def load_audio(path: str | Path) -> np.ndarray:
    """Read a WAV/FLAC/OGG/MP3 file and return a 16 kHz mono waveform."""
    wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return to_model_format(wave, sr)


# ---------------------------------------------------------------------------
# 2. Running YAMNet
# ---------------------------------------------------------------------------
def ensure_yamnet_weights() -> Path:
    """Download YAMNet's pretrained weights (15 MB) the first time we need them."""
    path = YAMNET_DIR / "yamnet.h5"
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading YAMNet weights to {path} ...")
        urllib.request.urlretrieve(YAMNET_WEIGHTS_URL, path)
    return path


@dataclass
class Window:
    """One 2-second slice of audio and its embedding."""
    start: float            # seconds from the start of the file
    end: float
    embedding: np.ndarray   # shape (1024,)


class YamnetEmbedder:
    """Loads YAMNet once, then turns waveforms into embeddings."""

    def __init__(self) -> None:
        from model.yamnet import params as yamnet_params
        from model.yamnet import yamnet as yamnet_lib

        self.model = yamnet_lib.yamnet_frames_model(yamnet_params.Params())
        self.model.load_weights(str(ensure_yamnet_weights()))

    def frame_embeddings(self, wave: np.ndarray) -> np.ndarray:
        """Return YAMNet's per-frame embeddings, shape (n_frames, 1024)."""
        # Pad very short audio so YAMNet produces at least one full frame.
        min_len = int(YAMNET_FRAME_LEN * SAMPLE_RATE)
        if len(wave) < min_len:
            wave = np.pad(wave, (0, min_len - len(wave)))
        _scores, embeddings, _spectrogram = self.model(wave)
        return embeddings.numpy()

    def windows(self, wave: np.ndarray,
                window_s: float = WINDOW_SECONDS, hop_s: float = HOP_SECONDS) -> list[Window]:
        """Cut audio into overlapping windows and give each one an embedding.

        We run YAMNet ONCE over the whole file (fast), then for each window we
        average the YAMNet frames whose centre falls inside that window.
        """
        frames = self.frame_embeddings(wave)
        centres = np.arange(len(frames)) * YAMNET_FRAME_HOP + YAMNET_FRAME_LEN / 2
        duration = max(len(wave) / SAMPLE_RATE, window_s)

        out: list[Window] = []
        start = 0.0
        while start + window_s <= duration + 1e-6:
            inside = (centres >= start) & (centres < start + window_s)
            if inside.any():
                out.append(Window(start, start + window_s, frames[inside].mean(axis=0)))
            start += hop_s
        if not out:   # safety net: very short audio -> one window using every frame
            out.append(Window(0.0, duration, frames.mean(axis=0)))
        return out
