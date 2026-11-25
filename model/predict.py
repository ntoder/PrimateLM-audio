"""Use the trained classifier on new audio.

As a command:
    python model/predict.py path/to/recording.wav

As a library (the demo and robot do this):
    from model.predict import PrimateClassifier
    clf = PrimateClassifier()
    result = clf.predict_wave(wave)   # wave = 16 kHz mono numpy array
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import ARTIFACTS_DIR, DETECTION_THRESHOLD, MIN_WINDOWS_IN_A_ROW  # noqa: E402
from model.embed import YamnetEmbedder, load_audio  # noqa: E402


@dataclass
class WindowPrediction:
    start: float
    end: float
    label: str          # top label for this window
    confidence: float   # probability of the top label (0..1)
    probs: dict         # every label -> probability


@dataclass
class Detection:
    """A stretch of time where one (non-background) label was detected."""
    label: str
    start: float
    end: float
    confidence: float   # highest window confidence inside the stretch


def fmt_time(seconds: float) -> str:
    """12.0 -> '0:12'"""
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


class PrimateClassifier:
    def __init__(self, model_path: Path = ARTIFACTS_DIR / "classifier.joblib"):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.classes = list(self.model.classes_)
        self.label_names = bundle["label_names"]
        self.window_s = bundle["window_seconds"]
        self.hop_s = bundle["hop_seconds"]
        self.embedder = YamnetEmbedder()

    def nice(self, label: str) -> str:
        return self.label_names.get(label, label)

    def predict_windows(self, wave: np.ndarray) -> list[WindowPrediction]:
        """Classify every 2-second window (1-second step) of a 16 kHz waveform."""
        windows = self.embedder.windows(wave, self.window_s, self.hop_s)
        probs = self.model.predict_proba(np.stack([w.embedding for w in windows]))
        out = []
        for w, p in zip(windows, probs):
            i = int(p.argmax())
            out.append(WindowPrediction(w.start, w.end, self.classes[i], float(p[i]),
                                        dict(zip(self.classes, map(float, p)))))
        return out

    def overall(self, windows: list[WindowPrediction]) -> dict[str, float]:
        """One score per label for the whole recording: the average of the window
        probabilities. This is exactly how the per-clip test scores were measured.
        (For long recordings, look at detections() instead - a short call can
        disappear in an average.)"""
        mean = {c: float(np.mean([w.probs[c] for w in windows])) for c in self.classes}
        return dict(sorted(mean.items(), key=lambda kv: -kv[1]))

    def detections(self, windows: list[WindowPrediction],
                   threshold: float = DETECTION_THRESHOLD,
                   min_windows: int = MIN_WINDOWS_IN_A_ROW) -> list[Detection]:
        """Turn window predictions into a list of detections.

        1. Keep windows whose top label is not 'background' and whose confidence
           is at least `threshold`.
        2. Merge back-to-back kept windows with the same label into one stretch,
           e.g. gibbon at 4-6 s and 5-7 s -> gibbon at 0:04-0:07.
        3. Throw away stretches made of fewer than `min_windows` windows. On our
           test set this halves false alarms on background audio.
        4. Join surviving stretches of the same label that touch or overlap
           (a one-window dip in the middle of a long call shouldn't split it in two).
        """
        runs: list[tuple[Detection, int]] = []   # (detection, number of windows in it)
        prev = None
        for w in windows:
            keep = w.label != "background" and w.confidence >= threshold
            if keep and prev is not None and runs and runs[-1][0].label == w.label \
                    and w.start <= runs[-1][0].end:
                det, n = runs[-1]
                det.end = w.end
                det.confidence = max(det.confidence, w.confidence)
                runs[-1] = (det, n + 1)
            elif keep:
                runs.append((Detection(w.label, w.start, w.end, w.confidence), 1))
            prev = w if keep else None

        merged: list[Detection] = []
        for d in (d for d, n in runs if n >= min_windows):
            if merged and merged[-1].label == d.label and d.start <= merged[-1].end:
                merged[-1].end = max(merged[-1].end, d.end)
                merged[-1].confidence = max(merged[-1].confidence, d.confidence)
            else:
                merged.append(d)
        return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify primate calls in an audio file.")
    ap.add_argument("audio", help="WAV/MP3/FLAC file")
    ap.add_argument("--threshold", type=float, default=DETECTION_THRESHOLD)
    ap.add_argument("--min-windows", type=int, default=MIN_WINDOWS_IN_A_ROW)
    args = ap.parse_args()

    clf = PrimateClassifier()
    wave = load_audio(args.audio)
    windows = clf.predict_windows(wave)

    print(f"\n{args.audio}  ({len(wave) / 16000:.1f} s, {len(windows)} windows)\n")
    print("Top 3 for the whole recording:")
    for label, p in list(clf.overall(windows).items())[:3]:
        print(f"  {clf.nice(label):38s} {p:5.1%}  {'#' * int(p * 30)}")

    dets = clf.detections(windows, args.threshold, args.min_windows)
    print(f"\nDetections (confidence >= {args.threshold}, >= {args.min_windows} windows in a row):")
    for d in dets:
        print(f"  {clf.nice(d.label)} at {fmt_time(d.start)}-{fmt_time(d.end)}  (max {d.confidence:.0%})")
    if not dets:
        print("  none - only background or uncertain windows")


if __name__ == "__main__":
    main()
