"""Find the audio clips on disk and split them into train / validation / test.

Each clip's label comes from its folder name, e.g. female.gibbon/ -> "gibbon".

IMPORTANT idea - splitting by RECORDING, not by clip:
Clips are cut from 2-hour field recordings. Two clips from the same recording
share the same background hiss, weather and recorder. If one went into training
and the other into testing, the model could "recognise the recording" instead of
the animal, and our test score would look better than it really is. So every
clip from one recording goes into the same split.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import EXTRA_TEST_DIR, FOLDER_TO_LABEL, TRAIN_CLIPS_DIR  # noqa: E402

SEED = 42  # fixed random seed so the split is the same every time

# Filenames look like: female.gibbon_S10_20180620_080003_1236.64_1254.21_.wav
#   recorder unit = S10, recording = S10_20180620_080003 (one 2-hour file)
RECORDING_RE = re.compile(r"_([A-Z]\d+_\d{8}_\d{6})_")


@dataclass
class Clip:
    path: Path
    label: str
    recording: str   # which 2-hour field recording the clip was cut from


def list_clips(folder: Path, extra_folder_names: dict[str, str] | None = None) -> list[Clip]:
    """Return every .wav clip under folder/<class>/ with its label and recording id."""
    mapping = {**FOLDER_TO_LABEL, **(extra_folder_names or {})}
    clips = []
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        if class_dir.name not in mapping:
            print(f"  (skipping unknown folder {class_dir.name})")
            continue
        # skip hidden macOS "._" metadata files that came inside the zip
        for wav in sorted(p for p in class_dir.glob("*.wav") if not p.name.startswith(".")):
            m = RECORDING_RE.search(wav.name)
            clips.append(Clip(wav, mapping[class_dir.name], m.group(1) if m else wav.stem))
    return clips


def training_clips() -> list[Clip]:
    """The main dataset (Danum Valley, 2018)."""
    return list_clips(TRAIN_CLIPS_DIR)


def maliau_clips() -> list[Clip]:
    """Extra test set from a DIFFERENT forest (Maliau Basin, 2019).
    Its gibbon folder is called 'female.gibbon' and some clips are named 'duet'
    (male + female gibbons singing together) - both count as 'gibbon'."""
    return list_clips(EXTRA_TEST_DIR)


def split_by_recording(clips: list[Clip]) -> dict[str, list[Clip]]:
    """Split into about 70% train / 15% validation / 15% test, keeping each
    recording in one split and keeping class proportions similar in each split."""
    labels = np.array([c.label for c in clips])
    groups = np.array([c.recording for c in clips])
    # 7 folds of ~14% each: fold 0 -> test, fold 1 -> validation, folds 2-6 -> train
    folds = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    fold_of = np.empty(len(clips), dtype=int)
    for k, (_, idx) in enumerate(folds.split(np.zeros(len(clips)), labels, groups)):
        fold_of[idx] = k
    return {
        "test": [c for c, f in zip(clips, fold_of) if f == 0],
        "val": [c for c, f in zip(clips, fold_of) if f == 1],
        "train": [c for c, f in zip(clips, fold_of) if f >= 2],
    }
