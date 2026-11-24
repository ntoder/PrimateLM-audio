"""Shared settings for PrimateLM-audio.

Every part of the project (training, demo, robot) imports these values, so the
model always sees audio prepared in exactly the same way.
"""
from pathlib import Path

# ---- Folders -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent        # the repo folder
DATA_DIR = ROOT / "data"                             # downloaded audio (git-ignored)
RAW_DIR = DATA_DIR / "raw"                           # unzipped dataset files
CACHE_DIR = DATA_DIR / "cache"                       # saved YAMNet embeddings
ARTIFACTS_DIR = ROOT / "model" / "artifacts"         # trained model + reports (committed)
YAMNET_DIR = DATA_DIR / "yamnet"                     # YAMNet weights (downloaded)

# ---- Audio ---------------------------------------------------------------
SAMPLE_RATE = 16_000      # YAMNet expects 16,000 samples per second, mono
WINDOW_SECONDS = 2.0      # the model looks at 2 seconds at a time
HOP_SECONDS = 1.0         # sliding window step: 1 s = 50% overlap
DETECTION_THRESHOLD = 0.6 # below this confidence we say "uncertain"

# ---- Dataset -------------------------------------------------------------
# Borneo loud-calls dataset, Clink & Hamid Ahmad, Zenodo, CC-BY-4.0.
ZENODO_RECORD = "14213067"
ZENODO_FILES = ["trainingclips.zip", "testclipsmaliau.zip"]

# Where the unzipped clips live. Each has one sub-folder per class.
TRAIN_CLIPS_DIR = RAW_DIR / "trainingclips" / "trainingclips"     # Danum Valley, 2018
EXTRA_TEST_DIR = RAW_DIR / "testclipsmaliau" / "testclipsmaliau"   # Maliau Basin, 2019 (different forest!)

# Dataset folder name -> short label used by the model.
FOLDER_TO_LABEL = {
    "female.gibbon": "gibbon",
    "long.argus": "great_argus",
    "hornbill.helmeted": "helmeted_hornbill",
    "hornbill.rhino": "rhinoceros_hornbill",
    "noise": "background",
}
LABELS = list(FOLDER_TO_LABEL.values())   # fixed order used everywhere

# Short label -> friendly name for the demo, docs and robot output.
LABEL_NAMES = {
    "gibbon": "Northern grey gibbon (female) 🐒",
    "great_argus": "Great argus pheasant",
    "helmeted_hornbill": "Helmeted hornbill",
    "rhinoceros_hornbill": "Rhinoceros hornbill",
    "background": "Background / no call",
}
PRIMATE_LABELS = {"gibbon"}   # which labels count as "primate detected"
