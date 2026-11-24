# Quick Start

From a fresh clone to a working demo. Everything runs on a normal laptop CPU with **Python 3.11**.

## 1. Get the code and install

```bash
git clone https://github.com/ntoder/PrimateLM-audio.git
cd PrimateLM-audio
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download the data

```bash
python model/download_data.py
```

This downloads the openly licensed dataset into `data/` (which git ignores) and prints its license.

## 3. Train and evaluate

```bash
python model/train.py
```

This computes YAMNet embeddings for every clip, trains the classifier, and writes:

- `model/artifacts/classifier.joblib` — the trained classifier
- `model/artifacts/eval_report.md` — accuracy, per-class F1
- `model/artifacts/confusion_matrix.png`

## 4. Run the demo

```bash
python demo/app.py
```

Open the URL it prints (usually <http://127.0.0.1:7860>).

## 5. View this docs site locally

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Open <http://127.0.0.1:8000>.

!!! note "Work in progress"
    Steps 2–4 are being built now. Track progress in [PROGRESS.md](https://github.com/ntoder/PrimateLM-audio/blob/main/PROGRESS.md).
