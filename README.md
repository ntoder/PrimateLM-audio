# PrimateLM-audio 🐒

A school robotics team project: a small, honest **primate-call classifier** with a documentation site, an interactive Gradio demo, and a Raspberry Pi live-detection script. Inspired by [NatureLM-audio](https://github.com/earthspecies/NatureLM-audio).

**Pipeline:** audio → 16 kHz → 2-second windows → [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) embeddings → small MLP classifier → label + confidence.

**Classes:** Northern grey gibbon · great argus pheasant · helmeted hornbill · rhinoceros hornbill · background.

**Results on held-out recordings:** 84.8% clip accuracy, macro-F1 0.72, gibbon F1 0.82. In a forest the model never heard: 80.4% accuracy. Full report: [`model/artifacts/eval_report.md`](model/artifacts/eval_report.md).

📖 Docs site: https://ntoder.github.io/PrimateLM-audio/ · 📋 Progress: [PROGRESS.md](PROGRESS.md)

---

## Setup from a fresh clone

You need **Python 3.11** and about 1.5 GB of free disk space (TensorFlow is ~600 MB; the dataset is ~400 MB). No GPU is needed.

```bash
git clone https://github.com/ntoder/PrimateLM-audio.git
cd PrimateLM-audio
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r demo/requirements.txt
```

`demo/requirements.txt` installs everything needed for training and the demo. Use `requirements.txt` if you only want to train.

### 1. Download the dataset (~400 MB)

```bash
python model/download_data.py
```

This checks the license from Zenodo (CC-BY-4.0), downloads, verifies checksums, and unzips into `data/` (git-ignored).

### 2. Train and evaluate (~4 minutes on a laptop CPU)

```bash
python model/train.py
```

This writes `model/artifacts/classifier.joblib`, `eval_report.md`, `metrics.json` and confusion-matrix images. The YAMNet embeddings are cached in `data/cache/`, so a second run takes ~10 seconds.

Optional: rebuild the Examples page from the new model.

```bash
python model/make_examples.py
```

### 3. Classify a file from the command line

```bash
python model/predict.py demo/examples/01_gibbon.flac
```

### 4. Run the web demo

```bash
python demo/app.py
```

Open http://127.0.0.1:7860. Upload or record audio, or click an example clip.

### 5. Run the robot detector (simulating a microphone with a file)

```bash
python robot/live_detect.py --wav demo/examples/01_gibbon.flac -v
```

On a Raspberry Pi with a USB mic, see [robot/README.md](robot/README.md).

### 6. View the docs site locally

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Open http://127.0.0.1:8000/PrimateLM-audio/.

---

## Deploying

**Docs → GitHub Pages:** on GitHub, open **Settings → Pages** and set **Source** to **GitHub Actions** (one time). After that, every push to `main` rebuilds the site with `.github/workflows/docs.yml`.

**Demo → Hugging Face Spaces (free CPU):**

```bash
python demo/prepare_space.py
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload <your-username>/PrimateLM-audio build/space . --repo-type=space
```

The first command builds `build/space/`; then create a Gradio Space on huggingface.co and upload it.

## Repo layout

```
docs/                 MkDocs site (9 pages) + assets
model/
  config.py           shared settings: labels, sample rate, window size, thresholds
  download_data.py    verified dataset download
  dataset.py          find clips, split by recording
  embed.py            audio loading + YAMNet embeddings
  train.py            train + evaluate + write report
  evaluate.py         metrics and confusion matrices
  predict.py          use the model (CLI + library for demo/robot)
  make_examples.py    builds the docs Examples page
  plots.py            spectrogram drawing
  yamnet/             YAMNet model code (Apache-2.0, from tensorflow/models)
  artifacts/          trained classifier + evaluation report (committed)
demo/                 Gradio app, example clips, Hugging Face Space builder
robot/                Raspberry Pi live detector (stdout / MQTT / HTTP)
```

## Credits

- **Data:** Clink, D. J. & Hamid Ahmad, A. (2024). *A labelled dataset of the loud calls of four vertebrates collected using passive acoustic monitoring in Malaysian Borneo.* Zenodo. [doi:10.5281/zenodo.14213067](https://doi.org/10.5281/zenodo.14213067). CC-BY-4.0. The example clips in `demo/examples/` and `docs/assets/examples/` are 10-second excerpts of this dataset.
- **YAMNet:** Google, Apache-2.0.
- **Inspiration and site structure:** NatureLM-audio by the Earth Species Project.
