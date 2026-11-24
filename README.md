# PrimateLM-audio

A school robotics team project: a small, honest **primate-vocalization classifier** with a documentation site, an interactive Gradio demo, and a Raspberry Pi live-detection script. Inspired by [NatureLM-audio](https://github.com/earthspecies/NatureLM-audio).

Pipeline: audio → 16 kHz → YAMNet embeddings → small classifier → label + confidence.

> 🚧 Work in progress — see [PROGRESS.md](PROGRESS.md).

## Repo layout

```
docs/      MkDocs site content          model/   data download, training, evaluation
demo/      Gradio app                   robot/   Raspberry Pi live detector
mkdocs.yml requirements-docs.txt        .github/workflows/docs.yml  (GitHub Pages)
```

## Docs site (works now)

```bash
git clone https://github.com/ntoder/PrimateLM-audio.git
cd PrimateLM-audio
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-docs.txt
mkdocs serve
```

Then open http://127.0.0.1:8000.

To publish: on GitHub go to **Settings → Pages → Source: GitHub Actions**. Every push to `main` then rebuilds the site.

Model, demo and robot setup commands will be added here as each step is finished.
