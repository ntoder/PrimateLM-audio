"""Build a ready-to-upload Hugging Face Space folder at build/space/.

A Space needs: app.py, requirements.txt and a README.md with settings at the top.
This script copies only what the demo needs (no dataset, no docs):

    python demo/prepare_space.py

Then upload it (needs a free Hugging Face account and `pip install huggingface_hub`):

    huggingface-cli login
    huggingface-cli upload <your-username>/PrimateLM-audio build/space . --repo-type=space
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "space"

SPACE_README = """---
title: PrimateLM-audio
emoji: 🐒
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: demo/app.py
pinned: false
license: apache-2.0
short_description: Detect gibbon calls and Bornean forest sounds in audio
---

# PrimateLM-audio demo

A school robotics team project. Upload or record audio; the model (YAMNet embeddings +
a small classifier) says whether it hears a Northern grey gibbon, one of three Bornean
birds, or background. Code and results: https://github.com/ntoder/PrimateLM-audio

Example audio: Clink & Hamid Ahmad (2024), Zenodo 10.5281/zenodo.14213067, CC-BY-4.0.
"""


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)

    # Python code + the trained classifier (but not the dataset or caches)
    shutil.copytree(ROOT / "model", OUT / "model",
                    ignore=shutil.ignore_patterns("__pycache__", "*.npz", "*.png", "*.md", "metrics.json"))
    shutil.copy(ROOT / "model" / "yamnet" / "README.md", OUT / "model" / "yamnet" / "README.md")
    shutil.copytree(ROOT / "demo", OUT / "demo",
                    ignore=shutil.ignore_patterns("__pycache__", "prepare_space.py", "requirements.txt"))

    # Spaces install from requirements.txt in the root of the Space.
    reqs = (ROOT / "requirements.txt").read_text() + "gradio==5.50.0\n"
    (OUT / "requirements.txt").write_text(reqs)
    (OUT / "README.md").write_text(SPACE_README)

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"Space folder ready: {OUT}  ({size:.1f} MB)")
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print("  ", f.relative_to(OUT))


if __name__ == "__main__":
    main()
