"""Build the Examples page of the docs from real TEST-SET clips.

Run after training:
    python model/make_examples.py

For every class it takes the first clip the model got RIGHT and the first clip
it got WRONG (if any) - no hand-picking. For each one it saves:
  * a short MP3 (first 10 s) in docs/assets/examples/
  * a spectrogram PNG
  * a card in docs/examples.md (between the EXAMPLES:START / END markers)
The same clips are copied to demo/examples/ for the Gradio demo's example buttons.

Audio credit: Clink & Hamid Ahmad, Zenodo 10.5281/zenodo.14213067, CC-BY-4.0.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import ARTIFACTS_DIR, DATA_DIR, LABELS, ROOT, SAMPLE_RATE  # noqa: E402
from model.embed import load_audio  # noqa: E402
from model.plots import spectrogram_figure  # noqa: E402
from model.predict import PrimateClassifier  # noqa: E402

OUT_DIR = ROOT / "docs" / "assets" / "examples"
DEMO_DIR = ROOT / "demo" / "examples"
PAGE = ROOT / "docs" / "examples.md"
MAX_SECONDS = 10


def main() -> None:
    preds = np.load(ARTIFACTS_DIR / "test_predictions.npz")
    rows = list(zip(preds["paths"], preds["true"], preds["pred"]))

    # Pick: first correct + first wrong per class, in a fixed order.
    picks = []
    for lab in LABELS:
        right = [r for r in rows if r[1] == lab and r[2] == lab]
        wrong = [r for r in rows if r[1] == lab and r[2] != lab]
        picks += right[:1] + wrong[:1]

    for d in (OUT_DIR, DEMO_DIR):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    clf = PrimateClassifier()
    cards = []
    for rel_path, true, _pred in picks:
        wave = load_audio(DATA_DIR / rel_path)[: MAX_SECONDS * SAMPLE_RATE]
        # Re-run the model on the (possibly shortened) clip so the card shows
        # exactly what the model says about the audio you can play.
        windows = clf.predict_windows(wave)
        top3 = list(clf.overall(windows).items())[:3]
        pred, conf = top3[0]

        stem = f"{len(cards) + 1:02d}_{true}"
        sf.write(OUT_DIR / f"{stem}.mp3", wave, SAMPLE_RATE, format="MP3")
        sf.write(DEMO_DIR / f"{stem}.mp3", wave, SAMPLE_RATE, format="MP3")
        fig = spectrogram_figure(wave, height=2.2)
        fig.savefig(OUT_DIR / f"{stem}.png", dpi=90)
        plt.close(fig)

        ok = pred == true
        others = " · ".join(f"{clf.nice(l).split(' (')[0]} {p:.0%}" for l, p in top3[1:])
        cards.append(f"""<div class="example-card">
  <img src="../assets/examples/{stem}.png" alt="Spectrogram of a {clf.nice(true)} clip" loading="lazy">
  <audio controls preload="none" src="../assets/examples/{stem}.mp3"></audio>
  <div class="row"><span>True label</span><strong>{clf.nice(true)}</strong></div>
  <div class="row"><span>Prediction</span><span class="{'ok' if ok else 'bad'}">{clf.nice(pred)} {'✓' if ok else '✗'}</span></div>
  <div class="row"><span>Confidence</span><strong>{conf:.0%}</strong></div>
  <div class="row"><span>Next guesses</span><span>{others}</span></div>
  <div class="row"><span>Source file</span><code>{Path(rel_path).name[:38]}…</code></div>
</div>""")
        print(f"  {stem}: predicted {pred} ({conf:.0%})")

    block = ("<!-- EXAMPLES:START -->\n<div class=\"example-grid\">\n" + "\n".join(cards)
             + "\n</div>\n<!-- EXAMPLES:END -->")
    page = PAGE.read_text()
    start, end = page.index("<!-- EXAMPLES:START -->"), page.index("<!-- EXAMPLES:END -->") + len("<!-- EXAMPLES:END -->")
    PAGE.write_text(page[:start] + block + page[end:])
    print(f"Wrote {len(cards)} example cards to {PAGE}")


if __name__ == "__main__":
    main()
