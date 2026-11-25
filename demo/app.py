"""PrimateLM-audio interactive demo (Gradio).

Run locally from the repo folder:
    python demo/app.py
then open the address it prints (usually http://127.0.0.1:7860).

What happens when you press "Analyse":
  1. Your audio (upload or microphone) is converted to 16 kHz mono.
  2. It is cut into 2-second windows that overlap by 50% (1-second steps).
  3. Every window goes through YAMNet + our classifier.
  4. We show: the spectrogram (with detections marked), the top-3 labels for
     the whole recording, and a timeline of detections for long recordings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.config import DETECTION_THRESHOLD, MIN_WINDOWS_IN_A_ROW, SAMPLE_RATE  # noqa: E402
from model.embed import to_model_format  # noqa: E402
from model.plots import spectrogram_figure  # noqa: E402
from model.predict import PrimateClassifier, fmt_time  # noqa: E402

MAX_SECONDS = 300           # longer audio is cut to 5 minutes to keep the free CPU fast
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

print("Loading model ...")
CLF = PrimateClassifier()   # loaded once when the app starts, reused for every request


def analyse(audio, threshold: float, min_windows: int):
    """Gradio calls this with audio = (sample_rate, numpy array) or None."""
    if audio is None:
        raise gr.Error("Please upload a file, record from the microphone, or click an example.")
    sr, data = audio
    wave = to_model_format(data, sr)
    note = ""
    if len(wave) > MAX_SECONDS * SAMPLE_RATE:
        wave = wave[: MAX_SECONDS * SAMPLE_RATE]
        note = f"\n\n*Only the first {MAX_SECONDS // 60} minutes were analysed.*"
    if len(wave) < 0.5 * SAMPLE_RATE:
        raise gr.Error("That recording is shorter than half a second - please use a longer clip.")

    windows = CLF.predict_windows(wave)
    dets = CLF.detections(windows, threshold, int(min_windows))

    # Top-3 labels for the whole recording (gr.Label draws confidence bars).
    overall = CLF.overall(windows)
    top3 = {CLF.nice(lab): p for lab, p in list(overall.items())[:3]}

    # Spectrogram with detections drawn as green bars along the top.
    fig = spectrogram_figure(wave, dets, CLF.label_names, height=3.0)
    plt.close(fig)  # free pyplot's copy; Gradio can still draw the figure

    # Timeline text.
    duration = len(wave) / SAMPLE_RATE
    lines = [f"**{duration:.1f} s analysed** in {len(windows)} windows "
             f"(2 s long, 1 s apart)."]
    if dets:
        lines.append("")
        for d in dets:
            lines.append(f"- **{CLF.nice(d.label)}** at {fmt_time(d.start)}–{fmt_time(d.end)} "
                         f"(peak confidence {d.confidence:.0%})")
    else:
        lines.append(f"\nNo calls detected (needs confidence ≥ {threshold:.0%} in "
                     f"{int(min_windows)}+ windows in a row). Mostly background.")

    # Per-window table for the curious.
    table = [[f"{fmt_time(w.start)}–{fmt_time(w.end)}", CLF.nice(w.label), f"{w.confidence:.0%}"]
             for w in windows]
    return top3, fig, "\n".join(lines) + note, table


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="green", secondary_hue="emerald")
    with gr.Blocks(title="PrimateLM-audio demo", theme=theme) as ui:
        gr.Markdown(
            "# 🐒 PrimateLM-audio\n"
            "Upload or record forest audio. The model listens in 2-second windows and tells you "
            "if it hears a **Northern grey gibbon**, one of three Bornean birds, or just background.\n\n"
            "Small school-project model: YAMNet embeddings + a small classifier, ~85% clip accuracy on "
            "held-out recordings. [Docs & results](https://ntoder.github.io/PrimateLM-audio/)")
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio (WAV / MP3)")
                with gr.Accordion("Detection settings", open=False):
                    threshold = gr.Slider(0.3, 0.95, value=DETECTION_THRESHOLD, step=0.05,
                                          label="Confidence threshold")
                    min_windows = gr.Slider(1, 4, value=MIN_WINDOWS_IN_A_ROW, step=1,
                                            label="Windows in a row needed")
                button = gr.Button("Analyse", variant="primary")
                examples = sorted(EXAMPLES_DIR.glob("*.flac"))
                if examples:
                    gr.Examples(
                        examples=[[str(p)] for p in examples], inputs=[audio],
                        example_labels=[p.stem[3:].replace("_", " ") + f" #{p.stem[:2]}" for p in examples],
                        label="Example clips from the test set (Clink & Hamid Ahmad, CC-BY-4.0)")
            with gr.Column(scale=2):
                top3 = gr.Label(num_top_classes=3, label="Top 3 for the whole recording")
                spec = gr.Plot(label="Spectrogram (green bars = detections)")
                timeline = gr.Markdown()
                with gr.Accordion("Every 2-second window", open=False):
                    table = gr.Dataframe(headers=["Time", "Top label", "Confidence"], interactive=False)

        button.click(analyse, [audio, threshold, min_windows], [top3, spec, timeline, table])
    return ui


if __name__ == "__main__":
    build_ui().launch()
