# PrimateLM-audio — progress

**Overall: 95% complete** (MVP done; 2 deploy clicks left for you) · last updated 2026-11-25

| # | Step | Status | Notes |
|---|------|--------|-------|
| 1 | Docs skeleton (MkDocs Material, 9 pages, green theme, Pages workflow) | ✅ done | `mkdocs build --strict` passes; light/dark + Mermaid checked |
| 2 | Data download (verified, openly licensed dataset + background class) | ✅ done | Borneo loud-calls, Zenodo 14213067, **CC-BY-4.0** verified via API; MD5-checked; 1,422 train clips + 209 Maliau test clips, 16 kHz |
| 3 | Training + evaluation (YAMNet embeddings → classifier, report) | ✅ done | `python model/train.py` from scratch in ~3.5 min on CPU. MLP chosen on val. **Test: 84.8% clip accuracy, macro-F1 0.72, gibbon F1 0.82.** New forest (Maliau): 80.4%. Report: `model/artifacts/eval_report.md` |
| 4 | Gradio demo (upload/mic, spectrogram, top-3, timeline) | ✅ done | `python demo/app.py` tested via API + browser screenshot. `python demo/prepare_space.py` builds a standalone Space folder (tested from a clean copy). Not yet uploaded to Hugging Face (needs your HF account) |
| 5 | Robot stub (Pi 4, rolling 2 s chunks, stdout/MQTT/HTTP) | ✅ done | `robot/live_detect.py --wav FILE` tested: stdout ✅, HTTP POST ✅, MQTT via test broker ✅, ~90 ms/window on x86. **Not yet tested on a real Pi or USB mic** |
| 6 | Deploy (GitHub Pages + Hugging Face Space, final README) | 🟡 almost | README tested from a fresh clone: download → train (same numbers) → predict → robot all pass. Pages workflow builds on GitHub but **deploy waits for you to enable Pages**. Space folder ready; **upload needs your Hugging Face account** |

## Decisions so far
- **Embedding model:** YAMNet (3.7M params, 16 kHz native, CPU + Raspberry Pi friendly). Weights download from storage.googleapis.com.
- **Not used:** ComParE 2021 Primates (no open download found), MarmAudio (~60 GB).
- **Docs stack:** MkDocs 1.6.1 + Material 9.8.2, same layout as the NatureLM-audio docs.

- **Detection rule:** threshold 0.6 **and** 2 windows in a row: halves false alarms on background (test: 9.3% → 4.6% of windows).
- **Split:** by 2-hour recording (StratifiedGroupKFold, seed 42) so test recordings are never seen in training.

- **Example audio is FLAC, not MP3:** MP3 compression dropped one gibbon clip's score from 74% to 49%.

- **Robot false alarms in a new forest:** on a 48 s Maliau test, default settings gave 1 real + 2 false detections; `--threshold 0.9` gave 1 real + 1 false.

## Open items
- **You:** GitHub → repo Settings → Pages → Source: **GitHub Actions**, then re-run the "Deploy docs" workflow (Actions tab).
- **You:** create a Hugging Face account/Space and upload `build/space` (commands in README → Deploying). Then put the Space link in `docs/demo.md` (the "Hosted demo" box).
- Add team member names in `docs/team.md`.
- Run `robot/live_detect.py` on the real Pi 4 + USB mic and note the ms/window.
- ~~Create the GitHub repo~~ (done, pushing after every step).
- Upload the demo to a Hugging Face Space (`python demo/prepare_space.py`, then the upload command in docs/demo.md).
- ~~Allow `zenodo.org`~~ (done). Later: allow `huggingface.co` (Space deploy) in the cloud environment.
