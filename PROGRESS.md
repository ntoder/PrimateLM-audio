# PrimateLM-audio — progress

**Overall: 30% complete** · last updated 2026-11-24

| # | Step | Status | Notes |
|---|------|--------|-------|
| 1 | Docs skeleton (MkDocs Material, 9 pages, green theme, Pages workflow) | ✅ done | `mkdocs build --strict` passes; light/dark + Mermaid checked |
| 2 | Data download (verified, openly licensed dataset + background class) | ✅ done | Borneo loud-calls, Zenodo 14213067, **CC-BY-4.0** verified via API; MD5-checked; 1,422 train clips + 209 Maliau test clips, 16 kHz |
| 3 | Training + evaluation (YAMNet embeddings → classifier, report) | ⏳ next | |
| 4 | Gradio demo (upload/mic, spectrogram, top-3, timeline) | ⬜ todo | |
| 5 | Robot stub (Pi 4, rolling 2 s chunks, stdout/MQTT/HTTP) | ⬜ todo | |
| 6 | Deploy (GitHub Pages + Hugging Face Space, final README) | ⬜ todo | |

## Decisions so far
- **Embedding model:** YAMNet (3.7M params, 16 kHz native, CPU + Raspberry Pi friendly). Weights download from storage.googleapis.com.
- **Not used:** ComParE 2021 Primates (no open download found), MarmAudio (~60 GB).
- **Docs stack:** MkDocs 1.6.1 + Material 9.8.2, same layout as the NatureLM-audio docs.

## Open items
- Create the GitHub repo `ntoder/PrimateLM-audio` so work can be pushed.
- ~~Allow `zenodo.org`~~ (done). Later: allow `huggingface.co` (Space deploy) in the cloud environment.
