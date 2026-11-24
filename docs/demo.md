# Interactive Demo

## Try PrimateLM-audio

Upload a recording or use your microphone, and the model tells you which primate (if any) is calling.

<a href="#" class="ext-btn">Launch Demo on Hugging Face Spaces</a>

!!! info "Coming soon"
    The hosted demo will be linked here once it is deployed (build step 4). Until then you can run it locally with `python demo/app.py`.

## How to use it

1. **Load audio:** upload a WAV/MP3 file, record from your microphone, or click one of the example clips.
2. **Look at the spectrogram:** a picture of the sound — time left to right, pitch bottom to top, loudness as brightness. Primate calls often show up as bright repeating shapes.
3. **Read the top-3 predictions:** each label has a confidence bar.
4. **Long recordings:** for audio longer than 2 seconds, the demo slides a 2-second window across the file (1-second steps) and shows a timeline, for example *"gibbon call at 0:04–0:06"*.

## Run it yourself

```bash
pip install -r requirements.txt
python demo/app.py
```
