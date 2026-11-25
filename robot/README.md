# Robot: live primate-call detection on a Raspberry Pi 4

`live_detect.py` listens through a USB microphone, keeps the last 2 seconds of audio, and once
per second runs the same model as the demo. Detections are printed as JSON lines and can be sent
to MQTT and/or an HTTP endpoint. Full docs: `docs/robot.md` (Robot Integration page on the site).

## Try it on a laptop first (no microphone needed)

```bash
python robot/live_detect.py --wav demo/examples/01_gibbon.flac -v
```

## On the Pi

```bash
sudo apt install -y libportaudio2 git
git clone https://github.com/ntoder/PrimateLM-audio.git
cd PrimateLM-audio
python3 -m venv .venv
source .venv/bin/activate
pip install -r robot/requirements.txt
python robot/live_detect.py --list-devices
python robot/live_detect.py --device 1 -v
```

The trained classifier (`model/artifacts/classifier.joblib`) is in the repo, and YAMNet's
weights (15 MB) download on the first run, so you don't need the dataset on the robot.
