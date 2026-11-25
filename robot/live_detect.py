"""Live primate-call detector for a robot (Raspberry Pi 4, CPU only).

It listens in 1-second chunks, always keeps the LAST 2 SECONDS of audio, and
runs the same model as the demo on that 2-second window once per second
(2-second windows with 50% overlap). Detections are printed as JSON lines and
can also be sent over MQTT or HTTP.

Examples:
    # Simulate the microphone with a WAV file (works on any laptop):
    python robot/live_detect.py --wav demo/examples/01_gibbon.flac

    # Same, but as slow as real life (1 chunk per second):
    python robot/live_detect.py --wav recording.wav --realtime

    # Real USB microphone on the Pi (list devices with --list-devices):
    python robot/live_detect.py --device 1

    # Also publish to an MQTT broker and/or POST to a web server:
    python robot/live_detect.py --device 1 --mqtt localhost --http http://192.168.1.20:8000/detections

Output (one JSON object per line on stdout; status messages go to stderr):
    {"event": "start", "time": "...", "label": "gibbon", "confidence": 0.97,
     "window_start_s": 7.0, "window_end_s": 9.0}
    {"event": "end", "time": "...", "label": "gibbon", "confidence": 1.0,
     "start_s": 6.0, "end_s": 13.0}
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import DETECTION_THRESHOLD, MIN_WINDOWS_IN_A_ROW, SAMPLE_RATE  # noqa: E402
from model.embed import load_audio, to_model_format  # noqa: E402
from model.predict import PrimateClassifier  # noqa: E402

CHUNK_SECONDS = 1.0    # new audio arrives 1 second at a time
WINDOW_SECONDS = 2.0   # the model always looks at the last 2 seconds


def log(*args) -> None:
    """Status messages go to stderr, so stdout only contains detection JSON."""
    print(*args, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Audio sources: each one yields 1-second chunks of 16 kHz mono audio
# ---------------------------------------------------------------------------
def chunks_from_file(path: str, realtime: bool):
    """Pretend a file is a microphone: hand it out 1 second at a time."""
    wave = load_audio(path)
    step = int(CHUNK_SECONDS * SAMPLE_RATE)
    for i in range(0, len(wave) - step + 1, step):
        if realtime:
            time.sleep(CHUNK_SECONDS)
        yield wave[i:i + step]


def chunks_from_mic(device: int | None):
    """Read a USB microphone with the `sounddevice` library.
    Many USB mics only support 44.1 or 48 kHz, so we record at the mic's own
    rate and resample each chunk to 16 kHz."""
    import sounddevice as sd   # imported here so --wav works without it

    info = sd.query_devices(device, "input")
    mic_rate = int(info["default_samplerate"])
    log(f"Listening on '{info['name']}' at {mic_rate} Hz (Ctrl+C to stop)")
    q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):   # runs in a background thread
        if status:
            log("audio warning:", status)
        q.put(indata[:, 0].copy())

    block = int(CHUNK_SECONDS * mic_rate)
    with sd.InputStream(device=device, channels=1, samplerate=mic_rate, blocksize=block,
                        dtype="float32", callback=callback):
        while True:
            yield to_model_format(q.get(), mic_rate)


# ---------------------------------------------------------------------------
# Outputs: stdout always, MQTT and HTTP optional
# ---------------------------------------------------------------------------
class Publisher:
    def __init__(self, mqtt_host: str | None, mqtt_topic: str, http_url: str | None):
        self.mqtt = None
        self.topic = mqtt_topic
        self.http_url = http_url
        if mqtt_host:
            import paho.mqtt.client as mqtt   # pip install paho-mqtt
            host, _, port = mqtt_host.partition(":")
            self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.mqtt.connect(host, int(port or 1883))
            self.mqtt.loop_start()             # handle the network in the background
            log(f"MQTT: publishing to {host}:{port or 1883} topic '{mqtt_topic}'")

    def send(self, message: dict) -> None:
        line = json.dumps(message)
        print(line, flush=True)
        if self.mqtt:
            # wait up to 2 s so the message is really sent (matters if we exit right after)
            self.mqtt.publish(self.topic, line, qos=1).wait_for_publish(timeout=2)
        if self.http_url:
            try:   # never let a network problem crash the robot
                req = urllib.request.Request(self.http_url, data=line.encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=2).close()
            except OSError as e:
                log(f"HTTP POST failed: {e}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(chunks, clf: PrimateClassifier, pub: Publisher, threshold: float,
        min_windows: int, verbose: bool) -> None:
    window_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    buffer = deque(maxlen=window_len)      # holds only the most recent 2 seconds
    t = 0.0                                # seconds of audio received so far
    run_label, run_len, run_start, run_conf = None, 0, 0.0, 0.0
    announced = False

    for chunk in chunks:
        buffer.extend(chunk)
        t += len(chunk) / SAMPLE_RATE
        if len(buffer) < window_len:
            continue                       # wait until we have 2 full seconds

        t0 = time.perf_counter()
        w = clf.predict_windows(np.array(buffer, dtype=np.float32))[0]
        ms = (time.perf_counter() - t0) * 1000
        start, end = t - WINDOW_SECONDS, t
        hit = w.label != "background" and w.confidence >= threshold
        if verbose:
            log(f"[{start:6.1f}-{end:6.1f}s] {w.label:20s} {w.confidence:4.0%}  ({ms:.0f} ms)")

        if hit and w.label == run_label:
            run_len += 1
            run_conf = max(run_conf, w.confidence)
        else:
            # The previous run (if any) is over.
            if announced:
                pub.send({"event": "end", "time": datetime.now().isoformat(timespec="seconds"),
                          "label": run_label, "confidence": round(run_conf, 3),
                          "start_s": run_start, "end_s": start + WINDOW_SECONDS - CHUNK_SECONDS})
            announced = False
            run_label, run_len, run_start, run_conf = (w.label, 1, start, w.confidence) if hit \
                else (None, 0, 0.0, 0.0)

        # Announce once, as soon as the run is long enough (default: 2 windows in a row).
        if run_label and run_len >= min_windows and not announced:
            pub.send({"event": "start", "time": datetime.now().isoformat(timespec="seconds"),
                      "label": run_label, "name": clf.nice(run_label),
                      "confidence": round(w.confidence, 3),
                      "window_start_s": start, "window_end_s": end})
            announced = True

    if announced:   # audio ended in the middle of a call
        pub.send({"event": "end", "time": datetime.now().isoformat(timespec="seconds"),
                  "label": run_label, "confidence": round(run_conf, 3),
                  "start_s": run_start, "end_s": t})


def main() -> None:
    ap = argparse.ArgumentParser(description="Live primate-call detector for a robot.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--wav", help="read audio from this file instead of a microphone")
    src.add_argument("--device", type=int, default=None, help="microphone device number")
    src.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--realtime", action="store_true", help="with --wav: feed 1 chunk per second")
    ap.add_argument("--threshold", type=float, default=DETECTION_THRESHOLD)
    ap.add_argument("--min-windows", type=int, default=MIN_WINDOWS_IN_A_ROW)
    ap.add_argument("--mqtt", metavar="HOST[:PORT]", help="publish detections to this MQTT broker")
    ap.add_argument("--mqtt-topic", default="primatelm/detections")
    ap.add_argument("--http", metavar="URL", help="POST each detection as JSON to this URL")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every window to stderr")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    log("Loading model ...")
    clf = PrimateClassifier()
    pub = Publisher(args.mqtt, args.mqtt_topic, args.http)
    chunks = chunks_from_file(args.wav, args.realtime) if args.wav else chunks_from_mic(args.device)
    try:
        run(chunks, clf, pub, args.threshold, args.min_windows, args.verbose)
    except KeyboardInterrupt:
        log("Stopped.")


if __name__ == "__main__":
    main()
