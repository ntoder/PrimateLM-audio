# Usage

This page explains what you can do with PrimateLM-audio and what its limits are. For the commands to install and run everything, see [Quick Start](quick_start.md).

## What goes in, what comes out

| | |
|---|---|
| **Input** | Any audio file (WAV, MP3, FLAC, OGG) or live microphone audio. It is converted to mono, 16 kHz. |
| **Window** | The model looks at **2 seconds** at a time. Longer audio is cut into 2-second windows that overlap by 50% (a new window every 1 second). |
| **Output** | For each window: a label (one of the classes on the [Datasets](datasets.md) page) and a confidence between 0 and 1. |

## Three ways to use it

=== "Command line"

    ```bash
    python model/predict.py path/to/recording.wav
    ```
    Prints the top-3 labels for a short clip, or a timeline of detections for a long one.

=== "Web demo"

    ```bash
    python demo/app.py
    ```
    Opens a Gradio page in your browser. See [Interactive Demo](demo.md).

=== "Robot"

    ```bash
    python robot/live_detect.py --wav path/to/recording.wav
    ```
    Simulates a live microphone from a file. On a Raspberry Pi, drop `--wav` to use the USB mic. See [Robot Integration](robot.md).

## Reading the confidence

The confidence is the classifier's probability for its top label. It is **not** a guarantee: a confident answer can still be wrong, especially for sounds the model never saw in training (for example, monkey species that are not in our dataset). We treat a window as a detection only when confidence is above a threshold (default 0.6); below that the demo shows "uncertain".

## Limits — please read

- The model only knows the classes it was trained on. A howler monkey will be forced into one of the known classes or "background".
- All training audio comes from one dataset, recorded with one type of recorder in one region. Accuracy on phone recordings or zoo recordings will likely be lower than the test-set numbers.
- 2-second windows can cut a call in half. Long calls are usually detected in at least one window.
