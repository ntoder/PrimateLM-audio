"""Train the PrimateLM-audio classifier from scratch and write an evaluation report.

Run from the repo folder (after `python model/download_data.py`):
    python model/train.py

Steps:
  1. Find all clips and split them 70/15/15 by recording (see dataset.py).
  2. Cut every clip into 2-second windows (1 s step) and get a YAMNet embedding
     for each window. This is the slow part (a few minutes), so results are
     cached in data/cache/ and reused next time.
  3. Train two small classifiers on the TRAIN windows:
       - logistic regression (straight-line boundaries)
       - a small MLP neural network (one hidden layer)
     and keep whichever scores better on the VALIDATION set.
  4. Score the chosen model ONCE on the TEST set, plus the extra Maliau test set
     from a different forest. Write model/artifacts/eval_report.md, metrics.json
     and confusion-matrix images.
  5. Save the classifier to model/artifacts/classifier.joblib.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import evaluate as ev  # noqa: E402
from model.config import (ARTIFACTS_DIR, CACHE_DIR, DATA_DIR, HOP_SECONDS, LABEL_NAMES,  # noqa: E402
                          LABELS, WINDOW_SECONDS)
from model.dataset import SEED, Clip, maliau_clips, split_by_recording, training_clips  # noqa: E402
from model.embed import YamnetEmbedder, load_audio  # noqa: E402


# ---------------------------------------------------------------------------
# Step 2: embeddings (with a cache so re-running is fast)
# ---------------------------------------------------------------------------
_EMBEDDER: YamnetEmbedder | None = None


def get_embedder() -> YamnetEmbedder:
    """Load YAMNet only once, and only if we actually need it."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = YamnetEmbedder()
    return _EMBEDDER


def embed_clips(clips: list[Clip], name: str):
    """Return (X, y, clip_id): one row per 2-second window.
    X = embeddings (n_windows, 1024), y = label of the clip the window came from,
    clip_id = index of that clip in `clips` (used to combine windows per clip)."""
    cache = CACHE_DIR / f"{name}_w{WINDOW_SECONDS}_h{HOP_SECONDS}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if list(d["paths"]) == [str(c.path) for c in clips]:
            print(f"  using cached embeddings: {cache.name}")
            return d["X"], d["y"], d["clip_id"]

    embedder = get_embedder()
    X, y, clip_id = [], [], []
    t0 = time.time()
    for i, clip in enumerate(clips):
        for w in embedder.windows(load_audio(clip.path)):
            X.append(w.embedding); y.append(clip.label); clip_id.append(i)
        if (i + 1) % 100 == 0 or i + 1 == len(clips):
            print(f"  {name}: {i + 1}/{len(clips)} clips ({time.time() - t0:.0f}s)")
    X, y, clip_id = np.array(X, dtype=np.float32), np.array(y), np.array(clip_id)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, clip_id=clip_id, paths=np.array([str(c.path) for c in clips]))
    return X, y, clip_id


# ---------------------------------------------------------------------------
# Step 3: the two candidate classifiers
# ---------------------------------------------------------------------------
def candidates():
    """Both models first standardise each of the 1,024 numbers (mean 0, std 1),
    which helps them train. class_weight='balanced' makes rare classes (like the
    rhinoceros hornbill) count as much as common ones."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000, random_state=SEED)),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-2, early_stopping=True,
                          max_iter=300, random_state=SEED)),
    }


def false_alarm_table(model, X, y, clip_id) -> list[dict]:
    """For the robot: how often does BACKGROUND audio trigger a detection, and how
    many gibbon windows do we catch, at different confidence thresholds?
    'two_in_a_row' requires two back-to-back windows from the same clip with the
    same label, like model/predict.py does."""
    probs = model.predict_proba(X)
    top, conf = model.classes_[probs.argmax(1)], probs.max(1)
    bg, gib = y == "background", y == "gibbon"
    same_as_prev = np.r_[False, (top[1:] == top[:-1]) & (clip_id[1:] == clip_id[:-1])]
    rows = []
    for th in (0.6, 0.8, 0.9):
        fa = bg & (top != "background") & (conf >= th)
        fa2 = fa & np.r_[False, fa[:-1]] & same_as_prev
        rows.append({"threshold": th,
                     "false_alarm_single": float(fa.sum() / max(bg.sum(), 1)),
                     "false_alarm_two_in_a_row": float(fa2.sum() / max(bg.sum(), 1)),
                     "gibbon_window_recall": float(((top == "gibbon") & (conf >= th) & gib).sum() / max(gib.sum(), 1))})
    return rows


def evaluate_set(model, X, y, clip_id, clips):
    """Score at two levels: every 2-second window, and every whole clip."""
    probs = model.predict_proba(X)
    classes = model.classes_
    win_pred = classes[probs.argmax(axis=1)]
    ids, clip_pred = ev.clip_predictions(probs, clip_id, classes)
    clip_true = np.array([clips[i].label for i in ids])
    return {
        "window": ev.scores(list(y), list(win_pred), LABELS),
        "clip": ev.scores(list(clip_true), list(clip_pred), LABELS),
        "_clip_true": clip_true, "_clip_pred": clip_pred, "_clip_ids": ids,
        "_probs": probs,
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- Step 1: clips and split
    print("1) Finding clips and splitting by recording ...")
    all_clips = training_clips()
    splits = split_by_recording(all_clips)
    extra = maliau_clips()
    for name, cl in [*splits.items(), ("maliau", extra)]:
        counts = {l: sum(c.label == l for c in cl) for l in LABELS}
        print(f"  {name:6s} {len(cl):5d} clips  {counts}")

    # ---- Step 2: embeddings
    print("2) Computing YAMNet embeddings for 2-second windows ...")
    data = {}
    for name, cl in [*splits.items(), ("maliau", extra)]:
        data[name] = embed_clips(cl, name)

    # ---- Step 3: train both candidates, pick the best on validation
    print("3) Training classifiers ...")
    Xtr, ytr, _ = data["train"]
    best_name, best_model, best_f1, val_results = None, None, -1.0, {}
    for name, model in candidates().items():
        model.fit(Xtr, ytr)
        r = evaluate_set(model, *data["val"], splits["val"])
        val_results[name] = {"window": r["window"], "clip": r["clip"]}
        f1 = r["clip"]["macro_f1"]
        print(f"  {name:20s} val clip macro-F1 = {f1:.3f}   val window accuracy = {r['window']['accuracy']:.3f}")
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1
    print(f"  -> keeping {best_name}")

    # ---- Step 4: final scores on test sets (used ONCE, never for choosing)
    print("4) Scoring on the held-out test sets ...")
    test = evaluate_set(best_model, *data["test"], splits["test"])
    mal = evaluate_set(best_model, *data["maliau"], extra)
    for name, r in [("test", test), ("maliau", mal)]:
        print(f"  {name:6s} clip accuracy {r['clip']['accuracy']:.3f}  macro-F1 {r['clip']['macro_f1']:.3f}"
              f" | window accuracy {r['window']['accuracy']:.3f}  macro-F1 {r['window']['macro_f1']:.3f}")

    ev.plot_confusion(test["_clip_true"], test["_clip_pred"], LABELS,
                      "Test set (Danum Valley) - per clip", ARTIFACTS_DIR / "confusion_matrix.png")
    ev.plot_confusion(mal["_clip_true"], mal["_clip_pred"], LABELS,
                      "Maliau Basin (new forest) - per clip", ARTIFACTS_DIR / "confusion_matrix_maliau.png")

    # ---- Step 5: save model + reports
    joblib.dump({
        "model": best_model, "model_type": best_name, "labels": LABELS,
        "label_names": LABEL_NAMES, "window_seconds": WINDOW_SECONDS, "hop_seconds": HOP_SECONDS,
    }, ARTIFACTS_DIR / "classifier.joblib", compress=3)

    def public(r):  # drop the helper arrays that start with "_"
        return {k: v for k, v in r.items() if not k.startswith("_")}

    dataset_info = json.loads((DATA_DIR / "dataset_info.json").read_text()) \
        if (DATA_DIR / "dataset_info.json").exists() else {}
    metrics = {
        "model_type": best_name, "dataset": dataset_info,
        "split_counts": {n: {l: sum(c.label == l for c in cl) for l in LABELS}
                         for n, cl in [*splits.items(), ("maliau", extra)]},
        "window_counts": {n: int(len(d[0])) for n, d in data.items()},
        "validation": val_results, "test": public(test), "maliau": public(mal),
        "false_alarms": {"test": false_alarm_table(best_model, *data["test"]),
                         "maliau": false_alarm_table(best_model, *data["maliau"])},
        "train_seconds": round(time.time() - t_start),
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS_DIR / "eval_report.md").write_text(make_report(metrics))

    # Save which test clips were used, so make_examples.py can pick honest examples.
    np.savez(ARTIFACTS_DIR / "test_predictions.npz",
             paths=np.array([str(splits["test"][i].path.relative_to(DATA_DIR)) for i in test["_clip_ids"]]),
             true=test["_clip_true"], pred=test["_clip_pred"])
    size_kb = (ARTIFACTS_DIR / "classifier.joblib").stat().st_size / 1024
    print(f"5) Saved classifier ({size_kb:.0f} KB) and report to {ARTIFACTS_DIR}")
    print(f"Done in {time.time() - t_start:.0f}s")


def make_report(m: dict) -> str:
    """Write the human-readable evaluation report (Markdown)."""
    t, mal = m["test"], m["maliau"]
    split_rows = "\n".join(
        f"| {n} | " + " | ".join(str(m["split_counts"][n][l]) for l in LABELS)
        + f" | {sum(m['split_counts'][n].values())} | {m['window_counts'][n]} |"
        for n in ["train", "val", "test", "maliau"])
    val_rows = "\n".join(
        f"| {n} | {v['clip']['macro_f1']:.3f} | {v['clip']['accuracy']:.3f} | {v['window']['accuracy']:.3f} |"
        for n, v in m["validation"].items())
    fa_rows = "\n".join(
        f"| {name} | {r['threshold']} | {r['false_alarm_single']:.1%} | {r['false_alarm_two_in_a_row']:.1%} "
        f"| {r['gibbon_window_recall']:.1%} |"
        for name in ("test", "maliau") for r in m["false_alarms"][name])
    return f"""# Evaluation report

Generated by `python model/train.py`. All numbers are on data the model never trained on.

**Chosen model:** `{m['model_type']}` on top of frozen YAMNet embeddings (2-second windows, 1-second hop).

## Data

Dataset: {m['dataset'].get('title', '?')} - {', '.join(m['dataset'].get('creators', []))},
DOI {m['dataset'].get('doi', '?')}, license {m['dataset'].get('license', '?')}.
Split by recording (no recording appears in two splits). `maliau` is a separate
test set from a different forest, recorded a year later.

| Split | {' | '.join(LABELS)} | Clips | Windows |
|---|{'---|' * len(LABELS)}---|---|
{split_rows}

## Model selection (validation set)

| Model | Clip macro-F1 | Clip accuracy | Window accuracy |
|---|---|---|---|
{val_rows}

## Test set - Danum Valley (same forest as training)

| Level | Accuracy | Macro-F1 |
|---|---|---|
| Per clip (average of its windows) | **{t['clip']['accuracy']:.3f}** | **{t['clip']['macro_f1']:.3f}** |
| Per 2-second window | {t['window']['accuracy']:.3f} | {t['window']['macro_f1']:.3f} |

Per-class scores (per clip):

{ev.markdown_table(t['clip'], LABEL_NAMES)}

![Confusion matrix](confusion_matrix.png)

## Extra test - Maliau Basin (new forest, never seen)

Only gibbon, rhinoceros hornbill and background clips exist in this set.

| Level | Accuracy | Macro-F1 |
|---|---|---|
| Per clip | **{mal['clip']['accuracy']:.3f}** | **{mal['clip']['macro_f1']:.3f}** |
| Per 2-second window | {mal['window']['accuracy']:.3f} | {mal['window']['macro_f1']:.3f} |

{ev.markdown_table(mal['clip'], LABEL_NAMES)}

![Confusion matrix Maliau](confusion_matrix_maliau.png)

## Detection settings (for the demo and robot)

Share of BACKGROUND 2-second windows that would wrongly trigger a detection, and
share of gibbon windows that are caught, at each confidence threshold.

| Test set | Threshold | False alarms (1 window) | False alarms (2 in a row) | Gibbon windows caught |
|---|---|---|---|---|
{fa_rows}

We use threshold 0.6 with 2 windows in a row by default.

## Caveats

- Window labels are inherited from the clip: a 2-second window inside a "gibbon"
  clip may actually be a pause between calls, so window scores are pessimistic.
- Small classes (rhinoceros hornbill, helmeted hornbill) have few test clips, so
  their scores can swing a lot with one or two mistakes.
- All audio comes from one recorder model in Sabah, Malaysia. Expect lower accuracy
  on phone, zoo or other-region recordings.
"""


if __name__ == "__main__":
    main()
