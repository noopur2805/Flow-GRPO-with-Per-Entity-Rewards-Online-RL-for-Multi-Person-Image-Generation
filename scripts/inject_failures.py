"""Validate the reward's sensitivity by injecting known failures into real footage.

Rationale
---------
A reward is only as trustworthy as the instrument computing it. Before reporting
that a generator fails on X% of entities, we need to know what fraction of
*known* failures this reward can actually see -- otherwise the number confounds
generator failure with detector failure.

So: take real crowd frames with ground-truth boxes (MOT20), plant identity
swaps, vanishes and clones at controlled rates, and measure how often the
per-entity score drops for the tampered entities. That yields a recall curve
as a function of event duration and injection rate.

This is the same discipline as measuring what post-training quantization does
to a calibrated uncertainty head: the headline metric can look fine while the
thing you actually rely on has degraded.

Usage
-----
    python scripts/inject_failures.py --mot20 /path/to/MOT20/train/MOT20-02 \
        --out results/injection.json --rates 0.05 0.1 0.2 --durations 3 10 30
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

from flow_grpo_entity.entity_reward import Detections, EntityRewardConfig, entity_scores


# --------------------------------------------------------------------------
# MOT20 ground truth
# --------------------------------------------------------------------------

def load_mot_gt(seq_dir: pathlib.Path):
    """Parse gt/gt.txt -> {frame: {track_id: xyxy}} for pedestrian entries."""
    gt = defaultdict(dict)
    path = seq_dir / "gt" / "gt.txt"
    for line in path.read_text().splitlines():
        f, tid, x, y, w, h, conf, cls, vis = (line.split(",") + ["1"] * 9)[:9]
        if float(conf) == 0 or int(float(cls)) != 1:
            continue
        gt[int(f)][int(tid)] = np.array(
            [float(x), float(y), float(x) + float(w), float(y) + float(h)]
        )
    return dict(gt)


# --------------------------------------------------------------------------
# failure injection (operates on detections, not pixels)
# --------------------------------------------------------------------------

def inject_swap(det: Detections, i: int, j: int) -> Detections:
    """Identity swap: entity i takes on entity j's appearance and vice versa."""
    emb = det.embeddings.copy()
    emb[[i, j]] = emb[[j, i]]
    return Detections(det.boxes.copy(), det.scores.copy(), emb, det.image_hw)


def inject_clone(det: Detections, i: int, j: int) -> Detections:
    """Clone collapse: entity j is rendered as a copy of entity i."""
    emb = det.embeddings.copy()
    emb[j] = emb[i]
    return Detections(det.boxes.copy(), det.scores.copy(), emb, det.image_hw)


def inject_merge(det: Detections, i: int, j: int) -> Detections:
    """Merged bodies: entity j collapses onto entity i's box."""
    boxes = det.boxes.copy()
    boxes[j] = boxes[i] + np.array([3.0, 3.0, 3.0, 3.0])
    return Detections(boxes, det.scores.copy(), det.embeddings.copy(), det.image_hw)


def inject_vanish(det: Detections, i: int, _j: int) -> Detections:
    """Vanish: the detector loses confidence in entity i (partial fade)."""
    scores = det.scores.copy()
    scores[i] *= 0.25
    return Detections(det.boxes.copy(), scores, det.embeddings.copy(), det.image_hw)


INJECTORS = {
    "swap": inject_swap,
    "clone": inject_clone,
    "merge": inject_merge,
    "vanish": inject_vanish,
}


# --------------------------------------------------------------------------
# synthetic appearance for GT boxes (no crops needed for the ordering test)
# --------------------------------------------------------------------------

def synth_detections(boxes_by_id: dict[int, np.ndarray], image_hw, rng, dim=16):
    """Build Detections from GT boxes with per-identity stable embeddings.

    Each track id gets a fixed random embedding plus small per-frame noise, so
    an untampered sequence has high distinctness by construction. That isolates
    what injection does from what the detector does. Run the same script with
    `--real-embeddings` once a ReID backbone is wired in to repeat the
    measurement on true appearance features.
    """
    ids = sorted(boxes_by_id)
    boxes = np.stack([boxes_by_id[i] for i in ids])
    base = {i: rng.normal(size=dim) for i in ids}
    emb = np.stack([base[i] + 0.05 * rng.normal(size=dim) for i in ids])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    scores = np.clip(rng.normal(0.9, 0.05, size=len(ids)), 0.3, 1.0)
    return Detections(boxes, scores, emb, image_hw), ids


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------

def run(seq_dir: pathlib.Path, rates, durations, image_hw, cfg, seed=0):
    gt = load_mot_gt(seq_dir)
    frames = sorted(gt)
    rng = np.random.default_rng(seed)
    results = []

    for kind, injector in INJECTORS.items():
        for rate in rates:
            for dur in durations:
                detected, planted = 0, 0
                for start in range(0, max(len(frames) - dur, 1), max(dur, 1)):
                    window = frames[start : start + dur]
                    if not window:
                        continue
                    ids = sorted(gt[window[0]])
                    if len(ids) < 4:
                        continue
                    n_target = max(1, int(round(rate * len(ids))))
                    targets = rng.choice(len(ids), size=min(n_target * 2, len(ids)),
                                         replace=False)
                    pairs = list(zip(targets[::2], targets[1::2]))
                    if not pairs:
                        continue

                    for f in window:
                        det, frame_ids = synth_detections(gt[f], image_hw, rng)
                        if len(frame_ids) < 4:
                            continue
                        base_scores = entity_scores(det, cfg)
                        tampered = det
                        touched = []
                        for i, j in pairs:
                            if i < len(frame_ids) and j < len(frame_ids):
                                tampered = injector(tampered, int(i), int(j))
                                touched += [int(i), int(j)]
                        if not touched:
                            continue
                        new_scores = entity_scores(tampered, cfg)
                        drop = base_scores - new_scores
                        # an event is "detected" if any touched entity's score
                        # drops by more than the largest untouched drop
                        untouched = np.setdiff1d(np.arange(len(drop)), touched)
                        noise = drop[untouched].max() if len(untouched) else 0.0
                        planted += 1
                        if drop[touched].max() > max(noise, 0.05):
                            detected += 1

                recall = detected / planted if planted else float("nan")
                results.append({"kind": kind, "rate": rate, "duration": dur,
                                "planted": planted, "detected": detected,
                                "recall": recall})
                print(f"{kind:7s} rate={rate:<5} dur={dur:<3} "
                      f"recall={recall:.3f} ({detected}/{planted})")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot20", required=True, type=pathlib.Path,
                    help="a MOT20 sequence dir containing gt/gt.txt")
    ap.add_argument("--out", default="results/injection.json")
    ap.add_argument("--rates", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--durations", nargs="+", type=int, default=[3, 10, 30])
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    a = ap.parse_args()

    res = run(a.mot20, a.rates, a.durations, (a.height, a.width), EntityRewardConfig())
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")
