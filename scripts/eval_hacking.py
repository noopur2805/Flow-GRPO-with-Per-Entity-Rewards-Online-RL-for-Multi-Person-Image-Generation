"""Reward-hacking analysis: training reward vs held-out per-entity quality.

Compares checkpoints (base / count-RL / entity-RL) on held-out prompts, and
plots the curve that matters: the reward each policy was trained on, against an
independent measure of per-entity quality it was not trained on.

If the counting policy's training reward rises while held-out per-entity quality
falls, that is reward hacking, measured rather than asserted.

Usage
-----
    python scripts/eval_hacking.py --samples base=dir1 count_rl=dir2 entity_rl=dir3 \
        --prompts dataset/crowd/test.txt --out results/hacking.json
"""
from __future__ import annotations
import argparse, glob, json, pathlib, re
import numpy as np

from flow_grpo_entity.entity_reward import (
    EntityRewardConfig, count_only_reward, entity_reward, mean_baseline_reward)


def n_from_prompt(p):
    m = re.search(r"exactly (\d+) people", p)
    return int(m.group(1)) if m else 10


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True,
                    help="name=dir pairs, one per checkpoint")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="results/hacking.json")
    a = ap.parse_args()

    from PIL import Image
    from flow_grpo_entity.detector import PersonScorer

    scorer, cfg = PersonScorer(), EntityRewardConfig()
    prompts = pathlib.Path(a.prompts).read_text().splitlines()
    report = {}

    for spec in a.samples:
        name, d = spec.split("=", 1)
        paths = sorted(glob.glob(str(pathlib.Path(d) / "*.png")))
        rows = []
        for i, p in enumerate(paths):
            img = np.array(Image.open(p).convert("RGB"))
            n = n_from_prompt(prompts[i % len(prompts)])
            det = scorer(img)
            b = entity_reward(det, n, cfg)
            rows.append({
                "count_reward": count_only_reward(det, n, cfg),   # training reward
                "entity_reward": b.reward,                        # training reward
                "tail": b.tail,                                   # held-out quality
                "mean_entity": mean_baseline_reward(det, n, cfg),
                "n_detected": b.n_detected, "n_requested": n,
                "worst_entity": float(b.per_entity.min()) if b.n_detected else 0.0,
            })
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        report[name] = {"n_samples": len(rows), "mean": agg}
        print(f"{name:10s} count={agg['count_reward']:.3f} "
              f"tail={agg['tail']:.3f} mean_entity={agg['mean_entity']:.3f} "
              f"worst={agg['worst_entity']:.3f} n_det={agg['n_detected']:.1f}")

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {a.out}")
    print("Hacking signature: count_rl raises count_reward above base while its "
          "tail/worst_entity fall below base.")
