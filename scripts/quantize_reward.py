"""Quantize the reward scorer and check the RL objective survives compression.

Why this matters
----------------
The reward model runs on every rollout, so it is a real cost in the RL loop and
an obvious quantization target. But a reward is not a classifier: what matters
is not its absolute accuracy, it is whether it still *orders* samples the same
way. If int8 preserves mean reward while shuffling the ranking within a GRPO
group, the advantages are computed against a corrupted objective and training
optimises the wrong thing -- silently.

So the metric here is Spearman rank correlation between fp32 and int8 rewards
over the same images, alongside throughput. Mean absolute reward error is
reported too, and is deliberately *not* the headline.

Usage
-----
    python scripts/quantize_reward.py --images results/samples/*.png \
        --prompts dataset/crowd/test.txt --out results/quant.json
"""
from __future__ import annotations
import argparse, glob, json, pathlib, re, time
import numpy as np
from scipy.stats import spearmanr, kendalltau

from flow_grpo_entity.entity_reward import EntityRewardConfig, entity_reward


def n_from_prompt(p: str) -> int:
    m = re.search(r"exactly (\d+) people", p)
    return int(m.group(1)) if m else 10


def score_all(scorer, images, n_req, cfg):
    rewards, t0 = [], time.perf_counter()
    for img, n in zip(images, n_req):
        rewards.append(entity_reward(scorer(img), n, cfg).reward)
    return np.array(rewards), time.perf_counter() - t0


def group_rank_agreement(fp32, int8, group_size=8):
    """Rank agreement *within* GRPO groups -- the quantity advantages depend on."""
    taus = []
    for s in range(0, len(fp32) - group_size + 1, group_size):
        a, b = fp32[s:s + group_size], int8[s:s + group_size]
        if np.ptp(a) < 1e-9:
            continue
        taus.append(kendalltau(a, b).statistic)
    return float(np.nanmean(taus)) if taus else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="results/quant.json")
    ap.add_argument("--group-size", type=int, default=8)
    a = ap.parse_args()

    from PIL import Image
    from flow_grpo_entity.detector import PersonScorer

    paths = sorted(sum([glob.glob(p) for p in a.images], []))
    images = [np.array(Image.open(p).convert("RGB")) for p in paths]
    prompts = pathlib.Path(a.prompts).read_text().splitlines()
    n_req = [n_from_prompt(prompts[i % len(prompts)]) for i in range(len(images))]
    cfg = EntityRewardConfig()

    fp32_scorer = PersonScorer(half=False)
    int8_scorer = PersonScorer(half=True)   # swap for a true int8 export

    r32, t32 = score_all(fp32_scorer, images, n_req, cfg)
    r8, t8 = score_all(int8_scorer, images, n_req, cfg)

    out = {
        "n_images": len(images),
        "throughput_speedup": t32 / max(t8, 1e-9),
        "sec_per_image_fp32": t32 / max(len(images), 1),
        "sec_per_image_int8": t8 / max(len(images), 1),
        "mean_abs_reward_error": float(np.abs(r32 - r8).mean()),
        "spearman_global": float(spearmanr(r32, r8).statistic),
        "kendall_within_group": group_rank_agreement(r32, r8, a.group_size),
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nHeadline is kendall_within_group, not mean_abs_reward_error:")
    print("GRPO normalises rewards within a group, so only the ordering matters.")
