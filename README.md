<h1 align="center"> Flow-GRPO with Per-Entity Rewards: <br> Online RL for Multi-Person Image Generation </h1>

# Training Flow Matching Models via Online RL with Per-Entity Rewards

An extension to [Flow-GRPO](https://github.com/yifan123/flow_grpo) for multi-person
image generation. Flow-GRPO supplies the RL machinery (ODE-to-SDE conversion,
group sampling, the GRPO loop); this repository supplies the reward.

## The problem

Flow-GRPO's shipped rewards are aggregate: one scalar per image. On a prompt
asking for 30 people, a counting reward is satisfied by 30 things a detector
fires on. Nothing in the objective distinguishes 30 distinct people from 30
merged, cloned blobs — so if the policy can find the cheaper solution, it will.
This is the same failure documented for video RL, where higher reward has been
reported alongside *lower* human preference than the base model.

## The reward

Each detected entity is scored on four terms — detector confidence,
distinctness from its nearest neighbour in appearance space, separation from
overlapping boxes, and plausible scale. The image's score is then the **CVaR
over the worst 20% of entities**, not the mean:

```
R = (CVaR_α{r_i}  +  w · count_adherence) / (1 + w)
```

Aggregating over the tail rather than the mean is the Group-DRO move: improve
the worst entities rather than the average one. The count term blocks the
degenerate solution of rendering one excellent person for a 30-person prompt.

### Why the tail, empirically

Response of the three rewards to a degraded minority (32 entities, cloned
appearance and low confidence on the bad fraction):

| degraded fraction | counting | mean-aggregated | tail-aggregated |
|---|---|---|---|
| 0.00 | 1.000 | 0.990 | 0.980 |
| 0.06 | 1.000 | 0.979 | 0.928 |
| 0.12 | 1.000 | 0.966 | 0.870 |
| 0.19 | 1.000 | 0.953 | 0.809 |
| 0.31 | 1.000 | 0.925 | 0.779 |

The counting reward is flat — it cannot see the failure at all. Mean
aggregation moves by 0.04 across the range; tail aggregation by 0.20.

The tail-vs-mean gap is **non-monotonic** in the degraded fraction: it peaks
around 20–30% and closes again once most entities are bad, because at that
point the mean has caught the failure too. Tail aggregation buys sensitivity
precisely in the regime that matters for crowds — foreground fine, background
degraded.

## Layout

```
flow_grpo_entity/entity_reward.py   per-entity scoring, CVaR, count adherence
flow_grpo_entity/detector.py        YOLO + ReID adapter -> Detections
reward_server/entity_server.py      Flow-GRPO-compatible remote reward server
scripts/make_prompts.py             prompt grid; held-out N and scenes in test
scripts/inject_failures.py          reward sensitivity via planted failures
scripts/quantize_reward.py          int8 reward scorer; rank agreement, not MAE
scripts/eval_hacking.py             training reward vs held-out entity quality
tests/test_entity_reward.py         reward ordering on synthetic detections
```

## Running

```bash
pytest tests/ -q                       # 16 tests, no GPU required
python scripts/make_prompts.py
python reward_server/entity_server.py --mode entity --port 8001
# ... and the matching --mode count server for the baseline
```

Training uses Flow-GRPO's own launcher with two configs differing *only* in the
reward endpoint. Fits a 12 GB GPU: SD3.5-M, LoRA rank 16, 512px, bf16,
`text_encoder_3=None`, gradient checkpointing, group size 4,
`sde_window_size=1` (Flow-GRPO-Fast).

## Validation before training

`scripts/inject_failures.py` plants identity swaps, clones, merges and vanishes
into MOT20 ground truth at controlled rates and durations, then measures how
often the per-entity score drops for the tampered entities. The output is a
recall curve — how much of a known failure this reward can actually see. Without
it, any failure rate reported on generated images confounds generator failure
with detector failure.

## Quantizing the reward

The reward model runs on every rollout. `scripts/quantize_reward.py` measures
throughput alongside **Kendall tau within GRPO groups** — because GRPO
normalises rewards inside a group, only the ordering affects the advantage. A
quantized reward with unchanged mean but shuffled within-group ranking is
optimising a corrupted objective, and mean absolute error will not show it.

## What this does not claim

- Images at 512px, not video, audio or 3D. One base model (SD3.5-M), LoRA only.
- The detector is the instrument; entities it cannot see are not scored. The
  injection study bounds this, it does not eliminate it.
- Term weights in `EntityRewardConfig` are a starting point, not tuned.
- Synthetic embeddings are used in the injection study by default so that
  injection effects are isolated from detector behaviour; rerun with real ReID
  features before quoting the recall numbers as end-to-end.

## Credit

Flow-GRPO (Liu et al.) for the RL framework and the ODE-to-SDE formulation.
GenEval for the counting baseline. MOT20 for the calibration footage.
