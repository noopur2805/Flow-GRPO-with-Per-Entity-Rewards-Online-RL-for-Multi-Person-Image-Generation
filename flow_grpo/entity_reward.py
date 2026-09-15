"""Per-entity reward with tail aggregation, for Flow-GRPO post-training.

Flow-GRPO's shipped rewards (GenEval counting, OCR, PickScore) are *aggregate*:
one scalar per image, computed over the whole frame. On multi-person prompts an
aggregate reward is cheap to satisfy in degenerate ways -- N detectable blobs
scores the same as N distinct people.

This module scores each detected entity separately, then aggregates over the
*worst* fraction of entities (CVaR) rather than the mean, so a handful of clean
foreground subjects cannot mask a degraded background.

Design notes
------------
* `entity_scores` returns the per-entity vector as well as the scalar, so the
  vector can be logged (for hacking analysis) and, later, used for
  entity-masked advantage attribution.
* Nothing here imports torch or a detector. Detections come in as plain arrays,
  which keeps the reward unit-testable without a GPU and lets the detector be
  swapped or quantized independently (see scripts/quantize_reward.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Detections", "EntityRewardConfig", "entity_scores", "entity_reward"]


@dataclass
class Detections:
    """Detections for a single generated image.

    boxes:      (n, 4) float array of xyxy boxes in pixel coordinates.
    scores:     (n,)   detector confidence in [0, 1].
    embeddings: (n, d) appearance embeddings, one per detection. Need not be
                unit norm; cosine distance is computed after normalisation.
    image_hw:   (H, W) of the image the detections came from.
    """

    boxes: np.ndarray
    scores: np.ndarray
    embeddings: np.ndarray
    image_hw: tuple[int, int]

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float64)
        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings.reshape(len(self.scores), -1)
        n = len(self.boxes)
        if not (len(self.scores) == len(self.embeddings) == n):
            raise ValueError("boxes, scores and embeddings must have equal length")

    def __len__(self) -> int:
        return len(self.boxes)


@dataclass
class EntityRewardConfig:
    """Weights and thresholds. Defaults are a starting point, not tuned."""

    # per-entity term weights (need not sum to 1; the score is renormalised)
    w_confidence: float = 1.0
    w_distinctness: float = 1.0
    w_separation: float = 1.0
    w_scale: float = 0.5

    # tail aggregation
    cvar_alpha: float = 0.2          # average over the worst 20% of entities
    min_tail_entities: int = 1

    # count adherence
    w_count: float = 1.0             # weight of the count term in the final reward
    count_tolerance: float = 1.0     # softness of the exponential penalty

    # thresholds
    distinct_margin: float = 0.35    # cosine distance at which two entities are
                                     # considered fully distinct
    iou_merge_threshold: float = 0.5 # IoU above which two boxes count as merged
    min_box_frac: float = 0.002      # box area below this frac of the image is
                                     # implausibly small -> penalised
    max_box_frac: float = 0.35

    # returned when no entity is detected at all
    empty_reward: float = 0.0

    def tail_k(self, n: int) -> int:
        return max(self.min_tail_entities, int(np.ceil(self.cvar_alpha * n)))


def _pairwise_cosine_distance(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    unit = emb / norms
    sim = np.clip(unit @ unit.T, -1.0, 1.0)
    return 1.0 - sim


def _pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    union = area[:, None] + area[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def entity_scores(det: Detections, cfg: EntityRewardConfig | None = None) -> np.ndarray:
    """Per-entity quality scores in [0, 1], one per detection.

    Four terms, each in [0, 1]:

    confidence   -- detector confidence. A person the detector is unsure about
                    is usually a person the generator rendered badly.
    distinctness -- cosine distance to the *nearest other* entity's embedding,
                    normalised by `distinct_margin`. Catches clone collapse,
                    where the generator emits N copies of one person.
    separation   -- 1 - max IoU with any other box. Catches merged bodies, the
                    dominant crowd failure reported for video generators.
    scale        -- box area as a fraction of the image, penalised outside a
                    plausible band. Catches blob artefacts that a detector
                    fires on but a human would not call a person.
    """
    cfg = cfg or EntityRewardConfig()
    n = len(det)
    if n == 0:
        return np.zeros(0)

    conf = np.clip(det.scores, 0.0, 1.0)

    if n == 1:
        distinct = np.ones(1)
        separation = np.ones(1)
    else:
        dist = _pairwise_cosine_distance(det.embeddings)
        np.fill_diagonal(dist, np.inf)
        nearest = dist.min(axis=1)
        distinct = np.clip(nearest / max(cfg.distinct_margin, 1e-12), 0.0, 1.0)

        iou = _pairwise_iou(det.boxes)
        np.fill_diagonal(iou, 0.0)
        worst_iou = iou.max(axis=1)
        # linear falloff: no penalty at IoU 0, zero score at the merge threshold
        separation = np.clip(1.0 - worst_iou / max(cfg.iou_merge_threshold, 1e-12), 0.0, 1.0)

    h, w = det.image_hw
    img_area = max(float(h) * float(w), 1e-12)
    box_area = np.clip(det.boxes[:, 2] - det.boxes[:, 0], 0, None) * np.clip(
        det.boxes[:, 3] - det.boxes[:, 1], 0, None
    )
    frac = box_area / img_area
    scale = np.ones(n)
    too_small = frac < cfg.min_box_frac
    too_large = frac > cfg.max_box_frac
    scale[too_small] = np.clip(frac[too_small] / max(cfg.min_box_frac, 1e-12), 0.0, 1.0)
    scale[too_large] = np.clip(
        (1.0 - frac[too_large]) / max(1.0 - cfg.max_box_frac, 1e-12), 0.0, 1.0
    )

    weights = np.array(
        [cfg.w_confidence, cfg.w_distinctness, cfg.w_separation, cfg.w_scale],
        dtype=np.float64,
    )
    terms = np.stack([conf, distinct, separation, scale], axis=1)
    return (terms @ weights) / max(weights.sum(), 1e-12)


def cvar(scores: np.ndarray, alpha: float, min_k: int = 1) -> float:
    """Mean of the worst ceil(alpha * n) scores.

    This is the lower-tail CVaR of the per-entity score distribution. Optimising
    it is the Group-DRO / risk-sensitive RL move: improve the worst entities
    rather than the average one.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    k = max(min_k, int(np.ceil(alpha * n)))
    k = min(k, n)
    return float(np.sort(scores)[:k].mean())


def count_adherence(n_detected: int, n_requested: int, tolerance: float = 1.0) -> float:
    """Exponential penalty on relative count error, in (0, 1]."""
    if n_requested <= 0:
        return 1.0 if n_detected == 0 else 0.0
    rel = abs(n_detected - n_requested) / float(n_requested)
    return float(np.exp(-rel / max(tolerance, 1e-12)))


@dataclass
class RewardBreakdown:
    reward: float
    per_entity: np.ndarray = field(repr=False)
    tail: float = 0.0
    count_term: float = 0.0
    n_detected: int = 0
    n_requested: int = 0


def entity_reward(
    det: Detections,
    n_requested: int,
    cfg: EntityRewardConfig | None = None,
) -> RewardBreakdown:
    """Scalar reward for one generated image, plus the per-entity breakdown.

    reward = tail_score ** (1 - beta) blended with count adherence, where the
    blend is a weighted geometric-style mix implemented additively for a
    well-behaved gradient-free signal:

        reward = (tail + w_count * count) / (1 + w_count)

    Both terms are in [0, 1], so the reward is too. The count term is *not*
    sufficient on its own -- that is exactly the baseline this reward is meant
    to improve on -- but dropping it entirely lets the policy satisfy the tail
    by emitting one well-rendered person for a 30-person prompt.
    """
    cfg = cfg or EntityRewardConfig()
    per_entity = entity_scores(det, cfg)
    n_det = len(per_entity)

    if n_det == 0:
        return RewardBreakdown(
            reward=cfg.empty_reward,
            per_entity=per_entity,
            tail=0.0,
            count_term=count_adherence(0, n_requested, cfg.count_tolerance),
            n_detected=0,
            n_requested=n_requested,
        )

    tail = cvar(per_entity, cfg.cvar_alpha, cfg.min_tail_entities)
    count_term = count_adherence(n_det, n_requested, cfg.count_tolerance)
    reward = (tail + cfg.w_count * count_term) / (1.0 + cfg.w_count)

    return RewardBreakdown(
        reward=float(reward),
        per_entity=per_entity,
        tail=float(tail),
        count_term=float(count_term),
        n_detected=n_det,
        n_requested=n_requested,
    )


def mean_baseline_reward(
    det: Detections, n_requested: int, cfg: EntityRewardConfig | None = None
) -> float:
    """Same per-entity scores, aggregated by *mean* instead of tail.

    This is the ablation that isolates the aggregation choice from the scoring
    choice. If mean and CVaR behave identically, tail aggregation is not what is
    doing the work and the claim should be dropped.
    """
    cfg = cfg or EntityRewardConfig()
    per_entity = entity_scores(det, cfg)
    if len(per_entity) == 0:
        return cfg.empty_reward
    count_term = count_adherence(len(per_entity), n_requested, cfg.count_tolerance)
    return float((per_entity.mean() + cfg.w_count * count_term) / (1.0 + cfg.w_count))


def count_only_reward(
    det: Detections, n_requested: int, cfg: EntityRewardConfig | None = None
) -> float:
    """GenEval-style counting baseline: did the detector find N people?

    Deliberately blind to whether those N are distinct, separated or plausible.
    This is the reward hypothesised to be hackable.
    """
    cfg = cfg or EntityRewardConfig()
    return count_adherence(len(det), n_requested, cfg.count_tolerance)
