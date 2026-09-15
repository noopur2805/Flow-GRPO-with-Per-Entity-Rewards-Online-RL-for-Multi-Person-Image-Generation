"""Reward ordering tests on synthetic detections.

The point of these tests is to establish, before any GPU time is spent, that
the reward *orders* failure modes correctly. A reward that cannot separate a
clean 10-person scene from 10 merged bodies will happily train a policy toward
the latter, and no amount of RL tuning will reveal that.

Each test constructs detections with a known defect and asserts the reward
moves in the expected direction relative to a clean control.
"""

import numpy as np
import pytest

from flow_grpo_entity.entity_reward import (
    Detections,
    EntityRewardConfig,
    count_only_reward,
    cvar,
    entity_reward,
    entity_scores,
    mean_baseline_reward,
)

H, W = 480, 640
RNG = np.random.default_rng(0)


def _grid_boxes(n: int, box_w: int = 40, box_h: int = 90, stride: int = 60):
    """n non-overlapping boxes laid out left to right, wrapping into rows."""
    boxes = []
    per_row = max(1, (W - box_w) // stride)
    for i in range(n):
        r, c = divmod(i, per_row)
        x1 = 10 + c * stride
        y1 = 10 + r * (box_h + 10)
        boxes.append([x1, y1, x1 + box_w, y1 + box_h])
    return np.array(boxes, dtype=float)


def _distinct_embeddings(n: int, d: int = 16):
    """Near-orthogonal embeddings -> every entity distinct."""
    emb = RNG.normal(size=(n, d))
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def clean(n: int, conf: float = 0.95) -> Detections:
    return Detections(
        boxes=_grid_boxes(n),
        scores=np.full(n, conf),
        embeddings=_distinct_embeddings(n),
        image_hw=(H, W),
    )


# --------------------------------------------------------------------------
# per-entity scoring
# --------------------------------------------------------------------------


def test_clean_scene_scores_near_one():
    s = entity_scores(clean(10))
    assert s.shape == (10,)
    assert s.min() > 0.85, f"clean entities should score high, got min {s.min():.3f}"


def test_clone_collapse_penalised():
    """N copies of one person: high confidence, no overlap, identical appearance."""
    n = 10
    det = clean(n)
    shared = _distinct_embeddings(1)
    det.embeddings = np.repeat(shared, n, axis=0)

    cloned = entity_scores(det)
    control = entity_scores(clean(n))
    assert cloned.mean() < control.mean() - 0.15, (
        "clone collapse must be penalised via the distinctness term; "
        f"cloned {cloned.mean():.3f} vs clean {control.mean():.3f}"
    )


def test_merged_bodies_penalised():
    """Two entities occupying nearly the same box."""
    n = 10
    det = clean(n)
    det.boxes[1] = det.boxes[0] + np.array([4.0, 4.0, 4.0, 4.0])  # heavy overlap

    s = entity_scores(det)
    control = entity_scores(clean(n))[0]
    assert s[0] < control - 0.25 and s[1] < control - 0.25, (
        f"merged pair should score well below clean ({control:.3f}), got {s[:2]}"
    )
    assert s[2:].min() > 0.85, "unaffected entities must keep their scores"


def test_low_confidence_penalised():
    det = clean(10)
    det.scores[3] = 0.2
    s = entity_scores(det)
    assert s[3] < s[0] - 0.15


def test_implausible_scale_penalised():
    det = clean(10)
    det.boxes[5] = [100.0, 100.0, 103.0, 106.0]  # ~18 px in a 307k px image
    s = entity_scores(det)
    assert s[5] < s[0], "tiny blob should be penalised by the scale term"


def test_empty_detections():
    det = Detections(
        boxes=np.zeros((0, 4)),
        scores=np.zeros(0),
        embeddings=np.zeros((0, 16)),
        image_hw=(H, W),
    )
    assert len(entity_scores(det)) == 0
    assert entity_reward(det, n_requested=10).reward == 0.0


# --------------------------------------------------------------------------
# tail aggregation
# --------------------------------------------------------------------------


def test_cvar_is_mean_of_worst_k():
    scores = np.array([1.0, 0.9, 0.8, 0.1, 0.2])
    # alpha=0.4 over 5 entities -> k=2 -> mean of {0.1, 0.2}
    assert cvar(scores, alpha=0.4) == pytest.approx(0.15)


def test_cvar_never_exceeds_mean():
    for _ in range(50):
        s = RNG.uniform(size=RNG.integers(1, 40))
        assert cvar(s, alpha=0.2) <= s.mean() + 1e-9


def test_tail_exposes_what_mean_hides():
    """The central claim: a degraded *minority* among many good entities.

    A handful of background people are cloned and low-confidence while the rest
    render cleanly. The mean barely moves; the tail collapses.

    Note the gap is non-monotonic in the bad fraction (see README): it peaks
    around 20-30% degraded and closes again once most entities are bad, because
    at that point the mean has caught the failure too. Tail aggregation buys you
    sensitivity precisely in the regime that matters for crowds -- foreground
    fine, background degraded.
    """
    n_total, n_bad = 32, 6
    n_good = n_total - n_bad
    det = clean(n_total)
    det.embeddings[n_good:] = np.repeat(_distinct_embeddings(1), n_bad, axis=0)  # cloned
    det.scores[n_good:] = 0.45

    scores = entity_scores(det)
    mean_agg = scores.mean()
    tail_agg = cvar(scores, alpha=0.2)

    assert tail_agg < mean_agg - 0.2, (
        "tail must separate from mean when a subpopulation is degraded; "
        f"mean {mean_agg:.3f}, tail {tail_agg:.3f}"
    )


# --------------------------------------------------------------------------
# full reward, and the baseline it is meant to beat
# --------------------------------------------------------------------------


def test_counting_baseline_is_blind_to_quality():
    """The hackability hypothesis, stated as a test.

    Ten clean people and ten merged clones both satisfy "there are ten people".
    The counting reward cannot tell them apart; the entity reward must.
    """
    good = clean(10)

    bad = clean(10)
    bad.embeddings = np.repeat(_distinct_embeddings(1), 10, axis=0)
    bad.boxes = _grid_boxes(10, stride=14)  # heavy mutual overlap
    bad.scores = np.full(10, 0.6)

    assert count_only_reward(good, 10) == pytest.approx(count_only_reward(bad, 10)), (
        "counting reward should be identical for both -- that is the problem"
    )

    r_good = entity_reward(good, 10).reward
    r_bad = entity_reward(bad, 10).reward
    assert r_good > r_bad + 0.1, f"entity reward failed to separate: {r_good:.3f} vs {r_bad:.3f}"


def test_undercount_penalised():
    r_full = entity_reward(clean(20), n_requested=20).reward
    r_short = entity_reward(clean(5), n_requested=20).reward
    assert r_full > r_short, "reward must penalise generating too few people"


def test_single_well_rendered_person_does_not_win():
    """Guards the degenerate solution the count term exists to block."""
    r_one = entity_reward(clean(1), n_requested=30).reward
    r_many = entity_reward(clean(30), n_requested=30).reward
    assert r_many > r_one + 0.2


def test_tail_and_mean_rewards_diverge_on_skewed_scenes():
    n_good, n_bad = 3, 27
    det = clean(n_good + n_bad)
    det.embeddings[n_good:] = np.repeat(_distinct_embeddings(1), n_bad, axis=0)
    det.scores[n_good:] = 0.45

    r_tail = entity_reward(det, n_good + n_bad).reward
    r_mean = mean_baseline_reward(det, n_good + n_bad)
    assert r_tail < r_mean, "tail aggregation must be stricter than mean here"


def test_reward_in_unit_interval():
    for n in (0, 1, 5, 30):
        det = clean(n) if n else Detections(
            np.zeros((0, 4)), np.zeros(0), np.zeros((0, 16)), (H, W)
        )
        r = entity_reward(det, n_requested=max(n, 1)).reward
        assert 0.0 <= r <= 1.0


def test_breakdown_vector_length_matches_detections():
    out = entity_reward(clean(12), n_requested=12)
    assert len(out.per_entity) == 12
    assert out.n_detected == 12


def test_config_alpha_changes_strictness():
    det = clean(20)
    det.scores[:4] = 0.3
    strict = entity_reward(det, 20, EntityRewardConfig(cvar_alpha=0.2)).reward
    loose = entity_reward(det, 20, EntityRewardConfig(cvar_alpha=1.0)).reward
    assert strict < loose
