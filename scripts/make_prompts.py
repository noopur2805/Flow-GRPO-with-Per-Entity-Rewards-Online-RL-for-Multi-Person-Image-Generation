"""Build the crowd prompt grid.

Held-out test set uses N values and scenes unseen in training, so the hacking
analysis is measured out of distribution rather than on memorised prompts.
"""
import itertools, random, pathlib

TRAIN_N = [5, 15, 30]
TEST_N = [8, 22]                      # held out
TRAIN_SCENES = ["a city plaza", "a train station concourse", "a market street"]
TEST_SCENES = ["a stadium entrance", "a museum lobby"]   # held out
MOTION = ["standing and talking", "walking in the same direction",
          "crossing in two directions"]


def build(ns, scenes, seed):
    rng = random.Random(seed)
    rows = [f"a photo of exactly {n} people {m} in {s}"
            for n, s, m in itertools.product(ns, scenes, MOTION)]
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    d = pathlib.Path("dataset/crowd"); d.mkdir(parents=True, exist_ok=True)
    tr, te = build(TRAIN_N, TRAIN_SCENES, 0), build(TEST_N, TEST_SCENES, 1)
    (d / "train.txt").write_text("\n".join(tr) + "\n")
    (d / "test.txt").write_text("\n".join(te) + "\n")
    print(f"train: {len(tr)}  test: {len(te)}")
