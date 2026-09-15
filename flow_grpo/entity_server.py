"""Remote reward server, matching Flow-GRPO's ddpo-style reward-server pattern.

Run in its own environment so detector/ReID dependencies never collide with the
trainer's diffusers/PEFT stack.

    python reward_server/entity_server.py --port 8001 --mode entity

Modes:
  entity  -- per-entity score, CVaR tail aggregation + count adherence
  mean    -- same per-entity scores, mean aggregation (aggregation ablation)
  count   -- GenEval-style counting only (the hackable baseline)
"""
from __future__ import annotations
import argparse, base64, io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

from flow_grpo_entity.entity_reward import (
    EntityRewardConfig, entity_reward, mean_baseline_reward, count_only_reward)
from flow_grpo_entity.detector import PersonScorer

app = Flask(__name__)
STATE = {}


def _decode(b64):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


@app.route("/reward", methods=["POST"])
def reward():
    payload = request.get_json()
    scorer, cfg, mode = STATE["scorer"], STATE["cfg"], STATE["mode"]
    out = []
    for item in payload["items"]:
        img = _decode(item["image"])
        n_req = int(item["n_requested"])
        det = scorer(img)
        if mode == "count":
            out.append({"reward": count_only_reward(det, n_req, cfg), "per_entity": []})
        elif mode == "mean":
            out.append({"reward": mean_baseline_reward(det, n_req, cfg), "per_entity": []})
        else:
            b = entity_reward(det, n_req, cfg)
            out.append({"reward": b.reward, "tail": b.tail, "count_term": b.count_term,
                        "n_detected": b.n_detected,
                        "per_entity": b.per_entity.round(4).tolist()})
    return jsonify({"rewards": out})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--mode", choices=["entity", "mean", "count"], default="entity")
    ap.add_argument("--det-weights", default="yolov8n.pt")
    ap.add_argument("--reid-weights", default=None)
    ap.add_argument("--int8", action="store_true", help="load the quantized scorer")
    ap.add_argument("--cvar-alpha", type=float, default=0.2)
    a = ap.parse_args()
    STATE["cfg"] = EntityRewardConfig(cvar_alpha=a.cvar_alpha)
    STATE["mode"] = a.mode
    STATE["scorer"] = PersonScorer(a.det_weights, reid_weights=a.reid_weights,
                                   half=not a.int8)
    app.run(host="0.0.0.0", port=a.port)
