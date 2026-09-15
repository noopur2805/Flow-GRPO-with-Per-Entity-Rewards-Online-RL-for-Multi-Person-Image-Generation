"""Detector + ReID adapter: image -> Detections.

Kept behind a thin interface so the scorer can be swapped or quantized without
touching the reward. `scripts/quantize_reward.py` relies on this separation.
"""
from __future__ import annotations
import numpy as np
from .entity_reward import Detections


class PersonScorer:
    """YOLO person detection + appearance embeddings.

    Embeddings come from a ReID backbone if torchreid/boxmot is available;
    otherwise falls back to a resized-crop colour histogram, which is weak but
    keeps the pipeline runnable for smoke tests.
    """

    def __init__(self, det_weights="yolov8n.pt", conf=0.25, device="cuda",
                 reid_weights=None, half=True):
        from ultralytics import YOLO
        self.model = YOLO(det_weights)
        self.conf, self.device, self.half = conf, device, half
        self.reid = None
        if reid_weights is not None:
            from boxmot.appearance.reid_auto_backend import ReidAutoBackend
            self.reid = ReidAutoBackend(weights=reid_weights, device=device,
                                        half=half).model

    def __call__(self, image: np.ndarray) -> Detections:
        r = self.model.predict(image, conf=self.conf, classes=[0],
                               device=self.device, half=self.half, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        emb = self._embed(image, boxes)
        return Detections(boxes, scores, emb, image.shape[:2])

    def _embed(self, image, boxes):
        if len(boxes) == 0:
            return np.zeros((0, 16))
        if self.reid is not None:
            return np.asarray(self.reid.get_features(boxes, image), dtype=np.float64)
        return np.stack([self._hist(image, b) for b in boxes])

    @staticmethod
    def _hist(image, box, bins=8):
        x1, y1, x2, y2 = [int(max(v, 0)) for v in box]
        crop = image[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
        if crop.size == 0:
            return np.zeros(bins * 3)
        h = [np.histogram(crop[..., c], bins=bins, range=(0, 255))[0] for c in range(3)]
        v = np.concatenate(h).astype(np.float64)
        return v / max(v.sum(), 1e-12)
