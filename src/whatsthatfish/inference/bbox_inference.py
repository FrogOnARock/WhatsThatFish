"""Thin wrapper around a trained YOLO detector.

Takes raw image bytes (single or batched), runs detection, and returns the
single highest-confidence box per image, clipped to the image bounds. Returns
None for an image with no detections so callers can treat it as a negative.
"""

import io
import logging

from PIL import Image
from ultralytics import YOLO
from ..models.detection import Detector, Dataset

logger = logging.getLogger(__name__)


class BoundingBoxInference:
    """Loads a YOLO detect model once and serves best-box predictions.

    The classifier pipeline only ever wants the most confident fish per frame,
    so this collapses YOLO's full detection set down to that one box.
    """

    def __init__(self, conf: float, model: Dataset = Dataset.LC1):
        logger.info("Loading YOLO model from %s (conf=%.2f)", model, conf)
        self.model = Detector(dataset=model)
        self.conf = conf

    def infer(self, data: bytes | list[bytes]):
        """Run detection and return the top-confidence box per image.

        Accepts one image's bytes or a list of them; always returns a list
        aligned to the input. Each entry is a dict of the best box (xyxy clipped
        to image bounds, plus conf and original w/h), or None when nothing was
        detected above the confidence threshold.
        """

        imgs_to_infer = []
        w = []
        h = []
        if isinstance(data, list):
            for img_bytes in data:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                w.append(img.size[0])
                h.append(img.size[1])
                imgs_to_infer.append(img)
        else:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            w.append(img.size[0])
            h.append(img.size[1])
            imgs_to_infer.append(img)

        results = self.model.predict(imgs_to_infer, conf=self.conf)

        if len(results) == 1:
            if results[0].boxes is None or len(results[0].boxes) == 0:
                return [None]

        def _has_boxes(result):
            return result.boxes is not None and len(result.boxes) > 0

        best_idx = [
            result.boxes.conf.argmax() if _has_boxes(result) else None
            for result in results
        ]
        xyxy = [
            result.boxes.xyxy[best_idx[i]].cpu().tolist()
            if _has_boxes(result)
            else [None] * 4
            for i, result in enumerate(results)
        ]
        conf = [
            result.boxes.conf[best_idx[i]].item() if _has_boxes(result) else None
            for i, result in enumerate(results)
        ]

        results = list(zip(xyxy, conf, w, h))

        return [
            {
                "x1": max(0.0, result[0][0]),
                "y1": max(0.0, result[0][1]),
                "x2": min(result[2], result[0][2]),
                "y2": min(result[3], result[0][3]),
                "conf": result[1],
                "w": result[2],
                "h": result[3],
            }
            if result[0][0] is not None
            else None
            for result in results
        ]
