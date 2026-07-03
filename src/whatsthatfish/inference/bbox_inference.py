"""Thin wrapper around a trained YOLO detector.

Takes raw image bytes (single or batched), runs detection, and returns the
single highest-confidence box per image, clipped to the image bounds. Returns
None for an image with no detections so callers can treat it as a negative.
"""

import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _letterbox(img: Image.Image, new_shape: int = 640, color: int = 114):
    """Resize a PIL RGB image into a square `new_shape` canvas, aspect-ratio
    preserving, gray-padded — matching Ultralytics' inference LetterBox so the
    ONNX sees the same pixels the torch predict() path fed it.

    Returns (canvas HxWx3 uint8, gain, (pad_x, pad_y)) — the gain and pads are
    exactly what a caller needs to map model-space boxes back to the original
    frame.
    """
    w0, h0 = img.size
    gain = min(new_shape / w0, new_shape / h0)
    new_w, new_h = round(w0 * gain), round(h0 * gain)
    resized = np.asarray(img.resize((new_w, new_h), Image.BILINEAR))
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    pad_x = (new_shape - new_w) // 2
    pad_y = (new_shape - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, gain, (pad_x, pad_y)


class BoundingBoxInference:
    """Loads a YOLO detect model once and serves best-box predictions.

    The classifier pipeline only ever wants the most confident fish per frame,
    so this collapses YOLO's full detection set down to that one box.
    """

    def __init__(self, conf: float, model=None):
        # Lazy import: keeps torch + Ultralytics out of the module-import path so
        # the slim serving container can import OnnxBoundingBoxInference (below)
        # without dragging in the training stack. This torch path is now the
        # PARITY ORACLE for the ONNX path, not the production server.
        from ..models.detection import Detector, Dataset

        model = model or Dataset.LC1
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


class OnnxBoundingBoxInference:
    """Torch-free best-box detection over the exported (FP32) detector ONNX.

    Mirrors BoundingBoxInference.infer's contract exactly (same list-of-dict /
    None output) so it drops into the serving app in its place — but runs on
    onnxruntime + numpy only, no torch/Ultralytics. NMS is baked into the graph
    (output0 is (N, 300, 6): [x1, y1, x2, y2, conf, cls] in letterboxed-input
    pixel space, conf-sorted, zero-padded to 300 rows).
    """

    def __init__(
        self,
        onnx_path,
        conf: float,
        img_size: int = 640,
        providers=("CPUExecutionProvider",),
    ):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=list(providers))
        self.conf = conf
        self.img_size = img_size

    def _to_original(self, box, gain, pad, w, h):
        """Map ONE detection box from letterboxed model space back to the original
        frame, then clip to [0, w] x [0, h].
        """
        pad_x, pad_y = pad
        x1, y1, x2, y2 = (
            max(0, (box[0] - pad_x) / gain),
            max(0, (box[1] - pad_y) / gain),
            min(w, (box[2] - pad_x) / gain),
            min(h, (box[3] - pad_y) / gain),
        )

        return x1, y1, x2, y2

    def infer(self, data: bytes | list[bytes]):
        items = data if isinstance(data, list) else [data]

        canvases, meta = [], []
        for img_bytes in items:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            canvas, gain, pad = _letterbox(img, self.img_size)
            canvases.append(canvas)
            meta.append((img.size[0], img.size[1], gain, pad))  # (w, h, gain, pad)

        # (N, H, W, 3) uint8 → (N, 3, H, W) float32 in [0,1] — the /255 the torch
        # predict path applies; the export did NOT bake normalization in.
        batch = np.stack(canvases).astype(np.float32) / 255.0
        batch = batch.transpose(0, 3, 1, 2)

        dets = self.session.run(None, {"images": batch})[0]  # (N, 300, 6)

        out = []
        for i, (w, h, gain, pad) in enumerate(meta):
            rows = dets[i]
            rows = rows[rows[:, 4] >= self.conf]  # drop zero-pad + sub-threshold
            if rows.shape[0] == 0:
                out.append(None)
                continue
            best = rows[rows[:, 4].argmax()]
            x1, y1, x2, y2 = self._to_original(best[:4], gain, pad, w, h)
            out.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": float(best[4]),
                    "w": w,
                    "h": h,
                }
            )
        return out
