from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image
from ..transforms.five_channel_conversion import AddMultiChannel
from ..transforms.letterbox_resize import LetterboxResize

from ..models.postprocess import build_predictions


def preprocess_numpy(imgs):
    """PIL crops → a single (N, 5, 320, 320) float32 batch, ready for an onnx feed.

    AddMultiChannel already returns a (5, 320, 320) float32 ndarray, so we
    letterbox + channel-stack each crop, then stack the batch dim on top.
    onnxruntime needs one contiguous ndarray, not a Python list of arrays.
    """
    seq = imgs if isinstance(imgs, (list, tuple)) else [imgs]
    letterbox = LetterboxResize(320)
    to_channels = AddMultiChannel()
    return np.stack([to_channels(letterbox(img)) for img in seq]).astype(np.float32)


class BaseClassInference:
    def __init__(self, crop_margin: float = 0.15):
        self.crop_margin = crop_margin

    def _crop_with_margin(self, image, bbox):
        """Crop `image` (PIL, RGB) to the detector box expanded by self.crop_margin.

        bbox is the JSONB `proposed_bbox`: absolute pixels {"x1","y1","x2","y2"},
        already clipped to image bounds upstream in bbox_inference.py.

        Goal: return image.crop((x1', y1', x2', y2')) where each side is pushed out
        by self.crop_margin * box_width (x) / box_height (y), then clamped to
        [0, W] / [0, H]. Context helps fine-grained ID (recovers clipped fins);
        too much reintroduces the full-frame problem this crop is meant to fix.
        """
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        w, h = image.size
        box_w, box_h = x2 - x1, y2 - y1
        x1 = max(x1 - box_w * self.crop_margin, 0)
        y1 = max(y1 - box_h * self.crop_margin, 0)
        x2 = min(x2 + box_w * self.crop_margin, w)
        y2 = min(y2 + box_h * self.crop_margin, h)
        return image.crop((int(x1), int(y1), int(x2), int(y2)))

    def infer(self, data: bytes | list[bytes], bbox: list[dict[str, float]]):

        imgs_to_infer = []
        if isinstance(data, list):
            for img_bytes in data:
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                imgs_to_infer.append(img)
        else:
            img = Image.open(BytesIO(data)).convert("RGB")
            imgs_to_infer.append(img)

        # A None box means the detector found no fish. Rather than failing, fall
        # back to classifying the full frame — out-of-distribution (the model is
        # trained on crops), so the caller flags these as low-trust downstream.
        prepared_images = [
            self._crop_with_margin(img, box) if box is not None else img
            for img, box in zip(imgs_to_infer, bbox)
        ]
        return self._run(prepared_images)

    def _run(self, prepared_images): ...


class ClassInference(BaseClassInference):
    def __init__(self, model: Path = None):
        super().__init__()
        # Lazy import: Classifier pulls torch. Keeping it out of module scope lets
        # the slim torch-free serving container import OnnxClassInference (below)
        # without the training stack. This torch path is the parity oracle now.
        from ..models.classifier import Classifier

        self.weights = model
        self.model = Classifier()

    def _run(self, prepared_images):
        return self.model.predict(images=prepared_images, weights=self.weights)


class OnnxClassInference(BaseClassInference):
    """Torch-free classifier inference: an onnxruntime session over the exported
    (INT8) classifier ONNX. Built once at app startup; `_run` feeds a stacked
    numpy batch and hands the three raw logit heads to the numpy postprocess."""

    def __init__(self, onnx_path, providers=("CPUExecutionProvider",)):
        super().__init__()
        # Lazy import so this module (and the torch ClassInference above) don't
        # pay for onnxruntime unless the ONNX path is actually constructed.
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=list(providers))

    def _run(self, prepared_images):
        batch = preprocess_numpy(prepared_images)  # (N, 5, 320, 320) float32
        species, genus, family = self.session.run(None, {"input": batch})
        return build_predictions(species, genus, family)
