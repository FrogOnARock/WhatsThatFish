from pathlib import Path
from io import BytesIO
from PIL import Image

from ..models.classifier import Classifier


class ClassInference:
    def __init__(self, model: Path = None, crop_margin: float = 0.15):
        self.weights = model
        self.model = Classifier()
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

        imgs_to_crop = list(zip(imgs_to_infer, bbox))
        cropped_images = [
            self._crop_with_margin(img[0], img[1]) for img in imgs_to_crop
        ]
        results = self.model.predict(images=cropped_images, weights=self.weights)
        return results
