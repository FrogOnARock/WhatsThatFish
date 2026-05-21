import io
from PIL import Image
from ultralytics import YOLO

class BoundingBoxInference:
    def __init__(self, model: str, conf: float, type='train'):
        self.model = YOLO(model, task='detect')
        self.conf = conf
        self.type = type

    def infer(self, data: bytes):

        img = Image.open(io.BytesIO(data)).convert("RGB")
        W, H = img.size

        results = self.model.predict(img, conf=self.conf, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None

        if self.type == "train":
            best_idx = results.boxes.conf.argmax()
            x1, y1, x2, y2 = results.boxes.xyxy[best_idx].cpu().tolist()
            conf = results.boxes.conf[best_idx].item()
        else:
            raise NotImplementedError

        return {
            "x1": max(0.0, x1),
            "y1": max(0.0, y1),
            "x2": min(float(W), x2),
            "y2": min(float(H), y2),
            "conf": conf,
        }








