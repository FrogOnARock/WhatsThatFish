"""
Tests for BoundingBoxInference.

YOLO model is mocked — no weights file required.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from whatsthatfish.src.inference.bbox_inference import BoundingBoxInference


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_image_bytes(w: int = 100, h: int = 80) -> bytes:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG")
    return buf.getvalue()


class _FakeBoxes:
    """Minimal stand-in for ultralytics Results.boxes with real tensors."""
    def __init__(self, xyxy: list[list[float]], conf: list[float]):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)

    def __len__(self):
        return len(self.conf)


@pytest.fixture
def inferrer():
    with patch("whatsthatfish.src.inference.bbox_inference.YOLO"):
        obj = BoundingBoxInference(model="fake.pt", conf=0.25)
    return obj


def _set_predictions(inferrer, boxes: _FakeBoxes | None):
    mock_result = MagicMock()
    mock_result.boxes = boxes
    inferrer.model.predict.return_value = [mock_result]


# ════════════════════════════════════════════════════════════════════════════════
# No-detection cases
# ════════════════════════════════════════════════════════════════════════════════

class TestNoDetection:

    def test_returns_none_when_boxes_is_none(self, inferrer):
        _set_predictions(inferrer, None)
        assert inferrer.infer(_make_image_bytes()) is None

    def test_returns_none_when_boxes_empty(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([], []))
        assert inferrer.infer(_make_image_bytes()) is None


# ════════════════════════════════════════════════════════════════════════════════
# Detection cases
# ════════════════════════════════════════════════════════════════════════════════

class TestDetection:

    def test_returns_dict_with_required_keys(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([[10.0, 10.0, 50.0, 60.0]], [0.8]))
        result = inferrer.infer(_make_image_bytes())
        assert set(result.keys()) == {"x1", "y1", "x2", "y2", "conf"}

    def test_selects_highest_confidence_box(self, inferrer):
        boxes = _FakeBoxes(
            [[5.0, 5.0, 30.0, 40.0], [20.0, 20.0, 70.0, 60.0]],
            [0.4, 0.9],
        )
        _set_predictions(inferrer, boxes)
        result = inferrer.infer(_make_image_bytes(w=100, h=80))
        assert result["conf"] == pytest.approx(0.9, abs=0.01)
        assert result["x1"] == pytest.approx(20.0, abs=0.01)

    def test_clips_x1_y1_to_zero(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([[-15.0, -8.0, 50.0, 40.0]], [0.7]))
        result = inferrer.infer(_make_image_bytes(w=100, h=80))
        assert result["x1"] == 0.0
        assert result["y1"] == 0.0

    def test_clips_x2_y2_to_image_dimensions(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([[10.0, 10.0, 200.0, 300.0]], [0.7]))
        result = inferrer.infer(_make_image_bytes(w=100, h=80))
        assert result["x2"] == pytest.approx(100.0, abs=0.01)
        assert result["y2"] == pytest.approx(80.0, abs=0.01)

    def test_valid_box_coordinates_unchanged(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([[10.0, 15.0, 60.0, 55.0]], [0.85]))
        result = inferrer.infer(_make_image_bytes(w=100, h=80))
        assert result["x1"] == pytest.approx(10.0, abs=0.01)
        assert result["y1"] == pytest.approx(15.0, abs=0.01)
        assert result["x2"] == pytest.approx(60.0, abs=0.01)
        assert result["y2"] == pytest.approx(55.0, abs=0.01)

    def test_conf_value_matches_best_box(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([[5.0, 5.0, 50.0, 50.0]], [0.63]))
        result = inferrer.infer(_make_image_bytes())
        assert result["conf"] == pytest.approx(0.63, abs=0.01)

    def test_single_box_is_selected(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([[10.0, 10.0, 40.0, 40.0]], [0.55]))
        result = inferrer.infer(_make_image_bytes())
        assert result is not None
