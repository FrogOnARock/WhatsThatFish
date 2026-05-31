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

from whatsthatfish.inference.bbox_inference import BoundingBoxInference


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
    with patch("whatsthatfish.inference.bbox_inference.YOLO"):
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
    # Single-image input returns list[dict] (length 1) after the batch refactor.

    def test_returns_list_with_required_keys(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([[10.0, 10.0, 50.0, 60.0]], [0.8]))
        result = inferrer.infer(_make_image_bytes())
        assert len(result) == 1
        assert set(result[0].keys()) == {"x1", "y1", "x2", "y2", "conf"}

    def test_selects_highest_confidence_box(self, inferrer):
        boxes = _FakeBoxes(
            [[5.0, 5.0, 30.0, 40.0], [20.0, 20.0, 70.0, 60.0]],
            [0.4, 0.9],
        )
        _set_predictions(inferrer, boxes)
        result = inferrer.infer(_make_image_bytes(w=100, h=80))
        assert result[0]["conf"] == pytest.approx(0.9, abs=0.01)
        assert result[0]["x1"] == pytest.approx(20.0, abs=0.01)

    def test_clips_x1_y1_to_zero(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([[-15.0, -8.0, 50.0, 40.0]], [0.7]))
        result = inferrer.infer(_make_image_bytes(w=100, h=80))
        assert result[0]["x1"] == 0.0
        assert result[0]["y1"] == 0.0

    def test_clips_x2_y2_to_image_dimensions(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([[10.0, 10.0, 200.0, 300.0]], [0.7]))
        result = inferrer.infer(_make_image_bytes(w=100, h=80))
        assert result[0]["x2"] == pytest.approx(100.0, abs=0.01)
        assert result[0]["y2"] == pytest.approx(80.0, abs=0.01)

    def test_valid_box_coordinates_unchanged(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([[10.0, 15.0, 60.0, 55.0]], [0.85]))
        result = inferrer.infer(_make_image_bytes(w=100, h=80))
        assert result[0]["x1"] == pytest.approx(10.0, abs=0.01)
        assert result[0]["y1"] == pytest.approx(15.0, abs=0.01)
        assert result[0]["x2"] == pytest.approx(60.0, abs=0.01)
        assert result[0]["y2"] == pytest.approx(55.0, abs=0.01)

    def test_conf_value_matches_best_box(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([[5.0, 5.0, 50.0, 50.0]], [0.63]))
        result = inferrer.infer(_make_image_bytes())
        assert result[0]["conf"] == pytest.approx(0.63, abs=0.01)

    def test_single_box_is_selected(self, inferrer):
        _set_predictions(inferrer, _FakeBoxes([[10.0, 10.0, 40.0, 40.0]], [0.55]))
        result = inferrer.infer(_make_image_bytes())
        assert result is not None


# ════════════════════════════════════════════════════════════════════════════════
# Batch inference
# ════════════════════════════════════════════════════════════════════════════════

def _set_batch_predictions(inferrer, box_sets: list):
    """Set mock predict return for a batch; each element is _FakeBoxes or None."""
    mock_results = []
    for boxes in box_sets:
        r = MagicMock()
        r.boxes = boxes
        mock_results.append(r)
    inferrer.model.predict.return_value = mock_results


class TestBatchInference:

    def test_all_detections_returns_list_of_dicts(self, inferrer):
        _set_batch_predictions(inferrer, [
            _FakeBoxes([[10.0, 10.0, 50.0, 50.0]], [0.8]),
            _FakeBoxes([[5.0, 5.0, 30.0, 30.0]], [0.7]),
        ])
        result = inferrer.infer([_make_image_bytes(), _make_image_bytes()])
        assert len(result) == 2
        assert all(r is not None for r in result)
        assert set(result[0].keys()) == {"x1", "y1", "x2", "y2", "conf"}

    def test_no_detection_in_one_image_yields_none_in_correct_position(self, inferrer):
        _set_batch_predictions(inferrer, [
            _FakeBoxes([[10.0, 10.0, 50.0, 50.0]], [0.8]),
            _FakeBoxes([], []),
        ])
        result = inferrer.infer([_make_image_bytes(), _make_image_bytes()])
        assert result[0] is not None
        assert result[1] is None

    def test_all_no_detections_returns_list_of_nones(self, inferrer):
        _set_batch_predictions(inferrer, [
            _FakeBoxes([], []),
            _FakeBoxes([], []),
            _FakeBoxes([], []),
        ])
        result = inferrer.infer([_make_image_bytes()] * 3)
        assert result == [None, None, None]

    def test_selects_best_confidence_per_image_independently(self, inferrer):
        _set_batch_predictions(inferrer, [
            _FakeBoxes([[1.0, 1.0, 10.0, 10.0], [5.0, 5.0, 50.0, 50.0]], [0.4, 0.9]),
            _FakeBoxes([[2.0, 2.0, 20.0, 20.0], [8.0, 8.0, 40.0, 40.0]], [0.6, 0.3]),
        ])
        result = inferrer.infer([_make_image_bytes(w=100, h=80)] * 2)
        assert result[0]["conf"] == pytest.approx(0.9, abs=0.01)
        assert result[1]["conf"] == pytest.approx(0.6, abs=0.01)

    def test_batch_clips_coordinates_to_respective_image_dimensions(self, inferrer):
        _set_batch_predictions(inferrer, [
            _FakeBoxes([[-5.0, -5.0, 200.0, 300.0]], [0.9]),
            _FakeBoxes([[-5.0, -5.0, 200.0, 300.0]], [0.9]),
        ])
        result = inferrer.infer([
            _make_image_bytes(w=100, h=80),
            _make_image_bytes(w=60, h=40),
        ])
        assert result[0]["x2"] == pytest.approx(100.0, abs=0.01)
        assert result[0]["y2"] == pytest.approx(80.0, abs=0.01)
        assert result[1]["x2"] == pytest.approx(60.0, abs=0.01)
        assert result[1]["y2"] == pytest.approx(40.0, abs=0.01)

    def test_none_boxes_attribute_yields_none_entry(self, inferrer):
        _set_batch_predictions(inferrer, [
            _FakeBoxes([[10.0, 10.0, 50.0, 50.0]], [0.8]),
            None,  # boxes attribute is None
        ])
        # Manually set second result's boxes to None
        inferrer.model.predict.return_value[1].boxes = None
        result = inferrer.infer([_make_image_bytes(), _make_image_bytes()])
        assert result[0] is not None
        assert result[1] is None
