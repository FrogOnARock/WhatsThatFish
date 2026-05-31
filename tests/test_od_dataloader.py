"""
Tests for object_detection_collate and ObjectDetectionDataset.

collate tests: pure tensor logic, no mocking needed.
dataset tests: DB and GCS dependencies mocked.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from whatsthatfish.models.data.od_dataloader import object_detection_collate
from whatsthatfish.models.data.od_dataset import ObjectDetectionDataset


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_image_bytes(h: int = 64, w: int = 64) -> bytes:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _image_tensor(h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.rand(3, h, w)


def _label_tensor(*rows) -> torch.Tensor:
    """Build an (N, 5) label tensor from rows of [class_id, x, y, w, h]."""
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(list(rows), dtype=torch.float32)


def _batch_item(label_rows=(), fname="test.jpg"):
    """Build a single (image, labels, filename) batch item."""
    return (_image_tensor(), _label_tensor(*label_rows), fname)


# ── Fake DB row ────────────────────────────────────────────────────────────────

def _fake_row(file_name: str, uiqm: float, annotations: list[dict]):
    row = MagicMock()
    row.file_name = file_name
    row.uiqm = uiqm
    row.annotation = annotations
    return row


# ════════════════════════════════════════════════════════════════════════════════
# object_detection_collate
# ════════════════════════════════════════════════════════════════════════════════

class TestCollate:

    def test_image_stack_shape(self):
        batch = [_batch_item() for _ in range(4)]
        result = object_detection_collate(batch)
        assert result["img"].shape == (4, 3, 64, 64)

    def test_batch_idx_assigned_per_image(self):
        batch = [
            _batch_item(([0, 0.5, 0.5, 0.2, 0.2],), "a.jpg"),
            _batch_item(([0, 0.3, 0.3, 0.1, 0.1],), "b.jpg"),
        ]
        result = object_detection_collate(batch)
        assert result["batch_idx"][0].item() == 0.0
        assert result["batch_idx"][1].item() == 1.0

    def test_bboxes_and_cls_shape(self):
        batch = [
            _batch_item(([0, 0.5, 0.5, 0.2, 0.2], [0, 0.1, 0.1, 0.05, 0.05]), "a.jpg"),
            _batch_item(([0, 0.3, 0.3, 0.1, 0.1],), "b.jpg"),
        ]
        result = object_detection_collate(batch)
        assert result["bboxes"].shape == (3, 4)
        assert result["cls"].shape == (3, 1)

    def test_all_negative_batch_returns_empty(self):
        batch = [_batch_item() for _ in range(4)]
        result = object_detection_collate(batch)
        assert result["bboxes"].shape == (0, 4)
        assert result["cls"].shape == (0, 1)
        assert result["batch_idx"].shape == (0,)

    def test_mixed_positive_negative_batch(self):
        batch = [
            _batch_item(([0, 0.5, 0.5, 0.2, 0.2],), "a.jpg"),
            _batch_item((), "b.jpg"),
            _batch_item(([0, 0.3, 0.3, 0.1, 0.1],), "c.jpg"),
        ]
        result = object_detection_collate(batch)
        assert result["bboxes"].shape == (2, 4)
        assert set(result["batch_idx"].tolist()) == {0.0, 2.0}

    def test_bbox_values_preserved(self):
        row = [0.0, 0.5, 0.5, 0.2, 0.2]
        batch = [_batch_item((row,), "a.jpg")]
        result = object_detection_collate(batch)
        assert torch.allclose(result["bboxes"][0], torch.tensor(row[1:], dtype=torch.float32))

    def test_im_file_list_matches_batch(self):
        batch = [_batch_item(fname=f"f{i}.jpg") for i in range(3)]
        result = object_detection_collate(batch)
        assert result["im_file"] == ["f0.jpg", "f1.jpg", "f2.jpg"]

    def test_result_keys_are_complete(self):
        batch = [_batch_item()]
        result = object_detection_collate(batch)
        assert {"img", "batch_idx", "cls", "bboxes", "im_file", "ori_shape", "ratio_pad"}.issubset(result.keys())


# ════════════════════════════════════════════════════════════════════════════════
# ObjectDetectionDataset
# ════════════════════════════════════════════════════════════════════════════════

FISH_ANN = [{"class_id": 0, "is_train": True, "norm_center_x": 0.5, "norm_center_y": 0.5, "norm_width": 0.2, "norm_height": 0.2}]
NEG_ANN  = [{"class_id": 1, "is_train": True, "norm_center_x": 0.0, "norm_center_y": 0.0, "norm_width": 0.0, "norm_height": 0.0}]

FAKE_ROWS = [
    _fake_row("fish_001.jpg", 0.8, FISH_ANN),
    _fake_row("neg_001.jpg",  0.4, NEG_ANN),
    _fake_row("fish_002.jpg", 0.6, FISH_ANN),
]

# v2 joint transform: accepts (PIL, BoundingBoxes) and returns (Tensor, BoundingBoxes)
_transform = v2.Compose([
    v2.Resize((64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


@pytest.fixture
def dataset():
    mock_session_instance = MagicMock()
    mock_session_instance.execute.return_value.all.return_value = FAKE_ROWS

    mock_session_factory = MagicMock(return_value=mock_session_instance)
    mock_get_session = MagicMock(return_value=mock_session_factory)

    mock_gcs_config = MagicMock()
    mock_gcs_config.bucket = "whats-that-fish"
    mock_gcs_config.prefixes.get.return_value = "object_detection"
    mock_config = MagicMock()
    mock_config.gcs = mock_gcs_config

    with patch("whatsthatfish.models.od_dataset.get_session_factory", mock_get_session), \
         patch("whatsthatfish.models.od_dataset.get_config", return_value=mock_config):
        ds = ObjectDetectionDataset(split="train", transforms=_transform)

    return ds


def _mock_bucket_for(image_bytes: bytes) -> MagicMock:
    mock_blob = MagicMock()
    mock_blob.download_as_bytes.return_value = image_bytes
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    return mock_bucket


class TestObjectDetectionDataset:

    def test_len(self, dataset):
        assert len(dataset) == 3

    def test_getitem_returns_three_values(self, dataset):
        mock_bucket = _mock_bucket_for(_make_image_bytes())
        with patch("whatsthatfish.models.od_dataset._bucket", mock_bucket):
            result = dataset[0]
        assert len(result) == 3

    def test_getitem_image_shape(self, dataset):
        mock_bucket = _mock_bucket_for(_make_image_bytes())
        with patch("whatsthatfish.models.od_dataset._bucket", mock_bucket):
            img, _, _ = dataset[0]
        assert img.shape == (3, 64, 64)
        assert img.dtype == torch.float32

    def test_getitem_positive_label_shape(self, dataset):
        mock_bucket = _mock_bucket_for(_make_image_bytes())
        with patch("whatsthatfish.models.od_dataset._bucket", mock_bucket):
            _, labels, _ = dataset[0]
        assert labels.shape == (1, 5)
        assert labels.dtype == torch.float32

    def test_getitem_negative_label_is_empty(self, dataset):
        mock_bucket = _mock_bucket_for(_make_image_bytes())
        with patch("whatsthatfish.models.od_dataset._bucket", mock_bucket):
            _, labels, _ = dataset[1]
        assert labels.shape == (0, 5)

    def test_getitem_filename_returned(self, dataset):
        mock_bucket = _mock_bucket_for(_make_image_bytes())
        with patch("whatsthatfish.models.od_dataset._bucket", mock_bucket):
            _, _, fname = dataset[0]
        assert fname == "fish_001.jpg"

    def test_uiqm_weights_length_matches_dataset(self, dataset):
        weights = [row.uiqm for row in dataset.data]
        assert len(weights) == len(dataset)

    def test_image_values_normalized_to_unit_range(self, dataset):
        mock_bucket = _mock_bucket_for(_make_image_bytes())
        with patch("whatsthatfish.models.od_dataset._bucket", mock_bucket):
            img, _, _ = dataset[0]
        assert img.min().item() >= 0.0
        assert img.max().item() <= 1.0
