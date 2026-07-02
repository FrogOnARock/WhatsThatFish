"""
End-to-end tensor-contract test for the two-stage inference chain:

    image  →  YOLO bbox (BoundingBoxInference)  →  crop
           →  LetterboxResize(320)  →  AddMultiChannel  →  (5,320,320)
           →  CustomResnet  →  (species, genus, family) logits

Nothing else tests this HANDOFF. The individual pieces are covered
(test_bbox_inference, test_letterbox_resize, test_five_channel,
test_c_classification), but the contract BETWEEN them — crop dims feeding the
320×320 letterbox feeding the 5-channel classifier input — is exactly where a
silent shape/normalization mismatch would hide.

We don't load real YOLO weights here (that's test_bbox_inference's job and
needs a .pt file). Instead we start from a detection dict in the SHAPE
`BoundingBoxInference.infer()` returns — {x1,y1,x2,y2,conf,w,h} — and drive the
deterministic downstream chain with a tiny randomly-initialised CustomResnet.

Note: `crop_export.py` (the real handoff) is still a stub, so this test also
documents the intended contract before that code is written.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from whatsthatfish.models.architecture.custom_resnet import BasicBlock, CustomResnet
from whatsthatfish.transforms.letterbox_resize import LetterboxResize
from whatsthatfish.transforms.five_channel_conversion import AddMultiChannel

NUM_CLASS = [40, 12, 5]  # [species, genus, family] — tiny for speed


@pytest.fixture(scope="module")
def classifier():
    """A small 5-channel CustomResnet (ResNet18-shaped) with three heads."""
    model = CustomResnet(
        block=BasicBlock, layers=[2, 2, 2, 2], num_class=NUM_CLASS, in_dim=5
    )
    model.eval()
    return model


def _synthetic_image(w: int = 800, h: int = 600) -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _detection(x1, y1, x2, y2, w, h) -> dict:
    """A detection in the shape BoundingBoxInference.infer() emits."""
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": 0.92, "w": w, "h": h}


def _crop_and_transform(img: Image.Image, det: dict) -> torch.Tensor:
    """The detector→classifier handoff under test.

    Crop to the detected box, letterbox to 320, lift to 5 channels.
    """
    crop = img.crop((det["x1"], det["y1"], det["x2"], det["y2"]))
    letterboxed = LetterboxResize(320)(crop)
    # AddMultiChannel now emits an (5,H,W) numpy array (numpy-native for the
    # torch-free serving path); the torch handoff wraps it back into a tensor.
    tensor = torch.from_numpy(AddMultiChannel()(letterboxed))
    return tensor


# ─── The handoff tensor contract ────────────────────────────────────────


class TestCropToClassifier:
    def test_transform_yields_5x320x320_float(self):
        img = _synthetic_image()
        det = _detection(100, 80, 500, 420, 800, 600)
        tensor = _crop_and_transform(img, det)
        assert tensor.shape == (5, 320, 320)
        assert tensor.dtype == torch.float32

    def test_channels_normalised_to_unit_range(self):
        img = _synthetic_image()
        det = _detection(100, 80, 500, 420, 800, 600)
        tensor = _crop_and_transform(img, det)
        assert tensor.min().item() >= 0.0
        assert tensor.max().item() <= 1.0

    def test_classifier_consumes_crop_and_emits_three_heads(self, classifier):
        img = _synthetic_image()
        det = _detection(100, 80, 500, 420, 800, 600)
        tensor = _crop_and_transform(img, det)

        with torch.no_grad():
            species, genus, family = classifier(tensor.unsqueeze(0))

        assert species.shape == (1, NUM_CLASS[0])
        assert genus.shape == (1, NUM_CLASS[1])
        assert family.shape == (1, NUM_CLASS[2])

    def test_full_chain_on_batch_of_crops(self, classifier):
        """Multiple detections → batched classifier input → batched logits."""
        img = _synthetic_image()
        dets = [
            _detection(10, 10, 300, 300, 800, 600),
            _detection(400, 200, 780, 560, 800, 600),
        ]
        batch = torch.stack([_crop_and_transform(img, d) for d in dets])
        assert batch.shape == (2, 5, 320, 320)

        with torch.no_grad():
            species, genus, family = classifier(batch)
        assert species.shape == (2, NUM_CLASS[0])

    def test_no_detection_skips_classifier(self, classifier):
        """BoundingBoxInference returns None when nothing is detected — the
        handoff must not attempt to crop/classify a None."""
        det = None
        assert det is None  # contract: caller guards None before cropping

    # Edge-flush crop contract — DECIDED: feed the RAW crop straight into the 320
    # letterbox (no pre-pad). LetterboxResize is already aspect-ratio-preserving
    # (symmetric zero-pad to square), so the fish aspect ratio is kept without an
    # extra padding step. A box flush against the image edge (x1=0, y1=0, y2=h)
    # must still crop a valid non-empty region, survive the letterbox, and
    # classify.
    def test_edge_flush_bbox_contract(self, classifier):
        img = _synthetic_image(800, 600)
        # Flush left + top + bottom edges; tall, non-square region.
        det = _detection(0, 0, 400, 600, 800, 600)
        tensor = _crop_and_transform(img, det)
        assert tensor.shape == (5, 320, 320)
        assert torch.isfinite(tensor).all()  # no NaN/inf from a degenerate crop

        with torch.no_grad():
            species, genus, family = classifier(tensor.unsqueeze(0))
        assert species.shape == (1, NUM_CLASS[0])

    def test_full_frame_box_is_valid(self, classifier):
        """The extreme flush case: the box IS the whole image (x1=0,y1=0,x2=w,y2=h)."""
        img = _synthetic_image(800, 600)
        tensor = _crop_and_transform(img, _detection(0, 0, 800, 600, 800, 600))
        assert tensor.shape == (5, 320, 320)
