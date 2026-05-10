"""
Tests for lcn_gradient_map: gradient_map and local_contrast_normalization.

No infrastructure required — all tests use in-memory numpy/cv2 image fixtures.
"""

import cv2
import numpy as np
import pytest

from whatsthatfish.src.preprocessing.lcn_gradient_map import (
    gradient_map,
    local_contrast_normalization,
)


# ── Image helpers ──────────────────────────────────────────────────────────────

def _encode(arr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", arr)
    return buf.tobytes()


def _uniform_bytes(value: int = 128, size: tuple[int, int] = (64, 64)) -> bytes:
    """Solid grey image — no edges, no contrast variation."""
    return _encode(np.full((size[0], size[1], 3), value, dtype=np.uint8))


def _step_edge_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    """Left half black, right half white — sharp vertical edge at centre column."""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:, size[1] // 2 :, :] = 255
    return _encode(img)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def uniform_bytes() -> bytes:
    return _uniform_bytes()


@pytest.fixture
def step_edge_bytes() -> bytes:
    return _step_edge_bytes()


# ════════════════════════════════════════════════════════════════════════════════
# gradient_map
# ════════════════════════════════════════════════════════════════════════════════

class TestGradientMap:

    def test_output_is_2d_uint8(self, step_edge_bytes):
        out = gradient_map(step_edge_bytes)
        assert out.ndim == 2
        assert out.dtype == np.uint8

    def test_output_shape_matches_input(self, step_edge_bytes):
        out = gradient_map(step_edge_bytes)
        assert out.shape == (64, 64)

    def test_values_in_valid_range(self, step_edge_bytes):
        out = gradient_map(step_edge_bytes)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_edge_pixels_are_nonzero(self, step_edge_bytes):
        """The column at the black/white boundary should have high gradient magnitude."""
        out = gradient_map(step_edge_bytes)
        mid = 64 // 2
        assert out[:, mid].mean() > 100

    def test_uniform_image_has_near_zero_gradient(self, uniform_bytes):
        """A flat image has no edges — all gradient values should be zero after normalisation."""
        out = gradient_map(uniform_bytes)
        # After min-max normalisation a flat gradient map collapses to all zeros
        assert out.max() == 0

    def test_corrupt_bytes_raises(self):
        with pytest.raises(Exception):
            gradient_map(b"not-an-image")


# ════════════════════════════════════════════════════════════════════════════════
# local_contrast_normalization
# ════════════════════════════════════════════════════════════════════════════════

class TestLocalContrastNormalization:

    def test_output_is_3channel_uint8(self, step_edge_bytes):
        out = local_contrast_normalization(step_edge_bytes)
        assert out.ndim == 3
        assert out.shape[2] == 3
        assert out.dtype == np.uint8

    def test_output_shape_matches_input(self, step_edge_bytes):
        out = local_contrast_normalization(step_edge_bytes)
        assert out.shape == (64, 64, 3)

    def test_values_in_valid_range(self, step_edge_bytes):
        out = local_contrast_normalization(step_edge_bytes)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_uniform_image_produces_flat_output(self, uniform_bytes):
        """A perfectly uniform image has no local contrast — LCN should flatten it."""
        out = local_contrast_normalization(uniform_bytes)
        # All pixels deviate from local mean by zero, so output is uniformly one value
        assert out.std() == pytest.approx(0.0, abs=1.0)

    def test_high_contrast_image_spans_full_range(self, step_edge_bytes):
        """A sharp black/white step should produce output spanning most of [0, 255]."""
        out = local_contrast_normalization(step_edge_bytes)
        assert out.max() - out.min() > 200

    def test_corrupt_bytes_raises(self):
        with pytest.raises(Exception):
            local_contrast_normalization(b"not-an-image")
