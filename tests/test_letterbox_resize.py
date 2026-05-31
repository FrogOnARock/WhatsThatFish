"""
Tests for LetterboxResize: PIL Image -> square PIL Image with zero padding.

No infrastructure required — all tests use in-memory PIL fixtures.
"""

import numpy as np
import pytest
from PIL import Image

from whatsthatfish.transforms.letterbox_resize import LetterboxResize


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def transform():
    return LetterboxResize(320)


def _pil(w: int, h: int) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _solid(w: int, h: int, color=(255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


# ════════════════════════════════════════════════════════════════════════════════
# Output dimensions
# ════════════════════════════════════════════════════════════════════════════════

class TestOutputDimensions:

    def test_square_input_stays_square(self, transform):
        out = transform(_pil(300, 300))
        assert out.size == (320, 320)

    def test_landscape_output_is_square(self, transform):
        out = transform(_pil(400, 200))
        assert out.size == (320, 320)

    def test_portrait_output_is_square(self, transform):
        out = transform(_pil(100, 300))
        assert out.size == (320, 320)

    def test_small_image_output_is_square(self, transform):
        out = transform(_pil(40, 30))
        assert out.size == (320, 320)

    def test_custom_size(self):
        t = LetterboxResize(224)
        out = t(_pil(500, 200))
        assert out.size == (224, 224)

    def test_odd_padding_remainder_produces_correct_size(self, transform):
        # Aspect ratio that produces odd pad_w or pad_h
        out = transform(_pil(100, 150))
        assert out.size == (320, 320)


# ════════════════════════════════════════════════════════════════════════════════
# Aspect ratio preservation
# ════════════════════════════════════════════════════════════════════════════════

class TestAspectRatio:

    def test_landscape_longer_side_fills_canvas(self, transform):
        """Wider dimension should fill the full 224px after scaling."""
        img = _solid(400, 200, color=(255, 0, 0))
        out = transform(img)
        arr = np.array(out)
        # Top and bottom rows should be black padding
        assert arr[0].max() == 0
        assert arr[-1].max() == 0
        # Middle row should contain the red content
        assert arr[160, 160, 0] > 200

    def test_portrait_longer_side_fills_canvas(self, transform):
        """Taller dimension should fill the full 224px after scaling."""
        img = _solid(100, 400, color=(0, 255, 0))
        out = transform(img)
        arr = np.array(out)
        # Left and right columns should be black padding
        assert arr[160, 0].max() == 0
        assert arr[160, -1].max() == 0
        # Centre column should contain the green content
        assert arr[160, 160, 1] > 200

    def test_square_input_has_no_padding(self, transform):
        """A square image should fill the canvas exactly — no black padding."""
        img = _solid(400, 400, color=(180, 120, 60))
        out = transform(img)
        arr = np.array(out)
        assert arr.min() > 0  # no black pixels anywhere


# ════════════════════════════════════════════════════════════════════════════════
# Padding symmetry
# ════════════════════════════════════════════════════════════════════════════════

class TestPaddingSymmetry:

    def test_landscape_padding_is_top_bottom(self, transform):
        """For landscape input, padding appears above and below the image."""
        img = _solid(400, 200, color=(200, 200, 200))
        out = transform(img)
        arr = np.array(out)

        # Find first and last non-black row
        row_has_content = arr.max(axis=(1, 2)) > 0
        first_content = np.argmax(row_has_content)
        last_content = len(row_has_content) - np.argmax(row_has_content[::-1]) - 1

        top_pad = first_content
        bottom_pad = 319 - last_content
        assert abs(top_pad - bottom_pad) <= 1  # symmetric within 1px (odd remainder)

    def test_portrait_padding_is_left_right(self, transform):
        """For portrait input, padding appears left and right of the image."""
        img = _solid(100, 400, color=(200, 200, 200))
        out = transform(img)
        arr = np.array(out)

        col_has_content = arr.max(axis=(0, 2)) > 0
        first_content = np.argmax(col_has_content)
        last_content = len(col_has_content) - np.argmax(col_has_content[::-1]) - 1

        left_pad = first_content
        right_pad = 319  - last_content
        assert abs(left_pad - right_pad) <= 1
