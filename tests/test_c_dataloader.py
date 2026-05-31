"""
Tests for classification collate functions.

Pure tensor/dict logic — no GCS, DB, or model loading required.
"""

import torch

from whatsthatfish.models.data.c_dataloader import collate_fn, collate_fn_ultralytics


# ── Helpers ────────────────────────────────────────────────────────────────────

def _img(c: int = 3, h: int = 224, w: int = 224) -> torch.Tensor:
    return torch.rand(c, h, w)


def _label(species: int = 0, genus: int = 1, subfamily: int = 2) -> dict:
    return {"species": species, "genus": genus, "subfamily": subfamily}


def _batch(n: int = 4, **label_kwargs) -> list:
    return [(_img(), _label(**label_kwargs)) for _ in range(n)]


# ════════════════════════════════════════════════════════════════════════════════
# collate_fn (custom multi-head)
# ════════════════════════════════════════════════════════════════════════════════

class TestCollateFn:

    def test_images_are_stacked(self):
        batch = _batch(4)
        imgs, _ = collate_fn(batch)
        assert imgs.shape == (4, 3, 224, 224)

    def test_labels_dict_has_three_keys(self):
        batch = _batch(2)
        _, labels = collate_fn(batch)
        assert set(labels.keys()) == {"species", "genus", "subfamily"}

    def test_each_label_tensor_length_matches_batch(self):
        batch = _batch(3)
        _, labels = collate_fn(batch)
        assert labels["species"].shape == (3,)
        assert labels["genus"].shape == (3,)
        assert labels["subfamily"].shape == (3,)

    def test_species_values_preserved(self):
        batch = [
            (_img(), _label(species=5)),
            (_img(), _label(species=12)),
            (_img(), _label(species=99)),
        ]
        _, labels = collate_fn(batch)
        assert labels["species"].tolist() == [5, 12, 99]

    def test_genus_values_preserved(self):
        batch = [(_img(), _label(genus=i)) for i in range(3)]
        _, labels = collate_fn(batch)
        assert labels["genus"].tolist() == [0, 1, 2]

    def test_subfamily_values_preserved(self):
        batch = [(_img(), _label(subfamily=i * 10)) for i in range(4)]
        _, labels = collate_fn(batch)
        assert labels["subfamily"].tolist() == [0, 10, 20, 30]

    def test_image_dtype_preserved(self):
        batch = _batch(2)
        imgs, _ = collate_fn(batch)
        assert imgs.dtype == torch.float32

    def test_single_item_batch(self):
        batch = [(_img(), _label(species=7, genus=2, subfamily=0))]
        imgs, labels = collate_fn(batch)
        assert imgs.shape == (1, 3, 224, 224)
        assert labels["species"].item() == 7


# ════════════════════════════════════════════════════════════════════════════════
# collate_fn_ultralytics (single species label for YOLO11-cls)
# ════════════════════════════════════════════════════════════════════════════════

class TestCollateFnUltralytics:

    def test_images_are_stacked(self):
        batch = _batch(4)
        imgs, _ = collate_fn_ultralytics(batch)
        assert imgs.shape == (4, 3, 224, 224)

    def test_labels_dict_has_only_cls_key(self):
        batch = _batch(2)
        _, labels = collate_fn_ultralytics(batch)
        assert set(labels.keys()) == {"cls"}
        assert "genus" not in labels
        assert "subfamily" not in labels

    def test_cls_tensor_length_matches_batch(self):
        batch = _batch(5)
        _, labels = collate_fn_ultralytics(batch)
        assert labels["cls"].shape == (5,)

    def test_cls_values_are_species_labels(self):
        batch = [(_img(), _label(species=i, genus=99, subfamily=88)) for i in range(3)]
        _, labels = collate_fn_ultralytics(batch)
        assert labels["cls"].tolist() == [0, 1, 2]
