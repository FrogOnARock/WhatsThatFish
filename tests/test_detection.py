"""Tests for the Detector facade: per-stage weight chaining + config resolution.

Pure construction — no training, GPU, or weight files needed. Confirms the
LILA → LC1 → LC2 weight mapping and that the lifecycle verbs exist after the
od_training → models/detection rename.
"""

from whatsthatfish.models.detection import Detector, Dataset


class TestDetectorConfig:
    def test_weights_chain_maps_each_stage(self):
        """Each stage reads the previous stage's output as its input."""
        assert Detector._WEIGHTS["lila"] == ("yolo11l.pt", "od_best.pt")
        assert Detector._WEIGHTS["lc1"] == ("od_best.pt", "lc1_best.pt")
        assert Detector._WEIGHTS["lc2"] == ("lc1_best.pt", "lc2_best.pt")

    def test_init_resolves_stage_weights_and_config(self):
        d = Detector(dataset=Dataset.LC1)
        assert d.dataset == "lc1"
        assert d.input_weights == "od_best.pt"
        assert d.output_weights == "lc1_best.pt"
        assert d.train_config_path.name == "lc1_train_config.yaml"

    def test_accepts_plain_string_dataset(self):
        """Dataset enum or its string value both resolve the same stage."""
        d = Detector(dataset="lc2")
        assert d.dataset == "lc2"
        assert d.input_weights == "lc1_best.pt"

    def test_facade_exposes_lifecycle_verbs(self):
        for verb in ("train", "tune", "predict"):
            assert callable(getattr(Detector, verb))
