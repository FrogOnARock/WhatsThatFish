"""Unit tests for the post-train promotion gate (pure logic + local store)."""

from whatsthatfish.models.promotion import (
    should_promote,
    gate_and_promote,
    PromotionStore,
    CLASSIFIER_FLOOR,
    CLASSIFIER_KEYS,
    DETECTOR_FLOOR,
    DETECTOR_KEYS,
)

# A passing classifier metric set (all six geo metrics present, above floor).
GOOD = {
    "geo_top1_species": 0.70,
    "geo_top3_species": 0.85,
    "geo_top1_genus": 0.80,
    "geo_top3_genus": 0.90,
    "geo_top1_family": 0.90,
    "geo_top3_family": 0.95,
}


class TestBootstrap:
    def test_promotes_when_floor_met_and_no_incumbent(self):
        ok, reasons = should_promote(GOOD, None, CLASSIFIER_FLOOR, CLASSIFIER_KEYS)
        assert ok
        assert reasons["bootstrap"] is True

    def test_blocks_when_species_top1_below_floor(self):
        new = {**GOOD, "geo_top1_species": 0.60}  # < 0.65 floor
        ok, reasons = should_promote(new, None, CLASSIFIER_FLOOR, CLASSIFIER_KEYS)
        assert not ok
        assert reasons["floor:geo_top1_species"]["pass"] is False

    def test_floor_ignores_other_heads_at_bootstrap(self):
        # genus/family low but species clears the floor → bootstrap still promotes
        new = {**GOOD, "geo_top1_genus": 0.10, "geo_top1_family": 0.10}
        ok, _ = should_promote(new, None, CLASSIFIER_FLOOR, CLASSIFIER_KEYS)
        assert ok


class TestNoRegression:
    def test_promotes_when_all_metrics_hold(self):
        new = {k: v + 0.01 for k, v in GOOD.items()}
        ok, _ = should_promote(new, GOOD, CLASSIFIER_FLOOR, CLASSIFIER_KEYS)
        assert ok

    def test_lateral_within_epsilon_still_promotes(self):
        # one metric dips by less than epsilon (0.005) → not a regression
        new = {**GOOD, "geo_top3_family": GOOD["geo_top3_family"] - 0.004}
        ok, _ = should_promote(new, GOOD, CLASSIFIER_FLOOR, CLASSIFIER_KEYS, epsilon=0.005)
        assert ok

    def test_regression_beyond_epsilon_blocks(self):
        new = {**GOOD, "geo_top1_genus": GOOD["geo_top1_genus"] - 0.05}
        ok, reasons = should_promote(new, GOOD, CLASSIFIER_FLOOR, CLASSIFIER_KEYS)
        assert not ok
        assert reasons["regress:geo_top1_genus"]["pass"] is False

    def test_floor_blocks_even_when_beating_incumbent(self):
        # new beats a weak incumbent everywhere but sits below the absolute floor
        weak = {k: v - 0.10 for k, v in GOOD.items()}
        new = {**weak, "geo_top1_species": 0.60}  # beats weak(0.60) but < 0.65 floor
        ok, reasons = should_promote(new, weak, CLASSIFIER_FLOOR, CLASSIFIER_KEYS)
        assert not ok
        assert reasons["floor:geo_top1_species"]["pass"] is False


class TestDetectorGate:
    def test_detector_targets_are_the_floor(self):
        passing = {"mAP@0.5": 0.78, "mAP@0.5:0.95": 0.52, "Recall@0.5": 0.91}
        assert should_promote(passing, None, DETECTOR_FLOOR, DETECTOR_KEYS)[0]

    def test_detector_recall_below_target_blocks(self):
        failing = {"mAP@0.5": 0.78, "mAP@0.5:0.95": 0.52, "Recall@0.5": 0.85}
        assert not should_promote(failing, None, DETECTOR_FLOOR, DETECTOR_KEYS)[0]


class TestStoreRoundtrip:
    def test_promote_writes_then_next_run_reads_incumbent(self, tmp_path):
        store = PromotionStore(tmp_path)
        weights = tmp_path / "best.pt"
        weights.write_bytes(b"fake-weights")

        # first run: no incumbent → bootstrap promote
        assert store.load_metrics("classifier") is None
        assert gate_and_promote(
            "classifier", GOOD, weights, CLASSIFIER_FLOOR, CLASSIFIER_KEYS, store
        )
        incumbent = store.load_metrics("classifier")
        assert incumbent["geo_top1_species"] == GOOD["geo_top1_species"]

        # second run: a regression is now blocked against the stored incumbent
        worse = {**GOOD, "geo_top1_species": GOOD["geo_top1_species"] - 0.05}
        assert not gate_and_promote(
            "classifier", worse, weights, CLASSIFIER_FLOOR, CLASSIFIER_KEYS, store
        )
        # incumbent unchanged after a blocked promotion
        assert store.load_metrics("classifier")["geo_top1_species"] == GOOD["geo_top1_species"]
