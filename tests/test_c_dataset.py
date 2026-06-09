"""
DB-backed tests for ClassificationDataset query semantics.

Covers:
- train/val split mapping (regression for the inversion: prepare_inat writes
  train=True for training rows, and the dataset must query accordingly)
- the detected-taxa filter: taxa with fewer than min_bbox_count rows holding
  a proposed_bbox are excluded entirely

Requires the test Postgres (docker compose -f docker-compose.test.yml up -d).
"""

import pytest

from whatsthatfish.database.models import (
    InatClassificationDataset,
    InatFilteredObservations,
    InatTaxa,
)
from whatsthatfish.models.datasets.c_dataset import ClassificationDataset

TEST_DATABASE_URL = "postgresql://test:test@localhost:5433/wtf_test"

BBOX = {"x1": 10, "y1": 10, "x2": 50, "y2": 50}

# (uuid, taxon_id, train, uiqm, proposed_bbox)
# taxon 100: detected (all rows have bboxes) — 2 train + 1 val
# taxon 200: undetected — one row never processed (SQL NULL) and one row the
#   detector found nothing in (JSON null, what the bbox proposal upsert
#   writes for no-detections); the filter must exclude both states
ROWS = [
    ("a" * 36, 100, True, 3.0, BBOX),
    ("b" * 36, 100, True, 5.0, BBOX),
    ("c" * 36, 100, False, 4.0, BBOX),
    ("d" * 36, 200, True, 9.0, "SQL_NULL"),
    ("e" * 36, 200, False, 8.0, None),  # ORM None → JSON null
]


@pytest.fixture
def seeded_db(session_factory, monkeypatch):
    """Seed both a detected and an undetected taxon; point the dataset's
    internal get_session_factory() at the test DB."""
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)

    with session_factory() as session:
        for taxon_id in (100, 200):
            session.add(
                InatTaxa(taxon_id=taxon_id, name=f"taxon {taxon_id}", rank="species")
            )
        for uuid, taxon_id, train, uiqm, bbox in ROWS:
            session.add(
                InatFilteredObservations(
                    photo_uuid=uuid,
                    photo_id=1,
                    observation_uuid=uuid,
                    observer_id=1,
                    taxon_id=taxon_id,
                    extension="jpg",
                    license="CC0",
                )
            )
            entry = InatClassificationDataset(
                photo_uuid=uuid,
                taxon_id=taxon_id,
                train=train,
                uiqm=uiqm,
                filename=f"{uuid}.jpg",
                zero_indexed_species=0,
                zero_indexed_genus=0,
                zero_indexed_subfamily=0,
            )
            # Leaving the attribute unset stores SQL NULL (never-processed
            # rows); setting it to None stores JSON null (no-detection rows).
            if bbox != "SQL_NULL":
                entry.proposed_bbox = bbox
            session.add(entry)
        session.commit()


class TestSplitSemantics:
    def test_train_split_loads_train_true_rows(self, seeded_db):
        ds = ClassificationDataset(split="train", min_bbox_count=1)
        assert {r.photo_uuid for r in ds.data} == {"a" * 36, "b" * 36}

    def test_val_split_loads_train_false_rows(self, seeded_db):
        ds = ClassificationDataset(split="val", min_bbox_count=1)
        assert {r.photo_uuid for r in ds.data} == {"c" * 36}

    def test_rows_ordered_by_uiqm_descending(self, seeded_db):
        ds = ClassificationDataset(split="train", min_bbox_count=1)
        assert [r.uiqm for r in ds.data] == [5.0, 3.0]

    def test_max_samples_limits_query(self, seeded_db):
        ds = ClassificationDataset(split="train", max_samples=1, min_bbox_count=1)
        assert len(ds) == 1
        assert ds.data[0].uiqm == 5.0  # highest quality kept first


class TestDetectedTaxaFilter:
    def test_undetected_taxa_are_excluded(self, seeded_db):
        """Taxon 200 has no proposed bboxes (one SQL NULL, one JSON null) —
        despite having the highest UIQM rows, none may appear in either split."""
        train = ClassificationDataset(split="train", min_bbox_count=1)
        val = ClassificationDataset(split="val", min_bbox_count=1)
        loaded = {r.photo_uuid for r in train.data} | {r.photo_uuid for r in val.data}
        assert "d" * 36 not in loaded
        assert "e" * 36 not in loaded

    def test_threshold_excludes_partially_detected_taxa(self, seeded_db):
        """taxon 100 has 3 bbox'd rows — a threshold above that drops it too."""
        ds = ClassificationDataset(split="train", min_bbox_count=4)
        assert len(ds) == 0
