"""
Tests for InatBoundingBox — WAL writing, JSONB serialization, batch inference, DB upsert.

BoundingBoxInference, DB sessions, and filesystem are fully controlled via
mocks and tmp_path — no model weights or Postgres instance required.
"""

import csv
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from whatsthatfish.inference.inat_bbox_proposal import InatBoundingBox


# ── Helpers ────────────────────────────────────────────────────────────────────

BBOX_A = {
    "x1": 10.0,
    "y1": 15.0,
    "x2": 80.0,
    "y2": 70.0,
    "conf": 0.85,
    "w": 100.0,
    "h": 80.0,
}
BBOX_B = {
    "x1": 5.0,
    "y1": 5.0,
    "x2": 40.0,
    "y2": 40.0,
    "conf": 0.60,
    "w": 100.0,
    "h": 80.0,
}


def _make_image_bytes(w: int = 100, h: int = 80) -> bytes:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _make_img_files(tmp_path: Path, count: int) -> list[dict]:
    """Create fake image files on disk and return img_dict entries."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)
    entries = []
    for i in range(count):
        fname = f"img_{i:04d}.jpg"
        (img_dir / fname).write_bytes(_make_image_bytes())
        entries.append({"photo_uuid": f"uuid-{i:04d}", "filename": fname})
    return entries


@pytest.fixture
def tmp_inat(tmp_path):
    """InatBoundingBox with all paths under tmp_path; model and DB mocked."""
    with (
        patch("whatsthatfish.inference.inat_bbox_proposal.BoundingBoxInference"),
        patch("whatsthatfish.inference.inat_bbox_proposal.get_session_factory"),
    ):
        obj = InatBoundingBox(
            mode="classification",
            img_folder_path="images",
            model_path="fake.pt",
            conf=0.25,
            data_path=tmp_path,
            wal_path="bbox_wal.csv",
            bbox_folder_path=tmp_path / "bboxes",
        )
    return obj


def _open_wal(obj: InatBoundingBox, tmp_path: Path):
    wal = open(tmp_path / "bbox_wal.csv", "w", newline="")
    obj.wal_file = wal
    return wal


def _read_wal(tmp_path: Path) -> list[dict]:
    return list(csv.DictReader(open(tmp_path / "bbox_wal.csv")))


# ════════════════════════════════════════════════════════════════════════════════
# log_bbox — WAL writing and JSON serialization
# ════════════════════════════════════════════════════════════════════════════════


class TestLogBbox:
    def test_writes_header_on_empty_file(self, tmp_inat, tmp_path):
        _open_wal(tmp_inat, tmp_path)
        tmp_inat.log_bbox([("uuid-001", BBOX_A)])
        tmp_inat.wal_file.flush()

        rows = _read_wal(tmp_path)
        assert set(rows[0].keys()) == {
            "photo_uuid",
            "proposed_bbox",
            "annotation",
            "conf",
        }

    def test_proposed_bbox_is_valid_json(self, tmp_inat, tmp_path):
        _open_wal(tmp_inat, tmp_path)
        tmp_inat.log_bbox([("uuid-001", BBOX_A)])
        tmp_inat.wal_file.flush()

        rows = _read_wal(tmp_path)
        bbox = json.loads(rows[0]["proposed_bbox"])
        assert bbox["x1"] == pytest.approx(10.0)
        assert bbox["y1"] == pytest.approx(15.0)
        assert bbox["x2"] == pytest.approx(80.0)
        assert bbox["y2"] == pytest.approx(70.0)

    def test_conf_written_correctly(self, tmp_inat, tmp_path):
        _open_wal(tmp_inat, tmp_path)
        tmp_inat.log_bbox([("uuid-001", BBOX_A)])
        tmp_inat.wal_file.flush()

        rows = _read_wal(tmp_path)
        assert float(rows[0]["conf"]) == pytest.approx(0.85)

    def test_photo_uuid_written_correctly(self, tmp_inat, tmp_path):
        _open_wal(tmp_inat, tmp_path)
        tmp_inat.log_bbox([("uuid-abc", BBOX_A)])
        tmp_inat.wal_file.flush()

        rows = _read_wal(tmp_path)
        assert rows[0]["photo_uuid"] == "uuid-abc"

    def test_none_detection_writes_empty_bbox_and_conf(self, tmp_inat, tmp_path):
        _open_wal(tmp_inat, tmp_path)
        tmp_inat.log_bbox([("uuid-001", None)])
        tmp_inat.wal_file.flush()

        rows = _read_wal(tmp_path)
        assert rows[0]["proposed_bbox"] == ""
        assert rows[0]["conf"] == ""

    def test_batch_writes_all_rows(self, tmp_inat, tmp_path):
        _open_wal(tmp_inat, tmp_path)
        tmp_inat.log_bbox(
            [
                ("uuid-001", BBOX_A),
                ("uuid-002", BBOX_B),
                ("uuid-003", None),
            ]
        )
        tmp_inat.wal_file.flush()

        rows = _read_wal(tmp_path)
        assert len(rows) == 3

    def test_mixed_batch_none_in_correct_position(self, tmp_inat, tmp_path):
        _open_wal(tmp_inat, tmp_path)
        tmp_inat.log_bbox([("uuid-001", BBOX_A), ("uuid-002", None)])
        tmp_inat.wal_file.flush()

        rows = _read_wal(tmp_path)
        assert json.loads(rows[0]["proposed_bbox"])["x1"] == pytest.approx(10.0)
        assert rows[1]["proposed_bbox"] == ""

    def test_proposed_bbox_uses_double_quotes_not_python_repr(self, tmp_inat, tmp_path):
        _open_wal(tmp_inat, tmp_path)
        tmp_inat.log_bbox([("uuid-001", BBOX_A)])
        tmp_inat.wal_file.flush()

        raw = open(tmp_path / "bbox_wal.csv").read()
        assert "'" not in raw.split("\n")[1]  # no single quotes in loaders row


# ════════════════════════════════════════════════════════════════════════════════
# JSONB roundtrip — write then deserialize as write_to_db would
# ════════════════════════════════════════════════════════════════════════════════


class TestJsonbRoundtrip:
    def _roundtrip(self, tmp_path: Path) -> list[dict]:
        rows = _read_wal(tmp_path)
        for row in rows:
            if row["proposed_bbox"]:
                row["proposed_bbox"] = json.loads(row["proposed_bbox"])
            if row["annotation"]:
                row["annotation"] = json.loads(row["annotation"])
        return rows

    def test_bbox_values_preserved_after_roundtrip(self, tmp_inat, tmp_path):
        _open_wal(tmp_inat, tmp_path)
        tmp_inat.log_bbox([("uuid-001", BBOX_A)])
        tmp_inat.wal_file.close()

        rows = self._roundtrip(tmp_path)
        bbox = rows[0]["proposed_bbox"]
        assert bbox["x1"] == pytest.approx(BBOX_A["x1"])
        assert bbox["y1"] == pytest.approx(BBOX_A["y1"])
        assert bbox["x2"] == pytest.approx(BBOX_A["x2"])
        assert bbox["y2"] == pytest.approx(BBOX_A["y2"])

    def test_none_roundtrip_stays_falsy(self, tmp_inat, tmp_path):
        _open_wal(tmp_inat, tmp_path)
        tmp_inat.log_bbox([("uuid-001", None)])
        tmp_inat.wal_file.close()

        rows = self._roundtrip(tmp_path)
        assert not rows[0]["proposed_bbox"]

    def test_multiple_bboxes_roundtrip_independently(self, tmp_inat, tmp_path):
        _open_wal(tmp_inat, tmp_path)
        tmp_inat.log_bbox([("uuid-001", BBOX_A), ("uuid-002", BBOX_B)])
        tmp_inat.wal_file.close()

        rows = self._roundtrip(tmp_path)
        assert rows[0]["proposed_bbox"]["x1"] == pytest.approx(BBOX_A["x1"])
        assert rows[1]["proposed_bbox"]["x1"] == pytest.approx(BBOX_B["x1"])
        assert float(rows[0]["conf"]) == pytest.approx(BBOX_A["conf"])
        assert float(rows[1]["conf"]) == pytest.approx(BBOX_B["conf"])


# ════════════════════════════════════════════════════════════════════════════════
# run_inference — batching and coverage
# ════════════════════════════════════════════════════════════════════════════════


class TestRunInference:
    def test_single_batch_processes_all_images(self, tmp_inat, tmp_path):
        img_dict = _make_img_files(tmp_path, 5)
        tmp_inat.model.infer.return_value = [BBOX_A] * 5
        _open_wal(tmp_inat, tmp_path)

        tmp_inat.run_inference(img_dict, batch_size=10)

        assert tmp_inat.model.infer.call_count == 1
        assert len(tmp_inat.model.infer.call_args[1]["data"]) == 5

    def test_multiple_batches_cover_all_images(self, tmp_inat, tmp_path):
        img_dict = _make_img_files(tmp_path, 7)
        tmp_inat.model.infer.side_effect = [
            [BBOX_A] * 4,
            [BBOX_A] * 3,
        ]
        _open_wal(tmp_inat, tmp_path)

        tmp_inat.run_inference(img_dict, batch_size=4)

        assert tmp_inat.model.infer.call_count == 2

    def test_last_batch_contains_remainder(self, tmp_inat, tmp_path):
        img_dict = _make_img_files(tmp_path, 5)
        tmp_inat.model.infer.side_effect = [
            [BBOX_A] * 4,
            [BBOX_A] * 1,
        ]
        _open_wal(tmp_inat, tmp_path)

        tmp_inat.run_inference(img_dict, batch_size=4)

        last_call_data = tmp_inat.model.infer.call_args_list[1][1]["data"]
        assert len(last_call_data) == 1

    def test_empty_img_dict_skips_inference(self, tmp_inat, tmp_path):
        _open_wal(tmp_inat, tmp_path)
        tmp_inat.run_inference([], batch_size=10)
        tmp_inat.model.infer.assert_not_called()

    def test_batch_size_equal_to_total_is_one_call(self, tmp_inat, tmp_path):
        img_dict = _make_img_files(tmp_path, 3)
        tmp_inat.model.infer.return_value = [BBOX_A] * 3
        _open_wal(tmp_inat, tmp_path)

        tmp_inat.run_inference(img_dict, batch_size=3)

        assert tmp_inat.model.infer.call_count == 1


# ════════════════════════════════════════════════════════════════════════════════
# _select_photo_uuid
# ════════════════════════════════════════════════════════════════════════════════


class TestSelectPhotoUuid:
    def _configure_session(self, tmp_inat, db_rows):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.execute.return_value.all.return_value = db_rows
        tmp_inat.session_factory.return_value = mock_session

    def test_returns_list_of_uuid_filename_dicts(self, tmp_inat):
        row = MagicMock()
        row.photo_uuid = "uuid-abc"
        row.filename = "12345.jpg"
        self._configure_session(tmp_inat, [row])

        result = tmp_inat._select_photo_uuid(["12345.jpg"])

        assert result == [{"photo_uuid": "uuid-abc", "filename": "12345.jpg"}]

    def test_multiple_rows_returned_in_order(self, tmp_inat):
        db_rows = []
        for i in range(3):
            r = MagicMock()
            r.photo_uuid = f"uuid-{i}"
            r.filename = f"file_{i}.jpg"
            db_rows.append(r)
        self._configure_session(tmp_inat, db_rows)

        result = tmp_inat._select_photo_uuid([f"file_{i}.jpg" for i in range(3)])

        assert len(result) == 3
        assert result[1]["photo_uuid"] == "uuid-1"

    def test_empty_file_list_returns_empty(self, tmp_inat):
        self._configure_session(tmp_inat, [])

        result = tmp_inat._select_photo_uuid([])

        assert result == []
