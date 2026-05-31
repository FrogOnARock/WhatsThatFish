import argparse
import csv
import json
import logging
from pathlib import Path
import os

from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select

from ..database.models import InatClassificationDataset, InatObjDetectionDataset, InatFilteredObservations, InatTaxa
from .bbox_inference import BoundingBoxInference
from ..database.config import get_session_factory

_CORAL_TAXON = "47533"

logger = logging.getLogger("BBOX Proposal")

_MODES = ("classification", "detection")

class InatBoundingBox:

    def __init__(self,
                 mode: str,
                 img_folder_path: str = None,
                 model_path: str = Path(__file__).parents[1] / "weights/od_best.pt",
                 conf: float = 0.25,
                 data_path: Path = Path(__file__).parents[1] / "data",
                 wal_path: str = "bbox_wal.csv",
                 bbox_folder_path: Path = Path(__file__).parents[1] / "data/classification_bboxs"
                 ):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

        self.model = BoundingBoxInference(model=model_path, conf=conf)
        self.session_factory = get_session_factory()
        self.img_folder = img_folder_path if img_folder_path else "classification_images" if mode == "classification"\
            else "inat_od_images"
        self.data_path = data_path
        self.wal_path = wal_path
        self.bbox_folder_path = bbox_folder_path
        self.wal_file = None
        self.dict_writer = None
        self.mode = mode
        self._target_model = InatClassificationDataset if mode == "classification" else InatObjDetectionDataset

    def write_to_db(self, max_params: int = 65535):
        logger.info("Writing WAL to database: %s", self.data_path / self.wal_path)
        with open(self.data_path / self.wal_path, "r") as file_path:
            reader = csv.DictReader(file_path)
            data = list(reader)
        logger.info("Upserting %d rows", len(data))
        for row in data:
            row["proposed_bbox"] = json.loads(row["proposed_bbox"]) if row["proposed_bbox"] else None
            row["annotation"] = json.loads(row["annotation"]) if row["annotation"] else None
            row["conf"] = float(row["conf"]) if row["conf"] else None

        num_cols = 4  # photo_uuid, proposed_bbox, conf, annotation
        chunk_size = max_params // num_cols
        with self.session_factory() as session:
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                stmt = insert(self._target_model).values(chunk)
                insert_stmt = stmt.on_conflict_do_update(
                    index_elements=[self._target_model.photo_uuid],
                    set_={
                        "proposed_bbox": stmt.excluded.proposed_bbox,
                        "conf": stmt.excluded.conf,
                        "annotation": stmt.excluded.annotation
                    }
                )
                session.execute(insert_stmt)
                logger.info("Upserted chunk %d/%d", min(i + chunk_size, len(data)), len(data))
            session.commit()
        logger.info("Database write complete")

    def annotation_normalization(self, row: dict[str, float]):
        return {
            "norm_center_x": ((row["x1"] + row["x2"]) / 2) / row["w"],
            "norm_center_y": ((row["y1"] + row["y2"]) / 2) / row["h"],
            "norm_width": (row["x2"] - row["x1"]) / row["w"],
            "norm_height": (row["y2"] - row["y1"]) / row["h"]
        }

    def log_bbox(self, data: list[tuple[str, dict[str, float]]]):

        out_data = [{
            "photo_uuid": row[0],
            "proposed_bbox": json.dumps({
                "x1": row[1].get("x1"),
                "y1": row[1].get("y1"),
                "x2": row[1].get("x2"),
                "y2": row[1].get("y2")
                }),
            "annotation": json.dumps([{**self.annotation_normalization(row[1]), "class_id": 0}]),
            "conf": row[1].get("conf")
            }
            if row[1] else
            {"photo_uuid": row[0],
             "proposed_bbox": "",
             "annotation": "",
             "conf": ""
             }
             for row in data]

        self.dict_writer = csv.DictWriter(self.wal_file, out_data[0].keys())

        if Path(self.data_path / self.wal_path).stat().st_size == 0:
            self.dict_writer.writeheader()
        self.dict_writer.writerows(out_data)

    def log_forced_negatives(self, photo_uuids: list[str]):
        """Write confirmed negative frames (coral) directly to WAL without inference.

        Coral images are guaranteed non-fish so we assign conf=1.0 and an empty
        annotation list, making them high-confidence negatives in OD training.
        """
        out_data = [
            {"photo_uuid": uuid, "proposed_bbox": "", "annotation": "[]", "conf": 1.0}
            for uuid in photo_uuids
        ]
        if not out_data:
            return
        self.dict_writer = csv.DictWriter(self.wal_file, out_data[0].keys())
        if Path(self.data_path / self.wal_path).stat().st_size == 0:
            self.dict_writer.writeheader()
        self.dict_writer.writerows(out_data)
        logger.info("Logged %d forced coral negatives (conf=1.0)", len(out_data))

    def run_inference(self,
                      img_dict: list[dict[str, str]],
                      batch_size: int = 64,
                      current_batch=0):
        total = len(img_dict)
        logger.info("Starting inference on %d images (batch_size=%d)", total, batch_size)
        for _ in range(current_batch, total, batch_size):
            batch = img_dict[current_batch:current_batch+batch_size]
            photo_uuids = [file.get("photo_uuid") for file in batch]
            file_path = [file.get("filename") for file in batch]
            files_to_infer = []
            for file in file_path:
                with open(self.data_path / self.img_folder / file, mode="rb") as img:
                    files_to_infer.append(img.read())

            inference = self.model.infer(data=files_to_infer)
            detections = sum(1 for r in inference if r is not None)
            logger.info("Batch %d/%d — %d/%d detections", current_batch + len(batch), total, detections, len(batch))
            combined = list(zip(photo_uuids, inference))
            self.log_bbox(combined)
            current_batch += batch_size

    def _select_photo_uuid(self, file_list: list[str]):
        with self.session_factory() as session:
            if self.mode == "detection":
                rows = session.execute(
                    select(
                        self._target_model.photo_uuid,
                        self._target_model.filename,
                        InatTaxa.ancestry,
                    )
                    .join(InatFilteredObservations, self._target_model.photo_uuid == InatFilteredObservations.photo_uuid)
                    .join(InatTaxa, InatFilteredObservations.taxon_id == InatTaxa.taxon_id)
                    .where(self._target_model.filename.in_(file_list))
                ).all()
                return [
                    {"photo_uuid": r.photo_uuid, "filename": r.filename, "is_coral": _CORAL_TAXON in (r.ancestry or "")}
                    for r in rows
                ]
            else:
                rows = session.execute(
                    select(self._target_model.photo_uuid, self._target_model.filename)
                    .where(self._target_model.filename.in_(file_list))
                ).all()
                return [{"photo_uuid": r.photo_uuid, "filename": r.filename} for r in rows]

    def run_bbox_proposals(self):
        filename_list = os.listdir(self.data_path / self.img_folder)
        logger.info("Found %d images in %s", len(filename_list), self.data_path / self.img_folder)
        img_dict = self._select_photo_uuid(filename_list)
        logger.info("%d images matched in database", len(img_dict))

        if self.mode == "detection":
            coral = [r["photo_uuid"] for r in img_dict if r["is_coral"]]
            fish  = [r for r in img_dict if not r["is_coral"]]
            logger.info("%d coral negatives (forced), %d fish images (inference)", len(coral), len(fish))
            self.log_forced_negatives(coral)
            self.run_inference(fish)
        else:
            self.run_inference(img_dict)

    def run_fine_tune_bboxes(self):
        logger.info("Starting bbox proposal run, WAL: %s", self.data_path / self.wal_path)
        Path(self.bbox_folder_path).mkdir(parents=True, exist_ok=True)
        with open(self.data_path / self.wal_path, "w") as self.wal_file:
            self.run_bbox_proposals()
        self.write_to_db()
        logger.info("Bbox proposal run complete")

if __name__ == '__main__':
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(
        description="Run bounding box proposal inference on iNat images",
        epilog=(
            "Examples:\n"
            "  Build LC1/LC2 OD training data:        --mode detection\n"
            "  Populate classifier crop bboxes:       --mode classification\n"
            "  Write existing WAL to DB only:         --mode classification --wal-only\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=_MODES,
        required=True,
        metavar="MODE",
        help=f"Target table. Choices: {', '.join(_MODES)}. 'classification' writes to inat_classification_dataset; 'detection' writes to inat_obj_detection_dataset",
    )
    parser.add_argument(
        "--wal-only",
        action="store_true",
        help="Skip inference and flush the existing WAL to the database",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        metavar="CONF",
        help="YOLO detection confidence threshold. Default: 0.25",
    )
    args = parser.parse_args()

    ibb = InatBoundingBox(mode=args.mode, conf=args.conf)
    if args.wal_only:
        ibb.write_to_db()
    else:
        ibb.run_fine_tune_bboxes()



