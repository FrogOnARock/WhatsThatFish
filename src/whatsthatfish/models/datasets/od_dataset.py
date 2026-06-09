import logging
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from sqlalchemy import select, desc
from PIL import Image
import torch
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from ...database.models import LilaImageQuality, LilaYolo, InatObjDetectionDataset
from ...database.config import get_session_factory

logger = logging.getLogger(__name__)


class ObjectDetectionDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        split: str = "train",
        transforms=None,
        max_samples: int = None,
        local_base_dir: str = None,
    ):
        self.session_factory = get_session_factory()
        self.session = self.session_factory()
        self.split = True if split == "train" else False
        self.transform = transforms
        self.max_samples = (
            max_samples
            if max_samples
            else 100000
            if dataset in ["lc1", "lc2"]
            else None
        )
        self.local_base_dir = (
            Path(local_base_dir)
            if local_base_dir
            else (Path(__file__).parents[2] / "data/inat_od_images")
            if dataset in ["lc1", "lc2"]
            else (Path(__file__).parents[2] / "data/od_images")
        )

        if dataset == "lila":
            query = (
                select(
                    LilaImageQuality.file_name,
                    LilaImageQuality.uiqm,
                    LilaYolo.annotation,
                )
                .join(LilaYolo, LilaImageQuality.file_name == LilaYolo.file_name)
                .where(LilaYolo.annotation[0]["is_train"].as_boolean() == self.split)
            )
        elif dataset == "lc1":
            query = (
                select(
                    InatObjDetectionDataset.filename.label("file_name"),
                    InatObjDetectionDataset.uiqm,
                    InatObjDetectionDataset.conf,
                    InatObjDetectionDataset.annotation,
                )
                .where(InatObjDetectionDataset.train == self.split)
                # Excludes JSONB null annotations (rows not yet populated by
                # bbox proposal) — there are no SQL NULLs in this column.
                .where(InatObjDetectionDataset.annotation != "null")
                .order_by(desc(InatObjDetectionDataset.uiqm))
            )
        elif dataset == "lc2":
            query = (
                select(
                    InatObjDetectionDataset.filename.label("file_name"),
                    InatObjDetectionDataset.uiqm,
                    InatObjDetectionDataset.conf,
                    InatObjDetectionDataset.annotation,
                )
                .where(InatObjDetectionDataset.train == self.split)
                .where(InatObjDetectionDataset.annotation != "null")
            )
        else:
            raise ValueError("Dataset must be one of lila, lc1, lc2.")

        if self.max_samples is not None:
            query = query.limit(self.max_samples)

        self.data = self.session.execute(query).all()
        count_positive = sum(
            [
                any([ann["class_id"] for ann in record.annotation if ann["class_id"]])
                == 0
                for record in self.data
            ]
        )
        count_negative = len(self.data) - count_positive
        logger.info(
            "Loaded %d records for dataset=%s split=%s. With a pos-neg split of pos=%s neg=%s",
            len(self.data),
            dataset,
            split,
            count_positive,
            count_negative,
        )

    @property
    def labels(self):
        # TODO what the fuck? Review tomorrow
        return [
            {
                "bboxes": np.array([[0, 0, 0, 0]])
                if not record.annotation
                else np.array(
                    [
                        [
                            ann["norm_center_x"],
                            ann["norm_center_y"],
                            ann["norm_width"],
                            ann["norm_height"],
                        ]
                        if ann["class_id"] == 0
                        else [0, 0, 0, 0]
                        for ann in record.annotation
                    ]
                ),
                "cls": np.array([1])
                if not record.annotation
                else np.array([ann["class_id"] for ann in record.annotation]),
            }
            for record in self.data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        labels = [
            [
                ann["class_id"],
                ann["norm_center_x"],
                ann["norm_center_y"],
                ann["norm_width"],
                ann["norm_height"],
            ]
            for ann in record.annotation
            if ann["class_id"] == 0
        ]

        label_tensor = (
            torch.zeros((0, 5), dtype=torch.float32)
            if not labels
            else torch.tensor(labels, dtype=torch.float32)
        )

        image_pil = Image.open(self.local_base_dir / record.file_name).convert("RGB")
        W_img, H_img = image_pil.size

        abs_boxes = label_tensor[:, 1:5] * torch.tensor([W_img, H_img, W_img, H_img])
        boxes = BoundingBoxes(
            abs_boxes, format=BoundingBoxFormat.CXCYWH, canvas_size=(H_img, W_img)
        )

        image_tensor, boxes = self.transform(image_pil, boxes)
        H_out, W_out = image_tensor.shape[-2], image_tensor.shape[-1]
        norm_boxes = boxes / torch.tensor([W_out, H_out, W_out, H_out])
        label_tensor = torch.cat([label_tensor[:, 0:1], norm_boxes], dim=1)

        return image_tensor, label_tensor, record.file_name
