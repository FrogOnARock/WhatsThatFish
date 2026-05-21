import io
from PIL import Image
import torch
from torch.utils.data import Dataset
from sqlalchemy import select, func
from google.cloud import storage
import os

from ..database.config import get_session_factory
from ..database.models import InatClassificationDataset
from ..config import get_config


_bucket = None
def init_gcs_worker(worker_id):
    global _bucket
    config = get_config().gcs
    client = storage.Client.from_service_account_json(os.environ.get("GCS_SECRET"))
    _bucket = client.bucket(config.bucket)


class ClassificationDataset(Dataset):
    def __init__(self,
                 split: str = "train",
                 transform = None,
                 max_samples: int = None,
                 ):

        self.transforms = transform
        self.gcs_config = get_config().gcs
        self.gcs_prefix = self.gcs_config.prefixes.get("gcs_train")
        self.session_factory = get_session_factory()
        self.session = self.session_factory()
        self.split = 0 if split == "train" else 1

        query = (
            select(InatClassificationDataset.photo_uuid, InatClassificationDataset.filename,
                   InatClassificationDataset.uiqm, InatClassificationDataset.subfamily, InatClassificationDataset.genus,
                   InatClassificationDataset.species, InatClassificationDataset.proposed_bbox).where(
                InatClassificationDataset.train == self.split).order_by(func.md5(InatClassificationDataset.photo_uuid))
            )
        if max_samples:
            query = query.limit(max_samples)

        self.data = self.session.execute(query).all()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        filename = record.filename
        bounding_box = record.proposed_bbox
        blob = _bucket.blob(self.gcs_prefix + "/" + filename)
        image_pil = Image.open(io.BytesIO(blob.download_as_bytes())).convert("RGB")

        W_img, H_img = image_pil.size
        bbox = bounding_box * [W_img, H_img, W_img, H_img]
        X = bbox[0] - (bbox[2] / 2)
        Y = bbox[1] - (bbox[3] / 2)

        img = image_pil.crop((X, Y, X + bbox[2], Y + bbox[3]))
        label = {
            "subfamily": record.subfamily,
            "genus": record.genus,
            "species": record.species
        }
        img_tensor, label_tensor = self.transforms(img, label)

        return img_tensor, label_tensor

