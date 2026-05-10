import io
import os
from torch.utils.data import Dataset
from sqlalchemy import select
from PIL import Image
from google.cloud import storage
import torch

from ..database.models import LilaImageQuality, LilaYolo
from ..config import get_config
from ..database.config import get_session_factory
from ..retry import transfer_retry


class ObjectDetectionDataset(Dataset):

    def __init__(self, split: str = "train", transforms = None):
        self.session = get_session_factory()
        self.session_factory = self.session()
        self.gcs_config = get_config().gcs
        self.gcs_prefix = self.gcs_config.prefixes.get("gcs_object_detection")
        self._gcs_bucket = None

        self.split = True if split == "train" else False
        self.transform = transforms

        self.data = self.session_factory.execute(
            select(LilaImageQuality.file_name, LilaImageQuality.uiqm, LilaYolo.annotation)
            .join(LilaYolo, LilaImageQuality.file_name == LilaYolo.file_name)
            .where(LilaYolo.annotation[0]["is_train"].as_boolean() == self.split)
        ).all()


    def __len__(self):
        return len(self.data)


    @transfer_retry
    def __getitem__(self, idx):

        if self._gcs_bucket is None:
            client = storage.Client.from_service_account_json(os.environ.get("GCS_SECRET"))
            self._gcs_storage = client.bucket(self.gcs_config.bucket)

        record = self.data[idx]
        labels = [
            [
            ann["class_id"],
            ann["norm_center_x"],
            ann["norm_center_y"],
            ann["norm_width"],
            ann["norm_height"]
            ]
            for ann in record.annotation if ann["class_id"] == 0
        ] # (n, 5) for each of the bounding boxes

        label_tensor = torch.zeros((0, 5), dtype=torch.float32) if not labels else torch.tensor(labels, dtype=torch.float32)

        filename = record.file_name
        blob = self._gcs_bucket.blob(self.gcs_prefix + filename)
        image = blob.download_as_bytes()
        image_tensor = self.transform(Image.open(io.BytesIO(image)))

        return image_tensor, label_tensor