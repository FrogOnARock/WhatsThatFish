from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from sqlalchemy import select, desc

from ...database.config import get_session_factory
from ...database.models import InatClassificationDataset


class ClassificationDataset(Dataset):
    def __init__(self,
                 split: str = "train",
                 transform=None,
                 max_samples: int = None,
                 local_base_dir: str = None,
                 ):

        self.transforms = transform
        self.local_base_dir = Path(local_base_dir) if local_base_dir else (Path(__file__).parents[2] / "loaders/classification_images")
        self.session_factory = get_session_factory()
        self.split = 0 if split == "train" else 1

        query = (
            select(
                InatClassificationDataset.photo_uuid,
                InatClassificationDataset.uiqm,
                InatClassificationDataset.filename,
                InatClassificationDataset.zero_indexed_subfamily,
                InatClassificationDataset.zero_indexed_genus,
                InatClassificationDataset.zero_indexed_species,
            ).where(InatClassificationDataset.train == self.split)
            .order_by(desc(InatClassificationDataset.uiqm))
        )
        if max_samples:
            query = query.limit(max_samples)

        with self.session_factory() as session:
            self.data = session.execute(query).all()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        image_pil = Image.open(self.local_base_dir / record.filename).convert("RGB")

        img_tensor = self.transforms(image_pil)

        label = {
            "subfamily": record.zero_indexed_subfamily,
            "genus": record.zero_indexed_genus,
            "species": record.zero_indexed_species,
        }
        return img_tensor, label
