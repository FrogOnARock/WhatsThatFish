from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from sqlalchemy import select, desc, func

from ...database.config import get_session_factory
from ...database.models import InatClassificationDataset


class ClassificationDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        transform=None,
        max_samples: int = None,
        local_base_dir: str = None,
        min_bbox_count: int = 250,
        crop_margin: float = 0.15,
    ):

        self.transforms = transform
        # Fraction of the detector box width/height added as context on each side
        # before cropping. Must match the margin used in the inference crop path
        # (consumer of inference/bbox_inference.py) or the train/serve gap reopens.
        self.crop_margin = crop_margin
        self.local_base_dir = (
            Path(local_base_dir)
            if local_base_dir
            else (Path(__file__).parents[2] / "data/classification_images")
        )
        self.session_factory = get_session_factory()
        # prepare_inat.assign_split: train=True → training, train=False → validation
        self.split = split == "train"

        # Exclude taxa the detector failed on: keep only taxa where at least
        # min_bbox_count images received a proposed bbox. `!= "null"` excludes
        # both JSON null (bbox proposal ran, no detection) and SQL NULL
        # (not yet processed — dropped by WHERE via NULL propagation).
        with self.session_factory() as session:
            rows = session.execute(
                select(InatClassificationDataset.taxon_id)
                .where(InatClassificationDataset.proposed_bbox != "null")
                .group_by(InatClassificationDataset.taxon_id)
                .having(func.count() >= min_bbox_count)
            )
            taxon_ids = [r.taxon_id for r in rows]

        query = (
            select(
                InatClassificationDataset.photo_uuid,
                InatClassificationDataset.uiqm,
                InatClassificationDataset.filename,
                InatClassificationDataset.proposed_bbox,
                InatClassificationDataset.zero_indexed_subfamily,
                InatClassificationDataset.zero_indexed_genus,
                InatClassificationDataset.zero_indexed_species,
            )
            .where(InatClassificationDataset.train == self.split)
            .where(InatClassificationDataset.taxon_id.in_(taxon_ids))
            # Drop detector misses: only train on rows that have a real crop, so
            # the classifier never sees a full frame it won't see at inference.
            .where(InatClassificationDataset.proposed_bbox != "null")
            .order_by(desc(InatClassificationDataset.uiqm))
        )
        if max_samples:
            query = query.limit(max_samples)

        with self.session_factory() as session:
            self.data = session.execute(query).all()

    def __len__(self):
        return len(self.data)

    def _crop_with_margin(self, image, bbox):
        """Crop `image` (PIL, RGB) to the detector box expanded by self.crop_margin.

        bbox is the JSONB `proposed_bbox`: absolute pixels {"x1","y1","x2","y2"},
        already clipped to image bounds upstream in bbox_inference.py.

        Goal: return image.crop((x1', y1', x2', y2')) where each side is pushed out
        by self.crop_margin * box_width (x) / box_height (y), then clamped to
        [0, W] / [0, H]. Context helps fine-grained ID (recovers clipped fins);
        too much reintroduces the full-frame problem this crop is meant to fix.
        """
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        x1, y1, x2, y2 = (
            min(x1 - (x1 * self.crop_margin / 2), 0),
            min(y1 - (y1 * self.crop_margin / 2), 0),
            max(x2 + (x2 * self.crop_margin / 2), image.size[0]),
            max(y2 + (y2 * self.crop_margin / 2), image.size[1]),
        )
        cropped_img = image.crop((x1, y1, x2, y2))
        return cropped_img

    def _get_one_photo(self, idx):
        record = self.data[idx]
        image_pil = Image.open(self.local_base_dir / record.filename).convert("RGB")
        image_pil = self._crop_with_margin(image_pil, record.proposed_bbox)

        img_tensor = self.transforms(image_pil)

        label = {
            "subfamily": record.zero_indexed_subfamily,
            "genus": record.zero_indexed_genus,
            "species": record.zero_indexed_species,
        }
        return img_tensor, label

    def __getitems__(self, indices: list[int]):
        """
        Use thread pools to execute multiple _get_one_photo calls at the same time
        rather than running individual __getitem__ processes
        """
        with ThreadPoolExecutor(max_workers=len(indices)) as executor:
            batch = list(executor.map(self._get_one_photo, indices))

        return batch

    def __getitem__(self, idx: int):
        """
        Retain original __getitem__ call
        """
        return self._get_one_photo(idx)
