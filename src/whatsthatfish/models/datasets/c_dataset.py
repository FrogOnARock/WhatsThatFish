from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from torch.utils.data import Dataset
from sqlalchemy import select, desc, func

from ...database.config import get_session_factory
from ...database.models import InatClassificationDataset

load_dotenv()


class ClassificationDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        transform=None,
        tuning: bool = False,
        local_base_dir: str = None,
        min_bbox_count: int = 100,
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
        self.tuning = tuning
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

        stmt = (
            select(
                InatClassificationDataset.photo_uuid,
                InatClassificationDataset.uiqm,
                InatClassificationDataset.filename,
                InatClassificationDataset.proposed_bbox,
                InatClassificationDataset.zero_indexed_family,
                InatClassificationDataset.zero_indexed_genus,
                InatClassificationDataset.zero_indexed_species,
                InatClassificationDataset.val_topup,
                func.row_number()
                .over(
                    partition_by=InatClassificationDataset.taxon_id,
                    order_by=desc(InatClassificationDataset.uiqm),
                )
                .label("row_num"),
            )
            .where(InatClassificationDataset.train == self.split)
            .where(InatClassificationDataset.taxon_id.in_(taxon_ids))
            # Drop detector misses: only train on rows that have a real crop, so
            # the classifier never sees a full frame it won't see at inference.
            .where(InatClassificationDataset.proposed_bbox != "null")
        )
        cte = stmt.cte("data")
        outer = select(
            cte.c.photo_uuid,
            cte.c.uiqm,
            cte.c.filename,
            cte.c.proposed_bbox,
            cte.c.zero_indexed_family,
            cte.c.zero_indexed_genus,
            cte.c.zero_indexed_species,
            cte.c.val_topup,
            cte.c.row_num,
        )

        if self.tuning:
            max_samples = 100 if self.split else 20
            # Filter on the CTE column, not outer.c.* — accessing .c on a Select
            # coerces it into an anonymous subquery (anon_1) and adds it as a second
            # FROM element alongside `data`, producing the cartesian-product warning.
            outer = outer.where(cte.c.row_num <= max_samples)

        with self.session_factory() as session:
            self.data = session.execute(outer).all()

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
        w, h = image.size
        box_w, box_h = x2 - x1, y2 - y1
        x1 = max(x1 - box_w * self.crop_margin, 0)
        y1 = max(y1 - box_h * self.crop_margin, 0)
        x2 = min(x2 + box_w * self.crop_margin, w)
        y2 = min(y2 + box_h * self.crop_margin, h)
        return image.crop((int(x1), int(y1), int(x2), int(y2)))

    def _get_one_photo(self, idx):
        record = self.data[idx]
        image_pil = Image.open(self.local_base_dir / record.filename).convert("RGB")
        image_pil = self._crop_with_margin(image_pil, record.proposed_bbox)

        img_tensor = self.transforms(image_pil)

        label = {
            "family": record.zero_indexed_family,
            "genus": record.zero_indexed_genus,
            "species": record.zero_indexed_species,
            # 1 = topped-up (IID) val row, 0 = geographic val or train. Carried so
            # the metrics can split geographic vs topped-up macro at eval time.
            "topup": int(bool(record.val_topup)),
        }
        return img_tensor, label

    def __getitems__(self, indices: list[int]):
        """
        Use thread pools to execute multiple _get_one_photo calls at the same time
        rather than running individual __getitem__ processes
        """
        with ThreadPoolExecutor(max_workers=8) as executor:
            batch = list(executor.map(self._get_one_photo, indices))

        return batch

    def __getitem__(self, idx: int):
        """
        Retain original __getitem__ call
        """
        return self._get_one_photo(idx)
