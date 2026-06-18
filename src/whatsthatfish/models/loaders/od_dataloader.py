"""Dataloader and Ultralytics trainer glue for the YOLO11 detector.

Bridges our ObjectDetectionDataset into the Ultralytics training loop: a collate
that emits the flat batch dict Ultralytics expects, a builder that wires UIQM
(×conf for LC stages) weighted sampling and augmentation, and Trainer/Validator
subclasses that swap in our dataloader and skip Ultralytics' redundant /255.
"""

from copy import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import v2
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

from ...models.datasets.od_dataset import ObjectDetectionDataset


def object_detection_collate(original_batch):
    """Collate per-image (img, label, file) tuples into Ultralytics' batch dict.

    Stacks images and flattens variable-length boxes across the batch into the
    parallel `batch_idx`/`cls`/`bboxes` tensors Ultralytics' loss expects, plus
    the `im_file`/`ori_shape`/`ratio_pad` bookkeeping the validator reads.
    Images with no boxes simply contribute nothing to the flattened label arrays.
    """

    img = torch.stack([item[0] for item in original_batch])
    batch_idx = []
    cls = []
    bboxes = []

    for idx, item in enumerate(original_batch):
        label_tensors = item[1]
        if label_tensors.shape[0] > 0:
            batch_idx.append(torch.full((label_tensors.shape[0],), idx))
            cls.append(label_tensors[:, 0:1])
            bboxes.append(label_tensors[:, 1:5])

    return {
        "img": img,
        "batch_idx": torch.cat(batch_idx, 0) if batch_idx else torch.zeros(0),
        "cls": torch.cat(cls, 0) if cls else torch.zeros((0, 1)),
        "bboxes": torch.cat(bboxes, 0) if bboxes else torch.zeros((0, 4)),
        "im_file": [item[2] for item in original_batch],
        "ori_shape": [tuple(img.shape[1:]) for img in img],
        "ratio_pad": [((1.0, 1.0), (0.0, 0.0)) for _ in original_batch],
    }


def od_dataloader(
    mode: str, dataset: str, batch_size: int = 16, max_samples: int = None
):
    """Build a detector DataLoader for the given split and curriculum dataset.

    Train mode adds colour/flip/scale jitter and a WeightedRandomSampler whose
    weights are UIQM (and ×conf for LC1/LC2) — biasing each epoch toward higher
    quality / higher confidence boxes. Val mode is plain resize, no sampler. The
    no-op `reset` attribute is attached so Ultralytics can call it harmlessly.
    """
    base_transform = [
        v2.Resize(size=(640, 640)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    if mode == "train":
        to_insert = [
            v2.ColorJitter(brightness=0.4, saturation=0.8, hue=0.015),
            v2.RandomHorizontalFlip(),
            v2.ScaleJitter(target_size=(640, 640), scale_range=(0.5, 2.0)),
        ]
        base_transform = to_insert + base_transform
        transform = v2.Compose(base_transform)
        od_dataset = ObjectDetectionDataset(
            dataset=dataset, transforms=transform, split=mode, max_samples=max_samples
        )
        if dataset in ("lc1", "lc2"):
            weights = [
                max(row.uiqm or 0.0, 1e-6) * max(row.conf or 0.0, 1e-6)
                for row in od_dataset.data
            ]
        else:
            weights = [max(row.uiqm or 0.0, 1e-6) for row in od_dataset.data]
        sampler = WeightedRandomSampler(weights, len(od_dataset), replacement=True)
    else:
        transform = v2.Compose(base_transform)
        od_dataset = ObjectDetectionDataset(
            dataset=dataset, transforms=transform, split=mode, max_samples=max_samples
        )
        sampler = None

    dataloader = DataLoader(
        dataset=od_dataset,
        sampler=sampler,
        shuffle=False,
        collate_fn=object_detection_collate,
        batch_size=batch_size,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    dataloader.reset = lambda: None
    return dataloader


class CustomDetectionValidator(DetectionValidator):
    """Validator that undoes our already-[0,1] images for the parent's /255."""

    def preprocess(self, batch):
        # images are already [0,1] from ToTensor(); scale up so parent's /255 restores [0,1]
        batch["img"] = batch["img"] * 255
        return super().preprocess(batch)


class CustomDetectionTrainer(DetectionTrainer):
    """Ultralytics DetectionTrainer wired to our dataset, dataloader and validator.

    `dataset`/`max_samples` are set on the instance before training to pick the
    curriculum stage. Overrides feed our weighted dataloader, move batches to the
    device without re-normalizing (images arrive already in [0,1]), and use the
    matching CustomDetectionValidator.
    """

    max_samples: int = None
    dataset: str = "lila"

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        return od_dataloader(
            dataset=self.dataset,
            mode=mode,
            batch_size=batch_size,
            max_samples=self.max_samples,
        )

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float()
        batch["cls"] = batch["cls"].to(self.device)
        batch["bboxes"] = batch["bboxes"].to(self.device)
        batch["batch_idx"] = batch["batch_idx"].to(self.device)
        return batch

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CustomDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
