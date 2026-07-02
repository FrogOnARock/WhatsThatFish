"""DataLoader builder for the hierarchical species classifier.

Wraps ClassificationDataset with the train/val transform pipelines (letterbox →
augment → 5-channel conversion) and a UIQM-weighted sampler on train, and offers
two collates: our hierarchical-label dict, or a slim species-only dict for the
Ultralytics classifier path.
"""

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from torchvision.transforms import v2
from ...models.datasets.c_dataset import ClassificationDataset
from ...transforms.five_channel_conversion import AddMultiChannel
from ...transforms.letterbox_resize import LetterboxResize


class NumpyToTensor:
    """Compose step: wrap AddMultiChannel's (5,H,W) numpy output into a float tensor.

    AddMultiChannel is numpy-native (for the torch-free serving path); the training
    pipeline needs tensors for the collate's `torch.stack`, so this is the last
    step after it. A module-level class (not a lambda) so it pickles for DataLoader
    workers.
    """

    def __call__(self, arr):
        return torch.from_numpy(arr).float()


def collate_fn(original_batch):
    """Stack the batch into (images, label dict) with all three taxonomic levels.

    The label dict carries family/genus/species index tensors plus the per-sample
    `topup` flag so eval can split geographic vs IID val macro accuracy.
    """

    img = torch.stack([item[0] for item in original_batch])
    labels = {
        "family": torch.tensor([item[1].get("family") for item in original_batch]),
        "genus": torch.tensor([item[1].get("genus") for item in original_batch]),
        "species": torch.tensor([item[1].get("species") for item in original_batch]),
        # Per-sample topped-up flag → lets metrics split geographic vs IID val macro.
        "topup": torch.tensor([item[1].get("topup", 0) for item in original_batch]),
    }
    return img, labels


def collate_fn_ultralytics(original_batch):
    """Species-only collate matching the Ultralytics classifier's {img, cls} format."""

    img = torch.stack([item[0] for item in original_batch])
    labels = {"cls": torch.tensor([item[1].get("species") for item in original_batch])}
    return {"img": img, "cls": labels}


def class_dataloader(
    mode: str = "custom", split: str = "train", batch: int = 16, tuning: bool = False
):
    """Build the classifier DataLoader for a split.

    Train applies the full augmentation stack (rotation, elastic, flip, jitter,
    sharpness) between letterbox and 5-channel conversion, plus a UIQM-weighted
    sampler; val is letterbox → 5-channel only. `mode` selects our hierarchical
    collate ("custom") or the slim Ultralytics species-only one. Output tensors
    are (5, 320, 320).
    """

    base_transform = [
        LetterboxResize(320),
        AddMultiChannel(),
        NumpyToTensor(),
    ]

    if split == "train":
        max_rotate, min_rotate = 90, 0
        add_transforms = [
            v2.RandomRotation(degrees=(min_rotate, max_rotate)),
            v2.ElasticTransform(alpha=75),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.4, contrast=0.2, hue=0.015, saturation=0.3),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ]
        transforms = (
            [LetterboxResize(320)] + add_transforms + [AddMultiChannel(), NumpyToTensor()]
        )
        transforms_composed = v2.Compose(transforms)
        class_dataset = ClassificationDataset(
            split=split, transform=transforms_composed, tuning=tuning
        )
        weights = [max(row.uiqm or 0.0, 1e-6) for row in class_dataset.data]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(class_dataset),
        )

    else:
        transforms_composed = v2.Compose(base_transform)
        class_dataset = ClassificationDataset(
            split=split, transform=transforms_composed, tuning=tuning
        )
        sampler = None

    if mode == "custom":
        collate_function = collate_fn
    else:
        collate_function = collate_fn_ultralytics

    dataloader = DataLoader(
        dataset=class_dataset,
        sampler=sampler,
        batch_size=batch,
        num_workers=8,
        prefetch_factor=4,
        collate_fn=collate_function,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader
