from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from torchvision.transforms import v2
from .c_dataset import ClassificationDataset, init_gcs_worker

def collate_fn(original_batch):

    img = torch.stack([item[0] for item in original_batch])
    labels = {
        "subfamily": torch.tensor([item[1].get("subfamily") for item in original_batch]),
        "genus": torch.tensor([item[1].get("genus") for item in original_batch]),
        "species": torch.tensor([item[1].get("species") for item in original_batch])
    }
    return img, labels

def collate_fn_ultralytics(original_batch):

    img = torch.stack([item[0] for item in original_batch])
    labels = {
        "cls": torch.tensor([item[1].get("species") for item in original_batch])
    }
    return img, labels


def class_dataloader(mode: str = "custom", split: str = "train", batch: int = 16, max_samples: int = None):

    base_transform = [
        v2.Resize(size=(224, 224)),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32)
    ]

    if split == "train":
        max_rotate, min_rotate = 90, 0
        add_transforms = [
            v2.RandomRotation(degrees=(min_rotate, max_rotate)),
            v2.ElasticTransform(alpha=75),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(brightness=0.4, contrast=0.2, hue=0.015, saturation=0.3),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ]
        transforms = add_transforms + base_transform
        transforms_composed = v2.Compose(transforms)
        class_dataset = ClassificationDataset(split=split, transform=transforms_composed, max_samples=max_samples)
        sampler = WeightedRandomSampler([max(row.uiqm, 1e-6) for row in class_dataset.data], num_samples=len(class_dataset))

    else:
        transforms_composed = v2.Compose(base_transform)
        class_dataset = ClassificationDataset(split=split, transform=transforms_composed, max_samples=max_samples)
        sampler = None

    if mode == "custom":
        collate_function = collate_fn
    else:
        collate_function = collate_fn_ultralytics

    class_dataloader = DataLoader(
        dataset=class_dataset,
        sampler=sampler,
        batch_size=batch,
        num_workers=8,
        prefetch_factor=2,
        worker_init_fn=init_gcs_worker,
        collate_fn=collate_function,
        shuffle=False,
        pin_memory=True
    )
    return class_dataloader


