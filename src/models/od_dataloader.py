import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

from .od_dataset import ObjectDetectionDataset

def object_detection_collate(original_batch):

    img = torch.stack([item[0] for item in original_batch])
    labels = []
    for idx, item in enumerate(original_batch):
        label_tensors = item[1]
        for label_tensor in label_tensors:
            labels.append(torch.cat([torch.tensor([idx], dtype=torch.float32), label_tensor]))

    labels = torch.stack(labels) if labels else torch.zeros((0,6), dtype=torch.float32)
    return img, labels

def return_dataloader():
    transform = transforms.Compose([
        transforms.Resize([640, 640]),
        transforms.ToTensor()
    ])
    od_dataset = ObjectDetectionDataset(transforms=transform)
    sampler = WeightedRandomSampler([row.uiqm for row in od_dataset.data], len(od_dataset), replacement=True)

    dataloader = DataLoader(dataset=od_dataset,
                            sampler=sampler,
                            collate_fn=object_detection_collate,
                            batch_size=16,
                            num_workers=4,
                            pin_memory=True)
    return dataloader

