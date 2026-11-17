"""Unaugmented Stage 1 data loading for Experiment 18."""

from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MagnetogramDataset(Dataset):
    def __init__(self, csv_file, image_directory):
        rows = pd.read_csv(csv_file)
        self.paths = [Path(image_directory) / value for value in rows["label"]]
        self.targets = rows["goes_class"].astype(int).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = TF.to_tensor(image.convert("L")).repeat(3, 1, 1)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, torch.tensor(self.targets[index], dtype=torch.long)


def build_loader(csv_file, image_directory, batch_size, num_workers, shuffle=False):
    dataset = MagnetogramDataset(csv_file, image_directory)
    options = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)


def class_counts(loader):
    return torch.bincount(torch.tensor(loader.dataset.targets), minlength=2)
