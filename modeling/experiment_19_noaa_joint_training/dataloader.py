"""Data validation and loading for the three-stage joint baseline."""

from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class MagnetogramDataset(Dataset):
    def __init__(self, csv_file, image_directory, mean=0.5, std=0.5):
        rows = pd.read_csv(csv_file)
        self.paths = [Path(image_directory) / value for value in rows["label"]]
        self.targets = rows["goes_class"].astype(int).tolist()
        self.mean, self.std = float(mean), float(std)
        if self.std <= 0:
            raise ValueError("Normalization standard deviation must be positive")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = TF.to_tensor(image.convert("L"))
        tensor = TF.normalize(tensor, (self.mean,), (self.std,)).repeat(3, 1, 1)
        return tensor, torch.tensor(self.targets[index], dtype=torch.long)


def build_loader(csv_file, image_directory, batch_size, workers, shuffle=False, **norm):
    dataset = MagnetogramDataset(csv_file, image_directory, **norm)
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)


def class_counts(loader):
    return torch.bincount(torch.tensor(loader.dataset.targets), minlength=2)


def validate_csv_splits(train_csv, holdout_csvs, validation_csv):
    """Reject malformed files and image overlap across train/evaluation splits."""
    paths = {"train": train_csv, **holdout_csvs, "validation": validation_csv}
    frames = {}
    for name, path in paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {name} CSV: {path}")
        frame = pd.read_csv(path)
        if list(frame.columns) != ["label", "goes_class"]:
            raise ValueError(f"Unexpected columns in {name}: {frame.columns.tolist()}")
        if frame.empty or frame["label"].duplicated().any():
            raise ValueError(f"{name} must be nonempty with unique image paths")
        if not frame["goes_class"].isin([0, 1]).all():
            raise ValueError(f"{name} contains nonbinary targets")
        frames[name] = frame

    names = list(frames)
    for index, left in enumerate(names):
        left_paths = set(frames[left]["label"])
        for right in names[index + 1:]:
            overlap = left_paths & set(frames[right]["label"])
            if overlap:
                raise ValueError(f"{left}/{right} overlap: {len(overlap)} images")
    return frames
