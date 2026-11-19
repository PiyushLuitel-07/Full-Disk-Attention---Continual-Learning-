"""Unaugmented Stage 1 data loading for Experiment 18."""

from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, RandomRotation


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MagnetogramDataset(Dataset):
    def __init__(self, csv_file, image_directory, augmentation="none"):
        rows = pd.read_csv(csv_file)
        self.paths = [Path(image_directory) / value for value in rows["label"]]
        self.targets = rows["goes_class"].astype(int).tolist()
        self.augmentation = augmentation
        self.augment_types = ["none"] * len(self.paths)

        # Preserve every original and append one dynamically rotated copy of
        # each FL sample. Class counts therefore reflect the effective
        # optimization dataset and can be used to recompute loss weights.
        if augmentation in ("rotation5_fl_copy", "rotation5_hflip_fl_copy"):
            flare_paths = [
                path
                for path, target in zip(self.paths, self.targets)
                if target == 1
            ]
            self.paths.extend(flare_paths)
            self.targets.extend([1] * len(flare_paths))
            appended_type = (
                "rotation_hflip_random"
                if augmentation == "rotation5_hflip_fl_copy"
                else "rotation"
            )
            self.augment_types.extend([appended_type] * len(flare_paths))
        elif augmentation == "rotation5_and_hflip_fl_copies":
            flare_paths = [
                path
                for path, target in zip(self.paths, self.targets)
                if target == 1
            ]
            # Add two distinct views per FL: one rotation-only copy and one
            # horizontal-flip-only copy.
            self.paths.extend(flare_paths)
            self.targets.extend([1] * len(flare_paths))
            self.augment_types.extend(["rotation"] * len(flare_paths))
            self.paths.extend(flare_paths)
            self.targets.extend([1] * len(flare_paths))
            self.augment_types.extend(["hflip"] * len(flare_paths))

        self.rotation = RandomRotation(
            degrees=5,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("L")
            augment_type = self.augment_types[index]
            rotate_all_sample = (
                self.augmentation == "rotation5" and torch.rand(()) < 0.5
            )
            if rotate_all_sample or augment_type in ("rotation", "rotation_hflip_random"):
                image = self.rotation(image)
            if (
                augment_type == "hflip"
                or (augment_type == "rotation_hflip_random" and torch.rand(()) < 0.5)
            ):
                image = TF.hflip(image)
            tensor = TF.to_tensor(image).repeat(3, 1, 1)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, torch.tensor(self.targets[index], dtype=torch.long)


def build_loader(
    csv_file,
    image_directory,
    batch_size,
    num_workers,
    shuffle=False,
    augmentation="none",
):
    dataset = MagnetogramDataset(csv_file, image_directory, augmentation=augmentation)
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
