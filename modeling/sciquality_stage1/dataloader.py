import random
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import InterpolationMode

'''
We use these values because pretrained MobileNet expects inputs with the same normalization used during
its original ImageNet training. Although the magnetogram is grayscale, it is copied into three channels, 
and each channel receives its corresponding ImageNet normalization. ie Z score noramlization is applied to each channel separately.
'''
IMAGENET_MEAN = (0.485, 0.456, 0.406) ## Red, green, blue means
IMAGENET_STD = (0.229, 0.224, 0.225)
VALID_AUGMENTATIONS = {
    "original", "horizontal_flip", "vertical_flip",
    "rotation_180", "small_rotation",
}

class MagnetogramDataset(Dataset):
    """Load one class with one fixed image transformation."""

    def __init__(
        self, csv_file, image_directory, class_value=None, augmentation="original"
    ):
        self.rows = pd.read_csv(csv_file)
        self.image_directory = Path(image_directory)
        self.augmentation = augmentation

        if not {"label", "goes_class"}.issubset(self.rows.columns):
            raise ValueError("CSV must contain label and goes_class columns.")
        if not self.rows["goes_class"].isin([0, 1]).all():
            raise ValueError("goes_class must contain only 0 (NF) and 1 (FL).")
        if augmentation not in VALID_AUGMENTATIONS:
            raise ValueError(f"Unknown augmentation: {augmentation}")
        if class_value is not None:
            self.rows = self.rows[self.rows["goes_class"] == class_value]
        self.rows = self.rows.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)
    
# Loads one magnetogram, converts it to grayscale and resizes it to 512×512.
    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.image_directory / row["label"]) as image:
            image = TF.resize(image.convert("L"), [512, 512])

        if self.augmentation == "horizontal_flip":
            image = TF.hflip(image)
        elif self.augmentation == "vertical_flip":
            image = TF.vflip(image)
        elif self.augmentation == "rotation_180":
            image = TF.vflip(TF.hflip(image))
        elif self.augmentation == "small_rotation":
            image = TF.rotate(
                image,
                random.uniform(-5, 5),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

        # Converts the grayscale image into three identical channels and applies ImageNet normalization.
        tensor = TF.to_tensor(image).repeat(3, 1, 1)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, torch.tensor(int(row["goes_class"]), dtype=torch.long)

'''
tensor: the processed 3×512×512 magnetogram.
torch.tensor(...): its class label, 0 for NF and 1 for FL.

Grayscale magnetogram: 1 × 512 × 512
                 ↓ repeat channel
MobileNet input:       3 × 512 × 512
'''

class RepeatedDataset(Dataset):
    """Repeat original samples in memory until the requested length."""

    def __init__(self, dataset, length):
        self.dataset = dataset
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]



'''
Guarantees all original FL images are used first, 
then cycles evenly through enabled augmented FL views until the required balanced length is reached.
'''
class ControlledFlareDataset(Dataset):
    """Use original FL images first, then cycle through augmented views."""

    def __init__(self, flare_views, length):
        self.views = flare_views
        self.original_count = len(flare_views[0])
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Guarantee that every original FL image appears once.
        if index < self.original_count:
            return self.views[0][index]

        extra_index = index - self.original_count
        augmented_views = self.views[1:] or self.views
        view = augmented_views[extra_index % len(augmented_views)]
        sample_index = (extra_index // len(augmented_views)) % self.original_count
        return view[sample_index]


def build_train_loader(
    csv_file, image_directory, augmentations, batch_size, num_workers=8
):
    """Create equal-size NF and controlled FL datasets for training."""
    # Original is always first; additional views are supplied by the sweep.
    added_views = dict.fromkeys(name for name in augmentations if name != "original")
    views = ["original", *added_views]
    nf_data = MagnetogramDataset(csv_file, image_directory, class_value=0)
    fl_views = [
        MagnetogramDataset(csv_file, image_directory, class_value=1, augmentation=name)
        for name in views
    ]

    # Fill only the class gap: no augmented files are written to disk.
    balanced_length = max(len(nf_data), len(fl_views[0]))
    training_data = ConcatDataset([
        RepeatedDataset(nf_data, balanced_length),
        ControlledFlareDataset(fl_views, balanced_length),
    ])
    return DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def build_evaluation_loader(
    csv_file, image_directory, batch_size, num_workers=8
):
    """Create an unaugmented loader for holdout, validation or test data."""
    dataset = MagnetogramDataset(csv_file, image_directory)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
