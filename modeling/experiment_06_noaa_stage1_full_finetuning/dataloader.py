"""Data loading for Experiment 6: full-data, full-model fine-tuning."""

from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MagnetogramDataset(Dataset):
    """Load original 512x512 magnetograms and binary NF/FL labels."""

    def __init__(self, csv_file, image_directory):
        rows = pd.read_csv(csv_file)
        self.paths = [Path(image_directory) / value for value in rows["label"]]
        self.targets = rows["goes_class"].astype(int).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # The stored images are already 512x512, so no resizing is required.
        with Image.open(self.paths[index]) as opened_image:
            image = opened_image.convert("L")

            # Repeat the grayscale channel because pretrained MobileNet expects
            # three input channels.
            tensor = TF.to_tensor(image).repeat(3, 1, 1)

        # Use the normalization expected by the ImageNet initialization.
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return tensor, target


def _loader(dataset, batch_size, num_workers, shuffle=False, sampler=None):
    """Create a DataLoader with shared performance settings."""
    settings = {
        "dataset": dataset,
        "batch_size": batch_size,
        # A distributed sampler controls ordering, so it cannot be combined
        # with DataLoader's own shuffle option.
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        settings.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**settings)


def build_loaders(
    train_csv,
    holdout_csv,
    image_directory,
    batch_size=64,
    num_workers=8,
    distributed=False,
    rank=0,
    world_size=1,
    seed=42,
):
    """Build the complete training loader and chronological holdout loader."""
    train_data = MagnetogramDataset(train_csv, image_directory)
    holdout_data = MagnetogramDataset(holdout_csv, image_directory)

    # When DDP is enabled, each GPU receives a different part of the training
    # set. Together, the GPUs still use all training images once per epoch.
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )

    return {
        # All 46,638 original training images are used in every epoch.
        "train": _loader(
            train_data,
            batch_size,
            num_workers,
            shuffle=True,
            sampler=train_sampler,
        ),
        # Holdout data remains unchanged and is never used for optimization.
        "holdout": _loader(holdout_data, batch_size, num_workers),
        "train_sampler": train_sampler,
        "class_counts": torch.bincount(
            torch.tensor(train_data.targets), minlength=2
        ),
    }


def build_evaluation_loader(
    csv_file,
    image_directory,
    batch_size=64,
    num_workers=8,
):
    """Build an unchanged loader for final validation or test evaluation."""
    dataset = MagnetogramDataset(csv_file, image_directory)
    return _loader(dataset, batch_size, num_workers)
