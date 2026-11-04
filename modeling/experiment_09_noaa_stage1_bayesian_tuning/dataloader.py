"""Data loading for Experiment 9: full-data mildly weighted fine-tuning."""

from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# MobileNet begins with ImageNet-pretrained weights, so its expected input
# normalization is retained while all model layers are fine-tuned.
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
        # Images are already stored at 512x512, so resizing is unnecessary.
        with Image.open(self.paths[index]) as opened_image:
            image = opened_image.convert("L")

            # MobileNet expects RGB-shaped input. Repeating the grayscale
            # channel changes only the tensor shape, not the image content.
            tensor = TF.to_tensor(image).repeat(3, 1, 1)

        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return tensor, target


def _loader(dataset, batch_size, num_workers, shuffle=False, sampler=None):
    """Create a DataLoader with the shared performance settings."""
    settings = {
        "dataset": dataset,
        "batch_size": batch_size,
        # DistributedSampler controls ordering when DDP is enabled.
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

    # Under DDP, each GPU processes a separate part of the same epoch. Across
    # all GPUs, the complete training set is still used once per epoch.
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
        # Use all 46,638 original images without balancing or augmentation.
        "train": _loader(
            train_data,
            batch_size,
            num_workers,
            shuffle=True,
            sampler=train_sampler,
        ),
        # Holdout images are evaluated only and never update the model.
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


