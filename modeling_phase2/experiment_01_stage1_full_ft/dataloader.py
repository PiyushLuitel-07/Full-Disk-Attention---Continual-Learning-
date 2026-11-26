"""Data loading for Phase 2 Experiment 1 (Stage 1 full fine-tuning)."""

import random
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms


class MagnetogramDataset(Dataset):
    """Load 512x512 magnetograms and their binary targets with symmetric grayscale normalization."""

    def __init__(self, csv_file, image_directory, grayscale_mean=0.5, grayscale_std=0.5, augment=False):
        rows = pd.read_csv(csv_file)
        self.paths = [Path(image_directory) / value for value in rows["label"]]
        self.targets = rows["goes_class"].astype(int).tolist()
        self.mean = (grayscale_mean, grayscale_mean, grayscale_mean)
        self.std = (grayscale_std, grayscale_std, grayscale_std)
        self.augment = (
            transforms.RandomChoice([
                transforms.Lambda(lambda img: img),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomRotation(degrees=5),
            ])
            if augment
            else None
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as opened_image:
            image = opened_image.convert("L")
            # Augment only FL (target == 1) images since they repeat every epoch; NF images already cycle.
            if self.augment is not None and self.targets[index] == 1:
                image = self.augment(image)
            # Repeat single-channel magnetogram into 3 identical channels, then map [0, 1] -> [-1, 1].
            tensor = TF.to_tensor(image).repeat(3, 1, 1)

        tensor = TF.normalize(tensor, self.mean, self.std)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return tensor, target


class CyclicBalancedSampler(Sampler):
    """Sample all FL images and an equal NF subset per epoch, cycling through all NF images."""

    def __init__(self, targets, seed=42):
        self.fl_indices = [i for i, value in enumerate(targets) if value == 1]
        self.nf_indices = [i for i, value in enumerate(targets) if value == 0]

        self.random = random.Random(seed)
        self.nf_pool = self.nf_indices.copy()
        self.random.shuffle(self.nf_pool)
        self.position = 0
        self.epoch = 0
        self.cycle = 1
        self.last_nf_indices = []

    def __len__(self):
        return 2 * len(self.fl_indices)

    def _refill(self, already_selected):
        """Reshuffle the NF pool and defer indices already used in the current epoch."""
        self.cycle += 1
        self.nf_pool = self.nf_indices.copy()
        self.random.shuffle(self.nf_pool)

        selected = set(already_selected)
        self.nf_pool.sort(key=lambda index: index in selected)
        self.position = 0

    def __iter__(self):
        selected_nf = []
        required = len(self.fl_indices)

        while len(selected_nf) < required:
            if self.position == len(self.nf_pool):
                self._refill(selected_nf)

            needed = required - len(selected_nf)
            available = len(self.nf_pool) - self.position
            take = min(needed, available)
            selected_nf.extend(self.nf_pool[self.position:self.position + take])
            self.position += take

        epoch_indices = self.fl_indices + selected_nf
        self.random.shuffle(epoch_indices)
        self.epoch += 1
        self.last_nf_indices = selected_nf
        return iter(epoch_indices)


def _loader(dataset, batch_size, num_workers, sampler=None, shuffle=False):
    """Create one DataLoader with shared performance settings."""
    settings = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": shuffle,
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
    seed=42,
    grayscale_mean=0.5,
    grayscale_std=0.5,
    augment=False,
):
    """Build balanced training, natural-training, and holdout loaders."""
    train_data = MagnetogramDataset(
        train_csv, image_directory, grayscale_mean=grayscale_mean, grayscale_std=grayscale_std, augment=augment
    )
    holdout_data = MagnetogramDataset(
        holdout_csv, image_directory, grayscale_mean=grayscale_mean, grayscale_std=grayscale_std
    )
    balanced_sampler = CyclicBalancedSampler(train_data.targets, seed=seed)

    return {
        "balanced_train": _loader(train_data, batch_size, num_workers, sampler=balanced_sampler),
        "natural_train": _loader(train_data, batch_size, num_workers),
        "holdout": _loader(holdout_data, batch_size, num_workers),
        "balanced_sampler": balanced_sampler,
    }


def build_evaluation_loader(
    csv_file,
    image_directory,
    batch_size=64,
    num_workers=8,
    grayscale_mean=0.5,
    grayscale_std=0.5,
):
    """Build an evaluation DataLoader for validation or test splits."""
    dataset = MagnetogramDataset(
        csv_file, image_directory, grayscale_mean=grayscale_mean, grayscale_std=grayscale_std
    )
    return _loader(dataset, batch_size, num_workers)
