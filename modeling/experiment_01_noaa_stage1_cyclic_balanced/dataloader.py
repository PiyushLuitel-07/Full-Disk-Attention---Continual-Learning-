"""Data loading for the NOAA Stage 1 diagnostic experiment."""

import random
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MagnetogramDataset(Dataset):
    """Load the verified 512x512 magnetograms and their binary labels."""

    def __init__(self, csv_file, image_directory):
        rows = pd.read_csv(csv_file)
        self.paths = [Path(image_directory) / value for value in rows["label"]]
        self.targets = rows["goes_class"].astype(int).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as opened_image:
            image = opened_image.convert("L")
            tensor = TF.to_tensor(image).repeat(3, 1, 1)

        # MobileNet's pretrained weights expect ImageNet normalization.
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return tensor, target


class CyclicBalancedSampler(Sampler):
    """Use every FL and an equal NF subset, cycling through shuffled NF images."""

    def __init__(self, targets, seed=42):
        # Keep the original row indices for each class.
        self.fl_indices = [i for i, value in enumerate(targets) if value == 1]
        self.nf_indices = [i for i, value in enumerate(targets) if value == 0]

        # Shuffle the complete NF pool once before the first epoch.
        self.random = random.Random(seed)
        self.nf_pool = self.nf_indices.copy()
        self.random.shuffle(self.nf_pool)
        self.position = 0 #This records the next unused position in the shuffled NF list.
        self.epoch = 0 # starting from the 0th epoch
        self.cycle = 1 # starting from 1st cycle
        self.last_nf_indices = [] # store the NF indices used in the last epoch

# Number of samples per epoch is twice the number of FL images.
    def __len__(self):
        return 2 * len(self.fl_indices)
    # Refill the NF pool if necessary at the start of a new epoch.
    # This runs when the sampler reaches the end of all 38,015 NF images.
    def _refill(self, already_selected):
        """Start a new shuffle without duplicating NF images within an epoch."""
        self.cycle += 1 # increment the cycle count each time the NF pool is refilled
        self.nf_pool = self.nf_indices.copy()
        self.random.shuffle(self.nf_pool)

        # Move NF images already used in this epoch to the end of the new pool.
        selected = set(already_selected)
        # '''Move NF images already used in this epoch to the end of the new pool.'''
        self.nf_pool.sort(key=lambda index: index in selected)
        self.position = 0 #Starts reading the newly shuffled pool from the beginning.
    # Select NF images for an epoch
    def __iter__(self):
        # Draw as many NF images as there are FL images.
        selected_nf = []
        required = len(self.fl_indices)

        while len(selected_nf) < required:
            # It checks whether every NF image in the current shuffled pool has been used:
            if self.position == len(self.nf_pool):
                # If yes, it reshuffles all NF images and starts a new cycle:
                self._refill(selected_nf)

            needed = required - len(selected_nf) #how many NF images are still required;
            available = len(self.nf_pool) - self.position #how many NF images are still available in the current pool
            take = min(needed, available) #how many NF images to take from the current pool
            selected_nf.extend(self.nf_pool[self.position:self.position + take])
            self.position += take

        # Combine every FL image with the selected NF images, then shuffle.
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


def build_loaders(train_csv, holdout_csv, image_directory, batch_size=128,
                  num_workers=8, seed=42):
    """Build balanced training, natural-training, and holdout loaders."""
    # Both training loaders share the same original Stage 1 dataset.
    train_data = MagnetogramDataset(train_csv, image_directory)
    holdout_data = MagnetogramDataset(holdout_csv, image_directory)
    balanced_sampler = CyclicBalancedSampler(train_data.targets, seed=seed)

    return {
        # Updates model parameters using all FL and an equal NF subset.
        "balanced_train": _loader(
            train_data, batch_size, num_workers, sampler=balanced_sampler
        ),
        # Measures performance on the complete natural training distribution.
        "natural_train": _loader(train_data, batch_size, num_workers),
        # Selects the best epoch without changing model parameters.
        "holdout": _loader(holdout_data, batch_size, num_workers),
        # Exposed so the training script can log NF coverage.
        "balanced_sampler": balanced_sampler,
    }


def build_evaluation_loader(csv_file, image_directory, batch_size=128,
                            num_workers=8):
    """Build an unchanged loader for final validation or testing."""
    dataset = MagnetogramDataset(csv_file, image_directory)
    return _loader(dataset, batch_size, num_workers)
