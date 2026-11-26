"""Load the original HMI images used by Experiment 15."""
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class MagnetogramDataset(Dataset):
    """Return an HMI magnetogram and its binary NF/FL target."""

    def __init__(self, csv_file, image_directory):
        if not Path(csv_file).is_file(): raise FileNotFoundError(f"Missing labels: {csv_file}")
        rows = pd.read_csv(csv_file)
        missing = {"label", "goes_class"} - set(rows.columns)
        if missing: raise ValueError(f"{csv_file} is missing columns: {sorted(missing)}")
        self.paths = [Path(image_directory) / path for path in rows["label"]]
        self.targets = rows["goes_class"].astype(int).tolist()
        if not self.paths or set(self.targets) - {0, 1}: raise ValueError(f"Invalid binary dataset: {csv_file}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = TF.to_tensor(image.convert("L")).repeat(3, 1, 1)
        return TF.normalize(tensor, MEAN, STD), torch.tensor(self.targets[index]).long()


def make_loader(csv_file, image_directory, batch_size, workers, shuffle=False):
    options = dict(dataset=MagnetogramDataset(csv_file, image_directory), batch_size=batch_size,
        num_workers=workers, shuffle=shuffle, pin_memory=torch.cuda.is_available())
    if workers: options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)


def class_counts(dataset):
    return torch.bincount(torch.tensor(dataset.targets), minlength=2).tolist()


def build_loaders(config, paths):
    batch, workers = int(config["batch_size"]), int(config["num_workers"])
    loaders = {
        "stage1_holdout": make_loader(paths["stage1_holdout"], paths["images"], batch, workers),
        "stage2_train": make_loader(paths["stage2_train"], paths["images"], batch, workers, True),
        "stage2_holdout": make_loader(paths["stage2_holdout"], paths["images"], batch, workers),
        "validation": make_loader(paths["validation"], paths["images"], batch, workers),
    }
    for name, loader in loaders.items():
        counts = class_counts(loader.dataset)
        if min(counts) == 0: raise ValueError(f"{name} must contain NF and FL samples: {counts}")
    sampling = {"method": "uniform_shuffle_without_replacement",
        "epoch_samples": len(loaders["stage2_train"].dataset), "replacement": False}
    return loaders, sampling
