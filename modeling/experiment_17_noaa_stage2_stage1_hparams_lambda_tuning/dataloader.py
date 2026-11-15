"""Self-contained original-image data loading for Experiment 17."""
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
        if not Path(csv_file).is_file(): raise FileNotFoundError(f"Missing label file: {csv_file}")
        rows = pd.read_csv(csv_file)
        missing = {"label", "goes_class"}.difference(rows.columns)
        if missing: raise ValueError(f"{csv_file} is missing columns: {sorted(missing)}")
        self.paths = [Path(image_directory) / value for value in rows["label"]]
        self.targets = rows["goes_class"].astype(int).tolist()
        if not self.paths: raise ValueError(f"Empty dataset: {csv_file}")
        if set(self.targets).difference({0, 1}): raise ValueError("Expected binary NF/FL targets")

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image_file:
            image = TF.to_tensor(image_file.convert("L")).repeat(3, 1, 1)
        return TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD), torch.tensor(self.targets[index]).long()


def loader(csv_file, image_directory, batch_size, workers, shuffle=False):
    settings = {"dataset": MagnetogramDataset(csv_file, image_directory), "batch_size": batch_size,
        "num_workers": workers, "shuffle": shuffle, "pin_memory": torch.cuda.is_available()}
    if workers > 0: settings.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**settings)


def class_counts(dataset):
    return torch.bincount(torch.tensor(dataset.targets), minlength=2).tolist()


def build_loaders(config, paths):
    batch, workers = int(config["batch_size"]), int(config["num_workers"])
    loaders = {
        "stage1_holdout": loader(paths["stage1_holdout_csv"], paths["image_directory"], batch, workers),
        "stage2_train": loader(paths["stage2_train_csv"], paths["image_directory"], batch, workers, shuffle=True),
        "stage2_holdout": loader(paths["stage2_holdout_csv"], paths["image_directory"], batch, workers),
        "future_validation": loader(paths["future_validation_csv"], paths["image_directory"], batch, workers),
    }
    for name, value in loaders.items():
        counts = class_counts(value.dataset)
        if min(counts) == 0: raise ValueError(f"{name} must contain both NF and FL examples; counts={counts}")
    metadata = {"method": "uniform_shuffle_without_replacement",
        "epoch_samples": len(loaders["stage2_train"].dataset), "replacement": False}
    return loaders, metadata
