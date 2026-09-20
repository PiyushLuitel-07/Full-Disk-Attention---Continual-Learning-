"""Datasets and DataLoaders for continual solar-flare learning."""

import random
import re
from pathlib import Path

import pandas as pd
import torch

from PIL import Image

from torch.utils.data import ConcatDataset, DataLoader, Dataset

import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------
# 1. BASIC IMAGE TRANSFORMATION
# ---------------------------------------------------------------------

class BasicTransform:
    """
    Resize a magnetogram and convert it to a PyTorch tensor.

    This transformation is used for:
        - original training images,
        - holdout images,
        - Fisher-information images.
    """

    def __init__(self, image_size=256):
        self.image_size = image_size

    def __call__(self, image):
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
        )

        return TF.to_tensor(image)


# ---------------------------------------------------------------------
# 2. FLARE-IMAGE AUGMENTATION
# ---------------------------------------------------------------------

class FlareTrainingTransform:
    """
    Randomly apply one augmentation to an FL training image.

    Available choices:
        1. Original image
        2. Horizontal flip
        3. Vertical flip
        4. Small rotation
        5. Polarity inversion

    This transformation must not be used for holdout or Fisher data.
    """

    def __init__(self, image_size=256, rotation_degrees=5):
        self.image_size = image_size
        self.rotation_degrees = rotation_degrees

    def __call__(self, image):
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
        )

        augmentation = random.choice(
            [
                "original",
                "horizontal_flip",
                "vertical_flip",
                "rotation",
                "polarity",
            ]
        )

        if augmentation == "horizontal_flip":
            image = TF.hflip(image)

        elif augmentation == "vertical_flip":
            image = TF.vflip(image)

        elif augmentation == "rotation":
            angle = random.uniform(
                -self.rotation_degrees,
                self.rotation_degrees,
            )

            image = TF.rotate(image, angle)

        # Convert the PIL image to a tensor with values from 0 to 1.
        image = TF.to_tensor(image)

        if augmentation == "polarity":
            # Swap the displayed positive and negative magnetic polarity.
            image = 1.0 - image

        return image


# ---------------------------------------------------------------------
# 3. MAIN MAGNETOGRAM DATASET
# ---------------------------------------------------------------------

class MagnetogramDataset(Dataset):
    """
    Load magnetogram paths and binary labels from a CSV file.

    Required CSV columns:
        label:
            Relative path to the JPG image.

        goes_class:
            0 for NF and 1 for FL.

    class_value:
        None loads both classes.
        0 loads only NF images.
        1 loads only FL images.
    """

    def __init__(
        self,
        csv_file,
        image_directory,
        transform=None,
        class_value=None,
    ):
        self.csv_file = Path(csv_file)
        self.image_directory = Path(image_directory)
        self.transform = transform

        annotations = pd.read_csv(self.csv_file)

        required_columns = {"label", "goes_class"}
        missing_columns = required_columns - set(annotations.columns)

        if missing_columns:
            raise ValueError(
                f"{self.csv_file} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if not annotations["goes_class"].isin([0, 1]).all():
            raise ValueError(
                f"{self.csv_file} contains labels other than 0 and 1."
            )

        if annotations["label"].isna().any():
            raise ValueError(
                f"{self.csv_file} contains missing image paths."
            )

        if class_value is not None:
            if class_value not in (0, 1):
                raise ValueError("class_value must be None, 0, or 1.")

            annotations = annotations[
                annotations["goes_class"] == class_value
            ]

        self.annotations = annotations.reset_index(drop=True)

        if len(self.annotations) == 0:
            raise ValueError(
                f"No samples were selected from {self.csv_file}."
            )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]

        image_path = self.image_directory / row["label"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Magnetogram was not found: {image_path}"
            )

        # Convert explicitly to one-channel grayscale.
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("L")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = TF.to_tensor(image)

        target = torch.tensor(
            int(row["goes_class"]),
            dtype=torch.long,
        )

        return image, target


# ---------------------------------------------------------------------
# 4. REPEAT A DATASET FOR CLASS BALANCING
# ---------------------------------------------------------------------

class RepeatedDataset(Dataset):
    """
    Repeat a smaller dataset until it reaches a requested length.

    This is used to repeat the minority class without creating duplicate
    files on disk.
    """

    def __init__(self, dataset, length):
        if len(dataset) == 0:
            raise ValueError("Cannot repeat an empty dataset.")

        if length <= 0:
            raise ValueError("Repeated dataset length must be positive.")

        self.dataset = dataset
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        actual_index = index % len(self.dataset)

        return self.dataset[actual_index]


# ---------------------------------------------------------------------
# 5. AUTOMATIC STAGE DISCOVERY
# ---------------------------------------------------------------------

def discover_stages(stage_directory):
    """
    Find every StageN_train.csv and matching StageN_holdout.csv file.

    Adding Stage4 later requires only:

        Stage4_train.csv
        Stage4_holdout.csv

    No Python stage list needs to be changed.
    """

    stage_directory = Path(stage_directory)
    stage_pattern = re.compile(r"Stage(\d+)_train\.csv$")

    stages = []

    for train_file in stage_directory.glob("Stage*_train.csv"):
        match = stage_pattern.match(train_file.name)

        if match is None:
            continue

        stage_number = int(match.group(1))

        holdout_file = (
            stage_directory
            / f"Stage{stage_number}_holdout.csv"
        )

        if not holdout_file.is_file():
            raise FileNotFoundError(
                f"Missing holdout file for Stage {stage_number}: "
                f"{holdout_file}"
            )

        stages.append(
            {
                "number": stage_number,
                "train_file": train_file,
                "holdout_file": holdout_file,
            }
        )

    stages.sort(key=lambda stage: stage["number"])

    if not stages:
        raise FileNotFoundError(
            f"No stage files were found in {stage_directory}"
        )

    return stages


# ---------------------------------------------------------------------
# 6. BUILD THE THREE LOADERS FOR ONE STAGE
# ---------------------------------------------------------------------

def build_stage_loaders(
    train_csv,
    holdout_csv,
    image_directory,
    batch_size=128,
    image_size=256,
    num_workers=4,
    pin_memory=True,
):
    """
    Build the training, holdout, and Fisher loaders for one stage.

    train_loader:
        Balanced NF and FL data.
        FL samples receive random augmentation.

    holdout_loader:
        Original untouched holdout images.
        No augmentation and no balancing.

    fisher_loader:
        Every original training image exactly once.
        No augmentation and no balancing.
    """

    basic_transform = BasicTransform(
        image_size=image_size
    )

    flare_transform = FlareTrainingTransform(
        image_size=image_size,
        rotation_degrees=5,
    )

    # ---------------------------------------------------------------
    # Training datasets
    # ---------------------------------------------------------------

    nf_training_data = MagnetogramDataset(
        csv_file=train_csv,
        image_directory=image_directory,
        transform=basic_transform,
        class_value=0,
    )

    fl_training_data = MagnetogramDataset(
        csv_file=train_csv,
        image_directory=image_directory,
        transform=flare_transform,
        class_value=1,
    )

    # Give both classes the same number of samples per training epoch.
    balanced_length = max(
        len(nf_training_data),
        len(fl_training_data),
    )

    balanced_nf_data = RepeatedDataset(
        dataset=nf_training_data,
        length=balanced_length,
    )

    balanced_fl_data = RepeatedDataset(
        dataset=fl_training_data,
        length=balanced_length,
    )

    balanced_training_data = ConcatDataset(
        [
            balanced_nf_data,
            balanced_fl_data,
        ]
    )

    # ---------------------------------------------------------------
    # Untouched holdout dataset
    # ---------------------------------------------------------------

    holdout_data = MagnetogramDataset(
        csv_file=holdout_csv,
        image_directory=image_directory,
        transform=basic_transform,
        class_value=None,
    )

    # ---------------------------------------------------------------
    # Complete original training data for Fisher information
    # ---------------------------------------------------------------

    fisher_data = MagnetogramDataset(
        csv_file=train_csv,
        image_directory=image_directory,
        transform=basic_transform,
        class_value=None,
    )

    # ---------------------------------------------------------------
    # DataLoader settings
    # ---------------------------------------------------------------

    loader_settings = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    # persistent_workers cannot be enabled when num_workers is zero.
    if num_workers > 0:
        loader_settings["persistent_workers"] = True

    train_loader = DataLoader(
        balanced_training_data,
        shuffle=True,
        **loader_settings,
    )

    holdout_loader = DataLoader(
        holdout_data,
        shuffle=False,
        **loader_settings,
    )

    fisher_loader = DataLoader(
        fisher_data,
        shuffle=False,
        **loader_settings,
    )

    return {
        "train": train_loader,
        "holdout": holdout_loader,
        "fisher": fisher_loader,
    }