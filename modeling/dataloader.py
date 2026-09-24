"""Datasets and DataLoaders for continual solar-flare learning."""

import random
import re
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset


# ---------------------------------------------------------------------
# 1. BASIC IMAGE TRANSFORMATION
# ---------------------------------------------------------------------

class BasicTransform:
    """
    Resize a magnetogram and convert it to a tensor.

    Used for:
        - original NF training images,
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
# 2. DISK-ONLY POLARITY INVERSION
# ---------------------------------------------------------------------

class DiskPolarityInversion:
    """
    Reverse magnetic polarity only inside the solar disk.

    The black background outside the Sun remains unchanged.
    """

    def __init__(
        self,
        center_x_ratio=0.50,
        center_y_ratio=0.50,
        radius_ratio=0.40,
    ):
        self.center_x_ratio = center_x_ratio
        self.center_y_ratio = center_y_ratio
        self.radius_ratio = radius_ratio

    def __call__(self, image):
        """
        Parameters
        ----------
        image:
            Tensor with shape [channels, height, width].
            Pixel values must be between 0 and 1.
        """

        if image.ndim != 3:
            raise ValueError(
                "Polarity inversion expects a tensor with shape "
                "[channels, height, width]."
            )

        _, height, width = image.shape

        center_x = self.center_x_ratio * (width - 1)
        center_y = self.center_y_ratio * (height - 1)

        radius = self.radius_ratio * min(
            height,
            width,
        )

        y_coordinates, x_coordinates = torch.meshgrid(
            torch.arange(
                height,
                device=image.device,
            ),
            torch.arange(
                width,
                device=image.device,
            ),
            indexing="ij",
        )

        disk_mask = (
            (x_coordinates - center_x) ** 2
            + (y_coordinates - center_y) ** 2
            <= radius ** 2
        )

        polarity_image = image.clone()

        # Reverse polarity only inside the solar disk.
        polarity_image[:, disk_mask] = (
            1.0 - polarity_image[:, disk_mask]
        )

        # Pixels outside the mask remain unchanged.
        return polarity_image


# ---------------------------------------------------------------------
# 3. FLARE-IMAGE TRAINING TRANSFORMATION
# ---------------------------------------------------------------------

class FlareTrainingTransform:
    """
    Apply one specified augmentation to an FL training image.

    Available choices:
        1. Original image
        2. Horizontal flip
        3. Vertical flip
        4. Small rotation
        5. Disk-only polarity inversion

    This transformation is not used for holdout or Fisher data.
    """

    def __init__(
        self,
        augmentation,
        image_size=256,
        rotation_degrees=5,
    ):
        available_augmentations = {
            "original",
            "horizontal_flip",
            "vertical_flip",
            "rotation",
            "polarity",
        }

        if augmentation not in available_augmentations:
            raise ValueError(
                f"Unknown augmentation: {augmentation}"
            )

        self.augmentation = augmentation
        self.image_size = image_size
        self.rotation_degrees = rotation_degrees

        self.polarity_inversion = DiskPolarityInversion(
            center_x_ratio=0.50,
            center_y_ratio=0.50,
            radius_ratio=0.40,
        )

    def __call__(self, image):
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
        )

        if self.augmentation == "horizontal_flip":
            image = TF.hflip(image)

        elif self.augmentation == "vertical_flip":
            image = TF.vflip(image)

        elif self.augmentation == "rotation":
            angle = random.uniform(
                -self.rotation_degrees,
                self.rotation_degrees,
            )

            image = TF.rotate(
                image,
                angle,
            )

        # Convert the PIL image to a tensor with values from 0 to 1.
        image = TF.to_tensor(image)

        if self.augmentation == "polarity":
            image = self.polarity_inversion(image)

        return image


# ---------------------------------------------------------------------
# 4. MAGNETOGRAM DATASET
# ---------------------------------------------------------------------

class MagnetogramDataset(Dataset):
    """
    Load magnetogram paths and binary labels from a CSV file.

    Required CSV columns
    --------------------
    label:
        Relative path to the JPG image.

    goes_class:
        0 for NF and 1 for FL.

    class_value
    -----------
    None:
        Load both classes.

    0:
        Load only NF images.

    1:
        Load only FL images.
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

        annotations = pd.read_csv(
            self.csv_file
        )

        required_columns = {
            "label",
            "goes_class",
        }

        missing_columns = (
            required_columns
            - set(annotations.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{self.csv_file} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if annotations["label"].isna().any():
            raise ValueError(
                f"{self.csv_file} contains missing image paths."
            )

        if not annotations["goes_class"].isin([0, 1]).all():
            raise ValueError(
                f"{self.csv_file} contains labels other than 0 and 1."
            )

        if class_value is not None:
            if class_value not in (0, 1):
                raise ValueError(
                    "class_value must be None, 0, or 1."
                )

            annotations = annotations[
                annotations["goes_class"] == class_value
            ]

        self.annotations = annotations.reset_index(
            drop=True
        )

        if len(self.annotations) == 0:
            raise ValueError(
                f"No samples were selected from {self.csv_file}."
            )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]

        image_path = (
            self.image_directory
            / row["label"]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Magnetogram was not found: {image_path}"
            )

        # Read the image and force it to one-channel grayscale.
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
# 5. REPEAT A DATASET FOR CLASS BALANCING
# ---------------------------------------------------------------------

class RepeatedDataset(Dataset):
    """
    Repeat a smaller dataset until it reaches the requested length.

    No new image files are created on disk.
    """

    def __init__(
        self,
        dataset,
        length,
    ):
        if len(dataset) == 0:
            raise ValueError(
                "Cannot repeat an empty dataset."
            )

        if length <= 0:
            raise ValueError(
                "Repeated dataset length must be positive."
            )

        self.dataset = dataset
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        actual_index = (
            index % len(self.dataset)
        )

        return self.dataset[actual_index]


class ControlledFlareDataset(Dataset):
    """
    Produce a balanced number of FL samples using controlled views.

    Every FL sample is selected cyclically. Its augmentation changes
    predictably across repetitions so that original, horizontal flip,
    vertical flip, rotation, and polarity views are all represented.
    """

    def __init__(self, flare_views, length):
        if not flare_views:
            raise ValueError("At least one FL view is required.")

        view_lengths = {len(view) for view in flare_views}

        if len(view_lengths) != 1:
            raise ValueError("All FL views must have the same length.")

        self.flare_views = flare_views
        self.original_flare_count = len(flare_views[0])
        self.number_of_views = len(flare_views)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample_index = index % self.original_flare_count
        repetition_number = index // self.original_flare_count

        # Shift the augmentation used for an image each time the FL
        # dataset repeats. This distributes all augmentation types
        # predictably without choosing one randomly.
        view_index = (
            sample_index + repetition_number
        ) % self.number_of_views

        return self.flare_views[view_index][sample_index]


# ---------------------------------------------------------------------
# 6. AUTOMATIC STAGE DISCOVERY
# ---------------------------------------------------------------------

def discover_stages(stage_directory):
    """
    Find every StageN_train.csv and StageN_holdout.csv pair.

    Adding a future stage requires only:

        Stage4_train.csv
        Stage4_holdout.csv
    """

    stage_directory = Path(
        stage_directory
    )

    if not stage_directory.is_dir():
        raise FileNotFoundError(
            f"Stage directory was not found: {stage_directory}"
        )

    stage_pattern = re.compile(
        r"Stage(\d+)_train\.csv$"
    )

    stages = []

    for train_file in stage_directory.glob(
        "Stage*_train.csv"
    ):
        match = stage_pattern.fullmatch(
            train_file.name
        )

        if match is None:
            continue

        stage_number = int(
            match.group(1)
        )

        holdout_file = (
            stage_directory
            / f"Stage{stage_number}_holdout.csv"
        )

        if not holdout_file.is_file():
            raise FileNotFoundError(
                f"Missing holdout file for Stage "
                f"{stage_number}: {holdout_file}"
            )

        stages.append(
            {
                "number": stage_number,
                "train_file": train_file,
                "holdout_file": holdout_file,
            }
        )

    stages.sort(
        key=lambda stage: stage["number"]
    )

    if not stages:
        raise FileNotFoundError(
            f"No stage files were found in "
            f"{stage_directory}"
        )

    return stages


# ---------------------------------------------------------------------
# 7. BUILD THE LOADERS FOR ONE STAGE
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
    Build three DataLoaders for one continual-learning stage.

    train:
        Balanced NF and FL samples.
        FL samples use controlled augmentation views.

    holdout:
        Untouched stage holdout.
        No augmentation or balancing.

    fisher:
        Complete original stage training data.
        No augmentation, repetition, or balancing.
    """

    basic_transform = BasicTransform(
        image_size=image_size
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

    augmentation_names = [
        "original",
        "horizontal_flip",
        "vertical_flip",
        "rotation",
        "polarity",
    ]

    flare_views = []

    for augmentation_name in augmentation_names:
        flare_transform = FlareTrainingTransform(
            augmentation=augmentation_name,
            image_size=image_size,
            rotation_degrees=5,
        )

        flare_views.append(
            MagnetogramDataset(
                csv_file=train_csv,
                image_directory=image_directory,
                transform=flare_transform,
                class_value=1,
            )
        )

    # Both classes will have this length during a training epoch.
    balanced_length = max(
        len(nf_training_data),
        len(flare_views[0]),
    )

    balanced_nf_data = RepeatedDataset(
        dataset=nf_training_data,
        length=balanced_length,
    )

    balanced_fl_data = ControlledFlareDataset(
        flare_views=flare_views,
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


# ---------------------------------------------------------------------
# 8. BUILD A PLAIN EVALUATION LOADER
# ---------------------------------------------------------------------

def build_evaluation_loader(
    csv_file,
    image_directory,
    batch_size=128,
    image_size=256,
    num_workers=4,
    pin_memory=True,
):
    """
    Build an untouched DataLoader for validation or testing.

    Both classes retain their natural distribution. Images are resized
    and converted to tensors, without augmentation or repetition.
    """

    evaluation_data = MagnetogramDataset(
        csv_file=csv_file,
        image_directory=image_directory,
        transform=BasicTransform(image_size=image_size),
        class_value=None,
    )

    loader_settings = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_settings["persistent_workers"] = True

    return DataLoader(
        evaluation_data,
        shuffle=False,
        **loader_settings,
    )
