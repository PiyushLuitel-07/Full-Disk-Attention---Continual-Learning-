"""CSV manifest and image-loading utilities.

The loader intentionally keeps preprocessing close to the original project:
grayscale image -> resize to 256 x 256 -> tensor in [0, 1]. Positive examples
can receive the same small rotations and flips used by the original trainer.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class Sample:
    relative_path: str
    target: int


def read_manifest(path: str | Path) -> list[Sample]:
    """Read the original two-column ``label,goes_class`` CSV contract."""

    manifest = Path(path)
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "goes_class"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{manifest} must contain columns {sorted(required)}")
        samples = []
        for row in reader:
            relative_path = row["label"].strip()
            parsed_path = Path(relative_path)
            if parsed_path.is_absolute() or ".." in parsed_path.parts:
                raise ValueError(f"Manifest paths must stay below image_root: {relative_path}")
            samples.append(Sample(relative_path, int(row["goes_class"])))

    if not samples:
        raise ValueError(f"Manifest is empty: {manifest}")
    invalid = [sample.target for sample in samples if sample.target not in (0, 1)]
    if invalid:
        raise ValueError(f"Targets must be 0 or 1; found {sorted(set(invalid))}")
    return samples


class MagnetogramDataset(Dataset):
    """Load one-channel HMI JPEGs from paths named by a manifest."""

    def __init__(
        self,
        samples: Iterable[Sample],
        image_root: str | Path,
        image_size: int = 256,
        augment_positive: bool = False,
        check_files: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.image_root = Path(image_root).expanduser().resolve()
        self.image_size = image_size
        self.augment_positive = augment_positive

        if image_size != 256:
            raise ValueError("Scientific parity requires image_size=256")
        if check_files:
            missing = [
                sample.relative_path
                for sample in self.samples
                if not (self.image_root / sample.relative_path).is_file()
            ]
            if missing:
                preview = "\n  ".join(missing[:5])
                raise FileNotFoundError(
                    f"{len(missing)} images referenced by the manifest are missing "
                    f"under {self.image_root}. First paths:\n  {preview}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]
        path = self.image_root / sample.relative_path
        with Image.open(path) as opened:
            image = opened.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )

        # Only FL examples are augmented, matching the original training idea.
        if self.augment_positive and sample.target == 1:
            operation = random.randrange(4)
            if operation == 1:
                image = TF.hflip(image)
            elif operation == 2:
                image = TF.vflip(image)
            elif operation == 3:
                image = TF.rotate(image, random.uniform(-5.0, 5.0))

        tensor = TF.to_tensor(image)
        target = torch.tensor(sample.target, dtype=torch.long)
        return tensor, target, sample.relative_path


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def make_train_loader(
    dataset: MagnetogramDataset,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Create a reproducible, class-balanced loader using weighted sampling."""

    counts = {
        label: sum(sample.target == label for sample in dataset.samples)
        for label in (0, 1)
    }
    if not all(counts.values()):
        raise ValueError(f"Training requires both classes; observed counts={counts}")

    weights = [1.0 / counts[sample.target] for sample in dataset.samples]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=2 * max(counts.values()),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def make_ordered_loader(
    dataset: MagnetogramDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
    )
