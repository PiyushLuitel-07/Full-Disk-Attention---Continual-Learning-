"""Generate tiny synthetic images for a software-only Stage 1 -> EWC demo.

These images are deliberately simple and MUST NOT be used for scientific
results. Their only purpose is to exercise the real model, Fisher estimation,
checkpoint hand-off, and EWC penalty without downloading the HMI corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


def rows_from_manifests(directory: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    for manifest in sorted(directory.glob("stage*.csv")):
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label, target = row["label"], int(row["goes_class"])
                if label in rows and rows[label] != target:
                    raise ValueError(f"Conflicting synthetic target for {label}")
                rows[label] = target
    return rows


def make_image(label: str, target: int, size: int) -> Image.Image:
    seed = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    radius = size * 0.43
    disk = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= radius**2
    array = np.full((size, size), 127.0)
    array[disk] += rng.normal(0.0, 13.0, int(disk.sum()))

    # A label-1 image contains a visible bipolar active region. Stage 2 moves it
    # to the other side of the disk, creating a tiny temporal domain shift.
    year = int(label.split("/", 1)[0])
    if target == 1:
        center_x = int(size * (0.36 if year <= 2012 else 0.64))
        center_y = int(size * 0.48)
        sigma = size * 0.045
        positive = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2))
        negative = np.exp(
            -((xx - (center_x + size * 0.09)) ** 2 + (yy - center_y) ** 2)
            / (2 * sigma**2)
        )
        array += 105 * positive - 105 * negative

    array[~disk] = 0
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def main() -> None:
    args = parse_args()
    rows = rows_from_manifests(args.manifest_dir.expanduser().resolve())
    root = args.image_root.expanduser().resolve()
    for relative_path, target in rows.items():
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        make_image(relative_path, target, args.size).save(
            destination, format="JPEG", quality=92
        )
    print(f"Wrote {len(rows)} synthetic smoke-test images under {root}")
    print("WARNING: these are not HMI observations and cannot support a scientific claim.")


if __name__ == "__main__":
    main()
