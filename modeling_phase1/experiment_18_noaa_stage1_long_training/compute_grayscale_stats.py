"""Compute Stage 1 grayscale pixel mean/std without loading all images at once."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from config import IMAGE_DIR, TRAIN_CSV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8-sig") as file:
        paths = [args.image_dir / row["label"] for row in csv.DictReader(file)]

    count = 0
    total = 0.0
    total_squared = 0.0
    for index, path in enumerate(paths, start=1):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        count += pixels.size
        total += pixels.sum()
        total_squared += np.square(pixels).sum()
        if index % 1000 == 0:
            print(f"Processed {index:,}/{len(paths):,} images", flush=True)

    mean = total / count
    variance = max(total_squared / count - mean * mean, 0.0)
    print(f"images={len(paths)}")
    print(f"pixels={count}")
    print(f"grayscale_mean={mean:.12f}")
    print(f"grayscale_std={variance ** 0.5:.12f}")


if __name__ == "__main__":
    main()
