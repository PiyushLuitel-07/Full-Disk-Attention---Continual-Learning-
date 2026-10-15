"""Convert detailed SciQuality labels into binary NF/FL training labels."""

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling" / "data_labels" / "sciquality"
INPUT_FILE = LABEL_DIR / "labels_sciquality_2010_2026.csv"
OUTPUT_FILE = LABEL_DIR / "labels_sciquality_2010_2026_binary.csv"


def binary_label(goes_class):
    """Return 1 for M1.0-or-stronger flares; otherwise return 0."""
    if goes_class.startswith("X"):
        return 1
    if goes_class.startswith("M") and float(goes_class[1:]) >= 1.0:
        return 1
    return 0


def main():
    with INPUT_FILE.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    # Keep the same columns used by the existing training pipeline.
    output_rows = [
        {"label": row["label"], "goes_class": binary_label(row["goes_class"])}
        for row in rows
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=["label", "goes_class"])
        writer.writeheader()
        writer.writerows(output_rows)

    counts = Counter(row["goes_class"] for row in output_rows)
    duplicates = len(output_rows) - len({row["label"] for row in output_rows})
    print(f"Created: {OUTPUT_FILE}")
    print(f"Images: {len(output_rows):,} | NF: {counts[0]:,} | FL: {counts[1]:,}")
    print(f"Duplicate image paths: {duplicates:,}")


if __name__ == "__main__":
    main()
