"""Convert detailed NOAA GOES labels into binary NF/FL labels."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    ROOT
    / "data_labeling"
    / "data_labels"
    / "noaa"
    / "labels_noaa_2010_2026.csv"
)
OUTPUT_FILE = INPUT_FILE.with_name("labels_noaa_2010_2026_binary.csv")


def binary_label(goes_class):
    """Map M/X events to FL (1); map A/B/C/NF to NF (0)."""
    goes_class = goes_class.strip().upper()
    if goes_class == "NF":
        return 0
    if goes_class[:1] in {"A", "B", "C"}:
        return 0
    if goes_class[:1] in {"M", "X"}:
        return 1
    raise ValueError(f"Unknown GOES class: {goes_class!r}")


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Detailed label file not found: {INPUT_FILE}")

    with INPUT_FILE.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"label", "goes_class"} <= set(reader.fieldnames):
            raise ValueError("Input file must contain label and goes_class columns.")

        rows = []
        seen_paths = set()
        duplicate_paths = 0
        counts = {0: 0, 1: 0}

        for row in reader:
            image_path = row["label"].strip()
            if not image_path:
                raise ValueError("Found an empty image path.")

            label = binary_label(row["goes_class"])
            rows.append({"label": image_path, "goes_class": label})
            counts[label] += 1

            # Count duplicate paths so the generated dataset can be audited.
            if image_path in seen_paths:
                duplicate_paths += 1
            seen_paths.add(image_path)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=["label", "goes_class"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {OUTPUT_FILE}")
    print(f"Images: {len(rows):,} | NF: {counts[0]:,} | FL: {counts[1]:,}")
    print(f"Duplicate image paths: {duplicate_paths:,}")


if __name__ == "__main__":
    main()
