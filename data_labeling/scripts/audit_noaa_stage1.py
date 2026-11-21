"""Audit NOAA Stage 1 labels, temporal splits, and effective sample counts."""

import argparse
import csv
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling/data_labels/noaa"
CATALOGUE = ROOT / "data_labeling/Catalogue/noaa_events.csv"
DETAILED = LABEL_DIR / "labels_noaa_2010_2026.csv"
BINARY = LABEL_DIR / "labels_noaa_2010_2026_binary.csv"
SPLITS = {
    "stage1_train": LABEL_DIR / "CL_Labels/Stage1_train.csv",
    "stage1_holdout": LABEL_DIR / "CL_Labels/Stage1_holdout.csv",
    "validation": LABEL_DIR / "CL_Labels/validation_2024_2025.csv",
    "test": LABEL_DIR / "CL_Labels/test_2025_2026_july.csv",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def image_time(label):
    name = Path(label).name
    return datetime.strptime(name, "HMI.m%Y.%m.%d_%H.%M.%S.jpg")


def flare_strength(goes_class):
    goes_class = goes_class.strip().upper()
    if goes_class == "NF":
        return -1, 0.0
    return {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}[goes_class[0]], float(goes_class[1:])


def binary_label(goes_class):
    return int(goes_class.strip().upper()[:1] in {"M", "X"})


def load_events():
    raw = read_csv(CATALOGUE)
    events, invalid = [], 0
    for row in raw:
        try:
            time = datetime.strptime(row["Begin"].strip(), "%Y-%m-%d %H:%M:%S")
            flare_strength(row["GOES class"])
        except (KeyError, ValueError):
            invalid += 1
            continue
        events.append((time, row["GOES class"].strip().upper()))
    events.sort()
    return raw, events, invalid


def episode_count(rows):
    times = sorted(image_time(row["label"]) for row in rows if int(row["goes_class"]) == 1)
    if not times:
        return 0
    return 1 + sum(current - previous > timedelta(hours=1, minutes=1)
                   for previous, current in zip(times, times[1:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, help="Optional server JPG root for existence checks")
    args = parser.parse_args()

    detailed_rows = read_csv(DETAILED)
    binary_rows = read_csv(BINARY)
    detailed = {row["label"]: row for row in detailed_rows}
    binary = {row["label"]: int(row["goes_class"]) for row in binary_rows}
    if len(detailed) != len(detailed_rows) or len(binary) != len(binary_rows):
        raise AssertionError("Duplicate image paths exist in a master label file")
    if detailed.keys() != binary.keys():
        raise AssertionError("Detailed and binary master label paths differ")

    mapping_errors = [label for label, row in detailed.items()
                      if binary_label(row["goes_class"]) != binary[label]]
    raw_catalogue, events, invalid_events = load_events()
    event_times = [event[0] for event in events]
    recompute_errors = []
    for label, row in detailed.items():
        time = image_time(label)
        first = bisect_right(event_times, time)
        last = bisect_left(event_times, time + timedelta(hours=24))
        expected = max((event[1] for event in events[first:last]),
                       key=flare_strength, default="NF")
        if expected != row["goes_class"].strip().upper():
            recompute_errors.append((label, row["goes_class"], expected))

    print("NOAA STAGE 1 LABEL AUDIT")
    print(f"Master images: {len(binary_rows):,}")
    print(f"Detailed-to-binary mapping errors: {len(mapping_errors):,}")
    print(f"24-hour label recomputation errors: {len(recompute_errors):,}")
    print(f"Catalogue rows: {len(raw_catalogue):,} | usable: {len(events):,} | invalid: {invalid_events:,}")
    print(f"Catalogue flags: {dict(Counter((row.get('Flag') or 'MISSING').strip() or 'MISSING' for row in raw_catalogue))}")

    previous_end = None
    for name, path in SPLITS.items():
        rows = read_csv(path)
        labels = [row["label"] for row in rows]
        duplicates = len(labels) - len(set(labels))
        missing = [label for label in labels if label not in binary]
        mismatches = [row for row in rows if row["label"] in binary
                      and int(row["goes_class"]) != binary[row["label"]]]
        if missing or mismatches:
            raise AssertionError(f"{name}: missing={len(missing)}, mismatched={len(mismatches)}")
        times = [image_time(label) for label in labels]
        fl = sum(int(row["goes_class"]) for row in rows)
        detailed_split = [detailed[label]["goes_class"].strip().upper() for label in labels]
        hard_c = sum(cls.startswith("C") and 4.0 <= flare_strength(cls)[1] < 10.0
                     for cls in detailed_split)
        negatives = len(rows) - fl
        gap = "N/A" if previous_end is None else str(min(times) - previous_end)
        print(f"\n{name}: {len(rows):,} images | {min(times)} to {max(times)} | prior gap={gap}")
        print(f"  NF={negatives:,} | FL={fl:,} ({fl / len(rows):.2%}) | duplicates={duplicates:,}")
        print(f"  Positive hourly episodes={episode_count(rows):,}")
        print(f"  C4.0-C9.9 hard negatives={hard_c:,} ({hard_c / max(negatives, 1):.2%} of negatives)")
        yearly = Counter((image_time(row["label"]).year, int(row["goes_class"])) for row in rows)
        print("  Yearly FL rate: " + ", ".join(
            f"{year}={yearly[year, 1] / (yearly[year, 0] + yearly[year, 1]):.1%}"
            for year in sorted({year for year, _ in yearly})))
        if args.image_root:
            absent = sum(not (args.image_root / label).is_file() for label in labels)
            empty = sum((args.image_root / label).is_file() and (args.image_root / label).stat().st_size == 0
                        for label in labels)
            print(f"  Server image check: missing={absent:,} | empty={empty:,}")
        previous_end = max(times)

    if mapping_errors or recompute_errors:
        raise AssertionError("Label audit failed; inspect the reported errors")
    print("\nPASS: stored labels reproduce the current 24-hour M/X labeling rule.")


if __name__ == "__main__":
    main()
