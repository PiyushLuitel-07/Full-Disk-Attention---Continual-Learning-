"""Create an audited 2010-2018 image/label manifest.

This script does not modify images or the repository's original label CSV.
It keeps only exact filename matches between the two sources.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IMAGE_DIRECTORY = PROJECT_ROOT / "downloaded_data" / "hmi_jpgs"
DEFAULT_LABEL_FILE = (
    PROJECT_ROOT
    / "data_labeling"
    / "data_labels"
    / "NEW_full_dataset_cleaned_1_hours_with_loc_and_time_new.csv"
)
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data_labeling"
    / "data_labels"
    / "continual_stages"
)

START_TIME = datetime(2010, 12, 1, 0, 0, 0)
END_TIME = datetime(2018, 12, 30, 23, 0, 0)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Match downloaded 2010-2018 JPGs to existing labels."
    )
    parser.add_argument(
        "--image-directory",
        type=Path,
        default=DEFAULT_IMAGE_DIRECTORY,
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=DEFAULT_LABEL_FILE,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    return parser.parse_args()


def timestamp_from_label(label):
    """Read a timestamp from YYYY/MM/DD/HMI.mYYYY.MM.DD_HH.MM.SS.jpg."""
    filename = Path(label).name
    return datetime.strptime(filename, "HMI.m%Y.%m.%d_%H.%M.%S.jpg")


def scan_images(image_directory):
    """Return nonempty JPG paths from the required time period."""
    images = {}

    for year in range(2010, 2019):
        year_directory = image_directory / str(year)

        if not year_directory.exists():
            continue

        for image_path in year_directory.rglob("*.jpg"):
            if image_path.stat().st_size == 0:
                continue

            relative_path = image_path.relative_to(image_directory).as_posix()
            timestamp = timestamp_from_label(relative_path)

            if START_TIME <= timestamp <= END_TIME:
                if relative_path in images:
                    raise ValueError(f"Duplicate downloaded image: {relative_path}")

                images[relative_path] = image_path.stat().st_size

    return images


def read_labels(label_file):
    """Load the original labels and reject duplicate label paths."""
    labels = {}

    with label_file.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required_columns = {"label", "goes_class"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{label_file} must contain label and goes_class columns"
            )

        for row in reader:
            label = row["label"].strip().replace("\\", "/")
            timestamp = timestamp_from_label(label)

            if not (START_TIME <= timestamp <= END_TIME):
                continue

            if label in labels:
                raise ValueError(f"Duplicate label row: {label}")

            original_class = row["goes_class"].strip().upper()
            if not original_class:
                raise ValueError(f"Empty GOES class for {label}")

            row["label"] = label
            row["original_goes_class"] = original_class
            row["goes_class"] = (
                1 if original_class.startswith(("M", "X")) else 0
            )
            labels[label] = row

    return labels


def write_lines(path, values):
    with path.open("w", encoding="utf-8") as file:
        for value in values:
            file.write(f"{value}\n")


def main():
    arguments = parse_arguments()

    image_directory = arguments.image_directory.resolve()
    label_file = arguments.label_file.resolve()
    output_directory = arguments.output_directory.resolve()

    if not image_directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_directory}")

    if not label_file.is_file():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    output_directory.mkdir(parents=True, exist_ok=True)

    images = scan_images(image_directory)
    labels = read_labels(label_file)

    matched_paths = sorted(set(images) & set(labels))
    missing_images = sorted(set(labels) - set(images))
    images_without_labels = sorted(set(images) - set(labels))

    if not matched_paths:
        raise RuntimeError("No image filenames matched the label CSV")

    master_path = output_directory / "master_2010_2018_matched.csv"
    missing_path = output_directory / "labels_without_images.txt"
    unlabelled_path = output_directory / "images_without_labels.txt"
    audit_path = output_directory / "master_2010_2018_audit.json"

    original_columns = list(next(iter(labels.values())).keys())
    original_columns.remove("original_goes_class")
    output_columns = [
        "label",
        "goes_class",
        "original_goes_class",
        "file_size_bytes",
    ]
    output_columns.extend(
        column
        for column in original_columns
        if column not in {"label", "goes_class"}
    )

    flare_count = 0

    with master_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()

        for label in matched_paths:
            row = labels[label].copy()
            row["file_size_bytes"] = images[label]
            flare_count += int(row["goes_class"])
            writer.writerow({column: row.get(column, "") for column in output_columns})

    write_lines(missing_path, missing_images)
    write_lines(unlabelled_path, images_without_labels)

    audit = {
        "period_start_utc": START_TIME.isoformat(sep=" "),
        "period_end_utc": END_TIME.isoformat(sep=" "),
        "downloaded_nonempty_jpgs": len(images),
        "label_rows": len(labels),
        "matched_rows": len(matched_paths),
        "flare_rows": flare_count,
        "nonflare_rows": len(matched_paths) - flare_count,
        "labels_without_images": len(missing_images),
        "images_without_labels": len(images_without_labels),
        "first_matched_image": matched_paths[0],
        "last_matched_image": matched_paths[-1],
    }

    with audit_path.open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)
        file.write("\n")

    print(json.dumps(audit, indent=2))
    print(f"\nMaster CSV: {master_path}")
    print(f"Audit report: {audit_path}")
    print(f"Missing-image list: {missing_path}")
    print(f"Unlabelled-image list: {unlabelled_path}")


if __name__ == "__main__":
    main()
