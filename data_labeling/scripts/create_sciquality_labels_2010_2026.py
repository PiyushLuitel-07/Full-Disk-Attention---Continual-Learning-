"""Create 24-hour flare labels from the NCEI SciQuality catalogue."""

import csv
import re
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
CATALOGUE = ROOT / "data_labeling" / "Catalogue" / "sciquality_events.csv"
OUTPUT_FILE = (
    ROOT
    / "data_labeling"
    / "data_labels"
    / "sciquality"
    / "labels_sciquality_2010_2026.csv"
)
START = datetime(2010, 12, 1)
END = datetime(2026, 7, 31, 23, 59, 59)


def image_time(path):
    """Read an observation time from an HMI JPG filename."""
    return datetime.strptime(path.name, "HMI.m%Y.%m.%d_%H.%M.%S.jpg")


def flare_strength(goes_class):
    """Return a sortable GOES class value, for example M2.0 > C9.0."""
    rank = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
    return rank[goes_class[0]], float(goes_class[1:])


def flare_location(location):
    """Convert a location such as N15W20 to signed longitude and latitude."""
    match = re.fullmatch(r"([NS])(\d+)([EW])(\d+)", location or "")
    if not match:
        return "UNK", "UNK"

    north_south, latitude, east_west, longitude = match.groups()
    latitude = int(latitude) * (1 if north_south == "N" else -1)
    longitude = int(longitude) * (1 if east_west == "W" else -1)
    return longitude, latitude


def main():
    # Collect every non-empty hourly image in the requested date range.
    images = sorted(
        (timestamp, path.relative_to(IMAGE_DIR).as_posix())
        for year in range(2010, 2027)
        for path in (IMAGE_DIR / str(year)).rglob("*.jpg")
        if path.stat().st_size > 0
        and START <= (timestamp := image_time(path)) <= END
    )

    with CATALOGUE.open(newline="", encoding="utf-8-sig") as file:
        catalogue_rows = list(csv.DictReader(file))

    # Events without a usable start time or class cannot label an image.
    events = []
    for event in catalogue_rows:
        if event["Start"] in {"", "N/A"} or event["Class"] in {"", "N/A"}:
            continue
        event["time"] = datetime.strptime(event["Start"], "%Y-%m-%d %H:%M:%S")
        event["fl_lon"], event["fl_lat"] = flare_location(event["Location"])
        events.append(event)

    events.sort(key=lambda event: event["time"])
    event_times = [event["time"] for event in events]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "label",
        "goes_class",
        "fl_lon",
        "fl_lat",
        "rest_fl",
        "rest_lon",
        "rest_lat",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for timestamp, label in images:
            # Select events strictly after the image and before t + 24 hours.
            first = bisect_right(event_times, timestamp)
            last = bisect_left(event_times, timestamp + timedelta(hours=24))
            window = sorted(
                events[first:last],
                key=lambda event: flare_strength(event["Class"]),
                reverse=True,
            )

            if not window:
                writer.writerow({
                    "label": label,
                    "goes_class": "NF",
                    "fl_lon": "unk",
                    "fl_lat": "unk",
                    "rest_fl": "",
                    "rest_lon": "",
                    "rest_lat": "",
                })
                continue

            strongest, rest = window[0], window[1:]
            writer.writerow({
                "label": label,
                "goes_class": strongest["Class"],
                "fl_lon": strongest["fl_lon"],
                "fl_lat": strongest["fl_lat"],
                "rest_fl": [event["Class"] for event in rest] if rest else "",
                "rest_lon": [event["fl_lon"] for event in rest] if rest else "",
                "rest_lat": [event["fl_lat"] for event in rest] if rest else "",
            })

    skipped = len(catalogue_rows) - len(events)
    print(f"Created: {OUTPUT_FILE}")
    print(f"Images labelled: {len(images):,}")
    print(f"Catalogue events used: {len(events):,}")
    print(f"Catalogue events skipped: {skipped:,}")


if __name__ == "__main__":
    main()
