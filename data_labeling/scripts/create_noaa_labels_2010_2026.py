"""Create 24-hour flare labels from the NOAA/SWPC event catalogue."""

import csv
import re
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
CATALOGUE = ROOT / "data_labeling" / "Catalogue" / "noaa_events.csv"
OUTPUT_FILE = (
    ROOT
    / "data_labeling"
    / "data_labels"
    / "noaa"
    / "labels_noaa_2010_2026.csv"
)
START = datetime(2010, 12, 1)
END = datetime(2026, 7, 31, 23, 59, 59)


def image_time(path):
    """Read the observation time from an HMI JPG filename."""
    return datetime.strptime(path.name, "HMI.m%Y.%m.%d_%H.%M.%S.jpg")


def flare_strength(goes_class):
    """Return a sortable value so, for example, M1.0 ranks above C9.9."""
    rank = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
    return rank[goes_class[0]], float(goes_class[1:])


def flare_location(location):
    """Convert N15W20 into signed (longitude, latitude) values."""
    match = re.fullmatch(r"([NS])(\d+)([EW])(\d+)", location or "")
    if not match:
        return "UNK", "UNK"

    north_south, latitude, east_west, longitude = match.groups()
    latitude = int(latitude) * (1 if north_south == "N" else -1)
    longitude = int(longitude) * (1 if east_west == "W" else -1)
    return longitude, latitude


def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
    if not CATALOGUE.is_file():
        raise FileNotFoundError(f"NOAA catalogue not found: {CATALOGUE}")

    # Use only real, non-empty images instead of generating expected timestamps.
    images = sorted(
        (timestamp, path.relative_to(IMAGE_DIR).as_posix())
        for path in IMAGE_DIR.rglob("HMI.m*.jpg")
        if path.stat().st_size > 0
        and START <= (timestamp := image_time(path)) <= END
    )

    with CATALOGUE.open(newline="", encoding="utf-8-sig") as file:
        catalogue_rows = list(csv.DictReader(file))

    required_columns = {"Begin", "GOES class", "Location"}
    if not catalogue_rows:
        raise ValueError(f"NOAA catalogue is empty: {CATALOGUE}")
    missing = required_columns - set(catalogue_rows[0])
    if missing:
        raise ValueError(f"NOAA catalogue is missing columns: {sorted(missing)}")

    # Rows without a usable beginning time or GOES class cannot label an image.
    events = []
    for row in catalogue_rows:
        begin = row["Begin"].strip()
        goes_class = row["GOES class"].strip().upper()
        if begin in {"", "N/A"} or goes_class in {"", "N/A"}:
            continue

        try:
            event_time = datetime.strptime(begin, "%Y-%m-%d %H:%M:%S")
            flare_strength(goes_class)  # Validate the GOES class.
        except (ValueError, KeyError):
            continue

        longitude, latitude = flare_location(row["Location"].strip().upper())
        events.append(
            {
                "time": event_time,
                "goes_class": goes_class,
                "fl_lon": longitude,
                "fl_lat": latitude,
            }
        )

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
            # Find events strictly after the image and before image time + 24 h.
            first = bisect_right(event_times, timestamp)
            last = bisect_left(event_times, timestamp + timedelta(hours=24))
            window = sorted(
                events[first:last],
                key=lambda event: flare_strength(event["goes_class"]),
                reverse=True,
            )

            if not window:
                writer.writerow(
                    {
                        "label": label,
                        "goes_class": "NF",
                        "fl_lon": "unk",
                        "fl_lat": "unk",
                        "rest_fl": "",
                        "rest_lon": "",
                        "rest_lat": "",
                    }
                )
                continue

            # Store the strongest event separately and preserve any other events.
            strongest, rest = window[0], window[1:]
            writer.writerow(
                {
                    "label": label,
                    "goes_class": strongest["goes_class"],
                    "fl_lon": strongest["fl_lon"],
                    "fl_lat": strongest["fl_lat"],
                    "rest_fl": [event["goes_class"] for event in rest] if rest else "",
                    "rest_lon": [event["fl_lon"] for event in rest] if rest else "",
                    "rest_lat": [event["fl_lat"] for event in rest] if rest else "",
                }
            )

    print(f"Created: {OUTPUT_FILE}")
    print(f"Images labelled: {len(images):,}")
    print(f"Catalogue events used: {len(events):,}")
    print(f"Catalogue events skipped: {len(catalogue_rows) - len(events):,}")


if __name__ == "__main__":
    main()
