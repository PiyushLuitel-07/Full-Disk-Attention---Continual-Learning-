"""Create 24-hour flare labels from the NOAA/SWPC event catalogue.

For a magnetogram observed at time T, include flares whose onset satisfies:

    T < NOAA Begin < T + 24 hours

The NOAA ``End`` time is deliberately not used for forecasting. A flare that
began before T is already occurring, so it is not a future flare even if it
ends after T. Peak and End remain useful for later duration analyses.
"""

import csv
import math
import re
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
CATALOGUE = ROOT / "data_labeling" / "Catalogue" / "noaa_events.csv"
OUTPUT_FILE = ROOT / "data_labeling/data_labels/noaa/labels_noaa_2010_2026.csv"
START = datetime(2010, 12, 1)
END = datetime(2026, 7, 31, 23, 59, 59)
FORECAST_HOURS = 24

# GOES letter classes represent physical peak-flux scales in W/m^2.
GOES_SCALES = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
GOES_PATTERN = re.compile(r"([ABCMX])(\d+(?:\.\d+)?)")


def image_time(path):
    """Read the requested observation time stored in an HMI JPG filename."""
    return datetime.strptime(path.name, "HMI.m%Y.%m.%d_%H.%M.%S.jpg")


def goes_peak_flux(goes_class):
    """Convert a GOES class such as M1.2 to peak flux in W/m^2.

    This physical conversion avoids letter-only mistakes. For example:
    M0.9 = 9.0e-6, C9.9 = 9.9e-6, and M1.0 = 1.0e-5 W/m^2.
    Therefore M0.9 is below the >=M1.0 binary flare threshold.
    """
    value = goes_class.strip().upper()
    match = GOES_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid GOES class: {goes_class!r}")

    category, magnitude_text = match.groups()
    magnitude = float(magnitude_text)
    if not math.isfinite(magnitude) or magnitude < 0:
        raise ValueError(f"Invalid GOES class: {goes_class!r}")
    return magnitude * GOES_SCALES[category]


def flare_location(location):
    """Convert a location such as N15W20 to signed longitude and latitude."""
    match = re.fullmatch(r"([NS])(\d+)([EW])(\d+)", location or "")
    if not match:
        return "UNK", "UNK"

    north_south, latitude, east_west, longitude = match.groups()
    latitude = int(latitude) * (1 if north_south == "N" else -1)
    longitude = int(longitude) * (1 if east_west == "W" else -1)
    return longitude, latitude


def load_images():
    """Return only real, non-empty JPG files, ordered chronologically."""
    images = []
    for path in IMAGE_DIR.rglob("HMI.m*.jpg"):
        if path.stat().st_size == 0:
            continue
        timestamp = image_time(path)
        if START <= timestamp <= END:
            images.append((timestamp, path.relative_to(IMAGE_DIR).as_posix()))
    return sorted(images)


def load_events():
    """Read valid NOAA events and remove exact duplicate catalogue records."""
    with CATALOGUE.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    # Only Begin and GOES class are required to create a forecasting label.
    # Peak, End, location, and AR may be missing in a valid NOAA record.
    required = {"Begin", "GOES class"}
    if not rows:
        raise ValueError(f"NOAA catalogue is empty: {CATALOGUE}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"NOAA catalogue is missing columns: {sorted(missing)}")

    events, seen, skipped, duplicates = [], set(), 0, 0
    for row_number, row in enumerate(rows, start=2):
        begin_text = row["Begin"].strip()
        class_text = row["GOES class"].strip().upper()

        # An event without an onset or class cannot define a forecast target.
        if begin_text in {"", "N/A"} or class_text in {"", "N/A"}:
            skipped += 1
            continue

        try:
            begin = datetime.strptime(begin_text, "%Y-%m-%d %H:%M:%S")
            peak_flux = goes_peak_flux(class_text)
        except ValueError as error:
            raise ValueError(f"Invalid NOAA catalogue row {row_number}: {error}") from error

        # NOAA occasionally contains the same physical event more than once.
        event_key = tuple(
            (row.get(column) or "").strip().upper()
            for column in ("Begin", "Peak", "End", "GOES class", "Location", "AR")
        )
        if event_key in seen:
            duplicates += 1
            continue
        seen.add(event_key)

        longitude, latitude = flare_location(row["Location"].strip().upper())
        events.append(
            {
                "begin": begin,
                "goes_class": class_text,
                "peak_flux": peak_flux,
                "fl_lon": longitude,
                "fl_lat": latitude,
            }
        )

    events.sort(key=lambda event: event["begin"])
    return rows, events, skipped, duplicates


def create_label_rows(images, events):
    """Create labels with an explicit 24-hour forecast window.

    The event cursor moves forward once through the chronological catalogue.
    ``active_events`` then contains only events satisfying
    ``image_time < Begin < image_time + 24 hours``. This is faster than
    repeatedly scanning the full catalogue while keeping the rule readable.
    """
    output_rows = []
    active_events = []
    next_event = 0

    for timestamp, image_label in images:
        window_end = timestamp + timedelta(hours=FORECAST_HOURS)

        # Match the professor's pwe=T+23:59:59 rule: an event exactly at
        # T+24 h belongs to the following window, not this one.
        while next_event < len(events) and events[next_event]["begin"] < window_end:
            active_events.append(events[next_event])
            next_event += 1

        # The lower boundary is exclusive: an event at or before T is not future.
        active_events = [event for event in active_events if event["begin"] > timestamp]
        window_events = sorted(
            active_events,
            key=lambda event: event["peak_flux"],
            reverse=True,
        )

        if not window_events:
            output_rows.append(
                {
                    "label": image_label,
                    "goes_class": "NF",
                    "fl_lon": "UNK",
                    "fl_lat": "UNK",
                    "rest_fl": "",
                    "rest_lon": "",
                    "rest_lat": "",
                }
            )
            continue

        strongest, rest = window_events[0], window_events[1:]
        output_rows.append(
            {
                "label": image_label,
                "goes_class": strongest["goes_class"],
                "fl_lon": strongest["fl_lon"],
                "fl_lat": strongest["fl_lat"],
                "rest_fl": [event["goes_class"] for event in rest] if rest else "",
                "rest_lon": [event["fl_lon"] for event in rest] if rest else "",
                "rest_lat": [event["fl_lat"] for event in rest] if rest else "",
            }
        )

    return output_rows


def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
    if not CATALOGUE.is_file():
        raise FileNotFoundError(f"NOAA catalogue not found: {CATALOGUE}")

    images = load_images()
    catalogue_rows, events, skipped, duplicates = load_events()
    labels = create_label_rows(images, events)

    columns = [
        "label", "goes_class", "fl_lon", "fl_lat",
        "rest_fl", "rest_lon", "rest_lat",
    ]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(labels)

    print(f"Created: {OUTPUT_FILE}")
    print(f"Images labelled: {len(labels):,}")
    print(f"Catalogue rows: {len(catalogue_rows):,}")
    print(f"Events used: {len(events):,}")
    print(f"Rows skipped for missing Begin/class: {skipped:,}")
    print(f"Exact duplicate events removed: {duplicates:,}")
    print("Forecast rule: image time < NOAA Begin < image time + 24 hours")


if __name__ == "__main__":
    main()
