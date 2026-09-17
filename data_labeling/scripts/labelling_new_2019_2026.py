"""Create 24-hour flare labels for downloaded 2019-July 2026 JPGs."""

import csv
import re
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
CATALOGUE = (
    ROOT
    / "data_labeling"
    / "Catalogue"
    / "goes_flares_catalogue_2019_2026.csv"
)
OUTPUT_FILE = (
    ROOT
    / "data_labeling"
    / "data_labels"
    / "labels_2019_2026_july.csv"
)
START = datetime(2019, 1, 1)
END = datetime(2026, 7, 31, 23)


def image_time(path):
    return datetime.strptime(path.name, "HMI.m%Y.%m.%d_%H.%M.%S.jpg")


def flare_strength(goes_class):
    rank = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
    return rank[goes_class[0]], float(goes_class[1:])


def flare_location(position):
    match = re.search(r"([NS])(\d+)([EW])(\d+)", position or "")
    if not match:
        return "UNK", "UNK"

    north_south, latitude, east_west, longitude = match.groups()
    latitude = int(latitude) * (1 if north_south == "N" else -1)
    longitude = int(longitude) * (1 if east_west == "W" else -1)
    return longitude, latitude


def main():
    images = sorted(
        (image_time(path), path.relative_to(IMAGE_DIR).as_posix())
        for year in range(2019, 2027)
        for path in (IMAGE_DIR / str(year)).rglob("*.jpg")
        if path.stat().st_size > 0 and START <= image_time(path) <= END
    )

    with CATALOGUE.open(newline="", encoding="utf-8-sig") as file:
        events = list(csv.DictReader(file))

    for event in events:
        event["time"] = datetime.strptime(
            event["event_start"], "%Y/%m/%d %H:%M:%S"
        )
        event["fl_lon"], event["fl_lat"] = flare_location(
            event["event_position"]
        )

    events.sort(key=lambda event: event["time"])
    event_times = [event["time"] for event in events]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "label", "goes_class", "fl_lon", "fl_lat",
        "rest_fl", "rest_lon", "rest_lat",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for timestamp, label in images:
            first = bisect_right(event_times, timestamp)
            last = bisect_left(event_times, timestamp + timedelta(hours=24))
            window = sorted(
                events[first:last],
                key=lambda event: flare_strength(event["event_GOES"]),
                reverse=True,
            )

            if window:
                strongest = window[0]
                rest = window[1:]
                row = {
                    "label": label,
                    "goes_class": strongest["event_GOES"],
                    "fl_lon": strongest["fl_lon"],
                    "fl_lat": strongest["fl_lat"],
                    "rest_fl": [event["event_GOES"] for event in rest],
                    "rest_lon": [event["fl_lon"] for event in rest],
                    "rest_lat": [event["fl_lat"] for event in rest],
                }
                if not rest:
                    row["rest_fl"] = row["rest_lon"] = row["rest_lat"] = ""
            else:
                row = {
                    "label": label,
                    "goes_class": "NF",
                    "fl_lon": "unk",
                    "fl_lat": "unk",
                    "rest_fl": "",
                    "rest_lon": "",
                    "rest_lat": "",
                }

            writer.writerow(row)

    print(f"Created {OUTPUT_FILE}")
    print(f"Images labelled: {len(images)}")


if __name__ == "__main__":
    main()
