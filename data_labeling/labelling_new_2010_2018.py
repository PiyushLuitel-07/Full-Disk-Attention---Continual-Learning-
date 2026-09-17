"""Create 24-hour flare labels for downloaded 2010-2018 JPG images."""

import csv
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
GOES_FILE = ROOT / "data_labeling" / "data_source" / "goes_flares_integrated.csv"
OUTPUT_FILE = ROOT / "data_labeling" / "data_labels" / "labels_2010_2018.csv"
START = datetime(2010, 12, 1)
END = datetime(2018, 12, 30, 23)


def image_time(path):
    return datetime.strptime(path.name, "HMI.m%Y.%m.%d_%H.%M.%S.jpg")


def flare_strength(goes_class):
    rank = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
    return rank[goes_class[0]], float(goes_class[1:])


def main():
    images = sorted(
        (image_time(path), path.relative_to(IMAGE_DIR).as_posix())
        for year in range(2010, 2019)
        for path in (IMAGE_DIR / str(year)).rglob("*.jpg")
        if path.stat().st_size > 0 and START <= image_time(path) <= END
    )

    with GOES_FILE.open(newline="", encoding="utf-8-sig") as file:
        events = list(csv.DictReader(file))

    for event in events:
        event["time"] = datetime.strptime(
            event["start_time"], "%Y-%m-%d %H:%M:%S"
        )

    events.sort(key=lambda event: event["time"])
    event_times = [event["time"] for event in events]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        columns = [
            "label", "goes_class", "fl_lon", "fl_lat",
            "rest_fl", "rest_lon", "rest_lat",
        ]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for timestamp, label in images:
            first = bisect_right(event_times, timestamp)
            last = bisect_left(event_times, timestamp + timedelta(hours=24))
            window = sorted(
                events[first:last],
                key=lambda event: flare_strength(event["goes_class"]),
                reverse=True,
            )

            if window:
                strongest = window[0]
                goes_class = strongest["goes_class"]
                longitude = strongest["fl_lon"] or "UNK"
                latitude = strongest["fl_lat"] or "UNK"
                rest_fl = [event["goes_class"] for event in window[1:]]
                rest_lon = [event["fl_lon"] or "UNK" for event in window[1:]]
                rest_lat = [event["fl_lat"] or "UNK" for event in window[1:]]
                if not rest_fl:
                    rest_fl = rest_lon = rest_lat = ""
            else:
                goes_class = "NF"
                longitude = "unk"
                latitude = "unk"
                rest_fl = ""
                rest_lon = ""
                rest_lat = ""

            writer.writerow({
                "label": label,
                "goes_class": goes_class,
                "fl_lon": longitude,
                "fl_lat": latitude,
                "rest_fl": rest_fl,
                "rest_lon": rest_lon,
                "rest_lat": rest_lat,
            })

    print(f"Created {OUTPUT_FILE}")
    print(f"Images labelled: {len(images)}")


if __name__ == "__main__":
    main()
