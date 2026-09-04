"""Bounded Helioviewer downloader for only the rows in selected manifests.

Unlike the original downloader, this script never loops over an open-ended date
range. It preserves the manifest's YYYY/MM/DD/file.jpg hierarchy and records
requested/returned timestamps plus the JP2 checksum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image


JP2_API = "https://api.helioviewer.org/v2/getJP2Image/"
METADATA_API = "https://api.helioviewer.org/v2/getClosestImage/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--jp2-root", type=Path)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--source-id", type=int, default=19)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def request_bytes(url: str, params: dict[str, object], timeout: int, retries: int) -> bytes:
    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                request_url, headers={"User-Agent": "fulldiskattention-continual/1.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} for {request_url}")
                return response.read()
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {request_url}") from last_error


def read_unique_paths(manifests: list[Path]) -> list[str]:
    paths: set[str] = set()
    for manifest in manifests:
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                paths.add(row["label"].strip())
    return sorted(paths)


def requested_timestamp(relative_path: str) -> datetime:
    return datetime.strptime(
        Path(relative_path).name, "HMI.m%Y.%m.%d_%H.%M.%S.jpg"
    )


def main() -> None:
    args = parse_args()
    image_root = args.image_root.expanduser().resolve()
    paths = read_unique_paths([path.expanduser().resolve() for path in args.manifest])
    if args.max_images:
        paths = paths[: args.max_images]
    if not paths:
        raise ValueError("The supplied manifests contain no image paths")
    provenance_rows: list[dict[str, object]] = []

    for number, relative_path in enumerate(paths, start=1):
        destination = image_root / relative_path
        timestamp = requested_timestamp(relative_path)
        requested_utc = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        metadata_bytes = request_bytes(
            METADATA_API,
            {"date": requested_utc, "sourceId": args.source_id},
            args.timeout,
            args.retries,
        )
        metadata = json.loads(metadata_bytes)
        returned = datetime.strptime(metadata["date"], "%Y-%m-%d %H:%M:%S")
        offset_seconds = abs(int((returned - timestamp).total_seconds()))
        if offset_seconds > 12 * 60:
            raise RuntimeError(
                f"Closest HMI observation is {offset_seconds}s from {requested_utc}"
            )

        jp2_bytes = request_bytes(
            JP2_API,
            {"date": requested_utc, "sourceId": args.source_id},
            args.timeout,
            args.retries,
        )
        with Image.open(io.BytesIO(jp2_bytes)) as decoded:
            image = decoded.convert("L").resize((512, 512), Image.Resampling.LANCZOS)
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="JPEG", quality=95)

        if args.jp2_root:
            jp2_path = args.jp2_root.expanduser().resolve() / Path(relative_path).with_suffix(
                ".jp2"
            )
            jp2_path.parent.mkdir(parents=True, exist_ok=True)
            jp2_path.write_bytes(jp2_bytes)

        provenance_rows.append(
            {
                "relative_jpg": relative_path,
                "requested_utc": requested_utc,
                "returned_utc": metadata["date"] + "Z",
                "offset_seconds": offset_seconds,
                "helioviewer_id": metadata.get("id", ""),
                "jp2_sha256": hashlib.sha256(jp2_bytes).hexdigest(),
            }
        )
        print(f"[{number}/{len(paths)}] {requested_utc} -> {destination}")

    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    with args.provenance.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(provenance_rows[0]))
        writer.writeheader()
        writer.writerows(provenance_rows)
    print(f"Downloaded {len(paths)} images; provenance: {args.provenance}")


if __name__ == "__main__":
    main()
