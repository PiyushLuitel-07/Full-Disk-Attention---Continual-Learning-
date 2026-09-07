import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from download_mag import download_from_manifests as downloader


class DownloaderTests(unittest.TestCase):
    def setUp(self) -> None:
        image = Image.new("L", (4, 4), color=127)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.image_bytes = buffer.getvalue()

    def _fake_request(self, url, params, _timeout, _retries):
        requested = str(params["date"])
        if url == downloader.METADATA_API:
            returned = (
                "2016-08-25 03:00:00"
                if requested == "2016-08-24T06:00:00Z"
                else requested.replace("T", " ").removesuffix("Z")
            )
            return json.dumps({"date": returned, "id": 19}).encode()
        return self.image_bytes

    def test_resilient_mode_skips_existing_and_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "label,target\n"
                "2016/08/23/HMI.m2016.08.23_08.00.00.jpg,0\n"
                "2016/08/24/HMI.m2016.08.24_06.00.00.jpg,1\n"
                "2016/08/26/HMI.m2016.08.26_04.00.00.jpg,0\n",
                encoding="utf-8",
            )
            image_root = root / "images"
            existing = image_root / "2016/08/23/HMI.m2016.08.23_08.00.00.jpg"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"already present")
            provenance = root / "provenance.csv"
            arguments = [
                "download_from_manifests.py",
                "--manifest",
                str(manifest),
                "--image-root",
                str(image_root),
                "--provenance",
                str(provenance),
                "--skip-existing",
                "--skip-unavailable",
            ]

            with patch.object(sys, "argv", arguments), patch.object(
                downloader, "request_bytes", side_effect=self._fake_request
            ), redirect_stdout(io.StringIO()):
                downloader.main()

            with provenance.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["status"] for row in rows],
                ["skipped_existing", "skipped_unavailable", "downloaded"],
            )
            downloaded = image_root / "2016/08/26/HMI.m2016.08.26_04.00.00.jpg"
            self.assertTrue(downloaded.is_file())

    def test_strict_mode_still_rejects_unavailable_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "label,target\n"
                "2016/08/24/HMI.m2016.08.24_06.00.00.jpg,1\n",
                encoding="utf-8",
            )
            arguments = [
                "download_from_manifests.py",
                "--manifest",
                str(manifest),
                "--image-root",
                str(root / "images"),
                "--provenance",
                str(root / "provenance.csv"),
            ]

            with patch.object(sys, "argv", arguments), patch.object(
                downloader, "request_bytes", side_effect=self._fake_request
            ), redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                RuntimeError, "Closest HMI observation"
            ):
                downloader.main()


if __name__ == "__main__":
    unittest.main()
