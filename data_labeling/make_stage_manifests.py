"""Create chronological Stage 1--4 manifests from the original paper folds.

No labels are regenerated. We only intersect the already committed fold files
with the chronological year ranges requested by the continual-learning plan:

* Stage 1: 2010--2012
* Stage 2: 2013--2014
* Stage 3: 2015--2016
* Stage 4: 2017--2018

The default remains two stages so existing commands and experiments do not
silently change. Pass ``--number-of-stages 4`` for the extended experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT.parent
    / "fulldiskattention"
    / "data_labeling"
    / "data_labels"
    / "simplified_data_labels"
)
STAGES = {
    1: (2010, 2012),
    2: (2013, 2014),
    3: (2015, 2016),
    4: (2017, 2018),
}
FILENAME_PATTERN = re.compile(r"HMI\.m(\d{4})\.(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})\.jpg$")


@dataclass(frozen=True)
class Row:
    label: str
    target: int

    @property
    def timestamp(self) -> datetime:
        match = FILENAME_PATTERN.search(Path(self.label).name)
        if not match:
            raise ValueError(f"Cannot parse HMI timestamp from {self.label!r}")
        return datetime(*(int(value) for value in match.groups()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument(
        "--number-of-stages",
        type=int,
        choices=(2, 4),
        default=2,
        help="Keep the existing two-stage plan or extend it through 2018",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "fold3",
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=0,
        help="0 keeps all rows; a positive value makes a small pilot manifest",
    )
    parser.add_argument(
        "--max-eval-per-class",
        type=int,
        default=0,
        help="0 keeps all evaluation rows",
    )
    parser.add_argument(
        "--fisher-per-class",
        type=int,
        default=64,
        help="Unique deterministic consolidation rows per class",
    )
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="If provided, fail when any generated train/eval image is missing",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"label", "goes_class"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected manifest schema in {path}")
        rows = [Row(item["label"].strip(), int(item["goes_class"])) for item in reader]
    if len({row.label for row in rows}) != len(rows):
        raise ValueError(f"Duplicate paths found in {path}")
    if any(row.target not in (0, 1) for row in rows):
        raise ValueError(f"Non-binary target found in {path}")
    return rows


def deterministic_cap(rows: list[Row], per_class: int, seed: int) -> list[Row]:
    """Take a reproducible class-stratified subset, then restore time order."""

    if per_class == 0:
        return sorted(rows, key=lambda row: row.timestamp)
    selected: list[Row] = []
    for target in (0, 1):
        candidates = [row for row in rows if row.target == target]
        # ``per_class`` is a maximum, not a request to fabricate observations.
        # This matters for Stage 4, which contains only 89 FL training rows in
        # Fold 3. Keep every available row when the requested cap is larger.
        count = min(per_class, len(candidates))
        selected.extend(random.Random(seed + target).sample(candidates, count))
    return sorted(selected, key=lambda row: row.timestamp)


def write_rows(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "goes_class"])
        writer.writeheader()
        writer.writerows(
            {"label": row.label, "goes_class": row.target} for row in rows
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(rows: list[Row]) -> dict[str, int]:
    class_counts = Counter(row.target for row in rows)
    return {"total": len(rows), "nf": class_counts[0], "fl": class_counts[1]}


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    train_source = source_dir / f"Fold{args.fold}_train.csv"
    eval_source = source_dir / f"Fold{args.fold}_val.csv"
    train_rows, eval_rows = read_rows(train_source), read_rows(eval_source)

    overlap = {row.label for row in train_rows} & {row.label for row in eval_rows}
    if overlap:
        raise ValueError(f"Original fold train/eval overlap contains {len(overlap)} paths")

    summary: dict[str, object] = {
        "outer_fold": args.fold,
        "seed": args.seed,
        "number_of_stages": args.number_of_stages,
        "source": {
            "train": str(train_source),
            "train_sha256": sha256(train_source),
            "eval": str(eval_source),
            "eval_sha256": sha256(eval_source),
        },
        "stages": {},
    }
    all_stage_paths: set[str] = set()

    selected_stages = list(STAGES.items())[: args.number_of_stages]
    for stage, (start_year, end_year) in selected_stages:
        stage_train_pool = [
            row for row in train_rows if start_year <= row.timestamp.year <= end_year
        ]
        stage_eval_pool = [
            row for row in eval_rows if start_year <= row.timestamp.year <= end_year
        ]
        stage_train = deterministic_cap(
            stage_train_pool, args.max_train_per_class, args.seed + 100 * stage
        )
        stage_eval = deterministic_cap(
            stage_eval_pool, args.max_eval_per_class, args.seed + 200 * stage
        )
        fisher_limit = min(
            args.fisher_per_class,
            min(sum(row.target == target for row in stage_train) for target in (0, 1)),
        )
        if fisher_limit <= 0:
            raise ValueError(f"Stage {stage} has no examples from one class")
        fisher_rows = deterministic_cap(
            stage_train, fisher_limit, args.seed + 300 * stage
        )

        train_path = output_dir / f"stage{stage}_train.csv"
        eval_path = output_dir / f"stage{stage}_eval.csv"
        fisher_path = output_dir / f"stage{stage}_fisher.csv"
        write_rows(train_path, stage_train)
        write_rows(eval_path, stage_eval)
        write_rows(fisher_path, fisher_rows)

        current_paths = {row.label for row in stage_train + stage_eval}
        if current_paths & all_stage_paths:
            raise ValueError(f"Stage {stage} overlaps an earlier chronological stage")
        all_stage_paths.update(current_paths)
        if not {row.label for row in fisher_rows}.issubset(
            {row.label for row in stage_train}
        ):
            raise AssertionError("Fisher rows must be a subset of Stage training rows")

        summary["stages"][str(stage)] = {
            "years": [start_year, end_year],
            "source_train_pool": counts(stage_train_pool),
            "source_eval_pool": counts(stage_eval_pool),
            "written_train": counts(stage_train),
            "written_eval": counts(stage_eval),
            "written_fisher": counts(fisher_rows),
            "files": {
                train_path.name: sha256(train_path),
                eval_path.name: sha256(eval_path),
                fisher_path.name: sha256(fisher_path),
            },
        }

    if args.image_root:
        image_root = args.image_root.expanduser().resolve()
        missing = [path for path in all_stage_paths if not (image_root / path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} generated paths are missing under {image_root}; "
                f"first path: {missing[0]}"
            )
        summary["validated_image_root"] = str(image_root)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["stages"], indent=2))
    print(f"Wrote chronological manifests to {output_dir}")


if __name__ == "__main__":
    main()
