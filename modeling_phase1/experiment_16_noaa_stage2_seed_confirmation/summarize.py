"""Aggregate Experiment 16 metrics as mean and sample standard deviation."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from config import RESULTS_DIR


METRICS = (
    "selected_epoch", "stage1_retention/css", "stage1_retention/tss", "stage1_retention/hss",
    "stage2_holdout/css", "stage2_holdout/tss", "stage2_holdout/hss",
    "stage2_holdout/recall", "stage2_holdout/false_positive_rate",
    "validation/css", "validation/tss", "validation/hss",
    "test/css", "test/tss", "test/hss", "test/recall", "test/false_positive_rate",
)


def read_metrics(path):
    return {row["metric"]: row["value"] for row in csv.DictReader(path.open())}


def main():
    grouped = defaultdict(list)
    for path in RESULTS_DIR.glob("candidate_*_seed_*/final_metrics.csv"):
        values = read_metrics(path)
        grouped[values["candidate_id"]].append(values)
    if not grouped:
        raise FileNotFoundError(f"No completed runs found under {RESULTS_DIR}")

    detail, summary = [], []
    for candidate, runs in sorted(grouped.items()):
        for run in sorted(runs, key=lambda values: int(float(values["random_seed"]))):
            detail.append({"candidate_id": candidate, "random_seed": int(float(run["random_seed"])),
                           **{metric: float(run[metric]) for metric in METRICS}})
        row = {"candidate_id": candidate, "completed_seeds": len(runs)}
        for metric in METRICS:
            numbers = [float(run[metric]) for run in runs]
            row[f"{metric}_mean"] = mean(numbers)
            row[f"{metric}_std"] = stdev(numbers) if len(numbers) > 1 else 0.0
        summary.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("seed_results.csv", detail), ("seed_summary.csv", summary)):
        with (RESULTS_DIR / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {RESULTS_DIR / name}")


if __name__ == "__main__":
    main()

