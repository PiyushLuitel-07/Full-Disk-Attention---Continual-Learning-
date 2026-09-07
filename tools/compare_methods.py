"""Compare matched EWC and fine-tuning continual-learning runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from modeling.method_comparison import compare_ewc_with_finetune


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "metrics" / "continual_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed continual summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ewc-run-dir", type=Path, required=True)
    parser.add_argument("--finetune-run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: <ewc-run-dir>/metrics/ewc_vs_finetune.json",
    )
    args = parser.parse_args()

    comparison = compare_ewc_with_finetune(
        load_summary(args.ewc_run_dir),
        load_summary(args.finetune_run_dir),
    )
    output = args.output or (
        args.ewc_run_dir / "metrics" / "ewc_vs_finetune.json"
    )
    write_json(output, comparison)

    print(
        "metric,forgetting_reduction,final_average_difference,"
        "backward_transfer_difference,average_incremental_performance_difference,"
        "average_learning_gain_difference,final_stage_difference"
    )
    for metric, values in comparison["metrics"].items():
        print(
            f"{metric},{values['forgetting_reduction']},"
            f"{values['final_average_difference']},"
            f"{values['backward_transfer_difference']},"
            f"{values['average_incremental_performance_difference']},"
            f"{values['average_learning_gain_difference']},"
            f"{values['final_stage_performance_difference']}"
        )
    print(f"Saved comparison: {output}")


if __name__ == "__main__":
    main()
