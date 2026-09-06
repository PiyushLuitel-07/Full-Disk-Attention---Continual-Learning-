"""Print the complete two-stage TSS/HSS evaluation matrix and forgetting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from modeling.continual_metrics import calculate_two_stage_metrics


def format_summary(summaries: dict[int, dict[str, Any]]) -> list[str]:
    """Return CSV lines for the complete two-stage metric matrix."""

    lines = [
        "metric,after_stage1_eval_stage1,after_stage1_eval_stage2,"
        "after_stage2_eval_stage1,after_stage2_eval_stage2"
    ]
    for metric in ("tss", "hss"):
        first = summaries.get(1, {}).get("evaluations", {}).get("stage1", {}).get(metric, "")
        future_before_learning = (
            summaries.get(1, {})
            .get("evaluations", {})
            .get("stage2", {})
            .get(metric, "")
        )
        old_after_new = summaries.get(2, {}).get("evaluations", {}).get("stage1", {}).get(metric, "")
        current = summaries.get(2, {}).get("evaluations", {}).get("stage2", {}).get(metric, "")
        lines.append(
            f"{metric},{first},{future_before_learning},{old_after_new},{current}"
        )
        if isinstance(first, (int, float)) and isinstance(old_after_new, (int, float)):
            lines.append(f"{metric}_forgetting,{first - old_after_new}")
    if 1 in summaries and 2 in summaries:
        result = calculate_two_stage_metrics(summaries[1], summaries[2])
        lines.extend(["", "continual_metric,tss,hss"])
        metric_names = (
            "final_average",
            "average_incremental_performance",
            "forgetting",
            "backward_transfer",
            "stage2_zero_shot",
            "stage2_after_training",
            "stage2_gain",
        )
        for name in metric_names:
            tss_value = result["continual_metrics"]["tss"][name]
            hss_value = result["continual_metrics"]["hss"][name]
            lines.append(f"{name},{tss_value},{hss_value}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries = {}
    for stage in (1, 2):
        path = args.run_dir / "metrics" / f"stage{stage}_summary.json"
        if path.exists():
            summaries[stage] = json.loads(path.read_text(encoding="utf-8"))

    if not summaries:
        raise FileNotFoundError(f"No stage summaries found below {args.run_dir}")
    print("\n".join(format_summary(summaries)))


if __name__ == "__main__":
    main()
