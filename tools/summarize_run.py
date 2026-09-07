"""Print TSS/HSS matrices and formal metrics for a completed CL run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from modeling.continual_metrics import calculate_continual_metrics


def _evaluation_stage_ids(summaries: dict[int, dict[str, Any]]) -> list[int]:
    ids: set[int] = set()
    for summary in summaries.values():
        for key in summary.get("evaluations", {}):
            match = re.fullmatch(r"stage([1-9][0-9]*)", key)
            if match:
                ids.add(int(match.group(1)))
    return sorted(ids)


def _value(summaries: dict[int, dict[str, Any]], i: int, j: int, metric: str) -> Any:
    return (
        summaries.get(i, {})
        .get("evaluations", {})
        .get(f"stage{j}", {})
        .get(metric, "")
    )


def format_summary(summaries: dict[int, dict[str, Any]]) -> list[str]:
    """Return CSV lines for the available metric matrix and final CL metrics."""

    trained_stage_ids = sorted(summaries)
    evaluated_stage_ids = _evaluation_stage_ids(summaries)
    columns = [
        f"after_stage{i}_eval_stage{j}"
        for i in trained_stage_ids
        for j in evaluated_stage_ids
    ]
    lines = ["metric," + ",".join(columns)]
    for metric in ("tss", "hss"):
        cells = [
            str(_value(summaries, i, j, metric))
            for i in trained_stage_ids
            for j in evaluated_stage_ids
        ]
        lines.append(f"{metric}," + ",".join(cells))

        # Preserve the compact line printed by the original two-stage tool.
        if trained_stage_ids == [1, 2] and evaluated_stage_ids == [1, 2]:
            first = _value(summaries, 1, 1, metric)
            retained = _value(summaries, 2, 1, metric)
            if isinstance(first, (int, float)) and isinstance(
                retained, (int, float)
            ):
                lines.append(f"{metric}_forgetting,{first - retained}")

    complete_ids = list(range(1, len(evaluated_stage_ids) + 1))
    if trained_stage_ids != complete_ids or evaluated_stage_ids != complete_ids:
        return lines
    matrix_is_complete = all(
        f"stage{evaluated}" in summaries[trained].get("evaluations", {})
        for trained in complete_ids
        for evaluated in complete_ids
    )
    if not matrix_is_complete:
        return lines

    result = calculate_continual_metrics(summaries)
    lines.extend(["", "continual_metric,tss,hss"])
    aggregate_names = (
        "final_average",
        "average_incremental_performance",
        "forgetting",
        "backward_transfer",
        "average_learning_gain",
        "final_stage_after_training",
    )
    for name in aggregate_names:
        lines.append(
            f"{name},"
            f"{result['continual_metrics']['tss'][name]},"
            f"{result['continual_metrics']['hss'][name]}"
        )

    nested_metrics = (
        ("average_after_each_stage", "average_after"),
        ("forgetting_by_stage", "forgetting"),
        ("backward_transfer_by_stage", "backward_transfer"),
        ("zero_shot_before_learning", "zero_shot_before_learning"),
        ("score_after_learning", "score_after_learning"),
        ("learning_gain_by_stage", "learning_gain"),
    )
    for source_name, display_name in nested_metrics:
        tss_values = result["continual_metrics"]["tss"][source_name]
        hss_values = result["continual_metrics"]["hss"][source_name]
        for stage_name in tss_values:
            lines.append(
                f"{display_name}_{stage_name},"
                f"{tss_values[stage_name]},{hss_values[stage_name]}"
            )

    if len(complete_ids) == 2:
        for name in ("stage2_zero_shot", "stage2_after_training", "stage2_gain"):
            lines.append(
                f"{name},"
                f"{result['continual_metrics']['tss'][name]},"
                f"{result['continual_metrics']['hss'][name]}"
            )
    return lines


def load_stage_summaries(run_dir: Path) -> dict[int, dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    metrics_dir = run_dir / "metrics"
    for path in metrics_dir.glob("stage*_summary.json"):
        match = re.fullmatch(r"stage([1-9][0-9]*)_summary\.json", path.name)
        if match:
            summaries[int(match.group(1))] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries = load_stage_summaries(args.run_dir)
    if not summaries:
        raise FileNotFoundError(f"No stage summaries found below {args.run_dir}")
    print("\n".join(format_summary(summaries)))


if __name__ == "__main__":
    main()
