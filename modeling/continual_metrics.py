"""Formal metrics for the focused two-stage continual-learning experiment.

For each base forecast metric (TSS or HSS), ``R_i,j`` means performance on
Stage j evaluation data after training through Stage i. The complete two-stage
matrix therefore contains R_1,1, R_1,2, R_2,1, and R_2,2.
"""

from __future__ import annotations

import math
from typing import Any, Callable


BASE_METRICS = ("tss", "hss")


def _score(
    summary: dict[str, Any],
    evaluated_stage: int,
    metric: str,
    trained_through_stage: int,
) -> float | None:
    try:
        value = summary["evaluations"][f"stage{evaluated_stage}"][metric]
    except KeyError as error:
        raise ValueError(
            f"Missing R_{trained_through_stage},{evaluated_stage} {metric}. "
            "The complete evaluation matrix must be recorded before calculating "
            "continual-learning metrics."
        ) from error
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"R_{trained_through_stage},{evaluated_stage} {metric} is not numeric"
        )
    return float(value) if math.isfinite(value) else None


def _calculate(
    values: tuple[float | None, ...],
    operation: Callable[..., float],
) -> float | None:
    """Propagate an undefined base score instead of inventing a CL value."""

    if any(value is None for value in values):
        return None
    return float(operation(*values))


def calculate_two_stage_metrics(
    stage1_summary: dict[str, Any],
    stage2_summary: dict[str, Any],
) -> dict[str, Any]:
    """Calculate stability, plasticity, and overall two-stage performance."""

    matrices: dict[str, Any] = {}
    derived: dict[str, Any] = {}
    for metric in BASE_METRICS:
        r11 = _score(stage1_summary, 1, metric, 1)
        r12 = _score(stage1_summary, 2, metric, 1)
        r21 = _score(stage2_summary, 1, metric, 2)
        r22 = _score(stage2_summary, 2, metric, 2)
        final_average = _calculate(
            (r21, r22), lambda old, new: (old + new) / 2.0
        )
        average_incremental = _calculate(
            (r11, final_average), lambda first, final: (first + final) / 2.0
        )

        matrices[metric] = {
            "after_stage1": {"eval_stage1": r11, "eval_stage2": r12},
            "after_stage2": {"eval_stage1": r21, "eval_stage2": r22},
        }
        derived[metric] = {
            "average_after_stage1": r11,
            "final_average": final_average,
            "average_incremental_performance": average_incremental,
            "forgetting": _calculate((r11, r21), lambda old, retained: old - retained),
            "backward_transfer": _calculate(
                (r11, r21), lambda old, retained: retained - old
            ),
            "stage2_zero_shot": r12,
            "stage2_after_training": r22,
            "stage2_gain": _calculate(
                (r12, r22), lambda zero_shot, learned: learned - zero_shot
            ),
        }

    return {
        "definitions": {
            "R_i_j": "base metric on eval Stage j after training through Stage i",
            "final_average": "(R_2_1 + R_2_2) / 2",
            "average_incremental_performance": "(R_1_1 + final_average) / 2",
            "forgetting": "R_1_1 - R_2_1; positive means old-stage loss",
            "backward_transfer": "R_2_1 - R_1_1; negative means forgetting",
            "stage2_gain": "R_2_2 - R_1_2",
        },
        "performance_matrices": matrices,
        "continual_metrics": derived,
    }
