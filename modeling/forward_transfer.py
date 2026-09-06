"""Forward-transfer metrics for the Stage-2-only reference experiment.

The standard two-stage forward-transfer measurement compares performance on
Stage 2 before learning Stage 2 with performance from a randomly initialized
model on the same Stage 2 evaluation rows::

    FWT = R_1,2 - b_2

The trained Stage-2-only model is retained as a separate reference.  Its score
is useful, but it is not the random baseline used in the FWT equation.
"""

from __future__ import annotations

import math
from typing import Any

from .continual_metrics import BASE_METRICS


def _metric(
    values: dict[str, Any],
    metric: str,
    description: str,
) -> float | None:
    if metric not in values:
        raise ValueError(f"Missing {description} {metric}")
    value = values[metric]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} {metric} is not numeric")
    return float(value) if math.isfinite(value) else None


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def calculate_forward_transfer(
    stage1_summary: dict[str, Any],
    random_baseline: dict[str, Any],
    stage2_only_after_training: dict[str, Any],
) -> dict[str, Any]:
    """Compare Stage 1 transfer with random and trained Stage-2 references."""

    try:
        zero_shot = stage1_summary["evaluations"]["stage2"]
    except KeyError as error:
        raise ValueError(
            "The source Stage 1 summary is missing its Stage 2 zero-shot evaluation "
            "R_1,2"
        ) from error

    metrics: dict[str, Any] = {}
    for metric in BASE_METRICS:
        r12 = _metric(zero_shot, metric, "R_1,2")
        b2 = _metric(random_baseline, metric, "random Stage 2 baseline b_2")
        scratch = _metric(
            stage2_only_after_training,
            metric,
            "trained Stage-2-only reference",
        )
        metrics[metric] = {
            "stage1_zero_shot_R_1_2": r12,
            "random_init_baseline_b_2": b2,
            "forward_transfer": _difference(r12, b2),
            "stage2_only_after_training": scratch,
        }

    return {
        "definitions": {
            "stage1_zero_shot_R_1_2": (
                "Stage 2 evaluation performance after learning only Stage 1"
            ),
            "random_init_baseline_b_2": (
                "Stage 2 evaluation performance before learning either stage"
            ),
            "forward_transfer": "R_1,2 - b_2; positive means beneficial forward transfer",
            "stage2_only_after_training": (
                "Randomly initialized reference after training only on Stage 2; "
                "reported separately and not used as b_2"
            ),
        },
        "metrics": metrics,
    }
