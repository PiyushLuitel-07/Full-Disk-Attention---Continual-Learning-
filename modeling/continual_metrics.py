"""Formal metrics for chronological continual-learning experiments.

For each base forecast metric (TSS or HSS), ``R_i,j`` means performance on
Stage j evaluation data after training through Stage i. The trainer records the
complete matrix, including future-stage zero-shot cells, without using those
future examples for optimization.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any


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


def _mean(values: list[float | None]) -> float | None:
    return _calculate(tuple(values), lambda *items: sum(items) / len(items))


def calculate_continual_metrics(
    summaries: Mapping[int, dict[str, Any]],
) -> dict[str, Any]:
    """Calculate standard matrix-based metrics through the final stage.

    ``summaries[i]`` must contain evaluations of every configured period after
    training through Stage ``i``. Stage IDs must be consecutive and start at 1.
    Undefined TSS/HSS cells propagate to every derived value that uses them.
    """

    stage_ids = sorted(summaries)
    if not stage_ids or stage_ids != list(range(1, len(stage_ids) + 1)):
        raise ValueError("Stage summaries must have consecutive IDs starting at 1")
    final_stage = stage_ids[-1]

    matrices: dict[str, Any] = {}
    derived: dict[str, Any] = {}
    for metric in BASE_METRICS:
        values = {
            trained_stage: {
                evaluated_stage: _score(
                    summaries[trained_stage],
                    evaluated_stage,
                    metric,
                    trained_stage,
                )
                for evaluated_stage in stage_ids
            }
            for trained_stage in stage_ids
        }
        matrices[metric] = {
            f"after_stage{trained_stage}": {
                f"eval_stage{evaluated_stage}": values[trained_stage][evaluated_stage]
                for evaluated_stage in stage_ids
            }
            for trained_stage in stage_ids
        }

        average_after_stage = {
            f"stage{trained_stage}": _mean(
                [values[trained_stage][task] for task in range(1, trained_stage + 1)]
            )
            for trained_stage in stage_ids
        }
        forgetting_by_stage: dict[str, float | None] = {}
        backward_transfer_by_stage: dict[str, float | None] = {}
        for task in range(1, final_stage):
            before_final = [values[when][task] for when in range(task, final_stage)]
            final_score = values[final_stage][task]
            forgetting_by_stage[f"stage{task}"] = _calculate(
                (*before_final, final_score),
                lambda *items: max(items[:-1]) - items[-1],
            )
            backward_transfer_by_stage[f"stage{task}"] = _calculate(
                (values[task][task], final_score),
                lambda learned, final: final - learned,
            )

        zero_shot_by_stage = {
            f"stage{task}": values[task - 1][task]
            for task in range(2, final_stage + 1)
        }
        learned_score_by_stage = {
            f"stage{task}": values[task][task]
            for task in range(2, final_stage + 1)
        }
        learning_gain_by_stage = {
            f"stage{task}": _calculate(
                (values[task - 1][task], values[task][task]),
                lambda zero_shot, learned: learned - zero_shot,
            )
            for task in range(2, final_stage + 1)
        }

        result: dict[str, Any] = {
            "average_after_each_stage": average_after_stage,
            "final_average": average_after_stage[f"stage{final_stage}"],
            "average_incremental_performance": _mean(
                list(average_after_stage.values())
            ),
            "forgetting": _mean(list(forgetting_by_stage.values()))
            if forgetting_by_stage
            else None,
            "backward_transfer": _mean(list(backward_transfer_by_stage.values()))
            if backward_transfer_by_stage
            else None,
            "forgetting_by_stage": forgetting_by_stage,
            "backward_transfer_by_stage": backward_transfer_by_stage,
            "zero_shot_before_learning": zero_shot_by_stage,
            "score_after_learning": learned_score_by_stage,
            "learning_gain_by_stage": learning_gain_by_stage,
            "average_learning_gain": _mean(list(learning_gain_by_stage.values()))
            if learning_gain_by_stage
            else None,
            "final_stage_after_training": values[final_stage][final_stage],
        }
        # Retain the original names so existing two-stage result readers remain
        # compatible with newly generated two-stage summaries.
        if final_stage == 2:
            result.update(
                {
                    "average_after_stage1": average_after_stage["stage1"],
                    "stage2_zero_shot": zero_shot_by_stage["stage2"],
                    "stage2_after_training": learned_score_by_stage["stage2"],
                    "stage2_gain": learning_gain_by_stage["stage2"],
                }
            )
        derived[metric] = result

    return {
        "definitions": {
            "R_i_j": "base metric on eval Stage j after training through Stage i",
            "average_after_each_stage": "A_i = mean(R_i,j for j <= i)",
            "final_average": "A_T = mean(R_T,j for j = 1..T)",
            "average_incremental_performance": "mean(A_i for i = 1..T)",
            "forgetting_by_stage": (
                "max(R_l,j for l = j..T-1) - R_T,j; positive means loss"
            ),
            "forgetting": "mean forgetting over old stages j = 1..T-1",
            "backward_transfer_by_stage": "R_T,j - R_j,j",
            "backward_transfer": "mean backward transfer over old stages",
            "learning_gain_by_stage": "R_j,j - R_(j-1),j for j >= 2",
            "average_learning_gain": "mean learning gain over Stages 2..T",
        },
        "number_of_stages": final_stage,
        "performance_matrices": matrices,
        "continual_metrics": derived,
    }


def calculate_two_stage_metrics(
    stage1_summary: dict[str, Any],
    stage2_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper for the original two-stage interface."""

    return calculate_continual_metrics({1: stage1_summary, 2: stage2_summary})
