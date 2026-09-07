"""Matched comparison of EWC and sequential fine-tuning results."""

from __future__ import annotations

from typing import Any, Callable


BASE_METRICS = ("tss", "hss")


def _difference(
    left: float | None,
    right: float | None,
    operation: Callable[[float, float], float],
) -> float | None:
    if left is None or right is None:
        return None
    return float(operation(float(left), float(right)))


def _optional_metric(values: dict[str, Any], name: str) -> float | None:
    value = values.get(name)
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def compare_ewc_with_finetune(
    ewc_summary: dict[str, Any],
    finetune_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare matched runs, rejecting changes to experimental controls."""

    if ewc_summary.get("method") != "ewc":
        raise ValueError("The EWC input must have method='ewc'")
    if finetune_summary.get("method") != "finetune":
        raise ValueError("The fine-tuning input must have method='finetune'")

    ewc_controls = ewc_summary.get("comparison_controls")
    finetune_controls = finetune_summary.get("comparison_controls")
    if ewc_controls is None or finetune_controls is None:
        raise ValueError("Both summaries must contain comparison_controls")
    if ewc_controls != finetune_controls:
        raise ValueError(
            "EWC and fine-tuning controls differ. Use identical stages, model, "
            "training settings, manifests, image root, fold, and seed."
        )

    stages = ewc_controls.get("stages", [])
    final_stage = len(stages)
    if final_stage < 2:
        raise ValueError("Comparison requires at least two chronological stages")

    results: dict[str, Any] = {}
    for metric in BASE_METRICS:
        try:
            ewc = ewc_summary["continual_metrics"][metric]
            finetune = finetune_summary["continual_metrics"][metric]
        except KeyError as error:
            raise ValueError(f"Missing {metric} continual metrics") from error
        ewc_final_stage = _optional_metric(ewc, "final_stage_after_training")
        finetune_final_stage = _optional_metric(
            finetune, "final_stage_after_training"
        )
        # Old two-stage summaries predate the generic field name.
        if final_stage == 2:
            if ewc_final_stage is None:
                ewc_final_stage = _optional_metric(ewc, "stage2_after_training")
            if finetune_final_stage is None:
                finetune_final_stage = _optional_metric(
                    finetune, "stage2_after_training"
                )
        metric_result = {
            "ewc": ewc,
            "finetune": finetune,
            # Positive means EWC forgot less than ordinary fine-tuning.
            "forgetting_reduction": _difference(
                finetune["forgetting"],
                ewc["forgetting"],
                lambda baseline, proposed: baseline - proposed,
            ),
            # Positive means EWC has the better value for these skill metrics.
            "final_average_difference": _difference(
                ewc["final_average"],
                finetune["final_average"],
                lambda proposed, baseline: proposed - baseline,
            ),
            "final_stage_performance_difference": _difference(
                ewc_final_stage,
                finetune_final_stage,
                lambda proposed, baseline: proposed - baseline,
            ),
            "backward_transfer_difference": _difference(
                ewc["backward_transfer"],
                finetune["backward_transfer"],
                lambda proposed, baseline: proposed - baseline,
            ),
            "average_incremental_performance_difference": _difference(
                _optional_metric(ewc, "average_incremental_performance"),
                _optional_metric(finetune, "average_incremental_performance"),
                lambda proposed, baseline: proposed - baseline,
            ),
            "average_learning_gain_difference": _difference(
                _optional_metric(ewc, "average_learning_gain"),
                _optional_metric(finetune, "average_learning_gain"),
                lambda proposed, baseline: proposed - baseline,
            ),
        }
        if final_stage == 2:
            metric_result["stage2_performance_difference"] = metric_result[
                "final_stage_performance_difference"
            ]
        results[metric] = metric_result

    return {
        "definitions": {
            "forgetting_reduction": "finetune forgetting - EWC forgetting; positive favors EWC",
            "final_average_difference": "EWC final average - finetune final average",
            "final_stage_performance_difference": "EWC R_T,T - finetune R_T,T",
            "backward_transfer_difference": "EWC BWT - finetune BWT",
            "average_incremental_performance_difference": "EWC AIP - finetune AIP",
            "average_learning_gain_difference": (
                "EWC average learning gain - finetune average learning gain"
            ),
        },
        "number_of_stages": final_stage,
        "comparison_controls": ewc_controls,
        "metrics": results,
    }
