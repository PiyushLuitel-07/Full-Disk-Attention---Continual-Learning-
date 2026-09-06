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

    results: dict[str, Any] = {}
    for metric in BASE_METRICS:
        try:
            ewc = ewc_summary["continual_metrics"][metric]
            finetune = finetune_summary["continual_metrics"][metric]
        except KeyError as error:
            raise ValueError(f"Missing {metric} continual metrics") from error
        results[metric] = {
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
            "stage2_performance_difference": _difference(
                ewc["stage2_after_training"],
                finetune["stage2_after_training"],
                lambda proposed, baseline: proposed - baseline,
            ),
            "backward_transfer_difference": _difference(
                ewc["backward_transfer"],
                finetune["backward_transfer"],
                lambda proposed, baseline: proposed - baseline,
            ),
        }

    return {
        "definitions": {
            "forgetting_reduction": "finetune forgetting - EWC forgetting; positive favors EWC",
            "final_average_difference": "EWC final average - finetune final average",
            "stage2_performance_difference": "EWC R_2_2 - finetune R_2_2",
            "backward_transfer_difference": "EWC BWT - finetune BWT",
        },
        "comparison_controls": ewc_controls,
        "metrics": results,
    }
