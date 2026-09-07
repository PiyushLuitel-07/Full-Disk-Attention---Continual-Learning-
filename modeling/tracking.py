"""Weights & Biases tracking for one continual-learning stage.

Each chronological stage runs as a separate command and receives its own W&B
run. The runs share one group name, which keeps the complete continual sequence
together in the W&B interface.

The API key is intentionally never read from a configuration file. W&B uses
the account previously configured with ``wandb login`` on the training server.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


TRAIN_METRICS = (
    "ce_loss",
    "raw_ewc_penalty",
    "weighted_ewc_penalty",
    "total_loss",
    "accuracy",
    "recall",
    "precision",
    "fpr",
    "tss",
    "hss",
    "tp",
    "fp",
    "tn",
    "fn",
)


def _artifact_name(value: str) -> str:
    """Convert a human-readable run group into a valid artifact name."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")


class WandbTracker:
    """Small adapter that keeps W&B-specific calls out of the training code."""

    def __init__(
        self,
        run: Any | None,
        wandb_module: Any | None,
        group: str,
        log_artifacts: bool,
    ) -> None:
        self.run = run
        self.wandb = wandb_module
        self.group = group
        self.should_log_artifacts = log_artifacts

    @property
    def enabled(self) -> bool:
        return self.run is not None

    @classmethod
    def start(
        cls,
        config: dict[str, Any],
        stage: int,
        device: str,
    ) -> "WandbTracker":
        tracking = config.get("tracking", {})
        if not tracking.get("enabled", False):
            return cls(None, None, "", False)

        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B tracking is enabled, but 'wandb' is not installed. Run "
                "'python -m pip install -r requirements.txt' inside the project "
                "environment."
            ) from error

        experiment = config["experiment"]
        method = experiment["method"]
        fold = experiment["outer_fold"]
        seed = experiment["seed"]
        configured_group = tracking.get("group")
        group = configured_group or (
            f"{experiment['name']}-fold{fold}-seed{seed}-{method}"
        )
        run_name = f"{group}-stage{stage}"
        tags = list(tracking.get("tags", []))
        tags.extend([method, f"fold-{fold}", f"stage-{stage}"])

        logged_config = {
            **config,
            "runtime": {
                "training_stage": stage,
                "device": device,
            },
        }
        run = wandb.init(
            entity=tracking["entity"],
            project=tracking["project"],
            name=run_name,
            group=group,
            job_type=f"train-stage-{stage}",
            tags=tags,
            notes=tracking.get("notes"),
            mode=tracking.get("mode", "online"),
            config=logged_config,
            save_code=bool(tracking.get("save_code", True)),
        )
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("trained_through_stage")
        run.define_metric("eval/*", step_metric="trained_through_stage")
        return cls(
            run,
            wandb,
            group,
            bool(tracking.get("log_artifacts", True)),
        )

    def log_training_epoch(self, stage: int, epoch_result: dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload: dict[str, Any] = {
            "stage": stage,
            "epoch": epoch_result["epoch"],
            "train/learning_rate": epoch_result["learning_rate"],
        }
        for metric in TRAIN_METRICS:
            if metric in epoch_result:
                payload[f"train/{metric}"] = epoch_result[metric]
        self.run.log(payload)

    def log_evaluation(
        self,
        trained_through_stage: int,
        evaluated_stage: int,
        result: dict[str, Any],
        prediction_rows: list[dict[str, Any]],
    ) -> None:
        if not self.enabled:
            return
        prefix = f"eval/stage_{evaluated_stage}"
        payload: dict[str, Any] = {"trained_through_stage": trained_through_stage}
        for metric, value in result.items():
            payload[f"{prefix}/{metric}"] = value

        targets = [int(row["target"]) for row in prediction_rows]
        predictions = [int(row["predicted_class"]) for row in prediction_rows]
        payload[f"{prefix}/confusion_matrix"] = self.wandb.plot.confusion_matrix(
            probs=None,
            y_true=targets,
            preds=predictions,
            class_names=["NF", "FL"],
            title=(
                f"After Stage {trained_through_stage}: "
                f"Stage {evaluated_stage} evaluation"
            ),
        )
        self.run.log(payload)

        # Summary values remain easy to compare after the run has finished.
        for metric, value in result.items():
            self.run.summary[
                f"after_stage_{trained_through_stage}/stage_{evaluated_stage}/{metric}"
            ] = value

    def log_fisher(self, stage: int, summary: dict[str, Any]) -> None:
        if not self.enabled:
            return
        layers = summary["layers"]
        element_count = sum(math.prod(item["shape"]) for item in layers.values())
        nonzero_count = sum(item["nonzero"] for item in layers.values())
        total_importance = sum(item["sum"] for item in layers.values())
        maximum_importance = max(item["max"] for item in layers.values())

        table = self.wandb.Table(
            columns=["parameter", "shape", "mean", "max", "sum", "nonzero"]
        )
        for name, item in layers.items():
            table.add_data(
                name,
                "x".join(str(value) for value in item["shape"]),
                item["mean"],
                item["max"],
                item["sum"],
                item["nonzero"],
            )

        self.run.log(
            {
                "trained_through_stage": stage,
                "fisher/sample_count": summary["sample_count"],
                "fisher/parameter_count": element_count,
                "fisher/nonzero_count": nonzero_count,
                "fisher/nonzero_fraction": (
                    nonzero_count / element_count if element_count else 0.0
                ),
                "fisher/total_importance": total_importance,
                "fisher/maximum_importance": maximum_importance,
                "fisher/per_parameter": table,
            }
        )

    def log_stage_summary(self, summary: dict[str, Any]) -> None:
        if not self.enabled:
            return
        for key in (
            "trained_through_stage",
            "method",
            "device",
            "train_examples_in_manifest",
            "epochs",
            "max_batches_per_epoch",
            "ewc_was_active",
            "ewc_lambda",
            "fisher_examples",
            "elapsed_seconds",
            "checkpoint",
            "ewc_state",
            "ewc_protected_stages_before_training",
            "ewc_anchors_after_stage",
        ):
            self.run.summary[key] = summary[key]

    def log_continual_metrics(self, summary: dict[str, Any]) -> None:
        """Log formal CL metrics and their complete performance matrix."""

        if not self.enabled:
            return
        number_of_stages = int(summary["number_of_stages"])
        matrix_columns = [
            f"R_{trained}_{evaluated}"
            for trained in range(1, number_of_stages + 1)
            for evaluated in range(1, number_of_stages + 1)
        ]
        table = self.wandb.Table(
            columns=[
                "base_metric",
                *matrix_columns,
                "final_average",
                "average_incremental_performance",
                "forgetting",
                "backward_transfer",
                "average_learning_gain",
            ]
        )
        payload: dict[str, Any] = {"trained_through_stage": number_of_stages}
        for base_metric, values in summary["continual_metrics"].items():
            matrix = summary["performance_matrices"][base_metric]
            table.add_data(
                base_metric,
                *[
                    matrix[f"after_stage{trained}"][f"eval_stage{evaluated}"]
                    for trained in range(1, number_of_stages + 1)
                    for evaluated in range(1, number_of_stages + 1)
                ],
                values["final_average"],
                values["average_incremental_performance"],
                values["forgetting"],
                values["backward_transfer"],
                values["average_learning_gain"],
            )
            for name, value in values.items():
                if isinstance(value, dict):
                    for stage_name, stage_value in value.items():
                        key = f"cl/{base_metric}_{name}/{stage_name}"
                        payload[key] = stage_value
                        self.run.summary[key] = stage_value
                else:
                    key = f"cl/{base_metric}_{name}"
                    payload[key] = value
                    self.run.summary[key] = value
        payload["cl/performance_matrix_and_summary"] = table
        self.run.log(payload)

    def log_forward_transfer(self, summary: dict[str, Any]) -> None:
        """Log random baselines and standard forward-transfer measurements."""

        if not self.enabled:
            return
        table = self.wandb.Table(
            columns=[
                "base_metric",
                "R_1_2",
                "random_baseline_b_2",
                "forward_transfer",
                "stage2_only_after_training",
            ]
        )
        payload: dict[str, Any] = {"trained_through_stage": 2}
        for base_metric, values in summary["metrics"].items():
            table.add_data(
                base_metric,
                values["stage1_zero_shot_R_1_2"],
                values["random_init_baseline_b_2"],
                values["forward_transfer"],
                values["stage2_only_after_training"],
            )
            for name, value in values.items():
                key = f"forward_transfer/{base_metric}_{name}"
                payload[key] = value
                self.run.summary[key] = value
        payload["forward_transfer/summary"] = table
        self.run.log(payload)

    def log_run_artifact(
        self,
        stage: int,
        run_dir: Path,
        config_path: Path,
    ) -> None:
        if not self.enabled or not self.should_log_artifacts:
            return
        artifact = self.wandb.Artifact(
            name=f"{_artifact_name(self.group)}-through-stage-{stage}",
            type="continual-learning-run",
            description=(
                f"Checkpoints, EWC state, metrics, predictions and provenance "
                f"through Stage {stage}."
            ),
            metadata={"trained_through_stage": stage},
        )
        artifact.add_dir(str(run_dir), name="run")
        artifact.add_file(str(config_path), name="config.json")
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.enabled:
            self.run.finish()
