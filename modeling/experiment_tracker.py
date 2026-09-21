"""Local and Weights & Biases tracking for continual-learning runs."""

import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import wandb


class ExperimentTracker:
    """Save one complete continual-learning experiment."""

    def __init__(self, config, results_directory, entity, project):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = f"ewc_continual_{timestamp}"
        self.directory = Path(results_directory) / self.name
        self.checkpoint_directory = self.directory / "checkpoints"
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)

        self.epoch_csv = self.directory / "epoch_metrics.csv"
        self.holdout_csv = self.directory / "holdout_metrics.csv"
        self.stage_csv = self.directory / "stage_metrics.csv"

        with (self.directory / "config.json").open("w") as config_file:
            json.dump(config, config_file, indent=2)

        self.config = config
        self.run = wandb.init(
            entity=entity,
            project=project,
            name=self.name,
            config=config,
            dir=str(self.directory),
            tags=["EWC", "continual-learning", "prototype"],
        )

        self.run.define_metric("global_epoch")
        self.run.define_metric("training/*", step_metric="global_epoch")
        self.run.define_metric(
            "current_holdout/*",
            step_metric="global_epoch",
        )
        self.run.define_metric("completed_stage")
        self.run.define_metric(
            "retention/*",
            step_metric="completed_stage",
        )

    @staticmethod
    def _append_csv(csv_path, row):
        """Append one dictionary as one CSV row."""
        write_header = not csv_path.exists()

        with csv_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=list(row),
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)

    @staticmethod
    def _history_on_cpu(ewc_history):
        """Copy Fisher values and old parameters to CPU."""
        saved_history = []

        for old_stage in ewc_history:
            saved_history.append(
                {
                    "stage": old_stage["stage"],
                    "fisher": {
                        name: value.detach().cpu()
                        for name, value in old_stage["fisher"].items()
                    },
                    "parameters": {
                        name: value.detach().cpu()
                        for name, value in old_stage["parameters"].items()
                    },
                }
            )

        return saved_history

    def log_epoch(
        self,
        global_epoch,
        stage_number,
        stage_epoch,
        train_metrics,
        holdout_metrics,
        learning_rate,
        epoch_seconds,
    ):
        """Save training and current-holdout metrics for one epoch."""
        row = {
            "global_epoch": global_epoch,
            "training_stage": stage_number,
            "stage_epoch": stage_epoch,
            **{
                f"train_{name}": value
                for name, value in train_metrics.items()
            },
            **{
                f"holdout_{name}": value
                for name, value in holdout_metrics.items()
            },
            "learning_rate": learning_rate,
            "epoch_seconds": epoch_seconds,
        }
        self._append_csv(self.epoch_csv, row)

        self.run.log(
            {
                "global_epoch": global_epoch,
                "training/stage": stage_number,
                "training/stage_epoch": stage_epoch,
                **{
                    f"training/{name}": value
                    for name, value in train_metrics.items()
                },
                **{
                    f"current_holdout/{name}": value
                    for name, value in holdout_metrics.items()
                },
                "system/learning_rate": learning_rate,
                "system/epoch_seconds": epoch_seconds,
            }
        )

    def log_holdouts(self, after_training_stage, holdout_results):
        """Save all historical holdout results after one stage."""
        wandb_values = {"completed_stage": after_training_stage}

        for evaluated_stage, metrics in holdout_results.items():
            self._append_csv(
                self.holdout_csv,
                {
                    "after_training_stage": after_training_stage,
                    "evaluated_stage": evaluated_stage,
                    **metrics,
                },
            )

            for name, value in metrics.items():
                wandb_values[
                    f"retention/stage_{evaluated_stage}_{name}"
                ] = value

        self.run.log(wandb_values)

    def save_stage(
        self,
        model,
        optimizer,
        ewc_history,
        stage_number,
        loaders,
        fisher_seconds,
    ):
        """Save one stage checkpoint and its stage-level information."""
        checkpoint_path = (
            self.checkpoint_directory
            / f"stage{stage_number}.pt"
        )

        torch.save(
            {
                "completed_stage": stage_number,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ewc_history": self._history_on_cpu(ewc_history),
                "config": self.config,
            },
            checkpoint_path,
        )

        stage_metrics = {
            "stage": stage_number,
            "balanced_training_samples": len(loaders["train"].dataset),
            "original_fisher_samples": len(loaders["fisher"].dataset),
            "holdout_samples": len(loaders["holdout"].dataset),
            "fisher_seconds": fisher_seconds,
            "checkpoint": str(checkpoint_path),
        }
        self._append_csv(self.stage_csv, stage_metrics)

        self.run.log(
            {
                "completed_stage": stage_number,
                **{
                    f"stage/{name}": value
                    for name, value in stage_metrics.items()
                    if name not in {"stage", "checkpoint"}
                },
            }
        )

        return checkpoint_path

    def finish(self, exit_code=0):
        self.run.finish(exit_code=exit_code)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.finish(exit_code=1 if exception_type else 0)
