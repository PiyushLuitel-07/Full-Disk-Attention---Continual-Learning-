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
        self.validation_csv = self.directory / "validation_metrics.csv"
        # remember the best historical performance on every stage’s holdout.
        self.best_holdout_tss = {}
        self.best_holdout_hss = {}
        self.best_holdout_css = {}

        self.run = wandb.init(
            entity=entity,
            project=project,
            name=self.name,
            config=config,
            dir=str(self.directory),
            tags=["EWC", "continual-learning", "prototype"],
        )
        self.config = dict(self.run.config)

        with (self.directory / "config.json").open("w") as config_file:
            json.dump(self.config, config_file, indent=2)

        # tells W&B what to use as the x-axis:
        self.run.define_metric("global_epoch")
        self.run.define_metric("training/*", step_metric="global_epoch")
        self.run.define_metric("current_holdout/*", step_metric="global_epoch")
        self.run.define_metric("completed_stage")
        self.run.define_metric("retention/*", step_metric="completed_stage")
        self.run.define_metric("forgetting/*", step_metric="completed_stage")

    @staticmethod
    def _append_csv(csv_path, row):
        """Append one dictionary as one CSV row."""
        write_header = not csv_path.exists()

        with csv_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(row))
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
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"holdout_{name}": value for name, value in holdout_metrics.items()},
            "learning_rate": learning_rate,
            "epoch_seconds": epoch_seconds,
        }
        self._append_csv(self.epoch_csv, row)

        self.run.log(
            {
                "global_epoch": global_epoch,
                "training/stage": stage_number,
                "training/stage_epoch": stage_epoch,
                **{f"training/{name}": value for name, value in train_metrics.items()},
                **{
                    f"current_holdout/{name}": value
                    for name, value in holdout_metrics.items()
                },
                "system/learning_rate": learning_rate,
                "system/epoch_seconds": epoch_seconds,
            }
        )

# Runs after completing a stage.
# It evaluates the current model on every available historical holdout.
    def log_holdouts(self, after_training_stage, holdout_results):
        """Save all historical holdout results after one stage."""
        wandb_values = {"completed_stage": after_training_stage}

        for evaluated_stage, metrics in holdout_results.items():
            # Forgetting is the drop from the best result seen for this holdout.
            previous_best_tss = self.best_holdout_tss.get(evaluated_stage, metrics["tss"])
            previous_best_hss = self.best_holdout_hss.get(evaluated_stage, metrics["hss"])
            previous_best_css = self.best_holdout_css.get(evaluated_stage, metrics["css"])

            tss_forgetting = max(0.0, previous_best_tss - metrics["tss"])
            hss_forgetting = max(0.0, previous_best_hss - metrics["hss"])
            css_forgetting = max(0.0, previous_best_css - metrics["css"])

            self.best_holdout_tss[evaluated_stage] = max(previous_best_tss, metrics["tss"])
            self.best_holdout_hss[evaluated_stage] = max(previous_best_hss, metrics["hss"])
            self.best_holdout_css[evaluated_stage] = max(previous_best_css, metrics["css"])

            self._append_csv(
                self.holdout_csv,
                {
                    "after_training_stage": after_training_stage,
                    "evaluated_stage": evaluated_stage,
                    **metrics,
                    "tss_forgetting": tss_forgetting,
                    "hss_forgetting": hss_forgetting,
                    "css_forgetting": css_forgetting,
                },
            )

            for name, value in metrics.items():
                wandb_values[f"retention/stage_{evaluated_stage}_{name}"] = value

            wandb_values[f"forgetting/stage_{evaluated_stage}_tss"] = tss_forgetting
            wandb_values[f"forgetting/stage_{evaluated_stage}_hss"] = hss_forgetting
            wandb_values[f"forgetting/stage_{evaluated_stage}_css"] = css_forgetting

            # Add the four confusion counts as one W&B bar chart.
            confusion_table = wandb.Table(
                columns=["Outcome", "Count"],
                data=[
                    ["True positive", metrics["tp"]],
                    ["True negative", metrics["tn"]],
                    ["False positive", metrics["fp"]],
                    ["False negative", metrics["fn"]],
                ],
            )
            wandb_values[
                "confusion_counts/"
                f"after_stage_{after_training_stage}_on_stage_{evaluated_stage}"
            ] = wandb.plot.bar(
                confusion_table,
                "Outcome",
                "Count",
                title=f"After Stage {after_training_stage}: Stage {evaluated_stage} holdout",
            )

            # Keep the latest stage-level values visible in the run summary.
            for name in ("loss", "f1", "tss", "hss", "css"):
                self.run.summary[
                    f"latest_retention/stage_{evaluated_stage}_{name}"
                ] = metrics[name]

            self.run.summary[
                f"latest_forgetting/stage_{evaluated_stage}_tss"
            ] = tss_forgetting
            self.run.summary[
                f"latest_forgetting/stage_{evaluated_stage}_hss"
            ] = hss_forgetting
            self.run.summary[
                f"latest_forgetting/stage_{evaluated_stage}_css"
            ] = css_forgetting

        self.run.log(wandb_values)

# It evaluates the final Stage 3 model on the future validation dataset:
    def log_validation(self, metrics, samples, label_csv):
        """Save final validation metrics for this complete training run."""
        self._append_csv(
            self.validation_csv,
            {
                "dataset": "validation_2024_2025",
                "samples": samples,
                "label_csv": str(label_csv),
                **metrics,
            },
        )

        self.run.log({f"validation/{name}": value for name, value in metrics.items()})

        for name, value in metrics.items():
            self.run.summary[f"validation/{name}"] = value

    def save_best_stage_model(
        self, model, optimizer, stage_number, stage_epoch, holdout_css
    ):
        """Save the best model observed within one training stage."""
        checkpoint_path = self.checkpoint_directory / f"stage{stage_number}_best.pt"
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model

        # Save everything required to resume from the best epoch.
        torch.save(
            {
                "stage": stage_number,
                "stage_epoch": stage_epoch,
                "holdout_css": holdout_css,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": self.config,
            },
            checkpoint_path,
        )

        return checkpoint_path

# This is done after training or early stopping so that the system continues using the best weights,
#  rather than the weights from the last epoch.
    @staticmethod
    def restore_best_stage_model(checkpoint_path, model, optimizer, device):
        """Restore the model and optimizer from a best-stage checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_to_restore = (
            model.module if isinstance(model, torch.nn.DataParallel) else model
        )
        model_to_restore.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    '''
    Saves the final state of a completed continual-learning stage.
    Examples:
    checkpoints/stage1.pt
    checkpoints/stage2.pt
    checkpoints/stage3.pt 
    '''
    def save_stage(
        self,
        model,
        optimizer,
        ewc_history,
        stage_number,
        loaders,
        fisher_seconds,
        best_stage_epoch,
        best_holdout_css,
        stopping_epoch,
        stopped_early,
    ):
        """Save one stage checkpoint and its stage-level information."""
        checkpoint_path = self.checkpoint_directory / f"stage{stage_number}.pt"
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model

        # Final stage checkpoints also contain the accumulated EWC history.
        torch.save(
            {
                "completed_stage": stage_number,
                "model_state_dict": model_to_save.state_dict(),
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
            "best_stage_epoch": best_stage_epoch,
            "best_holdout_css": best_holdout_css,
            "stopping_epoch": stopping_epoch,
            "stopped_early": stopped_early,
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
