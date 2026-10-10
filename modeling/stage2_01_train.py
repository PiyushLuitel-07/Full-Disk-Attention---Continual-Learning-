"""Train Stage 2 from the fixed, EWC-ready Stage 1 winner.

Every Stage 2 sweep trial must begin from the same selected Stage 1 weights,
protected Stage 1 parameters, and Stage 1 Fisher information. The Stage 1
learning rate, weight decay, and batch size are also held fixed. Only
``ewc_lambda`` is intended to vary between trials.

Within a trial, Stage 2 holdout CSS controls learning-rate scheduling, early
stopping, and best-epoch restoration. After restoration, the script evaluates
both Stage 1 holdout retention and future validation performance. The W&B
sweep can optimize ``validation/css`` while the recorded Stage 1 metrics allow
retention and forgetting to be reviewed before declaring the final winner.

The test set is deliberately excluded from this script.
"""

import random
import time

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from attention_model import Attn_Net
from dataloader import build_evaluation_loader, build_stage_loaders, discover_stages
from evaluation import evaluate
from ewc_train import (
    IMAGE_DIRECTORY,
    NUM_WORKERS,
    RESULTS_DIRECTORY,
    STAGE_DIRECTORY,
    VALIDATION_CSV,
    WANDB_ENTITY,
    WANDB_PROJECT,
    evaluate_learned_stages,
    train_one_epoch,
)
from experiment_tracker import ExperimentTracker


# All Stage 2 trials start from this exact Stage 1 winner and Fisher estimate.
STAGE1_EWC_CHECKPOINT = (
    RESULTS_DIRECTORY
    / "ewc_continual_20261010_043758"
    / "checkpoints"
    / "stage1_ewc_ready.pt"
)

# This value is replaced by wandb.config during the lambda sweep.
EWC_LAMBDA = 1.0


def select_stage(stages, stage_number):
    """Return exactly one discovered stage or fail with a clear error."""
    matches = [stage for stage in stages if stage["number"] == stage_number]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Stage {stage_number} dataset, "
            f"found {len(matches)}."
        )
    return matches[0]


def load_stage1_checkpoint(checkpoint_path, device):
    """Load and validate the fixed EWC-ready Stage 1 checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"EWC-ready Stage 1 checkpoint was not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    required_keys = {
        "completed_stage",
        "model_state_dict",
        "config",
        "ewc_history",
        "fisher_metadata",
    }
    missing_keys = required_keys - set(checkpoint)
    if missing_keys:
        raise ValueError(
            f"Stage 1 checkpoint is missing keys: {sorted(missing_keys)}"
        )
    if checkpoint["completed_stage"] != 1:
        raise ValueError(
            "Stage 2 must start from a completed Stage 1 checkpoint."
        )
    if len(checkpoint["ewc_history"]) != 1:
        raise ValueError(
            "The EWC-ready Stage 1 checkpoint must contain exactly one "
            "EWC history entry."
        )
    if checkpoint["ewc_history"][0].get("stage") != 1:
        raise ValueError("The EWC history entry is not labeled as Stage 1.")

    return checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_checkpoint = load_stage1_checkpoint(
        STAGE1_EWC_CHECKPOINT,
        device,
    )
    stage1_config = dict(stage1_checkpoint["config"])

    # These values are fixed from the selected Stage 1 trial. They are not
    # Stage 2 sweep dimensions.
    image_size = int(stage1_config["image_size"])
    batch_size = int(stage1_config["batch_size"])
    learning_rate = float(stage1_config["learning_rate"])
    weight_decay = float(stage1_config["weight_decay"])
    epochs_per_stage = int(stage1_config["epochs_per_stage"])
    early_stopping_patience = int(stage1_config["early_stopping_patience"])
    early_stopping_min_delta = float(
        stage1_config["early_stopping_min_delta"]
    )
    scheduler_factor = float(stage1_config["scheduler_factor"])
    scheduler_patience = int(stage1_config["scheduler_patience"])
    min_learning_rate = float(stage1_config["min_learning_rate"])
    random_seed = int(stage1_config["random_seed"])

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    number_of_gpus = torch.cuda.device_count()
    gpu_names = (
        [torch.cuda.get_device_name(i) for i in range(number_of_gpus)]
        if torch.cuda.is_available()
        else ["CPU"]
    )

    stages = discover_stages(STAGE_DIRECTORY)
    stage1 = select_stage(stages, 1)
    stage2 = select_stage(stages, 2)

    print(f"Device: {device}")
    print(f"GPUs available to PyTorch: {number_of_gpus}")
    for index, gpu_name in enumerate(gpu_names):
        print(f"  GPU {index}: {gpu_name}")
    print("Training mode: Stage 2 only")
    print(f"Fixed Stage 1 checkpoint: {STAGE1_EWC_CHECKPOINT}")
    print(f"Fixed batch size: {batch_size}")
    print(f"Fixed learning rate: {learning_rate:.8g}")
    print(f"Fixed weight decay: {weight_decay:.8g}")

    config = {
        "training_mode": "stage2_lambda_tuning",
        "source_stage1_checkpoint": str(STAGE1_EWC_CHECKPOINT),
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "epochs_per_stage": epochs_per_stage,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "ewc_lambda": EWC_LAMBDA,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "min_learning_rate": min_learning_rate,
        "random_seed": random_seed,
        "attention": True,
        "augmentations": stage1_config.get("augmentations", []),
        "fisher_data": "fixed Stage 1 Fisher from complete original training data",
        "stages": [2],
        "stage_periods": {2: "2018-01 to 2023-06"},
        "normalization": stage1_config.get("normalization", "0_to_1_scaling"),
        "stage_directory": str(STAGE_DIRECTORY),
        "validation_csv": str(VALIDATION_CSV),
        "image_directory": str(IMAGE_DIRECTORY),
        "device": str(device),
        "number_of_gpus": number_of_gpus,
        "gpus": gpu_names,
        "multi_gpu_method": "DataParallel" if number_of_gpus > 1 else "single_device",
        "pytorch_version": torch.__version__,
        "stage1_fisher_metadata": stage1_checkpoint["fisher_metadata"],
    }

    with ExperimentTracker(
        config=config,
        results_directory=RESULTS_DIRECTORY,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    ) as tracker:
        # Only lambda should be supplied by the Stage 2 sweep.
        ewc_lambda = float(tracker.config["ewc_lambda"])
        if ewc_lambda <= 0:
            raise ValueError("Stage 2 EWC lambda must be greater than zero.")

        # Recreate the model and restore the exact selected Stage 1 weights.
        model = Attn_Net(
            im_size=image_size,
            num_classes=2,
            attention=True,
            init="kaimingUniform",
        ).to(device)
        model.load_state_dict(stage1_checkpoint["model_state_dict"])
        if number_of_gpus > 1:
            model = nn.DataParallel(model)
            print(f"Using DataParallel on {number_of_gpus} GPUs")

        criterion = nn.CrossEntropyLoss()
        ewc_history = stage1_checkpoint["ewc_history"]

        # Stage 1 holdout remains untouched. It is evaluated immediately to
        # establish this run's retention baseline, then evaluated again after
        # Stage 2 using the same samples and transformation.
        stage1_holdout_loader = build_evaluation_loader(
            csv_file=stage1["holdout_file"],
            image_directory=IMAGE_DIRECTORY,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        stage2_loaders = build_stage_loaders(
            train_csv=stage2["train_file"],
            holdout_csv=stage2["holdout_file"],
            image_directory=IMAGE_DIRECTORY,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )

        print("\nEvaluating fixed Stage 1 retention baseline...")
        stage1_baseline_metrics = evaluate(
            model,
            stage1_holdout_loader,
            criterion,
            device,
        )
        tracker.log_holdouts(1, {1: stage1_baseline_metrics})
        stage1_baseline_css = stage1_baseline_metrics["css"]
        print(
            f"Stage 1 baseline | CSS={stage1_baseline_css:.4f} | "
            f"TSS={stage1_baseline_metrics['tss']:.4f} | "
            f"HSS={stage1_baseline_metrics['hss']:.4f}"
        )

        optimizer = SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=early_stopping_min_delta,
            threshold_mode="abs",
            min_lr=min_learning_rate,
        )

        best_holdout_css = float("-inf")
        best_stage_epoch = None
        best_checkpoint = None
        epochs_without_improvement = 0
        stopped_early = False
        stopping_epoch = epochs_per_stage

        print(f"\n{'=' * 60}\nTraining Stage 2\n{'=' * 60}")
        print(f"EWC lambda: {ewc_lambda:.8g}")

        for stage_epoch in range(1, epochs_per_stage + 1):
            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(
                model=model,
                data_loader=stage2_loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                ewc_history=ewc_history,
                ewc_lambda=ewc_lambda,
            )
            holdout_metrics = evaluate(
                model,
                stage2_loaders["holdout"],
                criterion,
                device,
            )
            epoch_seconds = time.perf_counter() - epoch_start

            tracker.log_epoch(
                global_epoch=stage_epoch,
                stage_number=2,
                stage_epoch=stage_epoch,
                train_metrics=train_metrics,
                holdout_metrics=holdout_metrics,
                learning_rate=optimizer.param_groups[0]["lr"],
                epoch_seconds=epoch_seconds,
            )

            if holdout_metrics["css"] > best_holdout_css + early_stopping_min_delta:
                best_holdout_css = holdout_metrics["css"]
                best_stage_epoch = stage_epoch
                epochs_without_improvement = 0
                best_checkpoint = tracker.save_best_stage_model(
                    model,
                    optimizer,
                    2,
                    stage_epoch,
                    best_holdout_css,
                )
            else:
                epochs_without_improvement += 1

            learning_rate_before_step = optimizer.param_groups[0]["lr"]
            scheduler.step(holdout_metrics["css"])
            learning_rate_after_step = optimizer.param_groups[0]["lr"]
            if learning_rate_after_step < learning_rate_before_step:
                print(
                    f"Reduced learning rate from {learning_rate_before_step:.6g} "
                    f"to {learning_rate_after_step:.6g}"
                )

            print(
                f"Epoch {stage_epoch}/{epochs_per_stage} | "
                f"loss={train_metrics['loss']:.4f} | "
                f"classification={train_metrics['classification_loss']:.4f} | "
                f"EWC={train_metrics['ewc_loss']:.4f} | "
                f"holdout CSS={holdout_metrics['css']:.4f} | "
                f"holdout HSS={holdout_metrics['hss']:.4f} | "
                f"holdout TSS={holdout_metrics['tss']:.4f} | "
                f"time={epoch_seconds:.1f}s"
            )

            if epochs_without_improvement >= early_stopping_patience:
                stopped_early = True
                stopping_epoch = stage_epoch
                print(
                    f"Early stopping Stage 2 at epoch {stage_epoch}: "
                    f"holdout CSS did not improve by at least "
                    f"{early_stopping_min_delta:.4f} for "
                    f"{early_stopping_patience} epochs."
                )
                break

        if best_checkpoint is None:
            raise RuntimeError("No valid Stage 2 checkpoint was created.")

        restored_checkpoint = tracker.restore_best_stage_model(
            best_checkpoint,
            model,
            optimizer,
            device,
        )
        print(
            "Restored Stage 2 best checkpoint from epoch "
            f"{restored_checkpoint['stage_epoch']} "
            f"(holdout CSS={restored_checkpoint['holdout_css']:.4f})"
        )

        holdout_results = evaluate_learned_stages(
            model,
            {
                1: stage1_holdout_loader,
                2: stage2_loaders["holdout"],
            },
            criterion,
            device,
        )
        tracker.log_holdouts(2, holdout_results)

        retained_stage1_css = holdout_results[1]["css"]
        stage1_css_forgetting = max(
            0.0,
            stage1_baseline_css - retained_stage1_css,
        )
        stage1_css_retention_ratio = (
            retained_stage1_css / stage1_baseline_css
            if stage1_baseline_css > 0
            else 0.0
        )
        tracker.run.log(
            {
                "completed_stage": 2,
                "selection/stage1_baseline_css": stage1_baseline_css,
                "selection/stage1_retained_css": retained_stage1_css,
                "selection/stage1_css_forgetting": stage1_css_forgetting,
                "selection/stage1_css_retention_ratio": (
                    stage1_css_retention_ratio
                ),
            }
        )
        tracker.run.summary["selection/stage1_baseline_css"] = (
            stage1_baseline_css
        )
        tracker.run.summary["selection/stage1_retained_css"] = (
            retained_stage1_css
        )
        tracker.run.summary["selection/stage1_css_forgetting"] = (
            stage1_css_forgetting
        )
        tracker.run.summary["selection/stage1_css_retention_ratio"] = (
            stage1_css_retention_ratio
        )

        # Stage 2 Fisher is intentionally deferred until the winning lambda is
        # selected. The Stage 2 checkpoint still carries the fixed Stage 1 EWC
        # history needed to reproduce this trial.
        checkpoint_path = tracker.save_stage(
            model=model,
            optimizer=optimizer,
            ewc_history=ewc_history,
            stage_number=2,
            loaders=stage2_loaders,
            fisher_seconds=0.0,
            best_stage_epoch=best_stage_epoch,
            best_holdout_css=best_holdout_css,
            stopping_epoch=stopping_epoch,
            stopped_early=stopped_early,
        )
        print(f"Stage 2 completed: {checkpoint_path}")
        print(
            f"Stage 1 retention | baseline CSS={stage1_baseline_css:.4f} | "
            f"after Stage 2 CSS={retained_stage1_css:.4f} | "
            f"forgetting={stage1_css_forgetting:.4f} | "
            f"retention ratio={stage1_css_retention_ratio:.4f}"
        )

        print("\nEvaluating restored Stage 2 model on validation data...")
        validation_loader = build_evaluation_loader(
            csv_file=VALIDATION_CSV,
            image_directory=IMAGE_DIRECTORY,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )
        tracker.log_validation(
            validation_metrics,
            len(validation_loader.dataset),
            VALIDATION_CSV,
        )
        print(
            f"Validation | CSS={validation_metrics['css']:.4f} | "
            f"TSS={validation_metrics['tss']:.4f} | "
            f"HSS={validation_metrics['hss']:.4f} | "
            f"loss={validation_metrics['loss']:.4f}"
        )
        print("\nStage 2 trial completed.")
        print(f"Local results: {tracker.directory}")


if __name__ == "__main__":
    main()
