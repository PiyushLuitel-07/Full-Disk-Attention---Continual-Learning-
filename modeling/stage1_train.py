"""Tune a fresh attention model on Stage 1 only."""

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
    BATCH_SIZE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EPOCHS_PER_STAGE,
    IMAGE_DIRECTORY,
    IMAGE_SIZE,
    LEARNING_RATE,
    MIN_LEARNING_RATE,
    NUM_WORKERS,
    RANDOM_SEED,
    RESULTS_DIRECTORY,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    STAGE_DIRECTORY,
    VALIDATION_CSV,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WEIGHT_DECAY,
    evaluate_learned_stages,
    train_one_epoch,
)
from experiment_tracker import ExperimentTracker


# Stage 1 has no previous task to protect, so EWC must be disabled.
EWC_LAMBDA = 0.0


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    number_of_gpus = torch.cuda.device_count()
    gpu_names = (
        [torch.cuda.get_device_name(i) for i in range(number_of_gpus)]
        if torch.cuda.is_available()
        else ["CPU"]
    )

    discovered_stages = discover_stages(STAGE_DIRECTORY)
    stage1_matches = [stage for stage in discovered_stages if stage["number"] == 1]
    if len(stage1_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Stage 1 dataset, found {len(stage1_matches)}."
        )
    stage = stage1_matches[0]
    stage_number = 1

    print(f"Device: {device}")
    print(f"GPUs available to PyTorch: {number_of_gpus}")
    for index, gpu_name in enumerate(gpu_names):
        print(f"  GPU {index}: {gpu_name}")
    print("Training mode: Stage 1 only")

    config = {
        "training_mode": "stage1_hyperparameter_tuning",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "epochs_per_stage": EPOCHS_PER_STAGE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "ewc_lambda": EWC_LAMBDA,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "min_learning_rate": MIN_LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "attention": True,
        "augmentations": [
            "original",
            "horizontal_flip",
            "vertical_flip",
            "rotation_180",
            "small_rotation",
        ],
        "fisher_data": "not calculated during Stage 1 sweep trials",
        "stages": [1],
        "stage_periods": {1: "2010-12 to 2017-12"},
        "normalization": "0_to_1_scaling",
        "stage_directory": str(STAGE_DIRECTORY),
        "validation_csv": str(VALIDATION_CSV),
        "image_directory": str(IMAGE_DIRECTORY),
        "device": str(device),
        "number_of_gpus": number_of_gpus,
        "gpus": gpu_names,
        "multi_gpu_method": "DataParallel" if number_of_gpus > 1 else "single_device",
        "pytorch_version": torch.__version__,
    }

    with ExperimentTracker(
        config=config,
        results_directory=RESULTS_DIRECTORY,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    ) as tracker:
        batch_size = int(tracker.config["batch_size"])
        epochs_per_stage = int(tracker.config["epochs_per_stage"])
        learning_rate = float(tracker.config["learning_rate"])
        weight_decay = float(tracker.config["weight_decay"])
        early_stopping_patience = int(tracker.config["early_stopping_patience"])
        early_stopping_min_delta = float(
            tracker.config["early_stopping_min_delta"]
        )
        scheduler_factor = float(tracker.config["scheduler_factor"])
        scheduler_patience = int(tracker.config["scheduler_patience"])
        min_learning_rate = float(tracker.config["min_learning_rate"])

        model = Attn_Net(
            im_size=IMAGE_SIZE,
            num_classes=2,
            attention=True,
            init="kaimingUniform",
        ).to(device)
        if number_of_gpus > 1:
            model = nn.DataParallel(model)
            print(f"Using DataParallel on {number_of_gpus} GPUs")

        criterion = nn.CrossEntropyLoss()
        loaders = build_stage_loaders(
            train_csv=stage["train_file"],
            holdout_csv=stage["holdout_file"],
            image_directory=IMAGE_DIRECTORY,
            batch_size=batch_size,
            image_size=IMAGE_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
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

        print(f"\n{'=' * 60}\nTraining Stage 1\n{'=' * 60}")
        for stage_epoch in range(1, epochs_per_stage + 1):
            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(
                model=model,
                data_loader=loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                ewc_history=[],
                ewc_lambda=EWC_LAMBDA,
            )
            holdout_metrics = evaluate(
                model,
                loaders["holdout"],
                criterion,
                device,
            )
            epoch_seconds = time.perf_counter() - epoch_start

            tracker.log_epoch(
                global_epoch=stage_epoch,
                stage_number=stage_number,
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
                    stage_number,
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
                    f"Early stopping Stage 1 at epoch {stage_epoch}: "
                    f"holdout CSS did not improve by at least "
                    f"{early_stopping_min_delta:.4f} for "
                    f"{early_stopping_patience} epochs."
                )
                break

        if best_checkpoint is None:
            raise RuntimeError("No valid Stage 1 checkpoint was created.")

        restored_checkpoint = tracker.restore_best_stage_model(
            best_checkpoint,
            model,
            optimizer,
            device,
        )
        print(
            "Restored Stage 1 best checkpoint from epoch "
            f"{restored_checkpoint['stage_epoch']} "
            f"(holdout CSS={restored_checkpoint['holdout_css']:.4f})"
        )

        holdout_results = evaluate_learned_stages(
            model,
            {1: loaders["holdout"]},
            criterion,
            device,
        )
        tracker.log_holdouts(stage_number, holdout_results)

        # Fisher is intentionally deferred until the winning Stage 1 trial is chosen.
        checkpoint = tracker.save_stage(
            model=model,
            optimizer=optimizer,
            ewc_history=[],
            stage_number=stage_number,
            loaders=loaders,
            fisher_seconds=0.0,
            best_stage_epoch=best_stage_epoch,
            best_holdout_css=best_holdout_css,
            stopping_epoch=stopping_epoch,
            stopped_early=stopped_early,
        )
        print(f"Stage 1 completed: {checkpoint}")

        print("\nEvaluating restored Stage 1 model on validation data...")
        validation_loader = build_evaluation_loader(
            csv_file=VALIDATION_CSV,
            image_directory=IMAGE_DIRECTORY,
            batch_size=batch_size,
            image_size=IMAGE_SIZE,
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
        print("\nStage 1 trial completed.")
        print(f"Local results: {tracker.directory}")


if __name__ == "__main__":
    main()
