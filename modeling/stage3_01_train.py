"""Train Stage 3 from the fixed, EWC-ready Stage 2 winner.

Stage 3 starts from the single Stage 2 model selected using validation CSS and
reviewed for Stage 1 retention. Its Stage 1 training hyperparameters and the
EWC lambda selected during Stage 2 are fixed; Stage 3 performs no additional
hyperparameter sweep.

The source checkpoint contains two EWC-history entries: one for Stage 1 and
one for Stage 2. Stage 3 training applies both entries so that important
parameters from both earlier stages are protected. Stage 3 holdout CSS controls
learning-rate scheduling, early stopping, and best-epoch restoration. After
restoration, the script evaluates Stage 1 and Stage 2 retention and the common
future validation set.

The test set is deliberately excluded. The final restored ``stage3.pt`` model
must be evaluated on the test set only once, in a separate command, after this
training run finishes.
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


# Stage 3 always starts from this selected Stage 2 model and its two Fisher
# estimates. Test-set results played no role in selecting this checkpoint.
STAGE2_EWC_CHECKPOINT = (
    RESULTS_DIRECTORY
    / "ewc_continual_20261010_130641"
    / "checkpoints"
    / "stage2_ewc_ready.pt"
)


def select_stage(stages, stage_number):
    """Return exactly one discovered stage or fail with a clear error."""
    matches = [stage for stage in stages if stage["number"] == stage_number]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Stage {stage_number} dataset, "
            f"found {len(matches)}."
        )
    return matches[0]


def load_stage2_checkpoint(checkpoint_path, device):
    """Load and validate the fixed EWC-ready Stage 2 checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"EWC-ready Stage 2 checkpoint was not found: {checkpoint_path}"
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
        "stage2_fisher_metadata",
    }
    missing_keys = required_keys - set(checkpoint)
    if missing_keys:
        raise ValueError(
            f"Stage 2 checkpoint is missing keys: {sorted(missing_keys)}"
        )
    if checkpoint["completed_stage"] != 2:
        raise ValueError(
            "Stage 3 must start from a completed Stage 2 checkpoint."
        )

    history_stages = [
        entry.get("stage") for entry in checkpoint["ewc_history"]
    ]
    if history_stages != [1, 2]:
        raise ValueError(
            "The EWC-ready Stage 2 checkpoint must contain exactly the "
            f"Stage 1 and Stage 2 history entries; found {history_stages}."
        )

    ewc_lambda = float(checkpoint["config"].get("ewc_lambda", 0.0))
    if ewc_lambda <= 0:
        raise ValueError(
            "The selected Stage 2 checkpoint must contain a positive "
            "ewc_lambda."
        )

    return checkpoint


def retention_statistics(baseline_css, retained_css):
    """Return nonnegative CSS forgetting and the retained CSS fraction."""
    forgetting = max(0.0, baseline_css - retained_css)
    retention_ratio = (
        retained_css / baseline_css if baseline_css > 0 else 0.0
    )
    return forgetting, retention_ratio


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage2_checkpoint = load_stage2_checkpoint(
        STAGE2_EWC_CHECKPOINT,
        device,
    )
    fixed_config = dict(stage2_checkpoint["config"])

    # All optimization settings come from the already selected earlier stages.
    # Stage 3 introduces no new tuning dimension.
    image_size = int(fixed_config["image_size"])
    batch_size = int(fixed_config["batch_size"])
    learning_rate = float(fixed_config["learning_rate"])
    weight_decay = float(fixed_config["weight_decay"])
    ewc_lambda = float(fixed_config["ewc_lambda"])
    epochs_per_stage = int(fixed_config["epochs_per_stage"])
    early_stopping_patience = int(fixed_config["early_stopping_patience"])
    early_stopping_min_delta = float(
        fixed_config["early_stopping_min_delta"]
    )
    scheduler_factor = float(fixed_config["scheduler_factor"])
    scheduler_patience = int(fixed_config["scheduler_patience"])
    min_learning_rate = float(fixed_config["min_learning_rate"])
    random_seed = int(fixed_config["random_seed"])

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
    stage3 = select_stage(stages, 3)

    print(f"Device: {device}")
    print(f"GPUs available to PyTorch: {number_of_gpus}")
    for index, gpu_name in enumerate(gpu_names):
        print(f"  GPU {index}: {gpu_name}")
    print("Training mode: Stage 3 only")
    print(f"Fixed Stage 2 checkpoint: {STAGE2_EWC_CHECKPOINT}")
    print(f"Fixed batch size: {batch_size}")
    print(f"Fixed learning rate: {learning_rate:.8g}")
    print(f"Fixed weight decay: {weight_decay:.8g}")
    print(f"Fixed EWC lambda: {ewc_lambda:.8g}")
    print("Protected EWC stages: [1, 2]")

    config = {
        "training_mode": "stage3_fixed_final_training",
        "source_stage2_checkpoint": str(STAGE2_EWC_CHECKPOINT),
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "epochs_per_stage": epochs_per_stage,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "ewc_lambda": ewc_lambda,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "min_learning_rate": min_learning_rate,
        "random_seed": random_seed,
        "attention": True,
        "augmentations": fixed_config.get("augmentations", []),
        "fisher_data": (
            "fixed Stage 1 and Stage 2 Fishers from complete original "
            "training data"
        ),
        "protected_ewc_stages": [1, 2],
        "stages": [3],
        "stage_periods": {3: "2023-07 to 2024-06"},
        "normalization": fixed_config.get(
            "normalization",
            "0_to_1_scaling",
        ),
        "stage_directory": str(STAGE_DIRECTORY),
        "validation_csv": str(VALIDATION_CSV),
        "image_directory": str(IMAGE_DIRECTORY),
        "device": str(device),
        "number_of_gpus": number_of_gpus,
        "gpus": gpu_names,
        "multi_gpu_method": (
            "DataParallel" if number_of_gpus > 1 else "single_device"
        ),
        "pytorch_version": torch.__version__,
        "stage2_fisher_metadata": stage2_checkpoint[
            "stage2_fisher_metadata"
        ],
    }
    if "fisher_metadata" in stage2_checkpoint:
        config["stage1_fisher_metadata"] = stage2_checkpoint[
            "fisher_metadata"
        ]

    with ExperimentTracker(
        config=config,
        results_directory=RESULTS_DIRECTORY,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    ) as tracker:
        # Recreate the model and restore the exact selected Stage 2 weights.
        model = Attn_Net(
            im_size=image_size,
            num_classes=2,
            attention=True,
            init="kaimingUniform",
        ).to(device)
        model.load_state_dict(stage2_checkpoint["model_state_dict"])
        if number_of_gpus > 1:
            model = nn.DataParallel(model)
            print(f"Using DataParallel on {number_of_gpus} GPUs")

        criterion = nn.CrossEntropyLoss()
        ewc_history = stage2_checkpoint["ewc_history"]

        # Earlier-stage holdouts are evaluation-only. Their pre-training
        # scores establish baselines for measuring Stage 3 forgetting.
        stage1_holdout_loader = build_evaluation_loader(
            csv_file=stage1["holdout_file"],
            image_directory=IMAGE_DIRECTORY,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        stage2_holdout_loader = build_evaluation_loader(
            csv_file=stage2["holdout_file"],
            image_directory=IMAGE_DIRECTORY,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        stage3_loaders = build_stage_loaders(
            train_csv=stage3["train_file"],
            holdout_csv=stage3["holdout_file"],
            image_directory=IMAGE_DIRECTORY,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )

        print("\nEvaluating fixed pre-Stage-3 retention baselines...")
        baseline_results = evaluate_learned_stages(
            model,
            {
                1: stage1_holdout_loader,
                2: stage2_holdout_loader,
            },
            criterion,
            device,
        )
        tracker.log_holdouts(2, baseline_results)
        for stage_number in (1, 2):
            metrics = baseline_results[stage_number]
            print(
                f"Stage {stage_number} baseline | "
                f"CSS={metrics['css']:.4f} | "
                f"TSS={metrics['tss']:.4f} | "
                f"HSS={metrics['hss']:.4f}"
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

        print(f"\n{'=' * 60}\nTraining Stage 3\n{'=' * 60}")
        print(f"EWC lambda: {ewc_lambda:.8g}")

        for stage_epoch in range(1, epochs_per_stage + 1):
            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(
                model=model,
                data_loader=stage3_loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                ewc_history=ewc_history,
                ewc_lambda=ewc_lambda,
            )
            holdout_metrics = evaluate(
                model,
                stage3_loaders["holdout"],
                criterion,
                device,
            )
            epoch_seconds = time.perf_counter() - epoch_start

            tracker.log_epoch(
                global_epoch=stage_epoch,
                stage_number=3,
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
                    3,
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
                    f"Early stopping Stage 3 at epoch {stage_epoch}: "
                    f"holdout CSS did not improve by at least "
                    f"{early_stopping_min_delta:.4f} for "
                    f"{early_stopping_patience} epochs."
                )
                break

        if best_checkpoint is None:
            raise RuntimeError("No valid Stage 3 checkpoint was created.")

        restored_checkpoint = tracker.restore_best_stage_model(
            best_checkpoint,
            model,
            optimizer,
            device,
        )
        print(
            "Restored Stage 3 best checkpoint from epoch "
            f"{restored_checkpoint['stage_epoch']} "
            f"(holdout CSS={restored_checkpoint['holdout_css']:.4f})"
        )

        holdout_results = evaluate_learned_stages(
            model,
            {
                1: stage1_holdout_loader,
                2: stage2_holdout_loader,
                3: stage3_loaders["holdout"],
            },
            criterion,
            device,
        )
        tracker.log_holdouts(3, holdout_results)

        selection_metrics = {"completed_stage": 3}
        for stage_number in (1, 2):
            baseline_css = baseline_results[stage_number]["css"]
            retained_css = holdout_results[stage_number]["css"]
            forgetting, retention_ratio = retention_statistics(
                baseline_css,
                retained_css,
            )
            prefix = f"selection/stage{stage_number}"
            stage_metrics = {
                f"{prefix}_baseline_css": baseline_css,
                f"{prefix}_retained_css": retained_css,
                f"{prefix}_css_forgetting": forgetting,
                f"{prefix}_css_retention_ratio": retention_ratio,
            }
            selection_metrics.update(stage_metrics)
            for name, value in stage_metrics.items():
                tracker.run.summary[name] = value

            print(
                f"Stage {stage_number} retention | "
                f"baseline CSS={baseline_css:.4f} | "
                f"after Stage 3 CSS={retained_css:.4f} | "
                f"forgetting={forgetting:.4f} | "
                f"retention ratio={retention_ratio:.4f}"
            )

        tracker.run.log(selection_metrics)

        # Stage 3 is the final planned stage, so no new Fisher calculation is
        # required. The checkpoint retains the Stage 1 and Stage 2 histories
        # that were applied during this training run.
        checkpoint_path = tracker.save_stage(
            model=model,
            optimizer=optimizer,
            ewc_history=ewc_history,
            stage_number=3,
            loaders=stage3_loaders,
            fisher_seconds=0.0,
            best_stage_epoch=best_stage_epoch,
            best_holdout_css=best_holdout_css,
            stopping_epoch=stopping_epoch,
            stopped_early=stopped_early,
        )
        print(f"Stage 3 completed: {checkpoint_path}")

        print("\nEvaluating restored Stage 3 model on validation data...")
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
        print("\nStage 3 training completed.")
        print(f"Local results: {tracker.directory}")


if __name__ == "__main__":
    main()
