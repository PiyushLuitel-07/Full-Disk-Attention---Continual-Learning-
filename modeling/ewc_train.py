"""Train the attention model sequentially with EWC."""

import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD

from attention_model import Attn_Net
from dataloader import build_stage_loaders, discover_stages
from evaluation import calculate_classification_metrics, evaluate
from ewc import calculate_fisher, ewc_penalty, save_parameters
from experiment_tracker import ExperimentTracker


# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIRECTORY = (
    PROJECT_ROOT
    / "data_labeling"
    / "data_labels"
    / "continual_stages_2010_2024_revised_chronological"
)
IMAGE_DIRECTORY = PROJECT_ROOT / "downloaded_data" / "hmi_jpgs"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"

WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = (
    "Full disk attention solar flare prediction "
    "with continual learning"
)

IMAGE_SIZE = 256
BATCH_SIZE = 128
NUM_WORKERS = 8
EPOCHS_PER_STAGE = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EWC_LAMBDA = 1.0
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 0.002
RANDOM_SEED = 42


# ---------------------------------------------------------------------
# 2. TRAIN ONE EPOCH
# ---------------------------------------------------------------------

def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    ewc_history,
    ewc_lambda,
):
    model.train()

    loss_sums = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "ewc_loss": 0.0,
    }
    predictions = []
    targets = []
    number_of_images = 0

    for images, batch_targets in data_loader:
        images = images.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        scores = model(images)[0]
        classification_loss = criterion(scores, batch_targets)
        ewc_loss = (
            ewc_lambda / 2.0
        ) * ewc_penalty(model, ewc_history)
        loss = classification_loss + ewc_loss

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_sums["loss"] += loss.item() * batch_size
        loss_sums["classification_loss"] += (
            classification_loss.item() * batch_size
        )
        loss_sums["ewc_loss"] += ewc_loss.item() * batch_size
        number_of_images += batch_size

        predictions.extend(scores.argmax(dim=1).detach().cpu().tolist())
        targets.extend(batch_targets.detach().cpu().tolist())

    metrics = calculate_classification_metrics(predictions, targets)

    for name, value in loss_sums.items():
        metrics[name] = value / number_of_images

    return metrics


def evaluate_learned_stages(
    model,
    holdout_loaders,
    criterion,
    device,
):
    """Evaluate every holdout encountered so far."""
    print("\nHistorical holdout evaluation")
    results = {}

    for stage_number, holdout_loader in holdout_loaders.items():
        metrics = evaluate(model, holdout_loader, criterion, device)
        results[stage_number] = metrics

        print(
            f"Stage {stage_number} holdout | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"TSS={metrics['tss']:.4f} | "
            f"HSS={metrics['hss']:.4f} | "
            f"CSS={metrics['css']:.4f}"
        )

    return results


# ---------------------------------------------------------------------
# 4. COMPLETE CONTINUAL-LEARNING PIPELINE
# ---------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_names = (
        [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else ["CPU"]
    )
    number_of_gpus = torch.cuda.device_count()

    stages = discover_stages(STAGE_DIRECTORY)
    stage_numbers = [stage["number"] for stage in stages]

    print(f"Device: {device}")
    print(f"GPUs available to PyTorch: {number_of_gpus}")
    for index, gpu_name in enumerate(gpu_names):
        print(f"  GPU {index}: {gpu_name}")
    print(f"Stages discovered: {stage_numbers}")

    config = {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "epochs_per_stage": EPOCHS_PER_STAGE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "ewc_lambda": EWC_LAMBDA,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "random_seed": RANDOM_SEED,
        "attention": True,
        "augmentations": [
            "original",
            "horizontal_flip",
            "vertical_flip",
            "rotation",
            "disk_only_polarity_inversion",
        ],
        "fisher_data": "complete original stage training data",
        "stages": stage_numbers,
        "stage_periods": {
            1: "2010-12 to 2017-12",
            2: "2018-01 to 2023-06",
            3: "2023-06 to 2024-06",
        },
        "normalization": "0_to_1_scaling",
        "stage_directory": str(STAGE_DIRECTORY),
        "image_directory": str(IMAGE_DIRECTORY),
        "device": str(device),
        "number_of_gpus": number_of_gpus,
        "gpus": gpu_names,
        "multi_gpu_method": (
            "DataParallel" if number_of_gpus > 1 else "single_device"
        ),
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
        ewc_lambda = float(tracker.config["ewc_lambda"])
        early_stopping_patience = int(
            tracker.config["early_stopping_patience"]
        )
        early_stopping_min_delta = float(
            tracker.config["early_stopping_min_delta"]
        )

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
        ewc_history = []
        holdout_loaders = {}
        global_epoch = 0

        for stage in stages:
            stage_number = stage["number"]
            print(f"\n{'=' * 60}")
            print(f"Training Stage {stage_number}")
            print(f"{'=' * 60}")

            loaders = build_stage_loaders(
                train_csv=stage["train_file"],
                holdout_csv=stage["holdout_file"],
                image_directory=IMAGE_DIRECTORY,
                batch_size=batch_size,
                image_size=IMAGE_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=device.type == "cuda",
            )
            holdout_loaders[stage_number] = loaders["holdout"]

            optimizer = SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            best_holdout_css = float("-inf")
            best_stage_epoch = None
            best_checkpoint = None
            epochs_without_improvement = 0
            stopped_early = False
            stopping_epoch = epochs_per_stage

            for stage_epoch in range(1, epochs_per_stage + 1):
                global_epoch += 1
                epoch_start = time.perf_counter()

                train_metrics = train_one_epoch(
                    model,
                    loaders["train"],
                    criterion,
                    optimizer,
                    device,
                    ewc_history,
                    ewc_lambda,
                )
                holdout_metrics = evaluate(
                    model,
                    loaders["holdout"],
                    criterion,
                    device,
                )
                epoch_seconds = time.perf_counter() - epoch_start

                tracker.log_epoch(
                    global_epoch,
                    stage_number,
                    stage_epoch,
                    train_metrics,
                    holdout_metrics,
                    optimizer.param_groups[0]["lr"],
                    epoch_seconds,
                )

                if (
                    holdout_metrics["css"]
                    > best_holdout_css + early_stopping_min_delta
                ):
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

                print(
                    f"Epoch {stage_epoch}/{epochs_per_stage} | "
                    f"loss={train_metrics['loss']:.4f} | "
                    f"EWC={train_metrics['ewc_loss']:.4f} | "
                    f"holdout CSS={holdout_metrics['css']:.4f} | "
                    f"holdout HSS={holdout_metrics['hss']:.4f} | "
                    f"holdout TSS={holdout_metrics['tss']:.4f} | "
                    f"time={epoch_seconds:.1f}s"
                )

                if (
                    epochs_without_improvement
                    >= early_stopping_patience
                ):
                    stopped_early = True
                    stopping_epoch = stage_epoch
                    print(
                        f"Early stopping Stage {stage_number} at epoch "
                        f"{stage_epoch}: holdout CSS did not improve by "
                        f"at least {early_stopping_min_delta:.4f} for "
                        f"{early_stopping_patience} epochs."
                    )
                    break

            restored_checkpoint = tracker.restore_best_stage_model(
                best_checkpoint,
                model,
                optimizer,
                device,
            )
            print(
                f"Restored Stage {stage_number} best checkpoint from "
                f"epoch {restored_checkpoint['stage_epoch']} "
                f"(holdout CSS={restored_checkpoint['holdout_css']:.4f})"
            )

            holdout_results = evaluate_learned_stages(
                model,
                holdout_loaders,
                criterion,
                device,
            )
            tracker.log_holdouts(stage_number, holdout_results)

            print(f"\nCalculating Stage {stage_number} Fisher information...")
            fisher_start = time.perf_counter()
            fisher = calculate_fisher(model, loaders["fisher"], device)
            fisher_seconds = time.perf_counter() - fisher_start

            ewc_history.append(
                {
                    "stage": stage_number,
                    "fisher": fisher,
                    "parameters": save_parameters(model),
                }
            )

            checkpoint = tracker.save_stage(
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
            )
            print(f"Stage {stage_number} completed: {checkpoint}")

        print("\nAll continual-learning stages completed.")
        print(f"Local results: {tracker.directory}")


if __name__ == "__main__":
    main()
