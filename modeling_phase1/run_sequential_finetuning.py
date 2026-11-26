"""Sequential Fine-Tuning Baseline across Stage 2 and Stage 3.

This script trains naive sequential fine-tuning (ewc_lambda = 0.0):
1. Starts from the exact Stage 1 winner (stage1_ewc_ready.pt).
2. Fine-tunes on Stage 2 until early stopping.
3. Evaluates Stage 1 retention, Stage 2 skill, validation, and test performance.
4. Takes the resulting Stage 2 checkpoint and immediately fine-tunes on Stage 3.
5. Evaluates Stage 1 and Stage 2 retention, Stage 3 skill, validation, and test performance.
6. Saves comprehensive comparison tables and metrics for the final report.
"""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
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
    PROJECT_ROOT,
    RESULTS_DIRECTORY,
    STAGE_DIRECTORY,
    VALIDATION_CSV,
    evaluate_learned_stages,
    train_one_epoch,
)


STAGE1_CHECKPOINT = (
    RESULTS_DIRECTORY
    / "ewc_continual_20261010_043758"
    / "checkpoints"
    / "stage1_ewc_ready.pt"
)

TEST_CSV = (
    PROJECT_ROOT
    / "data_labeling/data_labels/future_evaluation_labels/revised_chronological_2024_2026/test_2025_2026_july.csv"
)

OUT_DIR = RESULTS_DIRECTORY / "noaa_sequential_finetuning_baseline"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_metrics(m):
    return (
        f"CSS={m.get('css', 0.0):.4f} | "
        f"TSS={m.get('tss', 0.0):.4f} | "
        f"HSS={m.get('hss', 0.0):.4f} | "
        f"Acc={m.get('accuracy', 0.0):.4f} | "
        f"F1={m.get('f1', 0.0):.4f} | "
        f"Loss={m.get('loss', 0.0):.4f}"
    )


def train_single_stage(
    stage_number,
    model,
    train_loader,
    holdout_loader,
    device,
    criterion,
    lr,
    weight_decay,
    epochs,
    patience,
    min_delta,
    scheduler_factor,
    scheduler_patience,
):
    print(f"\n{'='*70}\nStarting Sequential Fine-Tuning on Stage {stage_number} (lambda=0.0)\n{'='*70}")
    print(f"Initial LR: {lr:.8g} | Weight Decay: {weight_decay:.8g} | Max Epochs: {epochs}")

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=min_delta,
        threshold_mode="abs",
        min_lr=1e-7,
    )

    best_css = -1.0
    best_epoch = 0
    best_state_dict = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            ewc_history=[],
            ewc_lambda=0.0,
        )
        train_sec = time.perf_counter() - t0

        holdout_m = evaluate(model, holdout_loader, criterion, device)
        cur_css = holdout_m["css"]
        cur_lr = optimizer.param_groups[0]["lr"]

        improved = (cur_css - best_css) >= min_delta
        if improved:
            best_css = cur_css
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            no_improve = 0
            mark = "★ BEST"
        else:
            no_improve += 1
            mark = f"({no_improve}/{patience})"

        print(
            f"Epoch {epoch:02d} | Train Loss={train_loss['loss']:.4f} ({train_sec:.1f}s) | "
            f"Holdout CSS={cur_css:.4f} (TSS={holdout_m['tss']:.4f}, HSS={holdout_m['hss']:.4f}) | "
            f"LR={cur_lr:.2e} | {mark}"
        )

        scheduler.step(cur_css)

        if no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Restoring epoch {best_epoch} (CSS={best_css:.4f})")
            break

    model.load_state_dict(best_state_dict)
    return model, best_epoch, best_css


def main():
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")

    # 1. Discover data stages
    stages = discover_stages(STAGE_DIRECTORY)
    stage_map = {s["number"]: s for s in stages}
    stage1_info = stage_map[1]
    stage2_info = stage_map[2]
    stage3_info = stage_map[3]

    # 2. Load winning Stage 1 checkpoint
    print(f"\nLoading Stage 1 checkpoint: {STAGE1_CHECKPOINT}")
    ckpt = torch.load(STAGE1_CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    image_size = int(cfg["image_size"])
    batch_size = int(cfg["batch_size"])
    stage1_lr = float(cfg["learning_rate"])
    weight_decay = float(cfg["weight_decay"])
    epochs = int(cfg.get("epochs_per_stage", 100))
    patience = int(cfg.get("early_stopping_patience", 12))
    min_delta = float(cfg.get("early_stopping_min_delta", 0.001))
    scheduler_factor = float(cfg.get("scheduler_factor", 0.3))
    scheduler_patience = int(cfg.get("scheduler_patience", 3))

    criterion = nn.CrossEntropyLoss()

    # Load Stage 1 baseline model
    model = Attn_Net(im_size=image_size, num_classes=2, attention=True, init="kaimingUniform").to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Loaders
    stage1_holdout_loader = build_evaluation_loader(
        stage1_info["holdout_file"], IMAGE_DIRECTORY, batch_size, image_size, NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    val_loader = build_evaluation_loader(
        VALIDATION_CSV, IMAGE_DIRECTORY, batch_size, image_size, NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    test_loader = None
    if TEST_CSV.exists():
        test_loader = build_evaluation_loader(
            TEST_CSV, IMAGE_DIRECTORY, batch_size, image_size, NUM_WORKERS, pin_memory=(device.type == "cuda")
        )

    # Initial baseline evaluation
    print("\nEvaluating initial Stage 1 checkpoint...")
    s1_baseline_m = evaluate(model, stage1_holdout_loader, criterion, device)
    s1_baseline_css = s1_baseline_m["css"]
    print(f"Stage 1 Baseline Holdout: {format_metrics(s1_baseline_m)}")

    summary_records = []

    # =========================================================================
    # STAGE 2 SEQUENTIAL FINE-TUNING
    # =========================================================================
    s2_loaders = build_stage_loaders(
        train_csv=stage2_info["train_file"],
        holdout_csv=stage2_info["holdout_file"],
        image_directory=IMAGE_DIRECTORY,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model, s2_best_epoch, s2_best_css = train_single_stage(
        stage_number=2,
        model=model,
        train_loader=s2_loaders["train"],
        holdout_loader=s2_loaders["holdout"],
        device=device,
        criterion=criterion,
        lr=stage1_lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
    )

    # Save Stage 2 checkpoint
    s2_ckpt_path = OUT_DIR / "stage2_finetune_best.pt"
    torch.save(
        {
            "stage": 2,
            "best_epoch": s2_best_epoch,
            "best_holdout_css": s2_best_css,
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        s2_ckpt_path,
    )
    print(f"\nSaved Stage 2 Fine-Tuned checkpoint: {s2_ckpt_path}")

    # Evaluate Stage 2 model across all splits
    print("\n--- Evaluating Stage 2 Fine-Tuned Model ---")
    s2_on_s1 = evaluate(model, stage1_holdout_loader, criterion, device)
    s2_on_s2 = evaluate(model, s2_loaders["holdout"], criterion, device)
    s2_on_val = evaluate(model, val_loader, criterion, device)
    s2_on_test = evaluate(model, test_loader, criterion, device) if test_loader else {}

    s1_retention_after_s2 = s2_on_s1["css"] / s1_baseline_css if s1_baseline_css > 0 else 0.0
    s1_forgetting_after_s2 = s1_baseline_css - s2_on_s1["css"]

    print(f"Stage 1 Holdout Retention: CSS={s2_on_s1['css']:.4f} | Retention Ratio={s1_retention_after_s2*100:.2f}% | Forgetting={s1_forgetting_after_s2:.4f}")
    print(f"Stage 2 Holdout Skill:     {format_metrics(s2_on_s2)}")
    print(f"Validation (2024-2025):    {format_metrics(s2_on_val)}")
    if s2_on_test:
        print(f"Test (2025-2026):          {format_metrics(s2_on_test)}")

    summary_records.append({
        "Model": "Sequential Fine-Tuning (Stage 2)",
        "Stage1_CSS": s2_on_s1["css"],
        "Stage1_Retention_Pct": s1_retention_after_s2 * 100,
        "Stage1_Forgetting": s1_forgetting_after_s2,
        "Stage2_CSS": s2_on_s2["css"],
        "Stage3_CSS": None,
        "Val_CSS": s2_on_val["css"],
        "Test_CSS": s2_on_test.get("css"),
    })

    # =========================================================================
    # STAGE 3 SEQUENTIAL FINE-TUNING
    # =========================================================================
    s3_loaders = build_stage_loaders(
        train_csv=stage3_info["train_file"],
        holdout_csv=stage3_info["holdout_file"],
        image_directory=IMAGE_DIRECTORY,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # Use stability learning rate for Stage 3: stage1_lr * 0.3
    s3_lr = stage1_lr * scheduler_factor

    model, s3_best_epoch, s3_best_css = train_single_stage(
        stage_number=3,
        model=model,
        train_loader=s3_loaders["train"],
        holdout_loader=s3_loaders["holdout"],
        device=device,
        criterion=criterion,
        lr=s3_lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
    )

    # Save Stage 3 checkpoint
    s3_ckpt_path = OUT_DIR / "stage3_finetune_best.pt"
    torch.save(
        {
            "stage": 3,
            "best_epoch": s3_best_epoch,
            "best_holdout_css": s3_best_css,
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        s3_ckpt_path,
    )
    print(f"\nSaved Stage 3 Fine-Tuned checkpoint: {s3_ckpt_path}")

    # Evaluate Stage 3 model across all splits
    print("\n--- Evaluating Stage 3 Fine-Tuned Model ---")
    s3_on_s1 = evaluate(model, stage1_holdout_loader, criterion, device)
    s3_on_s2 = evaluate(model, s2_loaders["holdout"], criterion, device)
    s3_on_s3 = evaluate(model, s3_loaders["holdout"], criterion, device)
    s3_on_val = evaluate(model, val_loader, criterion, device)
    s3_on_test = evaluate(model, test_loader, criterion, device) if test_loader else {}

    s1_retention_after_s3 = s3_on_s1["css"] / s1_baseline_css if s1_baseline_css > 0 else 0.0
    s1_forgetting_after_s3 = s1_baseline_css - s3_on_s1["css"]

    print(f"Stage 1 Holdout Retention: CSS={s3_on_s1['css']:.4f} | Retention Ratio={s1_retention_after_s3*100:.2f}% | Forgetting={s1_forgetting_after_s3:.4f}")
    print(f"Stage 2 Holdout Retention: CSS={s3_on_s2['css']:.4f}")
    print(f"Stage 3 Holdout Skill:     {format_metrics(s3_on_s3)}")
    print(f"Validation (2024-2025):    {format_metrics(s3_on_val)}")
    if s3_on_test:
        print(f"Test (2025-2026):          {format_metrics(s3_on_test)}")

    summary_records.append({
        "Model": "Sequential Fine-Tuning (Stage 3)",
        "Stage1_CSS": s3_on_s1["css"],
        "Stage1_Retention_Pct": s1_retention_after_s3 * 100,
        "Stage1_Forgetting": s1_forgetting_after_s3,
        "Stage2_CSS": s3_on_s2["css"],
        "Stage3_CSS": s3_on_s3["css"],
        "Val_CSS": s3_on_val["css"],
        "Test_CSS": s3_on_test.get("css"),
    })

    # Save summary files
    out_json = OUT_DIR / "summary_metrics.json"
    with open(out_json, "w") as f:
        json.dump(summary_records, f, indent=2)

    print("\n" + "="*80)
    print("SEQUENTIAL FINE-TUNING COMPLETED SUCCESSFULLY!")
    print(f"All checkpoints and summary saved to: {OUT_DIR}")
    print("="*80)
    for rec in summary_records:
        print(rec)


if __name__ == "__main__":
    main()
