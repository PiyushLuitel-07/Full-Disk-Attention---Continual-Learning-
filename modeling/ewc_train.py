from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD

from attention_model import Attn_Net
from dataloader import build_stage_loaders, discover_stages


# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGE_DIRECTORY = (
    PROJECT_ROOT
    / "data_labeling"
    / "data_labels"
    / "continual_stages_simplified_labels"
)

IMAGE_DIRECTORY = (
    PROJECT_ROOT
    / "downloaded_data"
    / "hmi_jpgs"
)

IMAGE_SIZE = 256
BATCH_SIZE = 128
NUM_WORKERS = 8
EPOCHS = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


# ---------------------------------------------------------------------
# 2. TRAIN FOR ONE EPOCH
# ---------------------------------------------------------------------

def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, targets in data_loader:
        images = images.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        # attention_model.py returns:
        # [scores, attention1, attention2, attention3]
        scores = model(images)[0]

        loss = criterion(scores, targets)

        loss.backward()
        optimizer.step()

        predictions = scores.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (
            predictions == targets
        ).sum().item()
        total_images += images.size(0)

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images

    return average_loss, accuracy


# ---------------------------------------------------------------------
# 3. EVALUATE THE HOLDOUT
# ---------------------------------------------------------------------

def evaluate(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(
                device,
                non_blocking=True,
            )
            targets = targets.to(
                device,
                non_blocking=True,
            )

            scores = model(images)[0]
            loss = criterion(scores, targets)

            predictions = scores.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (
                predictions == targets
            ).sum().item()
            total_images += images.size(0)

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images

    return average_loss, accuracy


# ---------------------------------------------------------------------
# 4. RUN THE STAGE 1 TEST
# ---------------------------------------------------------------------

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # Automatically discover all available stages.
    stages = discover_stages(STAGE_DIRECTORY)

    print(
        f"Stages discovered: "
        f"{[stage['number'] for stage in stages]}"
    )

    # For now, test only Stage 1.
    stage = stages[0]

    loaders = build_stage_loaders(
        train_csv=stage["train_file"],
        holdout_csv=stage["holdout_file"],
        image_directory=IMAGE_DIRECTORY,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = Attn_Net(
        im_size=IMAGE_SIZE,
        num_classes=2,
        attention=True,
        init="kaimingUniform",
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("\nTraining Stage 1")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        holdout_loss, holdout_accuracy = evaluate(
            model=model,
            data_loader=loaders["holdout"],
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss={train_loss:.4f} | "
            f"train accuracy={train_accuracy:.4f} | "
            f"holdout loss={holdout_loss:.4f} | "
            f"holdout accuracy={holdout_accuracy:.4f}"
        )

    print("\nStage 1 training test completed.")


if __name__ == "__main__":
    main()