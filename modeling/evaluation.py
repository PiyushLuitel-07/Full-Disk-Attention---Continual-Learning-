"""Classification metrics used by the training pipelines."""

import torch


def calculate_classification_metrics(predictions, targets):
    """Return all binary FL/NF metrics in one dictionary."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have equal lengths.")

    if not targets:
        raise ValueError("Cannot calculate metrics from empty inputs.")

    tn = fp = fn = tp = 0

    for prediction, target in zip(predictions, targets):
        prediction = int(prediction)
        target = int(target)

        if prediction not in (0, 1) or target not in (0, 1):
            raise ValueError("Metrics require binary values 0 and 1.")

        if target == 1 and prediction == 1:
            tp += 1
        elif target == 0 and prediction == 0:
            tn += 1
        elif target == 0 and prediction == 1:
            fp += 1
        else:
            fn += 1

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )
    tss = recall - false_positive_rate

    actual_fl = tp + fn
    actual_nf = tn + fp
    hss_denominator = (
        actual_fl * (fn + tn)
        + (tp + fp) * actual_nf
    )
    hss = (
        2 * (tp * tn - fn * fp) / hss_denominator
        if hss_denominator
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tss": float(tss),
        "hss": float(hss),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate(model, data_loader, criterion, device):
    """Evaluate a model without updating its parameters."""
    model.eval()

    total_loss = 0.0
    number_of_images = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for images, batch_targets in data_loader:
            images = images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            scores = model(images)[0]
            loss = criterion(scores, batch_targets)
            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            number_of_images += batch_size
            predictions.extend(scores.argmax(dim=1).cpu().tolist())
            targets.extend(batch_targets.cpu().tolist())

    metrics = calculate_classification_metrics(predictions, targets)
    metrics["loss"] = total_loss / number_of_images

    return metrics


def sklearn_Compatible_preds_and_targets(
    model_prediction_list,
    model_target_list,
):
    """Compatibility wrapper for the original training code."""
    predictions = []
    targets = []

    for batch in model_prediction_list:
        predictions.extend(
            batch.detach().cpu().reshape(-1).tolist()
        )

    for batch in model_target_list:
        targets.extend(
            batch.detach().cpu().reshape(-1).tolist()
        )

    metrics = calculate_classification_metrics(
        predictions,
        targets,
    )

    print(
        "TP:", metrics["tp"],
        "FP:", metrics["fp"],
        "TN:", metrics["tn"],
        "FN:", metrics["fn"],
    )

    return metrics["tss"], metrics["hss"]


def accuracy_score(prediction, target):
    """Compatibility wrapper returning the original TSS/HSS tuple."""
    metrics = calculate_classification_metrics(
        prediction,
        target,
    )

    print(
        "TP:", metrics["tp"],
        "FP:", metrics["fp"],
        "TN:", metrics["tn"],
        "FN:", metrics["fn"],
    )

    return metrics["tss"], metrics["hss"]
