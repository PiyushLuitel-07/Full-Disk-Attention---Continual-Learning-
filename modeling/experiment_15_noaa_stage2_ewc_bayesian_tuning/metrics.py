"""Binary flare metrics and evaluation for Experiment 15."""
import torch


def classification_metrics(predictions, targets):
    if len(predictions) != len(targets) or not targets: raise ValueError("Metrics require equal non-empty inputs")
    tn = fp = fn = tp = 0
    for prediction, target in zip(predictions, targets):
        prediction, target = int(prediction), int(target)
        if prediction not in (0, 1) or target not in (0, 1): raise ValueError("Metrics require binary values")
        if target == prediction == 1: tp += 1
        elif target == prediction == 0: tn += 1
        elif target == 0: fp += 1
        else: fn += 1
    total, actual_fl, actual_nf = tp + tn + fp + fn, tp + fn, tn + fp
    accuracy = (tp + tn) / total; precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / actual_fl if actual_fl else 0.0; fpr = fp / actual_nf if actual_nf else 0.0
    specificity, fnr = (tn / actual_nf if actual_nf else 0.0), (fn / actual_fl if actual_fl else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tss = recall - fpr; denominator = actual_fl * (fn + tn) + (tp + fp) * actual_nf
    hss = 2 * (tp * tn - fn * fp) / denominator if denominator else 0.0
    return {"accuracy": accuracy, "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
        "tss": tss, "hss": hss, "css": (max(0.0, tss) * max(0.0, hss)) ** 0.5,
        "false_positive_rate": fpr, "false_negative_rate": fnr,
        "actual_fl_rate": actual_fl / total, "predicted_fl_rate": (tp + fp) / total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluate(model, loader, criterion, device):
    model.eval(); loss = count = 0; predictions, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images); batch = len(images)
            loss += criterion(logits, labels).item() * batch; count += batch
            predictions.extend(logits.argmax(1).cpu().tolist()); targets.extend(labels.cpu().tolist())
    metrics = classification_metrics(predictions, targets); metrics["loss"] = loss / count
    return metrics
