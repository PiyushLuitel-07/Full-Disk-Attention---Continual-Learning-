"""Binary flare-classification metrics for Experiment 19."""


def roc_auc(scores, targets):
    """Compute binary ROC-AUC using average ranks, including tied scores."""
    positives = sum(targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        return float("nan")

    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    positive_rank_sum = sum(rank for rank, target in zip(ranks, targets) if target == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(scores, targets):
    """Compute binary average precision, grouping examples with tied scores."""
    positives = sum(targets)
    if not positives:
        return float("nan")
    ordered = sorted(zip(scores, targets), reverse=True)
    true_positives = predicted_positives = 0
    previous_recall = result = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        predicted_positives += end - start
        true_positives += sum(target for _, target in ordered[start:end])
        recall = true_positives / positives
        precision = true_positives / predicted_positives
        result += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return result


def classification_metrics(predictions, targets, scores=None):
    if len(predictions) != len(targets) or not targets:
        raise ValueError("Predictions and nonempty targets must have equal lengths")

    tn = fp = fn = tp = 0
    for prediction, target in zip(predictions, targets):
        if target == 1 and prediction == 1:
            tp += 1
        elif target == 0 and prediction == 0:
            tn += 1
        elif target == 0 and prediction == 1:
            fp += 1
        else:
            fn += 1

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    tss = recall - fpr
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fn * fp) / denominator if denominator else 0.0
    css = (max(0.0, tss) * max(0.0, hss)) ** 0.5
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tss": tss,
        "hss": hss,
        "css": css,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "predicted_fl_rate": (tp + fp) / total,
    }
    if scores is not None:
        if len(scores) != len(targets):
            raise ValueError("Scores and targets must have equal lengths")
        metrics.update({
            "roc_auc": roc_auc(scores, targets),
            "average_precision": average_precision(scores, targets),
        })
    return metrics
