"""Dependency-free binary metrics with safe one-class behavior."""

from __future__ import annotations

import math
from typing import Iterable


def binary_metrics(
    targets: Iterable[int], predictions: Iterable[int]
) -> dict[str, int | float]:
    targets = list(targets)
    predictions = list(predictions)
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Targets and predictions must have the same non-zero length")

    tn = fp = fn = tp = 0
    for target, prediction in zip(targets, predictions):
        if target not in (0, 1) or prediction not in (0, 1):
            raise ValueError("Binary metrics accept only labels 0 and 1")
        if target == 0 and prediction == 0:
            tn += 1
        elif target == 0 and prediction == 1:
            fp += 1
        elif target == 1 and prediction == 0:
            fn += 1
        else:
            tp += 1

    positives, negatives = tp + fn, tn + fp
    recall = tp / positives if positives else math.nan
    fpr = fp / negatives if negatives else math.nan
    precision = tp / (tp + fp) if tp + fp else math.nan
    tss = recall - fpr if positives and negatives else math.nan
    hss_denominator = positives * (fn + tn) + (tp + fp) * negatives
    hss = (
        2.0 * (tp * tn - fn * fp) / hss_denominator
        if hss_denominator
        else math.nan
    )

    return {
        "n": len(targets),
        "n_fl": positives,
        "n_nf": negatives,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(targets),
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "tss": tss,
        "hss": hss,
    }


def predictions_from_probabilities(
    probabilities: Iterable[float], threshold: float
) -> list[int]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    return [int(probability >= threshold) for probability in probabilities]

