from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = prediction == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if not selected.any():
            continue
        score += selected.mean() * abs(correctness[selected].mean() - confidence[selected].mean())
    return float(score)


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, object]:
    prediction = probabilities.argmax(axis=1)
    num_classes = probabilities.shape[1]
    class_labels = np.arange(num_classes)
    per_class_recall = recall_score(
        labels, prediction, labels=class_labels, average=None, zero_division=0
    )
    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(np.mean(per_class_recall)),
        "macro_f1": float(
            f1_score(labels, prediction, labels=class_labels, average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                labels, prediction, labels=class_labels, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(np.mean(per_class_recall)),
        "ece": expected_calibration_error(probabilities, labels),
        "nll": float(log_loss(labels, probabilities, labels=np.arange(num_classes))),
        "brier": multiclass_brier(probabilities, labels),
        "per_class_recall": per_class_recall.tolist(),
    }
    try:
        one_hot = np.eye(num_classes)[labels]
        metrics["macro_auroc"] = float(
            roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        metrics["macro_auroc"] = float("nan")
    return metrics


def continual_metrics(stage_accuracy: np.ndarray) -> Dict[str, float]:
    """Compute standard class/stage-incremental metrics from a T x T matrix.

    Row t stores accuracies after learning stage t; column j is the test accuracy on stage j.
    """
    matrix = np.asarray(stage_accuracy, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return {"average_accuracy": float("nan"), "forgetting": float("nan"), "bwt": float("nan")}
    final_row = matrix[-1]
    valid = np.isfinite(final_row)
    average_accuracy = float(np.nanmean(final_row))
    forgetting_values: List[float] = []
    bwt_values: List[float] = []
    for task in range(matrix.shape[1] - 1):
        trajectory = matrix[task:, task]
        trajectory = trajectory[np.isfinite(trajectory)]
        if len(trajectory) == 0 or not np.isfinite(final_row[task]):
            continue
        forgetting_values.append(float(np.max(trajectory) - final_row[task]))
        diagonal = matrix[task, task]
        if np.isfinite(diagonal):
            bwt_values.append(float(final_row[task] - diagonal))
    return {
        "average_accuracy": average_accuracy,
        "forgetting": float(np.mean(forgetting_values)) if forgetting_values else 0.0,
        "bwt": float(np.mean(bwt_values)) if bwt_values else 0.0,
    }


def bootstrap_ci(values: Sequence[float], confidence: float = 0.95, seed: int = 2026) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def paired_significance(proposed: Sequence[float], baseline: Sequence[float]) -> Dict[str, float]:
    proposed_array = np.asarray(proposed, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if len(proposed_array) < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "effect_size": float("nan")}
    statistic, p_value = stats.ttest_rel(proposed_array, baseline_array)
    differences = proposed_array - baseline_array
    effect = differences.mean() / (differences.std(ddof=1) + 1e-12)
    return {"statistic": float(statistic), "p_value": float(p_value), "effect_size": float(effect)}
