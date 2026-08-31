"""
metrics.py
─────────────────────────────────────────────────────────────────────────────
Calculates all evaluation metrics for the 6-class emotion classification task.

WHY THIS FILE EXISTS
────────────────────
All metric computation is isolated here so:
  1. Training, validation, and test loops call the same functions.
  2. Metrics are always computed identically — no subtle differences
     between how you measure val and test performance.
  3. Easy to verify: you can unit-test this file independently.

METRICS WE COMPUTE
───────────────────
Macro F1 (primary):
  Compute F1 per class, then take the unweighted mean.
  Treats all classes equally regardless of support.
  This is the correct primary metric for imbalanced multi-class problems.

Weighted Accuracy (WA):
  Total correct / total samples.
  What IEMOCAP papers before 2020 called "accuracy".
  Required for comparison with older papers.

Unweighted Accuracy (UA):
  Mean of per-class accuracy (equivalent to macro recall).
  What many IEMOCAP papers call "UA" — it is NOT the same as WA.
  Required for comparison with published results that report UA.
  UA = (recall_class1 + recall_class2 + ... + recall_class6) / 6

Weighted F1:
  F1 per class, weighted by class support.
  Reflects overall population performance; higher than Macro F1 for
  imbalanced datasets.

Per-class Precision, Recall, F1:
  Required for the per-class breakdown table (Table EX-5 in blueprint).
  Identifies which emotion classes are causing errors.

Confusion Matrix:
  A [6×6] numpy array.
  rows = true labels, columns = predicted labels.
  Used for Figure (heatmap) and Table EX-9 in blueprint.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from typing import List, Dict
from config import cfg, IDX_TO_LABEL, NUM_CLASSES


def compute_metrics(
    all_labels: List[int],   # true labels, integer 0–5
    all_preds:  List[int],   # predicted labels, integer 0–5
) -> Dict:
    """
    WHAT IT DOES
    ────────────
    Computes all evaluation metrics given lists of true and predicted labels.

    WHY IT IS NEEDED
    ────────────────
    Called at the end of every validation epoch and at test time.
    Returns a dictionary of all metrics that can be logged, compared, and
    saved to CSV.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    all_labels: list of integer true class labels (from the DataLoader)
    all_preds:  list of integer predicted class labels (from argmax(logits))
    Both lists must be the same length.

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    A dict with:
      "macro_f1"       → float: primary metric (equal weight per class)
      "weighted_f1"    → float: F1 weighted by class support
      "weighted_acc"   → float: standard accuracy (WA)
      "unweighted_acc" → float: mean per-class accuracy (UA)
      "precision_per_class" → dict: class_name → precision
      "recall_per_class"    → dict: class_name → recall
      "f1_per_class"        → dict: class_name → F1
      "confusion_matrix"    → np.ndarray [6, 6]
      "classification_report" → str: formatted sklearn report

    COMMON MISTAKES
    ───────────────
    - Calling compute_metrics() with LOGITS instead of PREDICTIONS.
      You must apply argmax before calling this function:
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
    - Computing metrics on GPU tensors (not converted to Python lists/numpy).
      All inputs must be Python lists or numpy arrays.
    - Using accuracy_score() as the primary metric for an imbalanced dataset.
      Always report Macro F1 as primary.
    """
    # Convert to numpy arrays for sklearn
    labels = np.array(all_labels)
    preds  = np.array(all_preds)

    # Guard: both arrays must be non-empty
    if len(labels) == 0:
        raise ValueError("compute_metrics received empty label list.")

    # ── Primary metrics ───────────────────────────────────────────────────────
    macro_f1    = f1_score(labels, preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    weighted_acc = accuracy_score(labels, preds)   # standard accuracy (WA)

    # Unweighted Accuracy (UA) = mean of per-class recall
    # sklearn recall_score with average="macro" computes this.
    unweighted_acc = recall_score(labels, preds, average="macro", zero_division=0)

    # ── Per-class metrics ─────────────────────────────────────────────────────
    # labels_list: [0, 1, 2, 3, 4, 5] — explicit list of all expected classes
    # This ensures classes absent from all_preds still appear in the output.
    labels_list = list(range(NUM_CLASSES))

    precision_per_class_arr = precision_score(
        labels, preds, labels=labels_list, average=None, zero_division=0
    )  # shape [6]
    recall_per_class_arr = recall_score(
        labels, preds, labels=labels_list, average=None, zero_division=0
    )  # shape [6]
    f1_per_class_arr = f1_score(
        labels, preds, labels=labels_list, average=None, zero_division=0
    )  # shape [6]

    # Convert arrays to dicts keyed by emotion name
    precision_per_class = {IDX_TO_LABEL[i]: float(precision_per_class_arr[i])
                           for i in range(NUM_CLASSES)}
    recall_per_class    = {IDX_TO_LABEL[i]: float(recall_per_class_arr[i])
                           for i in range(NUM_CLASSES)}
    f1_per_class        = {IDX_TO_LABEL[i]: float(f1_per_class_arr[i])
                           for i in range(NUM_CLASSES)}

    # ── Confusion matrix ──────────────────────────────────────────────────────
    conf_matrix = confusion_matrix(labels, preds, labels=labels_list)
    # Shape: [6, 6]. conf_matrix[i, j] = number of samples of class i
    # predicted as class j. Diagonal = correct. Off-diagonal = errors.

    # ── Full classification report (for logging) ──────────────────────────────
    target_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    cls_report = classification_report(
        labels, preds,
        labels=labels_list,
        target_names=target_names,
        zero_division=0
    )

    return {
        "macro_f1":             float(macro_f1),
        "weighted_f1":          float(weighted_f1),
        "weighted_acc":         float(weighted_acc),
        "unweighted_acc":       float(unweighted_acc),
        "precision_per_class":  precision_per_class,
        "recall_per_class":     recall_per_class,
        "f1_per_class":         f1_per_class,
        "confusion_matrix":     conf_matrix,
        "classification_report": cls_report,
    }


def format_metrics_for_log(metrics: dict, prefix: str = "") -> str:
    """
    WHAT IT DOES
    ────────────
    Returns a single-line string summary of the key metrics for logging.

    EXAMPLE OUTPUT
    ──────────────
    "[val] MacF1=0.6142 | WtF1=0.6683 | WA=68.41% | UA=64.87%"

    WHAT INPUT IT EXPECTS
    ─────────────────────
    metrics: the dict from compute_metrics()
    prefix:  a string like "val" or "test" prepended to the output
    """
    return (
        f"[{prefix}] "
        f"MacF1={metrics['macro_f1']:.4f} | "
        f"WtF1={metrics['weighted_f1']:.4f} | "
        f"WA={metrics['weighted_acc']*100:.2f}% | "
        f"UA={metrics['unweighted_acc']*100:.2f}%"
    )


def aggregate_fold_results(fold_results: List[Dict]) -> Dict:
    """
    WHAT IT DOES
    ────────────
    Aggregates results from all 5 LOSO folds into mean ± std.

    WHY IT IS NEEDED
    ────────────────
    The paper reports mean ± std across 5 folds (e.g. "0.704 ± 0.009").
    This function computes those statistics.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    fold_results: a list of 5 dicts, each the output of compute_metrics()
                  from one fold's test evaluation.

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    A dict with keys "macro_f1_mean", "macro_f1_std", "weighted_acc_mean", etc.
    for the 4 primary metrics.

    NOTE ON CONFUSION MATRIX AGGREGATION
    ──────────────────────────────────────
    The aggregate confusion matrix is the SUM of per-fold confusion matrices,
    normalised by row (shows average recall per class across all folds).
    """
    scalar_keys = ["macro_f1", "weighted_f1", "weighted_acc", "unweighted_acc"]
    agg = {}

    for key in scalar_keys:
        values = [fold[key] for fold in fold_results]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"]  = float(np.std(values, ddof=1))
        # ddof=1: Bessel's correction for sample standard deviation (N-1)
        # Use this for N=5 folds (small N). The paper should report this.

    # Aggregate per-class F1
    for emotion in cfg.emotion_classes:
        values = [fold["f1_per_class"][emotion] for fold in fold_results]
        agg[f"f1_{emotion}_mean"] = float(np.mean(values))
        agg[f"f1_{emotion}_std"]  = float(np.std(values, ddof=1))

    # Sum confusion matrices across folds
    conf_sum = sum(fold["confusion_matrix"] for fold in fold_results)
    # Normalise by row to get recall fractions
    row_sums = conf_sum.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1   # avoid division by zero for empty classes
    agg["confusion_matrix_normalised"] = conf_sum / row_sums

    return agg


def print_fold_summary(agg: Dict):
    """
    Prints a formatted summary table of aggregate metrics.

    EXAMPLE OUTPUT
    ──────────────
    ┌──────────────────────────────────────────────────────┐
    │              5-Fold LOSO Results Summary              │
    ├──────────────────────────────────────────────────────┤
    │  Macro F1       :  0.7041 ± 0.0092                   │
    │  Weighted F1    :  0.7534 ± 0.0080                   │
    │  Weighted Acc   :  76.43% ± 0.84%                    │
    │  Unweighted Acc :  73.12% ± 0.91%                    │
    ├──────────────────────────────────────────────────────┤
    │  Per-class F1:                                        │
    │    anger   : 0.831 ± 0.022                           │
    │    disgust : 0.667 ± 0.041                           │
    │    ...                                               │
    └──────────────────────────────────────────────────────┘
    """
    print("\n" + "=" * 56)
    print(f"{'5-Fold LOSO Results Summary':^56}")
    print("=" * 56)
    print(f"  Macro F1       : "
          f"{agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}")
    print(f"  Weighted F1    : "
          f"{agg['weighted_f1_mean']:.4f} ± {agg['weighted_f1_std']:.4f}")
    print(f"  Weighted Acc   : "
          f"{agg['weighted_acc_mean']*100:.2f}% ± {agg['weighted_acc_std']*100:.2f}%")
    print(f"  Unweighted Acc : "
          f"{agg['unweighted_acc_mean']*100:.2f}% ± {agg['unweighted_acc_std']*100:.2f}%")
    print("-" * 56)
    print("  Per-class F1:")
    for emotion in cfg.emotion_classes:
        m = agg[f"f1_{emotion}_mean"]
        s = agg[f"f1_{emotion}_std"]
        print(f"    {emotion:<12}: {m:.3f} ± {s:.3f}")
    print("=" * 56 + "\n")
