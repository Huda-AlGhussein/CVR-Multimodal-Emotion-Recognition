"""
eval_orchestrator.py
──────────────────────────────────────────────────────────────────────────────
Step 2: Orchestrator (Agent 6) evaluation on the full fold-1 test set.

Derives all metrics from fold1_test_agent_eval.csv — no model inference
re-run. The SupervisorAgent's decision for each utterance is fully determined
by AlertAgent.graded_agreement applied to the pre-computed pred_retrieval_agreement
(computed here from neighbor_labels + predicted, identical to plot_roc_pr.py).

Alert rule: alert fires when pred_retrieval_agreement < ALERT_THRESHOLD (0.6)
            i.e. fewer than 3/5 retrieved neighbors agree with the prediction.

Metrics reported
────────────────
  total_utterances     : rows in the test CSV (session 1, fold 1)
  alerts_fired_pct     : % of utterances where the alert fired
  alert_tp_pct         : % of all utterances where alert fired AND correct==False
                         (alert caught a real misclassification)
  alert_fp_pct         : % of all utterances where alert fired AND correct==True
                         (alert fired on a correct prediction — false alarm)

  Also reported as fractions of alert-positive utterances:
  alert_precision      : TP / (TP + FP)  = precision of the alert
  alert_recall         : TP / (TP + FN)  = recall over misclassifications

Output CSV columns
──────────────────
  fold1_orchestrator_eval.csv:
    utterance_id, ground_truth, predicted, correct,
    alert_fired, prediction_agreement, majority_neighbor_label

  fold1_orchestrator_eval_extended.csv (same columns, plus CVR context):
    ... + utterance_timecode, flight_phase, model_version
    utterance_timecode : None — placeholder pending CVR audio timestamps
    flight_phase        : None — placeholder pending FDR data
    model_version        : populated now (see MODEL_VERSION below)
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
ALERT_THRESHOLD = 0.6   # pred_retrieval_agreement must be >= this to suppress alert

BASE     = os.path.dirname(__file__)
TEST_CSV = os.path.join(BASE, "outputs", "fold1_test_agent_eval.csv")
OUT_CSV  = os.path.join(BASE, "outputs", "fold1_orchestrator_eval.csv")
OUT_CSV_EXTENDED = os.path.join(BASE, "outputs", "fold1_orchestrator_eval_extended.csv")

# [AGENT6-VERSION] Kept in sync with orchestrator.MODEL_VERSION by hand — not
# imported directly to keep this script lightweight (no torch/transformers
# dependency for a CSV-only evaluation).
MODEL_VERSION = "WavLM-base+LoRA_RoBERTa-base+LoRA_fold1"


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_pred_agreement(neighbor_labels_str: str, predicted: str) -> float:
    """Fraction of retrieved neighbors whose label matches the prediction."""
    neighbors = [s.strip() for s in str(neighbor_labels_str).split(";") if s.strip()]
    if not neighbors:
        return 0.0
    return sum(1 for lbl in neighbors if lbl == predicted) / len(neighbors)


def majority_neighbor_label(neighbor_labels_str: str) -> str:
    """
    Strict majority label (present in >50% of neighbors).
    Returns "TIE" if no single label achieves strict majority.
    """
    neighbors = [s.strip() for s in str(neighbor_labels_str).split(";") if s.strip()]
    if not neighbors:
        return "NONE"
    counts = Counter(neighbors)
    top_label, top_count = counts.most_common(1)[0]
    return top_label if top_count > len(neighbors) / 2 else "TIE"


# ── Load ──────────────────────────────────────────────────────────────────────
if not os.path.exists(TEST_CSV):
    print(f"ERROR: {TEST_CSV} not found.")
    sys.exit(1)

df = pd.read_csv(TEST_CSV)
n  = len(df)

# ── Derive orchestrator columns ───────────────────────────────────────────────
df["prediction_agreement"]   = df.apply(
    lambda r: compute_pred_agreement(r["neighbor_labels"], r["predicted"]), axis=1
)
df["alert_fired"]            = df["prediction_agreement"] < ALERT_THRESHOLD
df["majority_neighbor_label"] = df["neighbor_labels"].apply(majority_neighbor_label)

# Boolean convenience
correct     = df["correct"].astype(bool)
alert_fired = df["alert_fired"].astype(bool)

# ── Counts ────────────────────────────────────────────────────────────────────
n_alert = int(alert_fired.sum())
n_tp    = int((alert_fired &  ~correct).sum())   # alert fired, prediction was wrong
n_fp    = int((alert_fired &   correct).sum())   # alert fired, prediction was right
n_fn    = int((~alert_fired & ~correct).sum())   # no alert, prediction was wrong
n_tn    = int((~alert_fired &  correct).sum())   # no alert, prediction was right
n_error = int((~correct).sum())

alert_pct    = n_alert / n * 100
tp_pct       = n_tp    / n * 100          # TP as % of all utterances
fp_pct       = n_fp    / n * 100          # FP as % of all utterances
precision    = n_tp / n_alert             if n_alert > 0 else 0.0
recall       = n_tp / n_error             if n_error > 0 else 0.0
f1           = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# ── Print report ──────────────────────────────────────────────────────────────
print("=" * 62)
print("AGENT 6 ORCHESTRATOR EVALUATION — Fold-1 Test Set (Session 1)")
print("=" * 62)
print(f"  Alert threshold : pred_retrieval_agreement < {ALERT_THRESHOLD}")
print(f"  Source CSV      : fold1_test_agent_eval.csv")
print(f"  Derived from    : neighbor_labels + predicted (no re-inference)")
print()
print(f"  Total utterances processed : {n:>6}")
print(f"  Total misclassifications   : {n_error:>6}  ({n_error/n*100:.1f}% error rate)")
print()
print(f"  Alerts fired               : {n_alert:>6}  ({alert_pct:.1f}% of all utterances)")
print()
print("  Alert outcomes:")
print(f"    True  positives (alert + wrong pred) : {n_tp:>4}  "
      f"({tp_pct:.1f}% of all utts)")
print(f"    False positives (alert + right pred) : {n_fp:>4}  "
      f"({fp_pct:.1f}% of all utts)")
print(f"    False negatives (no alert + wrong)   : {n_fn:>4}  "
      f"({n_fn/n*100:.1f}% of all utts — missed errors)")
print(f"    True  negatives (no alert + right)   : {n_tn:>4}  "
      f"({n_tn/n*100:.1f}% of all utts)")
print()
print("  Alert performance (treating misclassification as positive class):")
print(f"    Precision (of fired alerts) : {precision:.4f}  "
      f"({n_tp}/{n_alert} alerts were real errors)")
print(f"    Recall    (over errors)     : {recall:.4f}  "
      f"({n_tp}/{n_error} errors were caught)")
print(f"    F1                          : {f1:.4f}")
print()

# Label distribution among alerted utterances
if n_alert > 0:
    alert_df = df[alert_fired]
    print("  Ground-truth distribution in alerted utterances:")
    for emotion, count in alert_df["ground_truth"].value_counts().items():
        print(f"    {emotion:<10} {count:>4}  ({count/n_alert*100:.1f}%)")
    print()
    print("  Majority neighbor label when alert fired (top 3 vs prediction):")
    disagreed = alert_df[alert_df["majority_neighbor_label"] != alert_df["predicted"]]
    print(f"    Alert + neighbor majority disagrees with prediction: "
          f"{len(disagreed)} / {n_alert} ({len(disagreed)/n_alert*100:.1f}%)")

print()
print(f"  Decision log saved to: {OUT_CSV}")
print(f"  Extended decision log (+ CVR context columns) saved to: {OUT_CSV_EXTENDED}")
print("=" * 62)

# ── Save decision log CSV ─────────────────────────────────────────────────────
out_cols = [
    "utterance_id",
    "ground_truth",
    "predicted",
    "correct",
    "alert_fired",
    "prediction_agreement",
    "majority_neighbor_label",
]
df[out_cols].to_csv(OUT_CSV, index=False)

# ── Save EXTENDED decision log CSV ─────────────────────────────────────────────
# [AGENT6-CVR-CONTEXT] Three columns added for future CVR/FDR integration:
#   utterance_timecode : None until CVR audio timestamps are wired in (blocked
#                        pending GCAA data access, same as Agent 4).
#   flight_phase        : None until FDR data is time-aligned with CVR audio.
#   model_version        : populated now — identifies the fusion checkpoint
#                        that produced these predictions.
df["utterance_timecode"] = None
df["flight_phase"]       = None
df["model_version"]      = MODEL_VERSION

out_cols_extended = out_cols + ["utterance_timecode", "flight_phase", "model_version"]
df[out_cols_extended].to_csv(OUT_CSV_EXTENDED, index=False)
