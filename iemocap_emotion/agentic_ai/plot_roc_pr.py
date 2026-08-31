"""
plot_roc_pr.py
──────────────────────────────────────────────────────────────────────────────
ROC and Precision-Recall curves for Agent 5 (AlertAgent).

Positive class : misclassification (correct == False)
Score          : 1 - pred_retrieval_agreement
                 Higher → neighbors disagree with prediction → alert more likely.
                 Alert fires when pred_retrieval_agreement < threshold,
                 i.e. risk_score > (1 - threshold).

Operating point: threshold = 0.6 on pred_retrieval_agreement
                 → risk_score cut = 0.4
                 → alert fires when risk_score > 0.4

Val  : fold1_val_agent_eval.csv  (pred_retrieval_agreement already present)
Test : fold1_test_agent_eval.csv (compute pred_retrieval_agreement from
       neighbor_labels + predicted columns)

Output: agentic_ai/outputs/roc_pr_curves.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
VAL_CSV  = os.path.join(BASE, "outputs", "fold1_val_agent_eval.csv")
TEST_CSV = os.path.join(BASE, "outputs", "fold1_test_agent_eval.csv")
OUT_PNG  = os.path.join(BASE, "outputs", "roc_pr_curves.png")

AGREE_THRESHOLD = 0.6      # operating point on pred_retrieval_agreement
RISK_CUT        = 1.0 - AGREE_THRESHOLD   # 0.4 — alert fires above this


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_pred_agreement(row: pd.Series) -> float:
    """
    Replicates AlertAgent._prediction_agreement() without importing the agent.
    neighbor_labels column is semicolon-separated, e.g. "neutral;anger;neutral".
    """
    neighbors = [s.strip() for s in str(row["neighbor_labels"]).split(";") if s.strip()]
    if not neighbors:
        return 0.0
    return sum(1 for lbl in neighbors if lbl == row["predicted"]) / len(neighbors)


def load_split(path: str, has_pred_agreement: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if has_pred_agreement:
        assert "pred_retrieval_agreement" in df.columns, (
            f"Expected pred_retrieval_agreement in {path}"
        )
    else:
        df["pred_retrieval_agreement"] = df.apply(compute_pred_agreement, axis=1)
    df["risk_score"] = 1.0 - df["pred_retrieval_agreement"]
    df["is_error"]   = (~df["correct"].astype(bool)).astype(int)   # 1 = misclassification
    return df


def operating_point(df: pd.DataFrame) -> tuple:
    """
    FPR, TPR (for ROC) and Precision, Recall (for PR) at the default threshold.
    Alert fires when risk_score > RISK_CUT  ⟺  pred_retrieval_agreement < 0.6
    """
    alert      = (df["risk_score"] > RISK_CUT).astype(int)
    y_true     = df["is_error"].to_numpy()
    y_pred     = alert.to_numpy()

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tpr

    return fpr, tpr, prec, rec, tp, fp, fn, tn


# ── Load data ─────────────────────────────────────────────────────────────────
val_df  = load_split(VAL_CSV,  has_pred_agreement=True)
test_df = load_split(TEST_CSV, has_pred_agreement=False)

# ── Curves ────────────────────────────────────────────────────────────────────
val_fpr,  val_tpr,  val_thresholds  = roc_curve(val_df["is_error"],  val_df["risk_score"])
test_fpr, test_tpr, test_thresholds = roc_curve(test_df["is_error"], test_df["risk_score"])

val_roc_auc  = auc(val_fpr,  val_tpr)
test_roc_auc = auc(test_fpr, test_tpr)

val_prec,  val_rec,  _ = precision_recall_curve(val_df["is_error"],  val_df["risk_score"])
test_prec, test_rec, _ = precision_recall_curve(test_df["is_error"], test_df["risk_score"])

val_ap  = average_precision_score(val_df["is_error"],  val_df["risk_score"])
test_ap = average_precision_score(test_df["is_error"], test_df["risk_score"])

# ── Operating points at threshold=0.6 ─────────────────────────────────────────
val_fpr_op,  val_tpr_op,  val_prec_op,  val_rec_op,  *val_counts  = operating_point(val_df)
test_fpr_op, test_tpr_op, test_prec_op, test_rec_op, *test_counts = operating_point(test_df)

# ── Print AUC values ───────────────────────────────────────────────────────────
print("=" * 60)
print("AUC VALUES — Agent 5 AlertAgent (threshold = 0.6)")
print("=" * 60)
print(f"  ROC-AUC   val : {val_roc_auc:.4f}")
print(f"  ROC-AUC   test: {test_roc_auc:.4f}")
print(f"  Avg Prec  val : {val_ap:.4f}  (PR-AUC)")
print(f"  Avg Prec  test: {test_ap:.4f}  (PR-AUC)")
print()
print("Operating point (pred_retrieval_agreement < 0.6 => alert):")
print(f"  VAL   FPR={val_fpr_op:.3f}  TPR={val_tpr_op:.3f}  "
      f"Prec={val_prec_op:.3f}  Rec={val_rec_op:.3f}")
vtp, vfp, vfn, vtn = val_counts
print(f"  VAL   TP={vtp}  FP={vfp}  FN={vfn}  TN={vtn}")
print(f"  TEST  FPR={test_fpr_op:.3f}  TPR={test_tpr_op:.3f}  "
      f"Prec={test_prec_op:.3f}  Rec={test_rec_op:.3f}")
ttp, tfp, tfn, ttn = test_counts
print(f"  TEST  TP={ttp}  FP={tfp}  FN={tfn}  TN={ttn}")
print("=" * 60)

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
ORANGE = "#EA580C"
GRAY   = "#9CA3AF"
MARKER = "D"
MS     = 9

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor("#FAFAFA")
for ax in axes:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors="#374151", labelsize=10)

# ── ROC ───────────────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(val_fpr,  val_tpr,  color=BLUE,   lw=2,
        label=f"Val  (AUC = {val_roc_auc:.3f})")
ax.plot(test_fpr, test_tpr, color=ORANGE, lw=2, linestyle="--",
        label=f"Test (AUC = {test_roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle=":", label="Random")

# Operating points
ax.plot(val_fpr_op,  val_tpr_op,  marker=MARKER, color=BLUE,   markersize=MS,
        zorder=5, label=f"Val  op  (FPR={val_fpr_op:.2f}, TPR={val_tpr_op:.2f})")
ax.plot(test_fpr_op, test_tpr_op, marker=MARKER, color=ORANGE, markersize=MS,
        zorder=5, label=f"Test op  (FPR={test_fpr_op:.2f}, TPR={test_tpr_op:.2f})")

# Annotate operating points
ax.annotate(f"t=0.6\nFPR={val_fpr_op:.2f}\nTPR={val_tpr_op:.2f}",
            xy=(val_fpr_op, val_tpr_op),
            xytext=(val_fpr_op + 0.08, val_tpr_op - 0.12),
            fontsize=8, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
ax.annotate(f"t=0.6\nFPR={test_fpr_op:.2f}\nTPR={test_tpr_op:.2f}",
            xy=(test_fpr_op, test_tpr_op),
            xytext=(test_fpr_op + 0.08, test_tpr_op + 0.06),
            fontsize=8, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("False Positive Rate", fontsize=11, color="#374151")
ax.set_ylabel("True Positive Rate", fontsize=11, color="#374151")
ax.set_title("ROC Curve — Agent 5 Alert\n(positive = misclassification)",
             fontsize=12, color="#111827", pad=10)
ax.legend(fontsize=8.5, framealpha=0.9, loc="lower right")
ax.grid(True, linestyle="--", linewidth=0.4, color="#E5E7EB")

# ── Precision-Recall ──────────────────────────────────────────────────────────
ax = axes[1]

# sklearn precision_recall_curve returns curves in descending-threshold order;
# for a readable plot, reverse so recall increases left to right.
ax.plot(val_rec[::-1],  val_prec[::-1],  color=BLUE,   lw=2,
        label=f"Val  (AP = {val_ap:.3f})")
ax.plot(test_rec[::-1], test_prec[::-1], color=ORANGE, lw=2, linestyle="--",
        label=f"Test (AP = {test_ap:.3f})")

# Baseline: random classifier at class-imbalance rate
val_baseline  = val_df["is_error"].mean()
test_baseline = test_df["is_error"].mean()
ax.axhline(val_baseline,  color=BLUE,   lw=1, linestyle=":",
           label=f"Val  baseline ({val_baseline:.2f})")
ax.axhline(test_baseline, color=ORANGE, lw=1, linestyle=":",
           label=f"Test baseline ({test_baseline:.2f})")

# Operating points
ax.plot(val_rec_op,  val_prec_op,  marker=MARKER, color=BLUE,   markersize=MS,
        zorder=5, label=f"Val  op  (Rec={val_rec_op:.2f}, Prec={val_prec_op:.2f})")
ax.plot(test_rec_op, test_prec_op, marker=MARKER, color=ORANGE, markersize=MS,
        zorder=5, label=f"Test op  (Rec={test_rec_op:.2f}, Prec={test_prec_op:.2f})")

ax.annotate(f"t=0.6\nRec={val_rec_op:.2f}\nPrec={val_prec_op:.2f}",
            xy=(val_rec_op, val_prec_op),
            xytext=(val_rec_op - 0.18, val_prec_op + 0.08),
            fontsize=8, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
ax.annotate(f"t=0.6\nRec={test_rec_op:.2f}\nPrec={test_prec_op:.2f}",
            xy=(test_rec_op, test_prec_op),
            xytext=(test_rec_op + 0.05, test_prec_op - 0.10),
            fontsize=8, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.0,   1.05)
ax.set_xlabel("Recall", fontsize=11, color="#374151")
ax.set_ylabel("Precision", fontsize=11, color="#374151")
ax.set_title("Precision-Recall Curve — Agent 5 Alert\n(positive = misclassification)",
             fontsize=12, color="#111827", pad=10)
ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right")
ax.grid(True, linestyle="--", linewidth=0.4, color="#E5E7EB")

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    "Score = 1 − pred_retrieval_agreement  |  Alert fires when score > 0.4  "
    "(pred_retrieval_agreement < 0.6)  |  Val = session 2, Test = session 1",
    ha="center", fontsize=8, color="#6B7280",
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
print(f"\nPlot saved to: {OUT_PNG}")
