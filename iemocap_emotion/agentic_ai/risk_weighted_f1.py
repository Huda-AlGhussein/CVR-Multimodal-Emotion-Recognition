"""
risk_weighted_f1.py
──────────────────────────────────────────────────────────────────────────────
Computes a risk-weighted F1 score alongside standard Macro F1.

RISK-WEIGHTED F1 — DEFINITION
───────────────────────────────
Standard F1 per class c treats every false negative (FN) equally.
Risk-weighted F1 penalises FNs proportional to the clinical/operational
risk of missing that emotion class:

    weighted_recall_c = TP_c / (TP_c + w_c * FN_c)
    precision_c       = TP_c / (TP_c + FP_c)          [unchanged]
    risk_F1_c         = 2 * P_c * R^w_c / (P_c + R^w_c)

    risk_MacroF1      = mean( risk_F1_c )  over all classes

Multiplying FN_c by w_c > 1 makes recall harder to achieve for
high-risk classes: the denominator grows faster, so the model needs
more TPs relative to FNs to maintain the same recall score.

WHY THIS FORMULATION
─────────────────────
The user's spec: "weight the false negative penalty by the class weight
of the true label." For utterances with true label c, each missed
prediction contributes w_c to the FN penalty rather than 1. This
aggregates to w_c * FN_c in the per-class recall denominator.

Precision is left unweighted because false positives (predicting anger
when the true label is something else) depend on the predicted class,
not the true class — the specification only mentions true-label weights.

PROVISIONAL RISK WEIGHTS (CVR domain, under review)
──────────────────────────────────────────────────────
  anger=3, fear=3, joy=2, sadness=2, neutral=1

Anger and fear are weighted highest as both are associated with
escalating distress responses in CVR contexts. Joy and sadness carry
intermediate weight. Neutral carries unit weight (baseline).

INPUT
──────
  agentic_ai/outputs/fold1_test_agent_eval.csv
  Columns used: ground_truth (str), predicted (str)

OUTPUT
───────
  agentic_ai/outputs/fold1_risk_weighted_f1.txt
"""

import os
import numpy as np
import pandas as pd

# ── Risk weights ──────────────────────────────────────────────────────────────
RISK_WEIGHTS = {
    "anger":   3,
    "fear":    3,
    "joy":     2,
    "sadness": 2,
    "neutral": 1,
}
CLASSES = ["anger", "fear", "joy", "neutral", "sadness"]   # alphabetical for consistency

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(__file__)
IN_CSV  = os.path.join(BASE, "outputs", "fold1_test_agent_eval.csv")
OUT_TXT = os.path.join(BASE, "outputs", "fold1_risk_weighted_f1.txt")


# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(IN_CSV)
y_true = df["ground_truth"].tolist()
y_pred = df["predicted"].tolist()
N      = len(df)


# ── Per-class confusion counts ────────────────────────────────────────────────
counts = {c: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for c in CLASSES}

for true, pred in zip(y_true, y_pred):
    for c in CLASSES:
        if true == c and pred == c:
            counts[c]["TP"] += 1
        elif true != c and pred == c:
            counts[c]["FP"] += 1
        elif true == c and pred != c:
            counts[c]["FN"] += 1
        else:
            counts[c]["TN"] += 1


# ── Standard per-class F1 ─────────────────────────────────────────────────────
def std_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0, prec, rec


# ── Risk-weighted per-class F1 ────────────────────────────────────────────────
def risk_f1(tp, fp, fn, w):
    """
    Weighted recall uses w * FN in the denominator.
    Precision unchanged — FPs are not class-risk-weighted per the spec.
    """
    prec      = tp / (tp + fp)         if (tp + fp)          > 0 else 0.0
    w_rec     = tp / (tp + w * fn)     if (tp + w * fn)      > 0 else 0.0
    denom     = prec + w_rec
    return 2 * prec * w_rec / denom    if denom               > 0 else 0.0, prec, w_rec


# ── Compute ───────────────────────────────────────────────────────────────────
results = {}
for c in CLASSES:
    tp = counts[c]["TP"]
    fp = counts[c]["FP"]
    fn = counts[c]["FN"]
    w  = RISK_WEIGHTS[c]

    f1_std,  prec_std,  rec_std  = std_f1(tp, fp, fn)
    f1_risk, prec_risk, w_rec    = risk_f1(tp, fp, fn, w)

    results[c] = {
        "w":        w,
        "TP": tp, "FP": fp, "FN": fn,
        "support":  tp + fn,
        # standard
        "prec_std": prec_std,
        "rec_std":  rec_std,
        "f1_std":   f1_std,
        # risk-weighted
        "prec_risk": prec_risk,   # same as prec_std (FP unchanged)
        "w_rec":     w_rec,
        "f1_risk":   f1_risk,
    }

macro_f1_std  = np.mean([results[c]["f1_std"]  for c in CLASSES])
macro_f1_risk = np.mean([results[c]["f1_risk"] for c in CLASSES])

# Weighted-by-risk macro (weighted average using risk weights)
risk_vals = np.array([results[c]["f1_risk"] for c in CLASSES])
w_vals    = np.array([RISK_WEIGHTS[c]        for c in CLASSES], dtype=float)
macro_f1_risk_weighted_avg = float(np.dot(risk_vals, w_vals) / w_vals.sum())

# ── Format report ─────────────────────────────────────────────────────────────
sep  = "=" * 72
sep2 = "-" * 72

lines = []
lines.append(sep)
lines.append("RISK-WEIGHTED F1 — Fold-1 Test Set (Agent 3 Fusion Model)")
lines.append("Session 1, 1,097 utterances")
lines.append(sep)
lines.append("")
lines.append("RISK WEIGHTS (provisional, CVR domain)")
lines.append(sep2)
for c in CLASSES:
    lines.append(f"  {c:<10} w = {RISK_WEIGHTS[c]}")
lines.append("")
lines.append("DEFINITION")
lines.append(sep2)
lines.append("  weighted_recall_c = TP_c / (TP_c + w_c * FN_c)")
lines.append("  precision_c       = TP_c / (TP_c + FP_c)          [unchanged]")
lines.append("  risk_F1_c         = 2 * P_c * R^w_c / (P_c + R^w_c)")
lines.append("  risk_MacroF1      = mean( risk_F1_c ) over all classes")
lines.append("")
lines.append("PER-CLASS BREAKDOWN")
lines.append(sep2)

hdr = (f"  {'Class':<10} {'w':>3}  {'Support':>8}  "
       f"{'Prec':>6}  {'StdRec':>7}  {'StdF1':>7}  "
       f"{'WtdRec':>7}  {'RiskF1':>7}  {'F1 delta':>9}")
lines.append(hdr)
lines.append("  " + "-" * 68)

for c in CLASSES:
    r = results[c]
    delta = r["f1_risk"] - r["f1_std"]
    lines.append(
        f"  {c:<10} {r['w']:>3}  {r['support']:>8}  "
        f"{r['prec_std']:>6.4f}  {r['rec_std']:>7.4f}  {r['f1_std']:>7.4f}  "
        f"{r['w_rec']:>7.4f}  {r['f1_risk']:>7.4f}  {delta:>+9.4f}"
    )

lines.append("  " + "-" * 68)
lines.append("")

lines.append("SUMMARY")
lines.append(sep2)
lines.append(f"  Standard  MacroF1              : {macro_f1_std:.6f}")
lines.append(f"  Risk      MacroF1 (mean)        : {macro_f1_risk:.6f}   "
             f"  (delta vs std: {macro_f1_risk - macro_f1_std:+.6f})")
lines.append(f"  Risk      MacroF1 (wtd avg)     : {macro_f1_risk_weighted_avg:.6f}   "
             f"  (risk-weighted average of per-class risk F1)")
lines.append("")
lines.append("INTERPRETATION")
lines.append(sep2)
lines.append("  Standard MacroF1 treats all classes equally when averaging.")
lines.append("  Risk MacroF1 (mean) still macro-averages but each class's")
lines.append("    per-class F1 is already penalised by its weight in the recall")
lines.append("    denominator — so high-risk classes drag the score down more")
lines.append("    if they have high FN rates.")
lines.append("  Risk MacroF1 (wtd avg) additionally weights the class F1s by")
lines.append("    risk score at the macro-average step, double-penalising")
lines.append("    high-risk classes. Use the mean variant for comparability")
lines.append("    with standard Macro F1 in papers.")
lines.append("")
lines.append("  A positive delta in RiskF1 - StdF1 for a class means the model")
lines.append("  happens to have better precision than (weighted) recall for that")
lines.append("  class, so the harmonic-mean numerics shift in its favour under")
lines.append("  the new weighting. A negative delta is the more common case:")
lines.append("  high-risk FNs increase the denominator, pulling recall down.")
lines.append("")
lines.append("CONFUSION SUMMARY (top misclassification pairs for risk classes)")
lines.append(sep2)
for true_c in ["anger", "fear"]:   # risk-weight=3 classes
    w = RISK_WEIGHTS[true_c]
    errors = df[(df["ground_truth"] == true_c) & (df["predicted"] != true_c)]
    if len(errors) == 0:
        lines.append(f"  {true_c} (w={w}): no errors")
        continue
    counts_err = errors["predicted"].value_counts()
    total_fn   = len(errors)
    lines.append(f"  {true_c} (w={w}): {total_fn} FN  ->  "
                 + ", ".join(f"{pred}:{n}" for pred, n in counts_err.items()))
lines.append(sep)
lines.append("")

report = "\n".join(lines)
print(report)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Saved to: {OUT_TXT}")
