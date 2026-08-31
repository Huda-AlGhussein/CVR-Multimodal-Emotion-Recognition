"""
evaluate_alert.py
─────────────────────────────────────────────────────────────────────────────
Evaluates AlertAgent rule performance using pre-computed per-utterance CSVs
produced by evaluate_agent.py.

Prints 2x2 contingency tables (alert/no-alert × correct/incorrect) with
FPR, TPR, precision, and F1 for both the val split (threshold-selection,
NOT reportable as a final result) and the test split (held-out, reportable).

PROVENANCE OF INPUT CSVs
──────────────────────────
  fold1_val_agent_eval.csv  — generate with:
      python agentic_ai/evaluate_agent.py --split val
  fold1_test_agent_eval.csv — generate with:
      python agentic_ai/evaluate_agent.py --split test  (or default)

  The val CSV was originally produced by an ad-hoc inline script during
  alert-rule threshold tuning (not by evaluate_agent.py). Re-run the above
  command to regenerate it reproducibly.

SPLIT DISCIPLINE
──────────────────
  val  → threshold selection only. These numbers justify which threshold was
          chosen. They MUST NOT be reported as the final performance figure.
  test → final reportable result. The threshold must be fixed before this is
          examined. Examining test to pick a threshold is data leakage.

USAGE
──────
  # Default: Rule A, threshold=0.6
  python agentic_ai/evaluate_alert.py

  # Sweep a different threshold
  python agentic_ai/evaluate_alert.py --threshold 0.4

  # Rule B (combined confidence + agreement)
  python agentic_ai/evaluate_alert.py --rule combined --threshold 0.6 --conf_threshold 0.8

Run from project root.
"""

import sys
import argparse
import pathlib
from collections import Counter

import numpy as np
import pandas as pd

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_OUTPUT_DIR   = _PROJECT_ROOT / "agentic_ai" / "outputs"

EMOTIONS = ["anger", "fear", "joy", "neutral", "sadness"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AlertAgent rule on val and test splits")
    p.add_argument("--fold",           type=int,   default=1,
                   help="LOSO fold (default: 1)")
    p.add_argument("--rule",           choices=["graded_agreement", "combined"],
                   default="graded_agreement",
                   help="Alert rule type (default: graded_agreement)")
    p.add_argument("--threshold",      type=float, default=0.6,
                   help="Prediction-retrieval agreement threshold for alert "
                        "(alert when agreement < threshold). Default: 0.6")
    p.add_argument("--conf_threshold", type=float, default=0.8,
                   help="Confidence threshold for Rule B combined (default: 0.8)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def add_pred_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds pred_retrieval_agreement column: fraction of top-k neighbors whose
    label matches the PREDICTION (not ground truth).

    The val CSV already has this column (written by the original ad-hoc script).
    The test CSV does not — we recompute it from neighbor_labels + predicted.
    """
    if "pred_retrieval_agreement" in df.columns:
        return df

    def _agree(row):
        labels = str(row["neighbor_labels"]).split(";")
        return sum(1 for l in labels if l == row["predicted"]) / len(labels)

    df = df.copy()
    df["pred_retrieval_agreement"] = df.apply(_agree, axis=1)
    return df


def add_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Adds max-softmax confidence column from per-class probability columns."""
    if "confidence" in df.columns:
        return df
    prob_cols = [f"p_{e}" for e in EMOTIONS if f"p_{e}" in df.columns]
    df = df.copy()
    df["confidence"] = df[prob_cols].max(axis=1)
    return df


def compute_alert_series(
    df:             pd.DataFrame,
    rule:           str,
    threshold:      float,
    conf_threshold: float,
) -> pd.Series:
    """Returns a boolean Series: True where the rule fires an alert."""
    # [AGENT5-RULE-A] graded_agreement: alert when prediction-retrieval agreement < threshold
    if rule == "graded_agreement":
        return df["pred_retrieval_agreement"] < threshold

    # [AGENT5-RULE-B] combined: alert when BOTH signals are below their thresholds
    return (df["confidence"] < conf_threshold) & (df["pred_retrieval_agreement"] < threshold)


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def report_2x2(
    df:           pd.DataFrame,
    alert_series: pd.Series,
    split_label:  str,
    reportable:   bool,
) -> dict:
    """
    Prints a 2x2 contingency table and returns the metric dict.

    Parameters
    ──────────
    reportable : if False, prints a warning that these numbers must not be
                 used as the final performance figure (val numbers only).
    """
    n_correct   = int(df["correct"].sum())
    n_incorrect = int((~df["correct"]).sum())
    n_total     = len(df)

    tp = int((~df["correct"] &  alert_series).sum())   # wrong pred, alert fires
    fp = int(( df["correct"] &  alert_series).sum())   # right pred, alert fires (false alarm)
    fn = int((~df["correct"] & ~alert_series).sum())   # wrong pred, no alert (missed)
    tn = int(( df["correct"] & ~alert_series).sum())   # right pred, no alert

    fpr  = fp / n_correct   if n_correct   > 0 else 0.0
    tpr  = tp / n_incorrect if n_incorrect > 0 else 0.0
    prec = tp / (tp + fp)   if (tp + fp)   > 0 else 0.0
    f1   = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0.0

    tag = "FINAL REPORTABLE RESULT" if reportable else "THRESHOLD SELECTION ONLY — do not report as final"

    print(f"\n  -- {split_label} [{tag}] --")
    print(f"  {'':20} {'Correct pred':>14}  {'Incorrect pred':>15}  {'Total':>6}")
    print(f"  {'ALERT (fired)':20} {fp:>7} ({fpr:5.1%})   {tp:>7} ({tpr:5.1%})   {fp+tp:>6}")
    print(f"  {'NO ALERT':20} {tn:>7} ({tn/n_correct:5.1%})   {fn:>7} ({fn/n_incorrect:5.1%})   {tn+fn:>6}")
    print(f"  {'Total':20} {n_correct:>14}  {n_incorrect:>15}  {n_total:>6}")
    print()
    print(f"  FPR      = {fpr:.3f}  ({fp}/{n_correct} correct predictions incorrectly flagged)")
    print(f"  TPR      = {tpr:.3f}  ({tp}/{n_incorrect} errors correctly flagged)")
    print(f"  Precision= {prec:.3f}  ({tp}/{tp+fp} alerts were genuine errors)")
    print(f"  F1       = {f1:.3f}")
    print(f"  n_alerts = {tp+fp}  ({(tp+fp)/n_total:.1%} of all utterances)")

    return dict(fpr=fpr, tpr=tpr, precision=prec, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn,
                n_correct=n_correct, n_incorrect=n_incorrect)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    val_csv  = _OUTPUT_DIR / f"fold{args.fold}_val_agent_eval.csv"
    test_csv = _OUTPUT_DIR / f"fold{args.fold}_test_agent_eval.csv"

    for path, label in [(val_csv, "val"), (test_csv, "test")]:
        if not path.exists():
            print(f"[ERROR] Missing {path}")
            print(f"  Generate it with: python agentic_ai/evaluate_agent.py --split {label}")
            raise SystemExit(1)

    val_df  = add_pred_agreement(add_confidence(pd.read_csv(val_csv)))
    test_df = add_pred_agreement(add_confidence(pd.read_csv(test_csv)))

    rule_desc = args.rule
    if args.rule == "combined":
        rule_desc += f" (agree<{args.threshold}, conf<{args.conf_threshold})"
    else:
        rule_desc += f" (agree<{args.threshold})"

    print(f"\n{'='*70}")
    print(f"  AlertAgent rule evaluation — fold {args.fold}")
    print(f"  Rule: {rule_desc}")
    print(f"{'='*70}")

    val_alert  = compute_alert_series(val_df,  args.rule, args.threshold, args.conf_threshold)
    test_alert = compute_alert_series(test_df, args.rule, args.threshold, args.conf_threshold)

    val_metrics  = report_2x2(val_df,  val_alert,  "VAL (session 2)",  reportable=False)
    test_metrics = report_2x2(test_df, test_alert, "TEST (session 1)", reportable=True)

    print(f"\n  Summary comparison:")
    print(f"  {'Metric':12}  {'VAL':>10}  {'TEST':>10}  {'Gap':>8}")
    print(f"  {'-'*46}")
    for key in ("fpr", "tpr", "precision", "f1"):
        v = val_metrics[key]
        t = test_metrics[key]
        print(f"  {key:12}  {v:>10.3f}  {t:>10.3f}  {t-v:>+8.3f}")
    print()


if __name__ == "__main__":
    main()
