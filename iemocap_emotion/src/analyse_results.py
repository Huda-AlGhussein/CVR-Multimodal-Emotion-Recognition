"""
analyse_results.py
──────────────────────────────────────────────────────────────────────────────
Generates per-class F1 tables, confusion matrices, and pairwise significance
tests (McNemar + Wilcoxon signed-rank) for all available model prediction CSVs.

Supports both datasets:
  MELD   — single test set, one CSV per model
  IEMOCAP — 5-fold LOSO; per-fold CSVs are pooled for class-level stats,
             and fold-level Macro F1 vectors are used for Wilcoxon

Usage
─────
  python src/analyse_results.py              # all available outputs
  python src/analyse_results.py --dataset meld
  python src/analyse_results.py --dataset iemocap

Outputs (written to outputs/analysis/)
───────────────────────────────────────
  per_class_f1.txt            — side-by-side F1 table (all datasets × models)
  confusion_<dataset>_<model>.txt  — normalised confusion matrix (text art)
  significance_tests.txt      — McNemar + Wilcoxon table
"""

import os
import sys
import glob
import argparse
import textwrap
from collections import defaultdict
from itertools import combinations

# Force UTF-8 output on Windows (cp1252 cannot encode box-drawing chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report
)
from statsmodels.stats.contingency_tables import mcnemar

# ── Label space (shared across both datasets) ────────────────────────────────
CLASSES   = ["anger", "fear", "joy", "neutral", "sadness"]
N_CLASSES = len(CLASSES)
IDX_TO_LABEL = {i: c for i, c in enumerate(CLASSES)}

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUTS_ROOT = "./outputs"
ANALYSIS_DIR = os.path.join(OUTPUTS_ROOT, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

MODELS   = ["text", "audio", "fusion"]
DATASETS = ["meld", "iemocap"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_meld_preds(model: str) -> pd.DataFrame | None:
    path = os.path.join(OUTPUTS_ROOT, model, "meld_test_predictions.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_iemocap_preds(model: str) -> pd.DataFrame | None:
    """
    Pools all available fold CSVs into one DataFrame.
    Each fold contributes its test-session utterances; together they cover all
    5 sessions without overlap (LOSO guarantee), so pooling is valid for
    aggregate metrics and per-class F1.
    """
    paths = sorted(glob.glob(
        os.path.join(OUTPUTS_ROOT, model, "fold*_test_predictions.csv")
    ))
    if not paths:
        return None
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df


def load_iemocap_fold_macro_f1s(model: str) -> list[float] | None:
    """
    Returns a list of 5 per-fold Macro F1 values (for Wilcoxon).
    Computed directly from the pooled per-fold CSV rows, NOT from the JSON,
    so the source is always the raw predictions (avoids JSON rounding).
    """
    paths = sorted(glob.glob(
        os.path.join(OUTPUTS_ROOT, model, "fold*_test_predictions.csv")
    ))
    if len(paths) < 2:   # need at least 2 folds for a meaningful test
        return None
    scores = []
    for p in paths:
        df = pd.read_csv(p)
        f1 = f1_score(df["true_label_idx"], df["pred_label_idx"],
                      average="macro", zero_division=0)
        scores.append(f1)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# PER-CLASS F1 TABLE
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_class_f1(df: pd.DataFrame) -> dict:
    y_true = df["true_label_idx"].to_numpy()
    y_pred = df["pred_label_idx"].to_numpy()
    f1s = f1_score(y_true, y_pred, labels=list(range(N_CLASSES)),
                   average=None, zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    wacc  = np.mean(y_true == y_pred)
    return {cls: float(f1s[i]) for i, cls in enumerate(CLASSES)} | {
        "MacroF1": macro,
        "Acc":     wacc,
    }


def build_f1_table(dataset: str) -> str | None:
    """
    Returns a formatted table with one column per model.
    Rows: per-class F1, then MacroF1, then Accuracy.
    Returns None if no CSVs found for this dataset.
    """
    loader = load_meld_preds if dataset == "meld" else load_iemocap_preds
    results = {}
    for model in MODELS:
        df = loader(model)
        if df is not None:
            results[model] = compute_per_class_f1(df)

    if not results:
        return None

    col_w  = 10
    row_w  = 12
    header = f"\n{'Per-class F1 — ' + dataset.upper():^{row_w + col_w * len(results)}}\n"
    header += f"{'':>{row_w}}" + "".join(f"{m:>{col_w}}" for m in results)
    header += "\n" + "─" * (row_w + col_w * len(results))

    rows = []
    for cls in CLASSES:
        row = f"  {cls:<{row_w - 2}}"
        for m in results:
            row += f"{results[m].get(cls, float('nan')):>{col_w}.4f}"
        rows.append(row)

    # separator before summary rows
    sep = "  " + "─" * (row_w - 2 + col_w * len(results))
    rows.append(sep)
    for key in ["MacroF1", "Acc"]:
        row = f"  {key:<{row_w - 2}}"
        for m in results:
            row += f"{results[m].get(key, float('nan')):>{col_w}.4f}"
        rows.append(row)

    return header + "\n" + "\n".join(rows) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def format_confusion_matrix(cm_norm: np.ndarray, title: str) -> str:
    """Renders a normalised confusion matrix as a text table."""
    col_w = 9
    label_w = 10
    lines = [f"\n{title}", "  Predicted →"]
    header = f"  {'True ↓':<{label_w}}" + "".join(f"{c:>{col_w}}" for c in CLASSES)
    lines.append(header)
    lines.append("  " + "─" * (label_w + col_w * N_CLASSES))
    for i, cls in enumerate(CLASSES):
        row = f"  {cls:<{label_w}}"
        for j in range(N_CLASSES):
            val = cm_norm[i, j]
            marker = " *" if i == j else "  "
            row += f"{val:>{col_w - 2}.3f}{marker}"
        lines.append(row)
    return "\n".join(lines) + "\n"


def compute_and_format_cms(dataset: str) -> str:
    loader = load_meld_preds if dataset == "meld" else load_iemocap_preds
    blocks = []
    for model in MODELS:
        df = loader(model)
        if df is None:
            continue
        y_true = df["true_label_idx"].to_numpy()
        y_pred = df["pred_label_idx"].to_numpy()
        cm     = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        title   = f"Confusion matrix — {dataset.upper()} / {model}  (row-normalised; * = diagonal)"
        blocks.append(format_confusion_matrix(cm_norm, title))
    return "\n".join(blocks) if blocks else ""


# ─────────────────────────────────────────────────────────────────────────────
# SIGNIFICANCE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """
    McNemar's test on matched utterance-level correctness.

    Both DataFrames must cover the same utterance set. We inner-join on
    utterance_id so mismatched rows (e.g. a fold that ran for one model but
    not the other) are excluded transparently.

    Returns: {stat, p_value, n_concordant, n_discordant, n_matched}
    """
    merged = df_a[["utterance_id", "correct"]].merge(
        df_b[["utterance_id", "correct"]],
        on="utterance_id",
        suffixes=("_a", "_b"),
    )
    a_right = merged["correct_a"].to_numpy(dtype=bool)
    b_right = merged["correct_b"].to_numpy(dtype=bool)

    # 2×2 contingency table
    #       B correct  B wrong
    # A correct  n00       n01
    # A wrong    n10       n11
    n00 = int(( a_right &  b_right).sum())
    n01 = int(( a_right & ~b_right).sum())
    n10 = int((~a_right &  b_right).sum())
    n11 = int((~a_right & ~b_right).sum())
    table = np.array([[n00, n01], [n10, n11]])

    result = mcnemar(table, exact=False, correction=True)
    return {
        "stat":          result.statistic,
        "p_value":       result.pvalue,
        "n_matched":     len(merged),
        "n_discordant":  n01 + n10,
        "a_only_right":  n01,   # A correct, B wrong
        "b_only_right":  n10,   # B correct, A wrong
    }


def wilcoxon_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """
    Wilcoxon signed-rank test on paired per-fold Macro F1 scores.
    Uses two-sided alternative (tests whether distributions differ).
    Requires ≥ 2 pairs with non-zero differences; returns NaN otherwise.
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    diffs = a - b
    if np.all(diffs == 0) or len(diffs) < 2:
        return {"stat": float("nan"), "p_value": float("nan"), "n_pairs": len(diffs)}
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError as e:
        return {"stat": float("nan"), "p_value": float("nan"), "note": str(e),
                "n_pairs": len(diffs)}
    return {"stat": stat, "p_value": p, "n_pairs": len(diffs)}


def build_significance_table(dataset: str) -> str:
    """
    Runs all pairwise tests for a dataset and formats an organised table.

    MELD:     McNemar only (single test set → no fold-level pairs for Wilcoxon).
    IEMOCAP:  Both McNemar (pooled) and Wilcoxon (fold-level Macro F1).
    """
    loader = load_meld_preds if dataset == "meld" else load_iemocap_preds
    dfs    = {}
    fold_f1s = {}
    for model in MODELS:
        df = loader(model)
        if df is not None:
            dfs[model] = df
        if dataset == "iemocap":
            fold_f1s[model] = load_iemocap_fold_macro_f1s(model)

    pairs = list(combinations(list(dfs.keys()), 2))
    if not pairs:
        return f"\n[{dataset.upper()}] Not enough models to test.\n"

    lines = []
    lines.append(f"\n{'Significance Tests — ' + dataset.upper():^80}")
    lines.append("─" * 80)
    lines.append(
        "Pair (A vs B): McNemar tests per-utterance agreement "
        "(all utterances pooled)."
    )
    if dataset == "iemocap":
        lines.append(
            "Wilcoxon tests paired per-fold Macro F1 (5 pairs); "
            "N/A for MELD (single split)."
        )
    lines.append("α = 0.05  |  * p<0.05  ** p<0.01  *** p<0.001\n")

    col_labels = ["Pair", "McNemar χ²", "p (McNemar)", "Sig",
                  "Wilcoxon W", "p (Wilcoxon)", "Sig",
                  "A-only✓", "B-only✓", "N matched"]
    col_widths = [20, 12, 13, 5, 12, 13, 5, 9, 9, 10]
    header = "".join(f"{h:>{w}}" for h, w in zip(col_labels, col_widths))
    lines.append(header)
    lines.append("─" * sum(col_widths))

    def sig_stars(p):
        if np.isnan(p): return "   "
        if p < 0.001:   return "***"
        if p < 0.01:    return " **"
        if p < 0.05:    return "  *"
        return "   "

    for a_name, b_name in pairs:
        df_a = dfs[a_name]
        df_b = dfs[b_name]
        mc   = mcnemar_test(df_a, df_b)
        pair_label = f"{a_name} vs {b_name}"

        if dataset == "iemocap" and fold_f1s.get(a_name) and fold_f1s.get(b_name):
            wc = wilcoxon_test(fold_f1s[a_name], fold_f1s[b_name])
        else:
            wc = {"stat": float("nan"), "p_value": float("nan"), "n_pairs": 0}

        mc_p_str = f"{mc['p_value']:.4f}" if not np.isnan(mc['p_value']) else "  N/A"
        wc_p_str = f"{wc['p_value']:.4f}" if not np.isnan(wc['p_value']) else "  N/A"
        mc_s_str = f"{mc['stat']:.2f}"    if not np.isnan(mc['stat'])    else "  N/A"
        wc_s_str = f"{wc['stat']:.2f}"    if not np.isnan(wc['stat'])    else "  N/A"

        vals = [
            pair_label,
            mc_s_str, mc_p_str, sig_stars(mc['p_value']),
            wc_s_str, wc_p_str, sig_stars(wc['p_value']),
            str(mc.get("a_only_right", "")),
            str(mc.get("b_only_right", "")),
            str(mc.get("n_matched", "")),
        ]
        lines.append("".join(f"{v:>{w}}" for v, w in zip(vals, col_widths)))

    lines.append("─" * sum(col_widths))
    lines.append(
        "\nA-only✓ = utterances A got right and B got wrong. "
        "B-only✓ = utterances B got right and A got wrong."
    )
    lines.append(
        "McNemar uses continuity-corrected χ² (Yates). "
        "Wilcoxon uses zero_method='wilcox', two-sided."
    )
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse MELD and IEMOCAP results.")
    parser.add_argument(
        "--dataset", choices=["meld", "iemocap", "all"], default="all",
        help="Which dataset(s) to analyse (default: all)."
    )
    args = parser.parse_args()

    datasets = (
        ["meld", "iemocap"] if args.dataset == "all"
        else [args.dataset]
    )

    all_output = []

    # ── Per-class F1 tables ────────────────────────────────────────────────
    f1_blocks = []
    for ds in datasets:
        tbl = build_f1_table(ds)
        if tbl:
            f1_blocks.append(tbl)
        else:
            f1_blocks.append(f"\n[{ds.upper()}] No prediction CSVs found — skipping.\n")

    sep = "=" * 80
    f1_section = sep + "\nPER-CLASS F1\n" + sep + "\n" + "\n".join(f1_blocks)
    all_output.append(f1_section)
    print(f1_section)

    # ── Confusion matrices ─────────────────────────────────────────────────
    cm_section = sep + "\nCONFUSION MATRICES (row-normalised)\n" + sep
    for ds in datasets:
        block = compute_and_format_cms(ds)
        if block:
            cm_section += "\n" + block
        else:
            cm_section += f"\n[{ds.upper()}] No prediction CSVs found — skipping.\n"

    all_output.append(cm_section)
    print(cm_section)

    # ── Significance tests ─────────────────────────────────────────────────
    sig_section = sep + "\nSIGNIFICANCE TESTS\n" + sep
    for ds in datasets:
        sig_section += build_significance_table(ds)

    all_output.append(sig_section)
    print(sig_section)

    # ── Save to file ───────────────────────────────────────────────────────
    out_path = os.path.join(ANALYSIS_DIR, "results_analysis.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_output))
    print(f"\n[Done] Full report saved to: {out_path}")


if __name__ == "__main__":
    main()
