"""
temperature_scaling.py
──────────────────────────────────────────────────────────────────────────────
Post-hoc calibration via temperature scaling for the three MELD models.

WHAT TEMPERATURE SCALING DOES
───────────────────────────────
A single scalar T is learned per model. The calibrated probability for class c
on sample i is:

    p_c^(T) = softmax( log(p) / T )_c

where p is the original softmax output and log(p) recovers the relative logit
structure (valid because softmax is shift-invariant: the per-sample log-sum-exp
constant cancels). T > 1 flattens the distribution (reduces overconfidence);
T < 1 sharpens it (reduces underconfidence).

T is learned by minimising the negative log-likelihood (NLL) on the MELD dev
set (meld_dev.csv, 937 utterances). scipy.optimize.minimize_scalar searches
T ∈ (0.05, 50.0) with the "bounded" method (Brent's algorithm).

CALIBRATION METRIC — ECE
──────────────────────────
Expected Calibration Error (Guo et al., 2017):
    ECE = Σ_b (|B_b| / n) × |acc(B_b) - conf(B_b)|
10 equal-width bins, 0.0–1.0.

STEPS
──────
1. Load MELD dev and test splits.
2. For each model: run inference → save probability CSVs for dev and test.
3. Learn T on dev set NLL.
4. Apply T to test probabilities.
5. Compute ECE before and after on test set.
6. Plot reliability diagrams (before / after) for all three models.

OUTPUT FILES
────────────
  outputs/{model}/meld_dev_predictions.csv   ← dev probs (used for T learning)
  outputs/{model}/meld_test_predictions.csv  ← overwritten with prob columns added
  outputs/analysis/calibration_plots_after_scaling.png
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import cfg, NUM_CLASSES, IDX_TO_LABEL
from utils import get_device
from dataset import AudioDataset, TextDataset, MultimodalDataset, build_dataloader
from models_text import RoBERTaLoRA
from models_audio import WavLMLoRA
from models_fusion import CrossModalAttentionFusion
from meld_preprocessing import load_meld_splits

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "./checkpoints"
SPLITS_DIR     = "./data_splits/meld"
OUT_PNG        = "./outputs/analysis/calibration_plots_after_scaling.png"
N_BINS         = 10
CLASSES        = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
PROB_COLS      = [f"p_{c}" for c in CLASSES]
MODELS         = ["text", "audio", "fusion"]


# ── Model helpers ─────────────────────────────────────────────────────────────

def build_model(model_type: str) -> torch.nn.Module:
    if model_type == "text":
        return RoBERTaLoRA()
    if model_type == "audio":
        return WavLMLoRA()
    return CrossModalAttentionFusion()


def build_loader(model_type: str, df: pd.DataFrame):
    if model_type == "text":
        ds = TextDataset(df)
    elif model_type == "audio":
        ds = AudioDataset(df)
    else:
        ds = MultimodalDataset(df)
    return build_dataloader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=False,
    )


def run_inference(model_type: str, df: pd.DataFrame, device: torch.device,
                  model: torch.nn.Module) -> np.ndarray:
    """
    Forward pass on df. Returns float32 array of shape [N, NUM_CLASSES]
    containing softmax probabilities. Reuses an already-loaded model.
    """
    loader  = build_loader(model_type, df)
    use_amp = cfg.use_amp and device.type == "cuda"
    all_probs = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.cuda.amp.autocast(enabled=use_amp):
                if model_type == "text":
                    logits = model(
                        input_ids=batch["input_ids"].to(device, non_blocking=True),
                        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                    )
                elif model_type == "audio":
                    logits = model(
                        waveform=batch["waveform"].to(device, non_blocking=True),
                        audio_mask=batch["audio_mask"].to(device, non_blocking=True),
                    )
                else:
                    logits = model(
                        input_ids=batch["input_ids"].to(device, non_blocking=True),
                        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                        waveform=batch["waveform"].to(device, non_blocking=True),
                        audio_mask=batch["audio_mask"].to(device, non_blocking=True),
                    )
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)   # [N, NUM_CLASSES]


def save_predictions_with_probs(df: pd.DataFrame, probs: np.ndarray, path: str):
    """
    Writes utterance_id, true_label_idx, true_label, pred_label_idx,
    pred_label, correct, p_anger, p_fear, p_joy, p_neutral, p_sadness.
    """
    pred_idx = probs.argmax(axis=1)
    rows = {
        "utterance_id":   df["utterance_id"].tolist(),
        "true_label_idx": df["label_idx"].tolist(),
        "true_label":     df["emotion"].tolist(),
        "pred_label_idx": pred_idx.tolist(),
        "pred_label":     [IDX_TO_LABEL[i] for i in pred_idx],
        "correct":        (pred_idx == df["label_idx"].to_numpy()).tolist(),
    }
    for j, col in enumerate(PROB_COLS):
        rows[col] = probs[:, j].tolist()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"    Saved: {path}")


# ── Temperature scaling ───────────────────────────────────────────────────────

def learn_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Learns scalar T ∈ (0.05, 50) minimising NLL on (probs, labels).

    log(p) recovers relative logit structure; dividing by T before softmax
    is equivalent to dividing the original logits by T (softmax is
    shift-invariant so the per-sample log-sum-exp constant cancels).
    """
    log_p = np.log(probs.clip(min=1e-12))   # [N, C]  — log-probabilities

    def nll(T):
        scaled   = log_p / T                        # [N, C]
        # log-softmax in numpy: log(softmax(x)) = x - log(sum(exp(x)))
        log_sum  = np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        log_soft = scaled - log_sum                 # [N, C]
        return -log_soft[np.arange(len(labels)), labels].mean()

    result = minimize_scalar(nll, bounds=(0.05, 50.0), method="bounded")
    return float(result.x)


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Rescale probabilities by temperature T. Returns new softmax probs."""
    log_p    = np.log(probs.clip(min=1e-12))
    scaled   = log_p / T
    exp_s    = np.exp(scaled - scaled.max(axis=1, keepdims=True))  # numerically stable
    return exp_s / exp_s.sum(axis=1, keepdims=True)


# ── Calibration metric ────────────────────────────────────────────────────────

def compute_calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS):
    """Returns bin_conf, bin_acc, bin_count, ece."""
    confs   = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)

    edges     = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf  = np.full(n_bins, np.nan)
    bin_acc   = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confs >= lo) & (confs < hi) if b < n_bins - 1 else (confs >= lo) & (confs <= hi)
        n = mask.sum()
        bin_count[b] = n
        if n > 0:
            bin_conf[b] = confs[mask].mean()
            bin_acc[b]  = correct[mask].mean()

    n_total = len(labels)
    ece = sum(
        (bin_count[b] / n_total) * abs(bin_acc[b] - bin_conf[b])
        for b in range(n_bins) if bin_count[b] > 0
    )
    return bin_conf, bin_acc, bin_count, float(ece)


# ── Reliability diagram (one axis) ───────────────────────────────────────────

def draw_reliability(ax, bin_conf, bin_acc, bin_count, ece, title,
                     acc, mean_conf, T=None, highlight=False):
    BLUE    = "#2563EB"
    BAR_CLR = "#BFDBFE" if not highlight else "#A5F3FC"
    GRAY    = "#9CA3AF"
    RED_SH  = "#FCA5A5"
    GRN_SH  = "#BBF7D0"

    edges      = np.linspace(0.0, 1.0, N_BINS + 1)
    centres    = (edges[:-1] + edges[1:]) / 2
    width      = edges[1] - edges[0]
    valid      = ~np.isnan(bin_acc)

    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")

    # Fraction-correct bars
    ax.bar(centres[valid], bin_acc[valid], width=width * 0.85,
           color=BAR_CLR, edgecolor=BLUE, linewidth=1.1, zorder=2)

    # Gap shading
    for b in range(N_BINS):
        if not np.isnan(bin_conf[b]):
            lo = min(bin_acc[b], bin_conf[b])
            hi = max(bin_acc[b], bin_conf[b])
            shade = RED_SH if bin_acc[b] < bin_conf[b] else GRN_SH
            ax.bar(centres[b], hi - lo, bottom=lo, width=width * 0.85,
                   color=shade, alpha=0.5, zorder=3)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1.4, linestyle="--", zorder=4)

    # Mean-confidence dots
    ax.scatter(bin_conf[valid], bin_conf[valid],
               marker="o", s=28, color=BLUE, zorder=5)

    # Sample-count annotations
    for b in range(N_BINS):
        if bin_count[b] > 0 and not np.isnan(bin_acc[b]):
            ax.text(centres[b], bin_acc[b] + 0.025, str(bin_count[b]),
                    ha="center", va="bottom", fontsize=6, color="#374151")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.18)
    ax.tick_params(colors="#374151", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.35, color="#E5E7EB", zorder=0)

    t_str  = f"  T={T:.3f}" if T is not None else ""
    ax.set_title(
        f"{title}{t_str}\nECE={ece:.4f}  Acc={acc:.3f}  ConfMean={mean_conf:.3f}",
        fontsize=9, color="#111827", pad=5,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}\n")

    cfg.dataset    = "meld"
    cfg.split_mode = "native"
    cfg.audio_cache_dir = cfg.meld_audio_dir + "_cache"

    _, dev_df_full, test_df_full = load_meld_splits(SPLITS_DIR)

    # Filter to audio_exists (mirrors training)
    dev_df  = dev_df_full[dev_df_full["audio_exists"]].reset_index(drop=True)
    test_df = test_df_full[test_df_full["audio_exists"]].reset_index(drop=True)
    print(f"MELD dev : {len(dev_df)} utterances")
    print(f"MELD test: {len(test_df)} utterances\n")

    # Storage for results
    all_results = {}   # model → {before: (bin_conf,bin_acc,bin_count,ece), after: ..., T: float}

    # ── Per-model inference + temperature scaling ──────────────────────────
    for model_type in MODELS:
        print(f"{'='*55}")
        print(f"  Model: {model_type}")
        print(f"{'='*55}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, model_type, "meld_best.pt")
        print(f"  Loading checkpoint: {ckpt_path}")
        model = build_model(model_type).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Dev inference
        print(f"  Running inference on dev  ({len(dev_df)} utterances)...")
        dev_probs  = run_inference(model_type, dev_df, device, model)
        dev_labels = dev_df["label_idx"].to_numpy()
        save_predictions_with_probs(
            dev_df, dev_probs,
            f"./outputs/{model_type}/meld_dev_predictions_probs.csv"
        )

        # Test inference
        print(f"  Running inference on test ({len(test_df)} utterances)...")
        test_probs  = run_inference(model_type, test_df, device, model)
        test_labels = test_df["label_idx"].to_numpy()
        save_predictions_with_probs(
            test_df, test_probs,
            f"./outputs/{model_type}/meld_test_predictions_probs.csv"
        )

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        # Learn T on dev NLL
        T = learn_temperature(dev_probs, dev_labels)
        print(f"  Learned temperature T = {T:.4f}")

        # Calibrate test probabilities
        test_probs_cal = apply_temperature(test_probs, T)

        # ECE before and after on test set
        bc_before, ba_before, bn_before, ece_before = compute_calibration(test_probs,     test_labels)
        bc_after,  ba_after,  bn_after,  ece_after  = compute_calibration(test_probs_cal, test_labels)

        acc_before = (test_probs.argmax(1)     == test_labels).mean()
        acc_after  = (test_probs_cal.argmax(1) == test_labels).mean()
        conf_before = test_probs.max(1).mean()
        conf_after  = test_probs_cal.max(1).mean()

        print(f"  ECE before: {ece_before:.4f}  ->  ECE after: {ece_after:.4f}  "
              f"(delta={ece_after - ece_before:+.4f})")
        print(f"  Acc unchanged: before={acc_before:.4f}  after={acc_after:.4f}")
        print(f"  Mean conf:     before={conf_before:.4f}  after={conf_after:.4f}\n")

        all_results[model_type] = {
            "T":          T,
            "before":     (bc_before, ba_before, bn_before, ece_before, acc_before, conf_before),
            "after":      (bc_after,  ba_after,  bn_after,  ece_after,  acc_after,  conf_after),
        }

    # ── Summary table ──────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  {'Model':<8}  {'T':>6}  "
          f"{'ECE before':>11}  {'ECE after':>10}  {'Delta ECE':>10}  {'Mean conf before':>17}  {'Mean conf after':>16}")
    print("-" * 65)
    for m in MODELS:
        r = all_results[m]
        T = r["T"]
        eb  = r["before"][3];  cb = r["before"][5]
        ea  = r["after"][3];   ca = r["after"][5]
        print(f"  {m:<8}  {T:>6.3f}  {eb:>11.4f}  {ea:>10.4f}  {ea-eb:>+10.4f}  "
              f"{cb:>17.4f}  {ca:>16.4f}")
    print("=" * 65)

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    fig.patch.set_facecolor("#FAFAFA")

    col_titles = ["Before temperature scaling", "After temperature scaling"]
    for col, label in enumerate(col_titles):
        fig.text(
            0.27 + col * 0.47, 0.985, label,
            ha="center", va="top", fontsize=12, color="#111827",
            fontweight="bold",
        )

    for row, model_type in enumerate(MODELS):
        r   = all_results[model_type]
        T   = r["T"]
        bef = r["before"]
        aft = r["after"]

        ax_bef = axes[row][0]
        ax_aft = axes[row][1]

        draw_reliability(ax_bef, *bef[:4], f"{model_type.capitalize()} — before",
                         bef[4], bef[5], T=None, highlight=False)
        draw_reliability(ax_aft, *aft[:4], f"{model_type.capitalize()} — after",
                         aft[4], aft[5], T=T, highlight=True)

        ax_bef.set_xlabel("Mean confidence (max softmax)", fontsize=9, color="#374151")
        ax_aft.set_xlabel("Mean confidence (max softmax)", fontsize=9, color="#374151")
        ax_bef.set_ylabel("Fraction correct", fontsize=9, color="#374151")

        # ECE delta annotation on the after panel
        delta = aft[3] - bef[3]
        colour = "#16A34A" if delta < 0 else "#DC2626"
        ax_aft.text(
            0.97, 0.05, f"ΔECE {delta:+.4f}",
            transform=ax_aft.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=colour, fontweight="bold",
        )

    fig.text(
        0.5, 0.005,
        "10 equal-width bins  |  Numbers above bars = sample count  |  "
        "Red shading = overconfident, green = underconfident  |  "
        "T learned on MELD dev set (NLL minimisation)",
        ha="center", fontsize=7.5, color="#6B7280",
    )
    fig.suptitle(
        "Temperature Scaling Calibration — MELD Test Set",
        fontsize=13, color="#111827", y=1.0,
    )

    plt.tight_layout(rect=[0, 0.015, 1, 0.985])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    print(f"\nPlot saved to: {OUT_PNG}")


if __name__ == "__main__":
    main()
