"""
plot_calibration.py
──────────────────────────────────────────────────────────────────────────────
Reliability diagrams (calibration plots) for the three MELD models.

WHY RE-INFERENCE IS NEEDED
────────────────────────────
The existing meld_test_predictions.csv files contain only hard predictions
(no per-class probabilities). Calibration requires per-sample confidence
scores (max softmax probability). This script loads each saved checkpoint,
runs a single forward pass over the MELD test set, collects softmax
probabilities, then plots the reliability diagrams.

No training happens. Gradients are disabled throughout.

CALIBRATION METRIC — ECE
─────────────────────────
Expected Calibration Error (Guo et al., 2017):
  ECE = Σ_b (|B_b| / n) × |acc(B_b) - conf(B_b)|
  where B_b = samples in confidence bin b,
        acc(B_b) = fraction correct in that bin,
        conf(B_b) = mean max-softmax confidence in that bin.

10 equal-width bins, 0.0–1.0. Bins with zero samples are skipped.
Lower ECE = better calibrated. A perfectly calibrated model has ECE = 0.

Output: outputs/analysis/calibration_plots.png

Usage
─────
  # Run from project root
  python src/plot_calibration.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src to path so config/model imports resolve from project root
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

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "./checkpoints"
SPLITS_DIR     = "./data_splits/meld"
OUT_PNG        = "./outputs/analysis/calibration_plots.png"
N_BINS         = 10
CLASSES        = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
MODELS         = ["text", "audio", "fusion"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_model(model_type: str) -> torch.nn.Module:
    if model_type == "text":
        return RoBERTaLoRA()
    if model_type == "audio":
        return WavLMLoRA()
    return CrossModalAttentionFusion()


def build_test_loader(model_type: str, test_df: pd.DataFrame):
    if model_type == "text":
        ds = TextDataset(test_df)
    elif model_type == "audio":
        ds = AudioDataset(test_df)
    else:
        ds = MultimodalDataset(test_df)
    return build_dataloader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=False,   # single-pass; no worker reuse needed
    )


def run_inference(model_type: str, test_df: pd.DataFrame, device: torch.device):
    """
    Load checkpoint, run forward pass over test set, return
    (confidences, correctness) arrays — one entry per utterance.
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, model_type, "meld_best.pt")
    print(f"  Loading {model_type} checkpoint: {ckpt_path}")

    model = build_model(model_type).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = build_test_loader(model_type, test_df)

    use_amp = cfg.use_amp and device.type == "cuda"
    all_confs   = []
    all_correct = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)

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

            probs = F.softmax(logits.float(), dim=1)          # [batch, 5]
            confs, preds = probs.max(dim=1)                    # [batch]

            all_confs.extend(confs.cpu().tolist())
            all_correct.extend((preds == labels).cpu().tolist())

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()

    return np.array(all_confs, dtype=np.float32), np.array(all_correct, dtype=np.float32)


def compute_calibration(confs: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """
    Returns bin_conf, bin_acc, bin_count, ece for plotting.
    Bins with zero samples are included as NaN (skipped in plot).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf  = np.full(n_bins, np.nan)
    bin_acc   = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        # include right edge in last bin
        mask = (confs >= lo) & (confs < hi) if b < n_bins - 1 else (confs >= lo) & (confs <= hi)
        n = mask.sum()
        bin_count[b] = n
        if n > 0:
            bin_conf[b] = confs[mask].mean()
            bin_acc[b]  = correct[mask].mean()

    # ECE: weighted average of |acc - conf| over non-empty bins
    n_total = len(confs)
    ece = 0.0
    for b in range(n_bins):
        if bin_count[b] > 0:
            ece += (bin_count[b] / n_total) * abs(bin_acc[b] - bin_conf[b])

    return bin_conf, bin_acc, bin_count, ece


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_calibration(results: dict, out_path: str):
    """
    results: {model_type: (bin_conf, bin_acc, bin_count, ece, confs, correct)}
    """
    BLUE    = "#2563EB"
    RED     = "#DC2626"
    GRAY    = "#9CA3AF"
    BAR_CLR = "#BFDBFE"   # light blue bar fill

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    fig.patch.set_facecolor("#FAFAFA")

    for ax, model_type in zip(axes, MODELS):
        bin_conf, bin_acc, bin_count, ece, confs, correct = results[model_type]
        overall_acc = correct.mean()

        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D1D5DB")
        ax.spines["bottom"].set_color("#D1D5DB")

        bin_edges  = np.linspace(0.0, 1.0, N_BINS + 1)
        bin_width  = bin_edges[1] - bin_edges[0]
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        # ── Bars: fraction correct per bin ────────────────────────────────
        valid = ~np.isnan(bin_acc)
        ax.bar(
            bin_centres[valid], bin_acc[valid],
            width=bin_width * 0.85,
            color=BAR_CLR, edgecolor=BLUE, linewidth=1.2,
            label="Fraction correct",
            zorder=2,
        )

        # ── Gap shading: difference from perfect calibration ──────────────
        for b in range(N_BINS):
            if not np.isnan(bin_conf[b]):
                lo = min(bin_acc[b], bin_conf[b])
                hi = max(bin_acc[b], bin_conf[b])
                color = RED if bin_acc[b] < bin_conf[b] else "#BBF7D0"
                ax.bar(
                    bin_centres[b], hi - lo,
                    bottom=lo,
                    width=bin_width * 0.85,
                    color=color, alpha=0.35,
                    zorder=3,
                )

        # ── Perfect calibration diagonal ──────────────────────────────────
        ax.plot([0, 1], [0, 1], color=GRAY, lw=1.5,
                linestyle="--", label="Perfect calibration", zorder=4)

        # ── Mean confidence dot per bin ────────────────────────────────────
        ax.scatter(
            bin_conf[valid], bin_conf[valid],
            marker="o", s=30, color=BLUE, zorder=5, label="Mean confidence"
        )

        # ── Sample count annotations ───────────────────────────────────────
        for b in range(N_BINS):
            if bin_count[b] > 0 and not np.isnan(bin_acc[b]):
                ax.text(
                    bin_centres[b], bin_acc[b] + 0.02,
                    str(bin_count[b]),
                    ha="center", va="bottom", fontsize=6.5, color="#374151"
                )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.15)
        ax.set_xlabel("Mean confidence (max softmax)", fontsize=10, color="#374151")
        if ax is axes[0]:
            ax.set_ylabel("Fraction correct", fontsize=10, color="#374151")
        ax.set_title(
            f"{model_type.capitalize()}\nECE = {ece:.4f}  |  Acc = {overall_acc:.3f}",
            fontsize=11, color="#111827", pad=8
        )
        ax.tick_params(colors="#374151", labelsize=9)
        ax.grid(True, linestyle="--", linewidth=0.4, color="#E5E7EB", zorder=0)

        if ax is axes[0]:
            ax.legend(fontsize=8, framealpha=0.85, loc="upper left")

    # ── Shared footer ─────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "Reliability diagrams — MELD test set  |  10 equal-width bins  |  "
        "Number above each bar = sample count  |  "
        "Red shading = overconfident, green = underconfident",
        ha="center", fontsize=8, color="#6B7280",
    )
    fig.suptitle(
        "Calibration — MELD Test Set (text / audio / fusion)",
        fontsize=13, color="#111827", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    print(f"\nPlot saved to: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}")

    # Load MELD test split
    cfg.dataset    = "meld"
    cfg.split_mode = "native"
    cfg.audio_cache_dir = cfg.meld_audio_dir + "_cache"
    _, _, test_df = load_meld_splits(SPLITS_DIR)
    # Keep only rows where audio exists (mirrors train_native_split filtering)
    test_df = test_df[test_df["audio_exists"]].reset_index(drop=True)
    print(f"MELD test set: {len(test_df)} utterances (audio_exists=True)\n")

    results = {}
    for model_type in MODELS:
        print(f"[{model_type}] Running inference...")
        confs, correct = run_inference(model_type, test_df, device)
        bin_conf, bin_acc, bin_count, ece = compute_calibration(confs, correct, N_BINS)
        results[model_type] = (bin_conf, bin_acc, bin_count, ece, confs, correct)
        print(f"  ECE={ece:.4f}  Acc={correct.mean():.4f}  "
              f"Mean conf={confs.mean():.4f}  n={len(confs)}\n")

    # Print summary table
    print("=" * 50)
    print(f"{'Model':<10} {'ECE':>8} {'Acc':>8} {'MeanConf':>10}")
    print("-" * 50)
    for m in MODELS:
        _, _, _, ece, confs, correct = results[m]
        print(f"{m:<10} {ece:>8.4f} {correct.mean():>8.4f} {confs.mean():>10.4f}")
    print("=" * 50)

    plot_calibration(results, OUT_PNG)


if __name__ == "__main__":
    main()
