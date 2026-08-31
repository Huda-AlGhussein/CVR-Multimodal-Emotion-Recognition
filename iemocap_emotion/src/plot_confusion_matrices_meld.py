"""
plot_confusion_matrices_meld.py
Plots row-normalised confusion matrices for the three MELD models as heatmaps.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

LABELS = ["anger", "fear", "joy", "neutral", "sadness"]

cms = {
    "Text-only": np.array([
        [0.522, 0.072, 0.177, 0.197, 0.032],
        [0.140, 0.340, 0.120, 0.260, 0.140],
        [0.087, 0.037, 0.701, 0.157, 0.017],
        [0.046, 0.053, 0.091, 0.776, 0.033],
        [0.159, 0.111, 0.115, 0.337, 0.279],
    ]),
    "Audio-only": np.array([
        [0.446, 0.026, 0.238, 0.159, 0.130],
        [0.180, 0.100, 0.260, 0.260, 0.200],
        [0.117, 0.032, 0.430, 0.266, 0.154],
        [0.110, 0.042, 0.217, 0.474, 0.158],
        [0.115, 0.082, 0.178, 0.212, 0.413],
    ]),
    "Fusion": np.array([
        [0.577, 0.078, 0.139, 0.133, 0.072],
        [0.140, 0.340, 0.100, 0.260, 0.160],
        [0.085, 0.035, 0.662, 0.184, 0.035],
        [0.076, 0.054, 0.085, 0.708, 0.076],
        [0.144, 0.139, 0.096, 0.226, 0.394],
    ]),
}

OUT = "./outputs/analysis/confusion_matrices_meld.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
fig.patch.set_facecolor("#FAFAFA")

for ax, (title, cm) in zip(axes, cms.items()):
    # Draw heatmap without annotations — we add them manually to bold diagonal
    sns.heatmap(
        cm,
        ax=ax,
        cmap="Blues",
        vmin=0.0, vmax=1.0,
        xticklabels=LABELS,
        yticklabels=LABELS,
        linewidths=0.4,
        linecolor="#E5E7EB",
        cbar=ax is axes[-1],          # colorbar only on rightmost panel
        annot=False,
    )

    # Annotate cells manually so diagonal can be bolded
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            val   = cm[i, j]
            diag  = (i == j)
            color = "white" if val > 0.55 else "#111827"
            ax.text(
                j + 0.5, i + 0.5,
                f"{val:.3f}",
                ha="center", va="center",
                fontsize=9.5,
                color=color,
                fontweight="bold" if diag else "normal",
            )

    ax.set_title(title, fontsize=13, color="#111827", pad=10, fontweight="semibold")
    ax.set_xlabel("Predicted", fontsize=10, color="#374151", labelpad=6)
    ax.set_ylabel("True" if ax is axes[0] else "", fontsize=10, color="#374151", labelpad=6)
    ax.tick_params(axis="both", labelsize=9, colors="#374151")
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_yticklabels(LABELS, rotation=0)

# Shared colorbar label
if hasattr(axes[-1], "collections") and axes[-1].collections:
    cbar = axes[-1].collections[0].colorbar
    if cbar is not None:
        cbar.set_label("Fraction of true-class utterances", fontsize=9, color="#374151")
        cbar.ax.tick_params(labelsize=8)

fig.suptitle(
    "Confusion Matrices — MELD Test Set (row-normalised)",
    fontsize=13, color="#111827", y=1.02,
)
fig.text(
    0.5, -0.02,
    "Diagonal cells (bold) = correctly classified fraction  |  "
    "Row sums to 1.0 per true class",
    ha="center", fontsize=8, color="#6B7280",
)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
print(f"Saved to: {OUT}")
