"""Generate result figures for the logbook from the computed CSVs.

Outputs (in reproduction/output/figures/):
  - results_mpjpe.png         per-sequence MPJPE bar chart (outliers highlighted)
  - results_depth_mask.png    per-sequence mask IoU + depth RMSE
"""
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reproduction" / "output"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DIFF_COLOR = {"easy": "#2e9e5b", "medium": "#e8a13a", "difficult": "#d9534f"}


def load_mpjpe():
    rows = []
    for line in (OUT / "all_sequences_metrics.csv").read_text().splitlines()[1:]:
        seq, diff, frames, m, med, mx, e2m, e2med, e2mx = line.split(",")
        rows.append((seq, diff, float(m), float(mx), float(e2m)))
    return rows


def load_depth():
    rows = []
    for line in (OUT / "depth_mask_all_sequences.csv").read_text().splitlines()[1:]:
        seq, diff, iou, mae, rmse = line.split(",")
        rows.append((seq, diff, float(iou), float(rmse)))
    return rows


def fig_mpjpe():
    rows = load_mpjpe()
    seqs = [r[0] for r in rows]
    vals = [r[2] for r in rows]
    colors = [DIFF_COLOR[r[1]] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(seqs, vals, color=colors)
    ax.axhline(5.0, color="gray", ls="--", lw=1, label="~5 mm (typical)")
    ax.set_xlabel("Sequence")
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("3D joint reproduction error per sequence")
    ax.set_ylim(0, max(vals) * 1.15)
    for s, v in zip(seqs, vals):
        ax.annotate(f"{v:.1f}", (s, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
    # highlight outliers
    for s in ("05", "11"):
        i = seqs.index(s)
        bars[i].set_edgecolor("black")
        bars[i].set_linewidth(2)
    from matplotlib.patches import Patch
    legend = [Patch(color=DIFF_COLOR[d], label=d) for d in ("easy", "medium", "difficult")]
    legend.append(plt.Line2D([], [], color="black", lw=2, label="outlier (05, 11)"))
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "results_mpjpe.png", dpi=130)
    plt.close(fig)


def fig_depth_mask():
    rows = load_depth()
    seqs = [r[0] for r in rows]
    iou = [r[2] for r in rows]
    rmse = [r[3] for r in rows]
    colors = [DIFF_COLOR[r[1]] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(seqs, iou, color=colors)
    ax1.set_xlabel("Sequence"); ax1.set_ylabel("Mask IoU")
    ax1.set_title("Foreground mask reproduction (IoU)")
    ax1.set_ylim(0, 1.0)
    for s, v in zip(seqs, iou):
        ax1.annotate(f"{v:.2f}", (s, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=7)

    ax2.bar(seqs, rmse, color=colors)
    ax2.set_xlabel("Sequence"); ax2.set_ylabel("Depth RMSE (mm)")
    ax2.set_title("Depth reproduction error (RMSE)")
    for s, v in zip(seqs, rmse):
        ax2.annotate(f"{v:.0f}", (s, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "results_depth_mask.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_mpjpe()
    fig_depth_mask()
    print(f"saved to {FIG}")
    for f in FIG.iterdir():
        print("  ", f.name)
