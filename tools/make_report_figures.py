"""Figures for the MSc final report (classification part).

Reads paper/per_fold.csv and the numbers in paper/fact_sheet.md (copied below,
regenerate the fact sheet with classification/paper_numbers.py if they change).
Writes report/figures/fig_*.png at 200 dpi.

Run:  python tools/make_report_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY, BLUE, ORANGE = "#a3a29c", "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def per_fold():
    rows = {}
    with open(ROOT / "paper" / "per_fold.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["model"], r["metric"])] = [float(r[f"fold{k}"]) for k in range(1, 10)]
    return rows


# ---------------------------------------------------------------- 1. pipeline
def fig_pipeline():
    steps = [
        ("RGB video\nof the infant", "#f0efec"),
        ("ViTPose\n15 body keypoints", "#f0efec"),
        ("Per-joint centring\n+ velocity, z-score\n250 frames x 60", "#f0efec"),
        ("1D-conv CNN\n(paper config 1)\n+ mirror /\nrotation aug", "#cde2fb"),
        ("16 seeds\naverage\nprobabilities", "#cde2fb"),
        ("FM+ / FM-", "#f0efec"),
    ]
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.set_xlim(0, 6 * 1.8); ax.set_ylim(0, 1.6); ax.axis("off")
    for i, (txt, col) in enumerate(steps):
        x = i * 1.8 + 0.1
        ax.add_patch(FancyBboxPatch((x, 0.3), 1.45, 1.0, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=col, ec=INK2, lw=0.8))
        ax.text(x + 0.725, 0.8, txt, ha="center", va="center", fontsize=8.5, color=INK)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.75, 0.8), xytext=(x + 1.5, 0.8),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    ax.text(3 * 1.8 + 0.1 + 1.55, 0.12, "blue = what this project changed", ha="center",
            fontsize=8, color=INK2)
    save(fig, "fig_pipeline.png")


# ------------------------------------------------- 2. paper configs vs ours
def fig_paper_vs_ours():
    paper = [90.66, 90.51, 89.44, 89.36, 88.57, 88.28, 88.22, 88.12, 88.11, 87.81]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.scatter(paper, [2] * 10, s=60, color=GREY, edgecolor="white", linewidth=1.5, zorder=3)
    ax.axvline(88.9, ymin=0.62, ymax=0.95, color=GREY, ls="--", lw=1)
    ax.text(88.9, 2.42, "mean 88.9", ha="center", fontsize=8, color=INK2)
    ax.text(90.66, 2.42, "published 90.7\n(best of 10)", ha="center", fontsize=8, color=INK2)
    ours = [("faithful reproduction\n(paper rule)", 89.4, None, GREY),
            ("final model, 16-seed ensemble\n(fresh seeds; bar = 95% CI over folds)", 92.0, (89.9, 94.1), BLUE)]
    for lab, v, ci, col in ours:
        if ci:
            ax.plot(ci, [1, 1], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([v], [1], s=80, color=col, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(v, 0.55, lab, ha="center", va="top", fontsize=8, color=INK)
    ax.set_yticks([1, 2]); ax.set_yticklabels(["this project", "Kulvicius et al.\n2025 (10 configs)"])
    ax.set_ylim(0, 2.8); ax.set_xlim(87, 94.6)
    ax.set_xlabel("balanced accuracy (%), 9 infant-wise folds")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "fig_paper_vs_ours.png")


# ------------------------------------------------------------- 3. per fold
def fig_per_fold(rows):
    base = rows[("Faithful reproduction, seeds 1", "paper")]
    final = rows[("FINAL: 8 plain + 8 aug, fresh seeds", "ens")]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    xs = range(1, 10)
    for x, a, b in zip(xs, base, final):
        ax.plot([x, x], [a, b], color=GRID, lw=2, zorder=1)
    ax.scatter(xs, base, s=55, color=GREY, edgecolor="white", linewidth=1.5, zorder=3,
               label="single model, paper's selection rule")
    ax.scatter(xs, final, s=55, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3,
               label="final model (16-seed ensemble)")
    ax.axhline(90.66, color=INK2, ls=":", lw=1)
    ax.text(9.6, 90.66, "paper\n90.7", va="center", fontsize=8, color=INK2)
    ax.set_xticks(list(xs)); ax.set_xlabel("test fold (4 infants each)")
    ax.set_ylabel("balanced accuracy (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    ax.set_ylim(82, 98); ax.set_xlim(0.5, 10.3)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Final model wins 8 of 9 folds (Wilcoxon p = 0.012)", fontsize=10, color=INK, loc="left")
    save(fig, "fig_per_fold.png")


# -------------------------------------------------------- 4. age confound
def fig_age_confound():
    items = [("age only (no video)", 98.6, ORANGE),
             ("14 static body numbers (no motion)", 90.5, ORANGE),
             ("limb proportions only", 75.4, ORANGE),
             ("hand-crafted motion statistics", 75.1, GREY),
             ("this project's CNN (motion, centred)", 92.0, BLUE)]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for i, (lab, v, col) in enumerate(items[::-1]):
        ax.barh(i, v - 50, left=50, height=0.55, color=col)
        ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([t[0] for t in items[::-1]])
    ax.set_xlim(50, 103); ax.axvline(50, color=INK2, lw=0.8)
    ax.set_xlabel("balanced accuracy (%), same 9 folds (50 = chance)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("The label is almost fully predicted by age (orange = shortcut features)",
                 fontsize=10, color=INK, loc="left")
    save(fig, "fig_age_confound.png")


# --------------------------------------------------------- 5. experiments
def fig_experiments():
    exps = [  # (name, diff vs matched baseline in pts, p, adopted)
        ("mirror + rotation augmentation (16 seeds)", +1.30, 0.023, True),
        ("extra 337 tuning clips, seeds 0", +0.78, 0.25, False),
        ("limb-relative features", +0.27, 0.46, False),
        ("other paper configurations (ensemble)", +0.04, 0.74, False),
        ("extra 337 tuning clips, fresh seeds", -0.47, 0.57, False),
        ("IMU auxiliary supervision", -0.89, 0.38, False),
        ("temporal crop 4-5 s", -1.00, 0.25, False),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for i, (lab, d, p, ok) in enumerate(exps[::-1]):
        ax.barh(i, d, height=0.55, color=BLUE if ok else GREY)
        ax.text(d + (0.06 if d >= 0 else -0.06), i, f"{d:+.2f}  (p={p:g})", va="center",
                ha="left" if d >= 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(exps))); ax.set_yticklabels([e[0] for e in exps[::-1]], fontsize=8.5)
    ax.set_xlim(-2.0, 2.3)
    ax.set_xlabel("change in ensemble balanced accuracy vs matched baseline (pts)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("Only augmentation survived (blue = adopted)", fontsize=10, color=INK, loc="left")
    save(fig, "fig_experiments.png")


def _box(ax, x, y, w, h, txt, col="#f0efec", size=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=col, ec=INK2, lw=0.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal")


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))


# ------------------------------------------------------ 6. system overview
def fig_overview():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    # data sources
    _box(ax, 0.1, 3.3, 2.6, 1.0, "MINI-RGBD\n12 sequences, RGB + depth,\nSMIL parameters", "#e7e6e1")
    _box(ax, 0.1, 0.4, 2.6, 1.0, "Kulvicius et al. (2025)\nViTPose keypoints,\n36 infants, 1,346 clips", "#e7e6e1")
    # parts
    _box(ax, 3.4, 3.3, 2.2, 1.0, "A. Forward\nreconstruction\n~5 mm (10/12)")
    _box(ax, 6.0, 3.3, 2.2, 1.0, "B. Registration\n82.2 -> 3.75 mm")
    _box(ax, 8.6, 3.3, 2.3, 1.0, "C. RGB-D preprocessing\ntable removal IoU 1.000\nlifting 26-35 mm")
    _box(ax, 3.4, 0.4, 2.2, 1.0, "D1. Faithful\nreproduction\n89.4%")
    _box(ax, 6.0, 0.4, 2.2, 1.0, "D2. Augmentation +\n16-seed ensemble\n92.0%", "#cde2fb", bold=True)
    _box(ax, 8.6, 0.4, 2.3, 1.0, "D3. Age-confound\nanalysis\nage alone 98.6%")
    for (x0, x1, y) in [(2.7, 3.4, 3.8), (5.6, 6.0, 3.8), (8.2, 8.6, 3.8),
                        (2.7, 3.4, 0.9), (5.6, 6.0, 0.9), (8.2, 8.6, 0.9)]:
        _arrow(ax, x0, y, x1, y)
    ax.text(5.5, 2.35, "3D body model and depth geometry (Parts A-C)   |   movement classification (Part D)",
            ha="center", fontsize=8.5, color=INK2)
    ax.plot([0.1, 10.9], [2.15, 2.15], color=GRID, lw=1)
    save(fig, "fig_overview.png")


# ------------------------------------------------------ 7. CNN architecture
def fig_cnn():
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 2.4); ax.axis("off")
    blocks = [("Input\n250 frames x 60\n(30 pos + 30 vel)", "#e7e6e1"),
              ("Conv block 1\n4 kernels, width 13", "#f0efec"),
              ("Conv block 2\n32 kernels, width 25", "#f0efec"),
              ("Conv block 3\n16 kernels, width 25", "#f0efec"),
              ("Average pool\nover time", "#f0efec"),
              ("Dense 128\n-> sigmoid\nP(FM+)", "#cde2fb")]
    w, gap = 1.55, 0.27
    for i, (t, c) in enumerate(blocks):
        x = 0.1 + i * (w + gap)
        _box(ax, x, 0.75, w, 1.15, t, c, size=8.2)
        if i < len(blocks) - 1:
            _arrow(ax, x + w, 1.32, x + w + gap, 1.32)
    ax.text(5.5, 0.35, "each conv block: Conv -> ReLU -> BatchNorm -> Dropout 0.2; kernels slide along time and are shared by all 60 columns",
            ha="center", fontsize=8, color=INK2)
    save(fig, "fig_cnn.png")


# ------------------------------------------------------ 8. improvement steps
def fig_steps():
    steps = [("First port", 86.5, GREY), ("Faithful\nreproduction", 89.4, GREY),
             ("+ augmentation\n(single model)", 90.2, GREY), ("+ 16-network\nensemble", 92.0, BLUE)]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    xs = range(len(steps))
    ax.plot(list(xs), [s[1] for s in steps], color=GRID, lw=2, zorder=1)
    for i, (lab, v, c) in enumerate(steps):
        ax.scatter([i], [v], s=90, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(i, v + 0.45, f"{v:.1f}%", ha="center", fontsize=9.5, color=INK,
                fontweight="bold" if c == BLUE else "normal")
    ax.axhline(90.66, color=INK2, ls=":", lw=1)
    ax.text(3.45, 90.66, "published\n90.7%", va="center", fontsize=8, color=INK2)
    ax.set_xticks(list(xs)); ax.set_xticklabels([s[0] for s in steps], fontsize=8.5)
    ax.set_ylim(85, 93.5); ax.set_xlim(-0.4, 3.9)
    ax.set_ylabel("balanced accuracy (%)")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Three steps took video-only accuracy from 86.5% to 92.0%", fontsize=10, color=INK, loc="left")
    save(fig, "fig_steps.png")


# ------------------------------------------------------ 9. cross-validation
def fig_cv():
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for k in range(9):
        for g in range(9):
            test = g == k
            ax.add_patch(FancyBboxPatch((g * 0.78 + 1.2, 8 - k), 0.7, 0.72,
                                        boxstyle="round,pad=0.01,rounding_size=0.05",
                                        fc=BLUE if test else "#f0efec", ec="white", lw=1))
        ax.text(1.05, 8 - k + 0.36, f"fold {k + 1}", ha="right", va="center", fontsize=8, color=INK2)
    ax.text(1.2 + 9 * 0.78 + 0.15, 4.4, "blue = 4 test infants\ngrey = 32 training infants\n(1/8 of training clips\nused for validation)",
            fontsize=8, color=INK2, va="center")
    ax.set_xlim(0, 11.2); ax.set_ylim(-0.2, 9.1); ax.axis("off")
    ax.set_title("Infant-wise 9-fold cross-validation: no infant is ever in both training and test",
                 fontsize=10, color=INK, loc="left")
    save(fig, "fig_cv.png")


if __name__ == "__main__":
    fig_steps()
    fig_cv()
    fig_overview()
    fig_cnn()
    fig_pipeline()
    fig_paper_vs_ours()
    fig_per_fold(per_fold())
    fig_age_confound()
    fig_experiments()
