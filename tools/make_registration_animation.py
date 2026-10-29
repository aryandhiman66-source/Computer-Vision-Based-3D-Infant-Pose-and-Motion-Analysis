"""Animated illustration of SMIL-style registration (for reports / slides).

A simplified 2D "puppet" (shape beta1 + pose theta + position/rotation) is fitted to
noisy camera points and OpenPose-like joint detections by minimising
    E = E_data + E_lm + E_beta + E_theta
with a simple coordinate-descent optimiser. This is a teaching illustration of the
idea in Hesse et al. (MICCAI 2018), not the real 3D SMIL fit.

Outputs (reproduction/output/figures/):
    registration_fit.gif          animation for PowerPoint / Word
    registration_storyboard.png   4 snapshots for printed reports / PDFs

Run:  python tools/make_registration_animation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = Path(__file__).resolve().parent.parent / "reproduction" / "output" / "figures"

GREY, AMBER, PURPLE, PURPLE_D, GREEN, CORAL = "#888780", "#BA7517", "#7F77DD", "#534AB7", "#1D9E75", "#D85A30"
KEYS = ["x", "y", "r", "b", "sh", "hp"]
LO = dict(x=-80, y=-60, r=-45, b=-2, sh=0, hp=0)
HI = dict(x=80, y=60, r=45, b=2, sh=170, hp=70)
TRUE = dict(x=10, y=5, r=6, b=0.7, sh=85, hp=22)
START = dict(x=-55, y=-30, r=-25, b=-1.2, sh=30, hp=55)


def model(p):
    """Return surface points (M,2), joints {name: (2,)}, bone segments [(a,b)]."""
    s = 1 + 0.14 * p["b"]
    J = {"neck": np.array([0, -58 * s]), "head": np.array([0, -90 * s]), "pelvis": np.array([0, 55 * s])}
    limbs = []
    for d, n in ((-1, "L"), (1, "R")):
        S = np.array([d * 36 * s, -46 * s])
        a, a2 = np.radians(p["sh"]), np.radians(p["sh"] + 30)
        E = S + np.array([d * np.sin(a), np.cos(a)]) * 40 * s
        W = E + np.array([d * np.sin(a2), np.cos(a2)]) * 34 * s
        H = np.array([d * 20 * s, 52 * s])
        h, h2 = np.radians(p["hp"]), np.radians(p["hp"] - 25)
        K = H + np.array([d * np.sin(h), np.cos(h)]) * 44 * s
        A = K + np.array([d * np.sin(h2), np.cos(h2)]) * 38 * s
        J.update({f"sho{n}": S, f"elb{n}": E, f"wri{n}": W, f"hip{n}": H, f"kne{n}": K, f"ank{n}": A})
        limbs += [(S, E, 8 * s), (E, W, 7 * s), (H, K, 11 * s), (K, A, 10 * s)]
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    pts = [np.c_[40 * s * np.cos(t), 62 * s * np.sin(t)]]
    t = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    pts.append(np.c_[J["head"][0] + 28 * s * np.cos(t), J["head"][1] + 28 * s * np.sin(t)])
    for a, b, w in limbs:
        v = b - a
        nrm = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-9)
        c = a + np.outer(np.linspace(0, 1, 5), v)
        pts += [c + nrm * w, c - nrm * w]
    pts = np.vstack(pts)
    c, sn = np.cos(np.radians(p["r"])), np.sin(np.radians(p["r"]))
    R = np.array([[c, -sn], [sn, c]])
    off = np.array([p["x"], p["y"]])
    tf = lambda q: q @ R.T + off
    bones = [("head", "neck"), ("neck", "pelvis"), ("neck", "shoL"), ("neck", "shoR"), ("shoL", "elbL"), ("elbL", "wriL"),
             ("shoR", "elbR"), ("elbR", "wriR"), ("pelvis", "hipL"), ("pelvis", "hipR"), ("hipL", "kneL"), ("kneL", "ankL"),
             ("hipR", "kneR"), ("kneR", "ankR")]
    JT = {k: tf(v) for k, v in J.items()}
    return tf(pts), JT, [(JT[a], JT[b]) for a, b in bones]


rng = np.random.default_rng(7)
_tp, _tj, _ = model(TRUE)
SCAN = _tp[rng.random(len(_tp)) < 0.85]                  # ~15% of the surface unseen (occlusion)
SCAN = SCAN + rng.uniform(-3, 3, SCAN.shape)
LM = {k: v + rng.uniform(-4, 4, 2) for k, v in _tj.items()}


def energy(p):
    pts, J, _ = model(p)
    d = np.sqrt(((SCAN[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(1)
    Ed = d.mean()
    El = np.mean([np.linalg.norm(LM[k] - J[k]) for k in LM])
    Eb = 3 * p["b"] ** 2
    Et = 4 * (((p["sh"] - 70) / 70) ** 2 + ((p["hp"] - 20) / 40) ** 2)
    return dict(Ed=Ed, El=El, Eb=Eb, Et=Et, E=Ed + El + Eb + Et)


def optimise():
    """Coordinate descent; returns the parameter state after every accepted move."""
    P, step = dict(START), dict(x=8, y=8, r=6, b=0.3, sh=10, hp=8)
    hist = [dict(P)]
    for _ in range(400):
        moved = False
        for k in KEYS:
            cur = energy(P)["E"]
            for d in (1, -1):
                q = dict(P)
                q[k] = float(np.clip(q[k] + d * step[k], LO[k], HI[k]))
                if energy(q)["E"] < cur - 1e-4:
                    P, moved = q, True
                    hist.append(dict(P))
                    break
        if not moved:
            step = {k: v * 0.6 for k, v in step.items()}
            if max(step.values()) < 0.02:
                break
    return hist


def draw_scene(ax, p, show_model=True, title=None):
    ax.clear()
    ax.set_xlim(-150, 150)
    ax.set_ylim(190, -150)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.scatter(SCAN[:, 0], SCAN[:, 1], s=14, c=GREY, zorder=1)
    if show_model:
        pts, J, bones = model(p)
        for a, b in bones:
            ax.plot([a[0], b[0]], [a[1], b[1]], c=PURPLE, lw=1.4, alpha=0.7, zorder=2)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c=PURPLE, zorder=3)
        for k, j in J.items():
            ax.plot([j[0], LM[k][0]], [j[1], LM[k][1]], c=AMBER, lw=0.6, ls=(0, (2, 2)), zorder=2)
        ax.scatter([j[0] for j in J.values()], [j[1] for j in J.values()], s=26, facecolors="white",
                   edgecolors=PURPLE_D, linewidths=1.3, zorder=4)
    ax.scatter([v[0] for v in LM.values()], [v[1] for v in LM.values()], marker="x", s=36, c=AMBER, linewidths=1.8, zorder=5)
    if title:
        ax.set_title(title, fontsize=12, color="#2C2C2A", pad=6)


def draw_bars(ax, e, e0):
    ax.clear()
    names = ["E_data  points far from surface", "E_lm  joints far from detections", "E_β  unusual body shape",
             "E_θ  unusual pose", "Total error E"]
    vals = [e["Ed"], e["El"], e["Eb"], e["Et"], e["E"]]
    scale = [40, 60, 12, 8, e0]
    cols = [GREY, AMBER, GREEN, PURPLE, CORAL]
    y = np.arange(len(names))[::-1]
    ax.barh(y, [1] * 5, color="#F1EFE8", height=0.55)
    ax.barh(y, [min(1, v / s) for v, s in zip(vals, scale)], color=cols, height=0.55)
    for yi, n, v in zip(y, names, vals):
        ax.text(0, yi + 0.42, n, fontsize=9.5, color="#444441", va="bottom",
                fontweight="bold" if n.startswith("Total") else "normal")
        ax.text(1.02, yi, f"{v:.1f}", fontsize=10, va="center", color="#2C2C2A")
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.6, 5)
    ax.axis("off")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hist = optimise()
    # ~90 optimisation frames, denser at the start where the motion is largest
    idx = np.unique(np.round(np.linspace(0, 1, 90) ** 1.6 * (len(hist) - 1)).astype(int))
    fit_frames = [hist[i] for i in idx]
    curve = [energy(h)["E"] for h in fit_frames]
    e0 = energy(START)["E"]
    fps = 12
    frames = ([("scan", START, None)] * int(1.5 * fps) + [("start", START, None)] * int(1.5 * fps)
              + [("fit", p, i) for i, p in enumerate(fit_frames)] + [("done", fit_frames[-1], len(fit_frames) - 1)] * int(3 * fps))

    fig = plt.figure(figsize=(12, 6), dpi=80)
    fig.patch.set_facecolor("white")
    ax_s = fig.add_axes([0.02, 0.10, 0.50, 0.80])
    ax_b = fig.add_axes([0.56, 0.42, 0.40, 0.48])
    ax_c = fig.add_axes([0.60, 0.12, 0.34, 0.22])
    head = fig.text(0.02, 0.95, "", fontsize=15, color="#2C2C2A", va="top")
    foot = fig.text(0.02, 0.03, "Illustrative 2D toy model of SMIL registration (Hesse et al., MICCAI 2018). Grey = depth-camera points, "
                    "orange x = OpenPose joints, purple = body model.", fontsize=9, color="#5F5E5A")

    def update(f):
        phase, p, i = frames[f]
        if phase == "scan":
            head.set_text("1. The camera sees a cloud of 3D points; OpenPose finds 2D joints")
            draw_scene(ax_s, p, show_model=False)
            ax_b.clear(); ax_b.axis("off"); ax_c.clear(); ax_c.axis("off")
            return
        e = energy(p)
        if phase == "start":
            head.set_text("2. Start: the body model (β = shape, θ = pose) is placed roughly")
        elif phase == "fit":
            head.set_text("3. The optimiser changes β and θ step by step to shrink the total error")
        else:
            head.set_text("4. Fitted: the model now explains the camera data. β and θ are the result")
        draw_scene(ax_s, p)
        draw_bars(ax_b, e, e0)
        ax_c.clear()
        if i is not None:
            ax_c.plot(range(i + 1), curve[: i + 1], c=CORAL, lw=2)
        ax_c.set_xlim(0, len(curve))
        ax_c.set_ylim(0, e0 * 1.05)
        ax_c.set_xlabel("optimisation step", fontsize=9, color="#5F5E5A")
        ax_c.set_ylabel("total error", fontsize=9, color="#5F5E5A")
        ax_c.tick_params(labelsize=8, colors="#5F5E5A")
        for sp in ("top", "right"):
            ax_c.spines[sp].set_visible(False)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    gif = OUT / "registration_fit.gif"
    anim.save(gif, writer=PillowWriter(fps=fps))
    plt.close(fig)

    # storyboard for printed reports
    snaps = [(START, "Start (rough placement)"), (fit_frames[len(fit_frames) // 6], "Early steps"),
             (fit_frames[len(fit_frames) // 2], "Halfway"), (fit_frames[-1], "Fitted")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), dpi=150)
    for ax, (p, t) in zip(axes, snaps):
        draw_scene(ax, p, title=f"{t}\ntotal error {energy(p)['E']:.1f}")
    fig.suptitle("Registration: fitting a body model to camera points and detected joints by minimising "
                 "E = E_data + E_lm + E_β + E_θ (illustrative 2D toy)", fontsize=11, color="#2C2C2A")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png = OUT / "registration_storyboard.png"
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    print(f"wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)\nwrote {png}")
    print("start error %.1f -> fitted %.1f ; true params %s ; fitted %s" % (
        e0, curve[-1], TRUE, {k: round(v, 2) for k, v in fit_frames[-1].items()}))


if __name__ == "__main__":
    main()
