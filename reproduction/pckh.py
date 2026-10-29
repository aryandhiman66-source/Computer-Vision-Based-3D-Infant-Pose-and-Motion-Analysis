"""Step 6: Reproduce the PCKh metric on our reconstruction vs dataset GT.

PCKh = fraction of joints whose error < alpha * head_segment_length.
Head sizes come from each sequence's tpose folder (2D px, 3D metres).
Sampled frames keep this fast (50 frames/sequence x 25 joints).

Run:  python reproduction/pckh.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import geometry
import loader

ALPHAS = [0.5, 1.0, 2.0]
N_FRAMES = 50


def difficulty(seq: str) -> str:
    n = int(seq)
    return "easy" if n <= 4 else ("medium" if n <= 9 else "difficult")


def head_sizes(seq: str):
    d = config.DATASET_DIR / seq / "tpose"
    h2 = float(np.loadtxt(d / "tpose_PCKh_value.txt"))        # px
    h3 = float(np.loadtxt(d / "tpose_PCKh_value_3D.txt"))     # m
    return h2, h3


def main():
    model = loader.load_smil_model()
    seqs = sorted(p.name for p in config.DATASET_DIR.glob("[0-9][0-9]"))
    nose_idx = 331

    rows = []
    print(f"{'seq':>3} {'group':>9} | {'PCKh3D@0.5':>10} {'PCKh3D@1.0':>10} {'PCKh3D@2.0':>10} | "
          f"{'PCKh2D@0.5':>10} {'PCKh2D@1.0':>10} {'PCKh2D@2.0':>10}")
    print("-" * 96)

    for seq in seqs:
        h2, h3 = head_sizes(seq)
        data0 = loader.load_frame(seq, 0)
        model.betas[:] = data0["betas"]

        err3d_all, err2d_all = [], []
        for f in range(0, 1000, 1000 // N_FRAMES):
            data = loader.load_frame(seq, f)
            model.pose[:] = data["pose"]
            model.trans[:] = data["trans"]
            verts = loader.as_np(model.r)
            Jtr = loader.as_np(model.J_transformed)

            gt3d = data["joints3d"][:24, :3]
            err3d_all.append(np.linalg.norm(Jtr - gt3d, axis=1))

            proj = geometry.project_points(np.vstack([Jtr, verts[nose_idx]]))
            err2d_all.append(np.linalg.norm(proj - data["joints2d"][:, :2], axis=1))

        err3d = np.concatenate(err3d_all)          # metres
        err2d = np.concatenate(err2d_all)          # px

        p3 = [float((err3d < a * h3).mean() * 100) for a in ALPHAS]
        p2 = [float((err2d < a * h2).mean() * 100) for a in ALPHAS]

        rows.append([seq, difficulty(seq), h3, h2, *p3, *p2])
        print(f"{seq:>3} {difficulty(seq):>9} | {p3[0]:>9.2f}% {p3[1]:>9.2f}% {p3[2]:>9.2f}% | "
              f"{p2[0]:>9.2f}% {p2[1]:>9.2f}% {p2[2]:>9.2f}%")

    out = config.OUTPUT_DIR / "pckh_all_sequences.csv"
    with open(out, "w") as fp:
        fp.write("sequence,difficulty,head3d_m,head2d_px,"
                 "pckh3d_0.5,pckh3d_1.0,pckh3d_2.0,pckh2d_0.5,pckh2d_1.0,pckh2d_2.0\n")
        for r in rows:
            fp.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
