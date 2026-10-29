"""Step 2: Inventory all 12 MINI-RGBD sequences (dataset-integrity check).

Reports, per sequence: frame count, file counts (rgb/depth/mask/joints),
parameter dimensions, and difficulty group. Writes dataset_integrity.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config


def difficulty(seq: str) -> str:
    n = int(seq)
    if n <= 4:
        return "easy"
    if n <= 9:
        return "medium"
    return "difficult"


def count_files(d: Path, pattern: str) -> int:
    return len(list(d.glob(pattern)))


def main():
    seqs = sorted(p.name for p in config.DATASET_DIR.glob("[0-9][0-9]"))
    rows = []
    print(f"{'seq':>3} {'group':>9} {'frames':>7} {'rgb':>5} {'depth':>6} {'mask':>5} "
          f"{'j3d':>5} {'j2d':>5} {'betas':>6} {'pose':>5} {'trans':>6} {'jrows':>6} {'tpose':>6}")
    print("-" * 96)

    for seq in seqs:
        d = config.DATASET_DIR / seq
        n_frames = count_files(d / "smil_params", "*_pose.txt")
        n_rgb = count_files(d / "rgb", "*.png")
        n_depth = count_files(d / "depth", "*.png")
        n_mask = count_files(d / "fg_mask", "*.png")
        n_j3d = count_files(d / "joints_3D", "*.txt")
        n_j2d = count_files(d / "joints_2Ddep", "*.txt")

        betas = np.loadtxt(d / "smil_shape_betas.txt")
        pose0 = np.loadtxt(d / "smil_params" / "00000_pose.txt")
        trans0 = np.loadtxt(d / "smil_params" / "00000_trans.txt")
        j3d0 = np.loadtxt(d / "joints_3D" / "syn_joints_3D_00000.txt")

        has_tpose = (d / "tpose" / "syn_joints_3D_tpose.txt").exists()

        rows.append([seq, difficulty(seq), n_frames, n_rgb, n_depth, n_mask,
                     n_j3d, n_j2d, betas.size, pose0.size, trans0.size,
                     j3d0.shape[0], has_tpose])

        print(f"{seq:>3} {difficulty(seq):>9} {n_frames:>7} {n_rgb:>5} {n_depth:>6} {n_mask:>5} "
              f"{n_j3d:>5} {n_j2d:>5} {betas.size:>6} {pose0.size:>5} {trans0.size:>6} "
              f"{j3d0.shape[0]:>6} {str(has_tpose):>6}")

    out = config.OUTPUT_DIR / "dataset_integrity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fp:
        fp.write("sequence,difficulty,frames,rgb,depth,mask,joints3d,joints2d,"
                 "betas_dim,pose_dim,trans_dim,joint_rows,has_tpose\n")
        for r in rows:
            fp.write(",".join(str(x) for x in r) + "\n")

    # integrity checks
    issues = []
    for r in rows:
        seq, grp, nf, nrgb, nd, nm, nj3, nj2, *_ = r
        for label, val in [("rgb", nrgb), ("depth", nd), ("mask", nm),
                           ("joints3d", nj3), ("joints2d", nj2)]:
            if val != nf:
                issues.append(f"seq {seq}: {label}={val} != frames={nf}")
    print(f"\n{'ALL CONSISTENT' if not issues else 'ISSUES:'}")
    for i in issues:
        print("  ", i)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
