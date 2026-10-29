"""Step 3: Reproduce ALL 12 sequences (3D + 2D joint metrics).

Loops every sequence and every frame, reconstructing the SMIL mesh from GT
parameters and measuring MPJPE (3D) and 2D reprojection error. Writes:
  - per-sequence summary -> output/all_sequences_metrics.csv
  - per-frame detail     -> output/{seq}/sequence_{seq}_metrics.csv

Run:  python reproduction/evaluate_all.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import geometry
import loader


def difficulty(seq: str) -> str:
    n = int(seq)
    return "easy" if n <= 4 else ("medium" if n <= 9 else "difficult")


def process_sequence(model, seq: str):
    """Reconstruct every frame; return (rows, aggregate)."""
    n = len(list((config.DATASET_DIR / seq / "smil_params").glob("*_pose.txt")))
    data0 = loader.load_frame(seq, 0)
    model.betas[:] = data0["betas"]
    nose_idx = 331

    mpjpe_all, e2d_all = [], []
    t0 = time.time()
    for f in range(n):
        data = loader.load_frame(seq, f)
        model.pose[:] = data["pose"]
        model.trans[:] = data["trans"]
        verts = loader.as_np(model.r)
        Jtr = loader.as_np(model.J_transformed)

        gt3d = data["joints3d"][:24, :3]
        mpjpe = np.linalg.norm(Jtr - gt3d, axis=1).mean() * 1000.0

        proj = geometry.project_points(np.vstack([Jtr, verts[nose_idx]]))
        e2d = np.linalg.norm(proj - data["joints2d"][:, :2], axis=1).mean()

        mpjpe_all.append(mpjpe)
        e2d_all.append(e2d)

    mpjpe_all = np.array(mpjpe_all)
    e2d_all = np.array(e2d_all)
    print(f"  {seq} ({difficulty(seq)}): MPJPE mean {mpjpe_all.mean():.3f} mm, "
          f"2D mean {e2d_all.mean():.3f} px  [{time.time()-t0:.1f}s]")
    return (mpjpe_all, e2d_all)


def main():
    model = loader.load_smil_model()
    seqs = sorted(p.name for p in config.DATASET_DIR.glob("[0-9][0-9]"))
    print(f"=== Reproducing all {len(seqs)} sequences ===")

    summary = []
    for seq in seqs:
        mpjpe, e2d = process_sequence(model, seq)
        summary.append([seq, difficulty(seq), len(mpjpe),
                        mpjpe.mean(), np.median(mpjpe), mpjpe.max(),
                        e2d.mean(), np.median(e2d), e2d.max()])

        # per-sequence per-frame detail CSV
        out = config.OUTPUT_DIR / seq
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"sequence_{seq}_metrics.csv", "w") as fp:
            fp.write("frame,mpjpe_mm,err2d_mean_px\n")
            for f, (m, e) in enumerate(zip(mpjpe, e2d)):
                fp.write(f"{f},{m:.4f},{e:.4f}\n")

    # combined summary
    out = config.OUTPUT_DIR / "all_sequences_metrics.csv"
    with open(out, "w") as fp:
        fp.write("sequence,difficulty,frames,mpjpe_mean_mm,mpjpe_median_mm,mpjpe_max_mm,"
                 "e2d_mean_px,e2d_median_px,e2d_max_px\n")
        for r in summary:
            fp.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    print("\n=== Per-sequence summary ===")
    print(f"{'seq':>3} {'group':>9} {'frames':>6} {'MPJPE mean':>11} {'MPJPE max':>10} "
          f"{'2D mean':>8} {'2D max':>7}")
    for r in summary:
        print(f"{r[0]:>3} {r[1]:>9} {r[2]:>6} {r[3]:>11.3f} {r[5]:>10.3f} {r[6]:>8.3f} {r[8]:>7.3f}")

    # difficulty aggregate
    print("\n=== By difficulty (MPJPE mm) ===")
    for grp in ["easy", "medium", "difficult"]:
        vals = [r[3] for r in summary if r[1] == grp]
        if vals:
            print(f"  {grp:>9}: mean {np.mean(vals):.3f}  (sequences {len(vals)})")

    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
