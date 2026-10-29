"""Batch reproduction + evaluation over a full MINI-RGBD sequence.

For every frame: reconstruct the SMIL mesh from GT pose/trans (betas constant),
compute 3D joint error (MPJPE) and 2D projection error, then:
  - write sequence_XX_metrics.csv
  - render an MP4 of RGB + projected joints vs GT joints
  - print an aggregate summary

Usage:
    python reproduction/evaluate_sequence.py [sequence] [--max-frames N] [--video]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import geometry
import loader


def count_frames(sequence: str) -> int:
    return len(list((config.DATASET_DIR / sequence / "smil_params").glob("*_pose.txt")))


def make_skeleton_lines() -> list[tuple[int, int]]:
    """Bones to draw for the 25-joint skeleton (indices into the 24 SMIL + nose)."""
    return [
        (0, 3), (3, 6), (6, 9), (9, 12),            # spine chain
        (0, 1), (1, 4), (4, 7), (7, 10),            # left leg
        (0, 2), (2, 5), (5, 8), (8, 11),            # right leg
        (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),   # left arm
        (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),   # right arm
        (12, 15), (15, 24),                          # head + nose
    ]


def main(sequence: str = "01", max_frames: int | None = None, make_video: bool = True):
    n_frames = count_frames(sequence)
    if max_frames:
        n_frames = min(n_frames, max_frames)

    print(f"=== Batch reproduction of MINI-RGBD sequence {sequence} ({n_frames} frames) ===")

    model = loader.load_smil_model()
    data0 = loader.load_frame(sequence, 0)
    betas = data0["betas"]
    model.betas[:] = betas
    nose_idx = 331
    bones = make_skeleton_lines()

    rows = []
    t0 = time.time()

    video_writer = None
    if make_video:
        import cv2
        out = config.OUTPUT_DIR / sequence
        out.mkdir(parents=True, exist_ok=True)
        vid_path = str(out / f"sequence_{sequence}_overlay.mp4")
        video_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                       (config.IMAGE_W, config.IMAGE_H))

    for f in range(n_frames):
        data = loader.load_frame(sequence, f)
        model.pose[:] = data["pose"]
        model.trans[:] = data["trans"]

        verts = loader.as_np(model.r)
        Jtr = loader.as_np(model.J_transformed)                      # (24,3)

        gt3d = data["joints3d"][:24, :3]
        mpjpe = np.linalg.norm(Jtr - gt3d, axis=1).mean() * 1000.0

        # 2D: project model joints + nose, compare to dataset joints_2Ddep (px)
        proj = geometry.project_points(np.vstack([Jtr, verts[nose_idx]]))
        err2d = np.linalg.norm(proj - data["joints2d"][:, :2], axis=1)
        rows.append((f, mpjpe, err2d.mean(), np.median(err2d), err2d.max()))

        if video_writer is not None:
            frame = data["rgb"].copy()
            # GT joints in green, ours in red
            for (a, b) in bones:
                cv2.line(frame, tuple(data["joints2d"][a, :2].astype(int)),
                         tuple(data["joints2d"][b, :2].astype(int)), (0, 200, 0), 2)
                cv2.line(frame, tuple(proj[a].astype(int)), tuple(proj[b].astype(int)),
                         (0, 0, 255), 2)
            for p in data["joints2d"][:, :2]:
                cv2.circle(frame, tuple(p.astype(int)), 3, (0, 200, 0), -1)
            for p in proj:
                cv2.circle(frame, tuple(p.astype(int)), 3, (0, 0, 255), -1)
            cv2.putText(frame, f"MPJPE {mpjpe:.1f} mm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if (f + 1) % 200 == 0:
            rate = (f + 1) / (time.time() - t0)
            print(f"    {f+1}/{n_frames} frames  ({rate:.1f} frames/s)")

    if video_writer is not None:
        video_writer.release()

    arr = np.array(rows)
    out = config.OUTPUT_DIR / sequence
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"sequence_{sequence}_metrics.csv"
    with open(csv_path, "w") as fp:
        fp.write("frame,mpjpe_mm,err2d_mean_px,err2d_median_px,err2d_max_px\n")
        for r in rows:
            fp.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")

    print("\n=== Summary ===")
    print(f"    frames processed : {n_frames}")
    print(f"    MPJPE  : mean {arr[:,1].mean():.3f}  median {np.median(arr[:,1]):.3f}  "
          f"max {arr[:,1].max():.3f} mm")
    print(f"    2D err : mean {arr[:,2].mean():.3f}  median {arr[:,3].mean():.3f}  "
          f"max {arr[:,4].max():.3f} px")
    print(f"    elapsed: {time.time()-t0:.1f} s")
    print(f"    saved  : {csv_path}" + (f" + video" if make_video else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence", nargs="?", default="01")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    main(a.sequence, a.max_frames, not a.no_video)
