"""Demo: (1) lifting 2D joints to 3D with the depth image, (2) removing the table.

(1) For each 2D joint pixel (u, v) we read the depth image there and back-project
    with the pinhole model -> a 3D point. Compared with the true 3D joints this shows
    the systematic "surface vs. joint centre" offset and occlusion errors.
(2) The table is found as the dominant plane in the full point cloud with RANSAC
    (what Open3D's `segment_plane` does), compared with the dataset's bg_plane.txt,
    and every point within 1 cm of the plane or behind it is removed. The result is
    compared with the dataset's foreground mask.

Run:  python tools/demo_lift_and_table.py [sequence] [frame]
Writes reproduction/output/figures/lift_and_table_<seq>_<frame>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reproduction"))
import config  # noqa: E402
import geometry  # noqa: E402
import loader  # noqa: E402


def lift_joints(joints2d, depth_mm, win=2):
    """2D joints (u, v) + depth image -> 3D points (m). Median over a small window
    makes it robust to single missing/noisy depth pixels."""
    out = []
    h, w = depth_mm.shape
    for u, v in joints2d[:, :2]:
        iu, iv = int(round(u)), int(round(v))
        patch = depth_mm[max(iv - win, 0):iv + win + 1, max(iu - win, 0):iu + win + 1].astype(float)
        patch = patch[patch > 0]
        z = np.median(patch) / config.DEPTH_SCALE if patch.size else np.nan
        out.append([(u - config.CX) * z / config.FX, (v - config.CY) * z / config.FY, z])
    return np.array(out)


def ransac_plane(pts, iters=500, thresh=0.01, seed=0):
    """Find the plane n.x + d = 0 supported by the most points (within `thresh` m)."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        cnt = (np.abs(pts @ n + d) < thresh).sum()
        if cnt > best_n:
            best, best_n = (n, d), cnt
    n, d = best
    inl = np.abs(pts @ n + d) < thresh               # refine with least squares on inliers
    c0 = pts[inl].mean(0)
    n = np.linalg.svd(pts[inl] - c0, full_matrices=False)[2][-1]
    d = -n @ c0
    if d > 0:                                        # orient like bg_plane.txt (camera side negative d)
        n, d = -n, -d
    return n, d, inl.mean()


def main(seq="01", frame=0):
    data = loader.load_frame(seq, frame)
    depth, mask = data["depth"], data["mask"]
    gt3d = data["joints3d"][:, :3]
    j2d = data["joints2d"]

    # ---------- (1) lifting ----------
    lifted = lift_joints(j2d, depth)
    err = np.linalg.norm(lifted - gt3d, axis=1) * 1000
    dz = (lifted[:, 2] - gt3d[:, 2]) * 1000             # + = lifted point is FURTHER than the true joint? (z away from camera)
    print("=== (1) Lifting 2D joints to 3D with the depth image ===")
    print(f"mean 3D error {np.nanmean(err):.1f} mm | median {np.nanmedian(err):.1f} mm | max {np.nanmax(err):.1f} mm")
    print(f"depth difference (surface - joint centre): mean {np.nanmean(dz):+.1f} mm  -> surface is closer to the camera")
    order = np.argsort(-err)
    for i in order[:5]:
        print(f"   worst: {config.JOINT_NAMES[i]:14s} error {err[i]:6.1f} mm  (depth diff {dz[i]:+.1f} mm)")
    print(f"dataset joints_2Ddep depth column = joint-centre depth, e.g. {config.JOINT_NAMES[0]}: "
          f"{j2d[0, 2]:.0f} mm vs depth image at that pixel {depth[int(round(j2d[0,1])), int(round(j2d[0,0]))]} mm")

    # ---------- (2) table removal ----------
    full = geometry.depth_to_pointcloud(depth, None, stride=2)          # every pixel, incl. table
    valid = full[:, 2] > 0
    full = full[valid]
    n, d, frac = ransac_plane(full)
    gt_plane = np.loadtxt(config.DATASET_DIR / seq / "bg_plane.txt", usecols=(1, 2, 3, 4))
    ang = np.degrees(np.arccos(np.clip(abs(n @ gt_plane[:3]) / np.linalg.norm(gt_plane[:3]), -1, 1)))
    print("\n=== (2) Table removal by plane fitting (RANSAC) ===")
    print(f"RANSAC plane : n = {np.round(n, 4)}, d = {d:.4f}  ({100*frac:.0f}% of points lie on it)")
    print(f"bg_plane.txt : n = {np.round(gt_plane[:3], 4)}, d = {gt_plane[3]:.4f}  -> angle between normals {ang:.2f} deg, "
          f"distance difference {abs(d - gt_plane[3])*1000:.1f} mm")
    signed = full @ n + d                                  # < 0 : between camera and table (the baby)
    keep = signed < -0.01                                  # at least 1 cm in front of the table
    # compare with the dataset mask on the same pixel grid
    h, w = depth.shape
    vv, uu = np.mgrid[0:h:2, 0:w:2]
    uu, vv = uu.ravel()[valid], vv.ravel()[valid]
    pred = np.zeros((h, w), bool)
    pred[vv[keep], uu[keep]] = True
    gtm = mask[::2, ::2] > 0
    pm = pred[::2, ::2]
    iou = (pm & gtm).sum() / max((pm | gtm).sum(), 1)
    print(f"kept {keep.sum()} of {len(full)} points as 'baby' | IoU with dataset fg_mask = {iou:.3f}")

    # ---------- figure ----------
    fig = plt.figure(figsize=(15, 5), dpi=130)
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(data["rgb"])
    ax.scatter(j2d[:, 0], j2d[:, 1], s=18, c="#BA7517", edgecolors="white", linewidths=0.5)
    ax.set_title("2D joints on the colour image\n(OpenPose would give these)", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    body = geometry.depth_to_pointcloud(depth, mask, stride=3)
    ax.scatter(body[:, 0], body[:, 2], -body[:, 1], s=0.5, c="#B4B2A9", alpha=0.5)
    ax.scatter(gt3d[:, 0], gt3d[:, 2], -gt3d[:, 1], s=22, c="#534AB7", label="true 3D joint (centre)")
    ax.scatter(lifted[:, 0], lifted[:, 2], -lifted[:, 1], s=22, c="#BA7517", marker="x", label="lifted from depth (surface)")
    ax.set_title(f"Lifting 2D -> 3D: mean error {np.nanmean(err):.0f} mm", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.view_init(elev=15, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    ax = fig.add_subplot(1, 3, 3)
    show = np.zeros((h // 2 + h % 2, w // 2 + w % 2, 3))
    tab = np.zeros(len(full), bool); tab[np.abs(signed) < 0.01] = True
    show[(vv[tab] // 2), (uu[tab] // 2)] = [0.85, 0.83, 0.78]
    show[(vv[keep] // 2), (uu[keep] // 2)] = [0.11, 0.62, 0.46]
    ax.imshow(show)
    ax.set_title(f"Table removal: grey = table plane, green = kept (baby)\nIoU vs dataset mask {iou:.3f}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    out = config.OUTPUT_DIR / "figures" / f"lift_and_table_{seq}_{frame:05d}.png"
    fig.savefig(out, facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "01", int(sys.argv[2]) if len(sys.argv) > 2 else 0)
