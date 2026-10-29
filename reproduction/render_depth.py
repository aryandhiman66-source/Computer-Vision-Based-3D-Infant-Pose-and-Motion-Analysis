"""Reproduce the depth + foreground-mask channels by z-buffer rendering the
reconstructed SMIL mesh, and compare against the dataset ground truth.

Usage:
    python reproduction/render_depth.py [sequence] [frame]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import loader


def rasterize_depth(verts: np.ndarray, faces: np.ndarray,
                    h: int = config.IMAGE_H, w: int = config.IMAGE_W) -> np.ndarray:
    """Z-buffer triangle rasterization -> depth in mm (inf where empty).

    verts: (N,3) camera-space meters; faces: (F,3) vertex indices.
    """
    u = config.FX * verts[:, 0] / verts[:, 2] + config.CX
    v = config.FY * verts[:, 1] / verts[:, 2] + config.CY
    z = verts[:, 2] * 1000.0                      # -> mm

    depth = np.full((h, w), np.inf, dtype=np.float64)
    eps = 1e-9

    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        z0, z1, z2 = z[i0], z[i1], z[i2]
        if z0 <= 0 or z1 <= 0 or z2 <= 0:
            continue

        u0, u1, u2 = u[i0], u[i1], u[i2]
        v0, v1, v2 = v[i0], v[i1], v[i2]

        xmin = max(0, int(np.floor(min(u0, u1, u2))))
        xmax = min(w - 1, int(np.ceil(max(u0, u1, u2))))
        ymin = max(0, int(np.floor(min(v0, v1, v2))))
        ymax = min(h - 1, int(np.ceil(max(v0, v1, v2))))
        if xmax < xmin or ymax < ymin:
            continue

        xs = np.arange(xmin, xmax + 1)
        ys = np.arange(ymin, ymax + 1)
        X, Y = np.meshgrid(xs, ys)                 # X:u, Y:v

        denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        if abs(denom) < 1e-12:
            continue
        w0 = ((v1 - v2) * (X - u2) + (u2 - u1) * (Y - v2)) / denom
        w1 = ((v2 - v0) * (X - u2) + (u0 - u2) * (Y - v2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
        if not inside.any():
            continue

        z_here = w0 * z0 + w1 * z1 + w2 * z2
        yy, xx = np.nonzero(inside)          # local indices into (ys, xs)
        px = xs[xx]                          # global pixel column (u)
        py = ys[yy]                          # global pixel row (v)
        cand = z_here[inside]
        cur = depth[py, px]
        better = cand < cur
        if better.any():
            depth[py[better], px[better]] = cand[better]

    return depth


def main(sequence: str = "01", frame: int = 0):
    print(f"=== Depth + mask reproduction for {sequence}/{frame:05d} ===")
    data = loader.load_frame(sequence, frame)
    model = loader.load_smil_model()
    model.betas[:] = data["betas"]
    model.pose[:] = data["pose"]
    model.trans[:] = data["trans"]
    verts = loader.as_np(model.r)
    faces = np.asarray(model.f, dtype=np.int64)

    print("    rasterizing 13776 triangles ...")
    import time
    t0 = time.time()
    rendered = rasterize_depth(verts, faces)
    print(f"    rasterization took {time.time()-t0:.2f} s")

    rendered_mask = np.isfinite(rendered)
    gt_depth = data["depth"].astype(np.float64)
    gt_mask = (data["mask"] > 0)

    # --- mask comparison ---
    inter = (rendered_mask & gt_mask).sum()
    union = (rendered_mask | gt_mask).sum()
    iou = inter / union if union else 0.0
    dice = 2 * inter / (rendered_mask.sum() + gt_mask.sum()) if (rendered_mask.sum() + gt_mask.sum()) else 0.0
    tp = inter
    fn = gt_mask.sum() - inter
    fp = rendered_mask.sum() - inter
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0

    # --- depth comparison (on pixels valid in BOTH) ---
    both = rendered_mask & gt_mask
    diff = rendered[both] - gt_depth[both]
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())

    print("\n=== Results ===")
    print(f"    mask IoU = {iou:.4f}   Dice = {dice:.4f}")
    print(f"    precision = {prec:.4f}   recall = {rec:.4f}")
    print(f"    depth (valid overlap, mm): MAE = {mae:.2f}   RMSE = {rmse:.2f}")
    print(f"    depth diff: median {np.median(diff):+.2f}  mean {diff.mean():+.2f} mm")

    _save(sequence, frame, data, rendered, rendered_mask, gt_depth, gt_mask)
    print("    saved outputs.")


def _save(sequence, frame, data, rendered, rendered_mask, gt_depth, gt_mask):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = config.OUTPUT_DIR / sequence / f"{frame:05d}"
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / "rendered_depth.png"), np.where(rendered_mask, rendered, 0).astype(np.uint16))
    cv2.imwrite(str(out / "rendered_mask.png"), (rendered_mask.astype(np.uint8) * 255))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(gt_depth, cmap="viridis"); axes[0, 0].set_title("GT depth")
    axes[0, 1].imshow(np.where(rendered_mask, rendered, np.nan), cmap="viridis")
    axes[0, 1].set_title("Rendered depth")
    diff = np.full_like(gt_depth, np.nan)
    both = rendered_mask & gt_mask
    diff[both] = rendered[both] - gt_depth[both]
    im = axes[0, 2].imshow(diff, cmap="RdBu_r", vmin=-15, vmax=15)
    axes[0, 2].set_title("Depth diff (mm)")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)
    axes[1, 0].imshow(gt_mask, cmap="gray"); axes[1, 0].set_title("GT mask")
    axes[1, 1].imshow(rendered_mask, cmap="gray"); axes[1, 1].set_title("Rendered mask")
    axes[1, 2].imshow(data["rgb"]); axes[1, 2].set_title("RGB")
    fig.suptitle(f"MINI-RGBD {sequence}/{frame:05d} — depth + mask reproduction")
    fig.tight_layout()
    fig.savefig(out / "depth_mask_comparison.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "01"
    fr = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(seq, fr)
