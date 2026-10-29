"""Reproduce MINI-RGBD Sequence 01, Frame 00000 from its ground-truth parameters.

This is the "Immediate Next Step" of the Paper 2 reproduction plan:

    betas + pose + trans  ->  SMIL  ->  mesh  ->  3D joints
                                              ->  project -> 2D joints
    depth + mask -> point cloud

and then compare our reconstruction against the supplied ground truth.

Usage (from smil-paper1 root):
    $env:PYTHONPATH = "<repo>/smil"
    python -W "ignore::DeprecationWarning" reproduction/run_frame.py [sequence] [frame]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import geometry
import loader


def reconstruct(model, betas: np.ndarray, pose: np.ndarray, trans: np.ndarray):
    """Set SMIL parameters and evaluate posed vertices + 3D joints."""
    model.betas[:] = betas
    model.pose[:] = pose
    model.trans[:] = trans
    verts = loader.as_np(model.r)                 # (6890, 3) posed + translated
    joints = loader.as_np(model.J_transformed)    # (24, 3) in kintree order
    faces = np.asarray(model.f, dtype=np.int64)
    return verts, joints, faces


def compare_3d_joints(model_joints: np.ndarray, gt_joints3d: np.ndarray):
    """Compare 24 reconstructed SMIL joints vs dataset rows 0..23 (meters)."""
    gt = gt_joints3d[: config.N_SMIL_JOINTS, :3]
    err = np.linalg.norm(model_joints - gt, axis=1)          # per-joint (m)
    mpjpe = err.mean() * 1000.0                               # mm
    return err, mpjpe


def find_nose_vertex(verts: np.ndarray, nose_gt: np.ndarray):
    """Identify which mesh vertex corresponds to the dataset `noseVertex` row."""
    d = np.linalg.norm(verts - nose_gt, axis=1)
    idx = int(np.argmin(d))
    return idx, d[idx]


def compare_2d(projected: np.ndarray, gt_joints2d: np.ndarray):
    """Pixel error between projected joints and dataset (u, v)."""
    gt = gt_joints2d[:, :2]
    err = np.linalg.norm(projected - gt, axis=1)
    return err


def main(sequence: str = "01", frame: int = 0):
    print(f"=== Reproducing MINI-RGBD {sequence}/{frame:05d} ===")
    data = loader.load_frame(sequence, frame)

    print("\n[1] SMIL reconstruction")
    model = loader.load_smil_model()
    verts, joints, faces = reconstruct(model, data["betas"], data["pose"], data["trans"])
    print(f"    vertices: {verts.shape}, faces: {faces.shape}, joints: {joints.shape}")
    print(f"    betas range: [{data['betas'].min():.3f}, {data['betas'].max():.3f}]  (n={data['betas'].size})")
    print(f"    trans: {data['trans']}")

    print("\n[2] 3D joint validation (model vs dataset, mm)")
    err3d, mpjpe = compare_3d_joints(joints, data["joints3d"])
    for name, e in zip(config.JOINT_NAMES, err3d):
        print(f"    {name:14s} {e*1000:8.3f} mm")
    print(f"    MPJPE = {mpjpe:.3f} mm")

    print("\n[3] noseVertex identification")
    nose_gt = data["joints3d"][config.N_SMIL_JOINTS, :3]
    nose_idx, nose_dist = find_nose_vertex(verts, nose_gt)
    print(f"    dataset nose pos: {nose_gt}")
    print(f"    closest vertex {nose_idx}: distance {nose_dist*1000:.4f} mm")

    print("\n[4] 2D projection validation (model vs dataset, px)")
    all3d = np.vstack([joints, verts[nose_idx]])            # (25, 3)
    projected = geometry.project_points(all3d)
    err2d = compare_2d(projected, data["joints2d"])
    print(f"    mean={err2d.mean():.3f}  median={np.median(err2d):.3f}  max={err2d.max():.3f} px")

    print("\n[5] Depth + mask -> point cloud vs mesh (approximate)")
    pts = geometry.depth_to_pointcloud(data["depth"], data["mask"], stride=4)
    from scipy.spatial import cKDTree
    tree = cKDTree(verts)
    d, _ = tree.query(pts, workers=-1)
    print(f"    foreground pts: {pts.shape[0]}  (stride=4)")
    print(f"    nearest-vertex dist: mean={d.mean()*1000:.2f} mm  median={np.median(d)*1000:.2f} mm  max={d.max()*1000:.2f} mm")

    _save_outputs(sequence, frame, data, verts, faces, joints, pts, projected, nose_idx)
    print("\nDone.")


def _save_outputs(sequence, frame, data, verts, faces, joints, pts, projected, nose_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out = config.OUTPUT_DIR / sequence / f"{frame:05d}"
    out.mkdir(parents=True, exist_ok=True)

    # --- 2D panel ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    ax.imshow(data["rgb"])
    ax.plot(projected[:, 0], projected[:, 1], "o", ms=3, color="red")
    ax.set_title("RGB + projected joints")

    ax = axes[1]
    ax.imshow(data["depth"], cmap="viridis")
    ax.plot(projected[:, 0], projected[:, 1], "o", ms=3, color="red")
    ax.set_title("Depth (mm) + projected joints")

    ax = axes[2]
    ax.imshow(data["mask"], cmap="gray")
    ax.set_title("Foreground mask")
    fig.suptitle(f"MINI-RGBD {sequence}/{frame:05d} reproduction")
    fig.tight_layout()
    fig.savefig(out / "panels_2d.png", dpi=110)
    plt.close(fig)

    # --- 3D panel ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, pts.shape[0] // 4000)
    ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2], s=1, c="lightsteelblue", label="depth point cloud")
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0.5, c="orange", label="SMIL mesh vertices")
    gt = data["joints3d"]
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=25, c="red", label="GT 3D joints")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.legend()
    ax.set_title("Depth point cloud + reconstructed SMIL mesh + GT joints")
    fig.tight_layout()
    fig.savefig(out / "overlay_3d.png", dpi=110)
    plt.close(fig)

    # --- mesh OBJ (for external inspection) ---
    obj_path = out / "smil_mesh.obj"
    with open(obj_path, "w") as fp:
        for v in verts:
            fp.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces + 1:  # OBJ faces are 1-based
            fp.write(f"f {f[0]} {f[1]} {f[2]}\n")
    print(f"    saved: {out}")


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "01"
    fr = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(seq, fr)
