"""Live code demo: reproduce MINI-RGBD 01/00000 step by step.

Run in front of your supervisor:
    $env:PYTHONPATH = "<repo>/smil"
    python -W "ignore::DeprecationWarning" reproduction/demo.py

Each numbered step prints what it computes, so you can narrate the pipeline.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import geometry
import loader
from render_depth import rasterize_depth

SEQ, FRAME = "01", 0

print("STEP 1 — Load the SMIL body model")
model = loader.load_smil_model()
print(f"    vertices {model.v_template.shape[0]}, faces {model.f.shape[0]}, "
      f"shape coeffs {model.betas.shape[0]}, pose params {model.pose.shape[0]}")

print("\nSTEP 2 — Load one frame's ground-truth parameters")
data = loader.load_frame(SEQ, FRAME)
print(f"    betas (beta) : {data['betas'].shape[0]} values  (body shape)")
print(f"    pose  (theta): {data['pose'].shape[0]} values  (24 joints x 3 axis-angle)")
print(f"    trans (T)    : {data['trans']}  (meters)")

print("\nSTEP 3 — Forward pass: beta + theta + T  ->  mesh + 3D joints")
model.betas[:] = data["betas"]   # shape
model.pose[:] = data["pose"]     # pose
model.trans[:] = data["trans"]   # global translation
verts = loader.as_np(model.r)                  # 6890 posed vertices (meters)
joints = loader.as_np(model.J_transformed)     # 24 3D joints (skeleton)
print(f"    mesh: {verts.shape}, joints: {joints.shape}")

print("\nSTEP 4 — Validate 3D joints against the dataset ground truth")
gt3d = data["joints3d"][:24, :3]
err = np.linalg.norm(joints - gt3d, axis=1)
print(f"    MPJPE = {err.mean()*1000:.2f} mm  (max {err.max()*1000:.2f} mm)")

print("\nSTEP 5 — Project joints to the image (pinhole camera)")
# u = fx*X/Z + cx ;  v = fy*Y/Z + cy
proj = geometry.project_points(np.vstack([joints, verts[331]]))   # + nose vertex 331
err2d = np.linalg.norm(proj - data["joints2d"][:, :2], axis=1)
print(f"    2D reprojection error = {err2d.mean():.2f} px")

print("\nSTEP 6 — Convert depth image + mask into a 3D point cloud")
# X = (u - cx)*Z/fx ;  Y = (v - cy)*Z/fy ;  Z = depth_mm / 1000
pts = geometry.depth_to_pointcloud(data["depth"], data["mask"], stride=4)
print(f"    foreground points = {pts.shape[0]} (subsampled)")

print("\nSTEP 7 — Render the mesh back to depth + mask (z-buffer rasterization)")
rendered = rasterize_depth(verts, np.asarray(model.f, dtype=np.int64))
rmask = np.isfinite(rendered)
gmask = data["mask"] > 0
gdepth = data["depth"].astype(np.float64)
iou = (rmask & gmask).sum() / (rmask | gmask).sum()
both = rmask & gmask
diff = rendered[both] - gdepth[both]
print(f"    mask IoU = {iou:.3f}")
print(f"    depth MAE = {np.abs(diff).mean():.2f} mm, RMSE = {np.sqrt((diff**2).mean()):.2f} mm")

print("\nSTEP 8 — Save the figure")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
out = config.OUTPUT_DIR / SEQ / f"{FRAME:05d}"
out.mkdir(parents=True, exist_ok=True)
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pts[::10, 0], pts[::10, 1], pts[::10, 2], s=1, c="lightsteelblue", label="depth point cloud")
ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0.4, c="orange", label="SMIL mesh")
ax.scatter(gt3d[:, 0], gt3d[:, 1], gt3d[:, 2], s=25, c="red", label="GT joints")
ax.legend(); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
fig.tight_layout()
fig.savefig(out / "demo_overlay_3d.png", dpi=110)
print(f"    saved {out / 'demo_overlay_3d.png'}")
print("\nDONE.")
