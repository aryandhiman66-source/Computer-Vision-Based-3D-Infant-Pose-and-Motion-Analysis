"""Diagnostic: which joint definition does MINI-RGBD use?

Compares dataset joints_3D (rows 0..23) against:
  A) LBS skeleton joints  (model.J_transformed)
  B) regressed joints     (model.J_regressor @ posed_vertices)
Also verifies the exact nose vertex.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproduction"))

import config
import loader

model = loader.load_smil_model()
data = loader.load_frame("01", 0)

model.betas[:] = data["betas"]
model.pose[:] = data["pose"]
model.trans[:] = data["trans"]
verts = loader.as_np(model.r)
Jtr = loader.as_np(model.J_transformed)

gt = data["joints3d"][:24, :3]

reg = model.J_regressor.dot(verts)
reg = np.asarray(reg, dtype=np.float64)

err_tr = np.linalg.norm(Jtr - gt, axis=1)
err_reg = np.linalg.norm(reg - gt, axis=1)

print("per-joint error (mm):")
print(f"{'joint':14s} {'LBS Jtr':>10s} {'regressed':>10s}")
for name, a, b in zip(config.JOINT_NAMES, err_tr, err_reg):
    print(f"{name:14s} {a*1000:10.3f} {b*1000:10.3f}")
print(f"{'MEAN':14s} {err_tr.mean()*1000:10.3f} {err_reg.mean()*1000:10.3f}")

# nose vertex
nose_gt = data["joints3d"][24, :3]
d = np.linalg.norm(verts - nose_gt, axis=1)
idx = int(np.argmin(d))
print(f"\nnose GT: {nose_gt}")
print(f"nearest vertex = {idx}, dist = {d[idx]*1000:.4f} mm")
print(f"vertex 331    dist = {np.linalg.norm(verts[331]-nose_gt)*1000:.4f} mm")

# also check: is regressor applied to SHAPED (rest) then LBS'd identical to Jtr?
print("\nregressed(posed) vs LBS-skeleton diff, mean:", np.linalg.norm(reg - Jtr, axis=1).mean()*1000, "mm")
