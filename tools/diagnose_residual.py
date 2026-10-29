"""Proper diagnostics for the ~4mm residual:

A) Bone lengths (rotation-invariant): zero-pose model vs T-pose GT  -> verifies SHAPE
B) Nearest-vertex test: are the dataset's 24 joints exact mesh vertices?
C) Procrustes: is the frame-00000 residual a rigid misalignment?
D) Per-joint nearest-vertex distances (frame 00000)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproduction"))

import config
import loader

model = loader.load_smil_model()
data = loader.load_frame("01", 0)

# joint hierarchy parents (from kintree)
parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

print("=== A) Bone lengths: zero-pose model vs T-pose GT (mm) ===")
model.betas[:] = data["betas"]
model.pose[:] = 0.0
model.trans[:] = 0.0
J_model = loader.as_np(model.J_transformed)
tpose = loader.load_txt(config.DATASET_DIR / "01" / "tpose" / "syn_joints_3D_tpose.txt")[:24, :3]
d_model, d_gt = [], []
for i in range(1, 24):
    d_model.append(np.linalg.norm(J_model[i] - J_model[parents[i]]))
    d_gt.append(np.linalg.norm(tpose[i] - tpose[parents[i]]))
d_model = np.array(d_model); d_gt = np.array(d_gt)
for name, a, b in zip(config.JOINT_NAMES[1:24], d_model, d_gt):
    print(f"    {name:14s} model {a*1000:7.2f}   GT {b*1000:7.2f}   diff {(a-b)*1000:7.2f}")
print(f"    mean |diff| = {np.abs(d_model-d_gt).mean()*1000:.3f} mm")

print("\n=== B/C) Frame 00000: nearest-vertex + Procrustes ===")
model.pose[:] = data["pose"]
model.trans[:] = data["trans"]
verts = loader.as_np(model.r)
Jtr = loader.as_np(model.J_transformed)
gt = data["joints3d"][:24, :3]

from scipy.spatial import cKDTree
tree = cKDTree(verts)
d_v, _ = tree.query(gt, k=1)
print("    nearest-vertex distance to each GT joint (mm):")
for name, e in zip(config.JOINT_NAMES[:24], d_v):
    print(f"        {name:14s} {e*1000:7.3f} mm")

# Procrustes (rigid alignment) between Jtr and gt
A = Jtr - Jtr.mean(0)
B = gt - gt.mean(0)
U, S, Vt = np.linalg.svd(A.T @ B)
R = Vt.T @ U.T
if np.linalg.det(R) < 0:
    Vt[-1] *= -1
    R = Vt.T @ U.T
resid = A @ R - B
print(f"    Procrustes residual: mean {np.linalg.norm(resid,axis=1).mean()*1000:.3f} mm "
      f"(rotation det={np.linalg.det(R):.4f})")
print("    -> if residual ~0, error is a rigid offset; if ~4mm, it is per-joint/deformation.")
