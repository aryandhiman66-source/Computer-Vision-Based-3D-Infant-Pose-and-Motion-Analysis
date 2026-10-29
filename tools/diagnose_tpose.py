"""Isolate shape vs pose discrepancy using the T-pose ground truth.

Reconstruct the sequence's infant in T-pose (pose=0) with the dataset betas and
compare the relative joint structure against syn_joints_3D_tpose.txt (which is
the same infant in T-pose). If shape/betas+model are consistent, errors -> ~0.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproduction"))

import config
import loader

model = loader.load_smil_model()
data = loader.load_frame("01", 0)

# T-pose reconstruction: dataset betas, zero pose, zero translation
model.betas[:] = data["betas"]
model.pose[:] = 0.0
model.trans[:] = 0.0
Jtr = loader.as_np(model.J_transformed)          # (24,3) in model frame (standing T-pose)
rel_model = Jtr - Jtr[0]

# GT tpose joints (camera frame), rows 0..23, no nose
tpose = loader.load_txt(config.DATASET_DIR / "01" / "tpose" / "syn_joints_3D_tpose.txt")
tpose = tpose[:24, :3]
rel_gt = tpose - tpose[0]

err = np.linalg.norm(rel_model - rel_gt, axis=1)
print("T-pose relative-joint error (model frame vs dataset, mm):")
for name, e in zip(config.JOINT_NAMES[:24], err):
    print(f"    {name:14s} {e*1000:8.3f} mm")
print(f"    MEAN = {err.mean()*1000:.3f} mm")

# Also try aligning shape by rigidly matching root-to-head scale? Just report.
print("\nIf MEAN is ~0, shape/model are consistent and the frame-00000 residual")
print("comes from pose/trans handling; if MEAN is ~4mm, the shape basis differs.")
