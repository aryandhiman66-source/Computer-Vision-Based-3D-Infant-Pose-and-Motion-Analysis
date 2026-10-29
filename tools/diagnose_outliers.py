"""Investigate why sequences 05 and 11 reproduce worse than the other 10.

Tests per sequence (frame 0):
  A) per-joint 3D error breakdown
  B) Procrustes rigid-fit residual (is it a rigid offset?)
  C) bone-length agreement (reconstructed skeleton vs dataset joints)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproduction"))

import config
import loader

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def analyze(seq, model):
    data = loader.load_frame(seq, 0)
    model.betas[:] = data["betas"]
    model.pose[:] = data["pose"]
    model.trans[:] = data["trans"]
    Jtr = loader.as_np(model.J_transformed)
    gt = data["joints3d"][:24, :3]

    err = np.linalg.norm(Jtr - gt, axis=1) * 1000.0

    # B) Procrustes
    A = Jtr - Jtr.mean(0)
    B = gt - gt.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    resid = np.linalg.norm(A @ R - B, axis=1).mean() * 1000.0

    # C) bone lengths
    bl_model = np.array([np.linalg.norm(Jtr[i] - Jtr[PARENTS[i]]) for i in range(1, 24)])
    bl_gt = np.array([np.linalg.norm(gt[i] - gt[PARENTS[i]]) for i in range(1, 24)])
    bl_diff = np.abs(bl_model - bl_gt).mean() * 1000.0

    return err, resid, bl_diff


def main():
    model = loader.load_smil_model()
    print(f"{'joint':14s} {'01':>8s} {'05':>8s} {'11':>8s}")
    errs = {}
    for seq in ["01", "05", "11"]:
        err, resid, bl = analyze(seq, model)
        errs[seq] = (err, resid, bl)

    for name in config.JOINT_NAMES[:24]:
        print(f"{name:14s} {errs['01'][0][config.JOINT_NAMES.index(name)]:8.2f} "
              f"{errs['05'][0][config.JOINT_NAMES.index(name)]:8.2f} "
              f"{errs['11'][0][config.JOINT_NAMES.index(name)]:8.2f}")

    print("\nProcrustes residual (rigid-fit, mm):")
    for s in ["01", "05", "11"]:
        print(f"   {s}: {errs[s][1]:.2f}")
    print("Bone-length mean |diff| (mm):")
    for s in ["01", "05", "11"]:
        print(f"   {s}: {errs[s][2]:.2f}")

    print("\nInterpretation:")
    print(" - If Procrustes ~0 -> rigid offset (translation/rotation).")
    print(" - If bone-length diff large -> the dataset skeleton itself differs (pose/shape).")
    print(" - If only some joints off -> possible joint-order issue.")


if __name__ == "__main__":
    main()
