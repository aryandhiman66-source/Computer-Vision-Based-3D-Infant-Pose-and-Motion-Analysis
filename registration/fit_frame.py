"""Step 9 (controlled recovery test): perturb GT global pose, then optimise it
back using the scan (point-cloud) loss, and verify recovery.

This validates the loss + optimiser BEFORE attempting a from-scratch fit.

Run:  python registration/fit_frame.py [sequence] [frame]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproduction"))

import config
import loader
from reproduction import geometry
from registration.losses import scan_loss


def mpjpe(model, gt3d: np.ndarray) -> float:
    """Reconstruct joints from current model state and return MPJPE (mm)."""
    Jtr = loader.as_np(model.J_transformed)
    return float(np.linalg.norm(Jtr - gt3d, axis=1).mean() * 1000.0)


def main(seq: str = "01", frame: int = 0):
    print(f"=== Controlled recovery test — {seq}/{frame:05d} ===")
    model = loader.load_smil_model()
    data = loader.load_frame(seq, frame)
    model.betas[:] = data["betas"]

    gt_pose = data["pose"].copy()
    gt_trans = data["trans"].copy()
    gt_rot = gt_pose[0:3].copy()
    gt3d = data["joints3d"][:24, :3]

    # observed points (target)
    points = geometry.depth_to_pointcloud(data["depth"], data["mask"], stride=4)
    print(f"    target point cloud: {points.shape[0]} points")

    def set_state(rot, trans):
        pose = gt_pose.copy()
        pose[0:3] = rot
        model.pose[:] = pose
        model.trans[:] = trans

    def loss(x):
        set_state(x[0:3], x[3:6])
        verts = loader.as_np(model.r)
        return scan_loss(verts, points)

    # 1) GT state
    set_state(gt_rot, gt_trans)
    gt_loss = loss(np.concatenate([gt_rot, gt_trans]))
    gt_err = mpjpe(model, gt3d)

    # 2) Perturbed state (translate +4cm, rotate global axis-angle +0.3)
    rng = np.random.default_rng(0)
    pert_trans = gt_trans + np.array([0.04, 0.04, 0.04])
    pert_rot = gt_rot + 0.3
    x0 = np.concatenate([pert_rot, pert_trans])
    set_state(pert_rot, pert_trans)
    pert_loss = loss(x0)
    pert_err = mpjpe(model, gt3d)

    print(f"    GT        : scan loss {gt_loss:.6f}  MPJPE {gt_err:.2f} mm")
    print(f"    Perturbed : scan loss {pert_loss:.6f}  MPJPE {pert_err:.2f} mm")

    # 3) Optimise
    from scipy.optimize import minimize
    res = minimize(loss, x0, method="L-BFGS-B", options={"maxiter": 150, "ftol": 1e-10})
    set_state(res.x[0:3], res.x[3:6])
    rec_err = mpjpe(model, gt3d)
    rec_loss = res.fun

    print(f"    Recovered : scan loss {rec_loss:.6f}  MPJPE {rec_err:.2f} mm")
    print(f"    optimiser : success={res.success}, message={res.message}")
    print(f"    trans error : GT {gt_trans} -> recovered {np.round(res.x[3:6], 4)}")
    print(f"    rot   error : |recovered - GT| = {np.linalg.norm(res.x[0:3] - gt_rot):.4f} rad")

    print("\n    => recovery quality (should approach the GT MPJPE of "
          f"{gt_err:.2f} mm)")


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "01"
    f = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(s, f)
