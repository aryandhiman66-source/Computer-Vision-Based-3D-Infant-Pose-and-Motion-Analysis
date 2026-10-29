"""POC: lift 2D ViTPose keypoints to 3D via SMIL (weak-perspective fitting).

Fits SMIL pose theta (+ scale + translation) so its 3D joints reproject onto
the raw 2D keypoints. Outputs reprojection error so we can judge if the 2D->3D
lifting is viable before scaling to the whole dataset.

Run:  python classification/smil3d.py <skeleton_file.txt>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "smil"))

from smil_webuser.serialization import load_model  # noqa: E402

# ViTPose 17 keypoint index -> SMIL joint index (SMIL kintree == SMPL order)
# 0 nose,5 L sh,6 R sh,7 L elb,8 R elb,9 L wri,10 R wri,11 L hip,12 R hip,13 L kne,14 R kne,15 L ank,16 R ank
V2S = {0: 15, 5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21,
       11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
SRC = sorted(V2S.keys())
DST = [V2S[i] for i in SRC]

model = load_model(str(ROOT / "smil" / "smil_web.pkl"))
model.betas[:] = 0.0


def joints(theta: np.ndarray) -> np.ndarray:
    model.pose[:] = theta
    model.trans[:] = 0.0
    return np.asarray(model.J_transformed, dtype=np.float64)   # (24,3)


def fit_frame(kps_2d: np.ndarray, maxiter: int = 150):
    """kps_2d: (17,2) raw pixel keypoints. Returns reprojection RMSE (normalized)."""
    pts = kps_2d.astype(np.float64).copy()
    # normalise: centre on mapped joints, scale by a robust reference
    centre = pts[SRC].mean(axis=0)
    pts -= centre
    ref = np.percentile(np.abs(pts[SRC]), 99)
    pts /= ref
    tgt = pts[SRC]                                      # (13,2) normalized targets

    # init: T-pose, global rotation to lie supine (guess about X)
    theta0 = np.zeros(72)
    theta0[0:3] = [np.pi / 2, 0.0, 0.0]

    def obj(x):
        theta = x[:72]
        s = float(np.exp(x[72]))
        tx, ty = x[73], x[74]
        J = joints(theta)
        pred = np.stack([s * J[DST, 0] + tx, s * J[DST, 1] + ty], axis=1)
        err = pred - tgt
        return float((err ** 2).sum()) + 0.01 * float((theta ** 2).sum())

    x0 = np.concatenate([theta0, [0.0, 0.0, 0.0]])     # log s = 0 -> s = 1
    res = None
    from scipy.optimize import minimize
    res = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-9})
    return res, tgt, ref, centre


def main():
    f = sys.argv[1] if len(sys.argv) > 1 else None
    if f is None:
        d = ROOT / "viddata" / "sensor_data" / "sensor_data" / "UMGDatensatz_slicesselected_vitpose_fm"
        f = str(next(d.iterdir()))
    kps = np.loadtxt(f)                                 # (250, 34)
    print(f"file: {Path(f).name}")

    for frame in [0, 50, 125]:
        kp = kps[frame].reshape(17, 2)
        res, tgt, ref, centre = fit_frame(kp)
        rmse_norm = float(np.sqrt(res.fun / len(SRC)))
        rmse_px = rmse_norm * ref
        print(f"  frame {frame:3d}: reproj RMSE {rmse_px:.1f} px  "
              f"(normalized {rmse_norm:.4f})  success={res.success}")


if __name__ == "__main__":
    main()
