"""Test whether sequences 05/11 differ by a uniform body-scale factor."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproduction"))

import config
import loader

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def bones(seq, model):
    data = loader.load_frame(seq, 0)
    model.betas[:] = data["betas"]
    model.pose[:] = data["pose"]
    model.trans[:] = data["trans"]
    Jtr = loader.as_np(model.J_transformed)
    gt = data["joints3d"][:24, :3]
    bm = np.array([np.linalg.norm(Jtr[i] - Jtr[PARENTS[i]]) for i in range(1, 24)])
    bg = np.array([np.linalg.norm(gt[i] - gt[PARENTS[i]]) for i in range(1, 24)])
    return bm, bg


def main():
    model = loader.load_smil_model()
    for seq in ["01", "05", "11"]:
        bm, bg = bones(seq, model)
        ratios = bg / bm
        # robust uniform scale = median ratio
        s = np.median(ratios)
        print(f"seq {seq}: bone length GT/model ratios: min {ratios.min():.3f} "
              f"median {s:.3f} max {ratios.max():.3f}")
        print(f"         spread (max/min) = {ratios.max()/ratios.min():.3f}  "
              f"(1.00 = perfectly uniform scale)")
        # residual after applying uniform scale
        resid = np.abs(bg - s * bm).mean() * 1000.0
        print(f"         mean |GT - s*model| per bone = {resid:.2f} mm")


if __name__ == "__main__":
    main()
