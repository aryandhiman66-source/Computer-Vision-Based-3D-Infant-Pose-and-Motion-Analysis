"""Step 4: Depth + mask evaluation across all 12 sequences (sampled frames).

Why sampled: the z-buffer render is ~0.6 s/frame and 12,000 frames would take
~2 h; 3 frames/sequence is enough to characterise depth/mask quality per sequence.

Run:  python reproduction/evaluate_depth_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import loader
from render_depth import rasterize_depth

FRAMES = [0, 250, 500]


def difficulty(seq: str) -> str:
    n = int(seq)
    return "easy" if n <= 4 else ("medium" if n <= 9 else "difficult")


def main():
    model = loader.load_smil_model()
    seqs = sorted(p.name for p in config.DATASET_DIR.glob("[0-9][0-9]"))
    print(f"{'seq':>3} {'group':>9} {'mask IoU':>9} {'depth MAE':>10} {'depth RMSE':>11}")
    print("-" * 46)
    summary = []
    for seq in seqs:
        ious, maes, rmses = [], [], []
        for f in FRAMES:
            data = loader.load_frame(seq, f)
            model.betas[:] = data["betas"]
            model.pose[:] = data["pose"]
            model.trans[:] = data["trans"]
            verts = loader.as_np(model.r)
            faces = np.asarray(model.f, dtype=np.int64)

            rendered = rasterize_depth(verts, faces)
            rmask = np.isfinite(rendered)
            gmask = data["mask"] > 0
            gdepth = data["depth"].astype(np.float64)
            inter = (rmask & gmask).sum(); union = (rmask | gmask).sum()
            iou = inter / union
            both = rmask & gmask
            diff = rendered[both] - gdepth[both]
            ious.append(iou); maes.append(np.abs(diff).mean()); rmses.append(np.sqrt((diff**2).mean()))

        iou_m, mae_m, rmse_m = np.mean(ious), np.mean(maes), np.mean(rmses)
        summary.append([seq, difficulty(seq), iou_m, mae_m, rmse_m])
        print(f"{seq:>3} {difficulty(seq):>9} {iou_m:>9.3f} {mae_m:>10.2f} {rmse_m:>11.2f}")

    out = config.OUTPUT_DIR / "depth_mask_all_sequences.csv"
    with open(out, "w") as fp:
        fp.write("sequence,difficulty,mask_iou_mean,depth_mae_mm,depth_rmse_mm\n")
        for r in summary:
            fp.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")

    print(f"\nsaved: {out}")
    for grp in ["easy", "medium", "difficult"]:
        vals = [r[2] for r in summary if r[1] == grp]
        if vals:
            print(f"  mask IoU {grp:>9}: mean {np.mean(vals):.3f}")


if __name__ == "__main__":
    main()
