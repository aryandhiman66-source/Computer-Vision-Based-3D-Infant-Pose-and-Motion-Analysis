"""Loaders for MINI-RGBD frame data and the SMIL model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import config


def ensure_smil_on_path() -> None:
    pkg = str(config.SMIL_PKG)
    if pkg not in sys.path:
        sys.path.insert(0, pkg)


def load_smil_model():
    """Load the SMIL model via the official loader (chumpy-based)."""
    ensure_smil_on_path()
    from smil_webuser.serialization import load_model
    return load_model(str(config.SMIL_MODEL))


def as_np(ch_array) -> np.ndarray:
    """Convert a chumpy expression (or numpy array) to a float64 numpy array."""
    arr = ch_array.r if hasattr(ch_array, "r") else ch_array
    return np.asarray(arr, dtype=np.float64)


def load_txt(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def load_frame(sequence: str, frame: int):
    """Load every ground-truth artifact for one frame of one sequence.

    Returns a dict with keys: rgb, depth, mask, joints3d, joints2d,
    pose, trans, betas, frame_dir.
    """
    seq_dir = config.DATASET_DIR / sequence
    f = f"{frame:05d}"

    import cv2

    rgb = cv2.cvtColor(cv2.imread(str(seq_dir / "rgb" / f"syn_{f}.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(seq_dir / "depth" / f"syn_{f}_depth.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(seq_dir / "fg_mask" / f"mask_{f}.png"), cv2.IMREAD_UNCHANGED)

    # joints_3D:  X Y Z jointID   (meters)
    joints3d = load_txt(seq_dir / "joints_3D" / f"syn_joints_3D_{f}.txt")
    # joints_2Ddep: X Y depth jointID  (pixels, mm)
    joints2d = load_txt(seq_dir / "joints_2Ddep" / f"syn_joints_2Ddep_{f}.txt")

    pose = load_txt(seq_dir / "smil_params" / f"{f}_pose.txt")
    trans = load_txt(seq_dir / "smil_params" / f"{f}_trans.txt")
    betas = load_txt(seq_dir / "smil_shape_betas.txt")

    return {
        "rgb": rgb,
        "depth": depth,
        "mask": mask,
        "joints3d": joints3d,      # (25, 4): x y z id
        "joints2d": joints2d,      # (25, 4): u v depth_mm id
        "pose": pose,              # (72,)
        "trans": trans,            # (3,)
        "betas": betas,            # (20,)
        "frame_dir": seq_dir,
    }
