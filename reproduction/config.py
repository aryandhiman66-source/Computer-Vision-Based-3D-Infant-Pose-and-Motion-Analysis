"""Configuration for reproducing the MINI-RGBD dataset (Paper 2).

Camera intrinsics are taken verbatim from the official `python/depth_to_3D.py`
script shipped with the dataset.
"""
from pathlib import Path

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "MINI-RGBD_web"
SMIL_MODEL = REPO_ROOT / "smil" / "smil_web.pkl"
SMIL_PKG = REPO_ROOT / "smil"          # add to PYTHONPATH / sys.path
OUTPUT_DIR = REPO_ROOT / "reproduction" / "output"

# --- Camera (pinhole, Kinect-V1-like) ---
FX = 588.67905803875317
FY = 590.25690113005601
CX = 322.22048191353628
CY = 237.46785983766890
IMAGE_W = 640
IMAGE_H = 480
DEPTH_SCALE = 1000.0   # depth PNG values are mm -> divide by this to get meters

# --- Joint names (order matches MINI-RGBD jointlist.txt) ---
# Rows 0..23 are the 24 SMIL joints (same order as the model kintree table).
# Row 24 is `noseVertex`, a mesh vertex used only for evaluation.
JOINT_NAMES = [
    "global", "leftThigh", "rightThigh", "spine",
    "leftCalf", "rightCalf", "spine1", "leftFoot", "rightFoot", "spine2",
    "leftToes", "rightToes", "neck", "leftShoulder", "rightShoulder", "head",
    "leftUpperArm", "rightUpperArm", "leftForeArm", "rightForeArm",
    "leftHand", "rightHand", "leftFingers", "rightFingers",
    "noseVertex",
]
N_SMIL_JOINTS = 24  # first 24 rows are SMIL joints, last is noseVertex
