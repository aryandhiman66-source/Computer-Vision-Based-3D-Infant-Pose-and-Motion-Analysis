"""Paths and constants for the MINI-RGBD -> sensor-fusion classification phase."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDDATA = ROOT / "viddata" / "data"
KPC_DIR = VIDDATA / "data_vitpose_kpc_9fold"     # centred skeleton features
SAMPLE_LIST_DIR = VIDDATA / "sample_list_vitpose_9fold"

N_FOLDS = 9
N_FRAMES = 250
N_KEYPOINTS = 15
N_COORDS = 2          # x, y  -> 30 position features per frame

CACHE_DIR = VIDDATA / "_cache"
