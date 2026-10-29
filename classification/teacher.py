"""Cheap sensor teacher for cross-modal distillation (train with sensors, test video-only).

Hand-crafted per-channel statistics of IMU (and optionally video skeleton) signals,
classified with gradient boosting. Out-of-fold (inner, infant-wise) predictions on each
outer fold's TRAINING clips are produced so the student never sees teacher outputs that
were fit on its own test infants.

Run:  python classification/teacher.py
Writes classification/runs/teacher/fold{k}.npz with p_train (aligned to the raw sample
list order), p_test, and the teacher's outer-fold test BA.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from faithful import _cached_txt

OUT = Path(__file__).resolve().parent / "runs" / "teacher"


def infant(name):
    return re.match(r"idxp(\d+)", name).group(1)


def channel_stats(x, fs):
    """x: (N,T,C) -> (N, C*8) amplitude, smoothness and band-power features."""
    d = np.diff(x, axis=1)
    spec = np.abs(np.fft.rfft(x - x.mean(1, keepdims=True), axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1], 1 / fs)
    bands = [(0.3, 1), (1, 2), (2, 4), (4, 8)]
    bp = [spec[:, (f >= a) & (f < b)].sum(1) for a, b in bands]
    tot = spec[:, f >= 0.3].sum(1) + 1e-9
    feats = [x.std(1), np.abs(d).mean(1), d.std(1), np.percentile(np.abs(d), 90, axis=1)]
    feats += [np.log(b + 1e-9) for b in bp] + [b / tot for b in bp][:0]
    return np.concatenate(feats, axis=1)


def load_modality(k, mod):
    if mod == "imu":
        d, T, C, fs = config.VIDDATA / "data_imu_9fold", 300, 36, 60
        stem = "input_imu_9fold_fold_{}_{}.txt"
    else:
        d, T, C, fs = config.KPC_DIR, 250, 30, 50
        stem = "input_vitpose_9fold_fold_{}_{}.txt"
    tr = np.reshape(_cached_txt(d / stem.format(k, "train")), (-1, T, C), order="F")
    te = np.reshape(_cached_txt(d / stem.format(k, "test")), (-1, T, C), order="F")
    return channel_stats(tr, fs), channel_stats(te, fs)


def labels(names):
    return np.array([int(n[65]) for n in names])


def clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=0)


def run(mods=("imu", "vid")):
    OUT.mkdir(parents=True, exist_ok=True)
    bas = []
    for k in range(1, 10):
        ntr = (config.SAMPLE_LIST_DIR / f"sample_list_vitpose_9fold_fold_{k}_train.txt").read_text().split()
        nte = (config.SAMPLE_LIST_DIR / f"sample_list_vitpose_9fold_fold_{k}_test.txt").read_text().split()
        parts = [load_modality(k, m) for m in mods]
        Xtr = np.concatenate([p[0] for p in parts], 1)
        Xte = np.concatenate([p[1] for p in parts], 1)
        ytr, yte = labels(ntr), labels(nte)
        # inner infant-wise 8-fold out-of-fold predictions on the training clips
        inf = np.array([infant(n) for n in ntr])
        uniq = np.unique(inf)
        groups = np.array_split(np.random.default_rng(k).permutation(uniq), 8)
        p_train = np.zeros(len(ntr))
        for g in groups:
            m = np.isin(inf, g)
            p_train[m] = clf().fit(Xtr[~m], ytr[~m]).predict_proba(Xtr[m])[:, 1]
        p_test = clf().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        ba = balanced_accuracy_score(yte, p_test > 0.5)
        oof = balanced_accuracy_score(ytr, p_train > 0.5)
        bas.append(ba)
        np.savez(OUT / f"fold{k}.npz", p_train=p_train, p_test=p_test, names_train=np.array(ntr))
        print(f"fold {k}: teacher test BA {ba:.4f}  inner-OOF train BA {oof:.4f}", flush=True)
    print(f"teacher {'+'.join(mods)} mean test BA {np.mean(bas):.4f}")


if __name__ == "__main__":
    run(tuple(sys.argv[1].split("+")) if len(sys.argv) > 1 else ("imu", "vid"))
