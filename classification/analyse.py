"""Offline analysis of saved runs (classification/runs/<tag>/fold*.npz).

For each tag reports, over the 9 test folds:
  * paper  : the single run with min validation loss (the paper's protocol)
  * mean   : average BA of all runs (expected single-run performance)
  * ens    : BA of the averaged logits of all runs (seed ensemble)
  * session: ensemble probabilities averaged per recording session
  * discord: accuracy on the clips whose label disagrees with the age group
             (FM+ at 4 weeks / FM- at 12-16 weeks) -- the only clips where
             movement, not age, decides the label
and a two-sided Wilcoxon signed-rank test between two tags on per-fold BA.

Run:  python classification/analyse.py paper_r8 [other_tag ...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RUNS = Path(__file__).resolve().parent / "runs"


def ba(y, score, thr=0.0):
    yhat = score > thr
    return 0.5 * ((yhat & (y == 1)).sum() / max((y == 1).sum(), 1)
                  + (~yhat & (y == 0)).sum() / max((y == 0).sum(), 1))


def session(name):
    return re.match(r"(idxp\d+_\d+d)", name).group(1)


def age(name):
    return int(re.search(r"_(\d+)d_", name).group(1))


def load(tag):
    return [np.load(RUNS / tag / f"fold{k}.npz") for k in range(1, 10)
            if (RUNS / tag / f"fold{k}.npz").exists()]


def summarise(tag):
    folds = load(tag)
    rows = {"paper": [], "mean": [], "ens": [], "session": []}
    disc_hit, disc_n = 0, 0
    for d in folds:
        y, L = d["y_test"], d["test_logits"]
        names = [str(n) for n in d["names_test"]]
        rows["paper"].append(ba(y, L[np.argmin(d["val_loss"])]))
        rows["mean"].append(np.mean([ba(y, l) for l in L]))
        p = 1 / (1 + np.exp(-L))
        pe = p.mean(0)
        rows["ens"].append(ba(y, pe, 0.5))
        sess = np.array([session(n) for n in names])
        ps = np.array([pe[sess == s].mean() for s in sess])
        rows["session"].append(ba(y, ps, 0.5))
        young = np.array([age(n) < 50 for n in names])
        disc = (young & (y == 1)) | (~young & (y == 0))
        disc_hit += ((pe[disc] > 0.5) == (y[disc] == 1)).sum()
        disc_n += disc.sum()
    out = {k: np.array(v) for k, v in rows.items()}
    print(f"\n[{tag}]  folds={len(folds)}  runs/fold={folds[0]['test_logits'].shape[0]}")
    for k, v in out.items():
        print(f"  {k:8s} BA {100 * v.mean():5.1f}%  per-fold {np.round(100 * v, 1).tolist()}")
    print(f"  discord  {disc_hit}/{disc_n} age-discordant clips correct")
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["paper_r8"]
    res = {t: summarise(t) for t in tags}
    base = tags[0]
    for t in tags[1:]:
        for k in ("paper", "ens"):
            a, b = res[base][k], res[t][k]
            n = min(len(a), len(b))
            if n >= 5 and not np.allclose(a[:n], b[:n]):
                p = wilcoxon(b[:n], a[:n]).pvalue
                print(f"  {t} vs {base} [{k}]: delta {100 * (b[:n] - a[:n]).mean():+.2f} pts, "
                      f"wins {(b[:n] > a[:n]).sum()}/{n}, Wilcoxon p={p:.4f}")
