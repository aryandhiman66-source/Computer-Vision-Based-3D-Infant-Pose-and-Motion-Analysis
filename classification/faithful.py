"""Keras-faithful re-implementation of the paper's video CNN, vectorised over runs.

Fixes vs the earlier port (reproduce_paper.py / beat2.PaperCNN_fast, since removed) (see LOGBOOK, 2026-10-21):
  * Conv2D((kw,1)) shares ONE kernel across all 60 feature columns. We fold the
    columns into the batch: (B,T,F) -> (B*F,1,T). (groups=60 Conv1d is NOT
    equivalent: it learns a separate kernel per column, ~60x the parameters.)
  * Keras layer order is Conv(relu) -> BN -> Dropout  (ReLU *before* BN).
  * Keras BN: momentum 0.99 (= torch 0.01), eps 1e-3.  Adam eps 1e-7.
  * Conv kernels orthogonal, Dense glorot-uniform, zero biases.
  * Keras fit(): reshuffle every epoch, keep partial last batch,
    val = last 1/8 of the fixed-shuffled train set, EarlyStopping(val_loss,
    patience 10, restore_best_weights), max 1000 epochs.

All R runs of a fold are trained simultaneously as one grouped network, each
with its own init, its own batch order and its own early stopping. Every run's
validation loss and test logits are saved, so selection / ensembling /
session-level analyses can be done offline (classification/analyse.py).

Run:  python classification/faithful.py --runs 20 --tag paper
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
AMP = DEVICE == "cuda"          # fp16 autocast: ~2x faster; BN/loss stay fp32
OUT_DIR = Path(__file__).resolve().parent / "runs"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _cached_txt(path: Path) -> np.ndarray:
    cache = config.CACHE_DIR / "faithful" / (path.parent.name + "__" + path.stem + ".npy")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return np.load(cache)
    arr = np.loadtxt(path).astype(np.float32)
    np.save(cache, arr)
    return arr


def sample_names(k: int, split: str) -> list[str]:
    p = config.SAMPLE_LIST_DIR / f"sample_list_vitpose_9fold_fold_{k}_{split}.txt"
    return p.read_text().split()


def load_hyperpar():
    """The authors' hyper-parameter-tuning set: 337 clips from 9 infants that are disjoint
    from all 36 cross-validation infants (never used for training in the paper)."""
    d = config.VIDDATA / "data_vitpose_kpc_hyperpar"
    X = np.reshape(_cached_txt(d / "input_vitpose_hyperpar.txt"), (-1, 250, 30), order="F")
    y = _cached_txt(d / "output_vitpose_hyperpar.txt")
    x = np.zeros((X.shape[0], 250, 60), np.float32)
    x[:, :, :30] = X
    x[:, 1:, 30:] = np.diff(X, axis=1)
    return x, y


def load_fold(k: int):
    """Paper-exact features: positions (30) + velocity (30), global z-score per
    group with train statistics. Returns train (fixed-shuffled), test, names."""
    d = config.KPC_DIR
    Xtr = _cached_txt(d / f"input_vitpose_9fold_fold_{k}_train.txt")
    Xte = _cached_txt(d / f"input_vitpose_9fold_fold_{k}_test.txt")
    ytr = _cached_txt(d / f"output_vitpose_9fold_fold_{k}_train.txt")
    yte = _cached_txt(d / f"output_vitpose_9fold_fold_{k}_test.txt")
    Xtr = np.reshape(Xtr, (-1, 250, 30), order="F")
    Xte = np.reshape(Xte, (-1, 250, 30), order="F")

    def feats(p):
        x = np.zeros((p.shape[0], 250, 60), np.float32)
        x[:, :, :30] = p
        x[:, 1:, 30:] = np.diff(p, axis=1)
        return x

    Xtr, Xte = feats(Xtr), feats(Xte)
    stats = []
    for sl in (slice(0, 30), slice(30, 60)):
        mu, sd = Xtr[:, :, sl].mean(), Xtr[:, :, sl].std()
        Xtr[:, :, sl] = (Xtr[:, :, sl] - mu) / sd
        Xte[:, :, sl] = (Xte[:, :, sl] - mu) / sd
        stats += [float(mu), float(sd)]

    idx = np.loadtxt(config.VIDDATA / "random_indices_9fold" /
                     f"random_indices_train_9fold_fold_{k}.txt", dtype=int) - 1
    names_tr = np.array(sample_names(k, "train"))[idx]
    return Xtr[idx], ytr[idx], Xte, yte, list(names_tr), sample_names(k, "test"), stats


# 15 kept ViTPose joints: nose, l/r eye, l/r shoulder, l/r elbow, l/r wrist,
# l/r hip, l/r knee, l/r ankle  ->  left<->right permutation for mirroring
MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]


def augment_batch(x, stats, rot_deg=8.0, p_flip=0.5):
    """Online skeleton augmentation on normalised features x (R,b,T,60).

    Random left/right mirror (skeletons are already rotated torso-vertical) and a
    small in-plane rotation. Both are linear, so they act identically on the
    mean-centred positions and on their velocities. No scaling: it would change
    velocity magnitude, the likely fidgety-movement signal."""
    mp, sp, mv, sv = stats
    R, b, T, _ = x.shape
    pos = (x[..., :30] * sp + mp).view(R, b, T, 15, 2)
    vel = (x[..., 30:] * sv + mv).view(R, b, T, 15, 2)
    flip = torch.rand(R, b, 1, 1, 1, device=x.device) < p_flip
    sign = torch.tensor([-1.0, 1.0], device=x.device)
    pos = torch.where(flip, pos[..., MIRROR, :] * sign, pos)
    vel = torch.where(flip, vel[..., MIRROR, :] * sign, vel)
    a = torch.randn(R, b, device=x.device) * math.radians(rot_deg)
    c, s_ = torch.cos(a)[..., None, None], torch.sin(a)[..., None, None]
    def rot(z):
        return torch.stack((c * z[..., 0] - s_ * z[..., 1], s_ * z[..., 0] + c * z[..., 1]), -1)
    pos, vel = rot(pos), rot(vel)
    return torch.cat(((pos.reshape(R, b, T, 30) - mp) / sp, (vel.reshape(R, b, T, 30) - mv) / sv), -1)


# --------------------------------------------------------------------------
# Model: R independent copies of the paper CNN in one grouped network
# --------------------------------------------------------------------------
class MultiPaperCNN(nn.Module):
    def __init__(self, R, n_feat=60, kn=(4, 32, 16), kw=(13, 25, 25), fc=128, p=0.2, n_aux=0):
        super().__init__()
        self.n_aux = n_aux
        self.R, self.Fn, self.kn, self.fc = R, n_feat, kn, fc
        cin = 1
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for c, w in zip(kn, kw):
            self.convs.append(nn.Conv1d(R * cin, R * c, w, padding=w // 2, groups=R))
            self.bns.append(nn.BatchNorm1d(R * c, momentum=0.01, eps=1e-3))
            cin = c
        d_in = n_feat * kn[-1]
        self.W1 = nn.Parameter(torch.empty(R, d_in, fc))
        self.b1 = nn.Parameter(torch.zeros(R, fc))
        self.bn_fc = nn.BatchNorm1d(R * fc, momentum=0.01, eps=1e-3)
        self.W2 = nn.Parameter(torch.empty(R, fc, 1))
        self.b2 = nn.Parameter(torch.zeros(R, 1))
        self.drop = nn.Dropout(p)
        if n_aux:
            self.Wa = nn.Parameter(torch.randn(R, d_in, n_aux) * (1.0 / math.sqrt(d_in)))
            self.ba = nn.Parameter(torch.zeros(R, n_aux))
        self._init()

    @torch.no_grad()
    def _init(self):
        R = self.R
        for conv in self.convs:
            w = conv.weight.view(R, -1, *conv.weight.shape[1:])
            for r in range(R):
                nn.init.orthogonal_(w[r])
            conv.bias.zero_()
        for W in (self.W1, self.W2):
            lim = math.sqrt(6.0 / (W.shape[1] + W.shape[2]))
            W.uniform_(-lim, lim)

    def forward(self, x, with_aux=False):
        """x: (R,B,T,F) per-run batches, or (B,T,F) shared -> logits (R,B)
        [, aux predictions (R,B,n_aux)]."""
        R = self.R
        if x.dim() == 3:
            x = x.unsqueeze(0).expand(R, *x.shape)
        _, B, T, Fn = x.shape
        h = x.permute(1, 3, 0, 2).reshape(B * Fn, R, T)          # columns -> batch
        for conv, bn in zip(self.convs, self.bns):
            h = self.drop(bn(F.relu(conv(h))))                    # Keras order
        h = h.mean(-1)                                            # AvgPool over time
        h = h.view(B, Fn, R, self.kn[-1]).permute(2, 0, 1, 3).reshape(R, B, -1)
        aux = torch.einsum("rbi,rio->rbo", h, self.Wa) + self.ba[:, None] if with_aux else None
        h = F.relu(torch.einsum("rbi,rio->rbo", h, self.W1) + self.b1[:, None])
        h = self.bn_fc(h.permute(1, 0, 2).reshape(B, R * self.fc))
        h = self.drop(h).view(B, R, self.fc).permute(1, 0, 2)
        out = (torch.einsum("rbi,rio->rbo", h, self.W2) + self.b2[:, None]).squeeze(-1)
        return (out, aux) if with_aux else out


@torch.no_grad()
def predict(model, X, chunk=16):
    """Shared-input logits (R,N), evaluated in chunks to bound GPU memory."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        return torch.cat([model(X[i:i + chunk]).float() for i in range(0, len(X), chunk)], 1)


def run_slices(model, R):
    """For every tensor in the state dict, a view (R, -1) of its per-run chunks."""
    return {k: v for k, v in model.state_dict().items() if v.dim() >= 1 and v.shape[0] % R == 0}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
# limb segments (child, parent) in the 15-joint layout: upper arm, forearm, thigh, shank
LIMBS = [(5, 3), (6, 4), (7, 5), (8, 6), (11, 9), (12, 10), (13, 11), (14, 12)]


def _raw(x, stats):
    """Normalised base features (...,T,60) -> raw positions, velocities (...,T,15,2)."""
    mp, sp, mv, sv = stats
    sh = x.shape[:-1]
    return (x[..., :30] * sp + mp).reshape(*sh, 15, 2), (x[..., 30:] * sv + mv).reshape(*sh, 15, 2)


def limb_relative(x, stats):
    """Limb-local motion: child-minus-parent joint trajectories for the 8 limb segments.
    The paper CNN never mixes feature columns before its dense layer, so it cannot
    form these itself. Inputs are per-joint mean-centred, so these are mean-centred
    deviations too -- no static limb length (body shape / age) is re-introduced."""
    pos, vel = _raw(x, stats)
    c, q = [a for a, _ in LIMBS], [b for _, b in LIMBS]
    sh = x.shape[:-1]
    return (pos[..., c, :] - pos[..., q, :]).reshape(*sh, 16), (vel[..., c, :] - vel[..., q, :]).reshape(*sh, 16)


def expand_features(x, stats, rstats):
    """(...,T,60) -> (...,T,92): base features + normalised limb-relative pos/vel."""
    rp, rv = limb_relative(x, stats)
    mrp, srp, mrv, srv = rstats
    return torch.cat((x, (rp - mrp) / srp, (rv - mrv) / srv), -1)


def crop_batch(x, stats, min_len):
    """Temporal-crop augmentation: each clip in the batch is replaced by a random
    window of length L ~ U[min_len, T] (same L within a batch), re-centred per joint
    like a real snippet. Velocities are unchanged differences of the same frames."""
    R, b, T, _ = x.shape
    L = int(torch.randint(min_len, T + 1, (1,)).item())
    if L >= T:
        return x
    start = torch.randint(0, T - L + 1, (R, b, 1, 1), device=x.device)
    idx = start + torch.arange(L, device=x.device).view(1, 1, L, 1)
    xc = torch.gather(x, 2, idx.expand(R, b, L, x.shape[-1]))
    mp, sp, _, _ = stats
    pos = xc[..., :30] * sp + mp
    pos = pos - pos.mean(2, keepdim=True)
    return torch.cat(((pos - mp) / sp, xc[..., 30:]), -1)


def teacher_targets(tag, k, names):
    """Teacher out-of-fold P(FM+) for the given (shuffled) training clip names."""
    d = np.load(OUT_DIR / tag / f"fold{k}.npz")
    lut = dict(zip([str(n) for n in d["names_train"]], d["p_train"]))
    return np.array([lut[n] for n in names], np.float32)


def imu_aux_targets(k, names_train, nval):
    """Cross-modal auxiliary targets: hand-crafted IMU motion statistics
    (amplitude, jerk, band power per sensor channel), z-scored on the training part.
    Used ONLY as a training signal -- inference is video-only."""
    from teacher import load_modality
    A_tr, _ = load_modality(k, "imu")                      # raw sample-list order
    raw = (config.SAMPLE_LIST_DIR / f"sample_list_vitpose_9fold_fold_{k}_train.txt").read_text().split()
    pos = {n: i for i, n in enumerate(raw)}
    A = A_tr[[pos[n] for n in names_train]].astype(np.float32)
    A[:, :4 * 36] = np.log(A[:, :4 * 36] + 1e-6)             # amplitude stats -> log
    mu, sd = A[:nval].mean(0), A[:nval].std(0) + 1e-6
    return (A - mu) / sd


def train_fold(k, R, seed, max_epochs=1000, patience=10, bs=4, aug=0.0, arch=None,
               teacher=None, alpha=0.0, aux_imu=0.0, rel=False, crop=0, extra=False, log=print):
    Xtr, ytr, Xte, yte, ntr, nte, stats = load_fold(k)
    nval = int(7 * Xtr.shape[0] / 8)
    Xt, yt, Xv, yv = Xtr[:nval], ytr[:nval], Xtr[nval:], ytr[nval:]
    if extra:
        # extra TRAINING clips only; validation split and normalisation stay identical
        Xh, yh = load_hyperpar()
        mp, sp, mv, sv = stats
        Xh[:, :, :30] = (Xh[:, :, :30] - mp) / sp
        Xh[:, :, 30:] = (Xh[:, :, 30:] - mv) / sv
        Xt, yt = np.concatenate([Xt, Xh]), np.concatenate([yt, yh.astype(np.float32)])
    if teacher:
        # distillation: soft targets on training clips only; val/test stay hard + video-only
        yt = (1 - alpha) * yt + alpha * teacher_targets(teacher, k, ntr[:nval])

    At_g = None
    if aux_imu > 0:
        At_g = torch.tensor(imu_aux_targets(k, ntr, nval)[:nval], device=DEVICE)

    torch.manual_seed(seed + 1000 * k)
    rng = np.random.default_rng(seed + 1000 * k)
    arch = dict(arch or {})
    if rel:
        arch["n_feat"] = 92
    model = MultiPaperCNN(R, **arch, n_aux=At_g.shape[1] if At_g is not None else 0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    scaler = torch.amp.GradScaler(enabled=AMP)

    Xt_g, yt_g = torch.tensor(Xt, device=DEVICE), torch.tensor(yt, device=DEVICE)
    Xv_g, yv_g = torch.tensor(Xv, device=DEVICE), torch.tensor(yv, device=DEVICE)
    Xte_g = torch.tensor(Xte, device=DEVICE)
    feat = lambda z: z
    if rel:
        with torch.no_grad():
            rp, rv = limb_relative(Xt_g, stats)                  # train-part statistics
            rstats = [rp.mean().item(), rp.std().item(), rv.mean().item(), rv.std().item()]
        feat = lambda z: expand_features(z, stats, rstats)
        Xv_e, Xte_e = feat(Xv_g), feat(Xte_g)
    else:
        Xv_e, Xte_e = Xv_g, Xte_g

    best_loss = np.full(R, np.inf)
    wait = np.zeros(R, int)
    active = np.ones(R, bool)
    stop_epoch = np.zeros(R, int)
    best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
    n = len(Xt)
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        perms = torch.tensor(np.stack([rng.permutation(n) for _ in range(R)]), device=DEVICE)
        # Keras keeps a partial last batch; torch BN cannot train on a batch of
        # 1, so a size-1 remainder is merged into the previous batch.
        starts = list(range(0, n, bs))
        if n % bs == 1 and len(starts) > 1:
            starts.pop()
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else n
            ib = perms[:, s:e]                                    # (R,b)
            xb = Xt_g[ib]
            if aug:
                xb = augment_batch(xb, stats, rot_deg=aug)
            if crop:
                xb = crop_batch(xb, stats, crop)
            xb = feat(xb)
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                if At_g is not None:
                    logits, aux = model(xb, with_aux=True)
                    logits, aux = logits.float(), aux.float()
                else:
                    logits = model(xb).float()
            loss = F.binary_cross_entropy_with_logits(logits, yt_g[ib], reduction="none").mean(1).sum()
            if At_g is not None:
                loss = loss + aux_imu * ((aux - At_g[ib]) ** 2).mean((1, 2)).sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(
                predict(model, Xv_e), yv_g.expand(R, -1), reduction="none").mean(1).cpu().numpy()
        improved = active & (vl < best_loss)
        if improved.any():
            cur = run_slices(model, R)
            for name, v in cur.items():
                bv = best_state[name].view(R, -1)
                vv = v.view(R, -1)
                for r in np.where(improved)[0]:
                    bv[r] = vv[r]
            best_loss[improved] = vl[improved]
            wait[improved] = 0
        wait[active & ~improved] += 1
        newly = active & (wait >= patience)
        stop_epoch[newly] = ep + 1
        active &= ~newly
        if not active.any():
            break
    stop_epoch[active] = ep + 1
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = predict(model, Xte_e).cpu().numpy()
        va_logits = predict(model, Xv_e).cpu().numpy()
    log(f"  fold {k}: {ep + 1} epochs, {time.time() - t0:.0f}s, "
        f"val_loss min {best_loss.min():.4f} median {np.median(best_loss):.4f}")
    return dict(val_loss=best_loss, test_logits=te_logits, val_logits=va_logits,
                y_test=yte, y_val=yv, names_test=np.array(nte),
                names_val=np.array(ntr[nval:]), stop_epoch=stop_epoch)


def balanced_accuracy(y, logits):
    yhat = logits > 0
    sens = (yhat & (y == 1)).sum() / max((y == 1).sum(), 1)
    spec = (~yhat & (y == 0)).sum() / max((y == 0).sum(), 1)
    return 0.5 * (sens + spec)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--folds", type=str, default="1-9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="paper")
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--kn", type=str, default="4,32,16")
    ap.add_argument("--kw", type=str, default="13,25,25")
    ap.add_argument("--fc", type=int, default=128)
    ap.add_argument("--teacher", type=str, default=None, help="runs/<tag> with teacher OOF probs")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--aug", type=float, default=0.0,
                    help="online mirror + rotation augmentation; value = rotation std in degrees (0 = off)")
    ap.add_argument("--rel", action="store_true", help="add limb-relative motion features (60 -> 92 columns)")
    ap.add_argument("--crop", type=int, default=0, help="temporal-crop augmentation, min window length in frames (0 = off)")
    ap.add_argument("--extra-hyperpar", action="store_true",
                    help="also train on the authors' 9 tuning infants (disjoint from all CV infants)")
    ap.add_argument("--aux-imu", type=float, default=0.0, help="weight of IMU-statistics auxiliary loss")
    a = ap.parse_args(argv)
    lo, hi = (int(v) for v in a.folds.split("-")) if "-" in a.folds else (int(a.folds),) * 2
    arch = dict(kn=tuple(int(v) for v in a.kn.split(",")),
                kw=tuple(int(v) for v in a.kw.split(",")), fc=a.fc)
    out = OUT_DIR / a.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    print(f"device {DEVICE}  tag {a.tag}  runs {a.runs}  arch {arch}")
    for k in range(lo, hi + 1):
        res = train_fold(k, a.runs, a.seed, max_epochs=a.max_epochs, aug=a.aug, arch=arch,
                         teacher=a.teacher, alpha=a.alpha, aux_imu=a.aux_imu,
                         rel=a.rel, crop=a.crop, extra=a.extra_hyperpar)
        np.savez(out / f"fold{k}.npz", **res)
        sel = int(np.argmin(res["val_loss"]))
        bas = [balanced_accuracy(res["y_test"], l) for l in res["test_logits"]]
        ens = balanced_accuracy(res["y_test"], res["test_logits"].mean(0))
        print(f"  fold {k}: paper-selected BA {bas[sel]:.4f} | mean-run {np.mean(bas):.4f} "
              f"| ensemble {ens:.4f}", flush=True)


if __name__ == "__main__":
    main()
