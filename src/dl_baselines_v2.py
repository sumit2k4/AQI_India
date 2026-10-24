"""Re-tuned sequence baselines plus a Transformer encoder baseline (revision,
reviewer item 12.3). Same data, splits, inputs, loss and station embedding as
dl_baselines.py; the training recipe differs:

  * full pass over all training windows per epoch (no 400k subsample)
  * up to 20 epochs, early stopping with patience 5 on the 2024 calibration year
  * Adam 1e-3 with ReduceLROnPlateau (factor 0.5, patience 2), grad-clip 1.0
  * LSTM: 2 x 192 hidden, dropout 0.2;  TCN: 8 blocks (dilations to 128), 96 ch
  * Transformer: patch encoder (21 patches of 8 h), 4 layers, d_model 64, 4 heads,
    learned positional embedding, last-patch readout; optional mixed precision (--amp).
    Reported run: epochs 1-5 with --amp (best checkpoint = epoch 3), resumed in fp32
    for epochs 6-8 after CUDA illegal-memory-access faults with torch 1.9 on an RTX 2060

Usage: python src/dl_baselines_v2.py --models lstm2   (one process per model,
they can run concurrently on one GPU; training
resumes from results/<model>_ckpt.pt if interrupted), then python src/dl_baselines_v2.py --merge
to append lstm2 / tcn2 / transformer to results/preds_h{h}.parquet and write
results/dl_curves_v2.csv and results/cost_dl_v2.csv.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import dl_baselines as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS = D.HORIZONS
LOOK = D.LOOK
EPOCHS = 20
PATIENCE = 5
BATCH = 512
SEED = 42
USE_AMP = False   # set by --amp (Transformer only)
torch.manual_seed(SEED)
np.random.seed(SEED)


_OFF = np.arange(-D.LOOK + 1, 1)
_CAL = {}


def batch_fast(X, Xf, pairs, mu, sd, hourv, monv):
    """Vectorised equivalent of dl_baselines.batch_tensors (identical values)."""
    s, t = pairs[:, 0], pairs[:, 1]
    tt = t[:, None] + _OFF[None, :]
    key = (id(hourv), id(monv))
    if _CAL.get("key") != key:   # per-timestamp calendar table, computed once
        _CAL["key"] = key
        _CAL["tab"] = np.stack([np.sin(2 * np.pi * hourv / 24), np.cos(2 * np.pi * hourv / 24),
                                np.sin(2 * np.pi * monv / 12), np.cos(2 * np.pi * monv / 12)],
                               axis=1).astype("float32")
    xb = np.empty((len(pairs), D.LOOK, Xf.shape[2] + 4), dtype="float32")
    xb[:, :, :Xf.shape[2]] = Xf[s[:, None], tt]
    xb[:, :, Xf.shape[2]:] = _CAL["tab"][tt]
    ys = np.full((len(pairs), len(HORIZONS)), np.nan, dtype="float32")
    for j, h in enumerate(HORIZONS):
        th = np.clip(t + h, 0, X.shape[1] - 1)
        ys[:, j] = (X[s, th, 0] - mu) / sd
    return xb, s.astype("int64"), ys


class TransformerNet(nn.Module):
    """Patch-based Transformer encoder (PatchTST-style, Nie et al. 2023): the 168 h
    window is cut into 21 non-overlapping 8 h patches, each linearly embedded,
    with a learned positional embedding, 4 encoder layers and last-patch readout."""

    PATCH = 8

    def __init__(self, nfeat, nstation, d=64, heads=4, layers=4, emb=8):
        super().__init__()
        self.np = LOOK // self.PATCH
        self.inp = nn.Linear(nfeat * self.PATCH, d)
        self.pos = nn.Parameter(torch.zeros(1, self.np, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d,
                                           dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.emb = nn.Embedding(nstation, emb)
        self.head = nn.Sequential(nn.Linear(d + emb, 64), nn.ReLU(),
                                  nn.Linear(64, len(HORIZONS)))

    def forward(self, x, sid):
        b = x.shape[0]
        p = x.reshape(b, self.np, self.PATCH * x.shape[2])
        z = self.enc(self.inp(p) + self.pos)[:, -1]
        return self.head(torch.cat([z, self.emb(sid)], dim=1))


def run_model(name, net, X, Xf, idx, tr_pairs, ca_pairs, te_pairs, mu, sd,
              curves, cost):
    hourv, monv = idx.hour.values, idx.month.values
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    amp = USE_AMP and name == "transformer" and DEV == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    print("%s mixed precision: %s" % (name, amp), flush=True)
    nparams = sum(p.numel() for p in net.parameters())
    best, bad, t0 = np.inf, 0, time.time()
    ca_sub = ca_pairs[np.random.choice(len(ca_pairs), min(200_000, len(ca_pairs)),
                                       replace=False)]
    ckpt = os.path.join(RES, name + "_ckpt.pt")
    start, done = 0, False
    if os.path.exists(ckpt):
        c = torch.load(ckpt, map_location=DEV)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        if "scaler" in c:
            scaler.load_state_dict(c["scaler"])
        start, best, bad, done = c["epoch"], c["best"], c["bad"], c["done"]
        curves[:] = c["curves"]; t0 -= c["elapsed"]
        np.random.set_state(c["np_rng"]); torch.set_rng_state(c["torch_rng"].cpu())
        print("%s resumed after epoch %d (best %.4f, bad %d)" % (name, start, best, bad),
              flush=True)
    for ep in range(start, EPOCHS if not done else start):
        net.train()
        perm = np.random.permutation(len(tr_pairs))
        tl, nb = 0.0, 0
        for i in range(0, len(perm), BATCH):
            xb, sb, yb = batch_fast(X, Xf, tr_pairs[perm[i:i + BATCH]], mu, sd,
                                         hourv, monv)
            xb = torch.from_numpy(xb).to(DEV); sb = torch.from_numpy(sb).to(DEV)
            yb = torch.from_numpy(yb).to(DEV)
            mask = ~torch.isnan(yb)
            with torch.cuda.amp.autocast(enabled=amp):
                out = net(xb, sb)
            loss = (out.float()[mask] - yb[mask]).abs().mean()
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item(); nb += 1
        net.eval(); vl, vb = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ca_sub), BATCH):
                xb, sb, yb = batch_fast(X, Xf, ca_sub[i:i + BATCH], mu, sd,
                                             hourv, monv)
                xb = torch.from_numpy(xb).to(DEV); sb = torch.from_numpy(sb).to(DEV)
                yb = torch.from_numpy(yb).to(DEV)
                mask = ~torch.isnan(yb)
                with torch.cuda.amp.autocast(enabled=amp):
                    out = net(xb, sb)
                vl += (out.float()[mask] - yb[mask]).abs().mean().item(); vb += 1
        tr_l, va_l = tl / max(nb, 1), vl / max(vb, 1)
        sched.step(va_l)
        curves.append(dict(model=name, epoch=ep + 1, train_loss=tr_l, val_loss=va_l,
                           lr=opt.param_groups[0]["lr"]))
        print("%s epoch %d train %.4f val %.4f lr %.2e" % (
            name, ep + 1, tr_l, va_l, opt.param_groups[0]["lr"]), flush=True)
        pd.DataFrame(curves).to_csv(os.path.join(RES, "dl_curves_v2_%s.csv" % name), index=False)
        if va_l < best - 1e-4:
            best, bad = va_l, 0
            torch.save(net.state_dict(), os.path.join(RES, name + ".pt"))
        else:
            bad += 1
        done = bad >= PATIENCE
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=ep + 1, best=best, bad=bad, done=done, curves=list(curves),
                        elapsed=time.time() - t0, np_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state()), ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)
        if done:
            print("early stop", flush=True); break
    fit_s = time.time() - t0
    net.load_state_dict(torch.load(os.path.join(RES, name + ".pt")))
    net.eval()
    outs, t_inf = {}, 0.0
    for tag, pairs in (("cal", ca_pairs), ("test", te_pairs)):
        preds = np.zeros((len(pairs), len(HORIZONS)), dtype="float32")
        with torch.no_grad():
            for i in range(0, len(pairs), BATCH):
                xb, sb, _ = batch_fast(X, Xf, pairs[i:i + BATCH], mu, sd, hourv, monv)
                t1 = time.time()
                with torch.cuda.amp.autocast(enabled=amp):
                    out = net(torch.from_numpy(xb).to(DEV), torch.from_numpy(sb).to(DEV))
                preds[i:i + BATCH] = out.float().cpu().numpy()
                t_inf += time.time() - t1
        outs[tag] = preds * sd + mu
    cost.append(dict(model=name, fit_s=round(fit_s, 1), pred_s=round(t_inf, 1),
                     n_params=int(nparams), n_train=len(tr_pairs), best_val=best))
    return outs


def build_inputs():
    X, idx, sids, smap = D.build_arrays()
    pm_tr = X[:, (idx >= "2019-01-08") & (idx <= "2023-12-31"), 0]
    mu, sd = np.nanmean(pm_tr), np.nanstd(pm_tr)
    med = np.nanmedian(X, axis=(0, 1))
    Xf = np.empty((X.shape[0], X.shape[1], 5), dtype="float32")
    Xf[:, :, 1] = np.isnan(X[:, :, 0]).astype("float32")
    for kk, m0, s0 in [(0, mu, sd), (1, np.nanmean(X[:, :, 1]), np.nanstd(X[:, :, 1])),
                       (2, np.nanmean(X[:, :, 2]), np.nanstd(X[:, :, 2])),
                       (3, np.nanmean(X[:, :, 3]), np.nanstd(X[:, :, 3]))]:
        v = X[:, :, kk].copy(); v[np.isnan(v)] = med[kk]
        Xf[:, :, 0 if kk == 0 else kk + 1] = (v - m0) / max(s0, 1e-6)
    tr_pairs = D.make_samples(X, idx, ("2019-01-08", "2023-12-30"))
    ca_pairs = D.make_samples(X, idx, ("2024-01-01", "2024-12-30"))
    te_pairs = D.make_samples(X, idx, ("2025-01-01", "2025-12-30"))
    print("pairs train/cal/test:", len(tr_pairs), len(ca_pairs), len(te_pairs), flush=True)
    return X, Xf, idx, sids, smap, mu, sd, tr_pairs, ca_pairs, te_pairs


NETS = {"lstm2": lambda nfeat, ns: D.LSTMNet(nfeat, ns, nh=192),
        "tcn2": lambda nfeat, ns: TCN8(nfeat, ns),
        "transformer": lambda nfeat, ns: TransformerNet(nfeat, ns)}
NFEAT = 5 + 4


def train(models):
    """Train the selected models (one process per model is fine: each writes
    results/dlv2_<name>_{cal,test}.npy, dl_curves_v2_<name>.csv and
    cost_dl_v2_<name>.csv). Run `--merge` afterwards."""
    print("device:", DEV, flush=True)
    X, Xf, idx, sids, smap, mu, sd, tr_pairs, ca_pairs, te_pairs = build_inputs()
    for name in models:
        if os.path.exists(os.path.join(RES, "dlv2_%s_test.npy" % name)):
            print(name, "already trained, skipping", flush=True)
            continue
        curves, cost = [], []
        net = NETS[name](NFEAT, len(sids)).to(DEV)
        outs = run_model(name, net, X, Xf, idx, tr_pairs, ca_pairs, te_pairs,
                         mu, sd, curves, cost)
        pd.DataFrame(curves).to_csv(os.path.join(RES, "dl_curves_v2_%s.csv" % name), index=False)
        pd.DataFrame(cost).to_csv(os.path.join(RES, "cost_dl_v2_%s.csv" % name), index=False)
        for tag in ("cal", "test"):
            np.save(os.path.join(RES, "dlv2_%s_%s.npy" % (name, tag)), outs[tag])
        print(name, "saved", flush=True)
    print("DL V2 TRAIN DONE", models, flush=True)


def merge():
    """Append the saved predictions of every trained model to the preds files
    and concatenate the per-model curve / cost tables."""
    X, idx, sids, smap = D.build_arrays()
    ca_pairs = D.make_samples(X, idx, ("2024-01-01", "2024-12-30"))
    te_pairs = D.make_samples(X, idx, ("2025-01-01", "2025-12-30"))
    inv = {i: s for s, i in smap.items()}
    names = [n for n in NETS if os.path.exists(os.path.join(RES, "dlv2_%s_test.npy" % n))]
    results = {n: {tag: np.load(os.path.join(RES, "dlv2_%s_%s.npy" % (n, tag)))
                   for tag in ("cal", "test")} for n in names}
    for j, h in enumerate(HORIZONS):
        f = os.path.join(RES, "preds_h%d.parquet" % h)
        if not os.path.exists(f):
            print("no preds file for h=%d, skipping append" % h, flush=True)
            continue
        preds = pd.read_parquet(f)
        for name in names:
            frames = []
            for tag, pairs in (("cal", ca_pairs), ("test", te_pairs)):
                frames.append(pd.DataFrame({"ts": idx[pairs[:, 1]],
                                            "sid": [inv[s] for s in pairs[:, 0]],
                                            name: results[name][tag][:, j]}))
            col = pd.concat(frames, ignore_index=True)
            if name in preds.columns:
                preds = preds.drop(columns=[name])
            preds = preds.merge(col, on=["ts", "sid"], how="left")
        preds.to_parquet(f, index=False)
        print("appended", names, "to", f, flush=True)
    pd.concat([pd.read_csv(os.path.join(RES, "dl_curves_v2_%s.csv" % n)) for n in names],
              ignore_index=True).to_csv(os.path.join(RES, "dl_curves_v2.csv"), index=False)
    pd.concat([pd.read_csv(os.path.join(RES, "cost_dl_v2_%s.csv" % n)) for n in names],
              ignore_index=True).to_csv(os.path.join(RES, "cost_dl_v2.csv"), index=False)
    print("DL V2 MERGE DONE", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lstm2,tcn2,transformer")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision for the Transformer (see README)")
    a = ap.parse_args()
    global USE_AMP
    USE_AMP = a.amp
    if a.merge:
        merge()
    else:
        train([m.strip() for m in a.models.split(",") if m.strip()])


class TCN8(nn.Module):
    """Deeper TCN: 8 blocks, dilations 1..128 (receptive field > 168 h), 96 ch."""

    def __init__(self, nfeat, nstation, ch=96, emb=8):
        super().__init__()
        blocks, cin = [], nfeat
        for d in (1, 2, 4, 8, 16, 32, 64, 128):
            blocks.append(D.TCNBlock(cin, ch, d)); cin = ch
        self.tcn = nn.Sequential(*blocks)
        self.emb = nn.Embedding(nstation, emb)
        self.head = nn.Sequential(nn.Linear(ch + emb, 64), nn.ReLU(),
                                  nn.Linear(64, len(HORIZONS)))

    def forward(self, x, sid):
        z = self.tcn(x.transpose(1, 2))[:, :, -1]
        return self.head(torch.cat([z, self.emb(sid)], dim=1))


if __name__ == "__main__":
    main()
