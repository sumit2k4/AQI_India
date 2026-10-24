"""LSTM and TCN baselines for station-level PM2.5 forecasting (GPU).

Multi-output design: one model predicts all horizons {1,6,12,24} jointly from
a 168-h input window of [pm2.5, missing-mask, at, rh, ws, hour/month
harmonics] plus a station embedding. Trained on 2019-2023, early-stopped on
2024 (calibration year), evaluated once on 2025. Targets z-scored on train
statistics; L1 loss; missing targets masked.

Outputs: columns lstm/tcn appended to results/preds_h{h}.parquet,
per-epoch curves in results/dl_curves.csv, cost rows in results/cost_dl.csv.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS = [1, 6, 12, 24]
LOOK = 168
EPOCHS = 8
BATCH = 512
TRAIN_PER_EPOCH = 400_000
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_arrays():
    df = pd.read_parquet(os.path.join(PROC, "hourly_pm25.parquet"),
                         columns=["ts", "sid", "pm25", "at", "rh", "ws"])
    df = df.sort_values(["sid", "ts"])
    idx = pd.date_range(df.ts.min(), df.ts.max(), freq="1H")
    sids = sorted(df.sid.unique())
    smap = {s: i for i, s in enumerate(sids)}
    nT, nS = len(idx), len(sids)
    X = np.full((nS, nT, 4), np.nan, dtype="float32")  # pm25, at, rh, ws
    for k, col in enumerate(["pm25", "at", "rh", "ws"]):
        w = df.pivot(index="ts", columns="sid", values=col).reindex(idx)
        X[:, :, k] = w[sids].values.T.astype("float32")
    return X, idx, sids, smap


class LSTMNet(nn.Module):
    def __init__(self, nfeat, nstation, nh=128, emb=8):
        super().__init__()
        self.emb = nn.Embedding(nstation, emb)
        self.rnn = nn.LSTM(nfeat, nh, num_layers=2, batch_first=True,
                           dropout=0.2)
        self.head = nn.Sequential(nn.Linear(nh + emb, 64), nn.ReLU(),
                                  nn.Linear(64, len(HORIZONS)))

    def forward(self, x, sid):
        o, _ = self.rnn(x)
        z = torch.cat([o[:, -1], self.emb(sid)], dim=1)
        return self.head(z)


class TCNBlock(nn.Module):
    def __init__(self, cin, cout, dil):
        super().__init__()
        self.pad = 2 * dil
        self.c1 = nn.Conv1d(cin, cout, 3, dilation=dil)
        self.c2 = nn.Conv1d(cout, cout, 3, dilation=dil)
        self.act = nn.ReLU()
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = nn.functional.pad(x, (self.pad, 0))
        y = self.act(self.c1(y))
        y = nn.functional.pad(y, (self.pad, 0))
        y = self.act(self.c2(y))
        return y + self.skip(x)


class TCNNet(nn.Module):
    def __init__(self, nfeat, nstation, ch=64, emb=8):
        super().__init__()
        blocks, cin = [], nfeat
        for d in (1, 2, 4, 8, 16, 32):
            blocks.append(TCNBlock(cin, ch, d))
            cin = ch
        self.tcn = nn.Sequential(*blocks)
        self.emb = nn.Embedding(nstation, emb)
        self.head = nn.Sequential(nn.Linear(ch + emb, 64), nn.ReLU(),
                                  nn.Linear(64, len(HORIZONS)))

    def forward(self, x, sid):
        z = self.tcn(x.transpose(1, 2))[:, :, -1]
        z = torch.cat([z, self.emb(sid)], dim=1)
        return self.head(z)


def make_samples(X, idx, span):
    """Return (station_idx, time_idx) pairs whose window ends in span and
    which have any valid target."""
    t0, t1 = span
    tmask = (idx >= t0) & (idx <= t1)
    tids = np.where(tmask)[0]
    tids = tids[tids >= LOOK]
    pm = X[:, :, 0]
    pairs = []
    for s in range(X.shape[0]):
        cur_ok = ~np.isnan(pm[s, tids])          # need pm25(t) for regime/lag
        any_tgt = np.zeros(len(tids), dtype=bool)
        for h in HORIZONS:
            th = tids + h
            th = np.clip(th, 0, X.shape[1] - 1)
            any_tgt |= ~np.isnan(pm[s, th])
        ok = cur_ok & any_tgt
        pairs.append(np.stack([np.full(ok.sum(), s), tids[ok]], axis=1))
    return np.concatenate(pairs)


def batch_tensors(X, Xf, pairs, mu, sd, hourv, monv):
    s, t = pairs[:, 0], pairs[:, 1]
    wins = np.stack([Xf[si, ti - LOOK + 1:ti + 1] for si, ti in pairs])
    tt = np.arange(-LOOK + 1, 1)[None, :] + t[:, None]
    hh = hourv[tt]
    mm = monv[tt]
    cal = np.stack([np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24),
                    np.sin(2 * np.pi * mm / 12), np.cos(2 * np.pi * mm / 12)],
                   axis=2).astype("float32")
    xb = np.concatenate([wins, cal], axis=2)
    ys = np.full((len(pairs), len(HORIZONS)), np.nan, dtype="float32")
    for j, h in enumerate(HORIZONS):
        th = np.clip(t + h, 0, X.shape[1] - 1)
        ys[:, j] = (X[s, th, 0] - mu) / sd
    return xb, s.astype("int64"), ys


def run_model(name, net, X, Xf, idx, sids, tr_pairs, ca_pairs, te_pairs,
              mu, sd, curves, cost):
    hourv = idx.hour.values
    monv = idx.month.values
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    nparams = sum(p.numel() for p in net.parameters())
    best, patience, t_start = np.inf, 0, time.time()
    ca_sub = ca_pairs[np.random.choice(len(ca_pairs),
                                       min(100_000, len(ca_pairs)),
                                       replace=False)]
    for ep in range(EPOCHS):
        net.train()
        sub = tr_pairs[np.random.choice(len(tr_pairs),
                                        min(TRAIN_PER_EPOCH, len(tr_pairs)),
                                        replace=False)]
        tl, nb = 0.0, 0
        for i in range(0, len(sub), BATCH):
            xb, sb, yb = batch_tensors(X, Xf, sub[i:i + BATCH], mu, sd,
                                       hourv, monv)
            xb = torch.from_numpy(xb).to(DEV)
            sb = torch.from_numpy(sb).to(DEV)
            yb = torch.from_numpy(yb).to(DEV)
            mask = ~torch.isnan(yb)
            pred = net(xb, sb)
            loss = (pred[mask] - yb[mask]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
            nb += 1
        net.eval()
        vl, vb = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ca_sub), BATCH):
                xb, sb, yb = batch_tensors(X, Xf, ca_sub[i:i + BATCH], mu, sd,
                                           hourv, monv)
                xb = torch.from_numpy(xb).to(DEV)
                sb = torch.from_numpy(sb).to(DEV)
                yb = torch.from_numpy(yb).to(DEV)
                mask = ~torch.isnan(yb)
                vl += (net(xb, sb)[mask] - yb[mask]).abs().mean().item()
                vb += 1
        tr_l, va_l = tl / max(nb, 1), vl / max(vb, 1)
        curves.append(dict(model=name, epoch=ep + 1, train_loss=tr_l,
                           val_loss=va_l))
        print("%s epoch %d train %.4f val %.4f" % (name, ep + 1, tr_l, va_l),
              flush=True)
        if va_l < best - 1e-4:
            best, patience = va_l, 0
            torch.save(net.state_dict(), os.path.join(RES, name + ".pt"))
        else:
            patience += 1
            if patience >= 2:
                print("early stop", flush=True)
                break
    fit_s = time.time() - t_start
    net.load_state_dict(torch.load(os.path.join(RES, name + ".pt")))
    net.eval()
    # predict cal+test
    outs = {}
    t_inf = 0.0
    for tag, pairs in (("cal", ca_pairs), ("test", te_pairs)):
        preds = np.zeros((len(pairs), len(HORIZONS)), dtype="float32")
        with torch.no_grad():
            for i in range(0, len(pairs), BATCH):
                xb, sb, _ = batch_tensors(X, Xf, pairs[i:i + BATCH], mu, sd,
                                          hourv, monv)
                t1 = time.time()
                p = net(torch.from_numpy(xb).to(DEV),
                        torch.from_numpy(sb).to(DEV)).cpu().numpy()
                t_inf += time.time() - t1
                preds[i:i + BATCH] = p
        outs[tag] = preds * sd + mu
    cost.append(dict(model=name, fit_s=round(fit_s, 1),
                     pred_s=round(t_inf, 1), n_params=int(nparams),
                     n_train=len(tr_pairs)))
    return outs


def main():
    print("device:", DEV, flush=True)
    X, idx, sids, smap = build_arrays()
    pm_tr = X[:, (idx >= "2019-01-08") & (idx <= "2023-12-31"), 0]
    mu, sd = np.nanmean(pm_tr), np.nanstd(pm_tr)
    # feature tensor: [pm25_filled_z, missing_mask, at_z, rh_z, ws_z]
    med = np.nanmedian(X, axis=(0, 1))
    Xf = np.empty((X.shape[0], X.shape[1], 5), dtype="float32")
    Xf[:, :, 1] = np.isnan(X[:, :, 0]).astype("float32")
    for k, (kk, m0, s0) in enumerate(
            [(0, mu, sd),
             (1, np.nanmean(X[:, :, 1]), np.nanstd(X[:, :, 1])),
             (2, np.nanmean(X[:, :, 2]), np.nanstd(X[:, :, 2])),
             (3, np.nanmean(X[:, :, 3]), np.nanstd(X[:, :, 3]))]):
        v = X[:, :, kk].copy()
        v[np.isnan(v)] = med[kk]
        Xf[:, :, 0 if kk == 0 else kk + 1] = (v - m0) / max(s0, 1e-6)

    tr_pairs = make_samples(X, idx, ("2019-01-08", "2023-12-30"))
    ca_pairs = make_samples(X, idx, ("2024-01-01", "2024-12-30"))
    te_pairs = make_samples(X, idx, ("2025-01-01", "2025-12-30"))
    print("pairs train/cal/test:", len(tr_pairs), len(ca_pairs),
          len(te_pairs), flush=True)

    curves, cost = [], []
    nfeat = 5 + 4
    results = {}
    for name, cls in (("lstm", LSTMNet), ("tcn", TCNNet)):
        net = cls(nfeat, len(sids)).to(DEV)
        results[name] = run_model(name, net, X, Xf, idx, sids, tr_pairs,
                                  ca_pairs, te_pairs, mu, sd, curves, cost)
        pd.DataFrame(curves).to_csv(os.path.join(RES, "dl_curves.csv"),
                                    index=False)
        pd.DataFrame(cost).to_csv(os.path.join(RES, "cost_dl.csv"),
                                  index=False)

    # append to preds files
    inv = {i: s for s, i in smap.items()}
    for j, h in enumerate(HORIZONS):
        f = os.path.join(RES, "preds_h%d.parquet" % h)
        if not os.path.exists(f):
            print("no preds file for h=%d, skipping append" % h, flush=True)
            continue
        preds = pd.read_parquet(f)
        for name in ("lstm", "tcn"):
            frames = []
            for tag, pairs in (("cal", ca_pairs), ("test", te_pairs)):
                d = pd.DataFrame({
                    "ts": idx[pairs[:, 1]],
                    "sid": [inv[s] for s in pairs[:, 0]],
                    name: results[name][tag][:, j]})
                frames.append(d)
            col = pd.concat(frames, ignore_index=True)
            if name in preds.columns:
                preds = preds.drop(columns=[name])
            preds = preds.merge(col, on=["ts", "sid"], how="left")
        preds.to_parquet(f, index=False)
        print("appended lstm/tcn to", f, flush=True)
    print("DL DONE", flush=True)


if __name__ == "__main__":
    main()
