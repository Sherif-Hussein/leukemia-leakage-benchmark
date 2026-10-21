"""Fine-tuned EfficientNet-B0 on CPU: partial fine-tuning of the last stages on cached feature maps.

Stage 1 (--cache): run the frozen ImageNet stem + blocks 0-4 once per image (224 x 224, bicubic,
ImageNet normalization) and store the 112 x 14 x 14 feature maps (float16).
Stage 2 (training): fine-tune blocks 5-6, the 1x1 head convolution (1,280 channels) and a new linear
classifier, initialized from ImageNet, on the cached maps. Same protocol as the other experiments:
  * outer 5-fold CV on Discovery (image-level or patient-grouped folds)
  * inside each training fold, 15% of patients (grouped) or images (image split) form an inner
    validation set used ONLY for early stopping (best epoch by validation loss)
  * external prediction = mean of the 5 fold models; threshold for sensitivity >= 0.95 chosen on the
    out-of-fold Discovery predictions; Validation scored once
  * T3 (patient diagnosis) from the T1 model: mean cell probability per patient (as in Sidhom et al.)
Augmentation: random flips and 90-degree rotations of the cached maps. Optimizer AdamW (weight decay
1e-4), cosine learning-rate schedule from 3e-4 with 1 warm-up epoch, batch 128, class-weighted BCE.

  python scripts/run_finetune_cached.py --manifest manifests/apl_aml.csv --weights weights/efficientnet_b0_ra-3dd342df.pth \
      --cache-file features/apl_aml_effb0_block4.npy --cache            # once
  python scripts/run_finetune_cached.py ... --task apl_cell_diagnosis --splits image grouped --out results/finetune_t1
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench import metrics as MT
from leukbench.encoders import ImageList, default_transform
from leukbench.runner import parse_kv
from leukbench.splits import make_folds
from leukbench.tasks import load_tasks, select

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--tasks", default=str(Path(__file__).resolve().parents[1] / "configs" / "tasks.json"))
ap.add_argument("--weights", required=True)
ap.add_argument("--cache-file", required=True)
ap.add_argument("--cache", action="store_true", help="compute the cached feature maps and exit")
ap.add_argument("--task")
ap.add_argument("--splits", nargs="+", default=["grouped"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--batch-size", type=int, default=128)
ap.add_argument("--max-train", type=int, default=0, help="subsample training cells per fold (0 = all)")
ap.add_argument("--cv-where", nargs="*", default=["cohort=Discovery"])
ap.add_argument("--test-where", nargs="*", default=["cohort=Validation"])
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--pool", default="avg", choices=["avg", "attn"], help="global average pooling or gated attention pooling over the 7x7 positions")
ap.add_argument("--out")
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)


def backbone():
    import timm
    m = timm.create_model("efficientnet_b0", pretrained=True, pretrained_cfg_overlay=dict(file=a.weights))
    return m.eval()


man = pd.read_csv(a.manifest, keep_default_na=False)

# ------------------------------------------------------------------ stage 1: cache
if a.cache:
    from torch.utils.data import DataLoader
    m = backbone()
    frozen = nn.Sequential(m.conv_stem, m.bn1, *m.blocks[:5]).eval()
    N = len(man)
    store = np.lib.format.open_memmap(a.cache_file, mode="w+", dtype=np.float16, shape=(N, 112, 14, 14))
    ok_all = np.zeros(N, bool)
    dl = DataLoader(ImageList(man.path.tolist(), default_transform()), batch_size=64, num_workers=2)
    i0, t0 = 0, time.time()
    with torch.no_grad():
        for x, ok in dl:
            z = frozen(x).numpy().astype(np.float16)
            store[i0:i0 + len(z)] = z
            ok_all[i0:i0 + len(z)] = ok.numpy()
            i0 += len(z)
            if (i0 // 64) % 40 == 0:
                print(f"  cached {i0:,}/{N:,} ({time.time() - t0:.0f}s)", flush=True)
    store.flush()
    np.save(a.cache_file.replace(".npy", "_ok.npy"), ok_all)
    print(f"cached {N:,} maps, {(~ok_all).sum()} unreadable, {time.time() - t0:.0f}s")
    sys.exit(0)

# ------------------------------------------------------------------ stage 2: fine-tune
F = np.load(a.cache_file, mmap_mode="r")
F = np.asarray(F)                       # ~1.1 GB in RAM
OK = np.load(a.cache_file.replace(".npy", "_ok.npy"))
spec = load_tasks(a.tasks)[a.task]
cv = select(man, spec, parse_kv(a.cv_where))
te = select(man, spec, parse_kv(a.test_where))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
print(f"task {a.task}: CV {len(cv.y):,} cells ({cv.y.mean():.1%} positive, {len(set(cv.groups))} patients); "
      f"test {len(te.y):,} cells ({len(set(te.groups))} patients)", flush=True)


class AttnPool(nn.Module):
    """Gated attention pooling over spatial positions (Ilse et al. 2018), replacing global average pooling:
    each of the 7x7 positions gets a learned weight, so the classifier can focus on the nucleus or granules."""

    def __init__(self, c=1280, h=128):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(c, h), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(c, h), nn.Sigmoid())
        self.w = nn.Linear(h, 1)

    def forward(self, f):                      # f: [B, C, H, W]
        t = f.flatten(2).transpose(1, 2)       # [B, HW, C]
        a = torch.softmax(self.w(self.v(t) * self.u(t)), dim=1)
        return (a * t).sum(1)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        m = backbone()
        self.body = nn.Sequential(*m.blocks[5:], m.conv_head, m.bn2)
        self.pool = AttnPool() if a.pool == "attn" else None
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, 1)

    def forward(self, x):
        f = self.body(x)
        h = self.pool(f) if self.pool is not None else f.mean((2, 3))
        return self.fc(self.drop(h)).squeeze(-1)


def augment(x):
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))


@torch.no_grad()
def predict(model, rows, bs=512):
    model.eval()
    ps = []
    for i in range(0, len(rows), bs):
        x = torch.from_numpy(F[rows[i:i + bs]].astype(np.float32))
        ps.append(torch.sigmoid(model(x)).numpy())
    p = np.concatenate(ps) if ps else np.zeros(0)
    return np.where(OK[rows], p, np.nan)


def train(rows, y, val_rows, val_y, tag):
    rng = np.random.default_rng(a.seed)
    model = Head()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(rows) / a.batch_size)
    total, warm = a.epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    pos_w = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best, best_state, hist = np.inf, None, []
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(len(rows)); tl, t0, lrs = [], time.time(), []
        for b in range(steps_per_epoch):
            idx = perm[b * a.batch_size:(b + 1) * a.batch_size]
            x = augment(torch.from_numpy(F[rows[idx]].astype(np.float32)))
            loss = lossf(model(x), torch.from_numpy(y[idx].astype(np.float32)))
            opt.zero_grad(); loss.backward(); opt.step(); lrs.append(opt.param_groups[0]["lr"]); sched.step()
            tl.append(loss.item())
        pv = predict(model, val_rows); m = ~np.isnan(pv)
        vl = float(nn.functional.binary_cross_entropy(torch.tensor(pv[m]), torch.tensor(val_y[m].astype(np.float32))))
        va = MT.point_metrics(val_y[m], pv[m], 0.5)["auroc"]
        hist.append(dict(fold=tag, epoch=ep + 1, lr=float(np.mean(lrs)), lr_end=lrs[-1], train_loss=float(np.mean(tl)), val_loss=vl, val_auroc=va))
        print(f"    {tag} epoch {ep + 1}: lr {lrs[-1]:.2e} train loss {np.mean(tl):.3f} | val loss {vl:.3f} AUROC {va:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


rows_cv, y_cv, g_cv = cv.index, cv.y, cv.groups
rows_te, y_te, g_te = te.index, te.y, te.groups
summary, preds, history = [], [], []
for split in a.splits:
    fold = make_folds(y_cv, g_cv, split, 5, a.seed)
    oof = np.full(len(y_cv), np.nan); ext = []
    for k in range(5):
        tr = np.flatnonzero(fold != k); ho = np.flatnonzero(fold == k)
        # inner validation for early stopping (same split kind as the outer split)
        rng = np.random.default_rng(100 + k)
        if split == "grouped":
            ug = np.unique(g_cv[tr]); vg = rng.choice(ug, max(2, int(0.15 * len(ug))), replace=False)
            inner_v = tr[np.isin(g_cv[tr], vg)]; inner_t = tr[~np.isin(g_cv[tr], vg)]
        else:
            p = rng.permutation(tr); nv = int(0.15 * len(tr)); inner_v, inner_t = p[:nv], p[nv:]
        inner_t = inner_t[OK[rows_cv[inner_t]]]
        if a.max_train and len(inner_t) > a.max_train:
            inner_t = np.sort(rng.choice(inner_t, a.max_train, replace=False))
        t0 = time.time()
        model, h = train(rows_cv[inner_t], y_cv[inner_t], rows_cv[inner_v], y_cv[inner_v], f"{split}-f{k}")
        for r in h:
            r["split"] = split
        history += h
        oof[ho] = predict(model, rows_cv[ho])
        ext.append(predict(model, rows_te))
        print(f"  {split} fold {k}: held-out AUROC {MT.point_metrics(y_cv[ho][~np.isnan(oof[ho])], oof[ho][~np.isnan(oof[ho])], 0.5)['auroc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pe = np.nanmean(np.vstack(ext), 0)
    mcv, mte = ~np.isnan(oof), ~np.isnan(pe)
    thr = MT.threshold_for_sensitivity(y_cv[mcv], oof[mcv], 0.95)
    cvm = MT.point_metrics(y_cv[mcv], oof[mcv], thr)
    r = MT.bootstrap(y_te[mte], pe[mte], g_te[mte], thr, a.n_boot)
    tm = MT.point_metrics(y_te[mte], pe[mte], thr)
    summary.append(dict(level="cell", split=split, cv_auroc=cvm["auroc"], ext_auroc=tm["auroc"], ext_lo=r["auroc_lo"], ext_hi=r["auroc_hi"],
                        **{f"ext_{k}": tm[k] for k in ["sens", "spec", "f1", "npv", "acc", "ece", "fn", "fp"]}))
    preds += [dict(split=split, set="cv", row=i, patient_id=g, y=yy, p=pp) for i, g, yy, pp in zip(rows_cv, g_cv, y_cv, oof)]
    preds += [dict(split=split, set="test", row=i, patient_id=g, y=yy, p=pp) for i, g, yy, pp in zip(rows_te, g_te, y_te, pe)]
    print(f"== {split}: CV AUROC {cvm['auroc']:.3f} | external AUROC {tm['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] "
          f"recall {tm['sens']:.3f} spec {tm['spec']:.3f} F1 {tm['f1']:.3f}", flush=True)
    # patient level (T1 -> T3): mean cell probability per patient
    if a.task in ("apl_cell_diagnosis", "cnmc_all_vs_hem") and split == "grouped":
        P = lambda gg, pp, yy: pd.DataFrame({"g": gg, "p": pp, "y": yy}).groupby("g").agg(p=("p", "mean"), y=("y", "first"))
        pc, pt = P(g_cv, oof, y_cv), P(g_te, pe, y_te)
        thr_p = MT.threshold_for_sensitivity(pc.y.values, pc.p.values, 0.95)
        cvp = MT.point_metrics(pc.y.values, pc.p.values, thr_p)
        rp = MT.bootstrap(pt.y.values, pt.p.values, pt.index.values, thr_p, a.n_boot)
        tp = MT.point_metrics(pt.y.values, pt.p.values, thr_p)
        summary.append(dict(level="patient", split=split, cv_auroc=cvp["auroc"], ext_auroc=tp["auroc"], ext_lo=rp["auroc_lo"], ext_hi=rp["auroc_hi"],
                            **{f"ext_{k}": tp[k] for k in ["sens", "spec", "f1", "npv", "acc", "ece", "fn", "fp"]}))
        pt.reset_index().rename(columns={"g": "patient_id"}).to_csv(out / "patient_predictions_test.csv", index=False)
        pc.reset_index().rename(columns={"g": "patient_id"}).to_csv(out / "patient_predictions_cv.csv", index=False)
        print(f"== patient (mean cell prob): CV AUROC {cvp['auroc']:.3f} | external AUROC {tp['auroc']:.3f} [{rp['auroc_lo']:.3f}, {rp['auroc_hi']:.3f}] "
              f"recall {tp['sens']:.2f} ({int(tp['fn'])} missed) spec {tp['spec']:.2f} F1 {tp['f1']:.2f}", flush=True)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "predictions.csv.gz", index=False)
json.dump(vars(a), open(out / "config.json", "w"), indent=2)
print(f"wrote {out}")
