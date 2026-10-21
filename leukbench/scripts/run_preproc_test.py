"""Fast test of image preprocessing before feature extraction (APL-AML).

Variants: none | unsharp (edge sharpening) | clahe (contrast-limited adaptive histogram equalization on
lightness) | reinhard (colour normalization to reference LAB statistics computed from Discovery images).
For a fixed subset (up to --cells-per-patient random cells per patient, all patients), each variant is
passed through the frozen EfficientNet-B0; then, with identical models and splits:
  * cohort identifiability: AUROC for telling Validation from Discovery AML cells (grouped 5-fold CV)
  * cell level (T1): LR, patient-grouped 5-fold CV on Discovery (per-fold AUROC) and external AUROC
  * patient level (T3): pooled statistics + LR, 5-fold CV (per-fold AUROC) and external AUROC

  python scripts/run_preproc_test.py --manifest manifests/apl_aml.csv --weights weights/efficientnet_b0_ra-3dd342df.pth --out results/preproc_test
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from skimage import color, exposure
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench.encoders import IMAGENET_MEAN, IMAGENET_STD
from leukbench.splits import patient_folds

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--variants", nargs="+", default=["none", "unsharp", "clahe", "reinhard"])
ap.add_argument("--cells-per-patient", type=int, default=60)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--positive", default="APL"); ap.add_argument("--negative", default="AML")
ap.add_argument("--cohort-col", default="cohort"); ap.add_argument("--dev", default="Discovery"); ap.add_argument("--ext", default="Validation")
ap.add_argument("--black-bg-to-white", action="store_true", help="C-NMC: set black background to white before preprocessing")
ap.add_argument("--save-images", help="also write the preprocessed subset images (PNG) to this folder, and a manifest")
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

man = pd.read_csv(a.manifest, keep_default_na=False)
man["cohort"] = man[a.cohort_col].map({a.dev: "Discovery", a.ext: "Validation"})
man["diagnosis"] = man["diagnosis"].map({a.positive: "APL", a.negative: "AML"})
rng = np.random.default_rng(0)
idx = np.sort(np.concatenate([rng.choice(g.index.to_numpy(), min(a.cells_per_patient, len(g)), replace=False)
                              for _, g in man.groupby("patient_id")]))
sub = man.loc[idx].reset_index(drop=True)
print(f"subset: {len(sub):,} cells from {sub.patient_id.nunique()} patients", flush=True)


def load(p):
    try:
        im = np.asarray(Image.open(p).convert("RGB"))
        if a.black_bg_to_white:
            im = im.copy(); im[im.max(2) < 15] = 255
        return im
    except Exception:
        return None


# Reinhard reference: mean/SD of LAB over 300 random Discovery images
ref_paths = sub[sub.cohort == "Discovery"].path.sample(300, random_state=0)
_st = []
for im in map(load, ref_paths):          # keep only per-image statistics (low memory)
    if im is not None:
        l = color.rgb2lab(im / 255.0).reshape(-1, 3); _st.append((l.mean(0), l.std(0))); del l
REF_MU, REF_SD = np.mean([m for m, _ in _st], 0), np.mean([sd for _, sd in _st], 0)


# ---- Macenko stain normalization (Macenko et al., ISBI 2009)
IO, ALPHA, BETA = 240.0, 1.0, 0.15


def _bg(im):
    """Per-image background (illumination) level per channel: 99th percentile, as recommended for smears
    whose background is not pure white."""
    return np.percentile(im.reshape(-1, 3), 99, axis=0).astype(np.float64) + 1


def _od(im, io=None):
    io = _bg(im) if io is None else io
    return -np.log(np.clip((im.reshape(-1, 3).astype(np.float64) + 1) / io, 1e-6, 1))


def macenko_fit(od):
    """Stain matrix (2 x 3, rows = stain vectors in OD space) and 99th-percentile concentrations."""
    od = od[(od > BETA).all(1)]
    if len(od) < 50:
        return None, None
    _, V = np.linalg.eigh(np.cov(od.T)); V = V[:, [1, 2]]
    th = od @ V; phi = np.arctan2(th[:, 1], th[:, 0])
    lo, hi = np.percentile(phi, ALPHA), np.percentile(phi, 100 - ALPHA)
    v1 = V @ np.array([np.cos(lo), np.sin(lo)]); v2 = V @ np.array([np.cos(hi), np.sin(hi)])
    HE = np.array([v1, v2]) if v1[0] > v2[0] else np.array([v2, v1])        # first row: hematoxylin-like (more red OD)
    HE = HE / np.linalg.norm(HE, axis=1, keepdims=True)
    C = np.linalg.lstsq(HE.T, od.T, rcond=None)[0]
    return HE, np.percentile(C, 99, axis=1)


_ref_od = np.concatenate([_od(im)[::7].astype(np.float32) for im in map(load, ref_paths) if im is not None]).astype(np.float64)
REF_HE, REF_MAXC = macenko_fit(_ref_od)
REF_IO = np.median([_bg(im) for im in map(load, ref_paths[:100]) if im is not None], axis=0)


def macenko(im):
    od = _od(im)
    HE, maxc = macenko_fit(od)
    if HE is None:
        return im
    C = np.linalg.lstsq(HE.T, od.T, rcond=None)[0]
    C = C * (REF_MAXC / (maxc + 1e-6))[:, None]
    out = REF_IO[:, None] * np.exp(-(REF_HE.T @ C))
    return np.clip(out.T.reshape(im.shape), 0, 255).astype(np.uint8)


def preprocess(im, v):
    if v == "none":
        return im
    if v == "unsharp":
        return np.asarray(Image.fromarray(im).filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2)))
    if v == "clahe":
        lab = color.rgb2lab(im / 255.0)
        lab[..., 0] = exposure.equalize_adapthist(lab[..., 0] / 100.0, clip_limit=0.01) * 100.0
        return (np.clip(color.lab2rgb(lab), 0, 1) * 255).astype(np.uint8)
    if v == "macenko":
        return macenko(im)
    if v == "reinhard":
        lab = color.rgb2lab(im / 255.0); mu, sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
        lab = (lab - mu) / sd * REF_SD + REF_MU
        return (np.clip(color.lab2rgb(lab), 0, 1) * 255).astype(np.uint8)
    raise ValueError(v)


import timm
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, pretrained_cfg_overlay=dict(file=a.weights)).eval()
MEAN, STD = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)


def embed(v):
    f = out / f"emb_{v}.npy"
    if f.exists():
        return np.load(f)
    Z, batch, t0 = np.full((len(sub), 1280), np.nan, np.float32), [], time.time()
    def flush(ix):
        x = torch.stack([b for _, b in batch])
        with torch.no_grad():
            Z[[i for i, _ in batch]] = model(x).numpy()
    for i, p in enumerate(sub.path):
        im = load(p)
        if im is not None:
            im = Image.fromarray(preprocess(im, v)).resize((224, 224), Image.BICUBIC)
            batch.append((i, (torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255 - MEAN) / STD))
        if len(batch) == 64 or (i == len(sub) - 1 and batch):
            flush(i); batch.clear()
    np.save(f, Z); print(f"  embedded {v}: {time.time() - t0:.0f}s", flush=True)
    if v != "none":
        ex = sub.path.iloc[[0, len(sub) // 2]]
        tiles = [np.asarray(Image.fromarray(preprocess(load(p), vv)).resize((160, 160))) for p in ex for vv in ["none", v]]
        Image.fromarray(np.hstack(tiles)).save(out / f"example_{v}.png")
    return Z


clf = lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=4000, class_weight="balanced"))])
y_cell = (sub.diagnosis == "APL").astype(int).values
disc, val = (sub.cohort == "Discovery").values, (sub.cohort == "Validation").values
pt = sub.groupby("patient_id").agg(y=("diagnosis", lambda s: int(s.iloc[0] == "APL")), cohort=("cohort", "first")).reset_index()
rows = []
for v in a.variants:
    Z = np.nan_to_num(embed(v), nan=0.0)
    # cohort identifiability (AML cells)
    aml = np.flatnonzero((sub.diagnosis == "AML").values)
    p = cross_val_predict(clf(), Z[aml], (sub.cohort.values[aml] == "Validation").astype(int), groups=sub.patient_id.values[aml],
                          cv=StratifiedGroupKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    cid = roc_auc_score((sub.cohort.values[aml] == "Validation").astype(int), p)
    # cell level: grouped CV on Discovery, per fold, + external
    di = np.flatnonzero(disc); folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(di, y_cell[di], sub.patient_id.values[di]))
    cell_f = [roc_auc_score(y_cell[di[te]], clf().fit(Z[di[tr]], y_cell[di[tr]]).predict_proba(Z[di[te]])[:, 1]) for tr, te in folds]
    cell_ext = roc_auc_score(y_cell[val], clf().fit(Z[disc], y_cell[disc]).predict_proba(Z[val])[:, 1])
    # patient level: pooled stats
    P = np.stack([np.concatenate([Z[g].mean(0), Z[g].std(0), *np.percentile(Z[g], [10, 50, 90], axis=0)])
                  for g in [np.flatnonzero(sub.patient_id.values == pid) for pid in pt.patient_id]])
    pd_, pv_ = (pt.cohort == "Discovery").values, (pt.cohort == "Validation").values
    yd = pt.y.values[pd_]; Pd = P[pd_]; fold = patient_folds(yd, 5, 0)
    pat_f = [roc_auc_score(yd[fold == k], clf().fit(Pd[fold != k], yd[fold != k]).predict_proba(Pd[fold == k])[:, 1]) for k in range(5)]
    pat_ext = roc_auc_score(pt.y.values[pv_], clf().fit(Pd, yd).predict_proba(P[pv_])[:, 1])
    rows.append(dict(variant=v, cohort_id_auroc=cid, cell_cv_mean=np.mean(cell_f), cell_cv_sd=np.std(cell_f), cell_cv_min=np.min(cell_f), cell_ext=cell_ext,
                     patient_cv_mean=np.mean(pat_f), patient_cv_sd=np.std(pat_f), patient_cv_min=np.min(pat_f), patient_ext=pat_ext,
                     **{f"cell_f{k}": x for k, x in enumerate(cell_f)}, **{f"pat_f{k}": x for k, x in enumerate(pat_f)}))
    print(f"  {v:9s} cohort-ID {cid:.3f} | cell CV {np.mean(cell_f):.3f}±{np.std(cell_f):.3f} (min {np.min(cell_f):.3f}) ext {cell_ext:.3f} | "
          f"patient CV {np.mean(pat_f):.3f}±{np.std(pat_f):.3f} (min {np.min(pat_f):.3f}) ext {pat_ext:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out / "preproc_test.csv", index=False)
print(f"wrote {out}")
