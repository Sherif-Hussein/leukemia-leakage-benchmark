"""Macenko stain normalization of a whole dataset (writes PNGs + a new manifest with the same rows).

Reference stain (stain vectors, 99th-percentile concentrations, background level) is estimated from
300 random development images only. Mode 'patient': stain vectors of each patient are estimated once
from up to 20 of the patient's cells (stain varies between slides, not between cells of a slide);
mode 'cell': estimated separately for every image. Background (illumination) per channel is the 99th
percentile of each image, because smear backgrounds are not pure white.

  python scripts/normalize_images.py --manifest manifests/apl_aml.csv --mode patient --out-dir data/apl_macenko \
      --out-manifest manifests/apl_aml_macenko.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--mode", choices=["patient", "cell"], default="patient")
ap.add_argument("--dev-col", default="cohort"); ap.add_argument("--dev", default="Discovery")
ap.add_argument("--ref-manifest", help="manifest from which the reference stain is estimated (default: --manifest)")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--out-manifest", required=True)
a = ap.parse_args()
BETA, ALPHA = 0.15, 1.0
man = pd.read_csv(a.manifest, keep_default_na=False)
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)


def load(p):
    try:
        return np.asarray(Image.open(p).convert("RGB"))
    except Exception:
        return None


def bg(im):
    return np.percentile(im.reshape(-1, 3), 99, axis=0).astype(np.float64) + 1


def od(im, io=None):
    io = bg(im) if io is None else io
    return -np.log(np.clip((im.reshape(-1, 3).astype(np.float64) + 1) / io, 1e-6, 1))


def fit(o):
    o = o[(o > BETA).all(1)]
    if len(o) < 50:
        return None
    _, V = np.linalg.eigh(np.cov(o.T)); V = V[:, [1, 2]]
    th = o @ V; phi = np.arctan2(th[:, 1], th[:, 0])
    lo, hi = np.percentile(phi, ALPHA), np.percentile(phi, 100 - ALPHA)
    v1 = V @ np.array([np.cos(lo), np.sin(lo)]); v2 = V @ np.array([np.cos(hi), np.sin(hi)])
    HE = np.array([v1, v2]) if v1[0] > v2[0] else np.array([v2, v1])
    HE = HE / np.linalg.norm(HE, axis=1, keepdims=True)
    C = np.linalg.lstsq(HE.T, o.T, rcond=None)[0]
    return HE, np.percentile(C, 99, axis=1)


rng = np.random.default_rng(0)
rman = pd.read_csv(a.ref_manifest, keep_default_na=False) if a.ref_manifest else man
ref = rman[rman[a.dev_col] == a.dev].path.sample(300, random_state=0)   # same reference as the pilot (Discovery images only)
ref_ims = [im for im in map(load, ref) if im is not None]
REF_HE, REF_MAXC = fit(np.concatenate([od(im)[::7].astype(np.float32) for im in ref_ims]).astype(np.float64))
REF_IO = np.median([bg(im) for im in ref_ims[:100]], axis=0)
del ref_ims


def normalize(im, params):
    o = od(im)
    p = params if params is not None else fit(o)
    if p is None:
        return im
    HE, maxc = p
    C = np.linalg.lstsq(HE.T, o.T, rcond=None)[0] * (REF_MAXC / (maxc + 1e-6))[:, None]
    return np.clip((REF_IO[:, None] * np.exp(-(REF_HE.T @ C))).T.reshape(im.shape), 0, 255).astype(np.uint8)


paths, t0, n_fail = [], time.time(), 0
for pid, g in man.groupby("patient_id", sort=False):
    params = None
    if a.mode == "patient":
        sample = [im for im in map(load, g.path.iloc[:20]) if im is not None]
        params = fit(np.concatenate([od(im)[::3] for im in sample])) if sample else None
    for i, p in zip(g.index, g.path):
        q = out / f"{i:06d}.png"
        im = load(p)
        if im is None:
            n_fail += 1
        else:
            Image.fromarray(normalize(im, params)).save(q)
        paths.append((i, str(q)))
    if len(paths) % 2000 < len(g):
        print(f"  {len(paths):,}/{len(man):,} ({(time.time() - t0) / 60:.1f} min)", flush=True)
new = man.copy()
new.loc[[i for i, _ in paths], "path"] = [q for _, q in paths]
new.to_csv(a.out_manifest, index=False)
print(f"normalized {len(man):,} images ({n_fail} unreadable kept as missing) in {(time.time() - t0) / 60:.1f} min; "
      f"{(time.time() - t0) / man.patient_id.nunique():.1f} s per patient")
