"""Feature-engineering experiments for patient-level APL detection (T3).

Every transformation (correlation filter, PCA, L1 selection, cell-type classifier) is fitted on the
training patients of each Discovery fold only. A configuration is chosen by Discovery CV AUROC
(pre-declared rule); the Validation cohort is scored once for every configuration and reported.

Configurations
  eng_raw          engineered features, pooled (mean, SD, P10/50/90) + LR          [reference]
  eng_corrfilter   + drop one of each pair with |r| > 0.9 (fit on training cells)
  eng_slidenorm    + stain/colour features standardised within each slide (patient)
  eng_l1           + L1-regularised LR on pooled features (embedded selection)
  eng_stable       + correlation filter + slide normalisation
  eff_pca{16,32,64} EfficientNet-B0 embeddings reduced by PCA (fit on training cells), pooled + LR
  eff_raw          EfficientNet-B0, no reduction                                    [reference]
  fusion_pca32     engineered (filtered) + EfficientNet PCA-32
  composition      digital differential: fractions of 8 predicted cell classes from a cell-type
                   classifier trained on labelled training cells (out-of-fold within training patients)
  composition+eng  composition + engineered (filtered) pooled features
  mil_eng_stable   attention MIL on engineered (filtered + slide-normalised) cells
  mil_eff_pca32    attention MIL on EfficientNet PCA-32 cells

  python scripts/run_feature_engineering.py --manifest manifests/apl_aml.csv --hand features/apl_aml_hand.parquet \
     --effnet features/apl_aml_effnet.npy --out results/feature_engineering
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench import metrics as MT
from leukbench.mil import train_abmil
from leukbench.models import load_feature_block
from leukbench.splits import patient_folds

STAIN = r"_(H|E|L|a|b|sat)_(mean|std)$|^H_ratio|^L_diff"
CELL_CLASSES = {"Blast, no lineage spec": "blast", "Promyelocyte": "promyelocyte", "Myelocyte": "myelo", "Metamyelocyte": "myelo",
                "Monocyte": "mono", "Promonocyte": "mono", "Lymphocyte": "lympho", "Lymphocyte, variant": "lympho",
                "Prolymphocyte": "lympho", "Plasma cells": "other", "Segmented neutrophils": "neutro", "Band neutrophils": "neutro",
                "Erythroblast": "erythro", "Eosinophils": "other", "Basophil": "other"}

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--hand", required=True)
ap.add_argument("--effnet", required=True)
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--configs", nargs="*")
ap.add_argument("--max-cells", type=int, default=500)
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(1)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

man = pd.read_csv(a.manifest, keep_default_na=False)
Hdf = pd.read_parquet(a.hand) if a.hand.endswith(".parquet") else pd.read_csv(a.hand)
H = Hdf.to_numpy("float32"); E = load_feature_block(a.effnet)
stain_cols = np.array([bool(re.search(STAIN, c)) for c in Hdf.columns])
man["y"] = (man.diagnosis == "APL").astype(int)
ctype = man.cell_type.map(CELL_CLASSES)
pt = man.groupby("patient_id").agg(y=("y", "first"), cohort=("cohort", "first")).reset_index()
disc, val = pt[pt.cohort == "Discovery"].reset_index(drop=True), pt[pt.cohort == "Validation"].reset_index(drop=True)
rng0 = np.random.default_rng(0)
rows_of = {}
for pid, g in man.groupby("patient_id"):
    idx = g.index.to_numpy()
    rows_of[pid] = np.sort(rng0.choice(idx, a.max_cells, replace=False)) if len(idx) > a.max_cells else idx


# ---------------------------------------------------------------- cell-level transforms (fit on training rows only)
class CellTransform:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, rows):
        k = self.kind
        if k.startswith("eng"):
            X = self._eng(rows)
            self.med = np.nanmedian(X, 0)
            if "corr" in k or "stable" in k:
                Xf = np.where(np.isnan(X), self.med, X)
                C = np.abs(np.corrcoef(Xf[np.random.default_rng(0).choice(len(Xf), min(20000, len(Xf)), replace=False)].T))
                keep = np.ones(C.shape[0], bool)
                for i in range(C.shape[0]):
                    if keep[i]:
                        keep[(C[i] > 0.9) & (np.arange(C.shape[0]) > i)] = False
                self.keep = keep
            else:
                self.keep = np.ones(X.shape[1], bool)
        if k.startswith("eff") or k.startswith("fusion"):
            n = int(re.search(r"pca(\d+)", k).group(1)) if "pca" in k else 0
            if n:
                Xe = np.nan_to_num(E[rows], nan=0.0)
                self.sc = StandardScaler().fit(Xe)
                self.pca = PCA(n, random_state=0).fit(self.sc.transform(Xe))
            else:
                self.pca = None
            if k.startswith("fusion"):
                self.eng = CellTransform("eng_corrfilter").fit(rows)
        return self

    def _eng(self, rows):
        X = H[rows].copy()
        if "slidenorm" in self.kind or "stable" in self.kind:          # stain features standardised within each slide
            pids = man.loc[rows, "patient_id"].values
            for pid in np.unique(pids):
                m = pids == pid
                blk = X[np.ix_(m, stain_cols)]
                X[np.ix_(m, stain_cols)] = (blk - np.nanmedian(blk, 0)) / (np.nanstd(blk, 0) + 1e-6)
        return X

    def transform(self, rows):
        k = self.kind
        if k.startswith("eng"):
            X = self._eng(rows)
            return np.where(np.isnan(X), self.med, X)[:, self.keep]
        Xe = np.nan_to_num(E[rows], nan=0.0)
        Z = self.pca.transform(self.sc.transform(Xe)) if self.pca is not None else Xe
        if k.startswith("fusion"):
            return np.hstack([self.eng.transform(rows), Z])
        return Z


def pooled(X):
    return np.concatenate([np.nanmean(X, 0), np.nanstd(X, 0), *np.nanpercentile(X, [10, 50, 90], axis=0)])


def lr(penalty="l2", C=0.1):
    solver = "saga" if penalty == "l1" else "lbfgs"
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("c", LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=5000, class_weight="balanced"))])


# ---------------------------------------------------------------- composition ("digital differential")
COMP_CLASSES = sorted(set(CELL_CLASSES.values()))


def composition_model(train_pids, seed):
    """Cell-type classifier on labelled cells of training patients; composition of each training
    patient comes from an inner 3-fold patient-grouped CV (out-of-fold), test patients from the full model."""
    rows = np.concatenate([rows_of_all[p] for p in train_pids])
    lab = ctype.loc[rows]
    rows_l, lab = rows[lab.notna().values], lab.dropna().values
    feat = CellTransform("fusion_pca32").fit(rows_l)

    def fit(rr, yy):
        return Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))]).fit(feat.transform(rr), yy)

    def comp(model, pids):
        res = []
        for p in pids:
            pr = model.predict_proba(feat.transform(rows_of[p]))
            frac = np.bincount(pr.argmax(1), minlength=len(model.classes_)) / len(pr)
            res.append(np.concatenate([frac, pr.mean(0)]))
        return np.asarray(res)

    comp_train = np.zeros((len(train_pids), 2 * len(COMP_CLASSES)))
    pid_of_row = man.loc[rows_l, "patient_id"].values
    tp = np.asarray(train_pids)
    for tr_i, te_i in GroupKFold(3).split(tp, groups=tp):
        m = np.isin(pid_of_row, tp[tr_i])
        mdl = fit(rows_l[m], lab[m])
        comp_train[te_i] = comp(mdl, tp[te_i])
    full = fit(rows_l, lab)
    return comp_train, (lambda pids: comp(full, pids))


rows_all = {}
for pid, g in man.groupby("patient_id"):
    rows_all[pid] = g.index.to_numpy()
rows_of_all = rows_all


# ---------------------------------------------------------------- one configuration, CV + external
def run_config(cfg, seed):
    fold = patient_folds(disc.y.values, 5, seed)
    oof, ext = np.zeros(len(disc)), []
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tr_p, te_p, va_p = disc.patient_id.values[tr], disc.patient_id.values[te], val.patient_id.values
        ytr = disc.y.values[tr]
        if cfg.startswith("mil_"):
            kind = "eng_stable" if "eng" in cfg else "eff_pca32"
            T = CellTransform(kind).fit(np.concatenate([rows_of[p] for p in tr_p]))
            f = train_abmil([T.transform(rows_of[p]) for p in tr_p], ytr, seed=seed, epochs=40)
            oof[te] = f([T.transform(rows_of[p]) for p in te_p])
            ext.append(f([T.transform(rows_of[p]) for p in va_p]))
            continue
        if cfg.startswith("composition"):
            Ctr, comp_fn = composition_model(tr_p, seed)
            Xtr, Xte, Xva = Ctr, comp_fn(te_p), comp_fn(va_p)
            if cfg == "composition+eng":
                T = CellTransform("eng_corrfilter").fit(np.concatenate([rows_of[p] for p in tr_p]))
                P_ = lambda ps: np.asarray([pooled(T.transform(rows_of[p])) for p in ps])
                Xtr, Xte, Xva = np.hstack([Xtr, P_(tr_p)]), np.hstack([Xte, P_(te_p)]), np.hstack([Xva, P_(va_p)])
        else:
            kind = {"eng_l1": "eng_raw"}.get(cfg, cfg)
            T = CellTransform(kind).fit(np.concatenate([rows_of[p] for p in tr_p]))
            P_ = lambda ps: np.asarray([pooled(T.transform(rows_of[p])) for p in ps])
            Xtr, Xte, Xva = P_(tr_p), P_(te_p), P_(va_p)
        clf = (lr("l1", 0.5) if cfg == "eng_l1" else lr()).fit(Xtr, ytr)
        oof[te] = clf.predict_proba(Xte)[:, 1]
        ext.append(clf.predict_proba(Xva)[:, 1])
    return oof, np.mean(ext, axis=0)


CONFIGS = a.configs or ["eng_raw", "eng_corrfilter", "eng_slidenorm", "eng_l1", "eng_stable", "eff_raw", "eff_pca16", "eff_pca32",
                        "eff_pca64", "fusion_pca32", "composition", "composition+eng", "mil_eng_stable", "mil_eff_pca32"]
rows, preds = [], []
for cfg in CONFIGS:
    t0 = time.time()
    oofs, exts = [], []
    for seed in a.seeds:
        o, e = run_config(cfg, seed)
        oofs.append(o); exts.append(e)
    cv_aucs = [MT.point_metrics(disc.y.values, o, 0.5)["auroc"] for o in oofs]
    oof_m, ext_m = np.mean(oofs, 0), np.mean(exts, 0)
    rec = dict(config=cfg, cv_auroc=float(np.mean(cv_aucs)), cv_sd=float(np.std(cv_aucs)))
    for name, thr in [("sens95", MT.threshold_for_sensitivity(disc.y.values, oof_m, 0.95)), ("youden", MT.youden_threshold(disc.y.values, oof_m))]:
        r = MT.bootstrap(val.y.values, ext_m, val.patient_id.values, thr, a.n_boot)
        if name == "sens95":
            rec.update(ext_auroc=r["auroc"], ext_lo=r["auroc_lo"], ext_hi=r["auroc_hi"])
        rec.update({f"{name}_{k}": r[k] for k in ["sens", "spec", "f1", "npv", "acc", "fn", "fp"]})
    # post-hoc reference only (NOT deployable): best specificity at 100% recall if the threshold were re-set on site
    rec["oracle_spec_at_recall1"] = MT.point_metrics(val.y.values, ext_m, ext_m[val.y.values == 1].min() - 1e-9)["spec"]
    rows.append(rec)
    preds.append(pd.DataFrame(dict(config=cfg, patient_id=val.patient_id, y=val.y, p=ext_m)))
    print(f"  {cfg:16s} CV AUROC {rec['cv_auroc']:.3f}±{rec['cv_sd']:.3f} | external AUROC {rec['ext_auroc']:.3f} "
          f"[{rec['ext_lo']:.3f}, {rec['ext_hi']:.3f}] | sens95: recall {rec['sens95_sens']:.2f} spec {rec['sens95_spec']:.2f} | "
          f"youden: recall {rec['youden_sens']:.2f} spec {rec['youden_spec']:.2f} | oracle spec@recall1 {rec['oracle_spec_at_recall1']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(out / "feature_engineering.csv", index=False)
    pd.concat(preds).to_csv(out / "predictions.csv", index=False)
best = max(rows, key=lambda r: r["cv_auroc"])
print(f"selected by Discovery CV: {best['config']} (CV {best['cv_auroc']:.3f}) -> external AUROC {best['ext_auroc']:.3f}")
print(f"wrote {out}")
