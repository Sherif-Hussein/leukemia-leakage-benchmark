"""Scan-time confound analysis (APL-AML).

No scan dates are distributed with the data. The CellaVision analyzer numbers signed (classified) cells
sequentially in its database, so the median cell ID of a patient's signed cells is used as a time proxy.
Its validity is checked against the known temporal split (Validation patients were diagnosed later).

Analyses (Discovery cohort, patient level, pooled-statistics LR, 5-fold patient CV x 3 seeds):
  1. proxy validity: AUROC of the proxy for Validation vs Discovery
  2. confound strength: AUROC of the proxy for APL vs AML within Discovery
  3. colour -> time: among AML patients only, AUROC of stain/colour features (and others) for predicting
     early vs late scan (median split of the proxy), patient-level CV
  4. time-restricted CV: Discovery CV AUROC on all patients vs only patients inside the time window where
     both diagnoses occur (proxy between the 5th and 95th percentile of APL patients)
  5. adjustment: logistic regression of APL on the out-of-fold model score and the proxy rank; reports
     whether the score remains predictive (likelihood-ratio test) and the score's AUROC within proxy strata

  python scripts/run_date_confound.py --manifest manifests/apl_aml.csv --hand features/apl_aml_hand.parquet \
      --effnet features/apl_aml_effnet.npy [--hand-macenko features/apl_aml_hand_macenko.parquet] --out results/date_confound
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench.models import load_feature_block

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--hand", required=True)
ap.add_argument("--effnet", required=True)
ap.add_argument("--hand-macenko")
ap.add_argument("--max-cells", type=int, default=500)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
STAIN = r"_(H|E|L|a|b|sat)_(mean|std)$|^H_ratio|^L_diff"

man = pd.read_csv(a.manifest, keep_default_na=False)
Hdf = pd.read_parquet(a.hand); H = Hdf.to_numpy("float32"); E = np.nan_to_num(load_feature_block(a.effnet), nan=0.0)
stain = np.array([bool(re.search(STAIN, c)) for c in Hdf.columns])
HM = pd.read_parquet(a.hand_macenko).to_numpy("float32") if a.hand_macenko else None
man["sid"] = man.path.str.extract(r"/Signed slides/[^/]+/[A-Za-z]+_(\d{6,})\.jpg$")[0].astype(float)
pt = man.groupby("patient_id").agg(proxy=("sid", "median"), cohort=("cohort", "first"), dx=("diagnosis", "first")).reset_index()
pt["y"] = (pt.dx == "APL").astype(int)
rng = np.random.default_rng(0)
rows_of = {p: (np.sort(rng.choice(g.index, a.max_cells, replace=False)) if len(g) > a.max_cells else g.index.to_numpy()) for p, g in man.groupby("patient_id")}
res = {}

# 1-2
res["proxy_auroc_validation_vs_discovery"] = roc_auc_score(pt.cohort == "Validation", pt.proxy)
D = pt[pt.cohort == "Discovery"].reset_index(drop=True)
res["proxy_auroc_apl_within_discovery"] = roc_auc_score(D.y, D.proxy)
V = pt[pt.cohort == "Validation"].reset_index(drop=True)
res["proxy_auroc_apl_within_validation"] = roc_auc_score(V.y, V.proxy)


def pooled(X):
    X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
    return np.concatenate([np.nanmean(X, 0), np.nanstd(X, 0), *np.nanpercentile(X, [10, 50, 90], axis=0)])


def patient_matrix(F, pids, cols=None):
    return np.nan_to_num(np.stack([pooled(F[rows_of[p]] if cols is None else F[rows_of[p]][:, cols]) for p in pids]), nan=0.0)


clf = lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))])


def cv_scores(X, y, seeds=(0, 1, 2)):
    oof = np.zeros((len(seeds), len(y)))
    for i, s in enumerate(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            oof[i, te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof.mean(0), [roc_auc_score(y, o) for o in oof]


# 3. colour -> time among AML Discovery patients
A = D[D.dx == "AML"].reset_index(drop=True)
late = (A.proxy > A.proxy.median()).astype(int).values
reps = {"stain_colour": (H, stain), "shape_texture": (H, ~stain), "engineered_all": (H, None), "efficientnet": (E, None)}
if HM is not None:
    reps["stain_colour_macenko_images"] = (HM, stain)
for nm, (F, cols) in reps.items():
    X = patient_matrix(F, A.patient_id, np.flatnonzero(cols) if cols is not None else None)
    _, aucs = cv_scores(X, late)
    res[f"time_from_{nm}_auroc_AML_only"] = float(np.mean(aucs))

# 4-5. diagnosis models: full Discovery vs time-restricted window; adjustment for the proxy
lo, hi = np.percentile(D.proxy[D.y == 1], [5, 95])
win = ((D.proxy >= lo) & (D.proxy <= hi)).values
res["window_patients"] = int(win.sum()); res["window_apl"] = int(D.y[win].sum())
res["proxy_auroc_apl_within_window"] = roc_auc_score(D.y[win], D.proxy[win])
models = {"engineered": (H, None), "stain_colour_only": (H, stain), "shape_texture_only": (H, ~stain), "efficientnet": (E, None)}
if HM is not None:
    models["engineered_macenko_images"] = (HM, None)
rank = stats.rankdata(D.proxy) / len(D)
for nm, (F, cols) in models.items():
    c = np.flatnonzero(cols) if cols is not None else None
    X = patient_matrix(F, D.patient_id, c)
    oof, aucs = cv_scores(X, D.y.values)
    res[f"{nm}_cv_auroc_full"] = float(np.mean(aucs))
    oofw, aucsw = cv_scores(X[win], D.y.values[win])
    res[f"{nm}_cv_auroc_time_window"] = float(np.mean(aucsw))
    # adjustment: does the score add to the time proxy?
    z = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / np.clip(1 - oof, 1e-6, 1))
    m0 = LogisticRegression(C=1e6, max_iter=2000).fit(rank.reshape(-1, 1), D.y)
    m1 = LogisticRegression(C=1e6, max_iter=2000).fit(np.c_[rank, z], D.y)
    ll = lambda m, X: np.sum(D.y * np.log(m.predict_proba(X)[:, 1] + 1e-12) + (1 - D.y) * np.log(m.predict_proba(X)[:, 0] + 1e-12))
    lr = 2 * (ll(m1, np.c_[rank, z]) - ll(m0, rank.reshape(-1, 1)))
    res[f"{nm}_score_adds_to_time_LR_p"] = float(stats.chi2.sf(lr, 1))
    res[f"{nm}_score_vs_time_spearman"] = float(stats.spearmanr(oof, D.proxy).correlation)
R = pd.Series(res); R.to_csv(out / "date_confound.csv")
pd.set_option("display.width", 200)
print(R.round(4).to_string())
