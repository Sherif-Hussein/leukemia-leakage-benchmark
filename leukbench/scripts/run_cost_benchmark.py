"""Computational cost with and without PCA (APL-AML, one training fold of Discovery, 1 CPU thread).

For the unreduced EfficientNet-B0 embedding (1,280-d) and PCA with k components, measures:
  * PCA fitting time (training cells) and projection time per patient
  * pooled-statistics LR: training time, prediction time per patient, number of model parameters
  * attention MIL (40 epochs): training time, prediction time per patient, number of parameters
  * peak memory (resident set size increase) during training, and storage per cell (float32 bytes)
Each timing is the median of --repeats runs.

  python scripts/run_cost_benchmark.py --manifest manifests/apl_aml.csv --effnet features/apl_aml_effnet.npy --out results/cost
"""
import argparse
import gc
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench.mil import train_abmil
from leukbench.models import load_feature_block
from leukbench.splits import patient_folds

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--effnet", required=True)
ap.add_argument("--ks", nargs="+", type=int, default=[0, 64, 32, 16])
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(1)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

man = pd.read_csv(a.manifest, keep_default_na=False)
E = np.nan_to_num(load_feature_block(a.effnet), nan=0.0)
man["y"] = (man.diagnosis == "APL").astype(int)
pt = man[man.cohort == "Discovery"].groupby("patient_id").y.first().reset_index()
rng = np.random.default_rng(0)
rows_of = {p: (np.sort(rng.choice(g.index, 500, replace=False)) if len(g) > 500 else g.index.to_numpy()) for p, g in man.groupby("patient_id")}
fold = patient_folds(pt.y.values, 5, 0)
tr_p, te_p = pt.patient_id.values[fold != 0], pt.patient_id.values[fold == 0]
ytr = pt.y.values[fold != 0]
pooled = lambda X: np.concatenate([X.mean(0), X.std(0), *np.percentile(X, [10, 50, 90], axis=0)])
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB (peak so far)


def timed(f):
    ts, r = [], None
    for _ in range(a.repeats):
        gc.collect(); t0 = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), r


res = []
for k in a.ks:
    cells = np.concatenate([rows_of[p] for p in tr_p])
    if k:
        t_pca, (sc, pca) = timed(lambda: (lambda s: (s, PCA(k, random_state=0).fit(s.transform(E[cells]))))(StandardScaler().fit(E[cells])))
        tf = lambda rows: pca.transform(sc.transform(E[rows])).astype("float32")
        t_proj, _ = timed(lambda: [tf(rows_of[p]) for p in te_p])
        t_proj /= len(te_p)
    else:
        t_pca, t_proj, tf = 0.0, 0.0, (lambda rows: E[rows])
    Xtr = np.stack([pooled(tf(rows_of[p])) for p in tr_p]); Xte = np.stack([pooled(tf(rows_of[p])) for p in te_p])
    m0 = rss()
    t_lr, clf = timed(lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))]).fit(Xtr, ytr))
    t_lr_pred, _ = timed(lambda: clf.predict_proba(Xte)); t_lr_pred /= len(te_p)
    bags_tr = [tf(rows_of[p]) for p in tr_p]; bags_te = [tf(rows_of[p]) for p in te_p]
    t_mil, f = timed(lambda: train_abmil(bags_tr, ytr, seed=0, epochs=40))
    t_mil_pred, _ = timed(lambda: f(bags_te)); t_mil_pred /= len(te_p)
    d = 1280 if k == 0 else k
    mil_params = d * 128 + 128 + 2 * (128 * 64 + 64) + 64 + 1 + 128 + 1
    res.append(dict(representation="EfficientNet-B0, no PCA" if k == 0 else f"PCA-{k}", dims=d, pooled_inputs=5 * d,
                    storage_bytes_per_cell=4 * d, pca_fit_s=t_pca, pca_projection_ms_per_patient=1000 * t_proj,
                    lr_train_s=t_lr, lr_predict_ms_per_patient=1000 * t_lr_pred, lr_params=5 * d + 1,
                    mil_train_s=t_mil, mil_predict_ms_per_patient=1000 * t_mil_pred, mil_params=mil_params,
                    peak_rss_mb=rss(), train_patients=len(tr_p)))
    print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in res[-1].items()}, flush=True)
    pd.DataFrame(res).to_csv(out / "cost.csv", index=False)
print(f"wrote {out}")
