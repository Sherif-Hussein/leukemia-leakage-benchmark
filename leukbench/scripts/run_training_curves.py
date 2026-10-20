"""Training dynamics on the APL-AML Discovery cohort (diagnostics for the paper; nothing here is
used to select models or thresholds).

 1. ABMIL loss / AUROC per epoch      : 5 patient-grouped Discovery folds; training loss (class-weighted
                                         BCE, mean over bags) and held-out-fold loss/AUROC/accuracy after every
                                         epoch; inputs = engineered features, raw EfficientNet (1,280-d),
                                         EfficientNet + PCA-32 (fit on training cells of the fold).
 2. LightGBM loss per boosting round  : cell-level APL-vs-AML (M1 = engineered:LightGBM), patient-grouped folds,
                                         training vs held-out binary log-loss and AUROC over 400 rounds.
 3. Learning curve (sample size)      : pooled-statistics logistic regression trained on n Discovery patients
                                         (n = 16 ... all); AUROC on the held-out fold and on the Validation cohort.

  python scripts/run_training_curves.py --manifest manifests/apl_aml.csv --hand features/apl_aml_hand.parquet \
      --effnet features/apl_aml_effnet.npy --out results/training_curves
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench.mil import train_abmil
from leukbench.models import load_feature_block
from leukbench.splits import patient_folds

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--hand", required=True)
ap.add_argument("--effnet", required=True)
ap.add_argument("--max-cells", type=int, default=500)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--parts", nargs="+", default=["abmil", "lgbm", "size"])
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(1)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

man = pd.read_csv(a.manifest, keep_default_na=False)
H = load_feature_block(a.hand); E = np.nan_to_num(load_feature_block(a.effnet), nan=0.0)
man["y"] = (man.diagnosis == "APL").astype(int)
pt = man.groupby("patient_id").agg(y=("y", "first"), cohort=("cohort", "first")).reset_index()
disc = pt[pt.cohort == "Discovery"].reset_index(drop=True)
val = pt[pt.cohort == "Validation"].reset_index(drop=True)
rng0 = np.random.default_rng(0)
rows_of = {}
for pid, g in man.groupby("patient_id"):
    idx = g.index.to_numpy()
    rows_of[pid] = np.sort(rng0.choice(idx, a.max_cells, replace=False)) if len(idx) > a.max_cells else idx


class Rep:
    """Cell representation fitted on training rows only."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, rows):
        if self.kind == "eng":
            self.med = np.nanmedian(H[rows], 0)
        elif self.kind.startswith("eff_pca"):
            k = int(self.kind[7:])
            self.sc = StandardScaler().fit(E[rows])
            self.pca = PCA(k, random_state=0).fit(self.sc.transform(E[rows]))
        return self

    def __call__(self, rows):
        if self.kind == "eng":
            return np.where(np.isnan(H[rows]), self.med, H[rows]).astype("float32")
        if self.kind == "eff_raw":
            return E[rows]
        return self.pca.transform(self.sc.transform(E[rows])).astype("float32")


def pooled(X):
    return np.concatenate([X.mean(0), X.std(0), *np.percentile(X, [10, 50, 90], axis=0)])


# ------------------------------------------------------------------ 1. ABMIL epochs
if "abmil" in a.parts:
    hist = []
    for kind in ["eng", "eff_raw", "eff_pca32"]:
        for seed in a.seeds:
            fold = patient_folds(disc.y.values, 5, seed)
            for k in range(5):
                t0 = time.time()
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                tp, ep = disc.patient_id.values[tr], disc.patient_id.values[te]
                R = Rep(kind).fit(np.concatenate([rows_of[p] for p in tp]))
                f = train_abmil([R(rows_of[p]) for p in tp], disc.y.values[tr], seed=seed, epochs=a.epochs,
                                monitor={"val": ([R(rows_of[p]) for p in ep], disc.y.values[te])})
                for h in f.history:
                    hist.append(dict(model="abmil", input=kind, seed=seed, fold=k, **h))
                last = f.history[-1]
                print(f"  ABMIL {kind:9s} seed {seed} fold {k}: train loss {last['train_loss']:.3f}  "
                      f"val loss {last['val_loss']:.3f}  val AUROC {last['val_auroc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(hist).to_csv(out / "abmil_history.csv", index=False)

# ------------------------------------------------------------------ 2. LightGBM rounds (cell level)
if "lgbm" in a.parts:
    from lightgbm import LGBMClassifier
    cell = man[(man.cohort == "Discovery")].index.to_numpy()
    y, g = man.y.values[cell], man.patient_id.values[cell]
    hist = []
    for seed in a.seeds:
        for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(cell, y, g)):
            imp = SimpleImputer(strategy="median").fit(H[cell[tr]])
            Xtr, Xte = imp.transform(H[cell[tr]]), imp.transform(H[cell[te]])
            m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, class_weight="balanced", random_state=seed, verbose=-1, n_jobs=2)
            m.fit(Xtr, y[tr], eval_set=[(Xtr, y[tr]), (Xte, y[te])], eval_names=["train", "val"],
                  eval_metric=["binary_logloss", "auc"])
            r = m.evals_result_
            for i in range(400):
                hist.append(dict(model="lgbm_cell", seed=seed, fold=k, round=i + 1, lr=0.05,
                                 train_loss=r["train"]["binary_logloss"][i], val_loss=r["val"]["binary_logloss"][i],
                                 train_auroc=r["train"]["auc"][i], val_auroc=r["val"]["auc"][i]))
            print(f"  LightGBM seed {seed} fold {k}: train loss {r['train']['binary_logloss'][-1]:.3f} "
                  f"val loss {r['val']['binary_logloss'][-1]:.3f} val AUROC {r['val']['auc'][-1]:.3f}", flush=True)
    pd.DataFrame(hist).to_csv(out / "lgbm_history.csv", index=False)

# ------------------------------------------------------------------ 3. learning curve vs number of patients
if "size" in a.parts:
    res = []
    for seed in a.seeds:
        fold = patient_folds(disc.y.values, 5, seed)
        rng = np.random.default_rng(100 + seed)
        for k in range(5):
            tr_all, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            for n in [16, 24, 32, 48, len(tr_all)]:
                # stratified subsample of n training patients
                pos, neg = tr_all[disc.y.values[tr_all] == 1], tr_all[disc.y.values[tr_all] == 0]
                npos = max(2, round(n * len(pos) / len(tr_all)))
                tr = np.concatenate([rng.choice(pos, min(npos, len(pos)), replace=False),
                                     rng.choice(neg, min(n - npos, len(neg)), replace=False)])
                tp = disc.patient_id.values[tr]
                cells = np.concatenate([rows_of[p] for p in tp])
                for kind in ["eng", "eff_raw", "eff_pca32"]:
                    R = Rep(kind).fit(cells)
                    P = lambda pids: np.stack([pooled(R(rows_of[p])) for p in pids])
                    clf = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))])
                    clf.fit(P(tp), disc.y.values[tr])
                    res.append(dict(seed=seed, fold=k, n_train=len(tr), input=kind, n_inputs=P(tp[:1]).shape[1],
                                    train_auroc=roc_auc_score(disc.y.values[tr], clf.predict_proba(P(tp))[:, 1]),
                                    cv_auroc=roc_auc_score(disc.y.values[te], clf.predict_proba(P(disc.patient_id.values[te]))[:, 1]),
                                    ext_auroc=roc_auc_score(val.y.values, clf.predict_proba(P(val.patient_id.values))[:, 1])))
            print(f"  size curve seed {seed} fold {k} done", flush=True)
        pd.DataFrame(res).to_csv(out / "sample_size_curve.csv", index=False)
print(f"wrote {out}")
