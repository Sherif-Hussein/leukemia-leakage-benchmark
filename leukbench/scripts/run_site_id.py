"""Site-identifiability check (E3): how well do the features predict WHICH DATASET a cell came from?

High site accuracy means a representation encodes lab/scanner/stain signatures, which inflates
within-dataset scores and hurts transfer. Reported per feature block as balanced accuracy of a
multinomial logistic regression with dataset-stratified 5-fold CV (cells subsampled per dataset).

  python scripts/run_site_id.py --manifest merged/crosslab/manifest.csv \
     --features hand=merged/crosslab/hand.npy effnet=merged/crosslab/effnet.npy --out results/e3_site_id.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench.runner import load_blocks

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--features", nargs="+", required=True)
ap.add_argument("--per-dataset", type=int, default=3000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

man = pd.read_csv(a.manifest, keep_default_na=False)
blocks = load_blocks(a.features, len(man))
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(g.index, min(len(g), a.per_dataset), replace=False)
                      for _, g in man.groupby("dataset")])
y = man.loc[idx, "dataset"].to_numpy()
rows = []
sets = list(blocks.items())
if len(blocks) > 1:
    sets.append(("+".join(blocks), np.hstack(list(blocks.values()))))
for name, X in sets:
    clf = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                    ("c", LogisticRegression(C=0.1, max_iter=3000))])
    pred = cross_val_predict(clf, X[idx], y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    rows.append(dict(features=name, site_bal_acc=balanced_accuracy_score(y, pred), n_sites=len(np.unique(y)),
                     chance=1 / len(np.unique(y))))
    print(rows[-1])
pd.DataFrame(rows).to_csv(a.out, index=False)
