"""Local threshold recalibration on the independent external cohort (MLL Helmholtz): what a new laboratory gets
when it confirms m local APL cases (and m local AML cases) before using a model trained elsewhere.

Rules (the development threshold flags almost every external patient, so it can only be raised):
  R0  development threshold (recall >= 0.95 on Discovery out-of-fold scores)
  R1  just below the lowest score of the m local APL cases
For each m, B random draws; the remaining patients form the test set. Reports mean recall, specificity, F1,
P(at least one APL patient missed) and the expected recall m/(m+1).

  python scripts/run_local_recal_ext.py --results results --out results/mll_recal
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--ms", nargs="+", type=int, default=[1, 2, 3, 5, 8, 12])
ap.add_argument("--draws", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = Path(a.results); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


def thr95(y, p):
    pos = np.sort(p[y == 1]); return float(pos[int(np.floor(0.05 * len(pos)))]) - 1e-9


M = {}
pp = pd.read_csv(R / "mll_patient" / "predictions.csv")
for m, nm in [("P1", "Engineered, pooled LR"), ("PDF", "DinoBloom + engineered, pooled LR")]:
    c = pp[(pp.model == m) & (pp["eval"] == "cv")].groupby("patient_id").agg(y=("y", "first"), p=("p", "mean"))
    t = pp[(pp.model == m) & (pp["eval"] == "test")].groupby("patient_id").agg(y=("y", "first"), p=("p", "mean"))
    M[nm] = (thr95(c.y.values, c.p.values), t.y.values, t.p.values)
c = pd.read_csv(R / "mll_ensemble" / "ensemble_cv.csv"); t = pd.read_csv(R / "mll_ensemble" / "ensemble_test.csv")
M["Primary pipeline (ensemble)"] = (thr95(c.y.values, c.p.values), t.y.values, t.p.values)
c = pd.read_csv(R / "mll_finetune_macenko" / "patient_predictions_cv.csv"); t = pd.read_csv(R / "mll_finetune_macenko" / "patient_predictions_test.csv")
M["Macenko + fine-tuned CNN"] = (thr95(c.y.values, c.p.values), t.y.values, t.p.values)

rng = np.random.default_rng(0); rows = []
for nm, (t0, y, p) in M.items():
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    pr = p >= t0; tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = ((~pr) & (y == 1)).sum()
    rows.append(dict(model=nm, m=0, rule="R0 development threshold", sens=tp / len(pos), spec=1 - fp / len(neg), f1=2 * tp / (2 * tp + fp + fn), p_any_miss=float(fn > 0), n_test=len(y)))
    for m in a.ms:
        S = []
        for _ in range(a.draws):
            lp, ln = rng.choice(pos, m, replace=False), rng.choice(neg, m, replace=False)
            thr = p[lp].min() - 1e-9
            te = np.setdiff1d(np.arange(len(y)), np.r_[lp, ln]); yy, pr = y[te], p[te] >= thr
            tp = (pr & (yy == 1)).sum(); fp = (pr & (yy == 0)).sum(); fn = ((~pr) & (yy == 1)).sum(); tn = ((~pr) & (yy == 0)).sum()
            S.append((tp / (tp + fn), tn / (tn + fp), 2 * tp / (2 * tp + fp + fn), fn > 0))
        S = np.array(S, float)
        rows.append(dict(model=nm, m=m, rule="R1 local APL cases", sens=S[:, 0].mean(), spec=S[:, 1].mean(), f1=S[:, 2].mean(), p_any_miss=S[:, 3].mean(),
                         expected=m / (m + 1), n_test=len(y) - 2 * m))
D = pd.DataFrame(rows); D.to_csv(out / "mll_recal.csv", index=False)
print(D.round(3).to_string())
