"""Pre-specified patient-level ensemble (no weights or members tuned on any data).

Members (the same on every dataset): engineered features + pooled LR (P1), engineered features +
attention MIL (P4), and the fine-tuned EfficientNet-B0 (cell probabilities averaged per patient).
Each member's score is mapped to its percentile among that member's out-of-fold development scores
(so the external scores are never used to normalize), and the three percentiles are averaged.
The operating threshold (sensitivity >= 0.95) is chosen on the ensemble's out-of-fold development scores.

  python scripts/run_ensemble.py --patient results/v2_patient --finetune results/finetune_t1 --out results/ensemble_apl
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench import metrics as MT

ap = argparse.ArgumentParser()
ap.add_argument("--patient", required=True, help="run_patient_benchmark output folder (P1, P4)")
ap.add_argument("--finetune", required=True, help="run_finetune_cached output folder (patient_predictions_*.csv)")
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

pp = pd.read_csv(Path(a.patient) / "predictions.csv")
members = {}
for m in ["P1", "P4"]:
    q = pp[pp.model == m]
    cv = q[q["eval"] == "cv"].groupby("patient_id").agg(y=("y", "first"), p=("p", "mean"))
    te = q[q["eval"] == "test"].groupby("patient_id").agg(y=("y", "first"), p=("p", "mean"))
    members[m] = (cv, te)
ft = Path(a.finetune)
members["FT"] = (pd.read_csv(ft / "patient_predictions_cv.csv").set_index("patient_id")[["y", "p"]],
                 pd.read_csv(ft / "patient_predictions_test.csv").set_index("patient_id")[["y", "p"]])

ids_cv = sorted(set.intersection(*[set(c.index) for c, _ in members.values()]))
ids_te = sorted(set.intersection(*[set(t.index) for _, t in members.values()]))


def pct(ref, x):
    ref = np.sort(ref)
    return (np.searchsorted(ref, x, side="right") + np.searchsorted(ref, x, side="left")) / (2 * len(ref))


S_cv, S_te = [], []
for nm, (cv, te) in members.items():
    S_cv.append(pct(cv.loc[ids_cv, "p"].values, cv.loc[ids_cv, "p"].values))
    S_te.append(pct(cv.loc[ids_cv, "p"].values, te.loc[ids_te, "p"].values))
y_cv = members["P1"][0].loc[ids_cv, "y"].values; y_te = members["P1"][1].loc[ids_te, "y"].values
e_cv, e_te = np.mean(S_cv, 0), np.mean(S_te, 0)
thr = MT.threshold_for_sensitivity(y_cv, e_cv, 0.95)
rows = []
for nm, sc, st in [("Ensemble", e_cv, e_te)] + [(k, s_c, s_t) for k, s_c, s_t in zip(members, S_cv, S_te)]:
    t = MT.threshold_for_sensitivity(y_cv, sc, 0.95)
    b = MT.bootstrap(y_te, st, np.array(ids_te), t, a.n_boot)
    m = MT.point_metrics(y_te, st, t)
    rows.append(dict(model=nm, cv_auroc=MT.point_metrics(y_cv, sc, 0.5)["auroc"], ext_auroc=m["auroc"], lo=b["auroc_lo"], hi=b["auroc_hi"],
                     **{k: m[k] for k in ["sens", "spec", "f1", "npv", "acc", "fn", "fp"]}, n_pos=int(y_te.sum()), n=len(y_te)))
R = pd.DataFrame(rows)
R.to_csv(out / "ensemble.csv", index=False)
pd.DataFrame({"patient_id": ids_te, "y": y_te, "p": e_te}).to_csv(out / "ensemble_test.csv", index=False)
pd.DataFrame({"patient_id": ids_cv, "y": y_cv, "p": e_cv}).to_csv(out / "ensemble_cv.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(3).to_string())
