"""Patient-level bootstrap CIs for the leakage and shift components of the AUROC gap (cell-level tasks).

leakage = AUROC(image-level CV) - AUROC(patient-grouped CV): same development patients, paired bootstrap
          (development patients resampled with replacement, all their cells kept, both AUROCs recomputed).
shift   = AUROC(grouped CV) - AUROC(external): development and external patients resampled independently.
Each AUROC is the mean over seeds, as in the tables.

  python scripts/run_gap_bootstrap.py --results results --out results/gap_bootstrap [--boot 1000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = Path(a.results); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def prep(P):
    """dict seed -> (patient codes, y, p) for one split/eval."""
    d = {}
    for s, g in P.groupby("seed"):
        d[s] = (g.patient_id.values, g.y.values.astype(int), g.p.values)
    return d


def mean_auc(d, idx_of, pts):
    vals = []
    for s, (pid, y, p) in d.items():
        rows = np.concatenate([idx_of[s][q] for q in pts if q in idx_of[s]])
        vals.append(auc(y[rows], p[rows]))
    return float(np.mean(vals))


def index(d):
    return {s: pd.Series(np.arange(len(pid))).groupby(pid).apply(np.array).to_dict() for s, (pid, _, _) in d.items()}


rows = []
TASKS = [("T1 APL-AML: diagnosis on cells", "v2_cell_apl_cell_diagnosis", ["M1", "M1lr", "M2", "M4"]),
         ("T2 APL-AML: blast detection", "v2_cell_apl_cell_blast", ["M1", "M1lr", "M2", "M4"]),
         ("T4 C-NMC: cells", "cnmc_cell", ["M1lr", "M2", "M4"])]
rng = np.random.default_rng(0)
for task, folder, models in TASKS:
    P = pd.read_csv(R / folder / "predictions.csv.gz")
    P = P[~P.p.isna()]
    for m in models:
        q = P[P.model == m]
        ic, gc, ex = prep(q[(q.split == "image") & (q["eval"] == "cv")]), prep(q[(q.split == "grouped") & (q["eval"] == "cv")]), prep(q[(q.split == "grouped") & (q["eval"] == "test")])
        Iic, Igc, Iex = index(ic), index(gc), index(ex)
        dev = np.array(sorted(set(q[(q["eval"] == "cv")].patient_id))); ext = np.array(sorted(set(q[q["eval"] == "test"].patient_id)))
        a_ic, a_gc, a_ex = mean_auc(ic, Iic, dev), mean_auc(gc, Igc, dev), mean_auc(ex, Iex, ext)
        L, S = [], []
        for b in range(a.boot):
            db = rng.choice(dev, len(dev)); eb = rng.choice(ext, len(ext))
            try:
                gcb = mean_auc(gc, Igc, db)
                L.append(mean_auc(ic, Iic, db) - gcb); S.append(gcb - mean_auc(ex, Iex, eb))
            except ZeroDivisionError:
                continue
        L, S = np.array(L), np.array(S)
        rows.append(dict(task=task, model=m, image_cv=a_ic, grouped_cv=a_gc, ext=a_ex, leakage=a_ic - a_gc, leak_lo=np.nanpercentile(L, 2.5), leak_hi=np.nanpercentile(L, 97.5),
                         shift=a_gc - a_ex, shift_lo=np.nanpercentile(S, 2.5), shift_hi=np.nanpercentile(S, 97.5), boots=len(L)))
        print(f"{task:32s} {m:5s} leakage {rows[-1]['leakage']:+.3f} [{rows[-1]['leak_lo']:+.3f}, {rows[-1]['leak_hi']:+.3f}]  "
              f"shift {rows[-1]['shift']:+.3f} [{rows[-1]['shift_lo']:+.3f}, {rows[-1]['shift_hi']:+.3f}]", flush=True)
        pd.DataFrame(rows).to_csv(out / "gap_bootstrap.csv", index=False)
print("wrote", out)
