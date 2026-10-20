"""Same-cohort baselines from prior work, on the identical Discovery -> Validation split (APL-AML).

B1  Promyelocyte proportion per patient (the biomarker comparator of Sidhom et al. 2021,
    reproduced from their public code: signed promyelocytes / signed cells).
B2  Differential count: proportion of each signed cell type per patient -> logistic regression
    (the "differential count" comparator used by Dasdelen et al. 2026).
B3  CNN features + SVM (RBF) at cell level, averaged per patient (hybrid CNN + classical
    classifier design of Kasim et al. 2025; EfficientNet-B0 features instead of VGG16/InceptionV3).
B4  Single-cell classifier averaged per patient (single-cell design of Sidhom et al. 2021),
    for each cell-level representation: engineered, EfficientNet-B0, fusion.
Thresholds are chosen on Discovery (sensitivity >= 0.95) and applied unchanged to Validation.

  python scripts/run_baselines.py --manifest manifests/apl_aml.csv --hand features/apl_aml_hand.parquet \
     --effnet features/apl_aml_effnet.npy --out results/baselines
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench import metrics as MT
from leukbench.models import load_feature_block, make_classifier

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--hand", required=True)
ap.add_argument("--effnet", required=True)
ap.add_argument("--svm-cells", type=int, default=8000, help="training cells for the RBF-SVM (it scales quadratically)")
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

man = pd.read_csv(a.manifest, keep_default_na=False)
man["y"] = (man.diagnosis == "APL").astype(int)
pt = man.groupby("patient_id").agg(y=("y", "first"), cohort=("cohort", "first")).reset_index()
disc, val = pt[pt.cohort == "Discovery"], pt[pt.cohort == "Validation"]
rows = []


def score(name, design, p_disc, p_val, note=""):
    # operating point chosen on Discovery for sensitivity >= 0.95 (same rule as our models)
    thr = MT.threshold_for_sensitivity(disc.y.values, p_disc, 0.95)
    r = MT.bootstrap(val.y.values, p_val, val.patient_id.values, thr, a.n_boot)
    keep = ["auroc", "auroc_lo", "auroc_hi", "sens", "spec", "ppv", "npv", "f1", "acc", "bal_acc", "fn", "fp"]
    rows.append(dict(baseline=name, design=design, disc_auroc=MT.point_metrics(disc.y.values, p_disc, thr)["auroc"],
                     **{k: r[k] for k in keep}, note=note))
    print(f"  {name:55s} Discovery AUROC {rows[-1]['disc_auroc']:.3f} | Validation AUROC {r['auroc']:.3f} "
          f"[{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] sens {r['sens']:.2f} (missed {int(r['fn'])}) spec {r['spec']:.2f} "
          f"F1 {r['f1']:.2f} NPV {r['npv']:.2f} acc {r['acc']:.2f}", flush=True)


# ---------------- B1 promyelocyte proportion (no training)
signed = man[~man.cell_type.isin(["unlabelled"])]
prop = signed.groupby("patient_id").cell_type.apply(lambda s: (s == "Promyelocyte").mean())
score("Promyelocyte proportion", "Biomarker of Sidhom et al. [14]",
      disc.patient_id.map(prop).fillna(0).values, val.patient_id.map(prop).fillna(0).values)

# ---------------- B2 differential count + LR
excl = ["Arifact", "Unidentified", "Young Unidentified", "Thrombocyte aggregation"]
dc = signed[~signed.cell_type.isin(excl)].groupby(["patient_id", "cell_type"]).size().unstack(fill_value=0)
dc = dc.div(dc.sum(1), axis=0)
Xd, Xv = dc.reindex(disc.patient_id).fillna(0).values, dc.reindex(val.patient_id).fillna(0).values
lr = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))])
from sklearn.model_selection import cross_val_predict, StratifiedKFold
p_d = cross_val_predict(lr, Xd, disc.y.values, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
lr.fit(Xd, disc.y.values)
score("Differential count (%d cell types) + LR" % dc.shape[1], "Comparator of Dasdelen et al. [10]", p_d, lr.predict_proba(Xv)[:, 1],
      "Discovery AUROC is 5-fold CV")

# ---------------- cell-level models averaged per patient
H, E = load_feature_block(a.hand), load_feature_block(a.effnet)
blocks = {"Engineered": H, "EfficientNet-B0": E, "Fusion": np.hstack([H, E])}
cd = man.index[man.cohort == "Discovery"].to_numpy()
cv_ = man.index[man.cohort == "Validation"].to_numpy()


def patient_mean(rows_idx, p):
    s = pd.Series(p, index=man.loc[rows_idx, "patient_id"].values).groupby(level=0).mean()
    return s


def cell_model(name, design, X, clf, n_train=None, seed=0):
    rng = np.random.default_rng(seed)
    tr = cd if not n_train else np.sort(rng.choice(cd, min(n_train, len(cd)), replace=False))
    # Discovery patient scores from patient-grouped 5-fold CV (honest), Validation from a model fit on all Discovery
    from sklearn.model_selection import StratifiedGroupKFold
    p_cv = np.zeros(len(cd))
    g = man.loc[cd, "patient_id"].values
    for trf, tef in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(cd, man.loc[cd, "y"].values, g):
        trf_rows = cd[trf]
        if n_train and len(trf_rows) > n_train:
            trf_rows = np.sort(rng.choice(trf_rows, n_train, replace=False))
        m = clf().fit(X[trf_rows], man.loc[trf_rows, "y"].values)
        p_cv[tef] = m.predict_proba(X[cd[tef]])[:, 1]
    m = clf().fit(X[tr], man.loc[tr, "y"].values)
    pv = patient_mean(cv_, m.predict_proba(X[cv_])[:, 1])
    pdisc = patient_mean(cd, p_cv)
    score(name, design, pdisc.reindex(disc.patient_id).values, pv.reindex(val.patient_id).values,
          "cell probabilities averaged per patient")


svm = lambda: Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                        ("c", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=0))])
cell_model("EfficientNet-B0 features + RBF-SVM, cell mean", "Hybrid CNN + SVM design of Kasim et al. [4]", E, svm, a.svm_cells)
for bname, X in blocks.items():
    cell_model(f"{bname} + LR, cell mean", "Single-cell design of Sidhom et al. [14]", X, lambda: make_classifier("lr", 0))

pd.DataFrame(rows).to_csv(out / "baselines.csv", index=False)
print(f"wrote {out}/baselines.csv")
