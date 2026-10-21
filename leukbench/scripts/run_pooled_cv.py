"""Secondary analysis: repeated patient-level CV over ALL 106 patients (Discovery + Validation pooled).

Purpose: more statistical power for comparisons between representations (n = 106 test patients
instead of 24). It is NOT an external validation: all patients come from one hospital and analyzer,
and the temporal separation is given up. Every step (imputation, PCA, scaler, classifier, threshold)
is fitted inside the training folds.

Protocol: R repeats x 5 folds, stratified by diagnosis x cohort. In each outer training fold an inner
3-fold patient CV gives out-of-fold scores on which the sensitivity >= 0.95 threshold is chosen; the
outer fold model (fitted on all training patients) is then applied with that threshold.

  python scripts/run_pooled_cv.py --manifest manifests/apl_aml.csv --hand features/apl_aml_hand.parquet \
      --effnet features/apl_aml_effnet.npy --repeats 5 --out results/pooled_cv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench import metrics as MT
from leukbench.mil import train_abmil
from leukbench.models import load_feature_block

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--hand", required=True)
ap.add_argument("--effnet", required=True)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--max-cells", type=int, default=500)
ap.add_argument("--configs", nargs="+", default=["promyelocyte_fraction", "eng_raw", "eff_raw", "eff_pca16", "eff_pca32", "eff_pca64",
                                                  "fusion_pca32", "mil_eng"])
ap.add_argument("--n-boot", type=int, default=4000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(1)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

man = pd.read_csv(a.manifest, keep_default_na=False)
H = load_feature_block(a.hand); E = np.nan_to_num(load_feature_block(a.effnet), nan=0.0)
man["y"] = (man.diagnosis == "APL").astype(int)
pt = man.groupby("patient_id").agg(y=("y", "first"), cohort=("cohort", "first")).reset_index()
strata = pt.y.astype(str) + pt.cohort
rng0 = np.random.default_rng(0)
rows_of = {}
for pid, g in man.groupby("patient_id"):
    idx = g.index.to_numpy()
    rows_of[pid] = np.sort(rng0.choice(idx, a.max_cells, replace=False)) if len(idx) > a.max_cells else idx
# promyelocyte fraction among operator-signed cells (biomarker of Sidhom et al.; no training)
signed = man[man.cell_type != ""]
promyelo = signed.groupby("patient_id").cell_type.apply(lambda s: (s == "Promyelocyte").mean()).reindex(pt.patient_id).fillna(0).values


class Rep:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, rows):
        if "eng" in self.kind or "fusion" in self.kind:
            self.med = np.nanmedian(H[rows], 0)
        if "pca" in self.kind:
            k = int(self.kind.split("pca")[1])
            self.sc = StandardScaler().fit(E[rows]); self.pca = PCA(k, random_state=0).fit(self.sc.transform(E[rows]))
        return self

    def __call__(self, rows):
        parts = []
        if "eng" in self.kind or "fusion" in self.kind:
            parts.append(np.where(np.isnan(H[rows]), self.med, H[rows]))
        if self.kind == "eff_raw":
            parts.append(E[rows])
        if "pca" in self.kind:
            parts.append(self.pca.transform(self.sc.transform(E[rows])))
        return np.hstack(parts).astype("float32")


def pooled(X):
    return np.concatenate([X.mean(0), X.std(0), *np.percentile(X, [10, 50, 90], axis=0)])


def fit_predict(cfg, tr_pids, ytr, te_pids, seed):
    if cfg == "promyelocyte_fraction":
        idx = {p: i for i, p in enumerate(pt.patient_id)}
        return np.array([promyelo[idx[p]] for p in te_pids])
    kind = "eng" if cfg == "mil_eng" else cfg.replace("_nw", "")
    R = Rep(kind).fit(np.concatenate([rows_of[p] for p in tr_pids]))
    if cfg.startswith("mil"):
        f = train_abmil([R(rows_of[p]) for p in tr_pids], ytr, seed=seed, epochs=40)
        return f([R(rows_of[p]) for p in te_pids])
    Xtr = np.stack([pooled(R(rows_of[p])) for p in tr_pids]); Xte = np.stack([pooled(R(rows_of[p])) for p in te_pids])
    if cfg.endswith("_nw"):                 # keep relative variances of PCA scores (no whitening)
        mu, sd = Xtr.mean(0), (Xtr - Xtr.mean(0)).std() + 1e-9
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        clf = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit(Xtr, ytr)
    else:
        clf = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))]).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


res, allp = [], []
for cfg in a.configs:
    t0 = time.time()
    for rep in range(a.repeats):
        p_oof, thr_row = np.zeros(len(pt)), np.zeros(len(pt))
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(pt, strata)):
            tp, ytr = pt.patient_id.values[tr], pt.y.values[tr]
            # inner CV on the training patients -> threshold for sensitivity >= 0.95
            inner = np.zeros(len(tr))
            for itr, ite in StratifiedKFold(3, shuffle=True, random_state=100 + rep).split(tp, ytr):
                inner[ite] = fit_predict(cfg, tp[itr], ytr[itr], tp[ite], rep)
            thr_row[te] = MT.threshold_for_sensitivity(ytr, inner, 0.95)
            p_oof[te] = fit_predict(cfg, tp, ytr, pt.patient_id.values[te], rep)
        m = MT.point_metrics(pt.y.values, p_oof, 0.5)
        pred = (p_oof >= thr_row).astype(int); y = pt.y.values
        tp_, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
        tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
        sens, spec = tp_ / (tp_ + fn), tn / (tn + fp)
        res.append(dict(config=cfg, repeat=rep, auroc=m["auroc"], sens=sens, spec=spec, fn=fn, fp=fp,
                        f1=2 * tp_ / (2 * tp_ + fp + fn), npv=tn / max(tn + fn, 1), acc=(tp_ + tn) / len(y)))
        allp.append(pd.DataFrame(dict(config=cfg, repeat=rep, patient_id=pt.patient_id, cohort=pt.cohort, y=y, p=p_oof, thr=thr_row)))
    r = pd.DataFrame([x for x in res if x["config"] == cfg])
    print(f"  {cfg:22s} AUROC {r.auroc.mean():.3f}±{r.auroc.std():.3f} | recall {r.sens.mean():.3f} (missed {r.fn.mean():.1f}/34) "
          f"spec {r.spec.mean():.3f} F1 {r.f1.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(res).to_csv(out / "pooled_cv_repeats.csv", index=False)
    pd.concat(allp).to_csv(out / "pooled_cv_predictions.csv.gz", index=False)

# summary with bootstrap CI (patients) on repeat-averaged scores, and paired tests vs references
P = pd.concat(allp).groupby(["config", "patient_id"]).agg(y=("y", "first"), p=("p", "mean")).reset_index()
R = pd.DataFrame(res)
rows = []
for cfg in a.configs:
    q = P[P.config == cfg].sort_values("patient_id")
    b = MT.bootstrap(q.y.values, q.p.values, q.patient_id.values, 0.5, a.n_boot)
    rec = dict(config=cfg, auroc_mean_of_repeats=R[R.config == cfg].auroc.mean(), auroc_avg_scores=b["auroc"], lo=b["auroc_lo"], hi=b["auroc_hi"],
               **{k: R[R.config == cfg][k].mean() for k in ["sens", "spec", "fn", "fp", "f1", "npv", "acc"]})
    for ref in ["eff_raw", "eng_raw", "eff_raw_nw"]:
        if cfg != ref and ref in a.configs:
            r0 = P[P.config == ref].sort_values("patient_id")
            d = MT.paired_bootstrap_auc(q.y.values, q.p.values, r0.p.values, q.patient_id.values, a.n_boot)
            rec.update({f"d_vs_{ref}": d["delta"], f"d_vs_{ref}_lo": d["lo"], f"d_vs_{ref}_hi": d["hi"], f"p_vs_{ref}": d["p"]})
    rows.append(rec)
S = pd.DataFrame(rows)
S.to_csv(out / "pooled_cv_summary.csv", index=False)
pd.set_option("display.width", 250)
print(S.round(3).to_string())
print(f"wrote {out}")
