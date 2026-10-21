"""Revision-2 statistics: formal shift test, C-NMC bag-size confound, calibration of patient-level models.

1. Shift test (label-free, transductive). Cohort identification on control-class cells only (AML for
   APL-AML, HEM for C-NMC): logistic regression, patient-grouped 5-fold CV, AUROC for 'external vs
   development'. Null distribution: cohort labels permuted between control-class *patients* (all cells of a
   patient keep one label), same CV. p = (1 + #null >= observed) / (1 + B). The test reads the unlabelled
   external images (no diagnoses), i.e. it is transductive: it is run once, before any model sees external labels.
2. Bag-size confound (C-NMC): AUROC of the number of cells per subject for ALL vs HEM; Spearman between the
   model score and log(cell count); likelihood-ratio test of whether the score adds to log(count).
3. Calibration of patient-level probabilities on the external sets: Brier score, calibration intercept
   (calibration-in-the-large, slope fixed at 1) and slope (logistic recalibration), both on raw outputs and
   after Platt scaling fitted on development out-of-fold scores. Ensemble scores are percentiles, so only the
   Platt-mapped version is meaningful for them. Reliability diagrams (quantile bins).

  python scripts/run_revision_stats.py --work <work dir> --out results/revision_stats [--perms 200]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leukbench.models import load_feature_block

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--perms", type=int, default=200)
ap.add_argument("--perms-cnn", type=int, default=50)
ap.add_argument("--skip-shift", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
W = Path(a.work); R = W / "results"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
STAIN = r"_(H|E|L|a|b|sat)_(mean|std)$|^H_ratio|^L_diff"
EPS = 1e-6

# ------------------------------------------------------------------ 1. shift permutation test
DS = {"APL-AML": dict(man="apl_aml.csv", hand="apl_aml_hand.parquet", cnn="apl_aml_effnet.npy", cnn_mac="apl_aml_effnet_macenko.npy",
                      coh="cohort", dev="Discovery", ext="Validation", neg="AML"),
      "C-NMC": dict(man="cnmc.csv", hand="cnmc_hand.parquet", cnn="cnmc_effnet.npy", cnn_mac=None,
                    coh="split_role", dev="Development", ext="External", neg="HEM")}


def cohort_auc(X, yc, g, seed=0):
    clf = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))])
    p = cross_val_predict(clf, X, yc, groups=g, cv=StratifiedGroupKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return roc_auc_score(yc, p)


if not a.skip_shift:
    rows = []
    for ds, d in DS.items():
        man = pd.read_csv(W / "manifests" / d["man"], keep_default_na=False)
        H = pd.read_parquet(W / "features" / d["hand"])
        stain_cols = H.columns[H.columns.str.contains(STAIN)]
        blocks = {"Engineered: stain and colour": H[stain_cols].to_numpy("float32"), "Engineered (all)": H.to_numpy("float32"),
                  "EfficientNet-B0 (frozen)": np.nan_to_num(load_feature_block(str(W / "features" / d["cnn"])))}
        if d["cnn_mac"] and (W / "features" / d["cnn_mac"]).exists():
            blocks["EfficientNet-B0 on Macenko images"] = np.nan_to_num(load_feature_block(str(W / "features" / d["cnn_mac"])))
        idx = man.index[man.diagnosis == d["neg"]].to_numpy()
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, min(6000, len(idx)), replace=False))
        g = man.loc[idx, "patient_id"].values
        pt = pd.Series(man.loc[idx, d["coh"]].values == d["ext"], index=g).groupby(level=0).first()
        yc = pt.loc[g].astype(int).values
        for name, X in blocks.items():
            X = X[idx]
            obs = cohort_auc(X, yc, g)
            B = a.perms if X.shape[1] < 200 else a.perms_cnn
            null = []
            for b in range(B):
                perm = pd.Series(rng.permutation(pt.values), index=pt.index)
                null.append(cohort_auc(X, perm.loc[g].astype(int).values, g, seed=b + 1))
            null = np.array(null)
            pv = (1 + (null >= obs).sum()) / (1 + B)
            rows.append(dict(dataset=ds, representation=name, control_patients=len(pt), external_patients=int(pt.sum()), cells=len(idx),
                             cohort_auroc=obs, null_mean=null.mean(), null_95=np.percentile(null, 95), p_value=pv, permutations=B))
            print(f"  shift {ds:8s} {name:36s} AUROC {obs:.3f}  null {null.mean():.3f} (95th {np.percentile(null, 95):.3f})  p={pv:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(out / "shift_test.csv", index=False)

# ------------------------------------------------------------------ 2. C-NMC bag-size confound
man = pd.read_csv(W / "manifests" / "cnmc.csv", keep_default_na=False)
cnt = man.groupby("patient_id").agg(n=("path", "size"), y=("diagnosis", lambda s: int(s.iloc[0] == "ALL")), role=("split_role", "first"))
bag = []
scores = {"Engineered, pooled LR": ("cnmc_patient", "P1"), "Engineered, attention MIL": ("cnmc_patient", "P4"),
          "Fine-tuned EfficientNet-B0": ("finetune_cnmc", None), "Primary ensemble": ("ensemble_cnmc", None)}


def load_scores(folder, model, ev):
    f = R / folder
    if model:
        p = pd.read_csv(f / "predictions.csv"); p = p[(p.model == model) & (p["eval"] == ev)]
        return p.groupby("patient_id").agg(y=("y", "first"), p=("p", "mean"))
    nm = {"cv": "cv", "test": "test"}[ev]
    fn = f / (f"patient_predictions_{nm}.csv" if (f / f"patient_predictions_{nm}.csv").exists() else f"ensemble_{nm}.csv")
    return pd.read_csv(fn).set_index("patient_id")[["y", "p"]]


def lr_test(y, base, extra):
    m0 = LogisticRegression(C=1e6, max_iter=5000).fit(base, y); m1 = LogisticRegression(C=1e6, max_iter=5000).fit(np.c_[base, extra], y)
    ll = lambda m, X: np.sum(np.log(np.clip(np.where(y == 1, m.predict_proba(X)[:, 1], m.predict_proba(X)[:, 0]), 1e-12, 1)))
    return float(stats.chi2.sf(2 * (ll(m1, np.c_[base, extra]) - ll(m0, base)), 1))


for role, ev in [("Development", "cv"), ("External", "test")]:
    c = cnt[cnt.role == role]
    r = dict(set=role, subjects=len(c), median_cells_ALL=c.n[c.y == 1].median(), median_cells_HEM=c.n[c.y == 0].median(),
             count_auroc=roc_auc_score(c.y, c.n), mannwhitney_p=stats.mannwhitneyu(c.n[c.y == 1], c.n[c.y == 0]).pvalue)
    for nm, (folder, model) in scores.items():
        s = load_scores(folder, model, ev).join(c[["n"]], how="inner")
        lc = np.log(s.n.values).reshape(-1, 1); z = logit(np.clip(s.p.values, EPS, 1 - EPS)).reshape(-1, 1)
        r[f"{nm}: AUROC"] = roc_auc_score(s.y, s.p)
        r[f"{nm}: Spearman(score, log n)"] = stats.spearmanr(s.p, s.n).correlation
        r[f"{nm}: score adds to log n, LR p"] = lr_test(s.y.values, lc, z)
        # AUROC of the score within count strata (residual of score on log n)
        res = z.ravel() - np.polyval(np.polyfit(lc.ravel(), z.ravel(), 1), lc.ravel())
        r[f"{nm}: AUROC of count-adjusted score"] = roc_auc_score(s.y, res)
    bag.append(r)
bag = pd.DataFrame(bag); bag.to_csv(out / "cnmc_bag_size.csv", index=False)
print(bag.T.round(3).to_string(), flush=True)

# ------------------------------------------------------------------ 3. calibration
APL = {"Engineered, pooled LR": ("v2_patient", "P1"), "Engineered, attention MIL": ("v2_patient", "P4"),
       "Macenko + fine-tuned CNN": ("finetune_t1_macenko", None), "Secondary (Macenko + frozen CNN + PCA-64)": ("apl_fe_macenko", "eff_pca64_nw"),
       "Primary ensemble": ("ensemble_apl_macenko", None)}
CN = {"Engineered, pooled LR": ("cnmc_patient", "P1"), "Engineered, attention MIL": ("cnmc_patient", "P4"),
      "Fine-tuned CNN": ("finetune_cnmc", None), "Secondary (frozen CNN + PCA-64)": ("cnmc_fe_nw", "eff_pca64_nw"), "Primary ensemble": ("ensemble_cnmc", None)}


def load_any(folder, model, ev):
    if folder.endswith(("_fe_nw", "_fe_macenko")):
        if ev == "cv":
            return pd.read_csv(R / folder / f"oof_{model}.csv").set_index("patient_id")[["y", "p"]]
        p = pd.read_csv(R / folder / "predictions.csv"); return p[p.config == model].set_index("patient_id")[["y", "p"]]
    return load_scores(folder, model, ev)


def cal_metrics(y, p):
    p = np.clip(p, EPS, 1 - EPS); z = logit(p)
    slope_m = LogisticRegression(C=1e6, max_iter=5000).fit(z.reshape(-1, 1), y)
    # calibration-in-the-large: intercept with offset z (Newton on one parameter)
    b = 0.0
    for _ in range(50):
        q = expit(z + b); b += np.sum(y - q) / max(np.sum(q * (1 - q)), 1e-9)
    return dict(brier=float(np.mean((p - y) ** 2)), cal_intercept=float(b), cal_slope=float(slope_m.coef_[0, 0]),
                mean_pred=float(p.mean()), observed=float(y.mean()))


cal, curves = [], {}
for ds, spec in [("APL-AML", APL), ("C-NMC", CN)]:
    for nm, (folder, model) in spec.items():
        try:
            dv, te = load_any(folder, model, "cv"), load_any(folder, model, "test")
        except FileNotFoundError as e:
            print("  missing", nm, e); continue
        platt = LogisticRegression(C=1e6, max_iter=5000).fit(logit(np.clip(dv.p.values, EPS, 1 - EPS)).reshape(-1, 1), dv.y.values)
        pp = platt.predict_proba(logit(np.clip(te.p.values, EPS, 1 - EPS)).reshape(-1, 1))[:, 1]
        is_prob = "ensemble" not in nm.lower()
        r = dict(dataset=ds, model=nm, n=len(te), auroc=roc_auc_score(te.y, te.p))
        if is_prob:
            r.update({f"raw_{k}": v for k, v in cal_metrics(te.y.values, te.p.values).items()})
        r.update({f"platt_{k}": v for k, v in cal_metrics(te.y.values, pp).items()})
        cal.append(r); curves[(ds, nm)] = (te.y.values, te.p.values if is_prob else None, pp)
cal = pd.DataFrame(cal); cal.to_csv(out / "calibration.csv", index=False)
print(cal.round(3).to_string(), flush=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
cols = ["#8C8C8C", "#C7792B", "#2F6DB5", "#6A9F58", "#B5342F"]
for ax, ds in zip(axs, ["APL-AML", "C-NMC"]):
    ax.plot([0, 1], [0, 1], ls=":", c="k", lw=0.8)
    for i, ((d, nm), (y, praw, pp)) in enumerate([(k, v) for k, v in curves.items() if k[0] == ds]):
        bins = np.unique(np.quantile(pp, np.linspace(0, 1, 5)))
        idx = np.clip(np.digitize(pp, bins[1:-1]), 0, len(bins) - 2)
        mx = [pp[idx == j].mean() for j in range(len(bins) - 1) if (idx == j).any()]
        my = [y[idx == j].mean() for j in range(len(bins) - 1) if (idx == j).any()]
        ax.plot(mx, my, "o-", c=cols[i % len(cols)], lw=1.2 if "Primary" in nm else 0.8, ms=3, label=nm)
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("Predicted probability (Platt on development OOF)"); ax.set_ylabel("Observed fraction positive")
    ax.set_title(f"({'ab'[ds != 'APL-AML']}) {ds} external, {len(y)} patients, quartile bins", loc="left", fontsize=7.5)
axs[1].legend(frameon=False, fontsize=6, loc="lower right")
fig.tight_layout(); fig.savefig(out / "fig_reliability.pdf"); fig.savefig(out / "fig_reliability.png", dpi=200)
print("wrote", out)
