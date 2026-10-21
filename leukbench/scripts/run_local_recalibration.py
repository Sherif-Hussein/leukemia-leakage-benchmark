"""What can a new laboratory do when the development threshold does not transfer?

1. Few-shot local threshold recalibration. The new site contributes k confirmed local cases (k/2 APL,
   k/2 AML), drawn at random from the Validation cohort; the remaining 24 - k patients are the test set.
   Repeated B times. Threshold rules:
     R0  development threshold (sensitivity >= 0.95 on Discovery out-of-fold scores), no local cases
     R1  local only: just below the lowest score among the local APL cases
     R2  safety shift: min(R0, R1) - the threshold is only ever lowered, never raised
2. Prevalence-adjusted predictive values: PPV and NPV at APL prevalence 5%, 10% and 27% (Discovery).
3. Decision-curve analysis: scores mapped to risks by logistic (Platt) recalibration on the Discovery
   out-of-fold scores, shifted to the target prevalence (prior shift), net benefit on the Validation
   cohort reweighted to that prevalence.

  python scripts/run_local_recalibration.py --results results --out results/local_recalibration
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--ks", nargs="+", type=int, default=[2, 4, 6, 8])
ap.add_argument("--draws", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = Path(a.results); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
EPS = 1e-6


def load_models():
    M = {}
    names = {"eff_raw": "EfficientNet-B0, no PCA", "eff_pca16": "EfficientNet-B0 + PCA-16", "eff_pca32": "EfficientNet-B0 + PCA-32",
             "eff_pca64": "EfficientNet-B0 + PCA-64 (pre-declared)", "eng_raw": "Engineered, pooled LR"}
    fe = pd.read_csv(R / "feature_engineering" / "predictions.csv")
    for cfg, nm in names.items():
        f = R / "feature_engineering_oof" / f"oof_{cfg}.csv"
        if f.exists():
            o = pd.read_csv(f); t = fe[fe.config == cfg]
            M[nm] = (o.y.values, o.p.values, float(o.thr_sens95.iloc[0]), t.y.values, t.p.values)
    vp = pd.read_csv(R / "v2_patient" / "predictions.csv")
    for m, nm in [("P4", "Engineered, attention MIL")]:
        c = vp[(vp.model == m) & (vp["eval"] == "cv")].groupby("patient_id").agg(y=("y", "first"), p=("p", "mean"))
        t = vp[(vp.model == m) & (vp["eval"] == "test")].groupby("patient_id").agg(y=("y", "first"), p=("p", "mean"))
        thr = np.sort(c.p.values[c.y.values == 1])[int(np.floor(0.05 * (c.y.values == 1).sum()))] - 1e-9
        M[nm] = (c.y.values, c.p.values, float(thr), t.y.values, t.p.values)
    ft = R / "finetune_t1"
    if (ft / "patient_predictions_cv.csv").exists():
        c = pd.read_csv(ft / "patient_predictions_cv.csv"); t = pd.read_csv(ft / "patient_predictions_test.csv")
        thr = np.sort(c.p.values[c.y.values == 1])[int(np.floor(0.05 * (c.y.values == 1).sum()))] - 1e-9
        M["Fine-tuned EfficientNet-B0"] = (c.y.values, c.p.values, float(thr), t.y.values, t.p.values)
    return M


def rates(y, p, thr):
    pred = p >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    return tp, fn, tn, fp


M = load_models()
print("models:", list(M))
rng = np.random.default_rng(0)
rows = []
for nm, (yd, pd_, thr0, yv, pv) in M.items():
    pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
    tp, fn, tn, fp = rates(yv, pv, thr0)
    rows.append(dict(model=nm, k=0, rule="R0 development threshold", sens=tp / (tp + fn), spec=tn / (tn + fp), f1=2 * tp / (2 * tp + fp + fn),
                     npv=tn / max(tn + fn, 1), acc=(tp + tn) / len(yv), missed=fn / (tp + fn), p_any_miss=float(fn > 0), n_test=len(yv)))
    for k in a.ks:
        acc = {"R1 local cases only": [], "R2 safety shift": []}
        for _ in range(a.draws):
            loc = np.concatenate([rng.choice(pos, k // 2, replace=False), rng.choice(neg, k // 2, replace=False)])
            test = np.setdiff1d(np.arange(len(yv)), loc)
            t1 = pv[loc][yv[loc] == 1].min() - EPS
            for rule, thr in [("R1 local cases only", t1), ("R2 safety shift", min(thr0, t1))]:
                acc[rule].append(rates(yv[test], pv[test], thr))
        for rule, L in acc.items():
            L = np.array(L, float); tp, fn, tn, fp = L.T
            rows.append(dict(model=nm, k=k, rule=rule, sens=np.mean(tp / (tp + fn)), spec=np.mean(tn / (tn + fp)),
                             f1=np.mean(2 * tp / (2 * tp + fp + fn)), npv=np.mean(tn / np.maximum(tn + fn, 1)), acc=np.mean((tp + tn) / (tp + fn + tn + fp)),
                             missed=np.mean(fn / (tp + fn)), p_any_miss=np.mean(fn > 0), n_test=len(yv) - k))
T = pd.DataFrame(rows)
T.to_csv(out / "local_recalibration.csv", index=False)
pd.set_option("display.width", 250)
print(T[T.k.isin([0, 4])].round(3).to_string())

# prevalence-adjusted predictive values
pv_rows = []
for _, r in T[(T.k == 0) | ((T.k == 4) & (T.rule == "R2 safety shift"))].iterrows():
    for prev in [0.05, 0.10, 0.27]:
        se, sp = r.sens, r.spec
        ppv = se * prev / (se * prev + (1 - sp) * (1 - prev)) if se * prev + (1 - sp) * (1 - prev) > 0 else np.nan
        den = sp * (1 - prev) + (1 - se) * prev
        npv = sp * (1 - prev) / den if den > 0 else np.nan
        pv_rows.append(dict(model=r.model, k=r.k, rule=r.rule, prevalence=prev, ppv=ppv, npv=npv, sens=se, spec=sp))
PV = pd.DataFrame(pv_rows); PV.to_csv(out / "predictive_values.csv", index=False)

# decision curves
pts = np.round(np.arange(0.01, 0.61, 0.01), 2)
dca = []
for nm, (yd, pd_, thr0, yv, pv) in M.items():
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(np.clip(pd_, 1e-6, 1 - 1e-6)).reshape(-1, 1), yd)
    s_val = lr.decision_function(logit(np.clip(pv, 1e-6, 1 - 1e-6)).reshape(-1, 1))
    pi_dev = yd.mean()
    for prev in [0.10, 0.50]:
        risk = expit(s_val - logit(pi_dev) + logit(prev))
        for pt_ in pts:
            tp, fn, tn, fp = rates(yv, risk, pt_)
            se, sp = tp / (tp + fn), tn / (tn + fp)
            w = pt_ / (1 - pt_)
            dca.append(dict(model=nm, prevalence=prev, pt=pt_, net_benefit=se * prev - (1 - sp) * (1 - prev) * w,
                            treat_all=prev - (1 - prev) * w, sens=se, spec=sp))
D = pd.DataFrame(dca); D.to_csv(out / "decision_curves.csv", index=False)
print(D[(D.prevalence == 0.10) & (D.pt.isin([0.05, 0.10, 0.20]))].round(3).to_string())
print(f"wrote {out}")
