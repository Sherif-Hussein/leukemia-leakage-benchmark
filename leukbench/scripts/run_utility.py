"""Clinical-utility analyses requested in review (external cohorts, patient level).

1. Workload-sensitivity: refer the top q% of patients by score; recall (APL caught), specificity and PPV, with
   patient-bootstrap 95% CIs; workload needed for recall 0.95 and 1.00. Precision-recall curve and average precision.
2. Prevalence-adjusted PPV/NPV at the workload operating points (APL prevalence 5% and 10%).
3. Local recalibration with a SEPARATE calibration sample: the external cohort is split at random into a local
   calibration sample (n_pos APL, n_neg AML) and an evaluation sample (all others); thresholds are set on the
   calibration sample only and evaluated on the evaluation sample (2,000 draws). Rules:
     POS  just below the lowest calibration APL score (positive-only, protects recall; expected recall m/(m+1))
     BOTH highest threshold with calibration recall >= 0.9 that maximizes calibration specificity, using both classes
   Reports mean and 5th percentile of evaluation recall, mean specificity and referral fraction.
4. Exact numbers of consecutive detected local positives needed to claim recall >= r with 95% confidence.

  python scripts/run_utility.py --results results --out results/utility
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--draws", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = Path(a.results); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def bench(folder, model):
    p = pd.read_csv(R / folder / "predictions.csv"); t = p[(p.model == model) & (p["eval"] == "test")]
    return t.groupby("patient_id").agg(y=("y", "first"), p=("p", "mean")).reset_index()


C = {
    ("Independent (MLL)", "Primary pipeline"): pd.read_csv(R / "mll_ensemble" / "ensemble_test.csv"),
    ("Independent (MLL)", "DinoBloom + engineered"): bench("mll_patient", "PDF"),
    ("Independent (MLL)", "Engineered, pooled LR"): bench("mll_patient", "P1"),
    ("Independent (MLL)", "Fine-tuned CNN"): pd.read_csv(R / "mll_finetune_macenko" / "patient_predictions_test.csv"),
    ("APL-AML later cohort", "Primary pipeline"): pd.read_csv(R / "ensemble_apl_macenko" / "ensemble_test.csv"),
    ("APL-AML later cohort", "DinoBloom + engineered"): bench("apl_patient_dino", "PDF"),
    ("APL-AML later cohort", "DinoBloom, attention MIL"): bench("apl_patient_dino", "PDM"),
}
QS = [0.05, 0.10, 0.20, 0.30, 0.50]


def at_top(y, p, q):
    k = max(1, int(np.ceil(q * len(y)))); idx = np.argsort(-p, kind="stable")[:k]
    ref = np.zeros(len(y), bool); ref[idx] = True
    tp = (ref & (y == 1)).sum(); fp = (ref & (y == 0)).sum()
    return tp / (y == 1).sum(), 1 - fp / (y == 0).sum(), tp / max(ref.sum(), 1)


def workload_for(y, p, r):
    o = np.argsort(-p, kind="stable"); c = np.cumsum(y[o] == 1) / (y == 1).sum()
    return (np.argmax(c >= r - 1e-12) + 1) / len(y)


W, PR = [], {}
for (coh, nm), d in C.items():
    y, p = d.y.values.astype(int), d.p.values
    PR[(coh, nm)] = precision_recall_curve(y, p)
    row = dict(cohort=coh, model=nm, n=len(y), n_pos=int(y.sum()), auroc=roc_auc_score(y, p), ap=average_precision_score(y, p),
               prevalence=y.mean(), workload_r95=workload_for(y, p, 0.95), workload_r100=workload_for(y, p, 1.0))
    B = []
    for _ in range(a.boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        B.append([*[at_top(y[i], p[i], q)[0] for q in QS], average_precision_score(y[i], p[i]), workload_for(y[i], p[i], 0.95)])
    B = np.array(B)
    row["ap_lo"], row["ap_hi"] = np.percentile(B[:, len(QS)], [2.5, 97.5])
    row["w95_lo"], row["w95_hi"] = np.percentile(B[:, len(QS) + 1], [2.5, 97.5])
    for j, q in enumerate(QS):
        r, s, ppv = at_top(y, p, q)
        row[f"recall_top{int(q * 100)}"] = r; row[f"spec_top{int(q * 100)}"] = s; row[f"ppv_top{int(q * 100)}"] = ppv
        row[f"recall_top{int(q * 100)}_lo"], row[f"recall_top{int(q * 100)}_hi"] = np.percentile(B[:, j], [2.5, 97.5])
        for prev in (0.05, 0.10):   # prevalence-adjusted PPV / NPV for the sensitivity and specificity at this workload
            row[f"ppv{int(prev * 100)}_top{int(q * 100)}"] = r * prev / (r * prev + (1 - s) * (1 - prev) + 1e-12)
            row[f"npv{int(prev * 100)}_top{int(q * 100)}"] = s * (1 - prev) / (s * (1 - prev) + (1 - r) * prev + 1e-12)
    W.append(row)
W = pd.DataFrame(W); W.to_csv(out / "workload.csv", index=False)
print(W[["cohort", "model", "auroc", "ap", "workload_r95", "workload_r100"] + [f"recall_top{int(q * 100)}" for q in QS]].round(3).to_string())

# 3. local recalibration with a separate calibration sample (independent cohort only: 24 APL, 105 AML)
LR = []
for nm in ["Primary pipeline", "DinoBloom + engineered", "Engineered, pooled LR", "Fine-tuned CNN"]:
    d = C[("Independent (MLL)", nm)]; y, p = d.y.values.astype(int), d.p.values
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for npos, nneg in [(5, 20), (8, 30), (12, 50)]:
        res = {"POS": [], "BOTH": []}
        for _ in range(a.draws):
            cp, cn = rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)
            cal = np.r_[cp, cn]; ev = np.setdiff1d(np.arange(len(y)), cal)
            yc, pc, ye, pe = y[cal], p[cal], y[ev], p[ev]
            t_pos = pc[yc == 1].min() - 1e-9
            cand = np.unique(pc); best = None
            for t in cand[::-1]:
                rec = (pc[yc == 1] >= t).mean(); spc = (pc[yc == 0] < t).mean()
                if rec >= 0.9 and (best is None or spc > best[1]):
                    best = (t, spc)
            for rule, t in [("POS", t_pos), ("BOTH", best[0] - 1e-9)]:
                fl = pe >= t
                res[rule].append(((fl & (ye == 1)).sum() / (ye == 1).sum(), ((~fl) & (ye == 0)).sum() / (ye == 0).sum(), fl.mean(), ((~fl) & (ye == 1)).sum() > 0))
        for rule, v in res.items():
            v = np.array(v, float)
            LR.append(dict(model=nm, cal_pos=npos, cal_neg=nneg, rule=rule, recall_mean=v[:, 0].mean(), recall_p05=np.percentile(v[:, 0], 5),
                           spec_mean=v[:, 1].mean(), referred=v[:, 2].mean(), p_any_miss=v[:, 3].mean(), expected_pos_rule=npos / (npos + 1)))
LR = pd.DataFrame(LR); LR.to_csv(out / "local_recal_split.csv", index=False)
print(LR.round(3).to_string())

# 4. exact sample sizes: m consecutive detected positives give a one-sided 95% lower bound 0.05**(1/m) on recall
G = pd.DataFrame([dict(target_recall=r, positives_needed=int(np.ceil(np.log(0.05) / np.log(r))), expected_recall_rule=int(np.ceil(r / (1 - r))))
                  for r in (0.80, 0.90, 0.95)])
G.to_csv(out / "guarantee.csv", index=False); print(G.to_string())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.7))
cols = {"Primary pipeline": "#B03A2E", "DinoBloom + engineered": "#2E7D32", "Engineered, pooled LR": "#2F6DB5", "Fine-tuned CNN": "#8C8C8C", "DinoBloom, attention MIL": "#6A9F58"}
for k, coh in enumerate(["Independent (MLL)", "APL-AML later cohort"]):
    for (c_, nm), d in C.items():
        if c_ != coh:
            continue
        y, p = d.y.values, d.p.values; o = np.argsort(-p, kind="stable")
        ax[k].plot(np.arange(1, len(y) + 1) / len(y), np.cumsum(y[o] == 1) / y.sum(), color=cols[nm], lw=1.5 if nm.startswith("Primary") else 1, label=nm)
    ax[k].plot([0, 1], [0, 1], ":", color="#999", lw=0.8); ax[k].axhline(0.95, color="#B03A2E", ls=":", lw=0.6)
    ax[k].set_xlabel("Fraction of patients referred"); ax[k].set_ylabel("Fraction of APL patients captured")
    ax[k].set_title(f"({'ab'[k]}) {coh}", loc="left", fontsize=7.5); ax[k].legend(frameon=False, fontsize=6.2, loc="lower right")
for (c_, nm), (pr, rc, _) in PR.items():
    if c_ == "Independent (MLL)":
        ax[2].plot(rc, pr, color=cols[nm], lw=1.5 if nm.startswith("Primary") else 1)
ax[2].axhline(24 / 129, color="#999", ls=":", lw=0.8); ax[2].set_xlabel("Recall"); ax[2].set_ylabel("Precision (PPV)"); ax[2].set_ylim(0, 1.02)
ax[2].set_title("(c) Precision-recall (MLL)", loc="left", fontsize=7.5)
fig.tight_layout(); fig.savefig(out / "fig_workload.pdf"); fig.savefig(out / "fig_workload.png", dpi=200)
print("wrote", out)
