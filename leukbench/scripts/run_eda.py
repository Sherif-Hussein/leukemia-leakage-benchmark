"""Exploratory data analysis for the APL-AML cohort: cohort shift, redundancy and feature stability.

Outputs (CSV + PNG/PDF figures) to --out:
  shift_smd.csv          standardized mean difference Discovery vs Validation per engineered feature,
                         computed within each diagnosis (so class mix does not masquerade as shift)
  cohort_id.csv          how well cells can be assigned to their cohort (AML patients only), per representation
  redundancy.csv         highly correlated engineered-feature pairs (|r| > 0.9)
  univariate_auc.csv     patient-level univariate AUROC (APL vs AML) of each feature in Discovery and Validation
  fig_eda.(png|pdf)      (a) shift by feature group, (b) cohort identifiability, (c) PCA of patients, (d) stability
"""
import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench.models import load_feature_block

GROUPS = {"Nuclear shape": r"^nuc_(area|perimeter|circularity|eccentricity|solidity|extent|axis_ratio|major_axis)$|^nuc_fragments$",
          "Basophilic region": r"^bcell_|^nc_ratio_basophilic$|^basophilic_rim$",
          "Stain and colour": r"_(H|E|L|a|b|sat)_(mean|std)$|^H_ratio|^L_diff",
          "Chromatin texture": r"^glcm_|^nuc_entropy$", "Local binary patterns": r"^lbp_", "Nucleoli and granules": r"^nucleoli_|^granule_"}

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--hand", required=True)
ap.add_argument("--effnet", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})

man = pd.read_csv(a.manifest, keep_default_na=False)
H = pd.read_parquet(a.hand) if a.hand.endswith(".parquet") else pd.read_csv(a.hand)
E = load_feature_block(a.effnet)
group_of = {c: g for g, rx in GROUPS.items() for c in H.columns if re.search(rx, c)}

# ---- 1. shift: SMD Discovery vs Validation within diagnosis (cell level)
rows = []
for c in H.columns:
    smds = []
    for dx in ["AML", "APL"]:
        d = H.loc[(man.cohort == "Discovery") & (man.diagnosis == dx), c].dropna()
        v = H.loc[(man.cohort == "Validation") & (man.diagnosis == dx), c].dropna()
        sp = np.sqrt((d.var() + v.var()) / 2) + 1e-12
        smds.append((v.mean() - d.mean()) / sp)
    rows.append(dict(feature=c, group=group_of[c], smd_aml=smds[0], smd_apl=smds[1], abs_smd=np.mean(np.abs(smds))))
shift = pd.DataFrame(rows).sort_values("abs_smd", ascending=False)
shift.to_csv(out / "shift_smd.csv", index=False)

# ---- 2. cohort identifiability on AML patients only (patient-grouped CV)
aml = man.index[man.diagnosis == "AML"].to_numpy()
rng = np.random.default_rng(0)
aml = np.sort(rng.choice(aml, min(6000, len(aml)), replace=False))
yc = (man.loc[aml, "cohort"] == "Validation").astype(int).values
g = man.loc[aml, "patient_id"].values
cid = []
blocks = {"Engineered (all)": H.to_numpy("float32"),
          "Engineered: stain and colour": H[[c for c in H.columns if group_of[c] == "Stain and colour"]].to_numpy("float32"),
          "Engineered: shape + texture": H[[c for c in H.columns if group_of[c] in ("Nuclear shape", "Chromatin texture", "Local binary patterns")]].to_numpy("float32"),
          "EfficientNet-B0": E}
for name, X in blocks.items():
    clf = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))])
    p = cross_val_predict(clf, X[aml], yc, groups=g, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    cid.append(dict(representation=name, cohort_auroc=roc_auc_score(yc, p)))
    print(f"  cohort identifiability {name:32s} AUROC {cid[-1]['cohort_auroc']:.3f}", flush=True)
cid = pd.DataFrame(cid); cid.to_csv(out / "cohort_id.csv", index=False)

# ---- 3. redundancy
Cm = H.corr().abs()
pairs = [(i, j, Cm.loc[i, j]) for k, i in enumerate(Cm.index) for j in Cm.columns[k + 1:] if Cm.loc[i, j] > 0.9]
pd.DataFrame(pairs, columns=["f1", "f2", "abs_r"]).sort_values("abs_r", ascending=False).to_csv(out / "redundancy.csv", index=False)
print(f"  {len(pairs)} feature pairs with |r| > 0.9 among {H.shape[1]} features", flush=True)

# ---- 4. patient-level univariate AUROC stability
P = H.assign(patient_id=man.patient_id, cohort=man.cohort, y=(man.diagnosis == "APL").astype(int)).groupby("patient_id")
pm = P.mean(numeric_only=True); meta = P[["cohort"]].first()
uni = []
for c in H.columns:
    r = {}
    for coh in ["Discovery", "Validation"]:
        idx = meta.index[meta.cohort == coh]
        x, y = pm.loc[idx, c], pm.loc[idx, "y"]
        r[coh] = roc_auc_score(y, x.fillna(x.median())) if y.nunique() == 2 else np.nan
    uni.append(dict(feature=c, group=group_of[c], auc_discovery=r["Discovery"], auc_validation=r["Validation"]))
uni = pd.DataFrame(uni); uni.to_csv(out / "univariate_auc.csv", index=False)

# ---- figure
fig, ax = plt.subplots(1, 4, figsize=(7.16, 2.2), gridspec_kw=dict(width_ratios=[1.3, 1.25, 1, 1]))
gs = shift.groupby("group").abs_smd.median().sort_values()
ax[0].barh(range(len(gs)), gs.values, color="#8C8C8C")
ax[0].set_yticks(range(len(gs)), gs.index, fontsize=6.3); ax[0].set_xlabel("Median |SMD|, Validation vs Discovery")
ax[0].axvline(0.2, color="#D9822B", lw=0.8, ls="--"); ax[0].set_title("(a) Feature shift", loc="left", fontsize=7.5)
ax[1].barh(range(len(cid)), cid.cohort_auroc, color=["#2F6DB5" if "Eff" in s else "#8C8C8C" for s in cid.representation])
ax[1].set_yticks(range(len(cid)), [s.replace("Engineered: ", "Eng.: ").replace("Engineered (all)", "Eng. (all)") for s in cid.representation], fontsize=6.3)
ax[1].axvline(0.5, color="#555", lw=0.8); ax[1].set_xlim(0.4, 1); ax[1].set_xlabel("AUROC Discovery vs Validation")
ax[1].set_title("(b) Cohort identifiable?", loc="left", fontsize=7.5)
for i, v in enumerate(cid.cohort_auroc):
    ax[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=6.3)
Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(pm[H.columns]))
pc = PCA(2, random_state=0).fit_transform(Z)
for coh, mk in [("Discovery", "o"), ("Validation", "^")]:
    for yv, col in [(0, "#8C8C8C"), (1, "#2F6DB5")]:
        m = (meta.cohort.values == coh) & (pm.y.values == yv)
        ax[2].scatter(pc[m, 0], pc[m, 1], s=10, marker=mk, color=col, alpha=0.8, label=f"{coh[:4]}. {'APL' if yv else 'AML'}")
ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2"); ax[2].legend(fontsize=5.5, frameon=False, loc="best")
ax[2].set_title("(c) Patients (engineered)", loc="left", fontsize=7.5)
ax[3].scatter(uni.auc_discovery, uni.auc_validation, s=7, color="#8C8C8C")
ax[3].plot([0, 1], [0, 1], color="#ccc", lw=0.6); ax[3].axhline(0.5, color="#ccc", lw=0.6); ax[3].axvline(0.5, color="#ccc", lw=0.6)
ax[3].set_xlim(0.15, 0.9); ax[3].set_ylim(0.15, 0.9); ax[3].set_xlabel("AUROC Discovery"); ax[3].set_ylabel("AUROC Validation")
ax[3].set_title("(d) Feature stability", loc="left", fontsize=7.5)
fig.tight_layout(); fig.savefig(out / "fig_eda.pdf"); fig.savefig(out / "fig_eda.png", dpi=220)
r = np.corrcoef(uni.auc_discovery, uni.auc_validation)[0, 1]
print(f"  univariate AUROC Discovery vs Validation correlation r = {r:.2f}; features with AUROC>0.65 in both cohorts: "
      f"{int(((uni.auc_discovery > 0.65) & (uni.auc_validation > 0.65)).sum() + ((uni.auc_discovery < 0.35) & (uni.auc_validation < 0.35)).sum())}")
print(shift.groupby("group").abs_smd.median().round(3).to_string())
print(f"wrote {out}")
