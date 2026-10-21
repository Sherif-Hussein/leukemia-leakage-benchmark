"""SHAP explanation of the engineered-feature LightGBM cell classifier (TreeSHAP via LightGBM pred_contrib).

The model is trained on all development cells (same settings as the benchmark); SHAP values are computed for
the external cells. Outputs: mean |SHAP| per feature and per feature group, the direction of each feature's
effect (Spearman between feature value and SHAP value), and a summary figure.

  python scripts/run_shap.py --manifest manifests/cnmc.csv --hand features/cnmc_hand.parquet --label-col diagnosis \
      --positive ALL --dev-where split_role=Development --ext-where split_role=External --out results/shap_cnmc
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--hand", required=True)
ap.add_argument("--label-col", default="diagnosis")
ap.add_argument("--positive", required=True)
ap.add_argument("--dev-where", required=True)
ap.add_argument("--ext-where", required=True)
ap.add_argument("--rows", help="optional boolean column restricting cells (e.g. labeled blasts task)")
ap.add_argument("--title", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
GROUPS = {"Nuclear shape": r"^nuc_(area|perimeter|circularity|eccentricity|solidity|extent|axis_ratio|major_axis)$|^nuc_fragments$",
          "Basophilic region / N:C": r"^bcell_|^nc_ratio_basophilic$|^basophilic_rim$",
          "Stain and colour": r"_(H|E|L|a|b|sat)_(mean|std)$|^H_ratio|^L_diff",
          "Chromatin texture": r"^glcm_|^nuc_entropy$", "Local binary patterns": r"^lbp_", "Nucleoli and granules": r"^nucleoli_|^granule_"}
man = pd.read_csv(a.manifest, keep_default_na=False)
H = pd.read_parquet(a.hand)
k1, v1 = a.dev_where.split("="); k2, v2 = a.ext_where.split("=")
y = (man[a.label_col] == a.positive).astype(int).values
dev, ext = (man[k1] == v1).values, (man[k2] == v2).values
X = H.to_numpy("float32")
clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                     class_weight="balanced", random_state=0, verbose=-1, n_jobs=2).fit(X[dev], y[dev])
S = clf.predict(X[ext], pred_contrib=True)[:, :-1]          # last column = bias
grp = {c: next((g for g, rx in GROUPS.items() if re.search(rx, c)), "Other") for c in H.columns}
imp = pd.DataFrame({"feature": H.columns, "group": [grp[c] for c in H.columns], "mean_abs_shap": np.abs(S).mean(0),
                    "direction": [spearmanr(X[ext][:, j], S[:, j], nan_policy="omit").correlation for j in range(S.shape[1])]})
imp = imp.sort_values("mean_abs_shap", ascending=False); imp.to_csv(out / "shap_features.csv", index=False)
g = imp.groupby("group").mean_abs_shap.sum().sort_values(ascending=False); g.to_csv(out / "shap_groups.csv")
print(imp.head(15).round(3).to_string()); print(g.round(3).to_string())

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.0), gridspec_kw=dict(width_ratios=[1.4, 1]))
top = imp.head(15).iloc[::-1]
cols = ["#B03A2E" if d > 0 else "#2F6DB5" for d in top.direction]
ax[0].barh(range(len(top)), top.mean_abs_shap, color=cols)
ax[0].set_yticks(range(len(top)), top.feature, fontsize=6); ax[0].set_xlabel("Mean |SHAP| (log-odds)")
ax[0].set_title(f"(a) Top features{(' - ' + a.title) if a.title else ''}; red: higher value -> malignant", loc="left", fontsize=7)
ax[1].barh(range(len(g)), g.values[::-1], color="#8C8C8C"); ax[1].set_yticks(range(len(g)), g.index[::-1], fontsize=6.5)
ax[1].set_xlabel("Sum of mean |SHAP|"); ax[1].set_title("(b) Feature groups", loc="left", fontsize=7)
fig.tight_layout(); fig.savefig(out / "fig_shap.pdf"); fig.savefig(out / "fig_shap.png", dpi=200)
print("wrote", out)
