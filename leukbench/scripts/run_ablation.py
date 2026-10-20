"""Ablations that show where performance comes from (external cohort, patient-grouped protocol).

1. Engineered feature groups: "all", "all minus group G", and "group G only" with logistic
   regression; external AUROC with patient bootstrap CI and paired difference vs "all".
2. Calibration: external ECE / Brier before and after temperature scaling.
3. LightGBM gain importance of engineered features.

  python scripts/run_ablation.py --manifest manifests/apl_aml.csv --task apl_cell_blast \
     --hand features/apl_aml_hand.parquet --effnet features/apl_aml_effnet.npy \
     --cv-where cohort=Discovery --test-where cohort=Validation --out results/ablation_blast
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench import metrics as MT
from leukbench.models import load_feature_block, make_classifier
from leukbench.runner import parse_kv
from leukbench.tasks import load_tasks, select

GROUPS = {
    "Nuclear shape": r"^nuc_(area|perimeter|circularity|eccentricity|solidity|extent|axis_ratio|major_axis)$|^nuc_fragments$",
    "Basophilic region": r"^bcell_|^nc_ratio_basophilic$|^basophilic_rim$",
    "Stain and colour": r"_(H|E|L|a|b|sat)_(mean|std)$|^H_ratio|^L_diff",
    "Chromatin texture (GLCM, entropy)": r"^glcm_|^nuc_entropy$",
    "Local binary patterns": r"^lbp_",
    "Nucleoli and granules": r"^nucleoli_|^granule_",
}

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--tasks", default=str(Path(__file__).resolve().parents[1] / "configs" / "tasks.json"))
ap.add_argument("--task", required=True)
ap.add_argument("--hand", required=True)
ap.add_argument("--effnet")
ap.add_argument("--cv-where", nargs="*")
ap.add_argument("--test-where", nargs="*")
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
man = pd.read_csv(a.manifest, keep_default_na=False)
spec = load_tasks(a.tasks)[a.task]
H = pd.read_parquet(a.hand) if a.hand.endswith(".parquet") else pd.read_csv(a.hand)
cols = list(H.columns)
Hx = H.to_numpy("float32")
cv = select(man, spec, parse_kv(a.cv_where))
te = select(man, spec, parse_kv(a.test_where))
assigned = {c: g for g, rx in GROUPS.items() for c in cols if re.search(rx, c)}
missing = [c for c in cols if c not in assigned]
if missing:
    raise SystemExit(f"features not assigned to a group: {missing}")

tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(cv.index, cv.y, cv.groups))


def fit_eval(X, kind="lr"):
    """Fit on inner 80% of Discovery, temperature + threshold on inner 20%, score external."""
    Xc = X[cv.index]
    clf = make_classifier(kind, 0).fit(Xc[tr_i], cv.y[tr_i])
    p_va = clf.predict_proba(Xc[va_i])[:, 1]
    t = MT.fit_temperature(cv.y[va_i], p_va)
    thr = MT.threshold_for_sensitivity(cv.y[va_i], MT.apply_temperature(p_va, t), 0.95)
    raw = clf.predict_proba(X[te.index])[:, 1]
    return clf, raw, MT.apply_temperature(raw, t), thr


rows, preds = [], {}
_, _, p_all, thr = fit_eval(Hx)
preds["all"] = p_all
base = MT.bootstrap(te.y, p_all, te.groups, thr, a.n_boot)
rows.append(dict(setting="All engineered features", n_features=len(cols), **{k: base[k] for k in ["auroc", "auroc_lo", "auroc_hi"]},
                 delta=0.0, delta_lo=0.0, delta_hi=0.0, p=1.0))
for g in GROUPS:
    gi = [i for i, c in enumerate(cols) if assigned[c] == g]
    for label, keep in [(f"without {g}", [i for i in range(len(cols)) if i not in gi]), (f"{g} only", gi)]:
        _, _, p, thr = fit_eval(Hx[:, keep])
        r = MT.bootstrap(te.y, p, te.groups, thr, a.n_boot)
        d = MT.paired_bootstrap_auc(te.y, p, p_all, te.groups, a.n_boot)
        rows.append(dict(setting=label, n_features=len(keep), auroc=r["auroc"], auroc_lo=r["auroc_lo"],
                         auroc_hi=r["auroc_hi"], delta=d["delta"], delta_lo=d["lo"], delta_hi=d["hi"], p=d["p"]))
        print(f"  {label:45s} AUROC {r['auroc']:.3f}  delta vs all {d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] p={d['p']:.3f}", flush=True)
pd.DataFrame(rows).to_csv(out / "feature_groups.csv", index=False)

# calibration before / after temperature scaling
cal = []
blocks = {"Engineered (LR)": (Hx, "lr"), "Engineered (LightGBM)": (Hx, "lgbm")}
if a.effnet:
    E = load_feature_block(a.effnet)
    blocks["EfficientNet-B0 (LR)"] = (E, "lr")
    blocks["Fusion (LR)"] = (np.hstack([Hx, E]), "lr")
for name, (X, kind) in blocks.items():
    clf, raw, calp, _ = fit_eval(X, kind)
    cal.append(dict(model=name, ece_raw=MT.ece(te.y, raw), ece_scaled=MT.ece(te.y, calp),
                    brier_raw=float(np.mean((raw - te.y) ** 2)), brier_scaled=float(np.mean((calp - te.y) ** 2))))
    print(f"  calibration {name:25s} ECE {cal[-1]['ece_raw']:.3f} -> {cal[-1]['ece_scaled']:.3f}", flush=True)
    if kind == "lgbm":
        imp = clf.named_steps["clf"].booster_.feature_importance("gain")
        pd.DataFrame(dict(feature=cols, group=[assigned[c] for c in cols], gain=imp / imp.sum())) \
            .sort_values("gain", ascending=False).to_csv(out / "lgbm_importance.csv", index=False)
pd.DataFrame(cal).to_csv(out / "calibration.csv", index=False)
print(f"wrote {out}")
