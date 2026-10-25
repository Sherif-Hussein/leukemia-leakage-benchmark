"""Recompute the manuscript's headline external results from the released patient-level predictions.

No images or features are needed. Each line prints the recomputed value next to the value reported in the paper.
  python verify_claims.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

P = Path(__file__).resolve().parent / "leukbench" / "release" / "predictions"
TH = pd.read_csv(P / "thresholds.csv").set_index(["cohort", "model"]).development_threshold
ok = True


def load(coh, m):
    d = pd.read_csv(P / f"{coh}__{m}.csv"); return d.y.values.astype(int), d.score.values


def check(label, value, reported, tol=0.0015):
    global ok
    good = abs(value - reported) <= tol
    ok &= good
    print(f"{'OK ' if good else 'DIFF'} {label:72s} recomputed {value:.3f}   paper {reported:.3f}")


def at_threshold(coh, m):
    y, p = load(coh, m); t = TH[(coh, m)]; f = p >= t
    return (f & (y == 1)).sum() / (y == 1).sum(), (~f & (y == 0)).sum() / (y == 0).sum()


def top(y, p, q):
    k = int(np.ceil(q * len(y))); i = np.argsort(-p, kind="stable")[:k]; return y[i].sum() / y.sum()


def workload(y, p, r):
    o = np.argsort(-p, kind="stable"); c = np.cumsum(y[o]) / y.sum(); return (np.argmax(c >= r - 1e-12) + 1) / len(y)


print("Independent MLL cohort (24 APL, 105 other AML), models trained on APL-AML Discovery only")
y, p = load("mll", "primary_ensemble")
check("Primary pipeline AUROC", roc_auc_score(y, p), 0.862)
r, s = at_threshold("mll", "primary_ensemble"); check("Primary pipeline recall at development threshold", r, 1.000); check("Primary pipeline specificity at development threshold", s, 0.000)
check("Primary pipeline: APL captured in top 20% (15/24)", top(y, p, 0.20), 0.625)
check("Primary pipeline: workload for 95% recall", workload(y, p, 0.95), 0.605, 0.005)
y, p = load("mll", "dinobloom_engineered")
check("DinoBloom + engineered AUROC", roc_auc_score(y, p), 0.877)
r, s = at_threshold("mll", "dinobloom_engineered"); check("DinoBloom + engineered recall at development threshold", r, 1.000); check("DinoBloom + engineered specificity", s, 0.048, 0.005)
check("DinoBloom + engineered: APL captured in top 20% (17/24)", top(y, p, 0.20), 0.708, 0.002)
check("DinoBloom + engineered: workload for 95% recall", workload(y, p, 0.95), 0.752, 0.005)
check("Engineered pooled LR AUROC", roc_auc_score(*load("mll", "engineered_pooled_lr")), 0.832)
check("Fine-tuned CNN AUROC", roc_auc_score(*load("mll", "finetuned_cnn")), 0.737)
print("\nAPL-AML later (Validation) cohort, 12 APL / 12 AML (exploratory, reused test set)")
check("Primary pipeline AUROC", roc_auc_score(*load("apl_validation", "primary_ensemble")), 0.806)
r, s = at_threshold("apl_validation", "primary_ensemble"); check("Primary pipeline recall", r, 1.000); check("Primary pipeline specificity", s, 0.167, 0.002)
check("DinoBloom attention MIL AUROC", roc_auc_score(*load("apl_validation", "dinobloom_attention_mil")), 0.868)
print("\nC-NMC fold 2 (17 ALL, 14 normal subjects)")
check("Primary pipeline AUROC", roc_auc_score(*load("cnmc_fold2", "primary_ensemble")), 0.960)
check("DinoBloom + engineered AUROC", roc_auc_score(*load("cnmc_fold2", "dinobloom_engineered")), 0.962)
print("\nReverse transfer (exploratory): trained on MLL, tested on all 106 APL-AML patients")
check("DinoBloom + engineered AUROC", roc_auc_score(*load("apl_all_reverse", "dinobloom_engineered_trained_on_mll")), 0.776)
print("\nALL CLAIMS REPRODUCED" if ok else "\nSOME VALUES DIFFER - see DIFF lines")
