"""Metrics with patient-level uncertainty.

All confidence intervals resample PATIENTS (groups), not cells: cells from one patient are
correlated, so cell-level resampling gives intervals that are far too narrow.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.95) -> float:
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k]) - 1e-9


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def point_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """Threshold metrics. sens = recall; fn = missed positives (the clinically dangerous error)."""
    yhat = (p >= thr).astype(int)
    tp = ((yhat == 1) & (y == 1)).sum()
    tn = ((yhat == 0) & (y == 0)).sum()
    fp = ((yhat == 1) & (y == 0)).sum()
    fn = ((yhat == 0) & (y == 1)).sum()
    sens = tp / max(tp + fn, 1)                    # recall: share of sick patients/cells detected
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)                     # precision
    npv = tn / max(tn + fn, 1)                     # how far a negative call can be trusted
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
    out = dict(sens=sens, spec=spec, ppv=ppv, npv=npv, f1=f1, bal_acc=(sens + spec) / 2, acc=(tp + tn) / len(y),
               fn=fn, fp=fp, tp=tp, tn=tn)
    out["auroc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    out["ece"] = ece(y, p)
    out["brier"] = brier_score_loss(y, p) if len(np.unique(y)) == 2 else np.nan
    return {k: float(v) for k, v in out.items()}


def bootstrap(y, p, groups, thr, n_boot: int = 1000, seed: int = 0) -> dict:
    """Point estimates + 95% CI by resampling groups (patients) with replacement.

    thr may be a scalar or a per-row array (e.g. each CV fold's own validation threshold)."""
    thr = np.asarray(thr, dtype=float)
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(ug))]
    point = point_metrics(y, p, thr)
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([members[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        for k, v in point_metrics(y[idx], p[idx], thr if thr.ndim == 0 else thr[idx]).items():
            draws[k].append(v)
    out = {}
    for k, v in point.items():
        lo, hi = (np.percentile(draws[k], [2.5, 97.5]) if draws[k] else (np.nan, np.nan))
        out[k], out[k + "_lo"], out[k + "_hi"] = v, float(lo), float(hi)
    return out


def paired_bootstrap_auc(y, p1, p2, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference in AUROC (model 1 - model 2) with a patient-level paired bootstrap CI and p-value."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(ug))]
    d0 = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    ds = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))])
        if len(np.unique(y[idx])) < 2:
            continue
        ds.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    ds = np.asarray(ds)
    pval = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return dict(delta=float(d0), lo=float(np.percentile(ds, 2.5)), hi=float(np.percentile(ds, 97.5)),
                p=float(min(pval, 1.0)))


# ---------------------------------------------------------------- calibration
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_temperature(y_val: np.ndarray, p_val: np.ndarray) -> float:
    z = _logit(p_val)

    def nll(t):
        q = 1 / (1 + np.exp(-z / t))
        q = np.clip(q, 1e-7, 1 - 1e-7)
        return -np.mean(y_val * np.log(q) + (1 - y_val) * np.log(1 - q))

    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return 1 / (1 + np.exp(-_logit(p) / t))


def coverage_curve(y, p, levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)):
    """Selective prediction: keep cases whose confidence max(p, 1-p) >= level."""
    conf = np.maximum(p, 1 - p)
    yhat = (p >= 0.5).astype(int)
    rows = []
    for c in levels:
        m = conf >= c
        rows.append(dict(conf_level=c, coverage=float(m.mean()),
                         accuracy=float((yhat[m] == y[m]).mean()) if m.any() else np.nan))
    return rows
