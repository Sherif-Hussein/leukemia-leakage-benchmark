"""Cell-level benchmark: E1 (leakage audit), E2 (representations), E3 (cross-lab).

For every model x split type (image-level / patient-grouped) x seed:
  outer 5-fold CV over the CV rows. Inside each outer training fold, an inner 80/20 split
  (same split type) picks the decision threshold and the calibration temperature; the model
  is fit on the inner 80%. Out-of-fold predictions are pooled and scored with
  patient-level bootstrap CIs.
Optionally (--test-where), each model is also trained on all CV rows and scored once on a
held-out test set (held-out subjects, another cohort, or another dataset).

Example (E1 + E2 on the APL-AML cells):
  python scripts/run_cell_benchmark.py --manifest manifests/apl_aml.csv --task apl_cell_diagnosis \
     --features hand=features/apl_aml_hand.parquet effnet=features/apl_aml_effnet.npy \
     --models M1=hand:lgbm M2=effnet:lr M4=hand+effnet:lr \
     --splits image grouped --seeds 0 1 2 --cv-where cohort=Discovery --test-where cohort=Validation \
     --out results/e1_apl_cell_diagnosis
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench import metrics as MT
from leukbench.models import make_classifier, stack_blocks
from leukbench.runner import load_blocks, parse_kv, parse_models, save_json
from leukbench.splits import check_no_patient_overlap, make_folds
from leukbench.tasks import load_tasks, select

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--tasks", default=str(Path(__file__).resolve().parents[1] / "configs" / "tasks.json"))
ap.add_argument("--task", required=True)
ap.add_argument("--features", nargs="+", required=True)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--splits", nargs="+", default=["image", "grouped"], choices=["image", "grouped"])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--cv-where", nargs="*")
ap.add_argument("--test-where", nargs="*")
ap.add_argument("--threshold", default="sens95", choices=["sens95", "youden"])
ap.add_argument("--max-train-cells", type=int, default=0, help="subsample training cells (0 = all) for speed")
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
man = pd.read_csv(a.manifest, keep_default_na=False)
spec = load_tasks(a.tasks)[a.task]
blocks = load_blocks(a.features, len(man))
models = parse_models(a.models)
cv = select(man, spec, parse_kv(a.cv_where))
test = select(man, spec, parse_kv(a.test_where)) if a.test_where else None
print(f"task {a.task}: CV rows {len(cv.y):,} ({cv.y.mean():.1%} positive, {len(np.unique(cv.groups))} patients)"
      + (f"; test rows {len(test.y):,} ({len(np.unique(test.groups))} patients)" if test else ""))


def pick_threshold(y, p):
    return MT.threshold_for_sensitivity(y, p, 0.95) if a.threshold == "sens95" else MT.youden_threshold(y, p)


def fit_predict(X_tr, y_tr, g_tr, X_list, kind, clf_kind, seed):
    """Inner split -> fit, threshold, temperature. Returns calibrated probs for each X in X_list."""
    if a.max_train_cells and len(y_tr) > a.max_train_cells:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(y_tr), a.max_train_cells, replace=False))
        X_tr, y_tr, g_tr = X_tr[keep], y_tr[keep], g_tr[keep]
    if kind == "grouped":
        inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, va = next(inner.split(X_tr, y_tr, g_tr))
        if len(np.unique(y_tr[va])) < 2:                       # rare: retry with another seed
            tr, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed + 99).split(X_tr, y_tr, g_tr))
    else:
        tr, va = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X_tr, y_tr))
    clf = make_classifier(clf_kind, seed).fit(X_tr[tr], y_tr[tr])
    p_va = clf.predict_proba(X_tr[va])[:, 1]
    t = MT.fit_temperature(y_tr[va], p_va)
    thr = pick_threshold(y_tr[va], MT.apply_temperature(p_va, t))
    return [MT.apply_temperature(clf.predict_proba(X)[:, 1], t) for X in X_list], thr


summary, preds = [], []
for mname, (bnames, clf_kind) in models.items():
    X_all = stack_blocks(blocks, bnames)
    X_cv = X_all[cv.index]
    for kind in a.splits:
        for seed in a.seeds:
            t0 = time.time()
            fold = make_folds(cv.y, cv.groups, kind, a.folds, seed)
            overlap = check_no_patient_overlap(cv.groups, fold)
            oof = np.zeros(len(cv.y))
            thr_f = np.zeros(len(cv.y))
            for k in range(a.folds):
                te, tr = fold == k, fold != k
                (p,), thr = fit_predict(X_cv[tr], cv.y[tr], cv.groups[tr], [X_cv[te]], kind, clf_kind, seed)
                oof[te], thr_f[te] = p, thr
            # sens/spec use each fold's own validation threshold; AUROC is threshold-free
            res = MT.bootstrap(cv.y, oof, cv.groups, thr=thr_f, n_boot=a.n_boot, seed=seed)
            summary.append(dict(model=mname, blocks="+".join(bnames), clf=clf_kind, split=kind, seed=seed,
                                eval="cv", n=len(cv.y), patients=len(np.unique(cv.groups)),
                                patients_in_several_folds=overlap, **res))
            preds.append(pd.DataFrame(dict(model=mname, split=kind, seed=seed, eval="cv", row=cv.index,
                                           patient_id=cv.groups, y=cv.y, p=oof, thr=thr_f)))
            print(f"  {mname:4s} {kind:8s} seed {seed}  CV AUROC {res['auroc']:.3f} "
                  f"[{res['auroc_lo']:.3f}, {res['auroc_hi']:.3f}]  patients split across folds: {overlap}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)

            if test is not None:
                (p_te,), thr = fit_predict(X_cv, cv.y, cv.groups, [X_all[test.index]], kind, clf_kind, seed)
                res = MT.bootstrap(test.y, p_te, test.groups, thr, n_boot=a.n_boot, seed=seed)
                summary.append(dict(model=mname, blocks="+".join(bnames), clf=clf_kind, split=kind, seed=seed,
                                    eval="test", n=len(test.y), patients=len(np.unique(test.groups)),
                                    patients_in_several_folds=0, **res))
                preds.append(pd.DataFrame(dict(model=mname, split=kind, seed=seed, eval="test", row=test.index,
                                               patient_id=test.groups, y=test.y, p=p_te, thr=thr)))
                print(f"  {mname:4s} {kind:8s} seed {seed}  TEST AUROC {res['auroc']:.3f} "
                      f"[{res['auroc_lo']:.3f}, {res['auroc_hi']:.3f}]  sens {res['sens']:.3f} spec {res['spec']:.3f}",
                      flush=True)

pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
pd.concat(preds).to_csv(out / "predictions.csv.gz", index=False)
save_json(vars(a), out / "config.json")
print(f"wrote {out}/summary.csv and predictions.csv.gz")
