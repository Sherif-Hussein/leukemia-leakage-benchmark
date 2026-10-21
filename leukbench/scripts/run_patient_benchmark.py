"""Patient-level benchmark (E4): diagnose a patient from all of their cells.

Models (spec "NAME=blocks:kind"):
  pool_lr  summary statistics of cell features per patient -> logistic regression
  abmil    gated attention MIL over cells (one bag per patient)

Protocol: stratified K-fold over patients in the CV set. Inside each outer training fold an
inner 80/20 patient split picks the calibration temperature (and the CV threshold). With
--test-where, the external prediction is the average of all fold models over all seeds, and the
operating threshold is chosen on those models' out-of-fold Discovery predictions to reach
sensitivity >= --ext-sens (default 0.95). A Youden-threshold row is reported for reference. Attention weights for external patients are saved with
each cell's path and (if known) its cell type, so diagnostic-cell focus can be audited.

Example (APL vs AML, Discovery -> Validation cohort):
  python scripts/run_patient_benchmark.py --manifest manifests/apl_aml.csv --task apl_patient \
     --features hand=features/apl_aml_hand.parquet effnet=features/apl_aml_effnet.npy \
     --models P1=hand:pool_lr P2=effnet:pool_lr P3=hand+effnet:pool_lr P4=effnet:abmil P5=hand+effnet:abmil \
     --cv-where cohort=Discovery --test-where cohort=Validation --seeds 0 1 2 --out results/e4_apl_patient
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench import metrics as MT
from leukbench.mil import train_abmil
from leukbench.models import make_classifier, stack_blocks
from leukbench.runner import load_blocks, parse_kv, parse_models, save_json
from leukbench.splits import patient_folds
from leukbench.tasks import load_tasks, patient_table, select

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--tasks", default=str(Path(__file__).resolve().parents[1] / "configs" / "tasks.json"))
ap.add_argument("--task", required=True)
ap.add_argument("--features", nargs="+", required=True)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--cv-where", nargs="*")
ap.add_argument("--test-where", nargs="*")
ap.add_argument("--max-cells", type=int, default=500, help="cells sampled per patient bag (0 = all)")
ap.add_argument("--epochs", type=int, default=40, help="MIL epochs; 0 = choose by inner 3-fold patient CV (lowest mean validation loss, up to --max-epochs)")
ap.add_argument("--max-epochs", type=int, default=40)
ap.add_argument("--threshold", default="sens95", choices=["sens95", "youden"])
ap.add_argument("--ext-sens", type=float, default=0.95, help="target sensitivity for the external operating point")
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (small MIL models run fastest on 1)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

torch.set_num_threads(a.threads)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
man = pd.read_csv(a.manifest, keep_default_na=False)
spec = load_tasks(a.tasks)[a.task]
blocks = load_blocks(a.features, len(man))
models = parse_models(a.models)


def bags_for(task):
    pt = patient_table(task)
    rng = np.random.default_rng(0)
    rows = []
    for g in pt["patient_id"]:
        idx = task.index[task.groups == g]
        if a.max_cells and len(idx) > a.max_cells:
            idx = np.sort(rng.choice(idx, a.max_cells, replace=False))
        rows.append(idx)
    return pt, rows


cv = select(man, spec, parse_kv(a.cv_where))
pt_cv, rows_cv = bags_for(cv)
if a.test_where:
    te_task = select(man, spec, parse_kv(a.test_where))
    pt_te, rows_te = bags_for(te_task)
y_cv = pt_cv["y"].to_numpy()
print(f"task {a.task}: {len(pt_cv)} CV patients ({y_cv.sum()} positive), median {int(pt_cv.n_cells.median())} cells"
      + (f"; {len(pt_te)} test patients ({pt_te.y.sum()} positive)" if a.test_where else ""))


def pick_threshold(y, p):
    return MT.threshold_for_sensitivity(y, p, 0.95) if a.threshold == "sens95" else MT.youden_threshold(y, p)


CHOSEN_EPOCHS = []


def choose_epochs(X, rows, y, seed):
    """Nested early stopping: inner 3-fold patient CV inside the training patients; the epoch with the lowest
    mean validation loss (a proper scoring rule) is chosen, then the model is retrained on all of them."""
    from sklearn.model_selection import StratifiedKFold
    curves = []
    for itr, iva in StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        f = train_abmil([X[rows[i]] for i in itr], y[itr], seed=seed, epochs=a.max_epochs,
                        monitor={"val": ([X[rows[i]] for i in iva], y[iva])})
        curves.append([h["val_loss"] for h in f.history])
    return int(np.argmin(np.mean(curves, 0))) + 1


def fit_predict(X, tr_rows, y_tr, eval_sets, kind, seeds):
    """Fit on an inner 80% of training patients; threshold + temperature on the inner 20%.
    eval_sets: list of row-lists. Returns (list of prob arrays, threshold, attention of last set)."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seeds[0])
    itr, iva = next(sss.split(np.zeros(len(y_tr)), y_tr))
    if kind == "pool_lr":
        pool = lambda rows: pooled_features_rows(X, rows)
        clf = make_classifier("lr", seeds[0]).fit(pool([tr_rows[i] for i in itr]), y_tr[itr])
        predict = lambda rows: clf.predict_proba(pool(rows))[:, 1]
        att = None
    else:
        ep = a.epochs or choose_epochs(X, [tr_rows[i] for i in itr], y_tr[itr], seeds[0])
        CHOSEN_EPOCHS.append((mname, ep))
        fns = [train_abmil([X[tr_rows[i]] for i in itr], y_tr[itr], seed=s, epochs=ep) for s in seeds]
        predict = lambda rows: np.mean([f([X[r] for r in rows]) for f in fns], axis=0)
        att = lambda rows: [np.mean(z, axis=0) for z in zip(*[f([X[r] for r in rows], True)[1] for f in fns])]
    p_va = predict([tr_rows[i] for i in iva])
    t = MT.fit_temperature(y_tr[iva], p_va)
    thr = pick_threshold(y_tr[iva], MT.apply_temperature(p_va, t))
    probs = [MT.apply_temperature(predict(rows), t) for rows in eval_sets]
    attention = att(eval_sets[-1]) if att else None
    return probs, thr, attention


def pooled_features_rows(X, rows):
    return np.asarray([np.concatenate([np.nanmean(X[r], 0), np.nanstd(X[r], 0),
                                       *np.nanpercentile(X[r], [10, 50, 90], axis=0)]) for r in rows],
                      dtype="float32")


summary, preds = [], []
for mname, (bnames, kind) in models.items():
    X = stack_blocks(blocks, bnames)
    oofs, ext_probs, ext_att = [], [], []
    for seed in a.seeds:
        t0 = time.time()
        fold = patient_folds(y_cv, a.folds, seed)
        oof, thr_f = np.zeros(len(y_cv)), np.zeros(len(y_cv))
        for k in range(a.folds):
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            sets = [[rows_cv[i] for i in te]] + ([rows_te] if a.test_where else [])
            probs, thr, att = fit_predict(X, [rows_cv[i] for i in tr], y_cv[tr], sets, kind, [seed])
            oof[te], thr_f[te] = probs[0], thr
            if a.test_where:
                ext_probs.append(probs[1])
                if att is not None:
                    ext_att.append(att)
        oofs.append(oof)
        res = MT.bootstrap(y_cv, oof, pt_cv["patient_id"].to_numpy(), thr_f, a.n_boot, seed)
        summary.append(dict(model=mname, blocks="+".join(bnames), kind=kind, seed=seed, eval="cv",
                            patients=len(y_cv), **res))
        preds.append(pd.DataFrame(dict(model=mname, seed=seed, eval="cv", patient_id=pt_cv.patient_id, y=y_cv, p=oof)))
        print(f"  {mname:4s} {kind:7s} seed {seed}  CV AUROC {res['auroc']:.3f} [{res['auroc_lo']:.3f}, "
              f"{res['auroc_hi']:.3f}]  sens {res['sens']:.2f} spec {res['spec']:.2f}  ({time.time() - t0:.0f}s)",
              flush=True)

    if a.test_where:
        # External prediction = average of all fold models over all seeds (cross-validation ensemble).
        # The operating threshold is chosen on the SAME models' out-of-fold Discovery predictions,
        # targeting sensitivity >= --ext-sens: missing an APL patient is the costly error.
        y_te = pt_te["y"].to_numpy()
        oof_mean, p_te = np.mean(oofs, axis=0), np.mean(ext_probs, axis=0)
        thresholds = {"test": MT.threshold_for_sensitivity(y_cv, oof_mean, a.ext_sens),
                      "test_youden": MT.youden_threshold(y_cv, oof_mean)}
        for ev, thr in thresholds.items():
            res = MT.bootstrap(y_te, p_te, pt_te["patient_id"].to_numpy(), thr, a.n_boot, 0)
            summary.append(dict(model=mname, blocks="+".join(bnames), kind=kind, seed=-1, eval=ev, threshold=thr,
                                patients=len(y_te), **res))
            print(f"  {mname:4s} {kind:7s} EXTERNAL[{ev}] AUROC {res['auroc']:.3f} [{res['auroc_lo']:.3f}, {res['auroc_hi']:.3f}]"
                  f"  sens {res['sens']:.2f} (missed {int(res['fn'])}) spec {res['spec']:.2f} F1 {res['f1']:.2f}"
                  f" NPV {res['npv']:.2f} ECE {res['ece']:.3f}", flush=True)
        preds.append(pd.DataFrame(dict(model=mname, seed=-1, eval="test", patient_id=pt_te.patient_id, y=y_te, p=p_te)))
        if ext_att:
            att = [np.mean(z, axis=0) for z in zip(*ext_att)]
            recs = []
            for pid, rows, w in zip(pt_te.patient_id, rows_te, att):
                for r, wi in zip(rows, w):
                    recs.append(dict(patient_id=pid, row=r, path=man.at[r, "path"], cell_type=man.at[r, "cell_type"],
                                     attention=wi, attention_rel=wi * len(rows)))
            pd.DataFrame(recs).to_csv(out / f"attention_{mname}.csv.gz", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)          # save after every model
    pd.concat(preds).to_csv(out / "predictions.csv", index=False)

pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
pd.concat(preds).to_csv(out / "predictions.csv", index=False)
save_json(vars(a), out / "config.json")
if CHOSEN_EPOCHS:
    pd.DataFrame(CHOSEN_EPOCHS, columns=["model", "epochs"]).to_csv(out / "chosen_epochs.csv", index=False)
    print(pd.DataFrame(CHOSEN_EPOCHS, columns=["model", "epochs"]).groupby("model").epochs.describe())
print(f"wrote {out}/summary.csv")
