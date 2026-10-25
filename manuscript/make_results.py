"""Build LaTeX tables and result figures for the manuscript directly from LeukBench outputs (v2:
sensitivity-first operating points, recall/F1/NPV/missed cases, same-cohort comparison)."""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "leukbench"))
from leukbench.metrics import paired_bootstrap_auc  # noqa: E402

import os
R = Path(os.environ.get("LEUKBENCH_WORK", Path(__file__).resolve().parents[1])) / "results"
OUT = Path(__file__).resolve().parent
TAB = OUT / "tables"
TAB.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
GREY, BLUE, ORANGE, GREEN, RED = "#8C8C8C", "#2F6DB5", "#D9822B", "#2E7D32", "#B03A2E"
NAMES = {"M1": "Engineered + LightGBM", "M1lr": "Engineered + LR", "M2": "EfficientNet-B0 + LR", "M4": "Fusion + LR"}
TASKS = {"apl_cell_diagnosis": "T1 Diagnosis on cells", "apl_cell_blast": "T2 Blast detection"}
facts = {}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaN kept)."""
    p = np.asarray(pvals, float); m = ~np.isnan(p); adj = np.full_like(p, np.nan)
    idx = np.flatnonzero(m)[np.argsort(p[m])]; n = len(idx); run = 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (n - r) * p[i])); adj[i] = run
    return adj


def cpr(rate, n, d=2):
    """rate with exact (Clopper-Pearson) 95% CI from a proportion of n."""
    from scipy.stats import beta
    k = int(round(rate * n)); lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return f"{rate:.{d}f} ({lo:.{d}f}--{hi:.{d}f})"


def npv_s(npv, fn, spec, d=2):
    """NPV is undefined when nothing is called negative."""
    return "n/a" if (spec == 0 and fn == 0) else f"{npv:.{d}f}"


def ci(v, lo, hi, d=3):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ================================================================== cell level
cell = []
for t in TASKS:
    s = pd.read_csv(R / f"v2_cell_{t}" / "summary.csv")
    s["task"] = t
    cell.append(s)
cell = pd.concat(cell)
agg = cell.groupby(["task", "model", "split", "eval"]).agg(
    auroc=("auroc", "mean"), lo=("auroc_lo", "mean"), hi=("auroc_hi", "mean"), sens=("sens", "mean"), spec=("spec", "mean"),
    f1=("f1", "mean"), npv=("npv", "mean"), acc=("acc", "mean"), ece=("ece", "mean"), seeds=("seed", "nunique")).reset_index()
rows = []
for t, tname in TASKS.items():
    for m in NAMES:
        a = agg[(agg.task == t) & (agg.model == m)].set_index(["split", "eval"])
        ic, gc, ex = a.loc[("image", "cv")], a.loc[("grouped", "cv")], a.loc[("grouped", "test")]
        rows.append(dict(task=tname, model=NAMES[m], image_cv=ic.auroc, grouped_cv=gc.auroc, ext=ex.auroc, ext_lo=ex.lo,
                         ext_hi=ex.hi, sens=ex.sens, spec=ex.spec, f1=ex.f1, npv=ex.npv, acc=ex.acc, ece=ex.ece,
                         image_cv_acc=ic.acc, infl=ic.auroc - ex.auroc))
leak = pd.DataFrame(rows)
leak.to_csv(TAB / "leakage.csv", index=False)
facts["leak"] = leak.round(4).to_dict("records")

# ---- leakage vs shift decomposition, per seed, all cell-level tasks (both datasets)
GB = pd.read_csv(R / "gap_bootstrap" / "gap_bootstrap.csv") if (R / "gap_bootstrap" / "gap_bootstrap.csv").exists() else None


def decomp(summary_csv, task_label, models):
    s_ = pd.read_csv(summary_csv); out_ = []
    for m in models:
        q = s_[s_.model == m]
        if q.empty:
            continue
        per = []
        for sd in sorted(q.seed.unique()):
            g = q[q.seed == sd].set_index(["split", "eval"])
            if ("image", "test") not in g.index:
                continue
            per.append(dict(ic=g.loc[("image", "cv"), "auroc"], gc=g.loc[("grouped", "cv"), "auroc"], ex=g.loc[("grouped", "test"), "auroc"],
                            lx=g.loc[("image", "test"), "auroc"], lo=g.loc[("grouped", "test"), "auroc_lo"], hi=g.loc[("grouped", "test"), "auroc_hi"],
                            sens=g.loc[("grouped", "test"), "sens"], spec=g.loc[("grouped", "test"), "spec"], f1=g.loc[("grouped", "test"), "f1"]))
        P = pd.DataFrame(per)
        out_.append(dict(task=task_label, model=NAMES.get(m, m), image_cv=P.ic.mean(), grouped_cv=P.gc.mean(), ext=P.ex.mean(), lo=P.lo.mean(), hi=P.hi.mean(),
                         leaky_trained_ext=P.lx.mean(), leakage=(P.ic - P.gc).mean(), leakage_sd=(P.ic - P.gc).std(), shift=(P.gc - P.ex).mean(),
                         shift_sd=(P.gc - P.ex).std(), total=(P.ic - P.ex).mean(), sens=P.sens.mean(), spec=P.spec.mean(), f1=P.f1.mean(), seeds=len(P)))
        g_ = GB[(GB.task.str[:2] == task_label[:2]) & (GB.model == m)] if GB is not None else pd.DataFrame()
        out_[-1].update(dict(leak_lo=g_.leak_lo.iloc[0], leak_hi=g_.leak_hi.iloc[0], shift_lo=g_.shift_lo.iloc[0], shift_hi=g_.shift_hi.iloc[0]) if len(g_) else
                        dict(leak_lo=np.nan, leak_hi=np.nan, shift_lo=np.nan, shift_hi=np.nan))
    return out_


DEC = []
for t, lab in [("apl_cell_diagnosis", "T1 APL-AML: diagnosis on cells (patient label)"), ("apl_cell_blast", "T2 APL-AML: blast detection (cell label)")]:
    DEC += decomp(R / f"v2_cell_{t}" / "summary.csv", lab, ["M1", "M1lr", "M2", "M4"])
if (R / "cnmc_cell" / "summary.csv").exists():
    DEC += decomp(R / "cnmc_cell" / "summary.csv", "T4 C-NMC: cells (subject label)", ["M1lr", "M2", "M4"])
DEC = pd.DataFrame(DEC); DEC.to_csv(TAB / "leakage_decomposition.csv", index=False); facts["leakage_decomposition"] = DEC.round(4).to_dict("records")
L = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{How much of the gap between reported and real performance is due to leakage and how much to the external generalization gap (cell level, mean over seeds; 95\% CIs of the two components from 500 patient-level bootstrap resamples: paired over development patients for leakage, development and external patients resampled independently for the gap). Leakage = image-level CV AUROC $-$ patient-grouped CV AUROC (same development data, only the split differs). External generalization gap = patient-grouped CV AUROC $-$ external AUROC (new patients; temporal for APL-AML, a held-out subject fold for C-NMC); it mixes all differences between cohorts and is not a causal effect of any single one. Leaky-trained external: external AUROC of the models trained under image-level folds. Recall, specificity and F1 on the external set at the operating point fixed on development data.}",
     r"\label{tab:leakage}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llcccccccccc}", r"\toprule",
     r" & & \multicolumn{4}{c}{AUROC} & \multicolumn{2}{c}{Decomposition of the gap} & \multicolumn{3}{c}{External, fixed operating point} \\",
     r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}",
     r"Task & Model & Image CV & Grouped CV & External [95\% CI] & Leaky-trained ext. & Leakage & Ext.\ gap & Recall & Spec. & F1 \\", r"\midrule"]
for t, grp in DEC.groupby("task", sort=False):
    best = grp.ext.max()
    for j, r in enumerate(grp.itertuples()):
        e = ci(r.ext, r.lo, r.hi); e = r"\textbf{" + e + "}" if r.ext == best else e
        L.append(f"{t if j == 0 else ''} & {r.model} & {r.image_cv:.3f} & {r.grouped_cv:.3f} & {e} & {r.leaky_trained_ext:.3f} & "
                 f"{r.leakage:+.3f} [{r.leak_lo:+.3f}, {r.leak_hi:+.3f}] & {r.shift:+.3f} [{r.shift_lo:+.3f}, {r.shift_hi:+.3f}] & {r.sens:.2f} & {r.spec:.2f} & {r.f1:.2f} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}}", r"\end{table*}"]
(TAB / "tab_leakage.tex").write_text("\n".join(L))
print(DEC[["task", "model", "image_cv", "grouped_cv", "ext", "leaky_trained_ext", "leakage", "shift", "total"]].round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True)
order = list(NAMES.values())
for k, (ax, (t, tname)) in enumerate(zip(axes, TASKS.items())):
    sub = leak[leak.task == tname].set_index("model").loc[order]
    y = np.arange(len(order))[::-1]
    for yi, r in zip(y, sub.itertuples()):
        ax.plot([r.ext, r.image_cv], [yi, yi], color="#C9CED6", lw=2, zorder=1)
        ax.errorbar(r.ext, yi, xerr=[[r.ext - r.ext_lo], [r.ext_hi - r.ext]], fmt="none", ecolor=BLUE, elinewidth=0.8, capsize=2)
        ax.text(r.image_cv + 0.006, yi + 0.18, f"L {r.image_cv - r.grouped_cv:+.2f} / G {r.grouped_cv - r.ext:+.2f}", fontsize=6.0, color=ORANGE)
    ax.scatter(sub.image_cv, y, s=26, color=ORANGE, zorder=3, label="Image-level CV")
    ax.scatter(sub.grouped_cv, y, s=26, color=GREY, marker="s", zorder=3, label="Patient-grouped CV")
    ax.scatter(sub.ext, y, s=30, color=BLUE, marker="D", zorder=3, label="External cohort (95% CI)")
    ax.set_yticks(y, order); ax.set_xlim(0.45, 1.03); ax.axvline(1.0, color="#DDD", lw=0.6)
    ax.set_xlabel("AUROC"); ax.set_title(f"({'ab'[k]}) {tname}", loc="left", fontsize=8)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=6.8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "figures/fig_leakage.pdf"); fig.savefig(OUT / "figures/fig_leakage.png", dpi=220)

# ================================================================== patient level
ps = pd.read_csv(R / "v2_patient" / "summary.csv")
pmap = {"P1": ("Engineered", "Pooled + LR"), "P2": ("EfficientNet-B0", "Pooled + LR"), "P3": ("Fusion", "Pooled + LR"),
        "P4": ("Engineered", "Attention MIL"), "P5": ("EfficientNet-B0", "Attention MIL"), "P6": ("Fusion", "Attention MIL")}
prow = []
for m, (rep, ag) in pmap.items():
    cvv = ps[(ps.model == m) & (ps["eval"] == "cv")]
    ex = ps[(ps.model == m) & (ps["eval"] == "test")].iloc[0]
    yo = ps[(ps.model == m) & (ps["eval"] == "test_youden")].iloc[0]
    prow.append(dict(model=m, rep=rep, agg=ag, cv=cvv.auroc.mean(), cv_sd=cvv.auroc.std(), ext=ex.auroc, lo=ex.auroc_lo, hi=ex.auroc_hi,
                     sens=ex.sens, fn=int(ex.fn), fp=int(ex.fp), spec=ex.spec, f1=ex.f1, npv=ex.npv, acc=ex.acc, ece=ex.ece,
                     y_sens=yo.sens, y_fn=int(yo.fn), y_spec=yo.spec, y_f1=yo.f1, y_acc=yo.acc))
pt = pd.DataFrame(prow)
pt.to_csv(TAB / "patient.csv", index=False)
facts["patient"] = pt.round(4).to_dict("records")
L = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{Patient-level APL detection (T3). CV: Discovery, mean $\pm$ SD over three seeds. External: Validation cohort (12 APL, 12 AML) at the operating point fixed on Discovery (target sensitivity 0.95); the last three columns use a Youden threshold for reference.}",
     r"\label{tab:patient}", r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{llcccccccccc}", r"\toprule",
     r" & & & \multicolumn{6}{c}{External, sensitivity-first operating point} & \multicolumn{3}{c}{External, Youden threshold} \\",
     r"\cmidrule(lr){4-9}\cmidrule(lr){10-12}",
     r"Features & Aggregation & CV AUROC & AUROC [95\% CI] & Recall (missed) & Spec. & F1 & NPV & ECE & Recall (missed) & Spec. & Acc. \\", r"\midrule"]
best = pt.ext.max()
for r in pt.itertuples():
    e = ci(r.ext, r.lo, r.hi)
    e = r"\textbf{" + e + "}" if r.ext == best else e
    L.append(f"{r.rep} & {r.agg} & {r.cv:.3f}$\\pm${r.cv_sd:.3f} & {e} & {r.sens:.2f} ({r.fn}/12) & {r.spec:.2f} & {r.f1:.2f} & {npv_s(r.npv, r.fn, r.spec)} & {r.ece:.3f} & {r.y_sens:.2f} ({r.y_fn}/12) & {r.y_spec:.2f} & {r.y_acc:.2f} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(TAB / "tab_patient.tex").write_text("\n".join(L))

# ================================================================== same-cohort comparison
bl = pd.read_csv(R / "v2_baselines" / "baselines.csv")
facts["baselines"] = bl.round(4).to_dict("records")
best_m = pt.sort_values("ext", ascending=False).iloc[0]
C = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{Comparison with existing methods on the same APL-AML cohort. Upper block: results as reported by the original studies, whose protocols differ from ours. Lower blocks: methods re-run on the identical split (trained on Discovery, tested once on the 24 Validation patients) with the same operating-point rule (sensitivity $\geq$ 0.95 on Discovery). The primary pipeline is compared with each of its members. Recall and specificity with exact (Clopper--Pearson) 95\% CIs over 12 patients each; NPV n/a: no patient called negative.}",
     r"\label{tab:compare}", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{p{4.6cm}p{3.6cm}ccccccc}", r"\toprule",
     r"Method & Evaluation & AUROC [95\% CI] & Recall (95\% CI) [missed] & Spec. (95\% CI) & F1 & NPV & Acc. \\", r"\midrule",
     r"\multicolumn{8}{l}{\emph{As reported in the literature}} \\",
     r"Single-cell CNN, Sidhom \emph{et al.} \cite{sidhom2021apl} & Temporal validation, single-cell classifier & 0.743 & -- & -- & -- & -- & -- \\",
     r"Annotation-free MIL, Manescu \emph{et al.} \cite{manescu2023apl} & Random 3-fold CV, patient level & 0.96 $\pm$ 0.02 & -- & -- & -- & -- & -- \\",
     r"Ten clinicians (reader study in \cite{sidhom2021apl}) & Same 24 Validation patients, smear images & \multicolumn{6}{l}{shown only graphically; the authors' MIL model was equivalent or better} \\",
     r"\midrule", r"\multicolumn{8}{l}{\emph{Re-run on the identical split (Validation cohort, patient level)}} \\"]
labels = {"Promyelocyte proportion": r"Promyelocyte proportion (biomarker in \cite{sidhom2021apl})",
          "EfficientNet-B0 features + RBF-SVM, cell mean": r"CNN features + SVM (design of \cite{kasim2025multiclass})",
          "Engineered + LR, cell mean": r"Single-cell classifier, engineered (design of \cite{sidhom2021apl})",
          "EfficientNet-B0 + LR, cell mean": r"Single-cell classifier, EfficientNet-B0 (design of \cite{sidhom2021apl})",
          "Fusion + LR, cell mean": r"Single-cell classifier, fusion (design of \cite{sidhom2021apl})"}
for r in bl.itertuples():
    lab = labels.get(r.baseline, r.baseline.replace("Differential count (17 cell types) + LR", r"Differential count + LR (comparator in \cite{dasdelen2026caitomorph})"))
    C.append(f"{lab} & Same protocol & {ci(r.auroc, r.auroc_lo, r.auroc_hi)} & {cpr(r.sens, 12)} [{int(r.fn)}] & {cpr(r.spec, 12)} & {r.f1:.2f} & {npv_s(r.npv, r.fn, r.spec)} & {r.acc:.2f} \\\\")
C.append(r"\midrule")
C.append(r"\multicolumn{8}{l}{\emph{This work (pre-specified pipelines and their members)}} \\")
_ens = pd.read_csv(R / "ensemble_apl_macenko" / "ensemble.csv").set_index("model").loc["Ensemble"]
_ft = pd.read_csv(R / "finetune_t1_macenko" / "summary.csv").query("level == 'patient'").iloc[0]
_sec = pd.read_csv(R / "apl_fe_macenko" / "feature_engineering.csv").set_index("config").loc["eff_pca64_nw"]
cmp_rows = [(r"\textbf{Primary pipeline (ensemble)}", _ens.ext_auroc, _ens.lo, _ens.hi, _ens.sens, _ens.fn, _ens.spec, _ens.f1, _ens.npv, _ens.acc)]
for m, lab in [("P1", "Member: engineered, pooled LR"), ("P4", "Member: engineered, attention MIL")]:
    r = pt[pt.model == m].iloc[0]; cmp_rows.append((lab, r.ext, r.lo, r.hi, r.sens, r.fn, r.spec, r.f1, r.npv, r.acc))
cmp_rows.append(("Member: Macenko + fine-tuned CNN", _ft.ext_auroc, _ft.ext_lo, _ft.ext_hi, _ft.ext_sens, _ft.ext_fn, _ft.ext_spec, _ft.ext_f1, _ft.ext_npv, _ft.ext_acc))
cmp_rows.append(("Secondary pipeline (Macenko + frozen CNN + PCA-64)", _sec.ext_auroc, _sec.ext_lo, _sec.ext_hi, _sec.sens95_sens, _sec.sens95_fn, _sec.sens95_spec, _sec.sens95_f1, _sec.sens95_npv, _sec.sens95_acc))
for lab, au, lo, hi, se, fn, sp, f1, npv, acc in cmp_rows:
    C.append(f"{lab} & Same protocol & {ci(au, lo, hi)} & {cpr(se, 12)} [{int(fn)}] & {cpr(sp, 12)} & {f1:.2f} & {npv_s(npv, fn, sp)} & {acc:.2f} \\\\")
C += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(TAB / "tab_compare.tex").write_text("\n".join(C))

# confusion matrices on the external cohort (patient level)
pp = pd.read_csv(R / "v2_patient" / "predictions.csv")
cm_items = []
for r in bl.itertuples():
    if r.baseline in ("Promyelocyte proportion",) or r.baseline.startswith("Differential"):
        cm_items.append((r.baseline.replace(" (17 cell types) + LR", " + LR"), int(r.fn), int(12 - r.fn), int(r.fp), int(12 - r.fp)))
for m in ["P5", "P4", "P6"]:
    r = pt[pt.model == m].iloc[0]
    cm_items.append((f"{r.rep} + attention MIL", int(r.fn), int(12 - r.fn), int(r.fp), int(12 - r.fp)))
fig, axes = plt.subplots(1, len(cm_items), figsize=(7.16, 1.85))
for ax, (name, fn, tp, fp, tn) in zip(axes, cm_items):
    M = np.array([[tp, fn], [fp, tn]])
    ax.imshow([[1, 0], [0, 1]], cmap="Greys", vmin=0, vmax=6, alpha=0.25)
    for (i, j), v in np.ndenumerate(M):
        col = RED if (i, j) == (0, 1) and v > 0 else "#222"
        ax.text(j, i, str(v), ha="center", va="center", fontsize=10, color=col, fontweight="bold" if (i, j) == (0, 1) else None)
    ax.set_xticks([0, 1], ["APL", "AML"], fontsize=6.5); ax.set_yticks([0, 1], ["APL", "AML"], fontsize=6.5)
    ax.set_xlabel("Predicted", fontsize=6.5)
    ax.set_title(name.replace(" + attention MIL", "\n+ attention MIL").replace("Promyelocyte proportion", "Promyelocyte\nproportion").replace("Differential count + LR", "Differential\ncount + LR"), fontsize=6.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_ylabel("True", fontsize=6.5)
fig.text(0.5, 0.005, "External Validation cohort (12 APL, 12 AML) at the operating point fixed on Discovery. Red: APL patients missed.", ha="center", fontsize=6.5, color="#444")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "figures/fig_confusion.pdf"); fig.savefig(OUT / "figures/fig_confusion.png", dpi=220)

# ================================================================== paired tests + impact figure
def preds(t, model, split="grouped", seed=0):
    p = pd.read_csv(R / f"v2_cell_{t}" / "predictions.csv.gz")
    return p[(p.model == model) & (p.split == split) & (p["eval"] == "test") & (p.seed == seed)].sort_values("row")


tests = []
for t, tname in TASKS.items():
    a, b, c = preds(t, "M1"), preds(t, "M2"), preds(t, "M4")
    for name, x in [("Engineered (LightGBM) vs EfficientNet-B0", a), ("Fusion vs EfficientNet-B0", c)]:
        d = paired_bootstrap_auc(x.y.values, x.p.values, b.p.values, x.patient_id.values, 2000)
        tests.append(dict(task=tname, comparison=name, **d))
pp = pp[pp["eval"] == "test"]
P = lambda m: pp[pp.model == m].sort_values("patient_id")
for name, m1, m2 in [("Attention MIL vs pooled (engineered)", "P4", "P1"), ("Engineered vs EfficientNet-B0 (attention MIL)", "P4", "P5"),
                     ("Fusion vs EfficientNet-B0 (attention MIL)", "P6", "P5")]:
    d = paired_bootstrap_auc(P(m1).y.values, P(m1).p.values, P(m2).p.values, P(m1).patient_id.values, 4000)
    tests.append(dict(task="T3 Patient diagnosis", comparison=name, **d))
tests = pd.DataFrame(tests)
tests.to_csv(TAB / "paired_tests.csv", index=False)
facts["paired"] = tests.round(4).to_dict("records")

abl = []
for t, tname in TASKS.items():
    a = pd.read_csv(R / f"ablation_{t}" / "feature_groups.csv")
    a["task"] = tname
    abl.append(a)
abl = pd.concat(abl)
facts["ablation"] = abl.round(4).to_dict("records")
groups = list(dict.fromkeys(s.replace("without ", "") for s in abl.setting if s.startswith("without ")))
L = [r"\begin{table}[!t]", r"\centering",
     r"\caption{Feature-group ablation on the external cohort (engineered features, LR). $\Delta$: change in AUROC relative to all 93 features, paired patient bootstrap.}",
     r"\label{tab:ablation}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcccc}", r"\toprule",
     r" & \multicolumn{2}{c}{T2 Blast detection} & \multicolumn{2}{c}{T1 Diagnosis on cells} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
     r"Setting ($n$) & AUROC & $\Delta$ [95\% CI] & AUROC & $\Delta$ [95\% CI] \\", r"\midrule"]


def cv_(task, setting):
    r = abl[(abl.task == task) & (abl.setting == setting)].iloc[0]
    if setting.startswith("All"):
        return f"{r.auroc:.3f} & --", int(r.n_features)
    return f"{r.auroc:.3f} & {r.delta:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]{'*' if r.p < 0.05 else ''}", int(r.n_features)


v1, n = cv_(TASKS["apl_cell_blast"], "All engineered features"); v2, _ = cv_(TASKS["apl_cell_diagnosis"], "All engineered features")
L.append(f"All features ({n}) & {v1} & {v2} \\\\")
L.append(r"\midrule \multicolumn{5}{l}{\emph{Without one group}} \\")
for g in groups:
    v1, n = cv_(TASKS["apl_cell_blast"], f"without {g}"); v2, _ = cv_(TASKS["apl_cell_diagnosis"], f"without {g}")
    L.append(f"\\quad {g} ({n}) & {v1} & {v2} \\\\")
L.append(r"\midrule \multicolumn{5}{l}{\emph{One group only}} \\")
for g in groups:
    v1, n = cv_(TASKS["apl_cell_blast"], f"{g} only"); v2, _ = cv_(TASKS["apl_cell_diagnosis"], f"{g} only")
    L.append(f"\\quad {g} ({n}) & {v1} & {v2} \\\\")
L += [r"\bottomrule", r"\multicolumn{5}{l}{\footnotesize * $p<0.05$.}", r"\end{tabular}}", r"\end{table}"]
(TAB / "tab_ablation.tex").write_text("\n".join(L).replace("Chromatin texture (GLCM, entropy)", "Chromatin texture"))

items = [(r.comparison, r.task, r.delta, r.lo, r.hi) for r in tests.itertuples()]
for r in abl[(abl.task == TASKS["apl_cell_blast"]) & abl.setting.str.startswith("without")].itertuples():
    items.append((f"Remove {r.setting[8:].replace(' (GLCM, entropy)', '').lower()}", r.task, r.delta, r.delta_lo, r.delta_hi))
fig, (ax, axt) = plt.subplots(1, 2, figsize=(7.16, 3.5), gridspec_kw=dict(width_ratios=[3.2, 1.25]), sharey=True)
n_cmp = len(tests)
y = np.arange(len(items))[::-1].astype(float)
y[n_cmp:] -= 0.6
for yi, (lab, task, dv, lo, hi) in zip(y, items):
    sig = lo > 0 or hi < 0
    col = (GREEN if dv > 0 else RED) if sig else GREY
    ax.errorbar(dv, yi, xerr=[[dv - lo], [hi - dv]], fmt="o", color=col, ms=4, elinewidth=1, capsize=2)
    axt.text(0.0, yi, f"{dv:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=6.5, color=col if sig else "#444")
ax.set_yticks(y, [f"{lab}  ({task.split()[0]})" for lab, task, *_ in items], fontsize=6.6)
ax.axvline(0, color="#555", lw=0.8); ax.axhline(y[n_cmp - 1] - 0.8, color="#DDD", lw=0.8)
ax.set_xlim(-0.25, 0.45); ax.set_xlabel("Change in external AUROC")
ax.set_title("Design choices (top) and feature-group removal (bottom)", loc="left", fontsize=8)
axt.axis("off"); axt.set_xlim(0, 1); axt.set_title(r"$\Delta$ AUROC [95% CI]", loc="left", fontsize=7.5)
fig.text(0.01, 0.01, "Paired patient-level bootstrap on the external cohort. Green: significant gain; red: significant loss; grey: CI includes 0.", fontsize=6.3, color="#555")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "figures/fig_impact.pdf"); fig.savefig(OUT / "figures/fig_impact.png", dpi=220)

cal = []
for t, tname in TASKS.items():
    c = pd.read_csv(R / f"ablation_{t}" / "calibration.csv"); c["task"] = tname; cal.append(c)
cal = pd.concat(cal)
facts["calibration"] = cal.round(4).to_dict("records")
att = pd.read_csv(R / "v2_patient" / "attention_P4.csv.gz")
y_te = pp.drop_duplicates("patient_id").set_index("patient_id").y
att["dx"] = att.patient_id.map(y_te).map({1: "APL", 0: "AML"})
s = att[att.cell_type != "unlabelled"].groupby(["cell_type", "dx"]).attention_rel.agg(["mean", "count"]).reset_index()
facts["attention"] = s[s["count"] >= 15].round(3).to_dict("records")
(TAB / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
pd.set_option("display.width", 220)
print(leak.round(3).to_string()); print(pt.round(3).to_string()); print(bl.round(3).to_string()); print(tests.round(3).to_string())

# ================================================================== feature engineering (patient level)
fe_parts = [d for d in ["feature_engineering", "feature_engineering_part2"] if (R / d / "feature_engineering.csv").exists()]
fe = pd.concat([pd.read_csv(R / d / "feature_engineering.csv") for d in fe_parts], ignore_index=True).drop_duplicates("config", keep="last")
fp_ = pd.concat([pd.read_csv(R / d / "predictions.csv") for d in fe_parts], ignore_index=True).drop_duplicates(["config", "patient_id"], keep="last")
order = ["eng_raw", "eng_corrfilter", "eng_slidenorm", "eng_l1", "eng_stable", "eff_raw", "eff_pca16", "eff_pca32", "eff_pca64",
         "fusion_pca32", "composition", "composition+eng", "mil_eng_stable", "mil_eff_pca32"]
fe = fe.set_index("config").loc[[c for c in order if c in set(fe.config)]].reset_index()
NICE = {"eng_raw": ("Engineered, all 93", "--"), "eng_corrfilter": ("Engineered", "Correlation filter ($|r|>0.9$)"),
        "eng_slidenorm": ("Engineered", "Per-slide stain normalization"), "eng_l1": ("Engineered", "L1 (LASSO) selection"),
        "eng_stable": ("Engineered", "Filter + stain normalization"), "eff_raw": ("EfficientNet-B0, 1,280-d", "--"),
        "eff_pca16": ("EfficientNet-B0", "PCA, 16 components"), "eff_pca32": ("EfficientNet-B0", "PCA, 32 components"),
        "eff_pca64": ("EfficientNet-B0", "PCA, 64 components"), "fusion_pca32": ("Fusion", "Filter + PCA-32"),
        "composition": ("Digital differential", "8 predicted cell classes"), "composition+eng": ("Differential + engineered", "Filtered"),
        "mil_eng_stable": ("Engineered, attention MIL", "Filter + stain normalization"), "mil_eff_pca32": ("EfficientNet-B0, attention MIL", "PCA-32")}
base = {"eng": "eng_raw", "eff": "eff_raw", "fus": "eff_raw", "com": "eng_raw", "mil": "eng_raw"}
Pfe = lambda c: fp_[fp_.config == c].sort_values("patient_id")
dl = []
for r in fe.itertuples():
    b = base[r.config[:3]]
    if r.config == b:
        dl.append((np.nan, np.nan, np.nan, np.nan)); continue
    d = paired_bootstrap_auc(Pfe(r.config).y.values, Pfe(r.config).p.values, Pfe(b).p.values, Pfe(r.config).patient_id.values, 4000)
    dl.append((d["delta"], d["lo"], d["hi"], d["p"]))
fe[["delta", "dlo", "dhi", "dp"]] = dl
fe["dp_holm"] = holm(fe.dp.values)
fe.to_csv(TAB / "feature_engineering.csv", index=False)
facts["feature_engineering"] = fe.round(4).to_dict("records")
sel = fe.loc[fe.cv_auroc.idxmax()]
facts["fe_selected"] = {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v) for k, v in sel.to_dict().items()}
L = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{Feature engineering for patient-level APL detection. All steps are fitted inside the Discovery training folds; the Validation cohort was not used for fitting. PCA rows in this table use the whitened scaling of our first analysis (each component standardized), and the pooled LR of these experiments standardizes each feature, so its unreduced references differ from Table~\ref{tab:patient} and from the natural-scale analysis of the main text (engineered 0.729 vs.\ 0.757; EfficientNet-B0 0.743 vs.\ 0.722 and 0.708). $\Delta$: external AUROC change relative to the unreduced representation (paired patient bootstrap). The configuration with the highest Discovery CV AUROC (pre-declared selection rule) is marked $^{\star}$. Oracle column: specificity at 100\% recall if the threshold were re-set on the Validation cohort (reference only, not deployable).}",
     r"\label{tab:fe}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llccccccc}", r"\toprule",
     r"Representation & Engineering step & CV AUROC & External AUROC [95\% CI] & $\Delta$ [95\% CI] & Recall (missed) & Spec. & F1 & Oracle spec. \\", r"\midrule"]
for i, r in enumerate(fe.itertuples()):
    rep, step = NICE[r.config]
    star = r"$^{\star}$" if r.config == sel.config else ""
    d = "--" if np.isnan(r.delta) else f"{r.delta:+.3f} [{r.dlo:+.3f}, {r.dhi:+.3f}]" + ("*" if r.dp < 0.05 else "") + (r"$^{\dagger}$" if r.dp_holm < 0.05 else "")
    L.append(f"{rep}{star} & {step} & {r.cv_auroc:.3f} & {ci(r.ext_auroc, r.ext_lo, r.ext_hi)} & {d} & {r.sens95_sens:.2f} ({int(r.sens95_fn)}/12) & {r.sens95_spec:.2f} & {r.sens95_f1:.2f} & {r.oracle_spec_at_recall1:.2f} \\\\")
    if r.config in ("eng_stable", "eff_pca64"):
        L.append(r"\midrule")
L += [r"\bottomrule", r"\multicolumn{9}{l}{\footnotesize Recall, specificity and F1 at the operating point fixed on Discovery (target sensitivity 0.95). * $p<0.05$ unadjusted; $^{\dagger}$ $p<0.05$ after Holm correction across all comparisons in the table.}", r"\end{tabular}}", r"\end{table*}"]
(TAB / "tab_fe.tex").write_text("\n".join(L))
(TAB / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
print(fe[["config", "cv_auroc", "ext_auroc", "delta", "dlo", "dhi", "dp", "sens95_sens", "sens95_spec", "oracle_spec_at_recall1"]].round(3).to_string())

# ================================================================== PCA: before / after and why (sensitivity analysis over k)
pca_parts = [d for d in ["feature_engineering", "feature_engineering_pca"] if (R / d / "feature_engineering.csv").exists()]
if (R / "pca_analysis" / "pca_analysis.csv").exists() and "feature_engineering_pca" in pca_parts:
    pk = pd.concat([pd.read_csv(R / d / "feature_engineering.csv") for d in pca_parts], ignore_index=True).drop_duplicates("config", keep="last")
    pkp = pd.concat([pd.read_csv(R / d / "predictions.csv") for d in pca_parts], ignore_index=True).drop_duplicates(["config", "patient_id"], keep="last")
    pk = pk[pk.config.str.match(r"eff_(raw|pca\d+)$")].copy()
    pk["k"] = pk.config.map(lambda c: 1280 if c == "eff_raw" else int(c[7:]))
    pk = pk.sort_values("k").reset_index(drop=True)
    diag = pd.read_csv(R / "pca_analysis" / "pca_analysis.csv")
    pk = pk.merge(diag[["k", "explained_var", "cohort_id_auroc", "cell_apl_auroc_external"]], on="k", how="left")
    Pk = lambda c: pkp[pkp.config == c].sort_values("patient_id")
    dl = []
    for r in pk.itertuples():
        if r.config == "eff_raw":
            dl.append((np.nan,) * 4); continue
        d = paired_bootstrap_auc(Pk(r.config).y.values, Pk(r.config).p.values, Pk("eff_raw").p.values, Pk(r.config).patient_id.values, 4000)
        dl.append((d["delta"], d["lo"], d["hi"], d["p"]))
    pk[["delta", "dlo", "dhi", "dp"]] = dl
    pk["dp_holm"] = holm(pk.dp.values)
    pk["inputs"] = 5 * pk.k
    pk["gap"] = pk.cv_auroc - pk.ext_auroc
    pk.to_csv(TAB / "pca_sweep.csv", index=False)
    facts["pca_sweep"] = pk.round(4).to_dict("records")
    order_k = [1280] + [k for k in pk.k if k != 1280]
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Patient-level APL detection before and after PCA of the EfficientNet-B0 embeddings (pooled-statistics LR). PCA is fitted on the training cells of each Discovery fold. Inputs: number of patient-level features (five statistics per component) for about 66 training patients per fold. Cohort ID: AUROC for telling Validation from Discovery AML cells (0.5 = no acquisition signal). Cell APL: external cell-level AUROC of a cell classifier. $\Delta$: change in external AUROC relative to the unreduced embedding (paired patient bootstrap, 4,000 resamples). Components 16, 32 and 64 were specified in advance; the others form a post hoc sensitivity analysis.}",
         r"\label{tab:pca}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{rrrccccccccc}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Cell level} & \multicolumn{4}{c}{Patient level, AUROC} & \multicolumn{3}{c}{External, fixed operating point} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-9}\cmidrule(lr){10-12}",
         r"Components & Var. & Inputs & Cohort ID & Cell APL & CV & External [95\% CI] & Gap & $\Delta$ [95\% CI] & Recall (missed) & Spec. & F1 \\", r"\midrule"]
    for k in order_k:
        r = pk[pk.k == k].iloc[0]
        lab = "None (1,280)" if k == 1280 else str(k)
        d = "--" if np.isnan(r.delta) else f"{r.delta:+.3f} [{r.dlo:+.3f}, {r.dhi:+.3f}]" + ("*" if r.dp < 0.05 else "") + (r"$^{\dagger}$" if r.dp_holm < 0.05 else "")
        best = r.config == "eff_pca32"
        ext = ci(r.ext_auroc, r.ext_lo, r.ext_hi)
        L.append(f"{lab} & {r.explained_var:.2f} & {int(r.inputs):,} & {r.cohort_id_auroc:.3f} & {r.cell_apl_auroc_external:.3f} & {r.cv_auroc:.3f} & "
                 + (rf"\textbf{{{ext}}}" if best else ext) + f" & {r.gap:.3f} & {d} & {r.sens95_sens:.2f} ({int(r.sens95_fn)}/12) & {r.sens95_spec:.2f} & {r.sens95_f1:.2f} \\\\")
        if k == 1280:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\multicolumn{12}{l}{\footnotesize Var.: cumulative explained variance on Discovery cells. Gap: CV AUROC minus external AUROC. * $p<0.05$ unadjusted; $^{\dagger}$ $p<0.05$ after Holm correction across all comparisons in the table.}", r"\end{tabular}}", r"\end{table*}"]
    pass  # first-analysis PCA table superseded by tab_pca

    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.3))
    s = pk[pk.k != 1280]; raw = pk[pk.k == 1280].iloc[0]
    a0 = ax[0]
    a0.axvspan(13, 76, color="#F2F6FB", lw=0, zorder=0)
    a0.plot(s.k, s.cv_auroc, "o-", color=GREY, ms=3.5, label="Discovery CV")
    a0.plot(s.k, s.ext_auroc, "o-", color=BLUE, ms=3.5, label="External (Validation)")
    a0.fill_between(s.k, s.ext_lo, s.ext_hi, color=BLUE, alpha=0.12, lw=0)
    a0.axhline(raw.ext_auroc, color=BLUE, ls="--", lw=0.9); a0.axhline(raw.cv_auroc, color=GREY, ls="--", lw=0.9)
    a0.text(4.2, raw.ext_auroc + 0.012, "no PCA (external)", color=BLUE, fontsize=6.5)
    a0.set_xscale("log", base=2); a0.set_xticks(s.k); a0.set_xticklabels(s.k)
    a0.set_xlabel("PCA components"); a0.set_ylabel("Patient-level AUROC"); a0.set_ylim(0.2, 1.02)
    a0.legend(frameon=False, loc="lower right", fontsize=6.5); a0.set_title("(a) Patient-level APL detection", loc="left", fontsize=8)
    a1 = ax[1]
    a1.plot(s.k, s.gap, "o-", color=ORANGE, ms=3.5)
    a1.axhline(raw.gap, color=ORANGE, ls="--", lw=0.9); a1.text(40, raw.gap + 0.008, "no PCA", color=ORANGE, fontsize=6.5)
    a1b = a1.twinx(); a1b.spines["top"].set_visible(False)
    a1b.bar(s.k, s.inputs / 65.6, width=s.k * 0.35, color=GREY, alpha=0.35)
    a1b.axhline(1, color=GREY, lw=0.6, ls=":"); a1b.set_ylabel("Inputs per training patient", color=GREY, fontsize=6.8); a1b.set_yscale("log")
    a1.set_zorder(a1b.get_zorder() + 1); a1.patch.set_visible(False)
    a1.set_xscale("log", base=2); a1.set_xticks(s.k); a1.set_xticklabels(s.k)
    a1.set_xlabel("PCA components"); a1.set_ylabel("CV $-$ external AUROC", color=ORANGE)
    a1.set_title("(b) Generalization gap and model size", loc="left", fontsize=8)
    a2 = ax[2]
    a2.plot(s.k, s.cell_apl_auroc_external, "o-", color=GREEN, ms=3.5, label="APL signal (cell AUROC)")
    a2.plot(s.k, s.cohort_id_auroc, "o-", color=RED, ms=3.5, label="Cohort signal (cohort-ID AUROC)")
    a2.plot(s.k, s.explained_var, "s--", color=GREY, ms=3, lw=0.9, label="Explained variance")
    a2.axhline(raw.cell_apl_auroc_external, color=GREEN, ls=":", lw=0.9); a2.axhline(raw.cohort_id_auroc, color=RED, ls=":", lw=0.9)
    a2.set_xscale("log", base=2); a2.set_xticks(s.k); a2.set_xticklabels(s.k); a2.set_ylim(0.2, 1.08)
    a2.set_xlabel("PCA components"); a2.legend(frameon=False, fontsize=6.2, loc="upper left")
    a2.set_title("(c) What the components keep", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "fig_pca.pdf"); fig.savefig(OUT / "figures" / "fig_pca.png", dpi=200); plt.close(fig)
    print(pk[["config", "explained_var", "inputs", "cohort_id_auroc", "cell_apl_auroc_external", "cv_auroc", "ext_auroc", "gap", "delta", "dlo", "dhi", "dp", "sens95_sens", "sens95_fn", "sens95_spec"]].round(3).to_string())

# ================================================================== training dynamics (learning / loss curves)
TC = R / "training_curves"
if (TC / "abmil_history.csv").exists() and (TC / "lgbm_history.csv").exists() and (TC / "sample_size_curve.csv").exists():
    ah = pd.read_csv(TC / "abmil_history.csv"); lh = pd.read_csv(TC / "lgbm_history.csv"); sz = pd.read_csv(TC / "sample_size_curve.csv")
    INP = {"eng": ("Engineered (93)", GREEN), "eff_raw": ("EfficientNet-B0 (1,280)", GREY), "eff_pca32": ("EfficientNet-B0 + PCA-32", BLUE)}
    fig, ax = plt.subplots(2, 3, figsize=(7.16, 4.4))
    band = lambda a, x, m, s, c, **kw: (a.plot(x, m, color=c, **kw), a.fill_between(x, m - s, m + s, color=c, alpha=0.10, lw=0))
    for inp, (nm, c) in INP.items():
        g = ah[ah.input == inp].groupby("epoch")
        band(ax[0, 0], g.train_loss.mean().index, g.train_loss.mean(), g.train_loss.std(), c, label=nm, lw=1.2)
        band(ax[0, 1], g.val_loss.mean().index, g.val_loss.mean(), g.val_loss.std(), c, lw=1.2)
        e = g.val_loss.mean().idxmin(); ax[0, 1].plot([e], [g.val_loss.mean()[e]], "v", color=c, ms=5, mec="white", mew=0.5)
        band(ax[0, 2], g.val_auroc.mean().index, g.val_auroc.mean(), g.val_auroc.std(), c, lw=1.2)
    ax[0, 0].set_title("(a) MIL: training loss", loc="left", fontsize=8); ax[0, 0].set_ylabel("Weighted BCE loss")
    ax[0, 1].set_title("(b) MIL: held-out fold loss", loc="left", fontsize=8); ax[0, 1].set_ylabel("BCE loss")
    ax[0, 2].set_title("(c) MIL: held-out fold AUROC", loc="left", fontsize=8); ax[0, 2].set_ylabel("AUROC"); ax[0, 2].set_ylim(0.5, 1.0)
    ax[0, 0].legend(frameon=False, fontsize=6.3)
    for a in ax[0]:
        a.set_xlabel("Epoch")
    g = lh.groupby("round")
    band(ax[1, 0], g.train_loss.mean().index, g.train_loss.mean(), g.train_loss.std(), ORANGE, label="Training folds", lw=1.2)
    band(ax[1, 0], g.val_loss.mean().index, g.val_loss.mean(), g.val_loss.std(), BLUE, label="Held-out patients", lw=1.2)
    e = g.val_loss.mean().idxmin(); ax[1, 0].plot([e], [g.val_loss.mean()[e]], "v", color=BLUE, ms=5, mec="white", mew=0.5)
    ax[1, 0].set_xlabel("Boosting round"); ax[1, 0].set_ylabel("Log-loss"); ax[1, 0].legend(frameon=False, fontsize=6.3)
    ax[1, 0].set_title("(d) LightGBM (T1, cells): loss", loc="left", fontsize=8)
    for inp, (nm, c) in INP.items():
        q = sz[sz.input == inp].copy()
        q["n"] = q.n_train.where(q.n_train < 60, q.n_train.max())
        m = q.groupby("n").agg(cv=("cv_auroc", "mean"), cvs=("cv_auroc", "std"), ex=("ext_auroc", "mean"), exs=("ext_auroc", "std"), tr=("train_auroc", "mean"))
        band(ax[1, 1], m.index.values, m.cv.values, m.cvs.values, c, lw=1.2, marker="o", ms=3, label=nm)
        band(ax[1, 2], m.index.values, m.ex.values, m.exs.values, c, lw=1.2, marker="o", ms=3, label=nm)
        ax[1, 1].plot(m.index.values, m.tr.values, color=c, lw=0.8, ls=":")
    ax[1, 1].set_title("(e) Learning curve: held-out fold", loc="left", fontsize=8); ax[1, 1].set_ylabel("AUROC (dotted: training)")
    ax[1, 2].set_title("(f) Learning curve: external cohort", loc="left", fontsize=8); ax[1, 2].set_ylabel("AUROC"); ax[1, 2].legend(frameon=False, fontsize=6.3, loc="lower right")
    for a in ax[1, 1:]:
        a.set_xlabel("Training patients (pooled LR)"); a.set_ylim(0.4, 1.02)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "fig_training.pdf"); fig.savefig(OUT / "figures" / "fig_training.png", dpi=200); plt.close(fig)
    last = ah[ah.epoch == ah.epoch.max()].groupby("input")[["train_loss", "val_loss", "val_auroc"]].mean()
    best = ah.groupby(["input", "epoch"]).val_loss.mean().groupby("input").agg(["idxmin", "min"])
    au10 = ah[ah.epoch == 10].groupby("input").val_auroc.mean()
    lg = lh.groupby("round")[["train_loss", "val_loss", "val_auroc"]].mean()
    szs = sz.groupby(["input", "n_train"])[["train_auroc", "cv_auroc", "ext_auroc"]].mean().reset_index()
    facts["training"] = dict(abmil_last=last.round(4).to_dict(), abmil_best_epoch={k: int(v[1]) for k, v in best["idxmin"].items()},
                             abmil_best_val_loss=best["min"].round(4).to_dict(), abmil_auroc_epoch10=au10.round(4).to_dict(),
                             lgbm_best_round=int(lg.val_loss.idxmin()), lgbm_min_val_loss=round(float(lg.val_loss.min()), 4),
                             lgbm_last=lg.iloc[-1].round(4).to_dict(), lgbm_first=lg.iloc[0].round(4).to_dict(),
                             size=szs.round(4).to_dict("records"))
    print(json.dumps(facts["training"], indent=1, default=float)[:3000])
(TAB / "facts.json").write_text(json.dumps(facts, indent=1, default=float))

# ================================================================== REVISION 1
# ------------------------------------------------------------------ C-NMC replication (second dataset)
def cell_leak_rows(folders, names):
    s = pd.concat([pd.read_csv(R / f / "summary.csv") for f in folders if (R / f / "summary.csv").exists()])
    g = s.groupby(["model", "split", "eval"]).agg(auroc=("auroc", "mean"), lo=("auroc_lo", "mean"), hi=("auroc_hi", "mean"), sens=("sens", "mean"),
                                                   spec=("spec", "mean"), f1=("f1", "mean"), npv=("npv", "mean"), acc=("acc", "mean"), seeds=("seed", "nunique"))
    out = []
    for m, nm in names.items():
        if m not in g.index.get_level_values(0):
            continue
        a = g.loc[m]
        ic, gc, ex = a.loc[("image", "cv")], a.loc[("grouped", "cv")], a.loc[("grouped", "test")]
        out.append(dict(model=nm, image_cv=ic.auroc, grouped_cv=gc.auroc, ext=ex.auroc, lo=ex.lo, hi=ex.hi, sens=ex.sens, spec=ex.spec,
                        f1=ex.f1, npv=ex.npv, acc=ex.acc, infl=ic.auroc - ex.auroc, leak=ic.auroc - gc.auroc, shift=gc.auroc - ex.auroc, seeds=int(ex.seeds)))
    return pd.DataFrame(out)


if (R / "cnmc_cell" / "summary.csv").exists():
    cn = cell_leak_rows(["cnmc_cell", "cnmc_cell_lgbm"], NAMES)
    cn.to_csv(TAB / "cnmc_cell.csv", index=False); facts["cnmc_cell"] = cn.round(4).to_dict("records")
    ps = pd.read_csv(R / "cnmc_patient" / "summary.csv")
    PN = {"P1": ("Engineered", "Pooled + LR"), "P2": ("EfficientNet-B0", "Pooled + LR"), "P3": ("Fusion", "Pooled + LR"),
          "P4": ("Engineered", "Attention MIL"), "P5": ("EfficientNet-B0", "Attention MIL"), "P6": ("Fusion", "Attention MIL")}
    cvm = ps[ps["eval"] == "cv"].groupby("model").auroc.mean()
    ex = ps[ps["eval"] == "test"].set_index("model")
    cp = pd.DataFrame([dict(model=m, rep=PN[m][0], agg=PN[m][1], cv=cvm[m], ext=ex.loc[m, "auroc"], lo=ex.loc[m, "auroc_lo"], hi=ex.loc[m, "auroc_hi"],
                            sens=ex.loc[m, "sens"], fn=ex.loc[m, "fn"], spec=ex.loc[m, "spec"], f1=ex.loc[m, "f1"], npv=ex.loc[m, "npv"],
                            acc=ex.loc[m, "acc"], ece=ex.loc[m, "ece"]) for m in PN])
    cp.to_csv(TAB / "cnmc_subject.csv", index=False); facts["cnmc_subject"] = cp.round(4).to_dict("records")
    npos = int(pd.read_csv(R / "cnmc_fe" / "predictions.csv").groupby("patient_id").y.first().sum())
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Replication on the second dataset, C-NMC 2019 (ALL vs.\ normal B cells; development: official folds 0--1, 42 subjects; external: fold 2, 31 unseen subjects, 17 with ALL). Same features, models, protocol and operating-point rule as for APL-AML. Leakage: image-level minus subject-grouped CV AUROC; external gap: grouped CV minus external AUROC.}",
         r"\label{tab:cnmc}", r"\setlength{\tabcolsep}{3.5pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llcccccccc}", r"\toprule",
         r"\multicolumn{10}{l}{\emph{(a) Cell level: every cell carries its subject's diagnosis (mean of " + (str(int(cn.seeds.min())) if cn.seeds.min() == cn.seeds.max() else str(int(cn.seeds.min())) + r"--" + str(int(cn.seeds.max()))) + r" seeds)}} \\",
         r"Model & & Image CV & Grouped CV & External [95\% CI] & Recall & Spec. & F1 & Leakage & Ext.\ gap \\", r"\midrule"]
    for r in cn.itertuples():
        L.append(f"{r.model} & & {r.image_cv:.3f} & {r.grouped_cv:.3f} & {ci(r.ext, r.lo, r.hi)} & {r.sens:.2f} & {r.spec:.2f} & {r.f1:.2f} & {r.leak:+.3f} & {r.shift:+.3f} \\\\")
    L += [r"\midrule", r"\multicolumn{10}{l}{\emph{(b) Subject level: is this subject's sample leukemic?}} \\",
          r"Representation & Aggregation & CV AUROC & & External [95\% CI] & Recall (95\% CI) [missed] & Spec. (95\% CI) & F1 & NPV & \\", r"\midrule"]
    for r in cp.itertuples():
        L.append(f"{r.rep} & {r.agg} & {r.cv:.3f} & & {ci(r.ext, r.lo, r.hi)} & {cpr(r.sens, npos)} [{int(r.fn)}] & {cpr(r.spec, 31 - npos)} & {r.f1:.2f} & {npv_s(r.npv, r.fn, r.spec)} & \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TAB / "tab_cnmc.tex").write_text("\n".join(L))
    print(cn.round(3).to_string()); print(cp.round(3).to_string())


# ------------------------------------------------------------------ PCA sweep on both datasets
def pca_sweep(fe_dirs, diag_dir):
    fe = pd.concat([pd.read_csv(R / d / "feature_engineering.csv") for d in fe_dirs if (R / d / "feature_engineering.csv").exists()]).drop_duplicates("config", keep="last")
    pr = pd.concat([pd.read_csv(R / d / "predictions.csv") for d in fe_dirs if (R / d / "predictions.csv").exists()]).drop_duplicates(["config", "patient_id"], keep="last")
    fe = fe[fe.config.str.match(r"eff_(raw|pca\d+)(_nw)?$")].copy()
    fe["k"] = fe.config.map(lambda c: 1280 if c.startswith("eff_raw") else int(c.split("pca")[1].split("_")[0]))
    rawc = fe.config[fe.k == 1280].iloc[0]
    fe = fe.merge(pd.read_csv(R / diag_dir / "pca_analysis.csv")[["k", "explained_var", "cohort_id_auroc", "cell_apl_auroc_external"]], on="k")
    Pq = lambda c: pr[pr.config == c].sort_values("patient_id")
    dl = []
    for r in fe.itertuples():
        if r.config == rawc:
            dl.append((np.nan,) * 4); continue
        d = paired_bootstrap_auc(Pq(r.config).y.values, Pq(r.config).p.values, Pq(rawc).p.values, Pq(r.config).patient_id.values, 4000)
        dl.append((d["delta"], d["lo"], d["hi"], d["p"]))
    fe[["delta", "dlo", "dhi", "dp"]] = dl
    fe["dp_holm"] = holm(fe.dp.values); fe["gap"] = fe.cv_auroc - fe.ext_auroc
    return fe.sort_values("k").reset_index(drop=True)


if (R / "cnmc_fe" / "feature_engineering.csv").exists():
    sw = {("APL-AML", "std"): pca_sweep(["feature_engineering", "feature_engineering_pca"], "pca_analysis"),
          ("APL-AML", "nw"): pca_sweep(["apl_fe_nw"], "pca_analysis"),
          ("C-NMC", "std"): pca_sweep(["cnmc_fe"], "cnmc_pca_analysis"),
          ("C-NMC", "nw"): pca_sweep(["cnmc_fe_nw"], "cnmc_pca_analysis")}
    for (nm, v), d in sw.items():
        d.to_csv(TAB / f"pca_sweep_{nm.replace('-', '_')}_{v}.csv", index=False)
        facts[f"pca_{nm}_{v}"] = d.round(4).to_dict("records")
    cf = pd.read_csv(R / "cnmc_fe" / "feature_engineering.csv")
    facts["cnmc_fe"] = cf.round(4).to_dict("records")
    NPOS = {"APL-AML": 12, "C-NMC": 17}
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Patient/subject-level detection before and after PCA of the EfficientNet-B0 embeddings (pooled-statistics LR), on both datasets. PCA is fitted on training-fold cells only; the principal-component scores keep their relative variances (no whitening). Std.: external AUROC when each component is instead standardized to unit variance before the classifier (whitening), as in our first analysis. $\Delta$: change in external AUROC relative to no PCA under the same scaling (paired bootstrap). Recall, specificity at the operating point fixed on development data. Components 16, 32 and 64 were pre-specified.}",
         r"\label{tab:pca}", r"\setlength{\tabcolsep}{2.6pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{rcccccccccccc}", r"\toprule",
         r" & \multicolumn{6}{c}{APL-AML (24 external patients, 12 APL)} & \multicolumn{6}{c}{C-NMC (31 external subjects, 17 ALL)} \\",
         r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
         r"Components & Var. & Std. & External [95\% CI] & $\Delta$ & Recall (missed) & Spec. & Var. & Std. & External [95\% CI] & $\Delta$ & Recall (missed) & Spec. \\", r"\midrule"]
    ks = [1280, 8, 16, 32, 64, 128, 256]
    for k in ks:
        cells = ["None (1,280)" if k == 1280 else str(k)]
        for nm in ["APL-AML", "C-NMC"]:
            n_, s_ = sw[(nm, "nw")], sw[(nm, "std")]
            r = n_[n_.k == k]; q = s_[s_.k == k]
            if r.empty:
                cells += ["--"] * 6; continue
            r = r.iloc[0]
            d = "--" if np.isnan(r.delta) else f"{r.delta:+.3f}" + ("*" if r.dp < 0.05 else "") + (r"$^{\dagger}$" if r.dp_holm < 0.05 else "")
            cells += [f"{r.explained_var:.2f}", f"{q.iloc[0].ext_auroc:.3f}" if not q.empty else "--", ci(r.ext_auroc, r.ext_lo, r.ext_hi), d,
                      f"{r.sens95_sens:.2f} ({int(r.sens95_fn)}/{NPOS[nm]})", f"{r.sens95_spec:.2f}"]
        L.append(" & ".join(cells) + r" \\")
        if k == 1280:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\multicolumn{13}{l}{\footnotesize Var.: cumulative explained variance on development cells. * $p<0.05$ unadjusted; $^{\dagger}$ $p<0.05$ after Holm correction.}", r"\end{tabular}}", r"\end{table*}"]
    (TAB / "tab_pca.tex").write_text("\n".join(L))
    # compact before/after impact table (non-whitened PCA): no PCA, pre-specified 32 and 64, and the rule-selected k
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Impact of PCA on patient/subject-level detection (external sets). Before: unreduced 1,280-dimensional embedding. After: PCA fitted on development cells, scores kept at their natural scale. Rule: number of components with the highest development CV AUROC within the pre-specified set $\{16,32,64\}$; over the full sweep (4--256 and no PCA), development CV favoured 256 components on APL-AML and no reduction on C-NMC. Threshold metrics at the operating point fixed on development data.}",
         r"\label{tab:pcaimpact}", r"\setlength{\tabcolsep}{2.4pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llccccccc}", r"\toprule",
         r"Dataset & Representation & AUROC & $\Delta$AUROC & Acc. & Recall (missed) & Spec. & F1 & NPV \\", r"\midrule"]
    imp = []
    for nm in ["APL-AML", "C-NMC"]:
        d = sw[(nm, "nw")].set_index("k")
        rule_k = int(d.loc[[16, 32, 64]].cv_auroc.idxmax())   # pre-specified set
        sweep_k = int(d.cv_auroc.idxmax())                       # sensitivity analysis: whole sweep
        picks = [(1280, "Before: no PCA"), (32, "After: PCA-32"), (64, "After: PCA-64")]
        picks = [(k, lab + (" (rule)" if k == rule_k else "")) for k, lab in picks]
        if sweep_k not in (32, 64, 1280):
            picks.append((sweep_k, rf"PCA-{sweep_k} (best dev.\ CV, full sweep)"))
        for i, (k, lab) in enumerate(picks):
            r = d.loc[k]
            dlt = "--" if k == 1280 else f"{r.ext_auroc - d.loc[1280].ext_auroc:+.3f}"
            L.append(f"{nm if i == 0 else ''} & {lab} & {r.ext_auroc:.3f} & {dlt} & {r.sens95_acc:.2f} & {r.sens95_sens:.2f} ({int(r.sens95_fn)}/{NPOS[nm]}) & {r.sens95_spec:.2f} & {r.sens95_f1:.2f} & {r.sens95_npv:.2f} \\\\")
            imp.append(dict(dataset=nm, k=k, label=lab, auroc=r.ext_auroc, acc=r.sens95_acc, sens=r.sens95_sens, fn=r.sens95_fn, spec=r.sens95_spec, f1=r.sens95_f1, npv=r.sens95_npv))
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_pcaimpact.tex").write_text("\n".join(L))
    facts["pca_impact"] = pd.DataFrame(imp).round(4).to_dict("records")
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.3))
    for i, (nm, c) in enumerate([("APL-AML", BLUE), ("C-NMC", GREEN)]):
        for v, ls, lab in [("nw", "-", "PCA scores kept at natural scale"), ("std", "--", "each component standardized (whitened)")]:
            d = sw[(nm, v)]; s_ = d[d.k != 1280]; raw = d[d.k == 1280].iloc[0]
            ax[i].plot(s_.k, s_.ext_auroc, "o" + ls, color=c, ms=3.2, lw=1.2 if v == "nw" else 0.9, alpha=1 if v == "nw" else 0.6, label=lab)
            ax[i].axhline(raw.ext_auroc, color=c, ls=":" if v == "nw" else (0, (1, 3)), lw=1, alpha=1 if v == "nw" else 0.6)
            if v == "nw":
                ax[i].fill_between(s_.k, s_.ext_lo, s_.ext_hi, color=c, alpha=0.10, lw=0)
        ax[i].set_title(f"({'ab'[i]}) {nm}: external AUROC", loc="left", fontsize=8); ax[i].set_ylim(0.3, 1.02)
        ax[i].legend(frameon=False, fontsize=5.8, loc="lower right")
    ax[0].set_ylabel("Patient/subject AUROC")
    for nm, c in [("APL-AML", BLUE), ("C-NMC", GREEN)]:
        d = sw[(nm, "nw")]; s_ = d[d.k != 1280]
        ax[2].plot(s_.k, s_.cell_apl_auroc_external, "o-", color=c, ms=3.2, label=f"{nm}: disease signal")
        ax[2].plot(s_.k, s_.cohort_id_auroc, "s--", color=c, ms=2.8, lw=0.9, label=f"{nm}: cohort signal")
    ax[2].set_ylabel("Cell-level AUROC"); ax[2].set_ylim(0.4, 0.98); ax[2].legend(frameon=False, fontsize=5.6, loc="upper left")
    ax[2].set_title("(c) What the components keep", loc="left", fontsize=8)
    for a_ in ax:
        a_.set_xscale("log", base=2); a_.set_xticks([4, 8, 16, 32, 64, 128, 256]); a_.set_xticklabels([4, 8, 16, 32, 64, 128, 256]); a_.set_xlabel("PCA components")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "fig_pca_two.pdf"); fig.savefig(OUT / "figures" / "fig_pca_two.png", dpi=200); plt.close(fig)
    for key, d in sw.items():
        print(key); print(d[["config", "explained_var", "cv_auroc", "ext_auroc", "ext_lo", "ext_hi", "delta", "dlo", "dhi", "dp", "dp_holm", "sens95_sens", "sens95_fn", "sens95_spec", "sens95_f1"]].round(3).to_string())

# ------------------------------------------------------------------ fine-tuned CNN vs frozen / engineered (cell level, both datasets)
def ft_row(task_label, ftdir):
    sm = [p for p in [R / ftdir / "summary.csv", R / (ftdir + "_image") / "summary.csv"] if p.exists()]
    if not sm:
        return None
    s_ = pd.concat([pd.read_csv(p) for p in sm]); s_ = s_[s_.level == "cell"].set_index("split")
    g = s_.loc["grouped"]
    return dict(task=task_label, model="Fine-tuned EfficientNet-B0", image_cv=s_.loc["image", "cv_auroc"] if "image" in s_.index else np.nan,
                grouped_cv=g.cv_auroc, ext=g.ext_auroc, lo=g.ext_lo, hi=g.ext_hi, sens=g.ext_sens, spec=g.ext_spec, f1=g.ext_f1, seeds=1)


if (R / "finetune_t1" / "summary.csv").exists():
    rows = []
    for task, lab, ftdir in [("apl_cell_diagnosis", "T1 APL-AML, diagnosis on cells", "finetune_t1"), ("apl_cell_blast", "T2 APL-AML, blast detection", "finetune_t2")]:
        for r in leak[leak.task == TASKS[task]].itertuples():
            rows.append(dict(task=lab, model=r.model, image_cv=r.image_cv, grouped_cv=r.grouped_cv, ext=r.ext, lo=r.ext_lo, hi=r.ext_hi,
                             sens=r.sens, spec=r.spec, f1=r.f1, seeds=3))
        fr = ft_row(lab, ftdir)
        if fr:
            rows.append(fr)
    if "cnmc_cell" in facts:
        for r in cn.itertuples():
            rows.append(dict(task="T4 C-NMC, cells", model=r.model, image_cv=r.image_cv, grouped_cv=r.grouped_cv, ext=r.ext, lo=r.lo, hi=r.hi,
                             sens=r.sens, spec=r.spec, f1=r.f1, seeds=r.seeds))
        fr = ft_row("T4 C-NMC, cells", "finetune_cnmc")
        if fr:
            rows.append(fr)
    ft = pd.DataFrame(rows); ft.to_csv(TAB / "finetune.csv", index=False); facts["finetune_cell"] = ft.round(4).to_dict("records")
    fpat = {}
    for nm, d in [("APL-AML", "finetune_t1"), ("C-NMC", "finetune_cnmc")]:
        if (R / d / "summary.csv").exists():
            q = pd.read_csv(R / d / "summary.csv"); q = q[q.level == "patient"]
            if len(q):
                fpat[nm] = {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v) for k, v in q.iloc[0].to_dict().items()}
    facts["finetune_patient"] = fpat
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Engineered, frozen and fine-tuned features at cell level on both datasets. The fine-tuned model adapts the last two stages of EfficientNet-B0 (Table~\ref{S-tab:hparams}; one seed); the other models are averaged over three seeds. Leakage: image-level minus patient-grouped CV AUROC; external gap: grouped CV minus external AUROC. Recall, specificity and F1 on the external set at the operating point fixed on development data. Best external AUROC per task in bold.}",
         r"\label{tab:finetune}", r"\setlength{\tabcolsep}{3.5pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llcccccccc}", r"\toprule",
         r"Task & Model & Image CV & Grouped CV & External [95\% CI] & Leakage & Ext.\ gap & Recall & Spec. & F1 \\", r"\midrule"]
    for t, grp in ft.groupby("task", sort=False):
        for i, r in enumerate(grp.itertuples()):
            ic = "--" if np.isnan(r.image_cv) else f"{r.image_cv:.3f}"
            inf = ("--" if np.isnan(r.image_cv) else f"{r.image_cv - r.grouped_cv:+.3f}") + f" & {r.grouped_cv - r.ext:+.3f}"
            ext = ci(r.ext, r.lo, r.hi)
            L.append(f"{t if i == 0 else ''} & {r.model} & {ic} & {r.grouped_cv:.3f} & " + (rf"\textbf{{{ext}}}" if r.ext == grp.ext.max() else ext)
                     + f" & {inf} & {r.sens:.2f} & {r.spec:.2f} & {r.f1:.2f} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table*}"]
    (TAB / "tab_finetune.tex").write_text("\n".join(L))
    print(ft.round(3).to_string()); print(fpat)

# ------------------------------------------------------------------ reported (leaky) vs honest accuracy, threshold 0.5 as usually reported
def acc_auc(df):
    from sklearn.metrics import roc_auc_score
    y, p = df.y.values, df.p.values
    m = ~np.isnan(p)
    y, p = y[m], p[m]
    pred = p >= 0.5
    tp, fn = (pred & (y == 1)).sum(), (~pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
    return dict(acc=(pred == y).mean(), auroc=roc_auc_score(y, p), sens=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn))


rep_rows = []
SRC = [("T1 APL-AML, diagnosis on cells", "v2_cell_apl_cell_diagnosis", ["M1", "M4"]), ("T2 APL-AML, blast detection", "v2_cell_apl_cell_blast", ["M1", "M4"]),
       ("T4 C-NMC, cells", "cnmc_cell", ["M1lr", "M4"])]
for lab, d, models in SRC:
    f = R / d / "predictions.csv.gz"
    if not f.exists():
        continue
    P = pd.read_csv(f)
    for m in models:
        q = P[P.model == m]
        if q.empty:
            continue
        leak_r = [acc_auc(q[(q.split == "image") & (q["eval"] == "cv") & (q.seed == s_)]) for s_ in sorted(q.seed.unique())]
        hon = [acc_auc(q[(q.split == "grouped") & (q["eval"] == "test") & (q.seed == s_)]) for s_ in sorted(q.seed.unique())]
        mean = lambda L_, k: float(np.mean([x[k] for x in L_]))
        rep_rows.append(dict(task=lab, model=NAMES.get(m, m), leak_acc=mean(leak_r, "acc"), leak_auroc=mean(leak_r, "auroc"), leak_f1=mean(leak_r, "f1"),
                             ext_acc=mean(hon, "acc"), ext_auroc=mean(hon, "auroc"), ext_sens=mean(hon, "sens"), ext_f1=mean(hon, "f1")))
for lab, d in [("T1 APL-AML, diagnosis on cells", "finetune_t1"), ("T2 APL-AML, blast detection", "finetune_t2"), ("T4 C-NMC, cells", "finetune_cnmc")]:
    f, fi = R / d / "predictions.csv.gz", R / (d + "_image") / "predictions.csv.gz"
    if f.exists():
        P = pd.read_csv(f); h = acc_auc(P[(P.split == "grouped") & (P.set == "test")])
        lk = acc_auc(pd.read_csv(fi).query("split == 'image' and set == 'cv'")) if fi.exists() else None
        rep_rows.append(dict(task=lab, model="Fine-tuned EfficientNet-B0", leak_acc=lk["acc"] if lk else np.nan, leak_auroc=lk["auroc"] if lk else np.nan,
                             leak_f1=lk["f1"] if lk else np.nan, ext_acc=h["acc"], ext_auroc=h["auroc"], ext_sens=h["sens"], ext_f1=h["f1"]))
if rep_rows:
    RP = pd.DataFrame(rep_rows); order = {k: i for i, (k, _, _) in enumerate(SRC)}
    RP = RP.sort_values("task", key=lambda c: c.map(order), kind="stable"); RP.to_csv(TAB / "reported_vs_honest.csv", index=False)
    facts["reported_vs_honest"] = RP.round(4).to_dict("records")
    f_ = lambda v: "--" if np.isnan(v) else f"{v:.3f}"
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{The same models evaluated the way many studies report them (image-level 5-fold CV, accuracy at a 0.5 threshold) and on patients they have never seen (external set). Mean over seeds.}",
         r"\label{tab:reported}", r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llccccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{Image-level CV (leaky)} & \multicolumn{4}{c}{External set (new patients)} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-9}",
         r"Task & Model & Accuracy & AUROC & F1 & Accuracy & AUROC & Recall & F1 \\", r"\midrule"]
    for t, grp in RP.groupby("task", sort=False):
        for i, r in enumerate(grp.itertuples()):
            L.append(f"{t if i == 0 else ''} & {r.model} & {f_(r.leak_acc)} & {f_(r.leak_auroc)} & {f_(r.leak_f1)} & {r.ext_acc:.3f} & {r.ext_auroc:.3f} & {r.ext_sens:.3f} & {r.ext_f1:.3f} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table*}"]
    (TAB / "tab_reported.tex").write_text("\n".join(L))
    print(RP.round(3).to_string())

# ------------------------------------------------------------------ local recalibration and decision curves
LR_ = R / "local_recalibration"
if (LR_ / "local_recalibration.csv").exists():
    rc = pd.read_csv(LR_ / "local_recalibration.csv"); dc = pd.read_csv(LR_ / "decision_curves.csv"); pvv = pd.read_csv(LR_ / "predictive_values.csv")
    facts["recal"] = rc.round(4).to_dict("records"); facts["predictive_values"] = pvv.round(4).to_dict("records")
    facts["dca10"] = dc[(dc.prevalence == 0.10) & dc.pt.isin([0.02, 0.03, 0.05, 0.1])].round(4).to_dict("records")
    show = ["Macenko + frozen CNN, no PCA", "Secondary: Macenko + frozen CNN + PCA-64", "Engineered, pooled LR", "Engineered, attention MIL",
            "Macenko + fine-tuned EfficientNet-B0", "Primary: pre-specified ensemble"]
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Can a new laboratory restore the operating point with a few local cases? $k$ confirmed local patients ($k/2$ APL, $k/2$ AML) are drawn from the Validation cohort and the remaining $24-k$ patients are scored (2,000 random draws). R0: development threshold. R2: development threshold lowered, if needed, to just below the lowest score of the local APL cases. P(miss): probability that at least one APL patient in the test set is missed.}",
         r"\label{tab:recal}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccccccccc}", r"\toprule",
         r" & \multicolumn{4}{c}{R0, $k=0$} & \multicolumn{4}{c}{R2, $k=4$} & \multicolumn{4}{c}{R2, $k=8$} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
         r"Model & Recall & Spec. & F1 & P(miss) & Recall & Spec. & F1 & P(miss) & Recall & Spec. & F1 & P(miss) \\", r"\midrule"]
    for m in show:
        q = rc[rc.model == m]
        if q.empty:
            continue
        cells = []
        for k, rule in [(0, "R0 development threshold"), (4, "R2 safety shift"), (8, "R2 safety shift")]:
            z = q[(q.k == k) & (q.rule == rule)].iloc[0]
            cells += [f"{z.sens:.2f}", f"{z.spec:.2f}", f"{z.f1:.2f}", f"{z.p_any_miss:.2f}"]
        L.append((r"\textbf{" + m + "}" if m.startswith("Primary") else m) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TAB / "tab_recal.tex").write_text("\n".join(L))
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 3.0))
    r1 = rc[rc.rule == "R1 local cases only"].groupby("k").sens.mean()
    m_ = np.arange(1, 21)
    ax[0].plot(2 * m_, m_ / (m_ + 1), color=GREY, lw=1, label="Theory: $m/(m+1)$")
    ax[0].plot(r1.index, r1.values, "o", color=ORANGE, ms=4, label="Observed (all models)")
    ax[0].axhline(0.95, color=RED, ls=":", lw=0.8); ax[0].text(2, 0.955, "target 0.95", color=RED, fontsize=6)
    ax[0].set_xlabel("Local cases $k$ ($m=k/2$ APL)"); ax[0].set_ylabel("Expected recall"); ax[0].set_ylim(0.4, 1.0); ax[0].legend(frameon=False, fontsize=6.2, loc="lower right")
    ax[0].set_title("(a) Threshold from local cases only", loc="left", fontsize=8)
    cols = [GREY, "#7FA7D9", "#6A9F58", GREEN, ORANGE, RED]
    for m, c in zip(show, cols):
        q = rc[(rc.model == m) & ((rc.rule == "R2 safety shift") | (rc.k == 0))].sort_values("k")
        ax[1].plot(q.k, q.sens, "o-", color=c, ms=3, lw=1.6 if m.startswith("Primary") else 1.0, label=m.replace("Secondary: ", "Secondary: ").replace("Primary: pre-specified ensemble", "Primary (ensemble)"))
        ax[1].plot(q.k, q.spec, "s--", color=c, ms=2.5, lw=0.8)
    ax[1].set_xlabel("Local cases $k$"); ax[1].set_ylabel("Recall (solid) / specificity (dashed)"); ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_title("(b) Safety-shift rule (R2)", loc="left", fontsize=8)
    hl = ax[1].get_legend_handles_labels()
    d10 = dc[dc.prevalence == 0.10]
    for m, c in zip(show, cols):
        q = d10[(d10.model == m) & (d10.pt <= 0.3)]
        ax[2].plot(q.pt, q.net_benefit, color=c, lw=1.6 if m.startswith("Primary") else 1.0)
    q = d10[d10.model == show[0]]
    q = q[q.pt <= 0.3]
    ax[2].plot(q.pt, q.treat_all, color="black", lw=0.8, ls="--", label="Treat all"); ax[2].axhline(0, color="black", lw=0.6, label="Treat none")
    ax[2].set_ylim(-0.02, 0.1); ax[2].set_xlabel("Threshold probability"); ax[2].set_ylabel("Net benefit"); ax[2].legend(frameon=False, fontsize=6)
    ax[2].set_title("(c) Net benefit, 10% prevalence", loc="left", fontsize=8)
    fig.tight_layout(rect=(0, 0.15, 1, 1)); fig.legend(*hl, loc="lower center", ncol=3, frameon=False, fontsize=6.2)
    fig.savefig(OUT / "figures" / "fig_recal.pdf"); fig.savefig(OUT / "figures" / "fig_recal.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ pooled 106-patient CV
if (R / "pooled_cv" / "pooled_cv_summary.csv").exists():
    pc = pd.read_csv(R / "pooled_cv" / "pooled_cv_summary.csv")
    if (R / "pooled_cv_nw" / "pooled_cv_summary.csv").exists():
        nw = pd.read_csv(R / "pooled_cv_nw" / "pooled_cv_summary.csv")
        nw = nw[nw.config.str.endswith("_nw")]
        pc = pd.concat([pc, nw], ignore_index=True)
    # reference for the Delta column: unreduced embedding under the same scaling
    pc["d_ref"] = np.where(pc.config.str.endswith("_nw"), pc.get("d_vs_eff_raw_nw"), pc.get("d_vs_eff_raw"))
    pc["p_ref"] = np.where(pc.config.str.endswith("_nw"), pc.get("p_vs_eff_raw_nw"), pc.get("p_vs_eff_raw"))
    pc["p_ref_holm"] = holm(pc.p_ref.values)
    pc.to_csv(TAB / "pooled_cv.csv", index=False); facts["pooled_cv"] = pc.round(4).to_dict("records")
    PNM = {"promyelocyte_fraction": "Promyelocyte fraction (biomarker)", "eng_raw": "Engineered, pooled LR", "eff_raw": "EfficientNet-B0, no PCA",
           "eff_pca16": "EfficientNet-B0 + PCA-16", "eff_pca32": "EfficientNet-B0 + PCA-32", "eff_pca64": "EfficientNet-B0 + PCA-64",
           "fusion_pca32": "Engineered + PCA-32 fusion", "mil_eng": "Engineered, attention MIL",
           "eff_raw_nw": "EfficientNet-B0, no PCA$^{\\ddagger}$", "eff_pca16_nw": "EfficientNet-B0 + PCA-16$^{\\ddagger}$",
           "eff_pca32_nw": "EfficientNet-B0 + PCA-32$^{\\ddagger}$", "eff_pca64_nw": "EfficientNet-B0 + PCA-64$^{\\ddagger}$"}
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Secondary analysis with more statistical power: repeated (3$\times$) 5-fold patient-level CV over all 106 APL-AML patients. This is an internal analysis, not an external validation. $\Delta$: AUROC difference vs.\ the unreduced EfficientNet-B0 embedding (paired patient bootstrap on repeat-averaged scores). Recall, specificity and F1 at a threshold chosen for sensitivity $\ge0.95$ inside each training fold.}",
         r"\label{tab:pooled}", r"\setlength{\tabcolsep}{2.5pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Model & AUROC [95\% CI] & $\Delta$ vs.\ no PCA & Recall (missed/34) & Spec. & F1 \\", r"\midrule"]
    for r in pc.itertuples():
        d = "--" if np.isnan(r.d_ref) else f"{r.d_ref:+.3f}" + ("*" if r.p_ref < 0.05 else "") + (r"$^{\dagger}$" if r.p_ref_holm < 0.05 else "")
        L.append(f"{PNM.get(r.config, r.config)} & {ci(r.auroc_avg_scores, r.lo, r.hi)} & {d} & {r.sens:.2f} ({r.fn:.1f}) & {r.spec:.2f} & {r.f1:.2f} \\\\")
    L += [r"\bottomrule", r"\multicolumn{6}{l}{\footnotesize * $p<0.05$ unadjusted; $^{\dagger}$ after Holm correction. $^{\ddagger}$ PCA scores at natural scale; $\Delta$ vs.\ no PCA with the same scaling.}", r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_pooled.tex").write_text("\n".join(L))
    print(pc.round(3).to_string())
(TAB / "facts.json").write_text(json.dumps(facts, indent=1, default=float))

# ================================================================== IMPACT LADDER: what each addition changes (external sets)
# one canonical source per number: cell rows from the 3-seed cell benchmark / fine-tune summaries, patient rows from
# feature_engineering.csv (natural-scale PCA), the 3-seed patient benchmark (P1, P4) and the fine-tune / ensemble summaries
def ladder_rows():
    rows = []
    def add(ds, level, step, auroc, lo, hi, acc, sens, fn, spec, f1, npos):
        rows.append(dict(dataset=ds, level=level, step=step, auroc=auroc, lo=lo, hi=hi, acc=acc, sens=sens, fn=fn, spec=spec, f1=f1, npos=npos))
    for ds, ftdir in [("APL-AML", "finetune_t1"), ("C-NMC", "finetune_cnmc")]:
        src = leak[leak.task == TASKS["apl_cell_diagnosis"]].rename(columns={"ext_lo": "lo", "ext_hi": "hi"}) if ds == "APL-AML" else cn
        for m, lab in [("EfficientNet-B0 + LR", "Frozen CNN + LR"), ("Fusion + LR", "Frozen CNN + engineered (fusion)")]:
            r = src[src.model == m].iloc[0]
            add(ds, "cell", lab, r.ext, r.lo, r.hi, r.acc, r.sens, np.nan, r.spec, r.f1, np.nan)
        for d, lab in [(ftdir, "+ fine-tuning"), (ftdir + "_macenko", "+ fine-tuning + Macenko")]:
            f = R / d / "summary.csv"
            if f.exists():
                c = pd.read_csv(f).query("level == 'cell'").iloc[0]
                add(ds, "cell", lab, c.ext_auroc, c.ext_lo, c.ext_hi, c.ext_acc, c.ext_sens, np.nan, c.ext_spec, c.ext_f1, np.nan)
        npos = 12 if ds == "APL-AML" else 17
        fe_nw = R / ("apl_fe_nw" if ds == "APL-AML" else "cnmc_fe_nw") / "feature_engineering.csv"
        q = pd.read_csv(fe_nw).set_index("config")
        for c, lab in [("eff_raw_nw", "Frozen CNN, pooled + LR"), ("eff_pca64_nw", "+ PCA-64")]:
            r = q.loc[c]; add(ds, "patient", lab, r.ext_auroc, r.ext_lo, r.ext_hi, r.sens95_acc, r.sens95_sens, r.sens95_fn, r.sens95_spec, r.sens95_f1, npos)
        if ds == "APL-AML":
            q = pd.read_csv(R / "apl_fe_macenko" / "feature_engineering.csv").set_index("config")
            for c, lab in [("eff_raw_nw", "Frozen CNN + Macenko"), ("eff_pca64_nw", "Frozen CNN + Macenko + PCA-64 (secondary)")]:
                r = q.loc[c]; add(ds, "patient", lab, r.ext_auroc, r.ext_lo, r.ext_hi, r.sens95_acc, r.sens95_sens, r.sens95_fn, r.sens95_spec, r.sens95_f1, npos)
        for d, lab in [(ftdir, "Fine-tuned CNN, mean of cells"), (ftdir + "_macenko", "Fine-tuned CNN + Macenko")]:
            f = R / d / "summary.csv"
            if f.exists():
                c = pd.read_csv(f).query("level == 'patient'").iloc[0]
                add(ds, "patient", lab, c.ext_auroc, c.ext_lo, c.ext_hi, c.ext_acc, c.ext_sens, c.ext_fn, c.ext_spec, c.ext_f1, npos)
        ps_ = pd.read_csv(R / ("v2_patient" if ds == "APL-AML" else "cnmc_patient") / "summary.csv").query("eval == 'test'").set_index("model")
        for m, lab in [("P1", "Engineered, pooled + LR"), ("P4", "Engineered, attention MIL")]:
            r = ps_.loc[m]; add(ds, "patient", lab, r.auroc, r.auroc_lo, r.auroc_hi, r.acc, r.sens, r.fn, r.spec, r.f1, npos)
        e = pd.read_csv(R / ("ensemble_apl_macenko" if ds == "APL-AML" else "ensemble_cnmc") / "ensemble.csv").set_index("model").loc["Ensemble"]
        add(ds, "patient", "Primary: ensemble of the three members above" if ds == "C-NMC" else "Primary: ensemble (Macenko fine-tuned + engineered)",
            e.ext_auroc, e.lo, e.hi, e.acc, e.sens, e.fn, e.spec, e.f1, npos)
        if ds == "APL-AML":
            for d, lab in [("ensemble_apl", "Sensitivity: primary without Macenko"), ("ensemble_apl_all_macenko", "Sensitivity: primary, engineered also on Macenko images")]:
                f = R / d / "ensemble.csv"
                if f.exists():
                    e = pd.read_csv(f).set_index("model").loc["Ensemble"]
                    add(ds, "patient", lab, e.ext_auroc, e.lo, e.hi, e.acc, e.sens, e.fn, e.spec, e.f1, npos)
    return pd.DataFrame(rows)


LD = ladder_rows()
LD.to_csv(TAB / "impact_ladder.csv", index=False); facts["impact_ladder"] = LD.round(4).to_dict("records")
L = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{What each addition changes on new patients (external sets; cells: T1/T4, patients: T3/T5). Each block starts from a frozen ImageNet CNN; $\Delta$: AUROC change relative to the first row of the block. Threshold metrics at the operating point fixed on development data (target recall 0.95); patient rows give exact 95\% CIs for recall and specificity. Cell rows: mean of three seeds (frozen) or one seed (fine-tuned). The ensemble averages percentile scores of the fine-tuned CNN (with Macenko on APL-AML), engineered pooled LR and engineered attention MIL.}",
     r"\label{tab:ladder}", r"\setlength{\tabcolsep}{3.5pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lllcccccccc}", r"\toprule",
     r"Dataset & Level & Model & AUROC [95\% CI] & $\Delta$AUROC & Accuracy & Recall & Missed & Specificity & F1 \\", r"\midrule"]
for (ds, lv), grp in LD.groupby(["dataset", "level"], sort=False):
    base = grp.auroc.iloc[0]
    for i, r in enumerate(grp.itertuples()):
        miss = "--" if np.isnan(r.fn) else f"{int(r.fn)}/{int(r.npos)}"
        st = r"\textbf{" + r.step + "}" if r.step.startswith(("Primary", "Frozen CNN + Macenko + PCA")) else r.step
        L.append(f"{ds if i == 0 else ''} & {('Cell' if lv == 'cell' else 'Patient') if i == 0 else ''} & {st} & {ci(r.auroc, r.lo, r.hi)} & "
                 + ("--" if i == 0 else f"{r.auroc - base:+.3f}") + f" & {r.acc:.2f} & " + (f"{r.sens:.2f}" if np.isnan(r.npos) else cpr(r.sens, int(r.npos))) + f" & {miss} & " + (f"{r.spec:.2f}" if np.isnan(r.npos) else cpr(r.spec, 12 if ds == "APL-AML" else 14)) + f" & {r.f1:.2f} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}}", r"\end{table*}"]
(TAB / "tab_ladder.tex").write_text("\n".join(L))
groups = list(LD.groupby(["dataset", "level"], sort=False))
fig, ax = plt.subplots(2, 2, figsize=(7.16, 4.6), gridspec_kw=dict(height_ratios=[4, 8]))
for a_, ((ds, lv), grp) in zip(ax.T.ravel(), groups):
    y = np.arange(len(grp))[::-1]
    a_.errorbar(grp.auroc, y, xerr=[grp.auroc - grp.lo, grp.hi - grp.auroc], fmt="o", color=BLUE, ms=4, elinewidth=0.9, capsize=2, label="AUROC [95% CI]")
    a_.plot(grp.f1, y + 0.18, "s", color=ORANGE, ms=3.5, label="F1, fixed threshold")
    a_.plot(grp.sens, y - 0.18, "v", color=RED, ms=3.5, label="Recall, fixed threshold")
    a_.set_yticks(y, [s_.replace(" (secondary)", "\n(secondary)").replace("Primary: ensemble ", "Primary ensemble\n").replace("Primary: ensemble of the three members above", "Primary ensemble") for s_ in grp.step], fontsize=6.2)
    a_.axvline(grp.auroc.iloc[0], color="#BBB", lw=0.7, ls=":")
    a_.set_xlim(0.3, 1.02); a_.set_title(f"{ds}, {'cells' if lv == 'cell' else 'patients'}", fontsize=7.5, loc="left")
for a_ in ax[1]:
    a_.set_xlabel("External set")
hl_ = ax[0, 0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.legend(*hl_, loc="lower center", ncol=3, frameon=False, fontsize=6.5); fig.savefig(OUT / "figures" / "fig_ladder.pdf"); fig.savefig(OUT / "figures" / "fig_ladder.png", dpi=200); plt.close(fig)
print(LD.round(3).to_string())
(TAB / "facts.json").write_text(json.dumps(facts, indent=1, default=float))

# ================================================================== REVISION 1, part 2: preprocessing, epochs, trade-off, C-NMC EDA
import shutil
# ---- preprocessing comparison (frozen CNN on the pilot subset; APL-AML) + C-NMC check
pp = []
for d, nm in [("preproc_test", None), ("preproc_clahe", None), ("preproc_reinhard", None), ("preproc_macenko", None), ("preproc_macenko_pp", None)]:
    f = R / d / "preproc_test.csv"
    if f.exists():
        pp.append(pd.read_csv(f))
if pp:
    PP = pd.concat(pp).drop_duplicates("variant", keep="last").set_index("variant")
    PP.to_csv(TAB / "preprocessing.csv"); facts["preprocessing"] = PP.round(4).reset_index().to_dict("records")
    PN = {"none": "None", "unsharp": "Edge sharpening (unsharp mask)", "clahe": "Contrast enhancement (CLAHE)", "reinhard": "Reinhard colour normalization",
          "macenko": r"\textbf{Macenko stain normalization (selected)}", "macenko_pp": "Macenko, stain vectors per patient"}
    cn = pd.read_csv(R / "preproc_cnmc" / "preproc_test.csv").set_index("variant") if (R / "preproc_cnmc" / "preproc_test.csv").exists() else None
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Preprocessing before feature extraction (frozen EfficientNet-B0, up to 60 cells per patient, same folds and classifiers for every variant). Cohort shift: AUROC for assigning control-class cells to their cohort (0.5 = no detectable shift). Gap: development CV minus external AUROC. The rule fixed in advance selected the method that reduced the shift without lowering external performance. C-NMC, which showed no shift, is given for comparison.}",
         r"\label{tab:preproc}", r"\setlength{\tabcolsep}{3.2pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{Cells (T1)} & \multicolumn{3}{c}{Patients (T3)} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Preprocessing & Cohort shift & Dev.\ CV (SD; worst fold) & External & Gap & Dev.\ CV (SD; worst fold) & External & Gap \\", r"\midrule",
         r"\multicolumn{8}{l}{\emph{APL-AML (6,360 cells, 106 patients)}} \\"]
    for v in ["none", "unsharp", "clahe", "reinhard", "macenko", "macenko_pp"]:
        if v not in PP.index:
            continue
        r = PP.loc[v]
        L.append(f"{PN[v]} & {r.cohort_id_auroc:.3f} & {r.cell_cv_mean:.3f} ({r.cell_cv_sd:.3f}; {r.cell_cv_min:.3f}) & {r.cell_ext:.3f} & {r.cell_cv_mean - r.cell_ext:.3f} & "
                 f"{r.patient_cv_mean:.3f} ({r.patient_cv_sd:.3f}; {r.patient_cv_min:.3f}) & {r.patient_ext:.3f} & {r.patient_cv_mean - r.patient_ext:.3f} \\\\")
    if cn is not None:
        L += [r"\midrule", r"\multicolumn{8}{l}{\emph{C-NMC (3,773 cells, 73 subjects; no shift detected)}} \\"]
        for v in ["none", "macenko"]:
            r = cn.loc[v]
            L.append(f"{PN[v].replace(chr(92)+'textbf{','').replace(' (selected)}','')} & {r.cohort_id_auroc:.3f} & {r.cell_cv_mean:.3f} ({r.cell_cv_sd:.3f}; {r.cell_cv_min:.3f}) & {r.cell_ext:.3f} & {r.cell_cv_mean - r.cell_ext:.3f} & "
                     f"{r.patient_cv_mean:.3f} ({r.patient_cv_sd:.3f}; {r.patient_cv_min:.3f}) & {r.patient_ext:.3f} & {r.patient_cv_mean - r.patient_ext:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TAB / "tab_preproc.tex").write_text("\n".join(L))
    print(PP[["cohort_id_auroc", "cell_cv_mean", "cell_ext", "patient_cv_mean", "patient_ext"]].round(3).to_string())

# ---- fixed epochs vs nested early stopping (attention MIL)
es_rows = []
for ds, fx, es in [("APL-AML", "v2_patient", "v2_patient_es"), ("C-NMC", "cnmc_patient", "cnmc_patient_es")]:
    if not (R / es / "summary.csv").exists():
        continue
    a_ = pd.read_csv(R / fx / "summary.csv"); b_ = pd.read_csv(R / es / "summary.csv")
    ce = pd.read_csv(R / es / "chosen_epochs.csv") if (R / es / "chosen_epochs.csv").exists() else None
    for m, nm in [("P4", "Engineered"), ("P5", "EfficientNet-B0"), ("P6", "Fusion")]:
        x = a_[(a_.model == m) & (a_["eval"] == "test")]; z = b_[(b_.model == m) & (b_["eval"] == "test")]
        if x.empty or z.empty:
            continue
        x, z = x.iloc[0], z.iloc[0]
        cvx = a_[(a_.model == m) & (a_["eval"] == "cv")].auroc; cvz = b_[(b_.model == m) & (b_["eval"] == "cv")].auroc
        ep = ce[ce.model == m].epochs if ce is not None else pd.Series(dtype=float)
        es_rows.append(dict(dataset=ds, features=nm, fixed_ext=x.auroc, es_ext=z.auroc, fixed_cv_sd=cvx.std(), es_cv_sd=cvz.std(),
                            fixed_cv_min=cvx.min(), fixed_cv_max=cvx.max(), es_cv_min=cvz.min(), es_cv_max=cvz.max(),
                            fixed_recall=x.sens, es_recall=z.sens, fixed_f1=x.f1, es_f1=z.f1, fixed_ece=x.ece, es_ece=z.ece,
                            ep_med=ep.median() if len(ep) else np.nan, ep_min=ep.min() if len(ep) else np.nan, ep_max=ep.max() if len(ep) else np.nan))
if es_rows:
    ES = pd.DataFrame(es_rows); ES.to_csv(TAB / "early_stopping.csv", index=False); facts["early_stopping"] = ES.round(4).to_dict("records")
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Attention MIL trained for a fixed 40 epochs or with the number of epochs chosen by nested 3-fold CV inside each training fold (lowest mean validation loss). Development CV: range of the AUROC over three seeds. External AUROC, F1 at the development threshold and calibration error (ECE). Chosen epochs: over all folds and seeds.}",
         r"\label{tab:epochs}", r"\setlength{\tabcolsep}{2.2pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llccccccccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{Dev.\ CV AUROC (seed range)} & \multicolumn{2}{c}{External AUROC} & \multicolumn{2}{c}{F1} & \multicolumn{2}{c}{ECE} & Chosen epochs \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"Dataset & Features & Fixed & Nested & Fixed & Nested & Fixed & Nested & Fixed & Nested & median (range) \\", r"\midrule"]
    for ds, grp in ES.groupby("dataset", sort=False):
        for i, r in enumerate(grp.itertuples()):
            ep = "--" if np.isnan(r.ep_med) else f"{r.ep_med:.0f} ({r.ep_min:.0f}--{r.ep_max:.0f})"
            L.append(f"{ds if i == 0 else ''} & {r.features} & {r.fixed_cv_min:.3f}--{r.fixed_cv_max:.3f} & {r.es_cv_min:.3f}--{r.es_cv_max:.3f} & {r.fixed_ext:.3f} & {r.es_ext:.3f} & {r.fixed_f1:.2f} & {r.es_f1:.2f} & {r.fixed_ece:.2f} & {r.es_ece:.2f} & {ep} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_epochs.tex").write_text("\n".join(L))
    print(ES.round(3).to_string())

# ---- C-NMC EDA (composition table + figure)
if (R / "cnmc_eda" / "class_balance.csv").exists():
    cb = pd.read_csv(R / "cnmc_eda" / "class_balance.csv")
    shutil.copy(R / "cnmc_eda" / "fig_eda.pdf", OUT / "figures" / "fig_eda_cnmc.pdf"); shutil.copy(R / "cnmc_eda" / "fig_eda.png", OUT / "figures" / "fig_eda_cnmc.png")
    cid = pd.read_csv(R / "cnmc_eda" / "cohort_id.csv"); facts["cnmc_eda_cohort_id"] = cid.round(4).to_dict("records")
    sh = pd.read_csv(R / "cnmc_eda" / "shift_smd.csv"); facts["cnmc_eda_shift_by_group"] = sh.groupby("group").abs_smd.median().round(4).to_dict()
    L = [r"\begin{table}[!t]", r"\centering", r"\caption{Composition of the C-NMC 2019 data used here (official training folds).}", r"\label{tab:dataset_cnmc}",
         r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llrrr}", r"\toprule", r"Set & Class & Subjects & Cells & Cells per subject, median (range) \\", r"\midrule"]
    for r in cb.itertuples():
        L.append(f"{r.cohort} & {'ALL' if r.diagnosis == 'ALL' else 'Normal B cells'} & {r.patients} & {r.cells:,} & {getattr(r, '_5'):.0f} ({r.min:.0f}--{r.max:.0f}) \\\\")
    L += [r"\midrule", f"Total & & {cb.patients.sum()} & {cb.cells.sum():,} & \\\\", r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_dataset_cnmc.tex").write_text("\n".join(L))
(TAB / "facts.json").write_text(json.dumps(facts, indent=1, default=float))

# ================================================================== performance / complexity trade-off (APL-AML)
# measured on our CPU (2 vCPUs): seconds per cell for each image-level step
T_NORM = 30.5 * 60 / 25724          # Macenko, per cell, 1 thread
T_CNN = 21.9 * 60 / 25724           # EfficientNet-B0 forward pass, per cell, 2 threads
T_ENG = 16.1 * 60 / 25724           # engineered features, per cell, 2 processes
N_CELLS = 500
PRIMARY_TRAIN_S = 631.0   # mean seconds per fold of fine-tuning (log of finetune_t1_macenko); excludes the one-off cache (25 min)
cost = {}
for k, d in [(0, "cost_k0"), (64, "cost_k64")]:
    f = R / d / "cost.csv"
    if f.exists():
        cost[k] = pd.read_csv(f).iloc[0]
fm = R / "apl_fe_macenko" / "feature_engineering.csv"
if fm.exists() and cost:
    fe_m = pd.read_csv(fm).set_index("config")
    rows = []
    def perf(r):
        return dict(auroc=r.ext_auroc, acc=r.sens95_acc, sens=r.sens95_sens, fn=r.sens95_fn, spec=r.sens95_spec, f1=r.sens95_f1, npv=r.sens95_npv)
    if (R / "ensemble_apl_macenko" / "ensemble.csv").exists():
        e = pd.read_csv(R / "ensemble_apl_macenko" / "ensemble.csv").set_index("model").loc["Ensemble"]
        rows.append(dict(model="Primary: Macenko + fine-tuned CNN + engineered, ensemble", auroc=e.ext_auroc, acc=e.acc, sens=e.sens, fn=e.fn, spec=e.spec, f1=e.f1, npv=e.npv,
                         storage=5120 + 93 * 4, params=4007548 + 1281 + (93 * 128 + 128 + 2 * (128 * 64 + 64) + 64 + 1 + 128 + 1) + (5 * 93 + 1),
                         train_s=PRIMARY_TRAIN_S,
                         t_patient=N_CELLS * (T_NORM + T_CNN + T_ENG) + (cost[0].mil_predict_ms_per_patient + cost[0].lr_predict_ms_per_patient) / 1000))
    for c, lab, k in [("eff_raw_nw", "Macenko + frozen CNN, no PCA", 0), ("eff_pca64_nw", "Secondary: Macenko + frozen CNN + PCA-64", 64)]:
        if c in fe_m.index and k in cost:
            q = cost[k]
            rows.append(dict(model=lab, **perf(fe_m.loc[c]), storage=q.storage_bytes_per_cell, params=q.lr_params, train_s=q.lr_train_s + q.pca_fit_s,
                             t_patient=N_CELLS * (T_NORM + T_CNN) + (q.pca_projection_ms_per_patient + q.lr_predict_ms_per_patient) / 1000))
    TO = pd.DataFrame(rows); TO.to_csv(TAB / "tradeoff.csv", index=False); facts["tradeoff"] = TO.round(4).to_dict("records")
    facts["cost_steps_s_per_cell"] = dict(norm=T_NORM, cnn=T_CNN, eng=T_ENG)
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Performance and complexity on APL-AML (24 external patients). Time per patient assumes 500 cells on our CPU (2 vCPUs): Macenko normalization " + f"{T_NORM:.3f}" + r" s, CNN forward pass " + f"{T_CNN:.3f}" + r" s and engineered features " + f"{T_ENG:.3f}" + r" s per cell; classifier times are measured. Storage: bytes kept per cell after feature extraction. Parameters and training time refer to the patient-level classifier (including PCA fitting).}",
         r"\label{tab:tradeoff}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccccccccc}", r"\toprule",
         r" & \multicolumn{7}{c}{Performance (external, fixed threshold)} & \multicolumn{4}{c}{Complexity} \\", r"\cmidrule(lr){2-8}\cmidrule(lr){9-12}",
         r"Pipeline & AUROC & Acc. & Recall & Missed & Spec. & F1 & NPV & Storage/cell & Parameters & Training (s) & Time/patient (s) \\", r"\midrule"]
    for r in TO.itertuples():
        L.append(f"{r.model} & {r.auroc:.3f} & {r.acc:.2f} & {cpr(r.sens, 12)} & {int(r.fn)}/12 & {cpr(r.spec, 12)} & {r.f1:.2f} & {npv_s(r.npv, r.fn, r.spec)} & {int(r.storage):,} B & "
                 + ("--" if np.isnan(r.params) else f"{int(r.params):,}") + " & " + ("--" if np.isnan(r.train_s) else f"{r.train_s:.2f}") + f" & {r.t_patient:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TAB / "tab_tradeoff.tex").write_text("\n".join(L))
    print(TO.round(3).to_string())
(TAB / "facts.json").write_text(json.dumps(facts, indent=1, default=float))

# ================================================================== REVISION 2: shift test, scan-time confound, calibration, C-NMC bag size
RS = R / "revision_stats"
if (RS / "shift" / "shift_test.csv").exists():
    sh = pd.read_csv(RS / "shift" / "shift_test.csv"); sh.to_csv(TAB / "shift_test.csv", index=False); facts["shift_test"] = sh.round(4).to_dict("records")
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Label-free shift test. AUROC of a classifier that assigns control-class cells (AML or normal B cells, up to 6,000) to the external or development set, patient-grouped 5-fold CV. Null: cohort labels permuted between patients. The test reads unlabelled external images and is therefore transductive; it uses no external diagnoses.}",
         r"\label{tab:shift}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llcccc}", r"\toprule",
         r"Dataset & Representation & AUROC & Null mean (95th pct.) & $p$ & Perm. \\", r"\midrule"]
    for ds, grp in sh.groupby("dataset", sort=False):
        for i, r in enumerate(grp.itertuples()):
            pv = f"$\\leq${1 / (r.permutations + 1):.3f}" if r.p_value <= 1 / (r.permutations + 1) + 1e-12 else f"{r.p_value:.3f}"
            L.append(f"{ds if i == 0 else ''} & {r.representation} & {r.cohort_auroc:.3f} & {r.null_mean:.3f} ({r.null_95:.3f}) & {pv} & {r.permutations} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_shift.tex").write_text("\n".join(L)); print(sh.round(3).to_string())

if (R / "date_confound" / "date_confound.csv").exists():
    dcf = pd.read_csv(R / "date_confound" / "date_confound.csv", index_col=0).iloc[:, 0]
    facts["date_confound"] = dcf.round(4).to_dict()
    g = lambda k: dcf.get(k, np.nan)
    reps = [("engineered", "Engineered (all)"), ("stain_colour_only", "Engineered: stain and colour"), ("shape_texture_only", "Engineered: shape + texture"),
            ("efficientnet", "EfficientNet-B0 (frozen)"), ("engineered_macenko_images", "Engineered, on Macenko images")]
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Scan-time confounding in the APL-AML Discovery cohort (82 patients). Time proxy: median analyzer cell number of a patient's signed cells. Diagnosis models: pooled-statistics LR, patient-level 5-fold CV $\times$ 3 seeds, on all patients and on the patients scanned while both diagnoses were being scanned (time window). Score adds to time: likelihood-ratio test of the out-of-fold score added to a logistic model on the time rank. Time from features: AUROC for early vs.\ late scan among AML patients only.}",
         r"\label{tab:dateconf}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcccc}", r"\toprule",
         r"Representation & CV AUROC, all & CV AUROC, time window & Score adds to time, $p$ & Time from features (AML) \\", r"\midrule"]
    tkey = {"engineered": "engineered_all", "stain_colour_only": "stain_colour", "shape_texture_only": "shape_texture", "efficientnet": "efficientnet",
            "engineered_macenko_images": None}
    for k, lab in reps:
        if np.isnan(g(f"{k}_cv_auroc_full")):
            continue
        tf = g(f"time_from_{tkey[k]}_auroc_AML_only") if tkey[k] else np.nan
        pv = g(f"{k}_score_adds_to_time_LR_p")
        L.append(f"{lab} & {g(k + '_cv_auroc_full'):.3f} & {g(k + '_cv_auroc_time_window'):.3f} & " + (f"{pv:.3f}" if pv >= 0.001 else "$<$0.001") + " & " + ("--" if np.isnan(tf) else f"{tf:.3f}") + r" \\")
    L += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{Time proxy alone: Validation vs.\ Discovery {g('proxy_auroc_validation_vs_discovery'):.3f}; APL vs.\ AML in Discovery {g('proxy_auroc_apl_within_discovery'):.3f}; within the window {g('proxy_auroc_apl_within_window'):.3f} ({int(g('window_patients'))} patients, {int(g('window_apl'))} APL).}} \\",
          r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_dateconf.tex").write_text("\n".join(L)); print(dcf.round(3).to_string())

if (RS / "calibration.csv").exists():
    cal = pd.read_csv(RS / "calibration.csv"); cal.to_csv(TAB / "calibration.csv", index=False); facts["calibration"] = cal.round(4).to_dict("records")
    shutil.copy(RS / "fig_reliability.pdf", OUT / "figures" / "fig_reliability.pdf"); shutil.copy(RS / "fig_reliability.png", OUT / "figures" / "fig_reliability.png")
    f_ = lambda v, d=2: "--" if pd.isna(v) else f"{v:.{d}f}"
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Calibration of patient-level probabilities on the external sets. Raw: model output (not defined for the percentile-based ensemble). Platt: logistic mapping fitted on development out-of-fold scores. Intercept: calibration-in-the-large (0 = ideal; positive = risk underestimated); slope: 1 = ideal, $<1$ = overconfident. Prevalence differs between development and external sets (APL-AML 27\% vs.\ 50\%; C-NMC 71\% vs.\ 55\%), which alone shifts the intercept.}",
         r"\label{tab:calibration}", r"\setlength{\tabcolsep}{2.6pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llccccccc}", r"\toprule",
         r" & & & \multicolumn{3}{c}{Raw output} & \multicolumn{3}{c}{Platt (development)} \\", r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
         r"Dataset & Model & AUROC & Brier & Intercept & Slope & Brier & Intercept & Slope \\", r"\midrule"]
    for ds, grp in cal.groupby("dataset", sort=False):
        for i, r in enumerate(grp.itertuples()):
            L.append(f"{ds if i == 0 else ''} & {r.model} & {r.auroc:.3f} & {f_(r.raw_brier, 3)} & {f_(r.raw_cal_intercept)} & {f_(r.raw_cal_slope)} & {r.platt_brier:.3f} & {r.platt_cal_intercept:.2f} & {r.platt_cal_slope:.2f} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_calibration.tex").write_text("\n".join(L))

if (RS / "cnmc_bag_size.csv").exists():
    bg = pd.read_csv(RS / "cnmc_bag_size.csv").set_index("set"); facts["cnmc_bag_size"] = bg.round(4).reset_index().to_dict("records")
    models = ["Engineered, pooled LR", "Engineered, attention MIL", "Fine-tuned EfficientNet-B0", "Primary ensemble"]
    pfmt = lambda v: "$<$0.001" if v < 0.001 else f"{v:.3f}"
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Does the number of cells per subject explain C-NMC results? Count AUROC: AUROC of the cell count alone for ALL vs.\ normal. For each model: AUROC of the subject score, its Spearman correlation with log(cell count), the AUROC of the score after regressing out log(count), and the likelihood-ratio $p$-value for the score added to log(count).}",
         r"\label{tab:bagsize}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Development (42)} & \multicolumn{2}{c}{External (31)} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Quantity & Value & & Value & \\", r"\midrule",
         f"Median cells per subject, ALL / normal & {bg.loc['Development', 'median_cells_ALL']:.0f} / {bg.loc['Development', 'median_cells_HEM']:.0f} & & {bg.loc['External', 'median_cells_ALL']:.0f} / {bg.loc['External', 'median_cells_HEM']:.0f} & \\\\",
         f"Count AUROC (Mann--Whitney $p$) & {bg.loc['Development', 'count_auroc']:.3f} ({bg.loc['Development', 'mannwhitney_p']:.2f}) & & {bg.loc['External', 'count_auroc']:.3f} ({bg.loc['External', 'mannwhitney_p']:.3f}) & \\\\", r"\midrule",
         r"Model: AUROC / $\rho$(score, log count) / adjusted AUROC / $p$ & & & & \\"]
    for m in models:
        cells = []
        for st in ["Development", "External"]:
            cells.append(f"{bg.loc[st, m + ': AUROC']:.3f} / {bg.loc[st, m + ': Spearman(score, log n)']:+.2f} / {bg.loc[st, m + ': AUROC of count-adjusted score']:.3f} / {pfmt(bg.loc[st, m + ': score adds to log n, LR p'])}")
        L.append(f"\\quad {m} & \\multicolumn{{2}}{{c}}{{{cells[0]}}} & \\multicolumn{{2}}{{c}}{{{cells[1]}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_bagsize.tex").write_text("\n".join(L))
(TAB / "facts.json").write_text(json.dumps(facts, indent=1, default=float))

# ================================================================== REVISION 3: independent external test (MLL), DinoBloom, SHAP, improvement pilot
from sklearn.metrics import roc_curve
MLLR = R / "mll_patient"
if (MLLR / "summary.csv").exists():
    NP, NN = 24, 105
    ms = pd.read_csv(MLLR / "summary.csv").query("eval == 'test'").set_index("model")
    ens = pd.read_csv(R / "mll_ensemble" / "ensemble.csv").set_index("model").loc["Ensemble"]
    ftm = pd.read_csv(R / "mll_finetune_macenko" / "summary.csv").query("level == 'patient'").iloc[0]
    fem = pd.read_csv(R / "mll_fe_macenko" / "feature_engineering.csv").set_index("config").loc["eff_pca64_nw"]
    rows = [("\\textbf{Primary pipeline (pre-specified ensemble)}", ens.ext_auroc, ens.lo, ens.hi, ens.sens, ens.fn, ens.spec, ens.f1, ens.npv, ens.acc)]
    for m, lab in [("P1", "Member: engineered, pooled LR"), ("P4", "Member: engineered, attention MIL")]:
        r = ms.loc[m]; rows.append((lab, r.auroc, r.auroc_lo, r.auroc_hi, r.sens, r.fn, r.spec, r.f1, r.npv, r.acc))
    rows.append(("Member: Macenko + fine-tuned CNN", ftm.ext_auroc, ftm.ext_lo, ftm.ext_hi, ftm.ext_sens, ftm.ext_fn, ftm.ext_spec, ftm.ext_f1, ftm.ext_npv, ftm.ext_acc))
    rows.append(("Secondary pipeline (Macenko + frozen CNN + PCA-64)", fem.ext_auroc, fem.ext_lo, fem.ext_hi, fem.sens95_sens, fem.sens95_fn, fem.sens95_spec, fem.sens95_f1, fem.sens95_npv, fem.sens95_acc))
    for m, lab in [("PD", "DinoBloom-S, pooled LR"), ("PDF", "DinoBloom-S + engineered, pooled LR")]:
        r = ms.loc[m]; rows.append((lab, r.auroc, r.auroc_lo, r.auroc_hi, r.sens, r.fn, r.spec, r.f1, r.npv, r.acc))
    MT_ = pd.DataFrame(rows, columns=["model", "auroc", "lo", "hi", "sens", "fn", "spec", "f1", "npv", "acc"]); MT_.to_csv(TAB / "mll_external.csv", index=False)
    facts["mll_external"] = MT_.round(4).to_dict("records")
    shm = pd.read_csv(R / "mll_external" / "shift_test.csv"); facts["mll_shift"] = shm.round(4).to_dict("records")
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Independent external test (pre-registered): models trained on the APL-AML Discovery cohort only and applied once to the AML-Cytomorphology\_MLL\_Helmholtz cohort (another laboratory, country, scanner and stain; 24 APL and 105 other AML patients, up to 200 cells each). Label-free shift test: AUROC " + f"{shm.cohort_auroc.iloc[0]:.3f}" + r" for telling MLL from Discovery AML cells by stain and color ($p\leq0.03$), so Macenko normalization was applied to the CNN branch. Threshold metrics at the Discovery operating point (recall $\geq$0.95), with exact 95\% CIs; NPV n/a: no patient called negative.}",
         r"\label{tab:mll}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Model & AUROC [95\% CI] & Recall (95\% CI) [missed of 24] & Specificity (95\% CI) & F1 & NPV & Accuracy \\", r"\midrule"]
    for r in MT_.itertuples():
        L.append(f"{r.model} & {ci(r.auroc, r.lo, r.hi)} & {cpr(r.sens, NP)} [{int(r.fn)}] & {cpr(r.spec, NN)} & {r.f1:.2f} & {npv_s(r.npv, r.fn, r.spec)} & {r.acc:.2f} \\\\")
        if r.model.startswith("Secondary"):
            L.append(r"\midrule \multicolumn{7}{l}{\emph{Hematology foundation model (added in this revision, selected on C-NMC development data)}} \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TAB / "tab_mll.tex").write_text("\n".join(L))
    # figure: ROC curves + local recalibration on the independent cohort
    pp_ = pd.read_csv(MLLR / "predictions.csv")
    curves = {"Primary (ensemble)": pd.read_csv(R / "mll_ensemble" / "ensemble_test.csv")}
    for m, lab in [("PDF", "DinoBloom + engineered"), ("P1", "Engineered, pooled LR")]:
        curves[lab] = pp_[(pp_.model == m) & (pp_["eval"] == "test")].groupby("patient_id").agg(y=("y", "first"), p=("p", "mean")).reset_index()
    curves["Fine-tuned CNN"] = pd.read_csv(R / "mll_finetune_macenko" / "patient_predictions_test.csv")
    rc_ = pd.read_csv(R / "mll_recal" / "mll_recal.csv") if (R / "mll_recal" / "mll_recal.csv").exists() else None
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.8))
    cols = {"Primary (ensemble)": RED, "DinoBloom + engineered": GREEN, "Engineered, pooled LR": BLUE, "Fine-tuned CNN": GREY}
    from sklearn.metrics import roc_auc_score as _auc
    for lab, d in curves.items():
        fpr, tpr, _ = roc_curve(d.y, d.p); ax[0].plot(fpr, tpr, color=cols[lab], lw=1.5 if lab.startswith("Primary") else 1.0, label=f"{lab} ({_auc(d.y, d.p):.2f})")
    ax[0].plot([0, 1], [0, 1], ":", color="#999", lw=0.8); ax[0].set_xlabel("1 - specificity"); ax[0].set_ylabel("Recall")
    ax[0].set_title("(a) Independent cohort: 24 APL, 105 AML", loc="left", fontsize=8); ax[0].legend(frameon=False, fontsize=6, loc="lower right")
    if rc_ is not None:
        for nm, c in [("Primary pipeline (ensemble)", RED), ("DinoBloom + engineered, pooled LR", GREEN), ("Macenko + fine-tuned CNN", GREY)]:
            q = rc_[(rc_.model == nm) & (rc_.m > 0)]
            ax[1].plot(q.m, q.sens, "o-", color=c, ms=3, lw=1.1); ax[1].plot(q.m, q.spec, "s--", color=c, ms=2.5, lw=0.8)
        q = rc_[(rc_.model == "Primary pipeline (ensemble)") & (rc_.m > 0)]
        ax[1].plot(q.m, q.m / (q.m + 1), color="black", lw=0.7, ls=":", label="$m/(m+1)$")
        ax[1].set_xlabel("Confirmed local APL cases $m$"); ax[1].set_ylabel("Recall (solid) / specificity (dashed)"); ax[1].set_ylim(0, 1.02)
        ax[1].set_title("(b) Threshold from local APL cases", loc="left", fontsize=8); ax[1].legend(frameon=False, fontsize=6, loc="lower right")
        facts["mll_recal"] = rc_.round(4).to_dict("records")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "fig_mll.pdf"); fig.savefig(OUT / "figures" / "fig_mll.png", dpi=200); plt.close(fig)
    print(MT_.round(3).to_string())

# DinoBloom across the three cohorts (development CV and external), compared with the best previous model of each block
if (R / "cnmc_cell_dino" / "summary.csv").exists():
    D = []
    def cellrow(folder, model, lab):
        s = pd.read_csv(R / folder / "summary.csv"); s = s[(s.model == model) & (s.split == "grouped")]
        cv, ex = s[s["eval"] == "cv"], s[s["eval"] == "test"]
        return dict(block="C-NMC, cells", model=lab, cv=cv.auroc.mean(), ext=ex.auroc.mean(), lo=ex.auroc_lo.mean(), hi=ex.auroc_hi.mean(), sens=ex.sens.mean(), spec=ex.spec.mean(), f1=ex.f1.mean())
    for f, m, lab in [("cnmc_cell", "M2", "EfficientNet-B0 (ImageNet), frozen"), ("cnmc_cell_dino", "MD", "DinoBloom-S, frozen"), ("cnmc_cell", "M1lr", "Engineered"),
                      ("cnmc_cell_lgbm", "M1", "Engineered + LightGBM"), ("cnmc_cell", "M4", "EfficientNet-B0 + engineered"), ("cnmc_cell_dino", "MDF", "DinoBloom-S + engineered")]:
        if (R / f / "summary.csv").exists():
            D.append(cellrow(f, m, lab))
    ftc = pd.read_csv(R / "finetune_cnmc" / "summary.csv").query("level == 'cell'").iloc[0]
    D.insert(2, dict(block="C-NMC, cells", model="EfficientNet-B0, fine-tuned", cv=ftc.cv_auroc, ext=ftc.ext_auroc, lo=ftc.ext_lo, hi=ftc.ext_hi, sens=ftc.ext_sens, spec=ftc.ext_spec, f1=ftc.ext_f1))
    def patrow(folder, model, lab, block):
        s = pd.read_csv(R / folder / "summary.csv"); cv = s[(s.model == model) & (s["eval"] == "cv")].auroc.mean(); ex = s[(s.model == model) & (s["eval"] == "test")].iloc[0]
        return dict(block=block, model=lab, cv=cv, ext=ex.auroc, lo=ex.auroc_lo, hi=ex.auroc_hi, sens=ex.sens, spec=ex.spec, f1=ex.f1)
    for f, m, lab in [("cnmc_patient", "P1", "Engineered, pooled LR"), ("cnmc_patient_dino", "PD", "DinoBloom-S, pooled LR"), ("cnmc_patient_dino", "PDM", "DinoBloom-S, attention MIL"), ("cnmc_patient_dino", "PDF", "DinoBloom-S + engineered, pooled LR")]:
        D.append(patrow(f, m, lab, "C-NMC, subjects"))
    for f, m, lab in [("v2_patient", "P4", "Engineered, attention MIL"), ("apl_patient_dino", "PD", "DinoBloom-S, pooled LR"), ("apl_patient_dino", "PDM", "DinoBloom-S, attention MIL"), ("apl_patient_dino", "PDF", "DinoBloom-S + engineered, pooled LR")]:
        if (R / f / "summary.csv").exists():
            D.append(patrow(f, m, lab, "APL-AML, later cohort"))
    D = pd.DataFrame(D); D.to_csv(TAB / "dinobloom.csv", index=False); facts["dinobloom"] = D.round(4).to_dict("records")
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Hematology foundation model (DinoBloom-S, frozen, CLS token) compared with ImageNet, fine-tuned and engineered features. Selection by development CV only (bold: highest development CV in each block). External: C-NMC fold 2 and the APL-AML Validation cohort; recall and specificity at the development threshold.}",
         r"\label{tab:dino}", r"\setlength{\tabcolsep}{2.4pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llccccc}", r"\toprule",
         r"Block & Features & Dev.\ CV & External [95\% CI] & Recall & Spec. & F1 \\", r"\midrule"]
    for b, g in D.groupby("block", sort=False):
        best = g.cv.max()
        for i, r in enumerate(g.itertuples()):
            nm = r"\textbf{" + r.model + "}" if r.cv == best else r.model
            L.append(f"{b if i == 0 else ''} & {nm} & {r.cv:.3f} & {ci(r.ext, r.lo, r.hi)} & {r.sens:.2f} & {r.spec:.2f} & {r.f1:.2f} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_dino.tex").write_text("\n".join(L)); print(D.round(3).to_string())

# improvement pilot on C-NMC (development CV only) and SHAP
pil = []
for d, lab in [("finetune_cnmc", "Fine-tuned EfficientNet-B0 (reference)"), ("pilot_cnmc_crop", "+ crop to the cell")]:
    f = R / d / "summary.csv"
    if f.exists():
        q = pd.read_csv(f).query("level == 'cell'").iloc[0]
        pil.append(dict(variant=lab, cv=q.cv_auroc, subj=q.get("patient_cv_auroc", np.nan)))
if pil:
    pl = pd.DataFrame(pil); pl.to_csv(TAB / "pilot_cnmc.csv", index=False); facts["pilot_cnmc"] = pl.round(4).to_dict("records")
if (R / "shap_cnmc" / "fig_shap.pdf").exists():
    shutil.copy(R / "shap_cnmc" / "fig_shap.pdf", OUT / "figures" / "fig_shap.pdf"); shutil.copy(R / "shap_cnmc" / "fig_shap.png", OUT / "figures" / "fig_shap.png")
    facts["shap_cnmc_groups"] = pd.read_csv(R / "shap_cnmc" / "shap_groups.csv").round(4).to_dict("records")
(TAB / "facts.json").write_text(json.dumps(facts, indent=1, default=float))

# ================================================================== REVISION 4: clinical utility (workload), split-sample recalibration, reciprocal transfer
UT = R / "utility"
if (UT / "workload.csv").exists():
    wk = pd.read_csv(UT / "workload.csv"); facts["workload"] = wk.round(4).to_dict("records")
    shutil.copy(UT / "fig_workload.pdf", OUT / "figures" / "fig_workload.pdf"); shutil.copy(UT / "fig_workload.png", OUT / "figures" / "fig_workload.png")
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Can the ranking reduce the review workload? Patients are referred for urgent review in order of their score. Recall: fraction of APL patients among the referred ones caught when the top 10\%, 20\% or 30\% of patients are referred (patient-bootstrap 95\% CI); PPV at 10\% APL prevalence for the same operating point; workload needed to catch 95\% and all APL patients (bootstrap 95\% CI for 95\%); AP: average precision (base rate: 0.19 in the independent cohort, 0.50 in the later cohort). No threshold is fitted on these cohorts.}",
         r"\label{tab:workload}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llccccccc}", r"\toprule",
         r" & & & \multicolumn{3}{c}{Recall when referring the top} & PPV at 10\% & \multicolumn{2}{c}{Workload to catch} \\", r"\cmidrule(lr){4-6}\cmidrule(lr){8-9}",
         r"Cohort & Model & AP [95\% CI] & 10\% & 20\% & 30\% & prevalence (top 20\%) & 95\% of APL & all APL \\", r"\midrule"]
    for coh, g in wk.groupby("cohort", sort=False):
        for i, r in enumerate(g.itertuples()):
            rr = lambda q: f"{getattr(r, f'recall_top{q}'):.2f} [{getattr(r, f'recall_top{q}_lo'):.2f}, {getattr(r, f'recall_top{q}_hi'):.2f}]"
            L.append(f"{coh + f' ({r.n_pos}/{r.n})' if i == 0 else ''} & {r.model} & {ci(r.ap, r.ap_lo, r.ap_hi, 2)} & {rr(10)} & {rr(20)} & {rr(30)} & {r.ppv10_top20:.2f} & "
                     f"{r.workload_r95:.0%} [{r.w95_lo:.0%}, {r.w95_hi:.0%}] & {r.workload_r100:.0%} \\\\".replace("%", r"\%"))
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table*}"]
    (TAB / "tab_workload.tex").write_text("\n".join(L))
if (UT / "local_recal_split.csv").exists():
    ls_ = pd.read_csv(UT / "local_recal_split.csv"); facts["local_recal_split"] = ls_.round(4).to_dict("records")
    facts["guarantee"] = pd.read_csv(UT / "guarantee.csv").to_dict("records")
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Local recalibration with a separate calibration sample (independent cohort; 2,000 random splits). Thresholds are set on the calibration sample only and evaluated on all remaining patients. POS: just below the lowest calibration APL score; BOTH: threshold maximizing calibration specificity subject to calibration recall $\geq$0.9. P05: 5th percentile of evaluation recall over splits; P(miss): probability of missing at least one APL patient in the evaluation sample.}",
         r"\label{tab:recalsplit}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llccccccc}", r"\toprule",
         r"Model & Calibration (APL/AML) & Rule & Recall & P05 & Expected & Spec. & Referred & P(miss) \\", r"\midrule"]
    for (m, cp_, cn_), g in ls_.groupby(["model", "cal_pos", "cal_neg"], sort=False):
        for r in g.itertuples():
            L.append(f"{m} & {cp_}/{cn_} & {r.rule} & {r.recall_mean:.2f} & {r.recall_p05:.2f} & {r.expected_pos_rule:.2f} & {r.spec_mean:.2f} & {r.referred:.2f} & {r.p_any_miss:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_recalsplit.tex").write_text("\n".join(L))
RC = R / "reciprocal_mll_to_apl"
W_MAN = R.parent / "manifests"
if (RC / "summary.csv").exists():
    rs = pd.read_csv(RC / "summary.csv"); NM = {"P1": "Engineered, pooled LR", "P4": "Engineered, attention MIL", "PD": "DinoBloom-S, pooled LR", "PDF": "DinoBloom-S + engineered, pooled LR"}
    fw = pd.read_csv(R / "mll_patient" / "summary.csv")
    from sklearn.metrics import roc_auc_score as _ra
    _pp = pd.read_csv(RC / "predictions.csv"); _coh = pd.read_csv(W_MAN / "apl_aml.csv", keep_default_na=False).groupby("patient_id").cohort.first()
    def per_period(m):
        t = _pp[(_pp.model == m) & (_pp["eval"] == "test")].groupby("patient_id").agg(y=("y", "first"), p=("p", "mean"))
        t["c"] = _coh.loc[t.index].values
        return {f"rev_{c.lower()}": _ra(g.y, g.p) for c, g in t.groupby("c")}
    rows = []
    for m, nm in NM.items():
        a_ = fw[(fw.model == m) & (fw["eval"] == "test")].iloc[0]; b_ = rs[(rs.model == m) & (rs["eval"] == "test")].iloc[0]
        rows.append(dict(model=nm, fwd=a_.auroc, fwd_lo=a_.auroc_lo, fwd_hi=a_.auroc_hi, fwd_cv=fw[(fw.model == m) & (fw["eval"] == "cv")].auroc.mean(),
                         rev=b_.auroc, rev_lo=b_.auroc_lo, rev_hi=b_.auroc_hi, rev_cv=rs[(rs.model == m) & (rs["eval"] == "cv")].auroc.mean(),
                         rev_sens=b_.sens, rev_fn=b_.fn, rev_spec=b_.spec, **per_period(m)))
    RT = pd.DataFrame(rows); RT.to_csv(TAB / "reciprocal.csv", index=False); facts["reciprocal"] = RT.round(4).to_dict("records")
    npos_rev = int(round(RT.rev_sens.iloc[0] * 34 + RT.rev_fn.iloc[0])) if False else 34
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Reciprocal cross-laboratory transfer at patient level. Forward: trained on APL-AML Discovery (82 patients), tested on the independent MLL cohort (24 APL, 105 AML). Reverse: trained on the MLL cohort (129 patients; settings and threshold from patient-grouped CV within MLL), tested on all 106 APL-AML patients (34 APL). Dev.: development CV AUROC. Disc./Val.: reverse AUROC within the earlier (82) and later (24) APL-AML patients. Recall and specificity at the source-cohort threshold. The APL-AML cohort had been examined extensively during development, so the reverse direction is exploratory.}",
         r"\label{tab:reciprocal}", r"\setlength{\tabcolsep}{2.4pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{APL-AML $\rightarrow$ MLL} & \multicolumn{4}{c}{MLL $\rightarrow$ APL-AML (exploratory)} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
         r"Model & Dev. & External [95\% CI] & Dev. & All 106 [95\% CI] & Disc./Val. & Recall / Spec. \\", r"\midrule"]
    for r in RT.itertuples():
        L.append(f"{r.model} & {r.fwd_cv:.3f} & {ci(r.fwd, r.fwd_lo, r.fwd_hi)} & {r.rev_cv:.3f} & {ci(r.rev, r.rev_lo, r.rev_hi)} & {r.rev_discovery:.2f}/{r.rev_validation:.2f} & {r.rev_sens:.2f} / {r.rev_spec:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_reciprocal.tex").write_text("\n".join(L)); print(RT.round(3).to_string())
(TAB / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
