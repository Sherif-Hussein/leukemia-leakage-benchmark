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

L = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{Effect of the evaluation protocol at cell level (mean of three seeds). AUROC under image-level CV, patient-grouped CV and on the external Validation cohort; threshold metrics on the external cohort at the operating point fixed on Discovery (target sensitivity 0.95). Inflation = image-level CV AUROC minus external AUROC.}",
     r"\label{tab:leakage}", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{llccccccccc}", r"\toprule",
     r" & & \multicolumn{3}{c}{AUROC} & \multicolumn{5}{c}{External cohort, fixed operating point} & \\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){6-10}",
     r"Task & Model & Image CV & Grouped CV & External [95\% CI] & Recall & Spec. & F1 & NPV & Acc. & Inflation \\", r"\midrule"]
for i, t in enumerate(TASKS.values()):
    sub = leak[leak.task == t]
    best = sub.ext.max()
    for j, r in enumerate(sub.itertuples()):
        e = ci(r.ext, r.ext_lo, r.ext_hi)
        e = r"\textbf{" + e + "}" if r.ext == best else e
        L.append(f"{t if j == 0 else ''} & {r.model} & {r.image_cv:.3f} & {r.grouped_cv:.3f} & {e} & {r.sens:.2f} & {r.spec:.2f} & {r.f1:.2f} & {r.npv:.2f} & {r.acc:.2f} & +{r.infl:.3f} \\\\")
    L.append(r"\midrule" if i == 0 else r"\bottomrule")
L += [r"\end{tabular}", r"\end{table*}"]
(TAB / "tab_leakage.tex").write_text("\n".join(L))

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True)
order = list(NAMES.values())
for k, (ax, (t, tname)) in enumerate(zip(axes, TASKS.items())):
    sub = leak[leak.task == tname].set_index("model").loc[order]
    y = np.arange(len(order))[::-1]
    for yi, r in zip(y, sub.itertuples()):
        ax.plot([r.ext, r.image_cv], [yi, yi], color="#C9CED6", lw=2, zorder=1)
        ax.errorbar(r.ext, yi, xerr=[[r.ext - r.ext_lo], [r.ext_hi - r.ext]], fmt="none", ecolor=BLUE, elinewidth=0.8, capsize=2)
        ax.text(r.image_cv + 0.006, yi + 0.18, f"+{r.infl:.2f}", fontsize=6.5, color=ORANGE)
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
    L.append(f"{r.rep} & {r.agg} & {r.cv:.3f}$\\pm${r.cv_sd:.3f} & {e} & {r.sens:.2f} ({r.fn}/12) & {r.spec:.2f} & {r.f1:.2f} & {r.npv:.2f} & {r.ece:.3f} & {r.y_sens:.2f} ({r.y_fn}/12) & {r.y_spec:.2f} & {r.y_acc:.2f} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(TAB / "tab_patient.tex").write_text("\n".join(L))

# ================================================================== same-cohort comparison
bl = pd.read_csv(R / "v2_baselines" / "baselines.csv")
facts["baselines"] = bl.round(4).to_dict("records")
best_m = pt.sort_values("ext", ascending=False).iloc[0]
C = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{Comparison with existing methods on the same APL-AML cohort. Upper block: results as reported by the original studies, whose protocols differ from ours. Lower blocks: methods re-run on the identical split (trained on Discovery, tested once on the 24 Validation patients) with the same operating-point rule (sensitivity $\geq$ 0.95 on Discovery).}",
     r"\label{tab:compare}", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{p{4.6cm}p{3.6cm}ccccccc}", r"\toprule",
     r"Method & Evaluation & AUROC [95\% CI] & Recall (missed) & Spec. & F1 & NPV & Acc. \\", r"\midrule",
     r"\multicolumn{8}{l}{\emph{As reported in the literature}} \\",
     r"Single-cell CNN, Sidhom \emph{et al.} \cite{sidhom2021apl} & Temporal validation, cell level & 0.743 & -- & -- & -- & -- & -- \\",
     r"Annotation-free MIL, Manescu \emph{et al.} \cite{manescu2023apl} & Random 3-fold CV, patient level & 0.96 $\pm$ 0.02 & -- & -- & -- & -- & -- \\",
     r"\midrule", r"\multicolumn{8}{l}{\emph{Re-run on the identical split (Validation cohort, patient level)}} \\"]
labels = {"Promyelocyte proportion": r"Promyelocyte proportion (biomarker in \cite{sidhom2021apl})",
          "EfficientNet-B0 features + RBF-SVM, cell mean": r"CNN features + SVM (design of \cite{kasim2025multiclass})",
          "Engineered + LR, cell mean": r"Single-cell classifier, engineered (design of \cite{sidhom2021apl})",
          "EfficientNet-B0 + LR, cell mean": r"Single-cell classifier, EfficientNet-B0 (design of \cite{sidhom2021apl})",
          "Fusion + LR, cell mean": r"Single-cell classifier, fusion (design of \cite{sidhom2021apl})"}
for r in bl.itertuples():
    lab = labels.get(r.baseline, r.baseline.replace("Differential count (17 cell types) + LR", r"Differential count + LR (comparator in \cite{dasdelen2026caitomorph})"))
    C.append(f"{lab} & Same protocol & {ci(r.auroc, r.auroc_lo, r.auroc_hi)} & {r.sens:.2f} ({int(r.fn)}/12) & {r.spec:.2f} & {r.f1:.2f} & {r.npv:.2f} & {r.acc:.2f} \\\\")
C.append(r"\midrule")
C.append(r"\multicolumn{8}{l}{\emph{This work}} \\")
for r in pt.itertuples():
    if r.agg != "Attention MIL":
        continue
    e = ci(r.ext, r.lo, r.hi)
    e = r"\textbf{" + e + "}" if r.ext == best else e
    C.append(f"{r.rep} + attention MIL & Same protocol & {e} & {r.sens:.2f} ({r.fn}/12) & {r.spec:.2f} & {r.f1:.2f} & {r.npv:.2f} & {r.acc:.2f} \\\\")
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
fe = pd.read_csv(R / "feature_engineering" / "feature_engineering.csv")
fp_ = pd.read_csv(R / "feature_engineering" / "predictions.csv")
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
fe.to_csv(TAB / "feature_engineering.csv", index=False)
facts["feature_engineering"] = fe.round(4).to_dict("records")
sel = fe.loc[fe.cv_auroc.idxmax()]
facts["fe_selected"] = sel.round(4).to_dict()
L = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{Feature engineering for patient-level APL detection. All steps are fitted inside the Discovery training folds; the Validation cohort is scored once. $\Delta$: external AUROC change relative to the unreduced representation (paired patient bootstrap). The configuration with the highest Discovery CV AUROC (pre-declared selection rule) is marked $^{\star}$. Oracle column: specificity at 100\% recall if the threshold were re-set on the Validation cohort (reference only, not deployable).}",
     r"\label{tab:fe}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llccccccc}", r"\toprule",
     r"Representation & Engineering step & CV AUROC & External AUROC [95\% CI] & $\Delta$ [95\% CI] & Recall (missed) & Spec. & F1 & Oracle spec. \\", r"\midrule"]
for i, r in enumerate(fe.itertuples()):
    rep, step = NICE[r.config]
    star = r"$^{\star}$" if r.config == sel.config else ""
    d = "--" if np.isnan(r.delta) else f"{r.delta:+.3f} [{r.dlo:+.3f}, {r.dhi:+.3f}]" + ("*" if r.dp < 0.05 else "")
    L.append(f"{rep}{star} & {step} & {r.cv_auroc:.3f} & {ci(r.ext_auroc, r.ext_lo, r.ext_hi)} & {d} & {r.sens95_sens:.2f} ({int(r.sens95_fn)}/12) & {r.sens95_spec:.2f} & {r.sens95_f1:.2f} & {r.oracle_spec_at_recall1:.2f} \\\\")
    if r.config in ("eng_stable", "eff_pca64"):
        L.append(r"\midrule")
L += [r"\bottomrule", r"\multicolumn{9}{l}{\footnotesize Recall, specificity and F1 at the operating point fixed on Discovery (target sensitivity 0.95). * $p<0.05$.}", r"\end{tabular}", r"\end{table*}"]
(TAB / "tab_fe.tex").write_text("\n".join(L))
(TAB / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
print(fe[["config", "cv_auroc", "ext_auroc", "delta", "dlo", "dhi", "dp", "sens95_sens", "sens95_spec", "oracle_spec_at_recall1"]].round(3).to_string())
