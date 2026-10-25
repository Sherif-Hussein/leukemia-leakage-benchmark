import pandas as pd, numpy as np, sys
from pathlib import Path
W = Path(sys.argv[1]); O, N = W / "results", W / "repro"; rows = []
def add(name, a, b):
    rows.append(dict(result=name, reported=a, rerun=b, abs_diff=abs(a - b)))
for f in ["mll_patient", "reciprocal_mll_to_apl"]:
    if (N / f / "summary.csv").exists():
        a = pd.read_csv(O / f / "summary.csv").query("eval == 'test'").set_index("model"); b = pd.read_csv(N / f / "summary.csv").query("eval == 'test'").set_index("model")
        for m in a.index:
            for k in ["auroc", "sens", "spec"]:
                add(f"{f} {m} {k}", a.loc[m, k], b.loc[m, k])
if (N / "mll_fe_macenko" / "feature_engineering.csv").exists():
    a = pd.read_csv(O / "mll_fe_macenko/feature_engineering.csv").set_index("config"); b = pd.read_csv(N / "mll_fe_macenko/feature_engineering.csv").set_index("config")
    for c in a.index:
        for k in ["ext_auroc", "sens95_sens", "sens95_spec"]:
            add(f"secondary {c} {k}", a.loc[c, k], b.loc[c, k])
if (N / "mll_finetune_macenko" / "summary.csv").exists():
    a = pd.read_csv(O / "mll_finetune_macenko/summary.csv").set_index("level"); b = pd.read_csv(N / "mll_finetune_macenko/summary.csv").set_index("level")
    for lv in a.index:
        for k in ["ext_auroc", "ext_sens", "ext_spec"]:
            add(f"fine-tuned CNN ({lv}) {k}", a.loc[lv, k], b.loc[lv, k])
if (N / "mll_ensemble" / "ensemble.csv").exists():
    a = pd.read_csv(O / "mll_ensemble/ensemble.csv").set_index("model"); b = pd.read_csv(N / "mll_ensemble/ensemble.csv").set_index("model")
    for k in ["ext_auroc", "sens", "spec"]:
        add(f"primary ensemble {k}", a.loc["Ensemble", k], b.loc["Ensemble", k])
if (N / "utility" / "workload.csv").exists():
    a = pd.read_csv(O / "utility/workload.csv"); b = pd.read_csv(N / "utility/workload.csv")
    for i in range(len(a)):
        for k in ["recall_top20", "workload_r95"]:
            add(f"workload {a.cohort[i]} {a.model[i]} {k}", a.loc[i, k], b.loc[i, k])
D = pd.DataFrame(rows); D.to_csv(N / "comparison.csv", index=False)
print(D.round(4).to_string()); print("max abs difference:", D.abs_diff.max().round(6), "| identical:", int((D.abs_diff < 1e-9).sum()), "of", len(D))
