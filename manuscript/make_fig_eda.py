import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
import os
W = os.environ.get("LEUKBENCH_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # repo root (manifests/, features/, results/)
E = f"{W}/results/eda"
shift = pd.read_csv(f"{E}/shift_smd.csv"); cid = pd.read_csv(f"{E}/cohort_id.csv"); uni = pd.read_csv(f"{E}/univariate_auc.csv")
man = pd.read_csv(f"{W}/manifests/apl_aml.csv", keep_default_na=False); H = pd.read_parquet(f"{W}/features/apl_aml_hand.parquet")
fig, ax = plt.subplots(2, 2, figsize=(7.16, 4.6))
ax = ax.ravel()
gs = shift.groupby("group").abs_smd.median().sort_values()
ax[0].barh(range(len(gs)), gs.values, color="#8C8C8C")
ax[0].set_yticks(range(len(gs)), gs.index); ax[0].axvline(0.2, color="#D9822B", lw=0.8, ls="--")
ax[0].text(0.205, -0.6, "small effect", color="#D9822B", fontsize=6.5)
ax[0].set_xlabel("Median |standardized mean difference|"); ax[0].set_title("(a) Discovery-to-Validation shift by group", loc="left", fontsize=8)
lab = [s.replace("Engineered: ", "Engineered, ").replace("Engineered (all)", "Engineered, all") for s in cid.representation]
ax[1].barh(range(len(cid)), cid.cohort_auroc, color=["#2F6DB5" if "stain" in s else "#8C8C8C" for s in cid.representation])
for i, v in enumerate(cid.cohort_auroc): ax[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=6.8)
ax[1].set_yticks(range(len(cid)), lab); ax[1].axvline(0.5, color="#555", lw=0.8); ax[1].set_xlim(0.45, 0.85)
ax[1].text(0.505, len(cid) - 0.45, "0.5 = no shift", fontsize=6.3, color="#555")
ax[1].set_xlabel("AUROC, Validation vs. Discovery (AML cells)"); ax[1].set_title("(b) How recognizable is the cohort?", loc="left", fontsize=8)
P = H.assign(pid=man.patient_id, y=(man.diagnosis == "APL").astype(int), coh=man.cohort).groupby("pid")
pm = P.mean(numeric_only=True); coh = P.coh.first()
Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(pm[H.columns]))
pc = PCA(2, random_state=0).fit(Z); Zp = pc.transform(Z)
for c, mk in [("Discovery", "o"), ("Validation", "^")]:
    for yv, col, nm in [(0, "#8C8C8C", "AML"), (1, "#2F6DB5", "APL")]:
        m = (coh.values == c) & (pm.y.values == yv)
        ax[2].scatter(Zp[m, 0], Zp[m, 1], s=14, marker=mk, color=col, alpha=0.85, edgecolor="white", lw=0.3, label=f"{c} {nm}")
ax[2].set_xlabel(f"PC1 ({pc.explained_variance_ratio_[0]:.0%})"); ax[2].set_ylabel(f"PC2 ({pc.explained_variance_ratio_[1]:.0%})")
ax[2].legend(fontsize=6.3, frameon=False, loc="lower left"); ax[2].set_title("(c) Patients, PCA of mean engineered features", loc="left", fontsize=8)
st = ((uni.auc_discovery > 0.65) & (uni.auc_validation > 0.65)) | ((uni.auc_discovery < 0.35) & (uni.auc_validation < 0.35))
ax[3].scatter(uni.auc_discovery[~st], uni.auc_validation[~st], s=9, color="#B8B8B8", label="Unstable")
ax[3].scatter(uni.auc_discovery[st], uni.auc_validation[st], s=12, color="#2F6DB5", label=f"Stable in both cohorts ({st.sum()})")
ax[3].plot([0.1, 0.9], [0.1, 0.9], color="#ccc", lw=0.6); ax[3].axhline(0.5, color="#ddd", lw=0.6); ax[3].axvline(0.5, color="#ddd", lw=0.6)
r = np.corrcoef(uni.auc_discovery, uni.auc_validation)[0, 1]
ax[3].text(0.16, 0.84, f"r = {r:.2f}", fontsize=6.8)
ax[3].set_xlim(0.12, 0.9); ax[3].set_ylim(0.12, 0.9); ax[3].set_xlabel("Patient-level AUROC, Discovery"); ax[3].set_ylabel("Patient-level AUROC, Validation")
ax[3].legend(fontsize=6.3, frameon=False, loc="lower right"); ax[3].set_title("(d) Per-feature APL signal by cohort", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig_eda.pdf"); fig.savefig("figures/fig_eda.png", dpi=220); print("ok")
