import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "leukbench"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
W = os.environ.get("LEUKBENCH_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # repo root (manifests/, features/)
m = pd.read_csv(f"{W}/manifests/apl_aml.csv", keep_default_na=False)
GREY, BLUE, ORANGE = "#8C8C8C", "#2F6DB5", "#D9822B"

# ---- Fig: dataset composition (3 panels)
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw=dict(width_ratios=[1.0, 2.0, 1.1]))
p = m.groupby("patient_id").agg(dx=("diagnosis", "first"), coh=("cohort", "first"), n=("path", "size"))
t = p.groupby(["coh", "dx"]).size().unstack()
x = np.arange(2)
ax[0].bar(x - 0.18, t["AML"], 0.36, color=GREY, label="AML (non-APL)")
ax[0].bar(x + 0.18, t["APL"], 0.36, color=BLUE, label="APL")
for i, c in enumerate(t.index):
    ax[0].text(i - 0.18, t.loc[c, "AML"] + 1, t.loc[c, "AML"], ha="center", fontsize=7)
    ax[0].text(i + 0.18, t.loc[c, "APL"] + 1, t.loc[c, "APL"], ha="center", fontsize=7)
ax[0].set_xticks(x, ["Discovery", "Validation"], fontsize=6.5); ax[0].set_ylim(0, 75); ax[0].set_ylabel("Patients"); ax[0].set_title("(a) Patients per cohort", loc="left", fontsize=8)
ax[0].legend(frameon=False, fontsize=6.5, loc="upper center", ncol=1, bbox_to_anchor=(0.62, 1.02))
s = m[m.cell_type != "unlabelled"]
ct = s.groupby(["cell_type", "diagnosis"]).size().unstack(fill_value=0)
ct = ct[ct.sum(1) >= 50]
ct = ct.loc[ct.sum(1).sort_values().index]
y = np.arange(len(ct))
ax[1].barh(y - 0.2, ct["AML"], 0.4, color=GREY); ax[1].barh(y + 0.2, ct["APL"], 0.4, color=BLUE)
ax[1].set_yticks(y, [c.replace(", no lineage spec", "") for c in ct.index], fontsize=6.5)
ax[1].set_xlabel("Technologist-signed cells"); ax[1].set_title("(b) Signed cell types (>= 50 cells)", loc="left", fontsize=8)
mi = list(ct.index).index("Monocyte")
ax[1].annotate("1,228 vs 91", (ct.loc["Monocyte", "AML"], mi - 0.2), xytext=(8, -2), textcoords="offset points", fontsize=6.5, color=ORANGE)
ax[2].hist([p[p.dx == "AML"].n, p[p.dx == "APL"].n], bins=np.arange(0, 1101, 100), color=[GREY, BLUE], stacked=True)
ax[2].set_xlabel("Cells per patient"); ax[2].set_ylabel("Patients"); ax[2].set_title("(c) Bag size", loc="left", fontsize=8)
ax[2].axvline(p.n.median(), color="#444", lw=0.8, ls="--"); ax[2].text(p.n.median() + 20, ax[2].get_ylim()[1] * 0.9, f"median {int(p.n.median())}", fontsize=6.5)
fig.tight_layout(); fig.savefig("figures/fig_dataset.pdf"); fig.savefig("figures/fig_dataset.png", dpi=220)

# ---- Fig: segmentation examples
from leukbench.handcrafted import segment, SIZE
from skimage.segmentation import find_boundaries
from scipy import ndimage as ndi
from skimage.color import rgb2lab
classes = ["Blast, no lineage spec", "Promyelocyte", "Lymphocyte", "Monocyte", "Segmented neutrophils", "Erythroblast"]
names = ["Blast", "Promyelocyte", "Lymphocyte", "Monocyte", "Seg. neutrophil", "Erythroblast"]
fig, ax = plt.subplots(2, 6, figsize=(7.16, 2.75))
for j, c in enumerate(classes):
    paths = s[s.cell_type == c].path.tolist()
    for i, f in enumerate([paths[len(paths) // 4], paths[3 * len(paths) // 4]]):
        rgb = np.asarray(Image.open(f).convert("RGB").resize((SIZE, SIZE)))
        nuc, cell, _ = segment(rgb)
        L0 = rgb2lab(rgb)[..., 0]; d = ndi.distance_transform_edt(~nuc)
        ring = (d > 2) & (d <= 9) & (L0 < np.percentile(L0, 90) - 3)
        o = rgb.copy().astype(float)
        o[ring] = 0.55 * o[ring] + 0.45 * np.array([60, 120, 255])
        o[find_boundaries(cell)] = [20, 160, 60]; o[find_boundaries(nuc)] = [230, 30, 30]
        ax[i, j].imshow(o.astype(np.uint8)); ax[i, j].axis("off")
        if i == 0: ax[i, j].set_title(names[j], fontsize=7.5)
fig.text(0.5, 0.01, "red: nucleus   green: basophilic region   blue tint: perinuclear ring (cytoplasm proxy)", ha="center", fontsize=7)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig("figures/fig_segmentation.pdf", dpi=300); fig.savefig("figures/fig_segmentation.png", dpi=220)
print("ok")
