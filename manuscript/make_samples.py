"""Figure of sample images from both datasets (fig_samples.pdf). Random but fixed (seed 3) choices."""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
W = sys.argv[1]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
a = pd.read_csv(f"{W}/manifests/apl_aml.csv", keep_default_na=False)
c = pd.read_csv(f"{W}/manifests/cnmc.csv", keep_default_na=False)
rng = np.random.default_rng(3)
def pick(df, n=1):
    for _ in range(50):
        r = df.iloc[rng.integers(len(df))]
        try:
            im = Image.open(r.path).convert("RGB"); return im, r
        except Exception:
            continue
rows = [
 ("APL-AML, APL patients", [("Promyelocyte", a[(a.diagnosis == "APL") & (a.cell_type == "Promyelocyte")]),
                            ("Blast", a[(a.diagnosis == "APL") & (a.cell_type == "Blast, no lineage spec")]),
                            ("Blast", a[(a.diagnosis == "APL") & (a.cell_type == "Blast, no lineage spec")]),
                            ("Neutrophil", a[(a.diagnosis == "APL") & (a.cell_type == "Segmented neutrophils")]),
                            ("Lymphocyte", a[(a.diagnosis == "APL") & (a.cell_type == "Lymphocyte")]),
                            ("Unlabeled", a[(a.diagnosis == "APL") & (a.cell_type == "unlabelled")])]),
 ("APL-AML, other AML", [("Blast", a[(a.diagnosis == "AML") & (a.cell_type == "Blast, no lineage spec")]),
                         ("Blast", a[(a.diagnosis == "AML") & (a.cell_type == "Blast, no lineage spec")]),
                         ("Monocyte", a[(a.diagnosis == "AML") & (a.cell_type == "Monocyte")]),
                         ("Neutrophil", a[(a.diagnosis == "AML") & (a.cell_type == "Segmented neutrophils")]),
                         ("Smudge cell", a[(a.diagnosis == "AML") & (a.cell_type == "Smudge cells")]),
                         ("Unlabeled", a[(a.diagnosis == "AML") & (a.cell_type == "unlabelled")])]),
 ("C-NMC, ALL subjects", [("Lymphoblast", c[c.diagnosis == "ALL"])] * 6),
 ("C-NMC, normal subjects", [("Normal B cell", c[c.diagnosis == "HEM"])] * 6)]
fig, ax = plt.subplots(4, 6, figsize=(7.16, 5.0))
for i, (rl, items) in enumerate(rows):
    for j, (lab, df) in enumerate(items):
        im, r = pick(df)
        ax[i, j].imshow(im); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        coh = r.cohort if r.dataset == "apl_aml" else ("dev." if r.split_role == "Development" else "ext.")
        ax[i, j].set_title(f"{lab} ({coh})" if r.dataset == "apl_aml" else f"{lab} ({coh})", fontsize=6.3)
        for sp in ax[i, j].spines.values(): sp.set_linewidth(0.4)
    ax[i, 0].set_ylabel(rl, fontsize=7)
fig.tight_layout(h_pad=0.6, w_pad=0.3)
fig.savefig("figures/fig_samples.pdf", dpi=200); fig.savefig("figures/fig_samples.png", dpi=150)
