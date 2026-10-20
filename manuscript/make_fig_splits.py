import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
cols = {"P1": "#4C78A8", "P2": "#F58518", "P3": "#54A24B"}
fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.75))
def box(ax, x, y, w, h, title, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc="#F7F8FA", ec=ec, lw=1))
    ax.text(x + w / 2, y + h + 0.05, title, ha="center", va="bottom", fontsize=7.5)
def cells(ax, x0, y0, pats, n=6):
    k = 0
    for p in pats:
        for i in range(n):
            ax.add_patch(Circle((x0 + 0.07 * (k % 6), y0 - 0.09 * (k // 6)), 0.028, fc=cols[p], ec="none"))
            k += 1
for ax in axes:
    ax.set_xlim(0, 2); ax.set_ylim(-0.05, 0.85); ax.axis("off")
ax = axes[0]
ax.set_title("(a) Image-level split (common practice)", fontsize=9, color="#B03A2E", loc="left")
box(ax, 0.05, 0.12, 0.85, 0.42, "Training cells", "#B03A2E"); box(ax, 1.1, 0.12, 0.85, 0.42, "Test cells", "#B03A2E")
cells(ax, 0.17, 0.42, ["P1", "P2", "P3"], 4); cells(ax, 1.22, 0.42, ["P1", "P2", "P3"], 2)
ax.text(1.0, -0.02, "same patients on both sides: the model can recognise the patient", ha="center", fontsize=7, color="#B03A2E")
ax = axes[1]
ax.set_title("(b) Patient-grouped + external cohort (this work)", fontsize=9, color="#2E7D32", loc="left")
box(ax, 0.05, 0.12, 0.6, 0.42, "Training", "#2E7D32"); box(ax, 0.72, 0.12, 0.55, 0.42, "CV test", "#2E7D32")
box(ax, 1.37, 0.12, 0.58, 0.42, "External (once)", "#6A51A3")
cells(ax, 0.12, 0.42, ["P1", "P2"], 6); cells(ax, 0.8, 0.42, ["P3"], 6)
for i in range(12):
    ax.add_patch(Circle((1.45 + 0.07 * (i % 6), 0.42 - 0.09 * (i // 6)), 0.028, fc="#9E9AC8", ec="none"))
ax.text(1.66, 0.17, "24 unseen patients", ha="center", fontsize=6.5)
ax.text(1.0, -0.02, "no patient appears in more than one set", ha="center", fontsize=7, color="#2E7D32")
for p, c in cols.items():
    pass
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"cells of patient {p}") for p, c in cols.items()]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("figures/fig_splits.pdf"); fig.savefig("figures/fig_splits.png", dpi=220)
