"""Collect summary.csv files into paper-ready markdown tables.

  python scripts/make_tables.py results/e1_apl_cell_diagnosis results/e4_apl_patient --out results/tables.md

Across seeds, the point estimate is the mean over seeds and the 95% CI is the mean of the
per-seed patient-bootstrap bounds (seed spread is shown separately as +/- SD).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["auroc", "sens", "spec", "bal_acc", "ece"]

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, sort=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["seeds"] = len(g)
        for m in METRICS:
            if m not in g:
                continue
            v, lo, hi, sd = g[m].mean(), g[m + "_lo"].mean(), g[m + "_hi"].mean(), g[m].std(ddof=0)
            r[m] = f"{v:.3f} [{lo:.3f}, {hi:.3f}]" + (f" ±{sd:.3f}" if len(g) > 1 else "")
            r[m + "_val"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def md(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


parts = []
for d in a.dirs:
    s = pd.read_csv(Path(d) / "summary.csv")
    parts.append(f"## {Path(d).name}\n")
    keys = [c for c in ["model", "blocks", "clf", "kind", "split", "eval"] if c in s]
    t = agg(s, keys)
    parts.append(md(t, keys + ["seeds"] + METRICS) + "\n")
    if "split" in s and set(s["split"]) >= {"image", "grouped"}:
        rows = []
        for (m, e), g in t.groupby(["model", "eval"], sort=False):
            gi = g[g.split == "image"]["auroc_val"]
            gg = g[g.split == "grouped"]["auroc_val"]
            if len(gi) and len(gg):
                rows.append(dict(model=m, eval=e, image_level=f"{gi.iloc[0]:.3f}", patient_grouped=f"{gg.iloc[0]:.3f}",
                                 inflation=f"{gi.iloc[0] - gg.iloc[0]:+.3f}"))
        if rows:
            parts.append("**Leakage inflation (AUROC, image-level minus patient-grouped split)**\n")
            parts.append(md(pd.DataFrame(rows), ["model", "eval", "image_level", "patient_grouped", "inflation"]) + "\n")

Path(a.out).write_text("\n".join(parts))
print("\n".join(parts))
