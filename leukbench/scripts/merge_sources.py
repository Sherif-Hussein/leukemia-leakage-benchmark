"""Concatenate several datasets (manifests + their feature files, same row order) into one,
for cross-lab experiments (E3).

  python scripts/merge_sources.py --manifests manifests/lmu.csv manifests/bmc.csv manifests/cnmc.csv manifests/apl_aml.csv \
     --block hand features/lmu_hand.parquet features/bmc_hand.parquet features/cnmc_hand.parquet features/apl_aml_hand.parquet \
     --block effnet features/lmu_effnet.npy features/bmc_effnet.npy features/cnmc_effnet.npy features/apl_aml_effnet.npy \
     --out-dir merged/crosslab
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench.models import load_feature_block

ap = argparse.ArgumentParser()
ap.add_argument("--manifests", nargs="+", required=True)
ap.add_argument("--block", nargs="+", action="append", required=True, help="NAME file1 file2 ... (one per manifest)")
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()

out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
mans = [pd.read_csv(m, keep_default_na=False) for m in a.manifests]
pd.concat(mans, ignore_index=True).to_csv(out / "manifest.csv", index=False)
for blk in a.block:
    name, files = blk[0], blk[1:]
    if len(files) != len(mans):
        raise SystemExit(f"block {name}: {len(files)} files for {len(mans)} manifests")
    parts = []
    for f, m in zip(files, mans):
        X = load_feature_block(f)
        assert len(X) == len(m), f"{f} rows {len(X)} != manifest rows {len(m)}"
        parts.append(X)
    np.save(out / f"{name}.npy", np.vstack(parts).astype("float32"))
    print(f"{name}: {sum(len(p) for p in parts):,} rows")
print(f"wrote {out}")
