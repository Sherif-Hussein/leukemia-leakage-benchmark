"""Compute engineered features for every row of a manifest.

  python scripts/extract_handcrafted.py --manifest manifests/apl_aml.csv --out features/apl_aml_hand.parquet --workers 4

Output: one row per manifest row (same order), columns = feature names. Rows that fail are NaN
and are imputed later inside each training fold (never using test statistics).
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import sys
from functools import partial
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench.handcrafted import features_from_path

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--black-bg-to-white", action="store_true", help="pre-segmented cells on black background (C-NMC)")
a = ap.parse_args()

paths = pd.read_csv(a.manifest)["path"].tolist()
t0 = time.time()
rows = []
with Pool(a.workers) as pool:
    for i, r in enumerate(pool.imap(partial(features_from_path, black_bg=a.black_bg_to_white), paths, chunksize=32)):
        rows.append(r)
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"  {i + 1:,}/{len(paths):,}  {el / 60:.1f} min elapsed, ~{el / (i + 1) * (len(paths) - i - 1) / 60:.1f} min left", flush=True)
df = pd.DataFrame(rows).astype("float32")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(a.out) if a.out.endswith(".parquet") else df.to_csv(a.out, index=False)
print(f"wrote {a.out}: {df.shape[0]:,} rows x {df.shape[1]} features in {(time.time() - t0) / 60:.1f} min; "
      f"{int(df.isna().all(axis=1).sum())} failed rows")
