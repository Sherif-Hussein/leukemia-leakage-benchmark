"""Cache frozen-encoder embeddings for every row of a manifest.

  python scripts/extract_embeddings.py --manifest manifests/apl_aml.csv --encoder effnet_b0_imagenet \
         --out features/apl_aml_effnet.npy --device cuda
  python scripts/extract_embeddings.py --manifest manifests/apl_aml.csv --encoder dinobloom_b \
         --weights weights/DinoBloom-B.pth --out features/apl_aml_dinobloom_b.npy --device cuda
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench.encoders import build_encoder, embed

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--encoder", required=True)
ap.add_argument("--weights")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
a = ap.parse_args()

if a.threads:
    torch.set_num_threads(a.threads)
paths = pd.read_csv(a.manifest)["path"].tolist()
t0 = time.time()
z = embed(build_encoder(a.encoder, a.weights), paths, a.device, a.batch_size, a.workers)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.save(a.out, z)
print(f"wrote {a.out}: {z.shape} in {(time.time() - t0) / 60:.1f} min on {a.device}")
