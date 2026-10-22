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
ap.add_argument("--chunk", type=int, default=0, help="process the manifest in chunks of this many rows, each in a fresh process (avoids memory growth)")
a = ap.parse_args()

if a.threads:
    torch.set_num_threads(a.threads)
paths = pd.read_csv(a.manifest)["path"].tolist()
t0 = time.time()
if a.chunk:
    import subprocess, sys as _s, tempfile, os
    parts = []
    for st in range(0, len(paths), a.chunk):
        tmpm = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False).name; tmpo = tmpm.replace(".csv", ".npy")
        pd.DataFrame({"path": paths[st:st + a.chunk]}).to_csv(tmpm, index=False)
        cmd = [_s.executable, "-I", __file__, "--manifest", tmpm, "--encoder", a.encoder, "--out", tmpo, "--device", a.device,
               "--batch-size", str(a.batch_size), "--workers", str(a.workers), "--threads", str(a.threads)] + (["--weights", a.weights] if a.weights else [])
        subprocess.run(cmd, check=True); parts.append(np.load(tmpo)); os.remove(tmpm); os.remove(tmpo)
        print(f"  chunk done: {min(st + a.chunk, len(paths)):,}/{len(paths):,}", flush=True)
    z = np.concatenate(parts)
else:
    z = embed(build_encoder(a.encoder, a.weights), paths, a.device, a.batch_size, a.workers)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.save(a.out, z)
print(f"wrote {a.out}: {z.shape} in {(time.time() - t0) / 60:.1f} min on {a.device}")
