"""End-to-end smoke test on synthetic images laid out like C-NMC, AML-Hehr and LMU.

  python -I tests/smoke_test.py [work_dir]   # ~2-4 min on CPU

Checks that every builder, feature extractor and runner executes and that grouped splits
never put a patient in two folds. The numbers it prints are meaningless (synthetic data).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable, "-I"]


def cell(blast: bool, rng) -> Image.Image:
    im = Image.new("RGB", (120, 120), (240, 220, 210))
    d = ImageDraw.Draw(im)
    for _ in range(4):                                     # red cells
        x, y = rng.integers(0, 110, 2)
        d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=(220, 150, 160))
    r = 30 if blast else 22
    d.ellipse([60 - r - 8, 60 - r - 8, 60 + r + 8, 60 + r + 8], fill=(190, 180, 220))   # cytoplasm
    d.ellipse([60 - r, 60 - r, 60 + r, 60 + r], fill=(90 + int(rng.integers(0, 30)), 60, 140))  # nucleus
    return im


def run(args):
    print("$", " ".join(str(a) for a in args[2:]))
    subprocess.run(args, check=True)


def main():
    rng = np.random.default_rng(0)
    base = sys.argv[1] if len(sys.argv) > 1 else None          # optional working folder
    tmp = Path(tempfile.mkdtemp(prefix="leukbench_smoke_", dir=base))
    # --- C-NMC-like
    for fold in ["fold_0", "fold_1"]:
        for cls in ["all", "hem"]:
            p = tmp / "cnmc" / "C-NMC_training_data" / fold / cls
            p.mkdir(parents=True)
            for s in range(6):
                subj = f"{s}" if cls == "all" else f"H{s}"
                for i in range(6):
                    cell(cls == "all", rng).save(p / f"UID_{subj}{fold[-1]}_{i}_1_{cls}.bmp")
    pre = tmp / "cnmc" / "C-NMC_test_prelim_phase_data"
    pre.mkdir(parents=True)
    rows = []
    for i in range(1, 21):
        lab = int(i % 2)
        cell(lab == 1, rng).save(pre / f"{i}.bmp")
        rows.append(dict(Patient_ID=f"UID_{90 + i % 4}_{i}_1_{'all' if lab else 'hem'}.bmp", new_names=f"{i}.bmp", labels=lab))
    pd.DataFrame(rows).to_csv(tmp / "cnmc" / "labels.csv", index=False)
    # --- AML-Hehr-like
    meta = []
    for k in range(16):
        lab = ["control", "PML_RARA", "NPM1", "RUNX1_RUNX1T1"][k % 4]
        pid = f"P{k:03d}"
        p = tmp / "hehr" / "data" / lab / pid
        p.mkdir(parents=True)
        for i in range(12):
            cell(lab != "control" and rng.random() < 0.6, rng).save(p / f"image_{i}.tif")
        meta.append(dict(patient_id=pid, bag_label=lab))
    pd.DataFrame(meta).to_csv(tmp / "hehr" / "metadata.csv", index=False)
    # --- LMU-like (folder per class, no patient ids)
    for cls in ["MYO", "LYT", "NGS"]:
        p = tmp / "lmu" / cls
        p.mkdir(parents=True)
        for i in range(25):
            cell(cls == "MYO", rng).save(p / f"{cls}_{i:04d}.tiff")

    S = ROOT / "scripts"
    m, f, r = tmp / "manifests", tmp / "features", tmp / "results"
    run(PY + [S / "build_manifest.py", "cnmc", "--root", tmp / "cnmc", "--labels-csv", tmp / "cnmc" / "labels.csv", "--out", m / "cnmc.csv"])
    run(PY + [S / "build_manifest.py", "hehr", "--root", tmp / "hehr", "--out", m / "hehr.csv"])
    run(PY + [S / "build_manifest.py", "folder_per_class", "--root", tmp / "lmu", "--dataset", "lmu", "--out", m / "lmu.csv"])
    for ds in ["cnmc", "hehr", "lmu"]:
        run(PY + [S / "extract_handcrafted.py", "--manifest", m / f"{ds}.csv", "--out", f / f"{ds}_hand.parquet", "--workers", "2"])
    run(PY + [S / "run_cell_benchmark.py", "--manifest", m / "cnmc.csv", "--task", "cnmc_all_vs_hem",
              "--features", f"hand={f / 'cnmc_hand.parquet'}", "--models", "M1=hand:lgbm", "M1lr=hand:lr",
              "--splits", "image", "grouped", "--folds", "3", "--cv-where", "cohort=train",
              "--test-where", "cohort=prelim_test", "--n-boot", "50", "--out", r / "cnmc"])
    run(PY + [S / "run_patient_benchmark.py", "--manifest", m / "hehr.csv", "--task", "hehr_aml_vs_control",
              "--features", f"hand={f / 'hehr_hand.parquet'}", "--models", "P1=hand:pool_lr", "P2=hand:abmil",
              "--folds", "3", "--epochs", "3", "--n-boot", "50", "--out", r / "hehr"])
    run(PY + [S / "merge_sources.py", "--manifests", m / "cnmc.csv", m / "lmu.csv",
              "--block", "hand", f / "cnmc_hand.parquet", f / "lmu_hand.parquet", "--out-dir", tmp / "merged"])
    run(PY + [S / "run_cell_benchmark.py", "--manifest", tmp / "merged" / "manifest.csv", "--task", "blast_crosslab",
              "--features", f"hand={tmp / 'merged' / 'hand.npy'}", "--models", "M1=hand:lr", "--splits", "grouped",
              "--folds", "3", "--cv-where", "dataset=cnmc", "--test-where", "dataset=lmu", "--n-boot", "50",
              "--out", r / "crosslab"])
    run(PY + [S / "run_site_id.py", "--manifest", tmp / "merged" / "manifest.csv",
              "--features", f"hand={tmp / 'merged' / 'hand.npy'}", "--out", r / "site_id.csv"])
    run(PY + [S / "make_tables.py", r / "cnmc", r / "hehr", r / "crosslab", "--out", r / "tables.md"])
    s = pd.read_csv(r / "cnmc" / "summary.csv")
    assert (s.loc[s.split == "grouped", "patients_in_several_folds"] == 0).all(), "grouped split leaked patients"
    assert (s.loc[s.split == "image", "patients_in_several_folds"] > 0).any(), "image split should mix patients"
    print(f"\nSMOKE TEST PASSED  (workspace: {tmp})")


if __name__ == "__main__":
    main()
