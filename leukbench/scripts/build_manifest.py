"""Build a standard manifest CSV for one dataset.

Examples
  python scripts/build_manifest.py cnmc   --root data/C-NMC_Leukemia --labels-csv data/C-NMC_test_prelim_phase_data_labels.csv --out manifests/cnmc.csv
  python scripts/build_manifest.py hehr   --root data/AML_Hehr --out manifests/hehr.csv
  python scripts/build_manifest.py apl_aml --root data/DeepAPL/Data --out manifests/apl_aml.csv
  python scripts/build_manifest.py folder_per_class --root data/AML-Cytomorphology_LMU --dataset lmu --out manifests/lmu.csv
  python scripts/build_manifest.py folder_per_patient --root data/caitomorph/images --dataset caito \
         --meta data/caitomorph/metadata.csv --id-col patient --label-col coarse_class --out manifests/caito.csv
"""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench import manifests as M

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["cnmc", "hehr", "apl_aml", "folder_per_class", "folder_per_patient"])
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--labels-csv")
ap.add_argument("--dataset")
ap.add_argument("--patient-regex")
ap.add_argument("--meta")
ap.add_argument("--id-col")
ap.add_argument("--label-col")
ap.add_argument("--cohort-col")
a = ap.parse_args()

if a.kind == "cnmc":
    df = M.build_cnmc(a.root, a.labels_csv)
elif a.kind == "hehr":
    df = M.build_hehr(a.root)
elif a.kind == "apl_aml":
    df = M.build_apl_aml(a.root)
elif a.kind == "folder_per_class":
    df = M.build_folder_per_class(a.root, a.dataset, a.patient_regex)
else:
    df = M.build_folder_per_patient(a.root, a.dataset, a.meta, a.id_col, a.label_col, a.cohort_col)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"wrote {a.out}")
