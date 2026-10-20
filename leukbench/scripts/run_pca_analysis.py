"""Why does PCA on CNN embeddings help? Diagnostics on the APL-AML cohort.

For k in a grid (and the raw 1,280-d embedding):
  - explained variance of the first k components (PCA fitted on Discovery cells only)
  - cohort identifiability: AUROC of telling Validation from Discovery AML cells (patient-grouped CV),
    i.e. how much acquisition-specific signal the representation keeps
  - APL signal at cell level: AUROC of APL vs AML on Validation cells, LR trained on Discovery cells
  - model size of the patient-level classifier: 5 pooled statistics x k inputs vs 65 training patients per fold

  python scripts/run_pca_analysis.py --manifest manifests/apl_aml.csv --effnet features/apl_aml_effnet.npy --out results/pca_analysis
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench.models import load_feature_block

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--effnet", required=True)
ap.add_argument("--ks", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128, 256])
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

man = pd.read_csv(a.manifest, keep_default_na=False)
E = np.nan_to_num(load_feature_block(a.effnet), nan=0.0)
disc = (man.cohort == "Discovery").values
sc = StandardScaler().fit(E[disc])
Z = sc.transform(E)
pca = PCA(max(a.ks), random_state=0).fit(Z[disc])
cumvar = np.cumsum(pca.explained_variance_ratio_)
Zp = pca.transform(Z)

rng = np.random.default_rng(0)
aml = np.flatnonzero((man.diagnosis == "AML").values)
aml = np.sort(rng.choice(aml, min(6000, len(aml)), replace=False))
yc = (man.cohort.values[aml] == "Validation").astype(int)
g = man.patient_id.values[aml]
tr = np.flatnonzero(disc); te = np.flatnonzero(~disc)
tr = np.sort(rng.choice(tr, min(8000, len(tr)), replace=False))
y = (man.diagnosis == "APL").astype(int).values
clf = lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=4000, class_weight="balanced"))])

rows = []
for k in a.ks + [0]:
    X = E if k == 0 else Zp[:, :k]
    p = cross_val_predict(clf(), X[aml], yc, groups=g, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    cid = roc_auc_score(yc, p)
    m = clf().fit(X[tr], y[tr])
    cell_auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    dim = E.shape[1] if k == 0 else k
    rows.append(dict(k=dim, label="raw" if k == 0 else f"PCA-{k}", explained_var=1.0 if k == 0 else float(cumvar[k - 1]),
                     cohort_id_auroc=cid, cell_apl_auroc_external=cell_auc, patient_model_inputs=5 * dim,
                     inputs_per_training_patient=5 * dim / 65.6))
    print(f"  {rows[-1]['label']:8s} var {rows[-1]['explained_var']:.2f}  cohort-ID AUROC {cid:.3f}  cell APL AUROC (ext) {cell_auc:.3f}  "
          f"patient-model inputs {5 * dim}", flush=True)
pd.DataFrame(rows).to_csv(out / "pca_analysis.csv", index=False)
print(f"wrote {out}")
