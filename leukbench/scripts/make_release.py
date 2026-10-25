"""Release package: locked patient-level split files and patient-level predictions of the reported pipelines."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leukbench.splits import make_folds
W = Path(sys.argv[1]); R = W / "results"; out = Path(sys.argv[2]); (out / "splits").mkdir(parents=True, exist_ok=True); (out / "predictions").mkdir(parents=True, exist_ok=True)
for name, man, role in [("apl_aml", "apl_aml.csv", "cohort"), ("cnmc", "cnmc.csv", "split_role"), ("mll", "mll.csv", "cohort")]:
    m = pd.read_csv(W / "manifests" / man, keep_default_na=False)
    pt = m.groupby("patient_id").agg(set=(role, "first"), diagnosis=("diagnosis", "first"), cells=("path", "size")).reset_index()
    if name == "mll":
        pt["subtype"] = m.groupby("patient_id").subtype.first().values
    pt.to_csv(out / "splits" / f"{name}_patients.csv", index=False)
    dev = {"apl_aml": "Discovery", "cnmc": "Development"}.get(name)
    if dev:   # cell-level fold ids (patient-grouped and image-level, seeds 0-2) on the development set
        d = m[m[role] == dev].reset_index()
        y = (d.diagnosis == ("APL" if name == "apl_aml" else "ALL")).astype(int).values
        F = pd.DataFrame({"row": d["index"], "patient_id": d.patient_id})
        for kind in ["grouped", "image"]:
            for s in [0, 1, 2]:
                F[f"{kind}_seed{s}"] = make_folds(y, d.patient_id.values, kind, 5, s)
        F.to_csv(out / "splits" / f"{name}_cell_folds.csv.gz", index=False)
def bench(f, m):
    p = pd.read_csv(R / f / "predictions.csv"); return p[(p.model == m) & (p["eval"] == "test")].groupby("patient_id").agg(y=("y", "first"), score=("p", "mean")).reset_index()
P = {("mll", "primary_ensemble"): pd.read_csv(R / "mll_ensemble/ensemble_test.csv").rename(columns={"p": "score"}),
     ("mll", "dinobloom_engineered"): bench("mll_patient", "PDF"), ("mll", "engineered_pooled_lr"): bench("mll_patient", "P1"),
     ("mll", "finetuned_cnn"): pd.read_csv(R / "mll_finetune_macenko/patient_predictions_test.csv").rename(columns={"p": "score"}),
     ("apl_validation", "primary_ensemble"): pd.read_csv(R / "ensemble_apl_macenko/ensemble_test.csv").rename(columns={"p": "score"}),
     ("apl_validation", "dinobloom_attention_mil"): bench("apl_patient_dino", "PDM"),
     ("cnmc_fold2", "primary_ensemble"): pd.read_csv(R / "ensemble_cnmc/ensemble_test.csv").rename(columns={"p": "score"}),
     ("cnmc_fold2", "dinobloom_engineered"): bench("cnmc_patient_dino", "PDF"),
     ("apl_all_reverse", "dinobloom_engineered_trained_on_mll"): bench("reciprocal_mll_to_apl", "PDF")}
for (coh, m), d in P.items():
    d[["patient_id", "y", "score"]].to_csv(out / "predictions" / f"{coh}__{m}.csv", index=False)
(out / "README.md").write_text("""# Release files
* `splits/*_patients.csv`: every patient with its set (development / external), diagnosis and number of cells (MLL: genetic subtype).
* `splits/*_cell_folds.csv.gz`: 5-fold assignments of development cells (patient-grouped and image-level, seeds 0-2), as produced by `leukbench.splits.make_folds`.
* `predictions/<cohort>__<model>.csv`: patient-level scores of the reported pipelines on each external cohort (y = 1 for APL / ALL).
  Thresholds are those fixed on development data (see `docs/PRESPECIFICATION.md`, `docs/PRESPEC_EXTERNAL_MLL.md`).
* No images are redistributed; obtain them from the original sources (DeepAPL GitHub, TCIA C-NMC 2019, TCIA AML-Cytomorphology_MLL_Helmholtz).
""")
print("release written")
