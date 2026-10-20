# Leakage-Aware, Sensitivity-First Evaluation of Leukemia Detection in Blood Smears

Code, results and manuscript for the study *"Beyond 100% Accuracy: Leakage-Aware, Sensitivity-First
Evaluation of Engineered and Deep Features for Leukemia Detection in Blood Smears"* (IEEE Transactions
format, in preparation).

The study uses the public APL-AML cohort (106 patients with AML, 34 with APL; 25,724 single-cell images)
released by Sidhom et al., *npj Precision Oncology* 5:38 (2021), and asks how much of the reported
performance of blood-smear classifiers survives patient-level, temporally separated evaluation.

## Repository layout

| Folder | Contents |
| --- | --- |
| `leukbench/` | Python package and scripts: manifests, patient-grouped splits, engineered features, frozen CNN / DinoBloom encoders, cell- and patient-level models (LR, LightGBM, attention MIL), metrics with patient-level bootstrap, EDA, feature engineering, baselines. See `leukbench/README.md`. |
| `manuscript/` | IEEEtran LaTeX source (`main.tex`, `refs.bib`), figures, generated tables, figure/table generators and the compiled `main.pdf`. |
| `results/` | Outputs of every experiment: per-model summaries (`summary.csv`), predictions, configs, EDA and feature-engineering tables. |
| `manifests/apl_aml.csv` | One row per cell image: path (relative to the DeepAPL clone), patient, cohort, diagnosis, cell type. |

Raw images are not redistributed here; they are available from the original authors at
<https://github.com/sidhomj/DeepAPL>. Cached feature files (`features/`, ~140 MB) are excluded and can be
regenerated with the scripts.

## Reproduce

```bash
git clone https://github.com/sidhomj/DeepAPL            # data (Data/All, Data/master.csv)
pip install -r leukbench/requirements.txt
python -I leukbench/tests/smoke_test.py                  # must print SMOKE TEST PASSED

W=.                                                      # repository root
python -I leukbench/scripts/build_manifest.py apl_aml --root DeepAPL/Data --out $W/manifests/apl_aml.csv
python -I leukbench/scripts/extract_handcrafted.py --manifest $W/manifests/apl_aml.csv --out $W/features/apl_aml_hand.parquet --workers 4
python -I leukbench/scripts/extract_embeddings.py  --manifest $W/manifests/apl_aml.csv --encoder effnet_b0_imagenet --out $W/features/apl_aml_effnet.npy
```

Experiments (each writes to `results/<name>/`; exact commands are in each `config.json`):

| Experiment | Script |
| --- | --- |
| Leakage audit, cell level (T1, T2) | `run_cell_benchmark.py` |
| Patient-level APL detection (T3) | `run_patient_benchmark.py` |
| Baselines from prior work on the same split | `run_baselines.py` |
| Exploratory data analysis | `run_eda.py` |
| Feature engineering (PCA, selection, composition, normalization) | `run_feature_engineering.py` |
| Feature-group ablation and calibration | `run_ablation.py` |
| PCA diagnostics (explained variance, cohort identifiability, cell-level signal per k) | `run_pca_analysis.py` |
| Training dynamics: MIL loss/AUROC per epoch, LightGBM loss per round, sample-size learning curves | `run_training_curves.py` |

**Every setting of every component** (preprocessing, features, PCA, classifiers, optimizer, learning rate,
epochs, thresholds, bootstrap) and the exact command for each result folder are documented in
[`leukbench/docs/SETTINGS.md`](leukbench/docs/SETTINGS.md).

Tables and figures: `cd manuscript && LEUKBENCH_WORK=.. python make_results.py && python make_fig_eda.py`, then compile with
`pdflatex main && bibtex main && pdflatex main && pdflatex main` (or upload `manuscript/` to Overleaf).

## Evaluation principles

- Splits are made by **patient**; the Validation cohort is never used for fitting, selection, thresholds or calibration.
- Every scaler, imputer, PCA, feature filter and classifier is fitted inside the training fold only.
- Decision thresholds are fixed on the development data for **sensitivity ≥ 0.95**, and results report recall,
  missed cases, specificity, F1, NPV and calibration, not only accuracy.
- Confidence intervals resample **patients**, not cells.
