# Settings and implementation

This document describes every setting used to produce the results in the manuscript, where it lives in the code, and the exact command that produced each result folder. No model, threshold or calibration was fitted on the Validation cohort. It was, however, evaluated for many configurations during the first analysis and the revision, so it is reported as a reused test set (see PRESPECIFICATION.md, section 7).

## 1. Computing environment

| Item | Value |
|---|---|
| Hardware | Intel Xeon 2.80 GHz, 2 vCPUs, 7.8 GB RAM, no GPU |
| OS | Ubuntu 24.04.5 LTS |
| Python | 3.13 |
| Libraries | PyTorch 2.14.1 (CPU), scikit-learn 1.9.1, LightGBM 4.7.0, scikit-image 0.26.0, timm 1.0.30, pandas, numpy, matplotlib |
| Torch threads | 1 for MIL training (`--threads 1`): small models run fastest without thread oversubscription |
| Random seeds | 0, 1, 2 (fold assignment, cell sampling, model initialization); results are averaged over seeds |

## 2. Data

| Setting | Value | Code |
|---|---|---|
| Cohort | APL-AML (Sidhom et al., 2021), 106 patients, 25,724 cell images, CellaVision DM100 | `scripts/build_manifest.py` |
| Development / external | Discovery (82 patients, diagnosed before Jan 2019) / Validation (24 patients, later) | `manifests/apl_aml.csv`, column `cohort` |
| Unreadable images | 10 truncated JPEGs kept as rows with NaN features; imputed with training-fold medians | `leukbench/encoders.py` (`ImageList` returns `ok=False`) |
| Cell labels | Analyzer class verified by the operator; 15,632 labeled, 10,092 unlabeled | manifest column `cell_type` |

### Tasks (`configs/tasks.json`)

| Task | Unit | Positive class | Excluded |
|---|---|---|---|
| T1 `apl_cell_diagnosis` | cell | cell from an APL patient | none |
| T2 `apl_cell_blast` | cell | "Blast, no lineage spec" | artefacts, unidentified, smudge cells, platelets, unlabeled |
| T3 `apl_patient` | patient | APL | none (all cells form the bag) |

## 3. Splits and evaluation protocol (`leukbench/splits.py`)

* **Image-level CV**: `StratifiedKFold(5, shuffle=True, random_state=seed)` over cells, ignoring patients (reproduces the leaky protocol).
* **Patient-grouped CV**: `StratifiedGroupKFold(5, shuffle=True, random_state=seed)` with group = patient. `check_no_patient_overlap` asserts zero shared patients.
* **Patient-level tasks**: `StratifiedKFold(5)` over the 82 Discovery patients (`patient_folds`).
* **External test**: the Validation cohort (never used for fitting; evaluated for several configurations). For T3, the external prediction is the mean of all 15 fold models (5 folds × 3 seeds).
* **Calibration**: temperature scaling on an inner 80/20 split of each training fold, bounded search T ∈ [0.05, 20] (`metrics.fit_temperature`).

## 4. Engineered features, 93 per cell (`leukbench/handcrafted.py`)

| Step | Setting |
|---|---|
| Resize | RGB, bilinear, 192 × 192 (`SIZE = 192`) |
| Color deconvolution | `skimage.color.rgb2hed` (Ruifrok–Johnston H&E matrix) |
| Nucleus | Gaussian σ = 1 on hematoxylin → Otsu → opening (disk 2) → connected component nearest the center (within 0.30 × size); fallback: darkest 15 % of gray levels |
| Basophilic region | blue-dominant pixels (blue − red above an Otsu threshold within the foreground), within ~1 nuclear radius, opening (disk 2), closing (disk 4) |
| Perinuclear ring | distance 2–9 px from the nucleus, excluding pixels brighter than the 90th lightness percentile − 3 |
| GLCM | 33 gray levels, distances 1, 2, 4 px, angles 0, π/4, π/2, 3π/4; contrast, dissimilarity, homogeneity, energy, correlation (averaged over angles) |
| LBP | uniform, (P, R) = (8, 1) and (16, 2); normalized histograms |
| Groups | nuclear shape (9), basophilic region (10), stain/color (26), GLCM texture + entropy (16), LBP (28), nucleoli/granules (4) |
| Runtime | 16.1 min for 25,724 cells with 2 processes |

## 5. Deep features (`leukbench/encoders.py`)

| Setting | Value |
|---|---|
| Encoder | EfficientNet-B0, ImageNet weights (`timm`), frozen, global-average-pooled 1,280-d |
| Input | bicubic resize to 224 × 224, ImageNet mean/std normalization, no augmentation |
| Extraction | batch size 64, 2 workers, CPU, `torch.no_grad()`; 51.2 min for 25,724 cells |
| DinoBloom | loader implemented (teacher weights, `backbone.` prefix stripped, 257-token positional embedding); not evaluated because its weights were unreachable |

## 6. Feature engineering (`scripts/run_feature_engineering.py`, class `CellTransform`)

All steps are fitted on the **training cells of each fold** only.

| Config | What it does |
|---|---|
| `eng_raw` | 93 engineered features, median imputation |
| `eng_corrfilter` | drop the later feature of every pair with \|r\| > 0.9 (≤ 20,000 training cells sampled) |
| `eng_slidenorm` | per-slide median/SD standardization of the 26 stain features |
| `eng_l1` | L1-penalized LR on pooled features, C = 0.5, SAGA |
| `eng_stable` | correlation filter + slide normalization |
| `eff_raw` | 1,280-d EfficientNet-B0 embedding |
| `eff_pca{k}` | `StandardScaler` → `PCA(k, random_state=0, svd_solver="auto")`. k = 16, 32, 64 pre-specified; 4, 8, 128, 256 = sensitivity sweep |
| `fusion_pca32` | filtered engineered features + PCA-32 embedding |
| `composition` | "digital differential": an 8-class cell-type LR (on fusion-PCA32 features) predicts every cell; the patient is described by class fractions and mean probabilities. Training-patient compositions come from an inner 3-fold patient-grouped CV (out-of-fold) |
| `mil_*` | gated attention MIL on the given representation |

Patient vector for pooled models: mean, SD, 10th, 50th and 90th percentile of every cell feature (5 × dimensions); up to 500 cells per patient, sampled once with seed 0.

## 7. Classifiers

| Model | Settings | Code |
|---|---|---|
| Cell LR | L2, C = 0.1, L-BFGS, 3,000 iterations, class-balanced; median imputation and standardization in the pipeline | `leukbench/models.py` |
| Cell LightGBM | 400 trees, learning rate 0.05, 31 leaves, subsample 0.8 every iteration, colsample 0.8, class-balanced, 2 threads | `leukbench/models.py` |
| Pooled patient LR | L2, C = 0.1, 5,000 iterations, class-balanced, standardization | `scripts/run_feature_engineering.py` |
| Gated attention MIL | Linear(d→128)–ReLU–Dropout(0.25); gated attention (tanh × sigmoid, 64 units); linear classifier on the attention-weighted mean | `leukbench/mil.py` |
| MIL optimization | AdamW, learning rate 2e-4 (constant, no schedule), weight decay 1e-3, one bag per step, 40 epochs, no early stopping | `train_abmil` |
| MIL loss | BCE with positive weight n_AML / n_APL of the training fold | `train_abmil` |
| MIL augmentation | random removal of 20 % of a bag's cells per step (bags > 10 cells) | `train_abmil` |

### Re-run baselines (`scripts/run_baselines.py`)

* **B1** promyelocyte fraction per patient (biomarker comparator of Sidhom et al.), ranked directly.
* **B2** 17-class differential count → LR (C = 0.1).
* **B3** EfficientNet-B0 + RBF-SVM (C = 1, γ = scale, Platt probabilities, class-balanced, 8,000 training cells), cell scores averaged per patient.
* **B4** single-cell LR, cell scores averaged per patient.

## 8. Operating point and statistics (`leukbench/metrics.py`)

| Setting | Value |
|---|---|
| Threshold | highest threshold with sensitivity ≥ 0.95 on out-of-fold Discovery predictions (`threshold_for_sensitivity`); Youden threshold reported for reference |
| Threshold metrics | recall, missed positives (FN), specificity, F1, NPV, accuracy |
| Threshold-free | AUROC; ECE with 15 equal-width bins; Brier score |
| Oracle | specificity at 100 % recall if the threshold were re-set on Validation (post hoc reference only, not deployable) |
| CIs | patient-level bootstrap, 2,000 resamples (1,000 for cell-level CV) |
| Paired tests | paired patient-level bootstrap of ΔAUROC, 4,000 resamples; two-sided p |

## 9. Training-dynamics diagnostics (`scripts/run_training_curves.py`)

These are diagnostics only and never used to choose a model, epoch or threshold.

* **MIL curves**: 5 Discovery folds × 3 seeds; after every epoch, training loss (class-weighted BCE, mean over bags) and held-out-fold loss, AUROC and accuracy (`train_abmil(..., monitor=...)`, attached as `predict.history`). Inputs: engineered, raw EfficientNet-B0, EfficientNet-B0 + PCA-32.
* **LightGBM curves**: cell-level T1 with engineered features, patient-grouped folds; training vs held-out binary log-loss and AUROC per boosting round.
* **Sample-size learning curve**: pooled LR trained on n = 16, 24, 32, 48 and all (~66) Discovery patients per fold (stratified subsample); AUROC on training patients, the held-out fold and the Validation cohort.

## 10. PCA diagnostics (`scripts/run_pca_analysis.py`)

For k ∈ {4, 8, 16, 32, 64, 128, 256} and the raw embedding (scaler + PCA fitted on Discovery cells):

* explained variance of the first k components;
* **cohort identifiability**: AUROC of an LR telling Validation from Discovery **AML** cells (6,000 sampled, patient-grouped 5-fold CV); 0.5 = no acquisition signal;
* **cell-level APL signal**: LR trained on 8,000 Discovery cells, AUROC on all Validation cells;
* patient-model size: 5k inputs vs ~66 training patients per fold.

## 11. Commands

```bash
W=<work dir>   # contains manifests/, features/, results/
python scripts/extract_handcrafted.py --manifest $W/manifests/apl_aml.csv --out $W/features/apl_aml_hand.parquet
python scripts/extract_embeddings.py  --manifest $W/manifests/apl_aml.csv --encoder effnet_b0_imagenet --out $W/features/apl_aml_effnet.npy
F="hand=$W/features/apl_aml_hand.parquet effnet=$W/features/apl_aml_effnet.npy"
for t in apl_cell_blast apl_cell_diagnosis; do
  python scripts/run_cell_benchmark.py --manifest $W/manifests/apl_aml.csv --task $t --features $F \
    --models M1=hand:lgbm M1lr=hand:lr M2=effnet:lr M4=hand+effnet:lr --splits image grouped --seeds 0 1 2 \
    --cv-where cohort=Discovery --test-where cohort=Validation --out $W/results/v2_cell_$t
done
python scripts/run_patient_benchmark.py --manifest $W/manifests/apl_aml.csv --task apl_patient --features $F \
  --models P1=hand:pool_lr P2=effnet:pool_lr P3=hand+effnet:pool_lr P4=hand:abmil P5=effnet:abmil P6=hand+effnet:abmil \
  --cv-where cohort=Discovery --test-where cohort=Validation --seeds 0 1 2 --out $W/results/v2_patient
python scripts/run_baselines.py   --manifest $W/manifests/apl_aml.csv --effnet $W/features/apl_aml_effnet.npy --hand $W/features/apl_aml_hand.parquet --out $W/results/v2_baselines
python scripts/run_feature_engineering.py --manifest $W/manifests/apl_aml.csv --hand $W/features/apl_aml_hand.parquet --effnet $W/features/apl_aml_effnet.npy --seeds 0 1 2 --out $W/results/feature_engineering
python scripts/run_feature_engineering.py ... --configs eff_pca4 eff_pca8 eff_pca128 eff_pca256 --out $W/results/feature_engineering_pca
python scripts/run_pca_analysis.py    --manifest $W/manifests/apl_aml.csv --effnet $W/features/apl_aml_effnet.npy --out $W/results/pca_analysis
python scripts/run_training_curves.py --manifest $W/manifests/apl_aml.csv --hand $W/features/apl_aml_hand.parquet --effnet $W/features/apl_aml_effnet.npy --out $W/results/training_curves
python scripts/run_eda.py ...  # exploratory analysis, results/eda
LEUKBENCH_WORK=$W python ../manuscript/make_results.py   # tables, figures, facts.json
```

## 12. Result folders

| Folder | Content |
|---|---|
| `v2_cell_apl_cell_blast`, `v2_cell_apl_cell_diagnosis` | cell-level leakage audit (image vs grouped CV vs external), per seed |
| `v2_patient` | patient-level models P1–P6, CV and external, attention weights |
| `v2_baselines` | re-run designs from the literature on the same split |
| `feature_engineering`, `feature_engineering_part2` | feature-engineering configurations (patient level) |
| `feature_engineering_pca` | PCA sensitivity sweep (k = 4, 8, 128, 256) |
| `pca_analysis` | explained variance, cohort identifiability and cell-level APL signal per k |
| `training_curves` | MIL per-epoch history, LightGBM per-round history, sample-size learning curve |
| `ablation_*` | feature-group ablation and calibration |
| `eda` | exploratory data analysis |

## Revision 2 analyses

| Result folder | Script | What it does |
|---|---|---|
| `revision_stats` | `scripts/run_revision_stats.py --work <work> --out results/revision_stats --skip-shift` | C-NMC bag-size confound; calibration (Brier, intercept, slope; raw and Platt on development OOF); reliability diagrams |
| `revision_stats/shift` | `scripts/run_revision_stats.py --work <work> --out results/revision_stats/shift --perms 200 --perms-cnn 50` | label-free shift test with patient-level permutation p-values |
| `date_confound` | `scripts/run_date_confound.py` | scan-time proxy (median signed-cell number), confounding of APL with scan time, time-window CV, likelihood-ratio test |
| `preproc_clahe` | `scripts/run_preproc_test.py --variants clahe` | CLAHE arm of the preprocessing comparison |
| `local_recalibration` | `scripts/run_local_recalibration.py` | now uses natural-scale PCA and includes the primary and secondary pipelines |
| `gap_bootstrap` | `scripts/run_gap_bootstrap.py --boot 500` | patient-level bootstrap CIs of the leakage and shift components |
| `v2_patient_macenko_eng`, `ensemble_apl_all_macenko` | `run_patient_benchmark.py` on Macenko engineered features; `run_ensemble.py` | sensitivity: primary pipeline with engineered features on normalized images |
| `ensemble_apl` | `run_ensemble.py --patient results/v2_patient --finetune results/finetune_t1` | sensitivity: primary pipeline without Macenko normalization |

## Revision 3: foundation model and independent external test

| Result folder | Script | What it does |
|---|---|---|
| `cnmc_cell_dino`, `cnmc_patient_dino`, `apl_patient_dino` | `extract_embeddings.py --encoder dinobloom_s --chunk 7000`; `run_cell_benchmark.py`; `run_patient_benchmark.py` | DinoBloom-S (Zenodo 10908163, CLS token, original images) alone and with engineered features |
| `cnmc_cell_lgbm`, `pilot_cnmc_crop`, `shap_cnmc` | `run_cell_benchmark.py`; `run_finetune_cached.py --crop --no-test`; `run_shap.py` | C-NMC improvement pilot (development CV only) and SHAP explanation |
| `mll_external`, `mll_patient`, `mll_finetune_macenko`, `mll_ensemble`, `mll_fe_macenko`, `mll_recal` | see `docs/PRESPEC_EXTERNAL_MLL.md`; `run_local_recal_ext.py` | pre-registered independent test on AML-Cytomorphology_MLL_Helmholtz (24 APL, 105 AML), local recalibration |
