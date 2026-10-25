# Reproducibility check

## 1. Headline numbers from released predictions (seconds, no images)

`python verify_claims.py` recomputes 19 headline external results (independent MLL cohort, later APL-AML cohort, C-NMC,
reverse transfer) from `leukbench/release/predictions/` and the released development thresholds. Output on the
pinned version: **ALL CLAIMS REPRODUCED** (19/19 within rounding of the paper's values).

## 2. Clean-room re-run of the confirmatory and post-hoc analyses (25 October 2026)

All analyses on the independent MLL cohort and the reverse transfer were re-run from the stored features into a new,
empty folder with the commands in `reproduce.sh` (sections 6 and 7), and compared with the reported result folders
using `leukbench/scripts/compare_reproduction.py`:

| Re-run | Result folders | Numbers compared | Identical |
| --- | --- | --- | --- |
| Engineered and DinoBloom patient models, Discovery -> MLL | `mll_patient` | 12 (AUROC, recall, specificity of 4 models) | 12 |
| Reverse transfer, MLL -> APL-AML | `reciprocal_mll_to_apl` | 12 | 12 |
| Secondary pipeline (Macenko + frozen CNN +/- PCA-64) | `mll_fe_macenko` | 6 | 6 |
| Fine-tuned EfficientNet-B0 (cell and patient level) | `mll_finetune_macenko` | 6 | 6 |
| Primary pipeline (ensemble) | `mll_ensemble` | 3 | 3 |
| Workload (top 20%, workload for 95% recall) on both external APL cohorts | `utility` | 14 | 14 |
| **Total** | | **53** | **53 (maximum absolute difference 0.0)** |

The full comparison is in `results/reproduction_check/comparison.csv`. The re-run is deterministic because every
random component is seeded (folds, cell sampling, model initialization, bootstrap).

## 3. What was not re-run here, and why

* Feature extraction from images (engineered features, EfficientNet-B0, DinoBloom-S, Macenko normalization) was not
  repeated in the clean-room check because it takes many CPU hours; the commands are in `reproduce.sh`. Images are not
  redistributed and must be obtained from the original sources (see README).
* Earlier exploratory analyses (scan-time proxy, shift tests, PCA sweeps, early stopping) are reproducible with the
  commands in `leukbench/docs/SETTINGS.md` but were not part of this check.

## 4. Environment

Python 3.13.16; numpy 2.5.3; pandas 3.0.5; scikit-learn 1.9.1; scipy 1.18.1; PyTorch 2.14.1 (CPU); timm 1.0.30;
LightGBM 4.7.0; scikit-image 0.26.0. CPU only, 2 vCPUs, 8 GB RAM.
