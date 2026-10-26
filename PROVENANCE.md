# Provenance of the study

All times are Cairo time (UTC+03:00). Commit identifiers refer to branch `revision-1` (merged into `main`).

| Date and time | Commit | What was added | Test-set exposure at that point |
| --- | --- | --- | --- |
| 2026-10-20 07:03 | `8b2cb36` | First analysis: leakage audit (image vs. patient-grouped CV), patient-level models, baselines, EDA, feature engineering, first manuscript | APL-AML Validation scored for every configuration (cell and patient level, feature-engineering steps) |
| 2026-10-20 08:07 | `6f12d98` | PCA analysis (whitened scores), training curves, `SETTINGS.md` (stated DinoBloom "not evaluated, weights unreachable") | As above |
| 2026-10-20 08:18 | `ff3036b` | Author placeholders (version 1.0, archived as branch `archive/v1.0-review`) | As above |
| 2026-10-21 07:19 | `f47f3ec` | **Pre-specification** of the primary and secondary pipelines (`docs/PRESPECIFICATION.md`) | Written after first-analysis Validation results and after a preprocessing pilot whose subset included Validation patients (disclosed in section 5 of the file and in the paper) |
| 2026-10-21 09:58 | `8fe9eb7` | C-NMC replication, preprocessing comparison, fine-tuning, natural-scale PCA, pre-specified pipelines, supplement | C-NMC fold 2 scored for several configurations (never used for selection) |
| 2026-10-21 12:07 | `e9900bd` | Leakage vs. external generalization gap decomposition, scan-time confounding, label-free shift test, calibration, bag-size check | Exploratory analyses |
| 2026-10-21 15:29 | `be3fed9` | Bootstrap CIs for the decomposition, exact CIs, Macenko sensitivity analyses | Exploratory |
| 2026-10-21 ~19:00 | - | zenodo.org enabled in the computing environment; DinoBloom-S weights downloaded (Zenodo 10908163) | - |
| 2026-10-21 19:41 | `acf173b` | **Pre-specification of the independent MLL test** (`docs/PRESPEC_EXTERNAL_MLL.md`), DinoBloom loader | MLL cohort downloaded and extracted, **not scored** |
| 2026-10-22 05:41 | `7fce3d4` | Independent MLL test (scored once), DinoBloom on C-NMC, APL-AML and MLL, SHAP, C-NMC improvement pilot | MLL: confirmatory analysis per plan |
| 2026-10-25 21:24 | `1bc8d45` | Workload analysis, split-sample threshold adaptation, reverse transfer (train MLL, test APL-AML), release files | Post hoc on MLL scores (no model changed); reverse transfer exploratory (APL-AML previously examined) |
| 2026-10-25 21:56 | `d6b8cd8` | Documentation reconciled with the manuscript (`README.md`, `PROVENANCE.md`, `SETTINGS.md`), `reproduce.sh`, `verify_claims.py`, wording and figure fixes | No new analyses |
| 2026-10-25 | **`739c33f8f083e6836a42e234395baf43630a6763`** | **Code and results freeze cited in the manuscript.** Clean-room reproduction check (53/53 identical), figure and wording fixes | No new analyses |

## Statements superseded

* "The Validation cohort is never used for fitting, selection, thresholds or calibration" (README, version 1):
  correct for fitting, thresholds and calibration; **incorrect for selection**, because many configurations were
  compared on it. The paper treats it as a reused, exploratory test set.
* "DinoBloom ... not evaluated because its weights were unreachable" (`SETTINGS.md`, 20 October): correct until
  21 October; DinoBloom results were produced afterwards (commit `7fce3d4`).
* "The Validation cohort, scored once" (`SETTINGS.md`, first analysis): withdrawn on 21 October (commit `e9900bd`).

## Which manuscript claim comes from which result folder

| Claim | Result folder(s) | Script |
| --- | --- | --- |
| Leakage 0.04-0.11, external generalization gap 0.17-0.24 (T1) | `results/v2_cell_apl_cell_*`, `results/gap_bootstrap` | `run_cell_benchmark.py`, `run_gap_bootstrap.py` |
| Scan-time confounding | `results/date_confound` | `run_date_confound.py` |
| Primary pipeline, later APL-AML cohort (AUROC 0.806, 12/12, spec. 0.17) | `results/ensemble_apl_macenko` | `run_ensemble.py` |
| Independent MLL test (primary 0.862, 24/24, spec. 0.00; DinoBloom + engineered 0.877) | `results/mll_patient`, `mll_finetune_macenko`, `mll_ensemble`, `mll_fe_macenko` | `run_patient_benchmark.py`, `run_finetune_cached.py`, `run_ensemble.py`, `run_feature_engineering.py` |
| Workload (top 20%: 15-17 of 24; 60-75% for 95% recall) | `results/utility` | `run_utility.py` |
| Threshold adaptation on separate patients | `results/utility/local_recal_split.csv`, `results/mll_recal` | `run_utility.py`, `run_local_recal_ext.py` |
| Reverse transfer (0.75-0.78; MIL 0.54) | `results/reciprocal_mll_to_apl` | `run_patient_benchmark.py` |
| DinoBloom on C-NMC and APL-AML | `results/cnmc_cell_dino`, `cnmc_patient_dino`, `apl_patient_dino` | `extract_embeddings.py`, benchmarks |

Later commits on `paper-v4` change only the compiled manuscript (to insert this commit identifier) and documentation.
