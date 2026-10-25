# Leakage, Shift and the Operating Point: Evaluating Blood-Smear Leukemia Classifiers Across Time and Laboratories

Code, configurations, result files, release files and LaTeX sources for the manuscript of the same title
(IEEE Transactions format, under review). The study separates three questions for blood-smear leukemia
classifiers: **discrimination** (how well patients are ranked), **fixed-threshold performance** (what a threshold
chosen on development data does on new patients) and **review workload** (how many patients must be reviewed to
reach a sensitivity target).

> **Which version matches the paper?** The submitted manuscript corresponds to the commit named in the submission
> and listed in [`PROVENANCE.md`](PROVENANCE.md), which also gives the full history of the study: what was added when,
> and what each test set had been exposed to at that time. Earlier versions of this README and of
> `leukbench/docs/SETTINGS.md` described the first analysis (20 October 2026), before DinoBloom could be downloaded
> and before the Validation cohort's reuse was documented; those statements are superseded (see "Data use" below).

## Data (not redistributed)

| Cohort | Role | Source |
| --- | --- | --- |
| APL-AML (106 patients; 34 APL) | Development (Discovery, 82) and later temporal cohort (Validation, 24) | Sidhom et al. 2021, <https://github.com/sidhomj/DeepAPL> |
| C-NMC 2019 (73 subjects) | Replication: folds 0-1 development, fold 2 external | TCIA, doi:10.7937/tcia.2019.dc64i46r |
| AML-Cytomorphology_MLL_Helmholtz (24 APL, 105 other AML used) | Independent external test (prespecified, scored once) | Hehr et al. 2023; TCIA (CC BY 4.0) |
| DinoBloom-S weights | Hematology foundation model | Koch et al. 2024; Zenodo record 10908163 |

## Data use: what each test set was used for (honest statement)

* **No external cohort was used to fit a model, a threshold or a probability calibration.**
* **APL-AML Validation cohort: reused test set (exploratory).** It was evaluated for many configurations in the
  first analysis and the revisions, and a preprocessing pilot subset included its patients. Earlier documentation
  stated it was "never used for selection"; that statement was too strong and is withdrawn.
* **C-NMC fold 2: secondary.** Evaluated for several configurations, never used for selection.
* **MLL cohort: confirmatory.** The analysis plan (`leukbench/docs/PRESPEC_EXTERNAL_MLL.md`, commit `acf173b`) was
  committed before any MLL image was scored. Workload, split-sample threshold adaptation and reverse transfer on
  this cohort were added afterwards at reviewers' request (post hoc; no model changed).
* **DinoBloom** was not evaluated in the first analysis because its weights could not be downloaded in the computing
  environment; it was evaluated from 21 October 2026 (commits `acf173b`, `7fce3d4`) once access was enabled.

A per-analysis registry is in the supplement (Table "Analysis registry") and in `PROVENANCE.md`.

## Check the headline numbers in seconds (no images needed)

```bash
pip install numpy pandas scikit-learn scipy
python verify_claims.py          # recomputes the main external results from leukbench/release/predictions
```

## Reproduce everything

```bash
git clone https://github.com/sidhomj/DeepAPL            # APL-AML images
pip install -r leukbench/requirements.txt
bash reproduce.sh                                        # every step, in order, with the exact commands
```

`reproduce.sh` lists each command that produced each result folder (feature extraction, cell- and patient-level
benchmarks, fine-tuning, ensemble, DinoBloom, independent test, workload, threshold adaptation, reverse transfer,
tables and figures). Every setting is documented in [`leukbench/docs/SETTINGS.md`](leukbench/docs/SETTINGS.md).
Runs were done on a CPU-only machine (2 vCPUs, 8 GB RAM).

## Repository layout

| Folder | Contents |
| --- | --- |
| `leukbench/` | Python package and scripts (manifests, patient-grouped splits, engineered features, EfficientNet-B0 and DinoBloom encoders, LR / LightGBM / attention MIL, fine-tuning, metrics with patient-level bootstrap, analyses) |
| `leukbench/docs/` | `PRESPECIFICATION.md` (commit `f47f3ec`), `PRESPEC_EXTERNAL_MLL.md` (commit `acf173b`), `SETTINGS.md` |
| `leukbench/release/` | Locked patient-level split files, development cell folds, patient-level predictions of every reported pipeline on every external cohort |
| `results/` | Outputs of every experiment (summaries, predictions, configurations) |
| `manuscript/` | LaTeX sources, figures, tables, `make_results.py` (generates every table and figure from `results/`) |
| `manifests/` | Cell manifest of APL-AML with portable paths |

## Evaluation principles

* Splits are made by **patient**; every scaler, imputer, PCA, filter and classifier is fitted inside training folds.
* Decision thresholds are fixed on development data for **recall >= 0.95**; discrimination, fixed-threshold
  performance and workload are reported separately.
* Confidence intervals resample **patients**, not cells; recall and specificity also with exact binomial intervals.
* Analyses are labelled confirmatory or exploratory according to what their test set had been exposed to.
