# Release files
* `splits/*_patients.csv`: every patient with its set (development / external), diagnosis and number of cells (MLL: genetic subtype).
* `splits/*_cell_folds.csv.gz`: 5-fold assignments of development cells (patient-grouped and image-level, seeds 0-2), as produced by `leukbench.splits.make_folds`.
* `predictions/<cohort>__<model>.csv`: patient-level scores of the reported pipelines on each external cohort (y = 1 for APL / ALL).
  Thresholds are those fixed on development data (see `docs/PRESPECIFICATION.md`, `docs/PRESPEC_EXTERNAL_MLL.md`).
* No images are redistributed; obtain them from the original sources (DeepAPL GitHub, TCIA C-NMC 2019, TCIA AML-Cytomorphology_MLL_Helmholtz).
