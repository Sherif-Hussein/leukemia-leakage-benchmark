# Pre-specification of the final analysis (revision 1)

Written and committed **before** the full revision-1 run, on 2026-10-21 (Cairo time, UTC+03:00).
Nothing below may be changed after the full-run results are seen; any deviation is reported as such.

## 1. Primary model (one, reported as the main result)

| Stage | Specification |
|---|---|
| Stain normalization | **Macenko**, applied only if the label-free EDA shift test detects an acquisition shift (cohort-identification AUROC of control-class cells clearly above 0.5). Result of the test: APL-AML yes (0.75 on colour features), C-NMC no (0.44-0.52). Reference stain estimated from Discovery (development) images only. Applied to the CNN branch only; engineered features use the original images. |
| Normalization granularity | Per patient (stain vectors estimated once per patient) **if** the subset check in Section 4 passes; otherwise per cell. |
| Cell features | (a) 93 engineered features; (b) EfficientNet-B0 fine-tuned on the last two stages and head, average pooling, AdamW lr 3e-4, 1 warm-up epoch + cosine decay, 6 epochs with best-epoch checkpoint on 15% inner validation patients, up to 5,000 training cells per fold, seed 0. |
| Patient model | **Pre-specified ensemble**: mean of the percentile scores of (i) engineered pooled LR, (ii) engineered attention MIL (fixed 40 epochs), (iii) mean of fine-tuned cell probabilities. |
| Decision threshold | Highest threshold giving recall >= 0.95 on out-of-fold development predictions. |
| Primary metrics | External recall (missed patients), specificity, F1, accuracy, NPV, AUROC with patient-bootstrap 95% CI. |

## 2. Secondary model (efficiency option, pre-declared)

Macenko (same rule) -> frozen EfficientNet-B0 -> **PCA-64** (fitted on training-fold cells, component scores at natural scale, no whitening) -> pooled statistics -> LR -> same threshold rule. Reported with measured computational cost (storage, parameters, training time, memory, time per patient).

## 3. Not part of the final pipeline (reported as comparisons / ablations)

Image-level vs patient-level splits; frozen CNN; PCA sweep (8-256, whitened and natural scale); Reinhard, CLAHE, edge sharpening; nested early stopping; attention pooling (3 folds); pooled 106-patient CV; feature-engineering steps.

## 4. Per-patient vs per-cell Macenko: decision rule (label-free / development-only)

On the 6,360-cell pilot subset, per-patient Macenko is adopted if, compared with per-cell Macenko,
(a) cohort-identification AUROC is not higher by more than 0.02, and
(b) development (Discovery) CV AUROC at cell and patient level is not lower by more than 0.01.
The external (Validation) cohort is **not** used for this decision.

## 5. Disclosure

The preprocessing comparison (sharpening, CLAHE, Reinhard, Macenko) and a pilot of Macenko + PCA-64 were evaluated on a subset that included the APL-AML Validation cohort. This is disclosed in the paper. The choice of the primary model above does not rest on those external pilot numbers: Macenko is applied by the label-free shift rule, and the primary model is the configuration that was most consistent across both datasets before the pilot (fine-tuning and engineered features). C-NMC and the pooled analysis provide confirmation outside the APL-AML Validation cohort.

## 6. Outcome of the Section 4 rule (recorded 2026-10-21, before the full run)

Per-patient vs per-cell Macenko on the pilot subset (development data only):
cohort-identification AUROC 0.554 vs 0.543 (criterion a: pass); Discovery CV AUROC cells 0.859 vs 0.850 (pass),
patients 0.925 vs 0.942 (criterion b: **fail**, -0.017). **Decision: per-cell Macenko.**
Cost: per-cell normalization adds about 0.14 s per cell (about 68 s per 500-cell patient) on one CPU thread;
per-patient would add about 1 s per patient. This trade-off is reported in the paper.

## 7. Erratum and disclosure (added 2026-10-21, revision 2; the sections above are unchanged)

* Section 6 overstated the cost of per-cell Macenko normalization. Measured on the full run, it takes
  **0.071 s per cell** on one CPU thread (30.5 min for 25,724 cells), i.e. about **36 s per 500-cell patient**,
  not 0.14 s per cell / 68 s per patient. The decision itself (per-cell) rested on criterion (b), not on cost.
* Use of the APL-AML Validation cohort: it was never used to fit a model, a threshold or a calibration, but it was
  evaluated for many configurations in the first analysis (commit 8b2cb36 to ff3036b, 2026-10-20) and in the
  preprocessing pilot, before this pre-specification was written. The pre-specification therefore protects the
  final pipelines from being tuned on their own full-run results, but not from knowledge of earlier Validation
  results. The manuscript reports the Validation cohort as a reused test set.
