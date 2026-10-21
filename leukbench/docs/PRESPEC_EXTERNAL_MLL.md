# Pre-specification: independent external test on AML-Cytomorphology_MLL_Helmholtz

Written and committed **before** any model is scored on this cohort (2026-10-21, Cairo time).

## Data
* Source: AML-Cytomorphology_MLL_Helmholtz (Hehr et al., 2023; TCIA, CC BY 4.0), obtained from the Kaggle mirror
  `gchan357/human-aml-cytomorphology-dataset` (81,214 single-cell TIFF images, 144 x 144 px, 189 patients).
* Used: 24 patients with APL (PML::RARA) = positive; 105 patients with AML with NPM1, CBFB::MYH11 or RUNX1::RUNX1T1 = negative.
  The 60 healthy donors are excluded (the question is APL vs other AML). Up to 200 randomly drawn cells per patient (seed 0).
* No image or label of this cohort is used for training, threshold selection, calibration or model choice.

## Models (all trained on the APL-AML Discovery cohort only, exactly as in the paper)
1. **Primary pipeline** (unchanged): Macenko normalization if the label-free shift test is positive (reference stains
   from Discovery images), fine-tuned EfficientNet-B0 (mean cell probability), engineered features with pooled LR and
   with attention MIL (engineered features on original images), mean of percentile scores; threshold for recall >= 0.95
   on Discovery out-of-fold scores.
2. **Secondary pipeline** (unchanged): Macenko -> frozen EfficientNet-B0 -> PCA-64 (natural scale) -> pooled LR.
3. **DinoBloom variant** (added in revision 3, selected on C-NMC development data before this test): DinoBloom-S CLS
   features (original images) concatenated with engineered features, pooled LR.

## Shift test
Label-free: AML cells of Discovery vs AML cells of the MLL cohort, stain/colour engineered features, patient-grouped
5-fold CV, permutation p-value. If positive, Macenko is applied to the CNN branch (as pre-specified for APL-AML).

## Reported once
Patient-level AUROC (bootstrap 95% CI), recall with missed APL patients and specificity (exact 95% CIs), F1, NPV,
accuracy, at the Discovery threshold. Results are reported whatever they are; no model is changed afterwards.
Expected difficulty: different laboratory, scanner (Metafer, 40x), image size and staining; class balance 24:105.
