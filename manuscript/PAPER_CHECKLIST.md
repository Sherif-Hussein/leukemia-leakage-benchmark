# Paper checklist (agreed with the author, 21 Oct 2026)

Every revision of the manuscript must contain, clearly and in humanized wording:

1. **Dataset EDA** for every dataset: class balance (patients and cells), cells per patient, cell-type mix,
   feature shift between development and external sets, cohort identifiability, redundancy, feature stability.
2. **Preprocessing** used to handle each dataset issue, and why (e.g. unreadable files, C-NMC black background,
   stain/colour shift -> stain normalization if adopted), with before/after evidence.
3. **Why this model and not others**: evidence-based choice (literature + our own comparisons: engineered vs
   frozen CNN vs fine-tuned CNN vs foundation model; attention vs average pooling; MIL vs pooled).
4. **Before and after results** for every addition (PCA, fine-tuning, attention pooling, ensemble,
   preprocessing): AUROC, accuracy, recall (missed patients), specificity, F1, NPV, per fold and external.
5. **Defense of training settings**: number of epochs vs early stopping (nested inner CV), learning rate,
   thresholds; sensitivity analyses showing conclusions do not hinge on them.
6. **Results analysis**: why each component helps or not; fold/seed variability vs noise floor; dataset
   dependence; robustness (mean, worst fold, spread).
7. **Feature engineering** used and its impact, before and after.

Also: IEEE format, proper citations, block diagrams kept up to date, honest (no leakage, external sets untouched).
8. **End-to-end pipeline diagrams**: one per pipeline (primary and secondary), from input image to output
   decision, every step labelled with its setting and measured cost per patient.
9. **PCA framed as a complexity/performance trade-off** (agreed 21 Oct): PCA-64 is used to reduce complexity
   (storage, parameters, training time, memory), NOT to improve accuracy. Report a trade-off table
   (Macenko without PCA vs Macenko + PCA-64, primary as reference) with performance AND cost per step,
   and state the cost in performance explicitly (F1 -0.03, AUROC -0.06, recall unchanged on APL-AML).
