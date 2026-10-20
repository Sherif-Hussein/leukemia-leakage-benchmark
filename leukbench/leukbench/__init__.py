"""LeukBench: leakage-aware, cross-dataset benchmark for AI leukemia detection.

Modules
-------
manifests   build a standard CSV manifest (path, dataset, patient_id, ...) per dataset
tasks       turn a manifest + task spec into (rows, binary labels, groups)
splits      image-level vs patient-grouped folds, saved to disk and reused by every model
handcrafted engineered morphology / colour / texture features per cell
encoders    frozen deep encoders (ImageNet EfficientNet, DinoBloom) -> cached embeddings
models      scikit-learn pipelines (scaler + classifier fit inside the training fold only)
mil         gated attention MIL (ABMIL) for patient-level diagnosis
metrics     AUROC, sensitivity/specificity, ECE, Brier, patient-level bootstrap CIs,
            temperature scaling, paired bootstrap tests
"""

__version__ = "0.1.0"
