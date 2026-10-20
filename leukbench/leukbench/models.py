"""Classifiers as scikit-learn Pipelines.

Imputation, scaling and the classifier are one Pipeline, so every statistic (feature means,
scales) is learned on the training fold only. Concatenated feature blocks ("fusion") are
standardised per column, so neither block dominates by scale.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_classifier(kind: str, seed: int = 0):
    if kind == "lr":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")),
        ])
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                                   random_state=seed, verbose=-1, n_jobs=2)),
        ])
    raise ValueError(kind)


def load_feature_block(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype("float32")
    import pandas as pd
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    return df.to_numpy(dtype="float32")


def feature_names(path: str, prefix: str) -> list[str]:
    if path.endswith(".npy"):
        return [f"{prefix}_{i}" for i in range(np.load(path, mmap_mode="r").shape[1])]
    import pandas as pd
    cols = (pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, nrows=1)).columns
    return [f"{prefix}:{c}" for c in cols]


def stack_blocks(blocks: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """names like ['hand', 'effnet'] -> horizontally stacked matrix."""
    return np.hstack([blocks[n] for n in names])
