"""Split generation. Splits are created once, saved, and reused by every model.

image-level : StratifiedKFold on cells, ignoring patients  (what most 99-100% papers do; leaks)
grouped     : StratifiedGroupKFold with group = patient   (honest)
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_folds(y: np.ndarray, groups: np.ndarray, kind: str, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Return an int array: fold id (0..n_splits-1) for each row."""
    fold = np.full(len(y), -1, dtype=int)
    if kind == "image":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        it = cv.split(np.zeros(len(y)), y)
    elif kind == "grouped":
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        it = cv.split(np.zeros(len(y)), y, groups)
    else:
        raise ValueError(kind)
    for k, (_, te) in enumerate(it):
        fold[te] = k
    assert (fold >= 0).all()
    return fold


def check_no_patient_overlap(groups: np.ndarray, fold: np.ndarray) -> int:
    """Number of patients that appear in more than one fold (0 for grouped splits)."""
    n = 0
    for g in np.unique(groups):
        if len(np.unique(fold[groups == g])) > 1:
            n += 1
    return n


def patient_folds(patient_y: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Folds over patients (one row per patient) for patient-level tasks."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(patient_y), -1, dtype=int)
    for k, (_, te) in enumerate(cv.split(np.zeros(len(patient_y)), patient_y)):
        fold[te] = k
    return fold
