"""Task specifications: turn a manifest into rows, binary labels and groups.

A task spec (configs/tasks.json) looks like
    {
      "label_col": "cell_type",                 # column holding the class
      "positive": ["Blast, no lineage spec"],   # values counted as label 1
      "negative": "*rest*",                     # or an explicit list of values
      "exclude": ["Arifact", "unlabelled"],     # values dropped entirely
      "where": {"dataset": ["apl_aml"]},        # row filters (column -> allowed values)
      "level": "cell"                           # "cell" or "patient"
    }
Patient-level tasks take the label from the patient's diagnosis; all of a patient's cells
(labelled or not) form the bag.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TaskData:
    index: np.ndarray      # row indices into the manifest / feature matrices
    y: np.ndarray          # 0/1 labels (cell level) or per-row patient label (patient level)
    groups: np.ndarray     # patient ids ("" replaced by a unique id per row)
    frame: pd.DataFrame    # the selected manifest rows


def load_tasks(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def apply_where(df: pd.DataFrame, where: dict | None) -> pd.Series:
    keep = pd.Series(True, index=df.index)
    for col, allowed in (where or {}).items():
        allowed = [allowed] if isinstance(allowed, str) else allowed
        keep &= df[col].astype(str).isin([str(a) for a in allowed])
    return keep


def select(manifest: pd.DataFrame, spec: dict, extra_where: dict | None = None) -> TaskData:
    df = manifest.copy()
    keep = apply_where(df, spec.get("where")) & apply_where(df, extra_where)
    col = spec["label_col"]
    vals = df[col].astype(str)
    keep &= ~vals.isin(spec.get("exclude", []))
    pos = vals.isin(spec["positive"])
    neg_spec = spec.get("negative", "*rest*")
    neg = ~pos if neg_spec == "*rest*" else vals.isin(neg_spec)
    keep &= pos | neg
    sub = df[keep]
    y = pos[keep].astype(int).to_numpy()
    groups = sub["patient_id"].astype(str).to_numpy().copy()
    empty = groups == ""
    groups[empty] = [f"row{i}" for i in np.flatnonzero(empty)]   # no patient id -> each row is its own group
    if spec.get("level", "cell") == "cell" and empty.any():
        print(f"  note: {empty.sum():,} rows without patient id; grouped splits cannot protect them")
    return TaskData(index=sub.index.to_numpy(), y=y, groups=groups, frame=sub)


def patient_table(task: TaskData) -> pd.DataFrame:
    """One row per patient: patient_id, label, n_cells."""
    t = pd.DataFrame({"patient_id": task.groups, "y": task.y})
    out = t.groupby("patient_id").agg(y=("y", "first"), n_cells=("y", "size")).reset_index()
    if t.groupby("patient_id")["y"].nunique().max() > 1:
        raise ValueError("a patient has mixed labels in a patient-level task")
    return out
