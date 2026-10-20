"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import load_feature_block


def parse_kv(items: list[str] | None) -> dict:
    """['cohort=Discovery', 'dataset=lmu,bmc'] -> {'cohort': ['Discovery'], 'dataset': ['lmu', 'bmc']}"""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v.split(",")
    return out


def parse_models(items: list[str]) -> dict:
    """['M1=hand:lgbm', 'M4=hand+effnet:lr'] -> {'M1': (['hand'], 'lgbm'), ...}"""
    out = {}
    for it in items:
        name, spec = it.split("=", 1)
        blocks, clf = spec.split(":")
        out[name] = (blocks.split("+"), clf)
    return out


def load_blocks(items: list[str], n_rows: int) -> dict:
    """['hand=features/x.parquet', 'effnet=features/x.npy'] -> {'hand': array, ...}"""
    blocks = {}
    for it in items:
        name, path = it.split("=", 1)
        X = load_feature_block(path)
        if len(X) != n_rows:
            raise SystemExit(f"feature block {name} has {len(X)} rows, manifest has {n_rows}")
        blocks[name] = X
    return blocks


def save_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def fmt_ci(row: pd.Series, metric: str, digits: int = 3) -> str:
    v, lo, hi = row.get(metric), row.get(metric + "_lo"), row.get(metric + "_hi")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
