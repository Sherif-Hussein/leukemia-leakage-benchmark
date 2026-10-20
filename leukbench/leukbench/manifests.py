"""Build one standard manifest CSV per dataset.

Every manifest has the columns
    path        absolute path to one single-cell image
    dataset     short dataset name (cnmc, hehr, apl_aml, lmu, bmc, ...)
    patient_id  subject / patient identifier ("" when the dataset has none)
    cohort      dataset-defined cohort or phase ("train", "prelim_test", "Discovery", ...)
    diagnosis   patient-level diagnosis ("" when not applicable)
    cell_type   cell-level class ("unlabelled" when the cell has no class)

Dataset layouts were taken from the dataset papers / their official code. Each builder
prints a short summary; ALWAYS check it against the dataset documentation after download.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COLUMNS = ["path", "dataset", "patient_id", "cohort", "diagnosis", "cell_type"]


def _images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            yield p


def _finish(rows: list[dict], name: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS).fillna("")
    if df.empty:
        raise SystemExit(f"[{name}] no images found - check --root")
    print(f"[{name}] {len(df):,} images, {df['patient_id'].replace('', pd.NA).nunique()} patients")
    for col in ["cohort", "diagnosis", "cell_type"]:
        vc = df[col].value_counts()
        if 1 < len(vc) <= 30:
            print(f"  {col}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    return df


# --------------------------------------------------------------------------- C-NMC 2019
def build_cnmc(root: str, labels_csv: str | None = None) -> pd.DataFrame:
    """C-NMC 2019 (ISBI challenge, TCIA).

    Training data: <root>/.../fold_{0,1,2}/{all,hem}/UID_<subj>_<img>_<cell>_{all,hem}.bmp
    Preliminary test: <root>/.../C-NMC_test_prelim_phase_data/<n>.bmp with a labels CSV
    (columns Patient_ID, new_names, labels; labels 1 = ALL, 0 = hem). The subject ID is the
    token after "UID_" in the original file name (e.g. UID_11 or UID_H10).
    """
    root = Path(root)
    rows = []
    prelim_dir = None
    for p in _images(root):
        parts = [s.lower() for s in p.parts]
        if p.parent.name.lower() in {"all", "hem"} and any(s.startswith("fold_") for s in parts):
            m = re.match(r"UID_(H?\d+)_", p.name, re.I)
            if not m:
                continue
            rows.append(dict(path=str(p.resolve()), dataset="cnmc", patient_id=f"cnmc_{m.group(1).upper()}",
                             cohort="train", diagnosis="", cell_type=p.parent.name.lower()))
        elif "prelim" in str(p.parent).lower():
            prelim_dir = p.parent
    if labels_csv and prelim_dir is not None:
        lab = pd.read_csv(labels_csv)
        for _, r in lab.iterrows():
            m = re.match(r"UID_(H?\d+)_", str(r["Patient_ID"]), re.I)
            f = prelim_dir / str(r["new_names"])
            if m and f.exists():
                rows.append(dict(path=str(f.resolve()), dataset="cnmc", patient_id=f"cnmc_{m.group(1).upper()}",
                                 cohort="prelim_test", diagnosis="",
                                 cell_type="all" if int(r["labels"]) == 1 else "hem"))
    elif prelim_dir is not None:
        print("[cnmc] prelim test images found but no --labels-csv given; skipping them")
    return _finish(rows, "cnmc")


# --------------------------------------------------------------------------- AML-Hehr
def build_hehr(root: str) -> pd.DataFrame:
    """AML-Cytomorphology_MLL_Helmholtz (Hehr et al. 2023, TCIA).

    Layout used by the official SCEMILA code: <root>/metadata.csv (index column patient_id,
    column bag_label in {control, PML_RARA, NPM1, CBFB_MYH11, RUNX1_RUNX1T1}) and images in
    <root>/data/<bag_label>/<patient_id>/*.tif
    """
    root = Path(root)
    meta = pd.read_csv(root / "metadata.csv")
    rows = []
    for _, r in meta.iterrows():
        pdir = root / "data" / str(r["bag_label"]) / str(r["patient_id"])
        for p in _images(pdir):
            rows.append(dict(path=str(p.resolve()), dataset="hehr", patient_id=f"hehr_{r['patient_id']}",
                             cohort="all", diagnosis=str(r["bag_label"]), cell_type="unlabelled"))
    return _finish(rows, "hehr")


# --------------------------------------------------------------------------- APL-AML (Sidhom 2021)
def build_apl_aml(root: str) -> pd.DataFrame:
    """APL vs AML dataset of Sidhom et al. (npj Precis Oncol 2021; github.com/sidhomj/DeepAPL).

    <root>/master.csv (Patient_ID, Diagnosis APL/AML, Cohort Discovery/Validation) and
    <root>/All/<Patient_ID>/[<slide>/]{Signed slides/<cell type>, Unsigned slides}/*.jpg
    Signed cells carry a technologist cell-type label; unsigned cells are unlabelled.
    """
    root = Path(root)
    meta = pd.read_csv(root / "master.csv").set_index("Patient_ID")
    rows = []
    for p in _images(root / "All"):
        rel = p.relative_to(root / "All").parts
        pid = rel[0]
        if pid not in meta.index:
            continue
        signed = "Signed slides" in rel
        if signed:
            ct = p.parent.name if p.parent.name != "Signed slides" else "unlabelled"
        else:
            ct = "unlabelled"
        rows.append(dict(path=str(p.resolve()), dataset="apl_aml", patient_id=f"apl_{pid}",
                         cohort=meta.loc[pid, "Cohort"], diagnosis=meta.loc[pid, "Diagnosis"], cell_type=ct))
    return _finish(rows, "apl_aml")


# --------------------------------------------------------------------------- generic layouts
def build_folder_per_class(root: str, dataset: str, patient_regex: str | None = None) -> pd.DataFrame:
    """<root>/<CLASS>/*.img  (e.g. AML-Cytomorphology_LMU, BMC, MLL23).

    patient_regex: optional regex with one group applied to the file name to recover a patient ID.
    Without it, patient_id stays empty and the dataset must only be used as a whole-dataset hold-out.
    """
    root = Path(root)
    rx = re.compile(patient_regex) if patient_regex else None
    rows = []
    for p in _images(root):
        pid = ""
        if rx:
            m = rx.search(p.name)
            pid = f"{dataset}_{m.group(1)}" if m else ""
        rows.append(dict(path=str(p.resolve()), dataset=dataset, patient_id=pid, cohort="all",
                         diagnosis="", cell_type=p.parent.name))
    return _finish(rows, dataset)


def build_folder_per_patient(root: str, dataset: str, meta_csv: str, id_col: str, label_col: str,
                             cohort_col: str | None = None) -> pd.DataFrame:
    """<root>/<patient_id>/**/*.img plus a metadata CSV mapping patient -> diagnosis
    (e.g. the cAItomorph MLL release)."""
    root = Path(root)
    meta = pd.read_csv(meta_csv).astype({id_col: str}).set_index(id_col)
    rows = []
    for pdir in sorted(d for d in root.iterdir() if d.is_dir()):
        if pdir.name not in meta.index:
            continue
        r = meta.loc[pdir.name]
        for p in _images(pdir):
            rows.append(dict(path=str(p.resolve()), dataset=dataset, patient_id=f"{dataset}_{pdir.name}",
                             cohort=str(r[cohort_col]) if cohort_col else "all",
                             diagnosis=str(r[label_col]), cell_type="unlabelled"))
    return _finish(rows, dataset)
