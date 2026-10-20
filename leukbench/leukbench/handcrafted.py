"""Engineered single-cell features that mirror what hematologists assess.

Pipeline per image (resized to SIZE x SIZE so features are comparable across datasets):
 1. Nucleus mask: hematoxylin channel (rgb2hed) -> Otsu -> clean-up -> component nearest the centre.
 2. Basophilic cell region (blue-dominant pixels near the nucleus) and a perinuclear ring used as
    the cytoplasm proxy. Pale cytoplasm cannot be separated from red cells by colour alone, so no
    full cell outline / exact N:C ratio is claimed (a learned segmenter would be needed for that).
 3. Features: nuclear and cell shape, N:C ratio, stain / colour statistics per compartment,
    chromatin texture (GLCM, entropy), LBP micro-texture, nucleolus / granule proxies.

Nothing here reads image size, file type or dataset identity, so the features cannot encode
"which lab" except through genuine colour differences (checked by the site-ID experiment).
"""
from __future__ import annotations

import warnings

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2hed, rgb2hsv, rgb2lab
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk

SIZE = 192
_GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
_GLCM_DIST = [1, 2, 4]


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Version-stable replacement for skimage's remove_small_objects (keeps objects > min_size px)."""
    lab, n = ndi.label(mask)
    if n == 0:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    keep = sizes > min_size
    keep[0] = False
    return keep[lab]


def binary_opening(mask, se):
    return ndi.binary_opening(mask, structure=se)


def binary_closing(mask, se):
    return ndi.binary_closing(mask, structure=se)


def _central_component(mask: np.ndarray, prefer: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    lab = label(mask)
    n = int(lab.max())
    if n == 0:
        return mask, 0
    h, w = mask.shape
    best, best_score = 0, -np.inf
    for r in regionprops(lab):
        y, x = r.centroid
        d = np.hypot(y - h / 2, x - w / 2) / np.hypot(h / 2, w / 2)
        score = r.area * (1 - 0.8 * d)
        if prefer is not None:
            score += 10 * prefer[lab == r.label].sum()
        if score > best_score:
            best, best_score = r.label, score
    return lab == best, n


def segment(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (nucleus_mask, cell_mask, n_nuclear_fragments)."""
    hed = rgb2hed(rgb)
    h = ndi.gaussian_filter(hed[..., 0], 1.0)
    t = threshold_otsu(h)
    nuc = binary_opening(h > t, disk(2))
    nuc = remove_small_objects(nuc, 30)
    # lobulation: fragments of nucleus within the central region (segmented neutrophils have several)
    nuc_all = nuc.copy()
    nuc, _ = _central_component(nuc)
    if nuc.sum() < 30:                       # fall back to darkest region
        g = rgb.mean(axis=2)
        nuc, _ = _central_component(g < np.percentile(g, 15))
    cy, cx = ndi.center_of_mass(nuc)
    yy, xx = np.indices(nuc.shape)
    near = np.hypot(yy - cy, xx - cx) < 0.30 * SIZE
    n_frag = int(label(nuc_all & near).max())

    # "Basophilic cell region": pixels bluer than red cells / background (blue minus red above an
    # Otsu threshold), kept within ~1 nuclear radius of the nucleus. It captures blue cytoplasm
    # (blasts, lymphoid cells, monocytes) and excludes red cells; very pale cytoplasm may be missed,
    # so it is reported as a proxy, not a full cell outline.
    x = rgb.astype(float)
    grey = x.mean(axis=2)
    br = x[..., 2] - x[..., 0]
    fg = grey < threshold_otsu(grey) + 10
    t_br = threshold_otsu(br[fg]) if fg.sum() > 50 else 0.0
    cand = fg & (br > t_br - 5)
    req = np.sqrt(nuc.sum() / np.pi)
    cand &= ndi.distance_transform_edt(~nuc) <= max(req, 10)
    cand = binary_opening(cand, disk(2))
    cand = binary_closing(cand | nuc, disk(4))
    cell, _ = _central_component(ndi.binary_fill_holes(cand), prefer=nuc)
    cell = ndi.binary_fill_holes(cell | nuc)
    return nuc, cell, max(n_frag, 1)


def _major(r):
    return getattr(r, "axis_major_length", None) or r.major_axis_length


def _minor(r):
    v = getattr(r, "axis_minor_length", None)
    return v if v is not None else r.minor_axis_length


def _shape(mask: np.ndarray, prefix: str) -> dict:
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return {f"{prefix}_{k}": np.nan for k in ["area", "perimeter", "circularity", "eccentricity",
                                                  "solidity", "extent", "axis_ratio", "major_axis"]}
    r = props[0]
    a = r.area / mask.size
    per = r.perimeter / mask.shape[0]
    circ = 4 * np.pi * r.area / (r.perimeter ** 2 + 1e-6)
    return {
        f"{prefix}_area": a,
        f"{prefix}_perimeter": per,
        f"{prefix}_circularity": min(circ, 1.5),
        f"{prefix}_eccentricity": r.eccentricity,
        f"{prefix}_solidity": r.solidity,
        f"{prefix}_extent": r.extent,
        f"{prefix}_axis_ratio": _minor(r) / (_major(r) + 1e-6),
        f"{prefix}_major_axis": _major(r) / mask.shape[0],
    }


def _stats(values: np.ndarray, prefix: str) -> dict:
    if values.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan}
    return {f"{prefix}_mean": float(values.mean()), f"{prefix}_std": float(values.std())}


def _glcm(grey: np.ndarray, mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    out = {}
    if len(ys) < 20:
        for d in _GLCM_DIST:
            for p in _GLCM_PROPS:
                out[f"glcm_{p}_d{d}"] = np.nan
        return out
    crop = grey[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    q = np.clip((crop / 256 * 31).astype(np.uint8) + 1, 1, 32)   # 1..32, 0 = outside nucleus
    q[~m] = 0
    g = graycomatrix(q, _GLCM_DIST, [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=33, symmetric=True)
    g[0, :, :, :] = 0
    g[:, 0, :, :] = 0                                             # drop pairs touching background
    g = g.astype(float)
    s = g.sum(axis=(0, 1), keepdims=True)
    g = np.divide(g, s, out=np.zeros_like(g), where=s > 0)
    for p in _GLCM_PROPS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = graycoprops(g, p)                                    # (dist, angle)
        for i, d in enumerate(_GLCM_DIST):
            out[f"glcm_{p}_d{d}"] = float(np.nanmean(v[i]))
    return out


def _lbp(grey: np.ndarray, mask: np.ndarray) -> dict:
    out = {}
    for P, R in [(8, 1), (16, 2)]:
        code = local_binary_pattern(grey.astype(np.uint8), P, R, method="uniform")
        vals = code[mask]
        hist = np.bincount(vals.astype(int), minlength=P + 2)[: P + 2].astype(float)
        hist /= hist.sum() + 1e-9
        for i, v in enumerate(hist):
            out[f"lbp_P{P}R{R}_{i}"] = v
    return out


def features_from_rgb(rgb: np.ndarray) -> dict:
    nuc, cell, n_frag = segment(rgb)
    # cytoplasm proxy = perinuclear ring (2-9 px at 192 px) excluding bright background
    L0 = rgb2lab(rgb)[..., 0]
    dist = ndi.distance_transform_edt(~nuc)
    cyto = (dist > 2) & (dist <= 9) & (L0 < np.percentile(L0, 90) - 3)
    f: dict = {}
    f.update(_shape(nuc, "nuc"))
    f.update(_shape(cell, "bcell"))
    f["nc_ratio_basophilic"] = nuc.sum() / (cell.sum() + 1e-6)
    f["nuc_fragments"] = n_frag
    f["basophilic_rim"] = (cell & ~nuc).sum() / (nuc.sum() + 1e-6)

    hed = rgb2hed(rgb)
    lab = rgb2lab(rgb)
    hsv = rgb2hsv(rgb)
    for name, comp in [("nuc", nuc), ("cyto", cyto)]:
        f.update(_stats(hed[..., 0][comp], f"{name}_H"))
        f.update(_stats(hed[..., 1][comp], f"{name}_E"))
        f.update(_stats(lab[..., 0][comp], f"{name}_L"))
        f.update(_stats(lab[..., 1][comp], f"{name}_a"))
        f.update(_stats(lab[..., 2][comp], f"{name}_b"))
        f.update(_stats(hsv[..., 1][comp], f"{name}_sat"))
    # contrast between compartments (stain-robust ratios)
    f["H_ratio_nuc_cyto"] = (f["nuc_H_mean"] - f["cyto_H_mean"]) if cyto.any() else np.nan
    f["L_diff_nuc_cyto"] = (f["cyto_L_mean"] - f["nuc_L_mean"]) if cyto.any() else np.nan

    grey = rgb.astype(float).mean(axis=2)
    f.update(_glcm(grey, nuc))
    nv = grey[nuc]
    if nv.size:
        hist = np.histogram(nv, bins=32, range=(0, 256))[0].astype(float)
        hist /= hist.sum()
        f["nuc_entropy"] = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
    else:
        f["nuc_entropy"] = np.nan
    f.update(_lbp(grey, nuc))

    # nucleolus proxy: pale spots inside the nucleus; granule proxy: dark specks in cytoplasm
    h = hed[..., 0]
    if nv.size > 50:
        hn = h[nuc]
        pale = nuc & (h < hn.mean() - 1.5 * hn.std())
        pl = label(remove_small_objects(pale, 4))
        f["nucleoli_count"] = int(pl.max())
        f["nucleoli_area"] = float(pale.sum() / (nuc.sum() + 1e-6))
    else:
        f["nucleoli_count"], f["nucleoli_area"] = np.nan, np.nan
    if cyto.sum() > 50:
        gc = grey[cyto]
        dark = cyto & (grey < gc.mean() - 1.5 * gc.std())
        f["granule_count"] = int(label(remove_small_objects(dark, 3)).max())
        f["granule_area"] = float(dark.sum() / (cyto.sum() + 1e-6))
    else:
        f["granule_count"], f["granule_area"] = 0, 0.0
    return f


def features_from_path(path: str) -> dict:
    try:
        rgb = np.asarray(Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))
        return features_from_rgb(rgb)
    except Exception as e:  # corrupt image etc. -> all-NaN row, imputed later inside the training fold
        print(f"  feature error {path}: {e}")
        return {}
