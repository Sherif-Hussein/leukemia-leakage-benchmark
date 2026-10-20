# LeukBench — leakage-aware, cross-dataset benchmark for AI leukemia detection

Code for **Paper 1**: *Beyond 100%: a leakage-aware, cross-laboratory benchmark of handcrafted,
deep and foundation-model features for leukemia detection in blood smears.*

Every model sees **the same saved splits**, every statistic is learned **inside the training fold**,
and every confidence interval resamples **patients, not cells**.

## Install

```bash
pip install -r requirements.txt          # Python 3.10+; GPU optional except for run_finetune.py
python -I tests/smoke_test.py            # 2-4 min, synthetic data, must print SMOKE TEST PASSED
```

## Data

| Dataset | Get it from | Builder |
| --- | --- | --- |
| APL-AML (Sidhom 2021), 106 patients | `git clone https://github.com/sidhomj/DeepAPL` (images are in `Data/`) | `apl_aml --root DeepAPL/Data` |
| C-NMC 2019 | TCIA collection *C-NMC 2019* (training folds + preliminary test + labels CSV) | `cnmc --root ... --labels-csv ...` |
| AML-Hehr (189 patients) | TCIA *AML-Cytomorphology_MLL_Helmholtz* | `hehr --root ...` (expects `metadata.csv` + `data/<label>/<patient>/`) |
| AML-Cytomorphology_LMU | TCIA *AML-Cytomorphology_LMU* | `folder_per_class --dataset lmu` |
| BMC (bone marrow) | TCIA *Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer* | `folder_per_class --dataset bmc` |
| cAItomorph MLL release | doi.org/10.82296/hmgu-nefeli.9bv4e-3ag16 | `folder_per_patient --meta ... --id-col ... --label-col ...` |

TCIA downloads use the NBIA Data Retriever. **After building each manifest, read the printed
summary** (patients, classes) and compare with the dataset paper. Class codes used by the
cross-lab task are in `configs/tasks.json`; adjust them if a release names classes differently.

## Pipeline

```bash
# 1. manifests
python -I scripts/build_manifest.py apl_aml --root DeepAPL/Data --out manifests/apl_aml.csv

# 2. features (cached once, reused by every experiment)
python -I scripts/extract_handcrafted.py --manifest manifests/apl_aml.csv --out features/apl_aml_hand.parquet --workers 8
python -I scripts/extract_embeddings.py --manifest manifests/apl_aml.csv --encoder effnet_b0_imagenet --out features/apl_aml_effnet.npy
python -I scripts/extract_embeddings.py --manifest manifests/apl_aml.csv --encoder dinobloom_b \
       --weights weights/DinoBloom-B.pth --out features/apl_aml_dinobloom_b.npy        # weights: zenodo.org/records/10908163

# 3. E1 + E2: leakage audit and representation benchmark (cell level)
python -I scripts/run_cell_benchmark.py --manifest manifests/apl_aml.csv --task apl_cell_diagnosis \
   --features hand=features/apl_aml_hand.parquet effnet=features/apl_aml_effnet.npy dino=features/apl_aml_dinobloom_b.npy \
   --models M1=hand:lgbm M2=effnet:lr M3=dino:lr M4=dino+hand:lr \
   --splits image grouped --seeds 0 1 2 --cv-where cohort=Discovery --test-where cohort=Validation \
   --out results/e1_apl_cell_diagnosis

# 4. E4: patient-level diagnosis with attention MIL
python -I scripts/run_patient_benchmark.py --manifest manifests/apl_aml.csv --task apl_patient \
   --features hand=features/apl_aml_hand.parquet dino=features/apl_aml_dinobloom_b.npy \
   --models P1=hand:pool_lr P2=dino:pool_lr P3=dino+hand:pool_lr P4=dino:abmil P5=dino+hand:abmil \
   --cv-where cohort=Discovery --test-where cohort=Validation --seeds 0 1 2 --out results/e4_apl_patient

# 5. E3: cross-lab (merge sources, then leave one source out; repeat for each held-out source)
python -I scripts/merge_sources.py --manifests manifests/lmu.csv manifests/bmc.csv manifests/cnmc.csv manifests/apl_aml.csv \
   --block hand features/{lmu,bmc,cnmc,apl_aml}_hand.parquet --block dino features/{lmu,bmc,cnmc,apl_aml}_dinobloom_b.npy \
   --out-dir merged/crosslab
python -I scripts/run_cell_benchmark.py --manifest merged/crosslab/manifest.csv --task blast_crosslab \
   --features hand=merged/crosslab/hand.npy dino=merged/crosslab/dino.npy --models M1=hand:lgbm M3=dino:lr M4=dino+hand:lr \
   --splits grouped --cv-where dataset=lmu,bmc,apl_aml --test-where dataset=cnmc --out results/e3_test_cnmc
python -I scripts/run_site_id.py --manifest merged/crosslab/manifest.csv --features hand=merged/crosslab/hand.npy \
   dino=merged/crosslab/dino.npy --out results/e3_site_id.csv

# 6. M5 fine-tuned CNN (GPU)
python -I scripts/run_finetune.py --manifest manifests/cnmc.csv --task cnmc_all_vs_hem --splits image grouped \
   --cv-where cohort=train --test-where cohort=prelim_test --arch efficientnet_b1 --epochs 10 --out results/e1_cnmc_m5

# 7. tables
python -I scripts/make_tables.py results/e1_apl_cell_diagnosis results/e4_apl_patient --out results/tables.md
```

Model specs are `NAME=blocks:classifier`; `blocks` joined with `+` are concatenated (fusion).
Classifiers: `lr`, `lgbm` (cell level); `pool_lr`, `abmil` (patient level).

## What keeps the results honest

1. **Splits** are made once per seed (`StratifiedGroupKFold`, group = patient) and shared by all models.
   `--splits image` reproduces the leaky practice on purpose, for the leakage audit only.
2. **Threshold and calibration temperature** are chosen on an inner validation split of the training
   fold, then frozen. Test sets are touched once.
3. **Imputation and scaling** live inside each scikit-learn Pipeline, fit on training rows only.
4. **Confidence intervals** resample patients (1,000-2,000 bootstraps).
5. **Engineered features** never read image size, file type or dataset name. `run_site_id.py` measures how
   much any representation still reveals the source lab.

## Engineered features (93 per cell)

Nuclear shape (8), nuclear fragments / lobulation, basophilic cell-region shape (8) and its
nucleus-to-region ratio, stain optical density and Lab/HSV colour in nucleus and perinuclear ring,
GLCM chromatin texture (15), nuclear intensity entropy, LBP micro-texture (28), nucleolus and granule proxies.
Limitation: pale cytoplasm cannot be separated from red cells by colour alone, so the "cell" region is the
blue (basophilic) part only and cytoplasm features use a fixed perinuclear ring; an exact N:C ratio would need
a learned segmenter (e.g. Cellpose or SAM).

## Pilot results included (`results_pilot/`)

Run on the real APL-AML cohort (Discovery 82 patients → Validation 24 patients), CPU only,
handcrafted + ImageNet EfficientNet-B0 features. `tables.md` has every number with patient-level
95% CIs; `e4_apl_patient/attention_P4.csv.gz` has per-cell attention for the external patients.
DinoBloom (M3), fine-tuning (M5), the cross-lab experiment (E3) and C-NMC / AML-Hehr still need to
be run with the downloaded datasets and a GPU.
