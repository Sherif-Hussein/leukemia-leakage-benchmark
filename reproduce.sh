#!/usr/bin/env bash
# Reproduce every result folder used in the manuscript, in the order they were produced.
# Inputs: APL-AML images (DeepAPL clone), C-NMC 2019 and AML-Cytomorphology_MLL_Helmholtz images, EfficientNet-B0
# ImageNet weights (timm efficientnet_b0_ra-3dd342df.pth) and DinoBloom-S weights (Zenodo 10908163).
# Every setting is documented in leukbench/docs/SETTINGS.md. CPU-only; total run time about 1-2 days on 2 vCPUs.
set -euo pipefail
W=${W:-work}                    # working folder with manifests/, features/, results/
WEFF=${WEFF:-weights/efficientnet_b0_ra-3dd342df.pth}
WDINO=${WDINO:-weights/DinoBloom-S.pth}
cd leukbench
PY="python3 -u -I"
mkdir -p $W/manifests $W/features $W/results

# ---------------------------------------------------------------- 1. manifests
# build_manifest.py (see SETTINGS.md, "Manifests") writes one row per image with patient, cohort, diagnosis and cell type
# expected: $W/manifests/apl_aml.csv, cnmc.csv (split_role Development/External), mll.csv (cohort External_MLL)

# ---------------------------------------------------------------- 2. features
$PY scripts/extract_handcrafted.py --manifest $W/manifests/apl_aml.csv --out $W/features/apl_aml_hand.parquet --workers 2
$PY scripts/extract_handcrafted.py --manifest $W/manifests/cnmc.csv --out $W/features/cnmc_hand.parquet --workers 2 --black-bg-to-white
$PY scripts/extract_handcrafted.py --manifest $W/manifests/mll.csv --out $W/features/mll_hand.parquet --workers 2
for D in apl_aml cnmc; do
  $PY scripts/extract_embeddings.py --manifest $W/manifests/$D.csv --encoder effnet_b0_imagenet --weights $WEFF --out $W/features/${D}_effnet.npy
done
for D in apl_aml cnmc mll; do
  $PY scripts/extract_embeddings.py --manifest $W/manifests/$D.csv --encoder dinobloom_s --weights $WDINO --out $W/features/${D}_dinobloom_s.npy --batch-size 32 --chunk 7000
done
# Macenko stain normalization (reference stains from Discovery images only), then CNN features / fine-tuning caches
$PY scripts/normalize_images.py --manifest $W/manifests/apl_aml.csv --mode cell --out-dir $W/data/apl_macenko --out-manifest $W/manifests/apl_aml_macenko.csv
$PY scripts/normalize_images.py --manifest $W/manifests/mll.csv --ref-manifest $W/manifests/apl_aml.csv --mode cell --out-dir $W/data/mll_macenko --out-manifest $W/manifests/mll_macenko.csv
$PY scripts/extract_embeddings.py --manifest $W/manifests/apl_aml_macenko.csv --encoder effnet_b0_imagenet --weights $WEFF --out $W/features/apl_aml_effnet_macenko.npy
$PY scripts/extract_embeddings.py --manifest $W/manifests/mll_macenko.csv --encoder effnet_b0_imagenet --weights $WEFF --out $W/features/mll_effnet_macenko.npy
$PY scripts/extract_handcrafted.py --manifest $W/manifests/apl_aml_macenko.csv --out $W/features/apl_aml_hand_macenko.parquet --workers 1
FT="$PY scripts/run_finetune_cached.py --weights $WEFF --cache"
$FT --manifest $W/manifests/apl_aml.csv          --cache-file $W/features/apl_aml_effb0_block4.npy
$FT --manifest $W/manifests/cnmc.csv             --cache-file $W/features/cnmc_effb0_block4.npy
$FT --manifest $W/manifests/apl_aml_macenko.csv  --cache-file $W/features/apl_aml_effb0_block4_macenko.npy
$FT --manifest $W/manifests/mll_macenko.csv      --cache-file $W/features/mll_effb0_block4_macenko.npy
# combined APL-AML + MLL manifests and features (row-aligned): apl_mll*.csv, apl_mll_*.parquet/.npy
$PY scripts/assemble_mll.py $W

# ---------------------------------------------------------------- 3. leakage audit and decomposition (T1, T2, T4)
for T in apl_cell_diagnosis apl_cell_blast; do
  $PY scripts/run_cell_benchmark.py --manifest $W/manifests/apl_aml.csv --task $T --features hand=$W/features/apl_aml_hand.parquet effnet=$W/features/apl_aml_effnet.npy \
     --models M1=hand:lgbm M1lr=hand:lr M2=effnet:lr M4=hand+effnet:lr --splits image grouped --seeds 0 1 2 --out $W/results/v2_cell_$T
done
$PY scripts/run_cell_benchmark.py --manifest $W/manifests/cnmc.csv --task cnmc_all_vs_hem --features hand=$W/features/cnmc_hand.parquet effnet=$W/features/cnmc_effnet.npy \
   --models M1lr=hand:lr M2=effnet:lr M4=hand+effnet:lr --splits image grouped --seeds 0 1 2 --cv-where split_role=Development --test-where split_role=External --out $W/results/cnmc_cell
$PY scripts/run_gap_bootstrap.py --results $W/results --out $W/results/gap_bootstrap --boot 500

# ---------------------------------------------------------------- 4. patient level, fine-tuning, pre-specified pipelines
$PY scripts/run_patient_benchmark.py --manifest $W/manifests/apl_aml.csv --task apl_patient --features hand=$W/features/apl_aml_hand.parquet effnet=$W/features/apl_aml_effnet.npy \
   --models P1=hand:pool_lr P2=effnet:pool_lr P3=hand+effnet:pool_lr P4=hand:abmil P5=effnet:abmil P6=hand+effnet:abmil --cv-where cohort=Discovery --test-where cohort=Validation --seeds 0 1 2 --out $W/results/v2_patient
$PY scripts/run_patient_benchmark.py --manifest $W/manifests/cnmc.csv --task cnmc_patient --features hand=$W/features/cnmc_hand.parquet effnet=$W/features/cnmc_effnet.npy \
   --models P1=hand:pool_lr P2=effnet:pool_lr P3=hand+effnet:pool_lr P4=hand:abmil P5=effnet:abmil P6=hand+effnet:abmil --cv-where split_role=Development --test-where split_role=External --seeds 0 1 2 --out $W/results/cnmc_patient
$PY scripts/run_finetune_cached.py --manifest $W/manifests/apl_aml_macenko.csv --weights $WEFF --cache-file $W/features/apl_aml_effb0_block4_macenko.npy --task apl_cell_diagnosis --splits grouped --epochs 6 --max-train 5000 --out $W/results/finetune_t1_macenko
$PY scripts/run_finetune_cached.py --manifest $W/manifests/cnmc.csv --weights $WEFF --cache-file $W/features/cnmc_effb0_block4.npy --task cnmc_all_vs_hem --splits grouped --epochs 6 --max-train 5000 --cv-where split_role=Development --test-where split_role=External --out $W/results/finetune_cnmc
$PY scripts/run_ensemble.py --patient $W/results/v2_patient --finetune $W/results/finetune_t1_macenko --out $W/results/ensemble_apl_macenko
$PY scripts/run_ensemble.py --patient $W/results/cnmc_patient --finetune $W/results/finetune_cnmc --out $W/results/ensemble_cnmc
$PY scripts/run_feature_engineering.py --manifest $W/manifests/apl_aml_macenko.csv --hand $W/features/apl_aml_hand.parquet --effnet $W/features/apl_aml_effnet_macenko.npy --configs eff_raw_nw eff_pca64_nw --out $W/results/apl_fe_macenko

# ---------------------------------------------------------------- 5. DinoBloom (C-NMC and APL-AML)
$PY scripts/run_cell_benchmark.py --manifest $W/manifests/cnmc.csv --task cnmc_all_vs_hem --features hand=$W/features/cnmc_hand.parquet dino=$W/features/cnmc_dinobloom_s.npy \
   --models MD=dino:lr MDF=hand+dino:lr --splits image grouped --seeds 0 1 2 --cv-where split_role=Development --test-where split_role=External --out $W/results/cnmc_cell_dino
$PY scripts/run_patient_benchmark.py --manifest $W/manifests/cnmc.csv --task cnmc_patient --features hand=$W/features/cnmc_hand.parquet dino=$W/features/cnmc_dinobloom_s.npy \
   --models PD=dino:pool_lr PDF=hand+dino:pool_lr PDM=dino:abmil --cv-where split_role=Development --test-where split_role=External --seeds 0 1 2 --out $W/results/cnmc_patient_dino
$PY scripts/run_patient_benchmark.py --manifest $W/manifests/apl_aml.csv --task apl_patient --features hand=$W/features/apl_aml_hand.parquet dino=$W/features/apl_aml_dinobloom_s.npy \
   --models PD=dino:pool_lr PDF=hand+dino:pool_lr PDM=dino:abmil --cv-where cohort=Discovery --test-where cohort=Validation --seeds 0 1 2 --out $W/results/apl_patient_dino

# ---------------------------------------------------------------- 6. independent MLL test (prespecified, commit acf173b)
$PY scripts/run_patient_benchmark.py --manifest $W/manifests/apl_mll.csv --task apl_patient_mll --features hand=$W/features/apl_mll_hand.parquet dino=$W/features/apl_mll_dinobloom_s.npy \
   --models P1=hand:pool_lr P4=hand:abmil PDF=hand+dino:pool_lr PD=dino:pool_lr --cv-where cohort=Discovery --test-where cohort=External_MLL --seeds 0 1 2 --out $W/results/mll_patient
$PY scripts/run_finetune_cached.py --manifest $W/manifests/apl_mll_macenko.csv --weights $WEFF --cache-file $W/features/apl_mll_effb0_block4_macenko.npy --task apl_cell_diagnosis_mll --splits grouped --epochs 6 --max-train 5000 --cv-where cohort=Discovery --test-where cohort=External_MLL --out $W/results/mll_finetune_macenko
$PY scripts/run_ensemble.py --patient $W/results/mll_patient --finetune $W/results/mll_finetune_macenko --out $W/results/mll_ensemble
$PY scripts/run_feature_engineering.py --manifest $W/manifests/apl_mll_macenko.csv --hand $W/features/apl_mll_hand.parquet --effnet $W/features/apl_mll_effnet_macenko.npy --configs eff_raw_nw eff_pca64_nw --ext External_MLL --out $W/results/mll_fe_macenko

# ---------------------------------------------------------------- 7. post hoc: workload, threshold adaptation, reverse transfer
$PY scripts/run_local_recal_ext.py --results $W/results --out $W/results/mll_recal
$PY scripts/run_utility.py --results $W/results --out $W/results/utility
$PY scripts/run_patient_benchmark.py --manifest $W/manifests/apl_mll.csv --task apl_patient_mll --features hand=$W/features/apl_mll_hand.parquet dino=$W/features/apl_mll_dinobloom_s.npy \
   --models P1=hand:pool_lr P4=hand:abmil PD=dino:pool_lr PDF=hand+dino:pool_lr --cv-where cohort=External_MLL --test-where dataset=apl_aml --seeds 0 1 2 --max-cells 200 --out $W/results/reciprocal_mll_to_apl

# ---------------------------------------------------------------- 8. exploratory analyses (scan time, shift test, calibration, PCA, early stopping, ...)
#   see leukbench/docs/SETTINGS.md for the remaining commands (run_date_confound.py, run_revision_stats.py, run_pooled_cv.py,
#   run_local_recalibration.py, run_preproc_test.py, run_cost_benchmark.py, run_shap.py, run_eda.py)

# ---------------------------------------------------------------- 9. release files, tables, figures, PDFs
$PY scripts/make_release.py $W release
cd ../manuscript && LEUKBENCH_WORK=../leukbench/$W python3 make_results.py
pdflatex supplement && pdflatex main && bibtex supplement && bibtex main && pdflatex supplement && pdflatex main && pdflatex main
