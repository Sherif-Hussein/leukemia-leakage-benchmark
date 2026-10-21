#!/bin/bash
# Second dataset: C-NMC 2019 (ALL vs normal B cells; 72 subjects; different lab, stain and imaging).
# Development = official folds 0+1 (42 subjects); external = fold 2 (31 subjects), pre-specified.
# Every cell carries its subject's diagnosis, i.e. a patient-level label (like T1 in APL-AML).
set -e
W=${W:?set W to the work directory}
cd "$(dirname "$0")/.."
F="hand=$W/features/cnmc_hand.parquet effnet=$W/features/cnmc_effnet.npy"
COMMON="--manifest $W/manifests/cnmc.csv --cv-where split_role=Development --test-where split_role=External --seeds 0 1 2"

# 1. leakage audit at cell level (image-level vs subject-grouped CV vs external fold)
python3 -I scripts/run_cell_benchmark.py $COMMON --task cnmc_all_vs_hem --features $F \
  --models M1=hand:lgbm M1lr=hand:lr M2=effnet:lr M4=hand+effnet:lr --splits image grouped --out $W/results/cnmc_cell

# 2. subject-level diagnosis: pooled LR and attention MIL (same models as P1-P6)
python3 -I scripts/run_patient_benchmark.py $COMMON --task cnmc_patient --features $F \
  --models P1=hand:pool_lr P2=effnet:pool_lr P3=hand+effnet:pool_lr P4=hand:abmil P5=effnet:abmil P6=hand+effnet:abmil \
  --out $W/results/cnmc_patient

# 3. feature engineering incl. the full PCA sweep at subject level
python3 -I scripts/run_feature_engineering.py --manifest $W/manifests/cnmc.csv --hand $W/features/cnmc_hand.parquet \
  --effnet $W/features/cnmc_effnet.npy --positive ALL --cohort-col split_role --dev Development --ext External --seeds 0 1 2 \
  --configs eng_raw eng_corrfilter eff_raw eff_pca4 eff_pca8 eff_pca16 eff_pca32 eff_pca64 eff_pca128 eff_pca256 fusion_pca32 \
  --out $W/results/cnmc_fe

# 4. PCA diagnostics (cohort identifiability on normal cells, cell-level signal)
python3 -I scripts/run_pca_analysis.py --manifest $W/manifests/cnmc.csv --effnet $W/features/cnmc_effnet.npy \
  --positive ALL --negative HEM --cohort-col split_role --dev Development --ext External --out $W/results/cnmc_pca_analysis
echo "C-NMC done"
