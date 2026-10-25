"""Combine APL-AML (Discovery used for training) with the MLL external cohort, row-aligned features."""
import numpy as np, pandas as pd, sys
W = sys.argv[1]; M, F = f"{W}/manifests", f"{W}/features"
a, b = pd.read_csv(f"{M}/apl_aml.csv", keep_default_na=False), pd.read_csv(f"{M}/mll.csv", keep_default_na=False)
am, bm = pd.read_csv(f"{M}/apl_aml_macenko.csv", keep_default_na=False), pd.read_csv(f"{M}/mll_macenko.csv", keep_default_na=False)
cols = ["path", "dataset", "patient_id", "cohort", "diagnosis", "cell_type"]
pd.concat([a[cols], b[cols]], ignore_index=True).to_csv(f"{M}/apl_mll.csv", index=False)
pd.concat([am[cols], bm[cols]], ignore_index=True).to_csv(f"{M}/apl_mll_macenko.csv", index=False)
pd.concat([pd.read_parquet(f"{F}/apl_aml_hand.parquet"), pd.read_parquet(f"{F}/mll_hand.parquet")], ignore_index=True).to_parquet(f"{F}/apl_mll_hand.parquet")
for x, y, o in [("apl_aml_effnet_macenko.npy", "mll_effnet_macenko.npy", "apl_mll_effnet_macenko.npy"),
                ("apl_aml_dinobloom_s.npy", "mll_dinobloom_s.npy", "apl_mll_dinobloom_s.npy")]:
    np.save(f"{F}/{o}", np.concatenate([np.load(f"{F}/{x}"), np.load(f"{F}/{y}")]))
A = np.load(f"{F}/apl_aml_effb0_block4_macenko.npy", mmap_mode="r"); B = np.load(f"{F}/mll_effb0_block4_macenko.npy", mmap_mode="r")
C = np.lib.format.open_memmap(f"{F}/apl_mll_effb0_block4_macenko.npy", mode="w+", dtype=np.float16, shape=(len(A) + len(B),) + A.shape[1:])
C[:len(A)] = A; C[len(A):] = B; C.flush()
np.save(f"{F}/apl_mll_effb0_block4_macenko_ok.npy", np.concatenate([np.load(f"{F}/apl_aml_effb0_block4_macenko_ok.npy"), np.load(f"{F}/mll_effb0_block4_macenko_ok.npy")]))
print("assembled", len(a) + len(b), "rows")
