"""
Create Train/Val/Test Split
-------------------------------
Purpose: Assigns each row in dataset_index.csv (your final extracted
keypoint dataset) to train/val/test, stratified by class so each split
keeps the same safe/unsafe balance as the full dataset.

Known limitation: rider, session and take identity were not recorded, so
this is a PURE RANDOM split at the image level. Near-duplicate burst frames
and images of the same rider cannot be grouped together, so a small risk of
train/test leakage (identity or near-duplicate pose leakage) exists. This is
stated in the paper's limitations section.

Usage:
    python create_split.py
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_INDEX_CSV = "dataset_index.csv"
OUTPUT_CSV = "dataset_split.csv"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15   # must sum to 1.0 with the above

RANDOM_SEED = 42   # fixed for reproducibility — always produces the same split


def main():
    df = pd.read_csv(DATASET_INDEX_CSV)
    print(f"Loaded {len(df)} rows from {DATASET_INDEX_CSV}")
    print(df["class"].value_counts())

    rng = np.random.RandomState(RANDOM_SEED)
    splits = []

    # Stratify by class: shuffle each class's rows independently, then slice
    # into train/val/test proportionally, so both classes keep the same
    # ratio in every split as they have overall.
    for class_label, group in df.groupby("class"):
        n = len(group)
        shuffled_idx = rng.permutation(n)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        # whatever's left goes to test, avoids rounding gaps/overlaps
        n_test = n - n_train - n_val

        assignment = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
        assignment = assignment[np.argsort(shuffled_idx)]  # map back to original row order within group

        group = group.copy()
        group["split"] = assignment
        splits.append(group)

        print(f"  {class_label}: train={n_train}, val={n_val}, test={n_test} (of {n})")

    result = pd.concat(splits, ignore_index=True)
    result.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved split assignments to {OUTPUT_CSV}")
    print("\nOverall split totals:")
    print(result["split"].value_counts())
    print("\nPer-split class balance:")
    print(result.groupby(["split", "class"]).size().unstack())


if __name__ == "__main__":
    main()
