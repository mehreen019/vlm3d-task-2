import pandas as pd
import sys

# Load both files
train = pd.read_csv("ct_rate_2d/splits/train_slices.csv")
valid = pd.read_csv("ct_rate_2d/splits/valid_slices.csv")

# Combine them
combined = pd.concat([train, valid], ignore_index=True)
combined.to_csv("ct_rate_2d/splits/combined_slices.csv", index=False)

print(f"Combined {len(train)} + {len(valid)} = {len(combined)} slices")
