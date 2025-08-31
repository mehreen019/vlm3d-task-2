import pandas as pd

# Load the files
slices = pd.read_csv("ct_rate_2d/splits/valid_slices.csv")
labels = pd.read_csv("ct_rate_2d/multi_abnormality_labels.csv")

print("Slice file shape:", slices.shape)
print("Labels file shape:", labels.shape)
print("Slice columns:", slices.columns.tolist())
print("Label columns:", labels.columns.tolist())
print("Slice volume names sample:", slices['volume_name'].head(3).tolist())
print("Label volume names sample:", labels['VolumeName'].head(3).tolist())

# Check if files exist and are readable
import os
print("\nFile existence check:")
print("Slices file exists:", os.path.exists("ct_rate_2d/splits/valid_slices.csv"))
print("Labels file exists:", os.path.exists("ct_rate_2d/multi_abnormality_labels.csv"))