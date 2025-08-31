#!/usr/bin/env python3
"""
cv_make_folds_from_valid.py
---------------------------------
Build 5-fold (configurable) cross-validation at the VOLUME level
from a slice-level CSV (e.g., valid_slices.csv). Uses the correct
multi_abnormality_labels.csv for volume-level labels.

- Splits volumes (patients) into K folds (no leakage across slices)
- Multi-label aware: uses iterative-stratification if available;
  otherwise falls back to rarity-aware round-robin
- Writes per-fold CSVs:
    train_slices_fold{i}.csv, valid_slices_fold{i}.csv
- Also writes:
    cv_summary.csv  (per-fold counts)
    cv_pairs.json   (index of folds and file paths)
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


ID_CANDIDATES = ["VolumeName","volume_name","volume","case_id","study_id","patient_id","series_id"]


def infer_id_column(dfs: List[pd.DataFrame]) -> str:
    for df in dfs:
        if df is None: 
            continue
        for c in ID_CANDIDATES:
            if c in df.columns:
                return c
    # Final guess: low-uniqueness string column
    for df in dfs:
        if df is None:
            continue
        for c in df.columns:
            if df[c].dtype == object and df[c].nunique() < max(50, len(df)//20):
                return c
    raise ValueError("Could not infer a patient/volume ID column. Make sure your CSV has something like 'VolumeName' or 'volume_name'.")


def binary_label_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            try:
                vals = set(pd.unique(df[c].dropna().astype(int)))
                if vals.issubset({0,1}):
                    cols.append(c)
            except Exception:
                pass
    if not cols:
        raise ValueError("No binary (0/1) label columns found.")
    return cols


def build_volume_labels(id_col: str, slices_df: pd.DataFrame, vol_df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    # Always prefer the multi_abnormality_labels.csv if available
    if vol_df is not None and id_col in vol_df.columns and all(c in vol_df.columns for c in label_cols):
        return vol_df[[id_col]+label_cols].drop_duplicates(subset=[id_col]).reset_index(drop=True)
    # Fallback: OR over slices (label present if any slice has it)
    return slices_df[[id_col]+label_cols].groupby(id_col, as_index=False).max()


def stratified_split_ids(vol_labels: pd.DataFrame, slices_df: pd.DataFrame, label_cols: List[str], k: int, seed: int = 42) -> List[set]:
    """
    Balanced splitting that ensures equal slices across folds.
    Returns: list of sets of volume IDs, one set per fold (validation IDs).
    """
    id_col = vol_labels.columns[0] if vol_labels.columns[0] not in label_cols else "VolumeName"
    if id_col not in vol_labels.columns:
        id_col = [c for c in vol_labels.columns if c not in label_cols][0]
    
    # Get slice counts per volume
    slice_counts = slices_df[id_col].value_counts().to_dict()
    vol_labels = vol_labels.copy()
    vol_labels['slice_count'] = vol_labels[id_col].map(slice_counts)
    
    Y = vol_labels[label_cols].to_numpy()
    
    # Use slice-count aware splitting for balanced folds
    try:
        # Sort volumes by slice count (largest first) for better balancing
        sorted_volumes = vol_labels.sort_values('slice_count', ascending=False)
        
        folds = [set() for _ in range(k)]
        fold_slice_counts = [0] * k
        
        # Assign volumes to folds, balancing total slice count
        for _, row in sorted_volumes.iterrows():
            volume_id = row[id_col]
            volume_slices = row['slice_count']
            
            # Find fold with minimum current slice count
            target_fold = np.argmin(fold_slice_counts)
            folds[target_fold].add(volume_id)
            fold_slice_counts[target_fold] += volume_slices
        
        return folds
        
    except Exception:
        # Fallback to original method if balancing fails
        print("Slice balancing failed, using fallback method")
        rng = np.random.default_rng(seed)
        inv_freq = 1.0 / (Y.sum(axis=0) + 1e-9)
        scores = (Y * inv_freq).sum(axis=1)
        order = np.argsort(-scores)
        folds = [set() for _ in range(k)]
        for i, idx in enumerate(order):
            f = i % k
            folds[f].add(vol_labels.loc[idx, id_col])
        return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True, help="Path to slice-level CSV (e.g., valid_slices.csv)")
    ap.add_argument("--volumes", default=None, help="Path to multi_abnormality_labels.csv (required for proper labels)")
    ap.add_argument("--out", default=".", help="Output directory for per-fold CSVs")
    ap.add_argument("--k", type=int, default=5, help="Number of folds (default 5)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    if not args.volumes:
        raise ValueError("--volumes argument is required. Please provide path to multi_abnormality_labels.csv")
    
    if not Path(args.volumes).exists():
        raise FileNotFoundError(f"Volume labels file not found: {args.volumes}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    slices = pd.read_csv(args.slices)
    vol_df = pd.read_csv(args.volumes)

    # 1) infer ID + label columns
    id_col = infer_id_column([slices, vol_df])
    # Use labels from multi_abnormality_labels.csv
    label_cols = binary_label_columns(vol_df, exclude=[id_col, "split"])

    # 2) build volume labels from multi_abnormality_labels.csv
    vol_labels = build_volume_labels(id_col, slices, vol_df, label_cols)

    # 3) get validation ID sets per fold - with slice balancing
    fold_valid_ids = stratified_split_ids(vol_labels[[id_col]+label_cols], slices, label_cols, args.k, args.seed)

    # Calculate and display slice balance
    total_slices = len(slices)
    slice_counts = []
    for f, valid_ids in enumerate(fold_valid_ids):
        fold_slices = len(slices[slices[id_col].isin(valid_ids)])
        slice_counts.append(fold_slices)
    
    avg_slices = total_slices / args.k
    imbalance = max(slice_counts) - min(slice_counts)
    
    print(f"Total slices: {total_slices}")
    print(f"Target per fold: {avg_slices:.1f} slices")
    print(f"Maximum imbalance: {imbalance} slices")
    
    if imbalance > avg_slices * 0.2:  # More than 20% imbalance
        print("WARNING: Significant slice count imbalance between folds!")
    else:
        print("Good balance achieved across folds")

    # 4) write per-fold CSVs + summary/index
    pairs = []
    summary_rows = []
    all_ids = set(vol_labels[id_col])
    for f, valid_ids in enumerate(fold_valid_ids):
        train_ids = all_ids - valid_ids
        tr = slices[slices[id_col].isin(train_ids)].copy()
        va = slices[slices[id_col].isin(valid_ids)].copy()

        tr_path = out_dir / f"train_slices_fold{f}.csv"
        va_path = out_dir / f"valid_slices_fold{f}.csv"
        tr.to_csv(tr_path, index=False)
        va.to_csv(va_path, index=False)

        # summary per-label counts for the fold
        fold_vols = vol_labels[vol_labels[id_col].isin(valid_ids)].copy()
        pos_counts = fold_vols[label_cols].sum().to_dict()

        pairs.append({
            "fold": f,
            "train_csv": str(tr_path),
            "valid_csv": str(va_path),
            "train_volumes": int(len(train_ids)),
            "valid_volumes": int(len(valid_ids)),
            "train_slices": int(len(tr)),
            "valid_slices": int(len(va)),
        })
        row = {"fold": f, "num_volumes": int(len(valid_ids)), "num_slices": int(len(va))}
        row.update({f"pos_{k}": int(v) for k,v in pos_counts.items()})
        summary_rows.append(row)

    # write index + summary
    with open(out_dir / "cv_pairs.json", "w") as f:
        json.dump({"id_column": id_col, "label_columns": label_cols, "folds": pairs}, f, indent=2)
    pd.DataFrame(summary_rows).to_csv(out_dir / "cv_summary.csv", index=False)

    # stdout
    print(f"ID column: {id_col}")
    print(f"Label columns ({len(label_cols)}): {label_cols}")
    for p in pairs:
        print(f"Fold {p['fold']}: train_slices={p['train_slices']} valid_slices={p['valid_slices']} "
              f"(train_vols={p['train_volumes']}, valid_vols={p['valid_volumes']})")
    print(f"Wrote files to: {out_dir}")

if __name__ == "__main__":
    main()