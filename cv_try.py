#!/usr/bin/env python3
"""
cv_make_folds_from_valid.py
---------------------------------
Build 5-fold (configurable) cross-validation at the VOLUME level
from a slice-level CSV (e.g., valid_slices.csv). Optionally uses a
volume-level CSV (e.g., valid.csv) if provided.

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
    if vol_df is not None and id_col in vol_df.columns and all(c in vol_df.columns for c in label_cols):
        return vol_df[[id_col]+label_cols].drop_duplicates(subset=[id_col]).reset_index(drop=True)
    # OR over slices (label present if any slice has it)
    return slices_df[[id_col]+label_cols].groupby(id_col, as_index=False).max()


def stratified_split_ids(vol_labels: pd.DataFrame, label_cols: List[str], k: int, seed: int = 42) -> List[set]:
    """
    Try iterative-stratification; else fall back to rarity-aware round-robin.
    Returns: list of sets of volume IDs, one set per fold (validation IDs).
    """
    id_col = vol_labels.columns[0] if vol_labels.columns[0] not in label_cols else "VolumeName"
    if id_col not in vol_labels.columns:
        # best-effort pick
        id_col = [c for c in vol_labels.columns if c not in label_cols][0]
    Y = vol_labels[label_cols].to_numpy()

    # Attempt iterative-stratification (pip install iterative-stratification)
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        X = vol_labels[[id_col]].to_numpy()
        folds = [set() for _ in range(k)]
        for f, (_, va_idx) in enumerate(mskf.split(X, Y)):
            ids = vol_labels.loc[va_idx, id_col].tolist()
            folds[f].update(ids)
        # ensure no empty folds; if empty, fall back
        if any(len(s)==0 for s in folds):
            raise RuntimeError("Empty fold from iterative stratification; falling back.")
        return folds
    except Exception:
        pass

    # Rarity-aware round-robin fallback
    rng = np.random.default_rng(seed)
    inv_freq = 1.0 / (Y.sum(axis=0) + 1e-9)
    scores = (Y * inv_freq).sum(axis=1)  # rarer positives ⇒ higher score
    order = np.argsort(-scores)  # rarest-first
    folds = [set() for _ in range(k)]
    for i, idx in enumerate(order):
        f = i % k
        folds[f].add(vol_labels.loc[idx, id_col])
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True, help="Path to slice-level CSV (e.g., valid_slices.csv)")
    ap.add_argument("--volumes", default=None, help="Optional path to volume-level CSV (e.g., valid.csv)")
    ap.add_argument("--out", default=".", help="Output directory for per-fold CSVs")
    ap.add_argument("--k", type=int, default=5, help="Number of folds (default 5)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    slices = pd.read_csv(args.slices)
    vol_df  = pd.read_csv(args.volumes) if args.volumes and Path(args.volumes).exists() else None

    # 1) infer ID + label columns
    id_col = infer_id_column([slices, vol_df])
    # Prefer labels from volumes if present; else from slices
    if vol_df is not None and id_col in vol_df.columns:
        label_cols = binary_label_columns(vol_df, exclude=[id_col, "split"])
    else:
        label_cols = binary_label_columns(slices, exclude=[id_col, "split"])

    # 2) build volume labels (OR over slices if needed)
    vol_labels = build_volume_labels(id_col, slices, vol_df, label_cols)

    # 3) get validation ID sets per fold
    fold_valid_ids = stratified_split_ids(vol_labels[[id_col]+label_cols], label_cols, args.k, args.seed)

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
