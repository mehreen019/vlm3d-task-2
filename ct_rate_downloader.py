#!/usr/bin/env python3
"""
CT-RATE Dataset Downloader with Stratified Sampling
Designed to work within 80GB storage constraint while maintaining data diversity
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import argparse
from sklearn.model_selection import train_test_split
from collections import Counter
import logging
from typing import Dict, List, Tuple
from huggingface_hub import hf_hub_download, snapshot_download
import time
import shutil
import gzip
import lz4.frame
import nibabel as nib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CTRateDownloader:
    def __init__(self, data_dir: str = "./ct_rate_data", max_storage_gb: float = 80, use_compression: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_storage_bytes = max_storage_gb * 1024**3
        self.use_compression = use_compression
        
        # With compression, we can fit ~3-5x more data
        compression_factor = 4.0 if use_compression else 1.0
        self.avg_ct_size_bytes = (300 * 1024**2) / compression_factor  # Compressed size
        self.max_samples = int(self.max_storage_bytes * 0.9 / self.avg_ct_size_bytes)  # 90% of storage
        
        self.dataset_repo = "ibrahimhamamci/CT-RATE"
        self.base_url = "https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main"
        self.output_dir = self.data_dir / "ct_rate_volumes"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compression statistics
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'compression_ratio': 0
        }

        # 18 abnormalities
        self.abnormality_classes = [
            "Cardiomegaly", "Hiatal hernia", "Atelectasis", "Pulmonary fibrotic sequela",
            "Peribronchial thickening", "Interlobular septal thickening", "Medical material",
            "Pericardial effusion", "Lymphadenopathy", "Lung nodule", "Pleural effusion",
            "Consolidation", "Lung opacity", "Mosaic attenuation pattern", "Bronchiectasis",
            "Emphysema", "Arterial wall calcification", "Coronary artery wall calcification"
        ]

        
        logger.info(f"Initialized downloader for max {self.max_samples} samples (~{max_storage_gb}GB)")
        logger.info(f"Compression enabled: {use_compression}")
    
    def compress_nifti_volume(self, input_path: str, output_path: str) -> Tuple[int, int]:
        """Compress NIfTI volume using LZ4 for optimal speed/compression balance"""
        try:
            # Load the NIfTI file
            img = nib.load(input_path)
            data = img.get_fdata().astype(np.float32)
            
            # Get original size
            original_size = os.path.getsize(input_path)
            
            # Compress the data array using LZ4
            compressed_data = lz4.frame.compress(data.tobytes())
            
            # Save compressed data with metadata
            compressed_info = {
                'data': compressed_data,
                'shape': data.shape,
                'header': img.header,
                'affine': img.affine.tolist(),
                'dtype': str(data.dtype)
            }
            
            # Save to compressed format
            with open(output_path + '.lz4', 'wb') as f:
                import pickle
                pickle.dump(compressed_info, f)
            
            compressed_size = os.path.getsize(output_path + '.lz4')
            
            # Update statistics
            self.compression_stats['original_size'] += original_size
            self.compression_stats['compressed_size'] += compressed_size
            
            logger.debug(f"Compressed {input_path}: {original_size//1024//1024}MB -> {compressed_size//1024//1024}MB "
                        f"({compressed_size/original_size:.2f}x)")
            
            return original_size, compressed_size
            
        except Exception as e:
            logger.error(f"Compression failed for {input_path}: {e}")
            # Fallback to original file
            shutil.copy2(input_path, output_path)
            return 0, os.path.getsize(output_path)
    
    def decompress_nifti_volume(self, compressed_path: str) -> np.ndarray:
        """Decompress LZ4 compressed NIfTI volume"""
        try:
            with open(compressed_path, 'rb') as f:
                import pickle
                compressed_info = pickle.load(f)
            
            # Decompress data
            decompressed_bytes = lz4.frame.decompress(compressed_info['data'])
            data = np.frombuffer(decompressed_bytes, dtype=compressed_info['dtype'])
            data = data.reshape(compressed_info['shape'])
            
            return data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Decompression failed for {compressed_path}: {e}")
            return None

    def download_metadata(self) -> pd.DataFrame:
        """Download and merge train/val metadata"""
        merged_metadata_file = self.data_dir / "metadata.csv"

        if merged_metadata_file.exists():
            logger.info(f"Loading cached metadata from {merged_metadata_file}")
            return pd.read_csv(merged_metadata_file)

        split_files = {
            "train": "dataset/metadata/train_metadata.csv",
            "valid": "dataset/metadata/validation_metadata.csv",
        }

        dfs = []
        for split_name, remote_path in split_files.items():
            local_path = hf_hub_download(
                repo_id=self.dataset_repo,
                filename=remote_path,
                repo_type="dataset",
                cache_dir=str(self.data_dir)
            )
            df_split = pd.read_csv(local_path)
            df_split["split"] = split_name
            dfs.append(df_split)
            logger.info(f"Downloaded {remote_path} with {len(df_split)} samples")

        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(merged_metadata_file, index=False)
        logger.info(f"Merged metadata saved to {merged_metadata_file} with {len(df)} samples")
        return df
    
    def download_labels(self) -> pd.DataFrame:
        """Download multi-abnormality labels (train+val)"""
        merged_labels_file = self.data_dir / "multi_abnormality_labels.csv"

        if merged_labels_file.exists():
            logger.info(f"Loading cached labels from {merged_labels_file}")
            return pd.read_csv(merged_labels_file)

        split_files = {
            "train": "dataset/multi_abnormality_labels/train_predicted_labels.csv",
            "valid": "dataset/multi_abnormality_labels/valid_predicted_labels.csv",
        }

        dfs = []
        for split_name, remote_path in split_files.items():
            local_path = hf_hub_download(
                repo_id=self.dataset_repo,
                filename=remote_path,
                repo_type="dataset",
                cache_dir=str(self.data_dir)
            )
            df_split = pd.read_csv(local_path)
            df_split["split"] = split_name
            dfs.append(df_split)
            logger.info(f"Downloaded {remote_path} with {len(df_split)} rows")

        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(merged_labels_file, index=False)
        logger.info(f"Merged labels saved to {merged_labels_file} with {len(df)} rows")
        return df

    
    def analyze_data_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze the distribution of abnormalities in the dataset"""
        logger.info("Analyzing data distribution...")
        logger.info("Prining df.columns")
        logger.info(df.columns)
        
        # Calculate prevalence for each abnormality
        prevalences = {}
        for class_name in self.abnormality_classes:
            if class_name in df.columns:
                prevalences[class_name] = df[class_name].sum() / len(df)
        
        # Calculate co-occurrence patterns
        label_cols = [col for col in self.abnormality_classes if col in df.columns]
        label_matrix = df[label_cols].values
        
        # Count number of abnormalities per sample
        abnormality_counts = np.sum(label_matrix, axis=1)
        
        stats = {
            'total_samples': len(df),
            'prevalences': prevalences,
            'abnormality_count_distribution': Counter(abnormality_counts),
            'avg_abnormalities_per_sample': np.mean(abnormality_counts)
        }
        
        logger.info(f"Dataset stats: {stats['total_samples']} samples, "
                   f"avg {stats['avg_abnormalities_per_sample']:.2f} abnormalities per sample")
        
        return stats
    
    def stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform stratified sampling to maintain data distribution"""
        logger.info(f"Performing stratified sampling to select {self.max_samples} samples...")
        
        if len(df) <= self.max_samples:
            logger.info("Dataset smaller than storage limit, using all samples")
            return df
        
        # Create stratification key based on abnormality patterns
        label_cols = [col for col in self.abnormality_classes if col in df.columns]
        
        # Strategy 1: Stratify by number of abnormalities and most common abnormalities
        df['n_abnormalities'] = df[label_cols].sum(axis=1)
        
        # For stratification, focus on most prevalent abnormalities
        prevalences = {col: df[col].sum() / len(df) for col in label_cols}
        top_5_abnormalities = sorted(prevalences.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Create stratification groups
        stratify_cols = ['n_abnormalities'] + [col for col, _ in top_5_abnormalities]
        
        # Create stratification key
        df['strat_key'] = df[stratify_cols].apply(
            lambda row: '_'.join([f"{col}:{int(row[col])}" for col in stratify_cols]), axis=1
        )
        
        # Sample from each stratum proportionally
        sampled_dfs = []
        strat_counts = df['strat_key'].value_counts()
        
        for strat_key, count in strat_counts.items():
            strat_df = df[df['strat_key'] == strat_key]
            
            # Calculate how many samples to take from this stratum
            proportion = count / len(df)
            n_samples = max(1, int(proportion * self.max_samples))
            n_samples = min(n_samples, len(strat_df))
            
            sampled_strat = strat_df.sample(n=n_samples, random_state=42)
            sampled_dfs.append(sampled_strat)
        
        sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # If we're still over the limit, randomly sample
        if len(sampled_df) > self.max_samples:
            sampled_df = sampled_df.sample(n=self.max_samples, random_state=42)
        
        # Clean up temporary columns
        sampled_df = sampled_df.drop(['n_abnormalities', 'strat_key'], axis=1)
        
        logger.info(f"Selected {len(sampled_df)} samples via stratified sampling")
        
        return sampled_df
    
    def create_data_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/val/test splits while maintaining distribution"""
        logger.info("Creating train/val/test splits...")
        
        # Use the same stratification approach for splitting
        label_cols = [col for col in self.abnormality_classes if col in df.columns]
        df['n_abnormalities'] = df[label_cols].sum(axis=1)
        
        # First split: train+val vs test (80/20)
        try:
            train_val, test = train_test_split(
                df, test_size=0.2, random_state=42, 
                stratify=df['n_abnormalities']
            )
            
            # Second split: train vs val (75/25 of train_val = 60/20 overall)
            train, val = train_test_split(
                train_val, test_size=0.25, random_state=42,
                stratify=train_val['n_abnormalities']
            )
        except ValueError:
            # Fallback if stratification fails due to small sample sizes
            logger.warning("Stratified split failed, using random split")
            train_val, test = train_test_split(df, test_size=0.2, random_state=42)
            train, val = train_test_split(train_val, test_size=0.25, random_state=42)
        
        # Clean up temporary column
        train = train.drop('n_abnormalities', axis=1)
        val = val.drop('n_abnormalities', axis=1)
        test = test.drop('n_abnormalities', axis=1)
        
        logger.info(f"Created splits: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        return train, val, test
    
    def save_splits_and_metadata(self, train, val, test, stats):
        splits_dir = self.data_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        train.to_csv(splits_dir / "train.csv", index=False)
        val.to_csv(splits_dir / "valid.csv", index=False)
        test.to_csv(splits_dir / "test.csv", index=False)

        with open(splits_dir / "dataset_stats.json", 'w') as f:
            stats_serializable = {}
            for key, value in stats.items():
                if isinstance(value, dict):
                    # ensure both keys & values are JSON-safe
                    stats_serializable[key] = {
                        str(k): (
                            float(v) if isinstance(v, np.floating) 
                            else int(v) if isinstance(v, np.integer) 
                            else v
                        )
                        for k, v in value.items()
                    }
                else:
                    if isinstance(value, np.floating):
                        stats_serializable[key] = float(value)
                    elif isinstance(value, np.integer):
                        stats_serializable[key] = int(value)
                    else:
                        stats_serializable[key] = value

            json.dump(stats_serializable, f, indent=2)

        logger.info(f"Saved splits and metadata to {splits_dir}")

    


    def download_ct_volumes(self, df: pd.DataFrame, split: str):
        """
        Download CT volumes from HuggingFace dataset using hf_hub_download,
        reconstructing full nested paths from the CSV metadata.
        """
        for _, row in df.iterrows():
            volume_name = row["VolumeName"]
            split = row["split"]  # train or valid

            prefix = "_".join(volume_name.split("_")[:3])
            case_folder = "_".join(volume_name.split("_")[:2])
            repo_file_path = f"dataset/{split}/{case_folder}/{prefix}/{volume_name}"

            try:
                local_path = hf_hub_download(
                    repo_id="ibrahimhamamci/CT-RATE",
                    filename=repo_file_path,
                    repo_type="dataset"
                )

                # Ensure output directory exists
                final_path = os.path.join(self.output_dir, split, volume_name)
                os.makedirs(os.path.dirname(final_path), exist_ok=True)

                # Apply compression if enabled
                if self.use_compression:
                    # Compress the file
                    base_path = os.path.splitext(final_path)[0]  # Remove .gz extension
                    original_size, compressed_size = self.compress_nifti_volume(local_path, base_path)
                    print(f"✅ Downloaded & compressed: {base_path}.lz4 "
                          f"({compressed_size//1024//1024}MB, {compressed_size/original_size:.2f}x compression)")
                else:
                    # Copy file from cache to output_dir
                    shutil.copy2(local_path, final_path)
                    print(f"✅ Downloaded: {final_path}")

            except Exception as e:
                print(f"❌ Failed to download {repo_file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download CT-RATE dataset with stratified sampling and compression")
    parser.add_argument("--data-dir", default="./ct_rate_data", help="Data directory")
    parser.add_argument("--max-storage-gb", type=float, default=5, help="Maximum storage in GB")
    parser.add_argument("--download-volumes", action="store_true", help="Download actual CT volumes")
    parser.add_argument("--use-compression", action="store_true", default=True, help="Use lossless compression to save space")
    parser.add_argument("--no-compression", action="store_true", help="Disable compression")
    
    args = parser.parse_args()
    
    # Handle compression flags
    use_compression = args.use_compression and not args.no_compression
    
    # Initialize downloader
    downloader = CTRateDownloader(args.data_dir, args.max_storage_gb, use_compression)
    
    # Step 1: Download metadata
    meta_df = downloader.download_metadata()

    # Step 2: Download labels
    labels_df = downloader.download_labels()

    # Step 3: Merge them on VolumeName
    df = pd.merge(meta_df, labels_df, on=["VolumeName", "split"], how="inner")

    # Step 4: Analyze data distribution
    stats = downloader.analyze_data_distribution(df)

    # Step 5: Perform stratified sampling
    sampled_df = downloader.stratified_sample(df)

    
    # Step 5: Create data splits
    train, val, test = downloader.create_data_splits(sampled_df)
    
    # Step 6: Save splits and metadata
    downloader.save_splits_and_metadata(train, val, test, stats)
    
    # Step 7: Download actual volumes (if requested)
    if args.download_volumes:
        downloader.download_ct_volumes(train, "train")
        downloader.download_ct_volumes(val, "valid")
    
    # Print compression statistics
    if use_compression and args.download_volumes:
        original_gb = downloader.compression_stats['original_size'] / (1024**3)
        compressed_gb = downloader.compression_stats['compressed_size'] / (1024**3)
        if original_gb > 0:
            compression_ratio = original_gb / compressed_gb
            saved_gb = original_gb - compressed_gb
            logger.info("🗜️ Compression Statistics:")
            logger.info(f"   Original size: {original_gb:.2f} GB")
            logger.info(f"   Compressed size: {compressed_gb:.2f} GB")
            logger.info(f"   Compression ratio: {compression_ratio:.2f}x")
            logger.info(f"   Space saved: {saved_gb:.2f} GB ({(saved_gb/original_gb)*100:.1f}%)")
    
    logger.info("✅ Dataset preparation complete!")
    logger.info(f"📂 Data saved to: {args.data_dir}")
    logger.info(f"🚂 Train: {len(train)} samples")
    logger.info(f"✅ Val: {len(val)} samples") 
    logger.info(f"🧪 Test: {len(test)} samples")

if __name__ == "__main__":
    main()