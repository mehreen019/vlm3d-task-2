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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CTRateDownloader:
    def __init__(self, data_dir: str = "./ct_rate_data", max_storage_gb: float = 80):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_storage_bytes = max_storage_gb * 1024**3
        self.avg_ct_size_bytes = 300 * 1024**2  # Estimated 300MB per CT volume
        self.max_samples = int(self.max_storage_bytes * 0.9 / self.avg_ct_size_bytes)  # 90% of storage
        
        # CT-RATE 18 abnormality classes
        self.abnormality_classes = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Fracture', 'Hernia', 'Infiltration',
            'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
            'Support_Devices', 'Thickening', 'No_Finding'
        ]
        
        logger.info(f"Initialized downloader for max {self.max_samples} samples (~{max_storage_gb}GB)")
    
    def download_metadata(self) -> pd.DataFrame:
        """Download and parse CT-RATE metadata/annotations"""
        metadata_url = "https://physionet.org/files/ct-rate/1.0.0/"  # Adjust URL as needed
        
        logger.info("Downloading metadata...")
        
        # First try to download the metadata files
        # Note: You'll need to register on PhysioNet and get proper URLs
        # This is a template - adjust URLs based on actual CT-RATE distribution
        
        metadata_file = self.data_dir / "metadata.csv"
        
        if not metadata_file.exists():
            # Create dummy metadata for testing - replace with actual download
            logger.warning("Creating dummy metadata for testing. Replace with actual CT-RATE metadata download.")
            
            # Generate realistic dummy data for testing
            np.random.seed(42)
            n_samples = 10000  # Total samples in CT-RATE (adjust as needed)
            
            data = []
            for i in range(n_samples):
                # Generate realistic abnormality patterns
                sample = {
                    'study_id': f'CT_{i:06d}',
                    'patient_id': f'P_{i:04d}',
                    'file_path': f'volumes/CT_{i:06d}.nii.gz',
                    'file_size_mb': np.random.normal(300, 100),  # ~300MB average
                }
                
                # Generate correlated abnormality labels (some are more likely to co-occur)
                labels = np.zeros(18, dtype=int)
                
                # No Finding is exclusive
                if np.random.random() < 0.3:
                    labels[-1] = 1  # No_Finding
                else:
                    # Sample 1-4 abnormalities with realistic prevalences
                    prevalences = [0.1, 0.15, 0.08, 0.12, 0.2, 0.05, 0.03, 0.02, 0.01, 
                                 0.07, 0.06, 0.25, 0.04, 0.09, 0.05, 0.18, 0.04, 0.0]
                    
                    for j, prevalence in enumerate(prevalences[:-1]):  # Exclude No_Finding
                        if np.random.random() < prevalence:
                            labels[j] = 1
                
                # Add labels to sample
                for j, class_name in enumerate(self.abnormality_classes):
                    sample[class_name] = labels[j]
                
                data.append(sample)
            
            df = pd.DataFrame(data)
            df.to_csv(metadata_file, index=False)
            logger.info(f"Created dummy metadata with {len(df)} samples")
        else:
            df = pd.read_csv(metadata_file)
            logger.info(f"Loaded metadata with {len(df)} samples")
        
        return df
    
    def analyze_data_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze the distribution of abnormalities in the dataset"""
        logger.info("Analyzing data distribution...")
        
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
    
    def save_splits_and_metadata(self, train: pd.DataFrame, val: pd.DataFrame, 
                                test: pd.DataFrame, stats: Dict):
        """Save the data splits and metadata"""
        splits_dir = self.data_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        # Save splits
        train.to_csv(splits_dir / "train.csv", index=False)
        val.to_csv(splits_dir / "val.csv", index=False)
        test.to_csv(splits_dir / "test.csv", index=False)
        
        # Save metadata and statistics
        with open(splits_dir / "dataset_stats.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            stats_serializable = {}
            for key, value in stats.items():
                if isinstance(value, dict):
                    stats_serializable[key] = {k: float(v) if isinstance(v, np.floating) 
                                             else int(v) if isinstance(v, np.integer) 
                                             else v for k, v in value.items()}
                else:
                    stats_serializable[key] = value
            
            json.dump(stats_serializable, f, indent=2)
        
        logger.info(f"Saved splits and metadata to {splits_dir}")
    
    def download_ct_volumes(self, df: pd.DataFrame, split_name: str = "train"):
        """Download the actual CT volumes (placeholder - implement actual download logic)"""
        logger.info(f"Starting download of {len(df)} CT volumes for {split_name} split...")
        
        volumes_dir = self.data_dir / "volumes" / split_name
        volumes_dir.mkdir(parents=True, exist_ok=True)
        
        # This is where you'd implement actual download logic
        # For now, we'll create placeholder files
        
        logger.warning("This is a placeholder download function. "
                      "Implement actual CT-RATE download logic here.")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Downloading {split_name}"):
            study_id = row['study_id']
            
            # Placeholder: create empty files to simulate download
            # Replace this with actual download logic:
            # 1. Get download URL from CT-RATE
            # 2. Download the .nii.gz file
            # 3. Verify file integrity
            
            placeholder_file = volumes_dir / f"{study_id}.nii.gz"
            if not placeholder_file.exists():
                # Create small placeholder file
                placeholder_file.touch()
            
        logger.info(f"Completed download simulation for {split_name}")

def main():
    parser = argparse.ArgumentParser(description="Download CT-RATE dataset with stratified sampling")
    parser.add_argument("--data-dir", default="./ct_rate_data", help="Data directory")
    parser.add_argument("--max-storage-gb", type=float, default=80, help="Maximum storage in GB")
    parser.add_argument("--download-volumes", action="store_true", help="Download actual CT volumes")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = CTRateDownloader(args.data_dir, args.max_storage_gb)
    
    # Step 1: Download metadata
    df = downloader.download_metadata()
    
    # Step 2: Analyze data distribution
    stats = downloader.analyze_data_distribution(df)
    
    # Step 3: Perform stratified sampling
    sampled_df = downloader.stratified_sample(df)
    
    # Step 4: Create data splits
    train, val, test = downloader.create_data_splits(sampled_df)
    
    # Step 5: Save splits and metadata
    downloader.save_splits_and_metadata(train, val, test, stats)
    
    # Step 6: Download actual volumes (if requested)
    if args.download_volumes:
        downloader.download_ct_volumes(train, "train")
        downloader.download_ct_volumes(val, "val")
        downloader.download_ct_volumes(test, "test")
    
    logger.info("Dataset preparation complete!")
    logger.info(f"Data saved to: {args.data_dir}")
    logger.info(f"Train: {len(train)} samples")
    logger.info(f"Val: {len(val)} samples") 
    logger.info(f"Test: {len(test)} samples")

if __name__ == "__main__":
    main()