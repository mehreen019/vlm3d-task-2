#!/usr/bin/env python3
"""
Training Script for Multi-Abnormality Classification
VLM3D Task 2: Train and evaluate 18-class binary classification model
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Import our custom modules
from multi_abnormality_classifier import train_model, MultiAbnormalityModel, create_data_loaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories(config):
    """Create necessary directories"""
    dirs_to_create = [
        config['training']['checkpoint_dir'],
        config['training']['log_dir'],
        './results',
        './predictions'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def validate_data_files(config):
    """Validate that all required data files exist"""
    required_files = [
        config['data']['train_csv'],
        config['data']['val_csv']
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        logger.info("Please run the slice extractor first to generate the CSV files")
        return False
    
    # Check if data root exists
    if not Path(config['data']['data_root']).exists():
        logger.error(f"Data root directory not found: {config['data']['data_root']}")
        return False
    
    logger.info("All required data files found ✓")
    return True

def analyze_data_distribution(config):
    """Analyze the distribution of abnormalities in the dataset"""
    logger.info("Analyzing data distribution...")
    
    train_df = pd.read_csv(config['data']['train_csv'])
    val_df = pd.read_csv(config['data']['val_csv'])
    
    abnormality_classes = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
        'Emphysema', 'Fibrosis', 'Fracture', 'Hernia', 'Infiltration',
        'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
        'Support_Devices', 'Thickening', 'No_Finding'
    ]
    
    # Calculate prevalences
    train_prevalences = {}
    val_prevalences = {}
    
    for class_name in abnormality_classes:
        if class_name in train_df.columns:
            train_prev = train_df[class_name].mean()
            val_prev = val_df[class_name].mean()
            train_prevalences[class_name] = train_prev
            val_prevalences[class_name] = val_prev
    
    # Print summary
    logger.info(f"Dataset Summary:")
    logger.info(f"  Training slices: {len(train_df)}")
    logger.info(f"  Validation slices: {len(val_df)}")
    logger.info(f"  Training volumes: {train_df['volume_name'].nunique()}")
    logger.info(f"  Validation volumes: {val_df['volume_name'].nunique()}")
    
    logger.info("\nAbnormality Prevalences (Train | Val):")
    for class_name in abnormality_classes:
        train_prev = train_prevalences.get(class_name, 0)
        val_prev = val_prevalences.get(class_name, 0)
        logger.info(f"  {class_name:18s}: {train_prev:5.3f} | {val_prev:5.3f}")
    
    # Calculate multi-label statistics
    train_labels = train_df[abnormality_classes].values
    val_labels = val_df[abnormality_classes].values
    
    train_avg_labels = np.mean(np.sum(train_labels, axis=1))
    val_avg_labels = np.mean(np.sum(val_labels, axis=1))
    
    logger.info(f"\nMulti-label Statistics:")
    logger.info(f"  Avg abnormalities per slice (Train): {train_avg_labels:.2f}")
    logger.info(f"  Avg abnormalities per slice (Val): {val_avg_labels:.2f}")
    
    return {
        'train_prevalences': train_prevalences,
        'val_prevalences': val_prevalences,
        'train_avg_labels': train_avg_labels,
        'val_avg_labels': val_avg_labels
    }

def create_test_split_if_needed(config):
    """Create test split from validation data if test.csv doesn't exist"""
    test_csv_path = Path(config['data']['test_csv'])
    
    if test_csv_path.exists():
        logger.info(f"Test split already exists: {test_csv_path}")
        return
    
    logger.info("Creating test split from validation data...")
    
    val_df = pd.read_csv(config['data']['val_csv'])
    
    # Split validation into new val and test (50/50)
    from sklearn.model_selection import train_test_split
    
    # Stratify by number of abnormalities to maintain distribution
    abnormality_classes = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
        'Emphysema', 'Fibrosis', 'Fracture', 'Hernia', 'Infiltration',
        'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
        'Support_Devices', 'Thickening', 'No_Finding'
    ]
    
    # Calculate stratification key
    available_classes = [col for col in abnormality_classes if col in val_df.columns]
    val_df['n_abnormalities'] = val_df[available_classes].sum(axis=1)
    
    try:
        new_val, test = train_test_split(
            val_df.drop('n_abnormalities', axis=1),
            test_size=0.5,
            random_state=42,
            stratify=val_df['n_abnormalities']
        )
    except ValueError:
        # Fallback to random split if stratification fails
        logger.warning("Stratified split failed, using random split")
        new_val, test = train_test_split(
            val_df.drop('n_abnormalities', axis=1),
            test_size=0.5,
            random_state=42
        )
    
    # Save new splits
    new_val.to_csv(config['data']['val_csv'], index=False)
    test.to_csv(test_csv_path, index=False)
    
    logger.info(f"Created new validation split: {len(new_val)} slices")
    logger.info(f"Created test split: {len(test)} slices")

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Abnormality Classification Model")
    parser.add_argument("--config", 
                       default="config_multi_abnormality.yaml",
                       help="Configuration file path")
    parser.add_argument("--resume-from-checkpoint", 
                       help="Resume training from checkpoint")
    parser.add_argument("--data-analysis-only", 
                       action="store_true",
                       help="Only perform data analysis without training")
    parser.add_argument("--create-test-split", 
                       action="store_true",
                       help="Create test split from validation data")
    
    args = parser.parse_args()
    
    # Load configuration
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {args.config}")
    
    # Setup directories
    setup_directories(config)
    
    # Validate data files
    if not validate_data_files(config):
        logger.error("Data validation failed. Exiting.")
        return
    
    # Create test split if requested
    if args.create_test_split:
        create_test_split_if_needed(config)
    
    # Analyze data distribution
    data_stats = analyze_data_distribution(config)
    
    # Save data analysis
    import json
    with open('./results/data_analysis.json', 'w') as f:
        json.dump(data_stats, f, indent=2)
    
    if args.data_analysis_only:
        logger.info("Data analysis complete. Exiting.")
        return
    
    # Check if GPU is available
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.warning("GPU not available, training will use CPU (slower)")
        config['training']['gpus'] = 0
    
    # Set random seeds for reproducibility
    import pytorch_lightning as pl
    pl.seed_everything(config['training']['seed'])
    
    # Start training
    logger.info("=" * 60)
    logger.info("🚀 Starting Multi-Abnormality Classification Training")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Train model
        model, trainer = train_model(config)
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("🎉 Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Training duration: {training_duration}")
        logger.info(f"Best model saved in: {config['training']['checkpoint_dir']}")
        logger.info(f"Logs saved in: {config['training']['log_dir']}")
        logger.info(f"Results saved in: ./results/")
        
        # Print final metrics summary
        logger.info("\n📊 Final Results Summary:")
        logger.info("Check TensorBoard logs for detailed metrics:")
        logger.info(f"tensorboard --logdir {config['training']['log_dir']}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 