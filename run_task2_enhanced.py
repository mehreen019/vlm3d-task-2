#!/usr/bin/env python3
"""
Enhanced VLM3D Task 2: Multi-Abnormality Classification with Cross-Validation
Improved training pipeline for better AUROC performance (>60%)
"""

import os
import sys
import argparse
import logging
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config_multi_abnormality.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def check_environment():
    """Check if required dependencies are installed and show system info"""
    try:
        import torch
        import pytorch_lightning as pl
        import torchvision
        import pandas as pd
        import numpy as np
        import sklearn
        
        logger.info("✅ All required dependencies found")
        
        # Show PyTorch and CUDA info
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"PyTorch Lightning version: {pl.__version__}")
        logger.info(f"Scikit-learn version: {sklearn.__version__}")
        
        # Show CUDA info
        if torch.cuda.is_available():
            logger.info(f"CUDA available: True")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"  GPU {i}: {gpu_name}")
        else:
            logger.warning("CUDA not available - will use CPU")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def prepare_cross_validation_splits(labels_df, config, random_state=42):
    """
    Prepare cross-validation splits using stratified approach for multi-label data
    """
    logger.info("📊 Preparing cross-validation splits...")
    
    # Filter to training data only
    train_df = labels_df[labels_df['split'] == 'train'].copy()
    
    # Extract abnormality columns (exclude VolumeName and split)
    abnormality_cols = [col for col in train_df.columns if col not in ['VolumeName', 'split']]
    
    # Create volume-level labels for stratification 
    # We need to group by patient/volume to avoid data leakage
    volume_labels = []
    volume_names = []
    
    for _, row in train_df.iterrows():
        volume_name = row['VolumeName']
        # Extract patient ID from volume name (assumes format like "train_123_a_1.nii.gz")
        patient_id = '_'.join(volume_name.split('_')[:2])  # "train_123"
        
        if patient_id not in volume_names:
            volume_names.append(patient_id)
            # Create binary vector of abnormalities for this volume
            label_vector = row[abnormality_cols].values.astype(int)
            volume_labels.append(label_vector)
    
    volume_labels = np.array(volume_labels)
    
    logger.info(f"Total unique patients/volumes: {len(volume_names)}")
    logger.info(f"Abnormality prevalence across volumes:")
    for i, col in enumerate(abnormality_cols):
        prevalence = np.mean(volume_labels[:, i])
        logger.info(f"  {col}: {prevalence:.3f}")
    
    # Use stratified split based on most common abnormalities
    # Create stratification target using top 3 most common abnormalities
    top_abnormalities = []
    for i, col in enumerate(abnormality_cols):
        prevalence = np.mean(volume_labels[:, i])
        top_abnormalities.append((col, prevalence, i))
    
    top_abnormalities.sort(key=lambda x: x[1], reverse=True)
    top_3_indices = [x[2] for x in top_abnormalities[:3]]
    
    # Create stratification labels as combinations of top 3 abnormalities
    stratify_labels = []
    for labels in volume_labels:
        # Create string representation of top 3 abnormalities
        key = ''.join([str(labels[i]) for i in top_3_indices])
        stratify_labels.append(key)
    
    stratify_labels = np.array(stratify_labels)
    
    # Create cross-validation splits
    cv_folds = config['evaluation']['cv_folds']
    if config['evaluation']['stratified_cv']:
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    cv_splits = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(volume_names, stratify_labels)):
        train_volumes = [volume_names[i] for i in train_idx]
        val_volumes = [volume_names[i] for i in val_idx]
        
        # Map back to slice-level data
        train_slices = []
        val_slices = []
        
        for _, row in train_df.iterrows():
            volume_name = row['VolumeName']
            patient_id = '_'.join(volume_name.split('_')[:2])
            
            if patient_id in train_volumes:
                train_slices.append(row.to_dict())
            elif patient_id in val_volumes:
                val_slices.append(row.to_dict())
        
        cv_splits.append({
            'fold': fold,
            'train': pd.DataFrame(train_slices),
            'val': pd.DataFrame(val_slices)
        })
        
        logger.info(f"Fold {fold}: {len(train_slices)} train slices, {len(val_slices)} val slices")
    
    return cv_splits

def run_single_fold_training(fold_data, config, args, fold_num):
    """Run training for a single cross-validation fold"""
    logger.info(f"🔄 Training fold {fold_num + 1}/{config['evaluation']['cv_folds']}")
    
    # Import training module
    from train_multi_abnormality_model_enhanced import train_model_enhanced
    
    # Create fold-specific directories
    fold_checkpoint_dir = f"./checkpoints/fold_{fold_num}"
    fold_log_dir = f"./logs/fold_{fold_num}"
    os.makedirs(fold_checkpoint_dir, exist_ok=True)
    os.makedirs(fold_log_dir, exist_ok=True)
    
    # Set up training arguments for this fold
    class FoldTrainingArgs:
        def __init__(self):
            # Data paths and splits
            self.train_data = fold_data['train']
            self.val_data = fold_data['val']
            self.slice_dir = args.slice_dir
            self.fold_num = fold_num
            
            # Model configuration from config file
            self.model_name = config['model']['backbone']
            self.num_classes = config['model']['num_classes']
            self.dropout_rate = config['model']['dropout_rate']
            
            # Training configuration from config file
            self.batch_size = config['training']['batch_size']
            self.learning_rate = config['training']['learning_rate']
            self.weight_decay = config['training']['weight_decay']
            self.max_epochs = config['training']['max_epochs']
            self.early_stopping_patience = config['training']['early_stopping_patience']
            self.gradient_clip_val = config['training']['gradient_clip_val']
            self.accumulate_grad_batches = config['training']['accumulate_grad_batches']
            
            # Loss and sampling
            self.use_focal_loss = config['training']['use_focal_loss']
            self.use_weighted_sampling = config['training']['use_weighted_sampling'] 
            self.use_mixed_precision = config['training']['use_mixed_precision']
            
            # Hardware
            self.gpus = config['training']['gpus']
            self.num_workers = config['training']['num_workers']
            self.seed = config['training']['seed']
            
            # Directories
            self.checkpoint_dir = fold_checkpoint_dir
            self.log_dir = fold_log_dir
            
            # Enhanced preprocessing and augmentation
            self.use_advanced_preprocessing = True
            self.use_advanced_augmentation = True
            self.use_tta = True  # Test Time Augmentation
            
            # Advanced training techniques
            self.use_cosine_annealing = True
            self.use_label_smoothing = True
            self.label_smoothing_factor = 0.1
            self.use_mixup = True
            self.mixup_alpha = 0.2
            
            # Class balancing
            self.use_class_weights = True
            self.focal_alpha = 0.25
            self.focal_gamma = 2.0
            
            # Model architecture enhancements
            self.use_attention = 'se'  # Squeeze-Excitation
            self.use_multiscale = True
            self.use_dropout_schedule = True
            
            # Evaluation settings
            self.decision_threshold = config['evaluation']['decision_threshold']
    
    training_args = FoldTrainingArgs()
    
    # Run training for this fold
    try:
        model, trainer, metrics = train_model_enhanced(training_args)
        logger.info(f"✅ Fold {fold_num} completed successfully")
        return {
            'fold': fold_num,
            'model': model,
            'trainer': trainer,
            'metrics': metrics,
            'checkpoint_path': fold_checkpoint_dir
        }
    except Exception as e:
        logger.error(f"❌ Fold {fold_num} failed: {e}")
        return None

def aggregate_cv_results(fold_results):
    """Aggregate results across all CV folds"""
    logger.info("📊 Aggregating cross-validation results...")
    
    valid_folds = [r for r in fold_results if r is not None]
    if not valid_folds:
        logger.error("No valid fold results to aggregate")
        return None
    
    # Collect metrics from all folds
    all_metrics = {}
    for result in valid_folds:
        fold_metrics = result['metrics']
        for metric_name, value in fold_metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)
    
    # Calculate mean and std for each metric
    aggregated_metrics = {}
    for metric_name, values in all_metrics.items():
        aggregated_metrics[f"{metric_name}_mean"] = np.mean(values)
        aggregated_metrics[f"{metric_name}_std"] = np.std(values)
        aggregated_metrics[f"{metric_name}_folds"] = values
    
    # Calculate final ranking score using weights from config
    config = load_config()
    weights = config['evaluation']['metrics']
    
    ranking_score = (
        aggregated_metrics.get('auroc_macro_mean', 0) * weights['auroc_weight'] +
        aggregated_metrics.get('f1_macro_mean', 0) * weights['f1_weight'] +
        aggregated_metrics.get('precision_macro_mean', 0) * weights['precision_weight'] +
        aggregated_metrics.get('recall_macro_mean', 0) * weights['recall_weight'] +
        aggregated_metrics.get('accuracy_mean', 0) * weights['accuracy_weight']
    )
    
    aggregated_metrics['ranking_score'] = ranking_score
    aggregated_metrics['valid_folds'] = len(valid_folds)
    aggregated_metrics['total_folds'] = len(fold_results)
    
    return aggregated_metrics

def save_results(cv_results, args):
    """Save cross-validation results"""
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(results_dir / "cv_results_detailed.json", 'w') as f:
        json.dump(cv_results, f, indent=2, default=str)
    
    # Save summary
    summary = {
        'ranking_score': cv_results['ranking_score'],
        'auroc_macro': f"{cv_results.get('auroc_macro_mean', 0):.4f} ± {cv_results.get('auroc_macro_std', 0):.4f}",
        'f1_macro': f"{cv_results.get('f1_macro_mean', 0):.4f} ± {cv_results.get('f1_macro_std', 0):.4f}",
        'precision_macro': f"{cv_results.get('precision_macro_mean', 0):.4f} ± {cv_results.get('precision_macro_std', 0):.4f}",
        'recall_macro': f"{cv_results.get('recall_macro_mean', 0):.4f} ± {cv_results.get('recall_macro_std', 0):.4f}",
        'accuracy': f"{cv_results.get('accuracy_mean', 0):.4f} ± {cv_results.get('accuracy_std', 0):.4f}",
        'valid_folds': cv_results['valid_folds'],
        'total_folds': cv_results['total_folds']
    }
    
    with open(results_dir / "cv_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("📁 Results saved to ./results/")

def main():
    parser = argparse.ArgumentParser(description="Enhanced VLM3D Task 2 with Cross-Validation")
    
    # Configuration
    parser.add_argument("--config", default="config_multi_abnormality.yaml",
                       help="Configuration file path")
    
    # Data paths
    parser.add_argument("--slice-dir", default="./ct_rate_2d", 
                       help="Extracted slices directory")
    
    # Cross-validation settings
    parser.add_argument("--cv-folds", type=int, default=None,
                       help="Number of CV folds (overrides config)")
    parser.add_argument("--run-single-fold", type=int, default=None,
                       help="Run only a specific fold (0-indexed)")
    
    # Training settings
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing checkpoints")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"✅ Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        sys.exit(1)
    
    # Override CV folds if specified
    if args.cv_folds:
        config['evaluation']['cv_folds'] = args.cv_folds
    
    # Load data
    slice_dir = Path(args.slice_dir)
    labels_path = slice_dir / "multi_abnormality_labels.csv"
    
    if not labels_path.exists():
        logger.error(f"❌ Labels file not found: {labels_path}")
        sys.exit(1)
    
    labels_df = pd.read_csv(labels_path)
    logger.info(f"📊 Loaded labels for {len(labels_df)} samples")
    
    # Prepare cross-validation splits
    cv_splits = prepare_cross_validation_splits(labels_df, config, args.seed)
    
    logger.info("=" * 60)
    logger.info("🏥 Enhanced VLM3D Task 2: Cross-Validation Training")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"CV Folds: {config['evaluation']['cv_folds']}")
    logger.info(f"Model: {config['model']['backbone']}")
    logger.info(f"Slice directory: {args.slice_dir}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 60)
    
    # Run cross-validation training
    fold_results = []
    
    folds_to_run = [args.run_single_fold] if args.run_single_fold is not None else range(len(cv_splits))
    
    for fold_num in folds_to_run:
        if fold_num >= len(cv_splits):
            logger.warning(f"Fold {fold_num} doesn't exist, skipping")
            continue
            
        fold_data = cv_splits[fold_num]
        result = run_single_fold_training(fold_data, config, args, fold_num)
        fold_results.append(result)
    
    # Aggregate results if running all folds
    if args.run_single_fold is None:
        cv_results = aggregate_cv_results(fold_results)
        if cv_results:
            save_results(cv_results, args)
            
            logger.info("=" * 60)
            logger.info("🎉 Cross-Validation Results Summary")
            logger.info("=" * 60)
            logger.info(f"📊 Final Ranking Score: {cv_results['ranking_score']:.4f}")
            logger.info(f"🎯 AUROC (macro): {cv_results.get('auroc_macro_mean', 0):.4f} ± {cv_results.get('auroc_macro_std', 0):.4f}")
            logger.info(f"📈 F1 (macro): {cv_results.get('f1_macro_mean', 0):.4f} ± {cv_results.get('f1_macro_std', 0):.4f}")
            logger.info(f"🎯 Precision (macro): {cv_results.get('precision_macro_mean', 0):.4f} ± {cv_results.get('precision_macro_std', 0):.4f}")
            logger.info(f"🎯 Recall (macro): {cv_results.get('recall_macro_mean', 0):.4f} ± {cv_results.get('recall_macro_std', 0):.4f}")
            logger.info(f"✅ Accuracy: {cv_results.get('accuracy_mean', 0):.4f} ± {cv_results.get('accuracy_std', 0):.4f}")
            logger.info(f"📁 Valid/Total Folds: {cv_results['valid_folds']}/{cv_results['total_folds']}")
            logger.info("=" * 60)
            
            if cv_results.get('auroc_macro_mean', 0) > 0.60:
                logger.info("🎉 SUCCESS: AUROC > 0.60 achieved!")
            else:
                logger.warning("⚠️ AUROC < 0.60 - consider further improvements")
    else:
        logger.info(f"✅ Single fold {args.run_single_fold} training completed")

if __name__ == "__main__":
    main()