#!/usr/bin/env python3
"""
VLM3D Task 2: Multi-Abnormality Classification
Simple runner script that integrates with existing CT-RATE pipeline
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check if required dependencies are installed"""
    try:
        import torch
        import pytorch_lightning as pl
        import torchvision
        import pandas as pd
        import numpy as np
        logger.info("✅ All required dependencies found")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Please run: source venv_name/bin/activate (after running setup_env.sh)")
        return False

def check_data_structure(data_dir="./ct_rate_data", slice_dir="./ct_rate_2d"):
    """Check if data structure is correct"""
    data_dir = Path(data_dir)
    slice_dir = Path(slice_dir)
    
    required_files = [
        data_dir / "multi_abnormality_labels.csv",
        slice_dir / "splits" / "train_slices.csv", 
        slice_dir / "splits" / "valid_slices.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("❌ Missing required files:")
        for file_path in missing_files:
            logger.error(f"   {file_path}")
        logger.error("\nPlease run:")
        logger.error("1. python ct_rate_downloader.py --max-storage-gb 5 --download-volumes")
        logger.error("2. python 2d_slice_extractor.py")
        return False
    
    logger.info("✅ Data structure is correct")
    return True

def run_training(args):
    """Run the training pipeline"""
    logger.info("🚀 Starting Multi-Abnormality Classification Training")
    
    # Import the training module
    from train_multi_abnormality_model import train_model
    
    # Set up arguments for training
    class TrainingArgs:
        def __init__(self):
            self.data_dir = args.data_dir
            self.slice_dir = args.slice_dir
            self.model_name = args.model
            self.batch_size = args.batch_size
            self.learning_rate = args.learning_rate
            self.max_epochs = args.epochs
            self.early_stopping_patience = 10
            self.dropout_rate = 0.3
            self.weight_decay = 1e-5
            self.num_workers = 4
            self.use_mixed_precision = True
            self.checkpoint_dir = "./checkpoints"
            self.log_dir = "./logs"
    
    training_args = TrainingArgs()
    
    # Create output directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # Run training
    try:
        model, trainer = train_model(training_args)
        logger.info("🎉 Training completed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

def run_evaluation(checkpoint_path, args):
    """Run evaluation on trained model"""
    logger.info(f"📊 Evaluating model: {checkpoint_path}")
    
    from train_multi_abnormality_model import evaluate_model
    
    try:
        metrics = evaluate_model(checkpoint_path, args.slice_dir, args.data_dir)
        logger.info("🎉 Evaluation completed successfully!")
        return metrics
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return None

def run_prediction(checkpoint_path, args):
    """Run prediction on new data"""
    logger.info(f"🔮 Running prediction with model: {checkpoint_path}")
    logger.info(f"Input files: {len(args.predict_input)} {args.predict_type}")
    
    from predict_abnormalities import AbnormalityPredictor
    
    try:
        # Initialize predictor
        predictor = AbnormalityPredictor(checkpoint_path)
        
        # Run predictions
        if args.predict_type == "slices":
            results = predictor.predict_slices(args.predict_input, batch_size=args.batch_size)
        else:
            results = predictor.predict_volumes(args.predict_input, batch_size=args.batch_size)
        
        # Print summary
        predictor.print_summary(results)
        
        # Save results
        output_path = f"predictions_{args.predict_type}.json"
        predictor.save_predictions(results, output_path)
        
        logger.info("🎉 Prediction completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return False

def find_best_checkpoint():
    """Find the best model checkpoint"""
    checkpoint_dir = Path("./checkpoints")
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    
    # Return the most recent checkpoint
    return str(sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1])

def main():
    parser = argparse.ArgumentParser(description="VLM3D Task 2: Multi-Abnormality Classification")
    
    # Mode selection
    parser.add_argument("--mode", choices=["train", "evaluate", "both", "predict"], default="both",
                       help="Mode: train only, evaluate only, both, or predict")
    
    # Data paths
    parser.add_argument("--data-dir", default="./ct_rate_data", 
                       help="CT-RATE data directory")
    parser.add_argument("--slice-dir", default="./ct_rate_2d", 
                       help="Extracted slices directory")
    
    # Model parameters
    parser.add_argument("--model", default="resnet50", 
                       choices=["resnet50", "resnet101", "efficientnet_b0"],
                       help="Model backbone")
    parser.add_argument("--batch-size", type=int, default=5, 
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Number of epochs")
    
    # Evaluation and Prediction
    parser.add_argument("--checkpoint", 
                       help="Checkpoint path for evaluation/prediction (auto-detected if not specified)")
    parser.add_argument("--predict-input", nargs="+",
                       help="Input files for prediction (slices or volumes)")
    parser.add_argument("--predict-type", choices=["slices", "volumes"], default="slices",
                       help="Type of prediction input")
    
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check data structure
    if not check_data_structure(args.data_dir, args.slice_dir):
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("🏥 VLM3D Task 2: Multi-Abnormality Classification")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Slice directory: {args.slice_dir}")
    logger.info("=" * 60)
    
    success = True
    
    # Training
    if args.mode in ["train", "both"]:
        success = run_training(args)
        if not success:
            sys.exit(1)
    
    # Evaluation
    if args.mode in ["evaluate", "both"]:
        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            checkpoint_path = find_best_checkpoint()
            if not checkpoint_path:
                logger.error("❌ No checkpoint found for evaluation")
                sys.exit(1)
        
        metrics = run_evaluation(checkpoint_path, args)
        if metrics is None:
            sys.exit(1)
    
    # Prediction
    if args.mode == "predict":
        if not args.predict_input:
            logger.error("❌ --predict-input required for prediction mode")
            sys.exit(1)
        
        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            checkpoint_path = find_best_checkpoint()
            if not checkpoint_path:
                logger.error("❌ No checkpoint found for prediction")
                sys.exit(1)
        
        success = run_prediction(checkpoint_path, args)
        if not success:
            sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("🎉 Task 2 Pipeline Completed Successfully!")
    logger.info("=" * 60)
    
    if args.mode in ["evaluate", "both"]:
        logger.info("📊 Final Results:")
        if 'auroc_macro' in metrics:
            logger.info(f"   AUROC (macro): {metrics['auroc_macro']:.4f}")
        if 'f1_macro' in metrics:
            logger.info(f"   F1 (macro):    {metrics['f1_macro']:.4f}")
        if 'accuracy' in metrics:
            logger.info(f"   Accuracy:      {metrics['accuracy']:.4f}")
    
    logger.info("\n📁 Output files:")
    logger.info("   - Model checkpoints: ./checkpoints/")
    logger.info("   - Training logs: ./logs/")
    logger.info("   - Results: ./results/")
    logger.info("\n🔧 To view training progress:")
    logger.info("   tensorboard --logdir ./logs")

if __name__ == "__main__":
    main() 