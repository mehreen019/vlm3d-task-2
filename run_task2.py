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
    """Check if required dependencies are installed and show system info"""
    try:
        import torch
        import pytorch_lightning as pl
        import torchvision
        import pandas as pd
        import numpy as np
        
        logger.info("✅ All required dependencies found")
        
        # Show PyTorch and CUDA info
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"PyTorch Lightning version: {pl.__version__}")
        
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
        logger.error("Please run: conda run -n YOUR_ENV_NAME python run_task2.py ...")
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

def find_resume_checkpoint():
    """Find the most recent checkpoint to resume from"""
    checkpoint_dir = Path("./checkpoints")
    if not checkpoint_dir.exists():
        return None
    
    # Look for the most recent checkpoint
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    
    # Return the most recent checkpoint
    latest_checkpoint = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
    logger.info(f"🔄 Found existing checkpoint: {latest_checkpoint}")
    return str(latest_checkpoint)


def run_training(args):
    """Run the training pipeline with Colab-friendly checkpointing"""
    logger.info("🚀 Starting Multi-Abnormality Classification Training")
    
    # Check for existing checkpoint to resume from
    resume_checkpoint = None
    if args.resume and not args.force_restart:
        resume_checkpoint = find_resume_checkpoint()
        if resume_checkpoint:
            logger.info(f"📂 Resuming from checkpoint: {resume_checkpoint}")
        else:
            logger.info("📂 No checkpoint found, starting fresh training")
    
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
            self.early_stopping_patience = getattr(args, 'early_stopping_patience', 10)
            self.dropout_rate = 0.3
            self.weight_decay = 1e-5
            self.num_workers = 4
            self.use_mixed_precision = True
            self.checkpoint_dir = "./checkpoints"
            self.log_dir = "./logs"
            self.resume_from_checkpoint = resume_checkpoint  # Add resume capability

            # Advanced training arguments
            self.freeze_backbone = getattr(args, 'freeze_backbone', False)
            self.use_attention = getattr(args, 'use_attention', 'none')
            self.use_multiscale = getattr(args, 'use_multiscale', False)
            self.loss_type = getattr(args, 'loss_type', 'focal')
            self.progressive_unfreeze = getattr(args, 'progressive_unfreeze', False)
            self.unfreeze_epoch = getattr(args, 'unfreeze_epoch', 10)
            self.use_advanced_aug = getattr(args, 'use_advanced_aug', False)
            self.cutmix_prob = getattr(args, 'cutmix_prob', 0.5)
            self.gpu_device = getattr(args, 'gpu_device', None)
            
            # Colab-friendly settings
            self.save_every_n_epochs = getattr(args, 'save_every_n_epochs', 5)  # Save every 5 epochs
            self.save_final_weights = True
    
    training_args = TrainingArgs()
    
    # Create output directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # Run training
    try:
        model, trainer = train_model(training_args)
        
        # Save final model weights
        save_final_weights(model, args)
        
        logger.info("🎉 Training completed successfully!")
        logger.info("💾 Model weights saved!")
        return True
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

def save_final_weights(model, args):
    """Save final model weights in multiple formats"""
    import torch
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_dir = Path("./model_weights")
    weights_dir.mkdir(exist_ok=True)
    
    try:
        # Save PyTorch state dict (most portable)
        torch.save(model.state_dict(), 
                  weights_dir / f"model_weights_{args.model}_{timestamp}.pth")
        
        # Save complete model (includes architecture)
        torch.save(model, 
                  weights_dir / f"complete_model_{args.model}_{timestamp}.pt")
        
        # Save model info
        model_info = {
            'model_name': args.model,
            'timestamp': timestamp,
            'parameters': sum(p.numel() for p in model.parameters()),
            'architecture': str(model.__class__.__name__),
            'training_args': {
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'epochs': args.epochs,
                'loss_type': getattr(args, 'loss_type', 'focal'),
                'use_attention': getattr(args, 'use_attention', 'none')
            }
        }
        
        import json
        with open(weights_dir / f"model_info_{timestamp}.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"💾 Final weights saved to: {weights_dir}")
        logger.info(f"   - PyTorch weights: model_weights_{args.model}_{timestamp}.pth")
        logger.info(f"   - Complete model: complete_model_{args.model}_{timestamp}.pt") 
        logger.info(f"   - Model info: model_info_{timestamp}.json")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not save final weights: {e}")

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
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                       help="Early stopping patience (epochs without improvement)")

    # Colab-friendly checkpoint options
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume from the most recent checkpoint if available")
    parser.add_argument("--force-restart", action="store_true",
                       help="Force restart training from scratch (ignore existing checkpoints)")
    parser.add_argument("--save-every-n-epochs", type=int, default=5,
                       help="Save checkpoint every N epochs (for Colab timeout protection)")

    # Advanced training parameters
    parser.add_argument("--freeze-backbone", action="store_true",
                       help="Freeze backbone layers during training")
    parser.add_argument("--use-attention", choices=["none", "se", "cbam"], default="none",
                       help="Attention mechanism: none, se (Squeeze-Excitation), cbam (CBAM)")
    parser.add_argument("--use-multiscale", action="store_true",
                       help="Use multi-scale feature fusion")
    parser.add_argument("--loss-type", choices=["focal", "bce", "asl"], default="focal",
                       help="Loss function: focal, bce, asl (Asymmetric Loss)")
    parser.add_argument("--progressive-unfreeze", action="store_true",
                       help="Use progressive unfreezing of backbone layers")
    parser.add_argument("--unfreeze-epoch", type=int, default=10,
                       help="Epoch to unfreeze backbone in progressive unfreezing")
    parser.add_argument("--use-advanced-aug", action="store_true",
                       help="Use advanced augmentations (CutMix, MixUp)")
    parser.add_argument("--cutmix-prob", type=float, default=0.5,
                       help="Probability of applying CutMix augmentation")
    
    # GPU Configuration
    parser.add_argument("--gpu-device", type=int, default=None,
                       help="Specific GPU device ID to use (0, 1, etc.). If not specified, auto-selects NVIDIA GPU")
    
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
    if args.gpu_device is not None:
        logger.info(f"GPU device: {args.gpu_device} (manually specified)")
    else:
        logger.info("GPU device: Auto-select (will prefer NVIDIA over AMD)")
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
    logger.info("\n💡 For future runs with conda environment:")
    logger.info(f"   conda run -n YOUR_ENV_NAME python run_task2.py --gpu-device 1 --loss-type focal --freeze-backbone --epochs 50")
    logger.info("   (Replace YOUR_ENV_NAME with your actual conda environment name)")

if __name__ == "__main__":
    main() 