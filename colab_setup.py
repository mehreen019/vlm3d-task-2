#!/usr/bin/env python3
"""
Colab Setup Script for VLM3D Task 2
Handles Google Drive mounting and checkpoint backup
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_colab_environment():
    """Setup Colab environment with Drive mounting"""
    
    try:
        # Check if running in Colab
        import google.colab
        logger.info("🔍 Detected Google Colab environment")
        
        # Mount Google Drive
        from google.colab import drive
        drive.mount('/content/drive')
        logger.info("📁 Google Drive mounted successfully")
        
        # Create backup directory in Drive
        drive_backup_dir = Path("/content/drive/MyDrive/VLM3D_Task2_Backup")
        drive_backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📂 Backup directory created: {drive_backup_dir}")
        
        return str(drive_backup_dir)
        
    except ImportError:
        logger.info("ℹ️ Not running in Colab, skipping Drive setup")
        return None

def backup_to_drive(backup_dir=None):
    """Backup checkpoints and weights to Google Drive"""
    
    if not backup_dir:
        backup_dir = setup_colab_environment()
    
    if not backup_dir:
        logger.warning("⚠️ No backup directory available")
        return
    
    backup_path = Path(backup_dir)
    
    try:
        # Backup checkpoints
        checkpoint_dir = Path("./checkpoints")
        if checkpoint_dir.exists():
            checkpoint_backup = backup_path / "checkpoints"
            checkpoint_backup.mkdir(exist_ok=True)
            
            for checkpoint in checkpoint_dir.glob("*.ckpt"):
                shutil.copy2(checkpoint, checkpoint_backup / checkpoint.name)
                logger.info(f"📦 Backed up checkpoint: {checkpoint.name}")
        
        # Backup model weights
        weights_dir = Path("./model_weights")
        if weights_dir.exists():
            weights_backup = backup_path / "model_weights"
            weights_backup.mkdir(exist_ok=True)
            
            for weight_file in weights_dir.glob("*"):
                if weight_file.is_file():
                    shutil.copy2(weight_file, weights_backup / weight_file.name)
                    logger.info(f"💾 Backed up weights: {weight_file.name}")
        
        # Backup logs
        logs_dir = Path("./logs")
        if logs_dir.exists():
            logs_backup = backup_path / "logs"
            if logs_backup.exists():
                shutil.rmtree(logs_backup)
            shutil.copytree(logs_dir, logs_backup)
            logger.info("📊 Backed up training logs")
        
        logger.info(f"✅ Backup completed to: {backup_path}")
        
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")

def restore_from_drive(backup_dir=None):
    """Restore checkpoints and weights from Google Drive"""
    
    if not backup_dir:
        backup_dir = "/content/drive/MyDrive/VLM3D_Task2_Backup"
    
    backup_path = Path(backup_dir)
    
    if not backup_path.exists():
        logger.warning(f"⚠️ No backup found at: {backup_path}")
        return
    
    try:
        # Restore checkpoints
        checkpoint_backup = backup_path / "checkpoints"
        if checkpoint_backup.exists():
            checkpoint_dir = Path("./checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            for checkpoint in checkpoint_backup.glob("*.ckpt"):
                shutil.copy2(checkpoint, checkpoint_dir / checkpoint.name)
                logger.info(f"🔄 Restored checkpoint: {checkpoint.name}")
        
        # Restore model weights
        weights_backup = backup_path / "model_weights"
        if weights_backup.exists():
            weights_dir = Path("./model_weights")
            weights_dir.mkdir(exist_ok=True)
            
            for weight_file in weights_backup.glob("*"):
                if weight_file.is_file():
                    shutil.copy2(weight_file, weights_dir / weight_file.name)
                    logger.info(f"💾 Restored weights: {weight_file.name}")
        
        # Restore logs
        logs_backup = backup_path / "logs"
        if logs_backup.exists():
            logs_dir = Path("./logs")
            if logs_dir.exists():
                shutil.rmtree(logs_dir)
            shutil.copytree(logs_backup, logs_dir)
            logger.info("📊 Restored training logs")
        
        logger.info("✅ Restore completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Restore failed: {e}")

def auto_backup_every_n_epochs(n=10):
    """Auto-backup every N epochs (call this in training loop)"""
    
    try:
        # Simple epoch counting
        epoch_file = Path("./current_epoch.txt")
        current_epoch = 0
        
        if epoch_file.exists():
            current_epoch = int(epoch_file.read_text().strip())
        
        current_epoch += 1
        epoch_file.write_text(str(current_epoch))
        
        # Backup every N epochs
        if current_epoch % n == 0:
            logger.info(f"🔄 Auto-backup triggered at epoch {current_epoch}")
            backup_to_drive()
        
    except Exception as e:
        logger.warning(f"⚠️ Auto-backup failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Colab Setup and Backup")
    parser.add_argument("--action", choices=["setup", "backup", "restore"], default="setup",
                       help="Action to perform")
    parser.add_argument("--backup-dir", help="Custom backup directory path")
    
    args = parser.parse_args()
    
    if args.action == "setup":
        backup_dir = setup_colab_environment()
        if backup_dir:
            logger.info(f"🎯 Use this backup directory: {backup_dir}")
    
    elif args.action == "backup":
        backup_to_drive(args.backup_dir)
    
    elif args.action == "restore":
        restore_from_drive(args.backup_dir)