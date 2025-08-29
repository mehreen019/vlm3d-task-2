#!/usr/bin/env python3
"""
CT-CLIP Training Script for VLM3D Task 2
Trains the CT-CLIP model with aggressive loss to fix over-prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import os
import argparse
from ctclip_model import CTCLIPMultiAbnormalityModel, create_balanced_loss

class CTDataset(Dataset):
    """Simple CT dataset for testing"""

    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        # Simulate CT slices
        self.data = torch.randn(n_samples, 3, 224, 224)
        # Simulate multi-label targets (18 abnormalities)
        self.targets = torch.randint(0, 2, (n_samples, 18)).float()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'image': self.data[idx],
            'labels': self.targets[idx]
        }

def create_differential_optimizer(model, lr_backbone=1e-5, lr_classifier=1e-3, weight_decay=1e-4):
    """Create optimizer with different learning rates (CT-CLIP approach)"""

    optimizer = optim.AdamW([
        {
            'params': model.feature_extractor.parameters(),
            'lr': lr_backbone,
            'weight_decay': weight_decay
        },
        {
            'params': model.classifier.parameters(),
            'lr': lr_classifier,
            'weight_decay': weight_decay
        }
    ])

    return optimizer

def train_epoch(model, train_loader, criterion, optimizer, device, freeze_backbone=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    batch_count = 0

    if freeze_backbone:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    for batch in train_loader:
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        if batch_count % 10 == 0:
            print(".4f"
    avg_loss = total_loss / batch_count
    print(".4f"
    return avg_loss

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    batch_count = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Get predictions
            preds = torch.sigmoid(outputs).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels_np)

            total_loss += loss.item()
            batch_count += 1

    avg_loss = total_loss / batch_count

    # Concatenate all predictions and labels
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Calculate metrics
    pos_rate = np.sum(all_preds > 0.5) / all_preds.size

    print(".4f"    print(".1%"
    return avg_loss, pos_rate

def progressive_unfreezing_training(model, train_loader, val_loader, device, num_epochs=50):
    """Train with progressive unfreezing (CT-CLIP approach)"""

    # Phase 1: Train only classifier (backbone frozen)
    print("\n🧊 PHASE 1: Training Classifier Only (Backbone Frozen)")
    print("=" * 60)

    criterion = create_balanced_loss(alpha=0.85, gamma=4.0)
    optimizer = create_differential_optimizer(model, lr_backbone=0, lr_classifier=1e-3)

    for epoch in range(10):  # Train classifier for 10 epochs
        print(f"\nEpoch {epoch+1}/10")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, freeze_backbone=True)
        val_loss, pos_rate = validate_epoch(model, val_loader, criterion, device)

    # Phase 2: Progressive unfreezing
    print("\n🔥 PHASE 2: Progressive Unfreezing")
    print("=" * 60)

    unfreeze_schedule = [
        (15, 1e-5),   # Unfreeze at epoch 15 with low LR
        (25, 5e-5),   # Increase LR at epoch 25
        (35, 1e-4)    # Final LR increase at epoch 35
    ]

    current_lr_backbone = 0
    optimizer = create_differential_optimizer(model, lr_backbone=current_lr_backbone, lr_classifier=5e-4)

    for epoch in range(10, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Check if we should unfreeze
        for unfreeze_epoch, new_lr in unfreeze_schedule:
            if epoch == unfreeze_epoch:
                current_lr_backbone = new_lr
                optimizer = create_differential_optimizer(
                    model,
                    lr_backbone=current_lr_backbone,
                    lr_classifier=5e-4
                )
                print(f"🚀 Unfreezing backbone with LR: {new_lr}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, freeze_backbone=False)
        val_loss, pos_rate = validate_epoch(model, val_loader, criterion, device)

        # Early stopping based on positive rate
        if pos_rate < 0.4:  # If we're under-predicting too much
            print("⚠️ Positive rate too low, stopping to prevent under-prediction")
            break

    print("
🎉 Training complete!"    print("📊 Final model performance:"    print(f"   • Positive prediction rate should be around 50-60%"    print("   • Should show significant improvement over 97% baseline"
def quick_training_test():
    """Quick training test to verify everything works"""
    print("🧪 Quick Training Test")
    print("=" * 40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = CTCLIPMultiAbnormalityModel(ctclip_path="models/ctclip_classfine.pt")
    model.to(device)

    # Create datasets
    train_dataset = CTDataset(n_samples=200)
    val_dataset = CTDataset(n_samples=50)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Quick training
    criterion = create_balanced_loss(alpha=0.85, gamma=4.0)
    optimizer = create_differential_optimizer(model, lr_backbone=0, lr_classifier=1e-3)

    print("🚀 Starting quick training test...")

    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, freeze_backbone=True)
        val_loss, pos_rate = validate_epoch(model, val_loader, criterion, device)

    print("
✅ Quick training test successful!"    print("🎯 The aggressive loss should significantly reduce positive prediction rate"
def main():
    parser = argparse.ArgumentParser(description="CT-CLIP Training for VLM3D Task 2")
    parser.add_argument("--mode", choices=["quick_test", "full_train"], default="quick_test",
                       help="Training mode")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args = parser.parse_args()

    if args.mode == "quick_test":
        quick_training_test()
    elif args.mode == "full_train":
        print("🚀 Full training mode selected")
        # Create actual model and datasets here
        # This would use your real CT-RATE data
        print("💡 For full training, integrate this with your existing run_task2.py pipeline")

if __name__ == "__main__":
    main()
