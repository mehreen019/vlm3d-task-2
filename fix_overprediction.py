#!/usr/bin/env python3
"""
Emergency Fix for Over-prediction in VLM3D Task 2
Your model is predicting 98.7% positives - this script provides immediate solutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class ImbalanceFixedModel(nn.Module):
    """Model wrapper that applies bias correction for severe over-prediction"""
    
    def __init__(self, base_model, bias_correction=True, optimal_threshold=None):
        super().__init__()
        self.base_model = base_model
        self.bias_correction = bias_correction
        self.optimal_threshold = optimal_threshold or 0.5
        
        # Learned bias correction parameters
        self.class_bias = nn.Parameter(torch.zeros(18))  # 18 classes
        
    def forward(self, x):
        logits = self.base_model(x)
        
        if self.bias_correction:
            # Apply learned bias correction
            logits = logits + self.class_bias
            
        return logits

def create_balanced_loss():
    """Create a severely imbalanced-data-aware loss function"""
    
    class BalancedFocalLoss(nn.Module):
        def __init__(self, alpha=0.75, gamma=3.0, pos_weight=None):
            super().__init__()
            self.alpha = alpha  # Higher alpha = penalize positive predictions more
            self.gamma = gamma  # Higher gamma = focus more on hard examples
            self.pos_weight = pos_weight
            
        def forward(self, logits, targets):
            # Apply position weights if provided
            if self.pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(
                    logits, targets, 
                    pos_weight=self.pos_weight,
                    reduction='none'
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            
            # Convert to focal loss
            pt = torch.exp(-loss)
            
            # Apply alpha weighting (higher alpha = penalize positives more)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            
            # Apply focal term
            focal_loss = alpha_t * (1 - pt) ** self.gamma * loss
            
            return focal_loss.mean()
    
    return BalancedFocalLoss(alpha=0.85, gamma=4.0)  # Very aggressive against over-prediction

def calculate_pos_weights(train_labels):
    """Calculate position weights to severely penalize over-prediction"""
    # Count positive samples per class
    pos_counts = np.sum(train_labels, axis=0)
    neg_counts = len(train_labels) - pos_counts
    
    # Calculate weights (higher weight = more penalty for positive predictions)
    pos_weights = neg_counts / np.maximum(pos_counts, 1)  # Avoid division by zero
    
    # Apply aggressive scaling to combat over-prediction
    pos_weights = pos_weights * 2.0  # Double the penalty
    
    return torch.tensor(pos_weights, dtype=torch.float32)

def emergency_training_config():
    """Return emergency training configuration to fix over-prediction"""
    
    config = {
        "loss_function": "aggressive_focal",
        "learning_rate": 5e-5,  # Lower LR for stability
        "batch_size": 16,       # Smaller batch for more frequent updates
        "dropout_rate": 0.6,    # High dropout to prevent overfitting
        "weight_decay": 1e-3,   # High weight decay
        "focal_alpha": 0.85,    # Heavily penalize positive predictions
        "focal_gamma": 4.0,     # Focus on hard examples
        "early_stopping_patience": 5,  # Stop early if not improving
        "monitor_metric": "val_precision_macro",  # Focus on precision
        "threshold_search": True,  # Search for optimal threshold
    }
    
    return config

def quick_fix_commands():
    """Generate quick fix commands"""
    
    commands = [
        # Emergency Fix 1: Aggressive Focal Loss
        """python run_task2.py \\
  --model efficientnet_b0 \\
  --loss-type focal \\
  --freeze-backbone \\
  --batch-size 16 \\
  --learning-rate 5e-5 \\
  --epochs 30 \\
  --dropout-rate 0.6""",
        
        # Emergency Fix 2: ResNet with ASL
        """python run_task2.py \\
  --model resnet50 \\
  --loss-type asl \\
  --freeze-backbone \\
  --batch-size 16 \\
  --learning-rate 1e-5 \\
  --epochs 50""",
        
        # Emergency Fix 3: Very Conservative Training
        """python run_task2.py \\
  --model efficientnet_b0 \\
  --loss-type asl \\
  --freeze-backbone \\
  --batch-size 8 \\
  --learning-rate 1e-6 \\
  --epochs 100 \\
  --early-stopping-patience 10"""
    ]
    
    return commands

def analyze_overprediction_causes():
    """Analyze why the model is over-predicting"""
    
    causes = {
        "Data Imbalance": {
            "description": "Some classes have 96 positives, others have 0",
            "solution": "Use class weights or asymmetric loss",
            "severity": "HIGH"
        },
        
        "Learning Rate Too High": {
            "description": "Model learns to predict everything as positive",
            "solution": "Reduce LR to 1e-5 or 5e-5",
            "severity": "HIGH"
        },
        
        "Insufficient Regularization": {
            "description": "Model overfits to majority class pattern",
            "solution": "Increase dropout to 0.6, weight decay to 1e-3",
            "severity": "MEDIUM"
        },
        
        "Wrong Loss Function": {
            "description": "Standard losses don't handle extreme imbalance",
            "solution": "Use ASL with aggressive parameters",
            "severity": "HIGH"
        },
        
        "Threshold Issue": {
            "description": "0.5 threshold is wrong for imbalanced data",
            "solution": "Search for optimal threshold (likely 0.7-0.8)",
            "severity": "MEDIUM"
        }
    }
    
    return causes

def main():
    print("🚨 EMERGENCY FIX FOR OVER-PREDICTION")
    print("=" * 60)
    print("Your model is predicting 98.7% positives - this is severely broken!")
    print()
    
    # Analyze causes
    print("🔍 ROOT CAUSE ANALYSIS:")
    causes = analyze_overprediction_causes()
    for cause, info in causes.items():
        print(f"  • {cause} [{info['severity']}]")
        print(f"    Problem: {info['description']}")
        print(f"    Solution: {info['solution']}")
        print()
    
    # Emergency config
    print("⚡ EMERGENCY CONFIGURATION:")
    config = emergency_training_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Quick fixes
    print("🔧 IMMEDIATE FIXES TO TRY:")
    commands = quick_fix_commands()
    
    for i, cmd in enumerate(commands, 1):
        print(f"\n{i}. Emergency Fix {i}:")
        print(cmd)
    
    print()
    print("💡 RECOMMENDATIONS:")
    print("1. Try Emergency Fix 1 first (should show immediate improvement)")
    print("2. If still over-predicting, try Emergency Fix 3 (very conservative)")
    print("3. Monitor precision - it should be >40% after fix")
    print("4. Positive prediction rate should drop to <70%")
    print()
    print("🎯 SUCCESS CRITERIA:")
    print("  ✅ Positive prediction rate: <70% (currently 98.7%)")
    print("  ✅ Precision macro: >35% (currently 24.4%)")
    print("  ✅ AUROC macro: >60% (currently 45.1%)")
    print("  ✅ Hamming accuracy: >35% (currently 24.9%)")

if __name__ == "__main__":
    main()
