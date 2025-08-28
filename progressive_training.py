#!/usr/bin/env python3
"""
Progressive Training Strategies for Medical Image Classification
Implements freezing, progressive unfreezing, and layer-wise learning rates
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class ProgressiveTrainingMixin:
    """Mixin class for progressive training strategies"""
    
    def setup_progressive_training(self, 
                                 strategy: str = "progressive_unfreeze",
                                 freeze_epochs: int = 5,
                                 unfreeze_schedule: Dict[int, int] = None,
                                 layer_wise_lr: bool = True,
                                 discriminative_lr_factor: float = 0.1):
        """
        Setup progressive training strategy
        
        Args:
            strategy: 'freeze_backbone', 'progressive_unfreeze', 'layer_wise_lr'
            freeze_epochs: Number of epochs to keep backbone frozen initially
            unfreeze_schedule: Dict mapping epoch to number of layers to unfreeze
            layer_wise_lr: Whether to use different learning rates for different layers
            discriminative_lr_factor: Factor for layer-wise learning rate reduction
        """
        self.training_strategy = strategy
        self.freeze_epochs = freeze_epochs
        self.layer_wise_lr = layer_wise_lr
        self.discriminative_lr_factor = discriminative_lr_factor
        
        # Default unfreeze schedule for ResNet
        if unfreeze_schedule is None:
            self.unfreeze_schedule = {
                0: 0,      # Start with everything frozen
                5: 1,      # Unfreeze last ResNet block (layer4)
                10: 2,     # Unfreeze layer3
                15: 3,     # Unfreeze layer2
                20: 4,     # Unfreeze everything
            }
        else:
            self.unfreeze_schedule = unfreeze_schedule
        
        # Initialize freezing
        self.apply_freezing_strategy()
        
        logger.info(f"🧊 Progressive training setup: {strategy}")
        logger.info(f"   Freeze epochs: {freeze_epochs}")
        logger.info(f"   Layer-wise LR: {layer_wise_lr}")
        logger.info(f"   Unfreeze schedule: {self.unfreeze_schedule}")
    
    def apply_freezing_strategy(self):
        """Apply initial freezing strategy"""
        if self.training_strategy in ["freeze_backbone", "progressive_unfreeze"]:
            self.freeze_backbone_layers()
            logger.info("🧊 Backbone layers frozen for initial training")
    
    def freeze_backbone_layers(self, num_layers_to_keep_unfrozen: int = 0):
        """
        Freeze backbone layers, keeping only the last few unfrozen
        
        Args:
            num_layers_to_keep_unfrozen: Number of backbone layers to keep trainable
        """
        # Get backbone layers for ResNet
        if hasattr(self.backbone, 'layer4'):  # ResNet structure
            resnet_layers = [
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4
            ]
        else:
            # For Sequential backbone (already modified)
            resnet_layers = list(self.backbone.children())
        
        # Freeze layers
        for i, layer in enumerate(resnet_layers):
            freeze_layer = i < (len(resnet_layers) - num_layers_to_keep_unfrozen)
            
            for param in layer.parameters():
                param.requires_grad = not freeze_layer
            
            if freeze_layer:
                layer.eval()  # Set to eval mode to freeze batch norm
        
        # Always keep classifier trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        # Always keep attention trainable
        if hasattr(self, 'attention'):
            for param in self.attention.parameters():
                param.requires_grad = True
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"   Trainable parameters: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.1f}%)")
    
    def progressive_unfreeze_callback(self, current_epoch: int):
        """
        Callback to progressively unfreeze layers during training
        Called at the beginning of each epoch
        """
        if self.training_strategy != "progressive_unfreeze":
            return
        
        if current_epoch in self.unfreeze_schedule:
            num_layers_unfrozen = self.unfreeze_schedule[current_epoch]
            self.freeze_backbone_layers(num_layers_unfrozen)
            
            # Update learning rates if using layer-wise LR
            if self.layer_wise_lr and hasattr(self, 'trainer'):
                self.update_layer_wise_learning_rates(current_epoch)
            
            logger.info(f"🔓 Epoch {current_epoch}: Unfroze {num_layers_unfrozen} backbone layers")
    
    def get_layer_wise_learning_rates(self) -> List[Dict]:
        """
        Get parameter groups with different learning rates for different layers
        
        Returns:
            List of parameter groups for optimizer
        """
        if not self.layer_wise_lr:
            return [{'params': self.parameters(), 'lr': self.learning_rate}]
        
        param_groups = []
        base_lr = self.learning_rate
        
        # Classifier gets full learning rate
        param_groups.append({
            'params': list(self.classifier.parameters()),
            'lr': base_lr,
            'name': 'classifier'
        })
        
        # Attention gets full learning rate
        if hasattr(self, 'attention'):
            param_groups.append({
                'params': list(self.attention.parameters()),
                'lr': base_lr,
                'name': 'attention'
            })
        
        # Backbone layers get progressively smaller learning rates
        if hasattr(self.backbone, 'layer4'):  # ResNet structure
            backbone_components = [
                ('layer4', self.backbone.layer4),
                ('layer3', self.backbone.layer3),
                ('layer2', self.backbone.layer2),
                ('layer1', self.backbone.layer1),
                ('early_layers', [self.backbone.conv1, self.backbone.bn1])
            ]
        else:
            # For Sequential backbone
            backbone_layers = list(self.backbone.children())
            backbone_components = []
            for i, layer in enumerate(reversed(backbone_layers)):
                backbone_components.append((f'backbone_layer_{len(backbone_layers)-i-1}', layer))
        
        for i, (name, component) in enumerate(backbone_components):
            # Each deeper layer gets smaller learning rate
            layer_lr = base_lr * (self.discriminative_lr_factor ** (i + 1))
            
            if isinstance(component, list):
                params = []
                for comp in component:
                    params.extend(list(comp.parameters()))
            else:
                params = list(component.parameters())
            
            if params:  # Only add if there are parameters
                param_groups.append({
                    'params': params,
                    'lr': layer_lr,
                    'name': name
                })
        
        # Log learning rates
        logger.info("📊 Layer-wise learning rates:")
        for group in param_groups:
            logger.info(f"   {group['name']}: {group['lr']:.2e}")
        
        return param_groups
    
    def update_layer_wise_learning_rates(self, current_epoch: int):
        """Update learning rates when layers are unfrozen"""
        if not hasattr(self, 'trainer') or not self.trainer.optimizers:
            return
        
        optimizer = self.trainer.optimizers[0]
        
        # Get new parameter groups
        new_param_groups = self.get_layer_wise_learning_rates()
        
        # Update optimizer parameter groups
        optimizer.param_groups = new_param_groups
        
        logger.info(f"🔄 Updated layer-wise learning rates at epoch {current_epoch}")


class MedicalImageFineTuner(pl.LightningModule, ProgressiveTrainingMixin):
    """
    Enhanced model with progressive training for medical images
    """
    
    def __init__(self, 
                 base_model,
                 training_strategy: str = "progressive_unfreeze",
                 freeze_epochs: int = 5,
                 layer_wise_lr: bool = True,
                 warmup_epochs: int = 3,
                 **kwargs):
        
        super().__init__()
        
        # Copy attributes from base model
        for attr_name in dir(base_model):
            if not attr_name.startswith('_') and hasattr(base_model, attr_name):
                attr_value = getattr(base_model, attr_name)
                if not callable(attr_value):
                    setattr(self, attr_name, attr_value)
        
        # Copy model components
        self.backbone = base_model.backbone
        self.attention = base_model.attention if hasattr(base_model, 'attention') else None
        self.classifier = base_model.classifier
        
        # Copy loss and metrics
        self.base_loss_fn = base_model.base_loss_fn
        self.train_metrics = base_model.train_metrics
        self.val_metrics = base_model.val_metrics
        
        # Progressive training setup
        self.warmup_epochs = warmup_epochs
        self.setup_progressive_training(
            strategy=training_strategy,
            freeze_epochs=freeze_epochs,
            layer_wise_lr=layer_wise_lr
        )
        
        # Track training phase
        self.current_training_phase = "warmup"
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch"""
        current_epoch = self.current_epoch
        
        # Update training phase
        if current_epoch < self.warmup_epochs:
            self.current_training_phase = "warmup"
        elif current_epoch < self.freeze_epochs:
            self.current_training_phase = "frozen_backbone"
        else:
            self.current_training_phase = "progressive_unfreeze"
        
        # Apply progressive unfreezing
        self.progressive_unfreeze_callback(current_epoch)
        
        # Log current phase
        if current_epoch % 5 == 0:
            logger.info(f"🔄 Epoch {current_epoch}: Training phase = {self.current_training_phase}")
    
    def configure_optimizers(self):
        """Configure optimizer with layer-wise learning rates"""
        # Get parameter groups with layer-wise learning rates
        param_groups = self.get_layer_wise_learning_rates()
        
        # Use AdamW with different learning rates for different layers
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        if self.warmup_epochs > 0:
            # Warmup then cosine annealing
            def lr_lambda(epoch):
                if epoch < self.warmup_epochs:
                    # Linear warmup
                    return (epoch + 1) / self.warmup_epochs
                else:
                    # Cosine annealing
                    import math
                    progress = (epoch - self.warmup_epochs) / (self.trainer.max_epochs - self.warmup_epochs)
                    return 0.5 * (1 + math.cos(math.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs,
                eta_min=1e-6
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        
        if self.attention is not None:
            features = self.attention(features)
        
        logits = self.classifier(features)
        return logits


def create_progressive_training_strategies():
    """
    Define different progressive training strategies for medical images
    """
    strategies = {
        # Conservative approach: freeze backbone for longer
        "conservative": {
            "strategy": "progressive_unfreeze",
            "freeze_epochs": 10,
            "unfreeze_schedule": {
                0: 0,   # All frozen
                10: 1,  # Last layer
                20: 2,  # Last 2 layers
                30: 4,  # All layers
            },
            "layer_wise_lr": True,
            "discriminative_lr_factor": 0.1
        },
        
        # Aggressive approach: quick unfreezing
        "aggressive": {
            "strategy": "progressive_unfreeze", 
            "freeze_epochs": 3,
            "unfreeze_schedule": {
                0: 0,   # All frozen
                3: 1,   # Last layer
                6: 2,   # Last 2 layers
                9: 4,   # All layers
            },
            "layer_wise_lr": True,
            "discriminative_lr_factor": 0.2
        },
        
        # Balanced approach: moderate unfreezing
        "balanced": {
            "strategy": "progressive_unfreeze",
            "freeze_epochs": 5,
            "unfreeze_schedule": {
                0: 0,   # All frozen
                5: 1,   # Last layer
                10: 2,  # Last 2 layers
                15: 3,  # Last 3 layers
                20: 4,  # All layers
            },
            "layer_wise_lr": True,
            "discriminative_lr_factor": 0.15
        },
        
        # Medical-specific approach
        "medical_optimized": {
            "strategy": "progressive_unfreeze",
            "freeze_epochs": 8,
            "unfreeze_schedule": {
                0: 0,   # All frozen - learn domain-specific features first
                8: 1,   # High-level features
                15: 2,  # Mid-level features  
                25: 3,  # Low-level features
                35: 4,  # Fine-tune everything
            },
            "layer_wise_lr": True,
            "discriminative_lr_factor": 0.05  # Very different rates for medical
        }
    }
    
    return strategies


def apply_progressive_training(base_model, strategy_name: str = "balanced"):
    """
    Apply progressive training to a base model
    
    Args:
        base_model: The base model to enhance
        strategy_name: Name of strategy from create_progressive_training_strategies()
    
    Returns:
        Enhanced model with progressive training
    """
    strategies = create_progressive_training_strategies()
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    strategy_config = strategies[strategy_name]
    
    # Create enhanced model
    enhanced_model = MedicalImageFineTuner(
        base_model=base_model,
        **strategy_config
    )
    
    logger.info(f"🚀 Applied '{strategy_name}' progressive training strategy")
    
    return enhanced_model
