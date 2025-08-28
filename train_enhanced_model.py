#!/usr/bin/env python3
"""
Enhanced Multi-Abnormality Classification Model
Incorporates novel techniques: attention mechanisms, advanced augmentations, 
multi-scale training, and ensemble methods
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torchmetrics
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from PIL import Image
import cv2

# Import our custom modules
from enhanced_augmentations import AdvancedCTAugmentations, MixupCutmix, MixupCutmixLoss
from attention_mechanisms import create_attention_module, AttentionAggregator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedCTSliceDataset(Dataset):
    """Enhanced dataset with advanced augmentations and multi-scale loading"""
    
    def __init__(self, 
                 slice_df: pd.DataFrame,
                 data_root: str,
                 multi_abnormality_df: pd.DataFrame,
                 image_size: int = 224,
                 augment: bool = True,
                 use_advanced_aug: bool = True,
                 multi_scale_training: bool = True):
        
        self.slice_df = slice_df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.augment = augment
        self.use_advanced_aug = use_advanced_aug
        self.multi_scale_training = multi_scale_training
        self.image_size = image_size
        
        # Merge with abnormality labels
        self.slice_df = self._merge_labels(slice_df, multi_abnormality_df)
        
        # Define abnormality classes
        self.abnormality_classes = [
            "Cardiomegaly", "Hiatal hernia", "Atelectasis", "Pulmonary fibrotic sequela",
            "Peribronchial thickening", "Interlobular septal thickening", "Medical material",
            "Pericardial effusion", "Lymphadenopathy", "Lung nodule", "Pleural effusion",
            "Consolidation", "Lung opacity", "Mosaic attenuation pattern", "Bronchiectasis",
            "Emphysema", "Arterial wall calcification", "Coronary artery wall calcification"
        ]
        
        # Calculate class weights
        self.class_weights = self._calculate_class_weights()
        
        # Multi-scale sizes for training
        self.scale_sizes = [192, 224, 256, 288] if multi_scale_training else [image_size]
        
        # Setup augmentations
        self.setup_transforms()
        
        logger.info(f"Dataset initialized with {len(self.slice_df)} slices")
        self._print_class_distribution()
    
    def _merge_labels(self, slice_df: pd.DataFrame, multi_abnormality_df: pd.DataFrame) -> pd.DataFrame:
        """Merge slice data with abnormality labels"""
        # Extract volume name from slice data
        slice_df['volume_name'] = slice_df['slice_id'].str.split('_slice_').str[0]
        
        # Merge with labels
        merged_df = slice_df.merge(
            multi_abnormality_df, 
            left_on='volume_name', 
            right_on='VolumeName', 
            how='left'
        )
        
        # Fill missing labels with 0
        for class_name in self.abnormality_classes:
            if class_name in merged_df.columns:
                merged_df[class_name] = merged_df[class_name].fillna(0).astype(int)
        
        return merged_df
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data"""
        weights = []
        for class_name in self.abnormality_classes:
            if class_name in self.slice_df.columns:
                pos_count = self.slice_df[class_name].sum()
                neg_count = len(self.slice_df) - pos_count
                weight = neg_count / (pos_count + 1e-6)  # Add small epsilon
                weights.append(weight)
            else:
                weights.append(1.0)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _print_class_distribution(self):
        """Print class distribution for debugging"""
        logger.info("Class Distribution:")
        for class_name in self.abnormality_classes:
            if class_name in self.slice_df.columns:
                count = self.slice_df[class_name].sum()
                prevalence = count / len(self.slice_df)
                logger.info(f"  {class_name:25s}: {count:4d} ({prevalence:5.3f})")
    
    def setup_transforms(self):
        """Setup augmentation transforms"""
        if self.augment and self.use_advanced_aug:
            # Advanced CT-specific augmentations
            self.augment_fn = AdvancedCTAugmentations(
                rotation_range=15.0,
                zoom_range=(0.9, 1.1),
                brightness_range=0.1,
                contrast_range=(0.9, 1.1),
                elastic_deformation=True,
                ct_specific_aug=True
            )
        
        # Standard transforms
        self.base_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.slice_df)
    
    def __getitem__(self, idx):
        row = self.slice_df.iloc[idx]
        
        # Load slice
        slice_path = self.data_root / row['file_path']
        if not slice_path.exists():
            logger.error(f"Slice file not found: {slice_path}")
            slice_data = np.zeros((self.image_size, self.image_size))
        else:
            slice_data = np.load(slice_path)
        
        # Ensure slice is 2D
        if len(slice_data.shape) > 2:
            slice_data = slice_data.squeeze()
        
        # Multi-scale training
        if self.multi_scale_training and self.augment:
            target_size = random.choice(self.scale_sizes)
        else:
            target_size = self.image_size
        
        # Resize if needed
        if slice_data.shape[0] != target_size or slice_data.shape[1] != target_size:
            slice_data = cv2.resize(slice_data, (target_size, target_size))
        
        # Convert to 3-channel for pretrained models
        if len(slice_data.shape) == 2:
            slice_data = np.stack([slice_data] * 3, axis=-1)
        
        # Normalize to [0, 1] if needed
        if slice_data.max() > 1.0:
            slice_data = slice_data / 255.0
        
        # Apply advanced augmentations
        if self.augment and self.use_advanced_aug:
            slice_data = self.augment_fn(slice_data)
        
        # Convert to tensor
        slice_tensor = torch.from_numpy(slice_data.transpose(2, 0, 1)).float()
        
        # Apply normalization
        slice_tensor = self.base_transform(slice_tensor)
        
        # Get labels
        labels = torch.tensor([
            row.get(class_name, 0) for class_name in self.abnormality_classes
        ], dtype=torch.float32)
        
        return {
            'image': slice_tensor,
            'labels': labels,
            'slice_id': row['slice_id'],
            'volume_name': row.get('volume_name', ''),
            'image_size': target_size
        }


class EnhancedMultiAbnormalityModel(pl.LightningModule):
    """Enhanced model with attention mechanisms and advanced training techniques"""
    
    def __init__(self, 
                 model_name: str = "resnet50",
                 num_classes: int = 18,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 class_weights: torch.Tensor = None,
                 dropout_rate: float = 0.3,
                 attention_type: str = "cbam",
                 use_mixup: bool = True,
                 use_progressive_resizing: bool = True,
                 use_label_smoothing: bool = True):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.use_mixup = use_mixup
        self.use_progressive_resizing = use_progressive_resizing
        
        # Build enhanced model
        self.backbone = self._build_backbone(model_name)
        self.attention = self._build_attention(attention_type)
        self.classifier = self._build_classifier(dropout_rate)
        
        # Loss functions
        self.base_loss_fn = self._build_loss_function(use_label_smoothing)
        if use_mixup:
            self.mixup_cutmix = MixupCutmix(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5)
            self.mixup_loss_fn = MixupCutmixLoss(self.base_loss_fn)
        
        # Metrics
        self.train_metrics = self._create_metrics('train')
        self.val_metrics = self._create_metrics('val')
        
        # Progressive resizing schedule
        self.current_epoch_size = 224
        self.size_schedule = {0: 192, 20: 224, 40: 256, 60: 288} if use_progressive_resizing else {}
    
    def _build_backbone(self, model_name: str):
        """Build backbone model"""
        import torchvision.models as models
        
        if model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            self.feature_dim = 2048
        elif model_name == "resnet101":
            model = models.resnet101(pretrained=True)
            self.feature_dim = 2048
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove classifier
        if "resnet" in model_name:
            model = nn.Sequential(*list(model.children())[:-2])  # Remove avgpool and fc
        elif "efficientnet" in model_name:
            model.classifier = nn.Identity()
        
        return model
    
    def _build_attention(self, attention_type: str):
        """Build attention mechanism"""
        if attention_type == "none":
            return nn.Identity()
        
        return create_attention_module(attention_type, self.feature_dim)
    
    def _build_classifier(self, dropout_rate: float):
        """Build classification head with attention pooling"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 4),
            nn.Linear(256, self.num_classes)
        )
    
    def _build_loss_function(self, use_label_smoothing: bool):
        """Build loss function"""
        if use_label_smoothing:
            # Label smoothing for better generalization
            return nn.BCEWithLogitsLoss(
                pos_weight=self.class_weights,
                label_smoothing=0.1
            )
        else:
            return nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
    
    def _create_metrics(self, prefix: str):
        """Create torchmetrics for evaluation"""
        return nn.ModuleDict({
            'auroc': torchmetrics.AUROC(task='multilabel', num_labels=self.num_classes, average='macro'),
            'f1': torchmetrics.F1Score(task='multilabel', num_labels=self.num_classes, average='macro'),
            'precision': torchmetrics.Precision(task='multilabel', num_labels=self.num_classes, average='macro'),
            'recall': torchmetrics.Recall(task='multilabel', num_labels=self.num_classes, average='macro'),
            'accuracy': torchmetrics.Accuracy(task='multilabel', num_labels=self.num_classes, average='micro')
        })
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attended_features = self.attention(features)
        
        # Classify
        logits = self.classifier(attended_features)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        # Apply mixup/cutmix if enabled
        if self.use_mixup and self.training:
            images, labels_a, labels_b, lam = self.mixup_cutmix(images, labels)
            logits = self(images)
            loss = self.mixup_loss_fn(logits, labels_a, labels_b, lam)
            probs = torch.sigmoid(logits)
            
            # Update metrics with mixed labels
            mixed_labels = lam * labels_a + (1 - lam) * labels_b
            self.train_metrics['auroc'].update(probs, mixed_labels.int())
            self.train_metrics['f1'].update((probs > 0.5).float(), mixed_labels.int())
        else:
            logits = self(images)
            loss = self.base_loss_fn(logits, labels)
            probs = torch.sigmoid(logits)
            
            # Update metrics
            self.train_metrics['auroc'].update(probs, labels.int())
            self.train_metrics['f1'].update((probs > 0.5).float(), labels.int())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        logits = self(images)
        loss = self.base_loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        
        # Update metrics
        for metric_name, metric in self.val_metrics.items():
            if metric_name in ['auroc']:
                metric.update(probs, labels.int())
            else:
                metric.update((probs > 0.5).float(), labels.int())
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        # Log training metrics
        for metric_name, metric in self.train_metrics.items():
            value = metric.compute()
            self.log(f'train_{metric_name}', value, on_epoch=True)
            metric.reset()
        
        # Update progressive resizing
        if self.use_progressive_resizing and self.current_epoch in self.size_schedule:
            self.current_epoch_size = self.size_schedule[self.current_epoch]
            logger.info(f"Progressive resizing: updating to {self.current_epoch_size}x{self.current_epoch_size}")
    
    def on_validation_epoch_end(self):
        # Log validation metrics
        metrics = {}
        for metric_name, metric in self.val_metrics.items():
            value = metric.compute()
            self.log(f'val_{metric_name}', value, on_epoch=True)
            metrics[metric_name] = value.item()
            metric.reset()
        
        # Log combined score for model selection
        combined_score = (
            metrics.get('auroc', 0) * 0.4 + 
            metrics.get('f1', 0) * 0.3 + 
            metrics.get('precision', 0) * 0.15 + 
            metrics.get('recall', 0) * 0.15
        )
        self.log('val_combined_score', combined_score, on_epoch=True)
    
    def configure_optimizers(self):
        # Use AdamW with cosine annealing
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
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


def load_data(data_dir: str, slice_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training data and labels"""
    # Load slice data
    train_slices_df = pd.read_csv(Path(slice_dir) / "splits" / "train_slices.csv")
    val_slices_df = pd.read_csv(Path(slice_dir) / "splits" / "valid_slices.csv")
    
    # Load abnormality labels
    multi_labels_df = pd.read_csv(Path(data_dir) / "multi_abnormality_labels.csv")
    
    logger.info(f"Loaded {len(train_slices_df)} training slices")
    logger.info(f"Loaded {len(val_slices_df)} validation slices")
    logger.info(f"Loaded {len(multi_labels_df)} abnormality labels")
    
    return train_slices_df, val_slices_df, multi_labels_df


def create_enhanced_data_loaders(train_slices_df, val_slices_df, multi_labels_df, 
                                slice_dir: str, batch_size: int = 32, num_workers: int = 4):
    """Create enhanced data loaders with advanced augmentations"""
    
    # Create datasets
    train_dataset = EnhancedCTSliceDataset(
        slice_df=train_slices_df,
        data_root=slice_dir,
        multi_abnormality_df=multi_labels_df,
        augment=True,
        use_advanced_aug=True,
        multi_scale_training=True
    )
    
    val_dataset = EnhancedCTSliceDataset(
        slice_df=val_slices_df,
        data_root=slice_dir,
        multi_abnormality_df=multi_labels_df,
        augment=False,
        use_advanced_aug=False,
        multi_scale_training=False
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, train_dataset.class_weights


def train_enhanced_model(args):
    """Main training function with enhanced techniques"""
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    train_slices_df, val_slices_df, multi_labels_df = load_data(args.data_dir, args.slice_dir)
    
    # Create enhanced data loaders
    train_loader, val_loader, class_weights = create_enhanced_data_loaders(
        train_slices_df, val_slices_df, multi_labels_df, 
        args.slice_dir, args.batch_size, args.num_workers
    )
    
    # Initialize base enhanced model
    base_model = EnhancedMultiAbnormalityModel(
        model_name=args.model_name,
        num_classes=18,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        dropout_rate=args.dropout_rate,
        attention_type=getattr(args, 'attention_type', 'cbam'),
        use_mixup=getattr(args, 'use_mixup', True),
        use_progressive_resizing=getattr(args, 'use_progressive_resizing', True),
        use_label_smoothing=getattr(args, 'use_label_smoothing', True)
    )
    
    # Apply progressive training strategy
    progressive_strategy = getattr(args, 'progressive_strategy', 'balanced')
    if progressive_strategy != 'none':
        logger.info(f"🧊 Applying progressive training strategy: {progressive_strategy}")
        from progressive_training import apply_progressive_training
        model = apply_progressive_training(base_model, progressive_strategy)
    else:
        logger.info("🔥 Using full fine-tuning (no freezing)")
        model = base_model
    
    # Enhanced callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='enhanced-multi-abnormality-{epoch:02d}-{val_combined_score:.3f}',
        monitor='val_combined_score',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_combined_score',
        patience=args.early_stopping_patience,
        mode='max',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Trainer with advanced features
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16,  # Mixed precision training
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=pl.loggers.TensorBoardLogger(args.log_dir, name="enhanced_model"),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # Allow for faster training
        benchmark=True  # Optimize for consistent input sizes
    )
    
    # Train model
    logger.info("🚀 Starting enhanced training...")
    trainer.fit(model, train_loader, val_loader)
    
    logger.info("🎉 Enhanced training completed!")
    
    return model, trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Multi-Abnormality Classification")
    parser.add_argument("--data-dir", default="./ct_rate_data", help="Data directory")
    parser.add_argument("--slice-dir", default="./ct_rate_2d", help="Slice directory")
    parser.add_argument("--model-name", default="resnet50", help="Model backbone")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--dropout-rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", default="./logs", help="Log directory")
    parser.add_argument("--attention-type", default="cbam", help="Attention mechanism type")
    parser.add_argument("--use-mixup", action="store_true", default=True, help="Use mixup/cutmix")
    parser.add_argument("--use-progressive-resizing", action="store_true", default=True, help="Use progressive resizing")
    parser.add_argument("--use-label-smoothing", action="store_true", default=True, help="Use label smoothing")
    
    args = parser.parse_args()
    
    # Train enhanced model
    model, trainer = train_enhanced_model(args)
