#!/usr/bin/env python3
"""
Enhanced Multi-Abnormality Classification Training
Advanced techniques for improved AUROC performance (>60%)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, average_precision_score, classification_report
)
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from scipy import ndimage

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Advanced preprocessing for CT images
class CTImagePreprocessor:
    """Enhanced CT image preprocessing with HU normalization and enhancement"""
    
    def __init__(self, 
                 hu_min=-1000, hu_max=400,  # Typical range for chest CT
                 target_size=(224, 224),
                 enhance_contrast=True,
                 use_clahe=True):
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.target_size = target_size
        self.enhance_contrast = enhance_contrast
        self.use_clahe = use_clahe
        
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def normalize_hu(self, image):
        """Normalize HU values to 0-1 range"""
        # Clip to meaningful range
        image = np.clip(image, self.hu_min, self.hu_max)
        # Normalize to 0-1
        image = (image - self.hu_min) / (self.hu_max - self.hu_min)
        return image
    
    def enhance_image(self, image):
        """Apply image enhancement techniques"""
        if self.enhance_contrast:
            # Convert to uint8 for CLAHE
            image_uint8 = (image * 255).astype(np.uint8)
            
            if self.use_clahe:
                image_uint8 = self.clahe.apply(image_uint8)
            
            # Convert back to float32
            image = image_uint8.astype(np.float32) / 255.0
        
        return image
    
    def apply_windowing(self, image, window_center=-500, window_width=1400):
        """Apply CT windowing for soft tissue visualization"""
        min_val = window_center - window_width // 2
        max_val = window_center + window_width // 2
        
        windowed = np.clip(image, min_val, max_val)
        windowed = (windowed - min_val) / (max_val - min_val)
        
        return windowed
    
    def preprocess(self, image):
        """Complete preprocessing pipeline"""
        # Normalize HU values
        image = self.normalize_hu(image)
        
        # Apply windowing
        image = self.apply_windowing(image * (self.hu_max - self.hu_min) + self.hu_min)
        
        # Enhance contrast
        image = self.enhance_image(image)
        
        return image

# Enhanced Dataset with advanced augmentations
class EnhancedCTSliceDataset(Dataset):
    """Enhanced dataset with advanced preprocessing and augmentation"""
    
    def __init__(self, 
                 slice_data_df, 
                 slice_dir, 
                 abnormality_cols,
                 is_training=True,
                 use_advanced_aug=True,
                 target_size=(224, 224)):
        
        self.slice_data_df = slice_data_df.reset_index(drop=True)
        self.slice_dir = Path(slice_dir)
        self.abnormality_cols = abnormality_cols
        self.is_training = is_training
        self.use_advanced_aug = use_advanced_aug
        self.target_size = target_size
        
        # Initialize preprocessor
        self.preprocessor = CTImagePreprocessor(target_size=target_size)
        
        # Create slice paths
        self.slice_paths = []
        for _, row in self.slice_data_df.iterrows():
            volume_name = row['VolumeName']
            split = row['split']
            
            # Find slice files for this volume
            volume_slices = list(self.slice_dir.glob(f"slices/{split}/*{volume_name}*_slice_*.npy"))
            self.slice_paths.extend(volume_slices)
        
        logger.info(f"Dataset created with {len(self.slice_paths)} slices")
        
        # Advanced augmentations using albumentations
        if self.is_training and self.use_advanced_aug:
            self.transform = A.Compose([
                # Geometric transformations
                A.Rotate(limit=20, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.7),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                
                # Intensity transformations
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                
                # Morphological transformations
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
                
                # Resize to target size
                A.Resize(height=target_size[0], width=target_size[1], p=1.0),
                
                # Normalization for pretrained models
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Validation/test transforms
            self.transform = A.Compose([
                A.Resize(height=target_size[0], width=target_size[1], p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        # Create mapping from slice path to labels
        self.slice_to_labels = {}
        for _, row in self.slice_data_df.iterrows():
            volume_name = row['VolumeName']
            labels = row[self.abnormality_cols].values.astype(np.float32)
            self.slice_to_labels[volume_name] = labels
    
    def __len__(self):
        return len(self.slice_paths)
    
    def __getitem__(self, idx):
        slice_path = self.slice_paths[idx]
        
        # Extract volume name from slice path
        slice_filename = slice_path.name
        # Remove _slice_X.npy suffix and reconstruct volume name
        # Example: train_1_a_1.nii.gz_slice_45.npy -> train_1_a_1.nii.gz
        if '_slice_' in slice_filename:
            volume_name = slice_filename.split('_slice_')[0]
            # Handle case where .nii.gz is already in the name
            if not volume_name.endswith('.nii.gz'):
                volume_name += '.nii.gz'
        else:
            # Fallback to original logic
            volume_name = '_'.join(slice_filename.split('_')[:-2]) + '.nii.gz'
        
        # Load slice
        try:
            slice_data = np.load(slice_path)
        except Exception as e:
            logger.warning(f"Failed to load {slice_path}: {e}")
            # Return dummy data in case of failure
            slice_data = np.zeros((224, 224), dtype=np.float32)
        
        # Preprocess the slice
        slice_data = self.preprocessor.preprocess(slice_data)
        
        # Convert to 3-channel for pretrained models
        if len(slice_data.shape) == 2:
            slice_data = np.stack([slice_data, slice_data, slice_data], axis=-1)
        
        # Apply augmentations
        transformed = self.transform(image=slice_data)
        image = transformed['image']
        
        # Get labels
        if volume_name in self.slice_to_labels:
            labels = self.slice_to_labels[volume_name]
        else:
            raise KeyError(f"No labels found for {volume_name}. Available volumes: {list(self.slice_to_labels.keys())[:5]}...")
        
        return image, torch.tensor(labels, dtype=torch.float32)

# Advanced loss functions
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate focal loss for each class
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification"""
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
    
    def forward(self, x, y):
        # Calculate probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Calculate asymmetric loss
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Apply asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.gamma_pos > 0:
                los_pos *= (1 - xs_pos) ** self.gamma_pos
            if self.gamma_neg > 0:
                los_neg *= xs_pos ** self.gamma_neg
        
        loss = los_pos + los_neg
        return -loss.mean()

# Enhanced Model Architecture
class EnhancedMultiAbnormalityModel(pl.LightningModule):
    """Enhanced model with attention mechanisms and advanced training techniques"""
    
    def __init__(self, 
                 model_name='resnet50',
                 num_classes=18,
                 dropout_rate=0.3,
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 use_attention='se',
                 use_multiscale=True,
                 use_focal_loss=True,
                 use_label_smoothing=True,
                 label_smoothing_factor=0.1,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 class_weights=None,
                 use_cosine_annealing=True,
                 use_mixup=True,
                 mixup_alpha=0.2):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cosine_annealing = use_cosine_annealing
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_label_smoothing = use_label_smoothing
        self.label_smoothing_factor = label_smoothing_factor
        
        # Build backbone
        self.backbone = self._build_backbone(model_name)
        
        # Get feature dimension
        if 'resnet' in model_name:
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif 'efficientnet' in model_name:
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        
        # Attention mechanism
        if use_attention == 'se':
            self.attention = SEBlock(feature_dim)
        elif use_attention == 'cbam':
            self.attention = CBAMBlock(feature_dim)
        else:
            self.attention = nn.Identity()
        
        # Multi-scale features
        if use_multiscale:
            self.multiscale_features = MultiScaleFeatures(feature_dim)
            classifier_input_dim = feature_dim * 2
        else:
            self.multiscale_features = nn.Identity()
            classifier_input_dim = feature_dim
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(classifier_input_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(classifier_input_dim // 2),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(classifier_input_dim // 2, num_classes)
        )
        
        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            if class_weights is not None:
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        
        # Metrics
        self.train_metrics = self._create_metrics('train')
        self.val_metrics = self._create_metrics('val')
    
    def _build_backbone(self, model_name):
        """Build backbone model"""
        if model_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
        elif model_name == 'resnet101':
            backbone = models.resnet101(pretrained=True)
        elif model_name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return backbone
    
    def _create_metrics(self, stage):
        """Create torchmetrics for evaluation"""
        return nn.ModuleDict({
            'auroc': torchmetrics.AUROC(task='multilabel', num_labels=self.num_classes, average='macro'),
            'f1': torchmetrics.F1Score(task='multilabel', num_labels=self.num_classes, average='macro'),
            'precision': torchmetrics.Precision(task='multilabel', num_labels=self.num_classes, average='macro'),
            'recall': torchmetrics.Recall(task='multilabel', num_labels=self.num_classes, average='macro'),
            'accuracy': torchmetrics.Accuracy(task='multilabel', num_labels=self.num_classes, average='macro')
        })
    
    def mixup_data(self, x, y, alpha=1.0):
        """Apply MixUp augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y, lam
    
    def label_smoothing_loss(self, pred, target, smoothing=0.1):
        """Apply label smoothing"""
        confidence = 1.0 - smoothing
        smooth_positive = smoothing / 2
        smooth_negative = smoothing / 2
        
        # For multilabel, apply smoothing differently
        target_smooth = target * confidence + smooth_positive
        target_smooth = target_smooth * target + (1 - target) * smooth_negative
        
        return F.binary_cross_entropy_with_logits(pred, target_smooth)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        if hasattr(self, 'attention') and not isinstance(self.attention, nn.Identity):
            # For attention mechanisms that expect 4D input
            if len(features.shape) == 2:
                # Reshape for attention (assuming global avg pooling was applied)
                features = features.unsqueeze(-1).unsqueeze(-1)
            features = self.attention(features)
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # Multi-scale features
        features = self.multiscale_features(features)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Apply MixUp augmentation
        if self.use_mixup and self.training:
            x, y, lam = self.mixup_data(x, y, self.mixup_alpha)
        
        # Forward pass
        logits = self(x)
        
        # Calculate loss
        if self.use_label_smoothing:
            loss = self.label_smoothing_loss(logits, y, self.label_smoothing_factor)
        else:
            loss = self.criterion(logits, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update metrics
        probs = torch.sigmoid(logits)
        for metric_name, metric_fn in self.train_metrics.items():
            try:
                metric_fn.update(probs, y.int())
            except:
                pass
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Update metrics
        probs = torch.sigmoid(logits)
        for metric_name, metric_fn in self.val_metrics.items():
            try:
                metric_fn.update(probs, y.int())
            except:
                pass
        
        return loss
    
    def on_training_epoch_end(self):
        # Compute and log metrics
        for metric_name, metric_fn in self.train_metrics.items():
            try:
                value = metric_fn.compute()
                self.log(f'train_{metric_name}', value, on_epoch=True)
                metric_fn.reset()
            except:
                pass
    
    def on_validation_epoch_end(self):
        # Compute and log metrics
        for metric_name, metric_fn in self.val_metrics.items():
            try:
                value = metric_fn.compute()
                self.log(f'val_{metric_name}', value, on_epoch=True, prog_bar=True)
                metric_fn.reset()
            except:
                pass
    
    def configure_optimizers(self):
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        if self.use_cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer

# Attention Mechanisms
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.excitation(x).view(b, c, 1, 1)
        return x * y

class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class MultiScaleFeatures(nn.Module):
    """Multi-scale feature extraction"""
    def __init__(self, feature_dim):
        super(MultiScaleFeatures, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Assume x is already flattened features
        if len(x.shape) != 2:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        
        # Original features
        original = x
        
        # Multi-scale processing (simple version)
        # In practice, this would be more sophisticated
        expanded = x.unsqueeze(1)  # Add channel dim
        processed = self.conv1d(expanded).squeeze(1)
        
        # Concatenate original and processed features
        combined = torch.cat([original, processed], dim=1)
        
        return combined

def create_class_weights(labels_df, abnormality_cols):
    """Create class weights for imbalanced dataset"""
    weights = []
    for col in abnormality_cols:
        pos_count = labels_df[col].sum()
        neg_count = len(labels_df) - pos_count
        
        # Inverse frequency weighting
        pos_weight = len(labels_df) / (2 * pos_count) if pos_count > 0 else 1.0
        weights.append(pos_weight)
    
    return weights

def train_model_enhanced(args):
    """Enhanced training function with all improvements"""
    logger.info("🚀 Starting enhanced multi-abnormality classification training...")
    
    # Set random seeds
    pl.seed_everything(args.seed, workers=True)
    
    # Get abnormality columns
    abnormality_cols = [col for col in args.train_data.columns if col not in ['VolumeName', 'split']]
    logger.info(f"Training on {len(abnormality_cols)} abnormalities: {abnormality_cols}")
    
    # Create datasets
    train_dataset = EnhancedCTSliceDataset(
        args.train_data, 
        args.slice_dir, 
        abnormality_cols,
        is_training=True,
        use_advanced_aug=args.use_advanced_augmentation
    )
    
    val_dataset = EnhancedCTSliceDataset(
        args.val_data,
        args.slice_dir,
        abnormality_cols,
        is_training=False,
        use_advanced_aug=False
    )
    
    # Create class weights
    class_weights = None
    if args.use_class_weights:
        all_data = pd.concat([args.train_data, args.val_data])
        class_weights = create_class_weights(all_data, abnormality_cols)
        logger.info(f"Class weights: {[f'{w:.3f}' for w in class_weights]}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Create model
    model = EnhancedMultiAbnormalityModel(
        model_name=args.model_name,
        num_classes=len(abnormality_cols),
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_attention=args.use_attention,
        use_multiscale=args.use_multiscale,
        use_focal_loss=args.use_focal_loss,
        use_label_smoothing=args.use_label_smoothing,
        label_smoothing_factor=args.label_smoothing_factor,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        class_weights=class_weights,
        use_cosine_annealing=args.use_cosine_annealing,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f'fold_{args.fold_num}_best_auroc',
            monitor='val_auroc',
            mode='max',
            save_top_k=1,
            save_last=True
        ),
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f'fold_{args.fold_num}_best_f1',
            monitor='val_f1',
            mode='max',
            save_top_k=1
        ),
        EarlyStopping(
            monitor='val_auroc',
            mode='max',
            patience=args.early_stopping_patience,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Logger
    logger_tb = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f'fold_{args.fold_num}'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger_tb,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus if torch.cuda.is_available() else None,
        precision='16-mixed' if args.use_mixed_precision else 32,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=True,
        benchmark=False  # For reproducibility
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model and evaluate
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        model = EnhancedMultiAbnormalityModel.load_from_checkpoint(best_model_path)
        logger.info(f"Loaded best model from {best_model_path}")
    
    # Evaluate on validation set
    model.eval()
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            
            logits = model(x)
            probs = torch.sigmoid(logits)
            
            val_predictions.append(probs.cpu().numpy())
            val_targets.append(y.cpu().numpy())
    
    val_predictions = np.vstack(val_predictions)
    val_targets = np.vstack(val_targets)
    
    # Calculate metrics
    metrics = {}
    
    # AUROC
    try:
        auroc_macro = roc_auc_score(val_targets, val_predictions, average='macro')
        auroc_micro = roc_auc_score(val_targets, val_predictions, average='micro')
        metrics['auroc_macro'] = auroc_macro
        metrics['auroc_micro'] = auroc_micro
    except:
        metrics['auroc_macro'] = 0.5
        metrics['auroc_micro'] = 0.5
    
    # Other metrics (using threshold of 0.5)
    val_predictions_binary = (val_predictions > args.decision_threshold).astype(int)
    
    try:
        metrics['f1_macro'] = f1_score(val_targets, val_predictions_binary, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(val_targets, val_predictions_binary, average='micro', zero_division=0)
        metrics['precision_macro'] = precision_score(val_targets, val_predictions_binary, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(val_targets, val_predictions_binary, average='macro', zero_division=0)
        metrics['accuracy'] = accuracy_score(val_targets, val_predictions_binary)
    except:
        metrics.update({
            'f1_macro': 0, 'f1_micro': 0,
            'precision_macro': 0, 'recall_macro': 0,
            'accuracy': 0
        })
    
    # Log final metrics
    logger.info(f"✅ Fold {args.fold_num} Results:")
    logger.info(f"   AUROC (macro): {metrics['auroc_macro']:.4f}")
    logger.info(f"   F1 (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"   Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"   Recall (macro): {metrics['recall_macro']:.4f}")
    logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
    
    return model, trainer, metrics