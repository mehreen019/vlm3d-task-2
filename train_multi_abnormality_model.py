#!/usr/bin/env python3
"""
Multi-Abnormality Classification Training for VLM3D Task 2
Integrates with existing CT-RATE downloader and slice extractor pipeline
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
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
import random
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Advanced Attention Mechanisms
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block - Deterministic Version"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction

        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Use deterministic global average pooling
        y = torch.mean(x, dim=[2, 3])  # Global average pooling
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y

class CBAM(nn.Module):
    """Convolutional Block Attention Module - Deterministic Version"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.channels = channels
        self.reduction = reduction

        # Channel attention - use deterministic pooling
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention - deterministic implementation
        # Use mean and max operations instead of AdaptiveMaxPool2d
        avg_out = torch.mean(x, dim=[2, 3], keepdim=True)  # Global average pooling
        max_out = torch.max(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)[0]
        max_out = max_out.view(x.size(0), x.size(1), 1, 1)

        ca = self.channel_attention(avg_out + max_out)
        ca = torch.sigmoid(ca)

        x = x * ca

        # Spatial attention - deterministic
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        sa_input = torch.cat([avg_out, max_out], dim=1)

        sa = self.spatial_attention(sa_input)
        x = x * sa

        return x

# Advanced Loss Functions
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for Multi-Label Classification - Enhanced for Over-prediction"""
    def __init__(self, gamma_neg=6, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg  # Increased from 4 to 6 to penalize false positives more
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            los_pos *= one_sided_w
            los_neg *= one_sided_w

        loss = los_pos + los_neg
        return -loss.sum()

# Data Augmentation Functions
def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation for multi-label classification"""
    batch_size = x.size(0)
    indices = torch.randperm(batch_size)

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    # Random box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y, y[indices], lam

def rand_bbox(size, lam):
    """Generate random bounding box"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class CTSliceDataset(Dataset):
    """Dataset for loading CT slices with multi-label abnormalities"""
    
    def __init__(self,
                 slice_df: pd.DataFrame,
                 data_root: str,
                 multi_abnormality_df: pd.DataFrame = None,
                 transform=None,
                 augment: bool = False,
                 use_advanced_aug: bool = False,
                 cutmix_prob: float = 0.5):
        
        self.slice_df = slice_df.copy()
        self.data_root = Path(data_root)
        self.transform = transform
        self.augment = augment
        self.use_advanced_aug = use_advanced_aug
        self.cutmix_prob = cutmix_prob

        # 18 abnormality classes from CT-RATE
        self.abnormality_classes = [
            "Cardiomegaly", "Hiatal hernia", "Atelectasis", "Pulmonary fibrotic sequela",
            "Peribronchial thickening", "Interlobular septal thickening", "Medical material",
            "Pericardial effusion", "Lymphadenopathy", "Lung nodule", "Pleural effusion",
            "Consolidation", "Lung opacity", "Mosaic attenuation pattern", "Bronchiectasis",
            "Emphysema", "Arterial wall calcification", "Coronary artery wall calcification"
        ]

        
        # Load and merge multi-abnormality labels if provided
        if multi_abnormality_df is not None:
            self.slice_df = self._merge_labels(self.slice_df, multi_abnormality_df)
        
        # Ensure all label columns exist with default 0
        for col in self.abnormality_classes:
            if col not in self.slice_df.columns:
                self.slice_df[col] = 0
        
        # Calculate class weights for imbalanced data
        self.class_weights = self._calculate_class_weights()
        
        logger.info(f"Loaded dataset with {len(self.slice_df)} slices")
        self._print_class_distribution()
    
    def _merge_labels(self, slice_df: pd.DataFrame, multi_abnormality_df: pd.DataFrame) -> pd.DataFrame:
        """Merge slice metadata with multi-abnormality labels"""
        # The key is to match by VolumeName from slice metadata to multi-abnormality labels
        
        logger.info(f"Multi-abnormality DF columns: {list(multi_abnormality_df.columns)}")
        logger.info(f"Sample volume names in labels: {multi_abnormality_df['VolumeName'].head().tolist()}")
        logger.info(f"Sample volume names in slices: {slice_df['volume_name'].head().tolist()}")
        
        # Create a mapping from volume name to labels
        volume_labels = {}
        for _, row in multi_abnormality_df.iterrows():
            volume_name = row['VolumeName']
            labels = {}
            for class_name in self.abnormality_classes:
                if class_name in row.index:
                    labels[class_name] = int(row[class_name])
                else:
                    labels[class_name] = 0
            volume_labels[volume_name] = labels
        
        # Check for matches
        slice_volumes = set(slice_df['volume_name'])
        label_volumes = set(volume_labels.keys())
        matched_volumes = slice_volumes.intersection(label_volumes)
        logger.info(f"Slice volumes: {len(slice_volumes)}, Label volumes: {len(label_volumes)}, Matched: {len(matched_volumes)}")
        
        # Add labels to slice dataframe
        for class_name in self.abnormality_classes:
            slice_df[class_name] = slice_df['volume_name'].map(
                lambda x: volume_labels.get(x, {}).get(class_name, 0)
            )
        
        logger.info(f"Merged labels for {len(volume_labels)} volumes")
        return slice_df
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data"""
        weights = []
        for class_name in self.abnormality_classes:
            pos_count = self.slice_df[class_name].sum()
            neg_count = len(self.slice_df) - pos_count
            
            if pos_count == 0:
                weight = 1.0
            else:
                weight = neg_count / pos_count
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _print_class_distribution(self):
        """Print class distribution for debugging"""
        logger.info("Class Distribution:")
        for class_name in self.abnormality_classes:
            count = self.slice_df[class_name].sum()
            prevalence = count / len(self.slice_df)
            logger.info(f"  {class_name:18s}: {count:4d} ({prevalence:5.3f})")
    
    def __len__(self):
        return len(self.slice_df)
    
    def __getitem__(self, idx):
        row = self.slice_df.iloc[idx]
        
        # Load slice
        slice_path = self.data_root / row['file_path']
        if not slice_path.exists():
            logger.error(f"Slice file not found: {slice_path}")
            # Return zeros if file not found
            slice_data = np.zeros((224, 224))
        else:
            slice_data = np.load(slice_path)
        
        # Ensure slice is 2D
        if len(slice_data.shape) > 2:
            slice_data = slice_data.squeeze()
        
        # Convert to 3-channel for pretrained models
        if len(slice_data.shape) == 2:
            slice_data = np.stack([slice_data] * 3, axis=-1)
        
        # Normalize to [0, 1] if needed
        if slice_data.max() > 1.0:
            slice_data = slice_data / 255.0
        
        # Apply transforms
        if self.transform:
            # Convert to PIL for torchvision transforms
            from PIL import Image
            slice_data = (slice_data * 255).astype(np.uint8)
            slice_data = Image.fromarray(slice_data)
            slice_data = self.transform(slice_data)
        else:
            slice_data = torch.from_numpy(slice_data.transpose(2, 0, 1)).float()
        
        # Get labels
        labels = torch.tensor([
            row[class_name] for class_name in self.abnormality_classes
        ], dtype=torch.float32)
        
        return {
            'image': slice_data,
            'labels': labels,
            'slice_id': row['slice_id'],
            'volume_name': row['volume_name']
        }

class MultiAbnormalityModel(pl.LightningModule):
    """Multi-label classification model for abnormality detection"""
    
    def __init__(self,
                 model_name: str = "resnet50",
                 num_classes: int = 18,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 class_weights: torch.Tensor = None,
                 dropout_rate: float = 0.3,
                 freeze_backbone: bool = False,
                 use_attention: str = "none",  # "none", "se", "cbam"
                 use_multiscale: bool = False,
                 loss_type: str = "focal",  # "focal", "bce", "asl"
                 progressive_unfreeze: bool = False,
                 unfreeze_epoch: int = 10):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.freeze_backbone = freeze_backbone
        self.use_attention = use_attention
        self.use_multiscale = use_multiscale
        self.loss_type = loss_type
        self.progressive_unfreeze = progressive_unfreeze
        self.unfreeze_epoch = unfreeze_epoch
        self.current_epoch_num = 0

        # Build model components
        self.backbone = self._build_backbone(model_name)
        self.attention = self._build_attention(use_attention)
        self.classifier = self._build_classifier(dropout_rate)

        # Initialize loss function
        self.criterion = self._build_loss_function(loss_type)

        # Apply backbone freezing if requested
        if freeze_backbone and not progressive_unfreeze:
            self._freeze_backbone()
        
        # Metrics
        self.train_metrics = self._create_metrics('train')
        self.val_metrics = self._create_metrics('val')
        
        # Class names for logging
        self.class_names = [
            "Cardiomegaly", "Hiatal hernia", "Atelectasis", "Pulmonary fibrotic sequela",
            "Peribronchial thickening", "Interlobular septal thickening", "Medical material",
            "Pericardial effusion", "Lymphadenopathy", "Lung nodule", "Pleural effusion",
            "Consolidation", "Lung opacity", "Mosaic attenuation pattern", "Bronchiectasis",
            "Emphysema", "Arterial wall calcification", "Coronary artery wall calcification"
        ]
    
        def _build_backbone(self, model_name: str):
        """Build the backbone network with CT-CLIP integration"""
        if model_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 2048
        elif model_name == "resnet101":
            backbone = models.resnet101(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 2048
        elif model_name == "efficientnet_b0":
            # Try to load CT-CLIP weights if available
            backbone = models.efficientnet_b0(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])

            # Try to load CT-CLIP weights
            ctclip_paths = ['models/ctclip_classfine.pt', 'models/ctclip_vocabfine.pt']
            ctclip_loaded = False

            for ctclip_path in ctclip_paths:
                if os.path.exists(ctclip_path):
                    try:
                        state_dict = torch.load(ctclip_path, map_location='cpu')
                        # Load with strict=False in case of slight differences
                        backbone.load_state_dict(state_dict, strict=False)
                        print(f"🎯 Loaded CT-CLIP weights from {ctclip_path}")
                        ctclip_loaded = True
                        break
                    except Exception as e:
                        print(f"⚠️ Failed to load {ctclip_path}: {e}")
                        continue

            if not ctclip_loaded:
                print("ℹ️ Using ImageNet pretrained weights (CT-CLIP not found)")

            self.feature_dim = 1280
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return backbone
    
    def _build_classifier(self, dropout_rate: float):
        """Build the classification head"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, self.num_classes)
        )

    def _build_attention(self, attention_type: str):
        """Build attention mechanism"""
        if attention_type == "se":
            return SEBlock(self.feature_dim)
        elif attention_type == "cbam":
            return CBAM(self.feature_dim)
        else:
            return nn.Identity()

    def _build_loss_function(self, loss_type: str):
        """Build loss function based on type"""
        if loss_type == "focal":
            return "focal"  # We'll handle this in compute_loss
        elif loss_type == "bce":
            return "bce"
        elif loss_type == "asl":
            return AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        else:
            return "focal"

    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def on_train_epoch_start(self):
        """Handle progressive unfreezing"""
        self.current_epoch_num += 1

        if self.progressive_unfreeze and self.current_epoch_num == self.unfreeze_epoch:
            self._unfreeze_backbone()
            logger.info(f"Backbone unfrozen at epoch {self.current_epoch_num}")

    def _create_metrics(self, prefix: str):
        """Create torchmetrics for evaluation (removed AUROC due to NaN issues with rare classes)"""
        return nn.ModuleDict({
            'f1': torchmetrics.F1Score(task='multilabel', num_labels=self.num_classes, average='macro'),
            'precision': torchmetrics.Precision(task='multilabel', num_labels=self.num_classes, average='macro'),
            'recall': torchmetrics.Recall(task='multilabel', num_labels=self.num_classes, average='macro'),
            'accuracy': torchmetrics.Accuracy(task='multilabel', num_labels=self.num_classes, average='micro')
        })

    def _find_optimal_threshold(self, probs, labels):
        """Find optimal threshold using F1 score maximization"""
        best_threshold = 0.5
        best_f1 = 0

        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)

        for threshold in thresholds:
            y_pred = (probs > threshold).astype(int)
            f1 = f1_score(labels, y_pred, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def _calculate_metrics_with_threshold(self, probs, labels, y_pred, suffix=""):
        """Calculate metrics with specific threshold"""
        metrics = {}

        # Accuracy metrics
        hamming_accuracy = np.mean(labels == y_pred)
        exact_match_accuracy = np.mean(np.all(labels == y_pred, axis=1))
        sample_wise_accuracy = np.mean([
            accuracy_score(labels[i], y_pred[i]) for i in range(len(labels))
        ])

        metrics[f'hamming_accuracy{suffix}'] = hamming_accuracy
        metrics[f'exact_match_accuracy{suffix}'] = exact_match_accuracy
        metrics[f'sample_accuracy{suffix}'] = sample_wise_accuracy

        # Classification metrics
        metrics[f'f1_macro{suffix}'] = f1_score(labels, y_pred, average='macro', zero_division=0)
        metrics[f'f1_micro{suffix}'] = f1_score(labels, y_pred, average='micro', zero_division=0)
        metrics[f'precision_macro{suffix}'] = precision_score(labels, y_pred, average='macro', zero_division=0)
        metrics[f'recall_macro{suffix}'] = recall_score(labels, y_pred, average='macro', zero_division=0)

        return metrics

    def forward(self, x):
        features = self.backbone(x)

        # Apply attention if enabled
        if hasattr(self, 'attention') and self.attention is not None:
            features = self.attention(features)

        logits = self.classifier(features)
        return logits
    
    def compute_loss(self, logits, labels):
        """Compute loss based on configured loss type with aggressive CT-CLIP settings"""
        if isinstance(self.criterion, AsymmetricLoss):
            return self.criterion(logits, labels)
        elif self.criterion == "bce":
            # Add class weights to BCE for imbalanced data
            if self.class_weights is not None:
                # Convert weights to match logits shape
                weights = self.class_weights.to(logits.device).unsqueeze(0)
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels,
                    weight=weights.expand_as(logits),
                    reduction='mean'
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            return loss
        elif self.criterion == "focal":
            # Aggressive focal loss for CT-CLIP - heavily penalizes over-prediction
            alpha = 0.85  # Much higher alpha to penalize positive predictions heavily
            gamma = 4.0   # Higher gamma to focus on hard examples

            bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = alpha * (1 - pt) ** gamma * bce_loss

            # Apply class weights if available
            if self.class_weights is not None:
                weights = self.class_weights.to(logits.device).unsqueeze(0)
                focal_loss = focal_loss * weights.expand_as(focal_loss)

            return focal_loss.mean()
        else:
            # Default to aggressive focal loss for CT-CLIP
            alpha = 0.85  # Aggressive against over-prediction
            gamma = 4.0   # Focus on hard examples

            bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = alpha * (1 - pt) ** gamma * bce_loss

            return focal_loss.mean()
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        logits = self(images)
        loss = self.compute_loss(logits, labels)
        probs = torch.sigmoid(logits)
        
        # Update training metrics
        self.train_metrics['f1'].update(probs, labels.int())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        logits = self(images)
        loss = self.compute_loss(logits, labels)
        probs = torch.sigmoid(logits)
        
        # Update validation metrics
        for name, metric in self.val_metrics.items():
            metric.update(probs, labels.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss}
    
    def on_validation_epoch_end(self):
        """Log validation metrics"""
        for name, metric in self.val_metrics.items():
            value = metric.compute()
            self.log(f'val_{name}', value, on_epoch=True, prog_bar=True)
            metric.reset()
    
    def test_step(self, batch, batch_idx):
        """Test step (same as validation)"""
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        """Log test metrics"""
        for name, metric in self.val_metrics.items():
            value = metric.compute()
            self.log(f'test_{name}', value, on_epoch=True)
            metric.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

def get_transforms(augment: bool = False):
    """Get data transforms"""
    if augment:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def load_data(data_dir: str = "./ct_rate_data", slice_dir: str = "./ct_rate_2d"):
    """Load CT-RATE data and slice metadata"""
    data_dir = Path(data_dir)
    slice_dir = Path(slice_dir)
    
    # Load multi-abnormality labels
    labels_file = data_dir / "multi_abnormality_labels.csv"
    if labels_file.exists():
        multi_labels_df = pd.read_csv(labels_file)
        logger.info(f"Loaded multi-abnormality labels: {len(multi_labels_df)} rows")
    else:
        logger.warning(f"Multi-abnormality labels not found at {labels_file}")
        multi_labels_df = None
    
    # Load slice metadata
    train_slices_file = slice_dir / "splits" / "train_slices.csv"
    val_slices_file = slice_dir / "splits" / "valid_slices.csv"
    
    if not train_slices_file.exists():
        raise FileNotFoundError(f"Training slices metadata not found: {train_slices_file}")
    if not val_slices_file.exists():
        raise FileNotFoundError(f"Validation slices metadata not found: {val_slices_file}")
    
    train_slices_df = pd.read_csv(train_slices_file)
    val_slices_df = pd.read_csv(val_slices_file)
    
    logger.info(f"Loaded slice metadata: {len(train_slices_df)} train, {len(val_slices_df)} val")
    
    return train_slices_df, val_slices_df, multi_labels_df

def create_data_loaders(train_slices_df, val_slices_df, multi_labels_df,
                       slice_dir: str, batch_size: int = 32, num_workers: int = 4,
                       use_advanced_aug: bool = False, cutmix_prob: float = 0.5):
    """Create data loaders"""
    
    # Transforms
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    # Datasets
    train_dataset = CTSliceDataset(
        slice_df=train_slices_df,
        data_root=slice_dir,
        multi_abnormality_df=multi_labels_df,
        transform=train_transform,
        augment=True,
        use_advanced_aug=use_advanced_aug,
        cutmix_prob=cutmix_prob
    )

    val_dataset = CTSliceDataset(
        slice_df=val_slices_df,
        data_root=slice_dir,
        multi_abnormality_df=multi_labels_df,
        transform=val_transform,
        augment=False,
        use_advanced_aug=False,  # No augmentation for validation
        cutmix_prob=0.0
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_weights

def train_model(args):
    """Main training function"""
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    train_slices_df, val_slices_df, multi_labels_df = load_data(args.data_dir, args.slice_dir)
    
    # Create data loaders
    train_loader, val_loader, class_weights = create_data_loaders(
        train_slices_df, val_slices_df, multi_labels_df,
        args.slice_dir, args.batch_size, args.num_workers,
        use_advanced_aug=getattr(args, 'use_advanced_aug', False),
        cutmix_prob=getattr(args, 'cutmix_prob', 0.5)
    )
    
    # Initialize model
    model = MultiAbnormalityModel(
        model_name=args.model_name,
        num_classes=18,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        dropout_rate=args.dropout_rate,
        freeze_backbone=getattr(args, 'freeze_backbone', False),
        use_attention=getattr(args, 'use_attention', 'none'),
        use_multiscale=getattr(args, 'use_multiscale', False),
        loss_type=getattr(args, 'loss_type', 'focal'),
        progressive_unfreeze=getattr(args, 'progressive_unfreeze', False),
        unfreeze_epoch=getattr(args, 'unfreeze_epoch', 10)
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='multi_abnormality-{epoch:02d}-{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping_patience,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger_tb = TensorBoardLogger(
        save_dir=args.log_dir,
        name='multi_abnormality_classification'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices='auto',
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger_tb,
        precision=16 if args.use_mixed_precision else 32,
        gradient_clip_val=1.0,
        deterministic=False  # Disable strict determinism to avoid issues with attention modules
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test on validation set
    logger.info("Testing on validation set...")
    try:
        trainer.test(model, val_loader, ckpt_path='best')
    except Exception as e:
        logger.warning(f"Test step failed: {e}")
        logger.info("Continuing without test step...")
    
    return model, trainer

def evaluate_model(checkpoint_path: str, slice_dir: str, data_dir: str):
    """Evaluate trained model"""
    logger.info(f"Evaluating model from {checkpoint_path}")
    
    # Load data
    train_slices_df, val_slices_df, multi_labels_df = load_data(data_dir, slice_dir)
    
    # Create validation loader
    val_transform = get_transforms(augment=False)
    val_dataset = CTSliceDataset(
        slice_df=val_slices_df,
        data_root=slice_dir,
        multi_abnormality_df=multi_labels_df,
        transform=val_transform,
        augment=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load model
    model = MultiAbnormalityModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Generate predictions
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image']
            labels = batch['labels']
            
            if torch.cuda.is_available():
                images = images.cuda()
                model = model.cuda()
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # Quick sanity check of predictions
    logger.info("PREDICTION ANALYSIS:")
    logger.info(f"  Prediction range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
    logger.info(f"  Predictions > 0.5: {np.sum(all_probs > 0.5)}/{all_probs.size} ({100*np.sum(all_probs > 0.5)/all_probs.size:.1f}%)")
    logger.info(f"  All predictions == 0: {np.all(all_probs == 0)}")
    logger.info(f"  All predictions == 1: {np.all(all_probs == 1)}")
    logger.info(f"  Label distribution: {all_labels.sum(axis=0)} (positives per class)")

    # Calculate metrics
    metrics = {}
    
    # Overall metrics
    try:
        # Compute AUROC with NaN handling
        try:
            auroc_macro = roc_auc_score(all_labels, all_probs, average='macro')
            # Check if result is NaN (happens when classes have no positive samples)
            if np.isnan(auroc_macro):
                # Compute per-class AUROC and average only valid values
                per_class_auroc = roc_auc_score(all_labels, all_probs, average=None)
                valid_aurocs = per_class_auroc[~np.isnan(per_class_auroc)]
                if len(valid_aurocs) > 0:
                    auroc_macro = np.mean(valid_aurocs)
                    logger.warning(f"AUROC macro computed from {len(valid_aurocs)}/{len(per_class_auroc)} valid classes")
                else:
                    auroc_macro = 0.5  # Default to random performance
                    logger.warning("No valid AUROC classes found, using default value 0.5")
            metrics['auroc_macro'] = auroc_macro
        except Exception as e:
            logger.warning(f"Could not compute AUROC macro: {e}")
            metrics['auroc_macro'] = 0.5

        # AUROC micro is more robust
        try:
            metrics['auroc_micro'] = roc_auc_score(all_labels, all_probs, average='micro')
        except Exception as e:
            logger.warning(f"Could not compute AUROC micro: {e}")
            metrics['auroc_micro'] = 0.5

        # Threshold calibration - find optimal threshold
        optimal_threshold = self._find_optimal_threshold(all_probs, all_labels)
        logger.info(f"Optimal threshold found: {optimal_threshold:.3f}")

        # Use both 0.5 and optimal threshold for predictions
        y_pred_05 = (all_probs > 0.5).astype(int)
        y_pred_opt = (all_probs > optimal_threshold).astype(int)

        # Calculate metrics with both thresholds
        metrics_05 = self._calculate_metrics_with_threshold(all_probs, all_labels, y_pred_05, suffix="_05")
        metrics_opt = self._calculate_metrics_with_threshold(all_probs, all_labels, y_pred_opt, suffix="_opt")

        # Use 0.5 threshold for main metrics (backward compatibility)
        y_pred = y_pred_05

        # Calculate metrics with both thresholds
        metrics_05 = self._calculate_metrics_with_threshold(all_probs, all_labels, y_pred_05, suffix="_05")
        metrics_opt = self._calculate_metrics_with_threshold(all_probs, all_labels, y_pred_opt, suffix="_opt")

        # Merge metrics
        metrics.update(metrics_05)
        metrics.update(metrics_opt)

        # Keep exact match as 'accuracy' for backward compatibility
        metrics['accuracy'] = metrics_05['exact_match_accuracy_05']

        # Add threshold info
        metrics['optimal_threshold'] = optimal_threshold

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        # Set default values
        metrics.update({
            'auroc_macro': 0.5,
            'auroc_micro': 0.5,
            'f1_macro': 0.0,
            'f1_micro': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'accuracy': 0.0,
            'hamming_accuracy': 0.0,
            'exact_match_accuracy': 0.0,
            'sample_accuracy': 0.0
        })
    
    # Print results
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)

    # Show threshold comparison
    logger.info("THRESHOLD COMPARISON:")
    logger.info(f"  Optimal threshold: {metrics.get('optimal_threshold', 0.5):.3f}")

    logger.info("  Threshold 0.5 vs Optimal:")
    hamming_05 = metrics.get('hamming_accuracy_05', 0)
    hamming_opt = metrics.get('hamming_accuracy_opt', 0)
    f1_05 = metrics.get('f1_macro_05', 0)
    f1_opt = metrics.get('f1_macro_opt', 0)

    logger.info(f"    Hamming Acc: {hamming_05:.4f} → {hamming_opt:.4f} ({hamming_opt-hamming_05:+.4f})")
    logger.info(f"    F1 Macro:    {f1_05:.4f} → {f1_opt:.4f} ({f1_opt-f1_05:+.4f})")

    # Log primary metrics (using optimal threshold)
    logger.info("-" * 30)
    logger.info("PRIMARY METRICS (Optimal Threshold):")
    logger.info(f"hamming_accuracy     : {metrics.get('hamming_accuracy_opt', 0):.4f}")
    logger.info(f"f1_macro             : {metrics.get('f1_macro_opt', 0):.4f}")
    logger.info(f"precision_macro      : {metrics.get('precision_macro_opt', 0):.4f}")
    logger.info(f"recall_macro         : {metrics.get('recall_macro_opt', 0):.4f}")

    logger.info("-" * 30)
    logger.info("OVERALL METRICS:")
    logger.info(f"auroc_macro          : {metrics.get('auroc_macro', 0):.4f}")
    logger.info(f"auroc_micro          : {metrics.get('auroc_micro', 0):.4f}")

    # Add interpretation
    logger.info("-" * 30)
    hamming_opt = metrics.get('hamming_accuracy_opt', 0)
    exact_opt = metrics.get('exact_match_accuracy_opt', 0)

    logger.info("INTERPRETATION:")
    logger.info(f"  Hamming Acc: {hamming_opt:.1%} of individual predictions are correct")
    logger.info(f"  Exact Match: {exact_opt:.1%} of samples have ALL labels correct")
    logger.info(f"  (Low exact match is normal for multi-label with many classes)")

    # Performance analysis
    logger.info("-" * 30)
    logger.info("PERFORMANCE ANALYSIS:")
    pos_rate = np.sum(all_probs > 0.5) / all_probs.size
    logger.info(f"  Positive prediction rate: {pos_rate:.1%}")
    logger.info(f"  Model bias: {'Over-predicting positives' if pos_rate > 0.6 else 'Balanced' if pos_rate > 0.4 else 'Under-predicting positives'}")

    if metrics.get('auroc_macro', 0) < 0.6:
        logger.info("  ⚠️  AUROC < 0.6 suggests model needs improvement")
    if hamming_opt < 0.3:
        logger.info("  ⚠️  Low hamming accuracy suggests poor individual predictions")
    if metrics.get('precision_macro_opt', 0) < 0.3:
        logger.info("  ⚠️  Low precision suggests too many false positives")

    # Debug information
    logger.info("-" * 30)
    logger.info("DEBUG INFO:")
    logger.info(f"  Total samples: {len(all_labels)}")
    logger.info(f"  Predictions shape: {all_probs.shape}")
    logger.info(f"  Labels shape: {all_labels.shape}")
    logger.info(f"  Classes with positive samples: {np.sum(all_labels.sum(axis=0) > 0)}/{all_labels.shape[1]}")
    logger.info(f"  Average labels per sample: {all_labels.sum(axis=1).mean():.2f}")
    
    # Save results
    results_file = Path("./results") / "evaluation_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Multi-Abnormality Classification Training")
    
    # Data arguments
    parser.add_argument("--data-dir", default="./ct_rate_data", help="CT-RATE data directory")
    parser.add_argument("--slice-dir", default="./ct_rate_2d", help="Extracted slices directory")
    
    # Model arguments
    parser.add_argument("--model-name", default="resnet50", choices=["resnet50", "resnet101", "efficientnet_b0"])
    parser.add_argument("--dropout-rate", type=float, default=0.3)

    # Advanced model arguments
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone layers")
    parser.add_argument("--use-attention", choices=["none", "se", "cbam"], default="none",
                       help="Attention mechanism to use")
    parser.add_argument("--use-multiscale", action="store_true", help="Use multi-scale features")
    parser.add_argument("--loss-type", choices=["focal", "bce", "asl"], default="focal",
                       help="Loss function type")
    parser.add_argument("--progressive-unfreeze", action="store_true",
                       help="Use progressive unfreezing of backbone")
    parser.add_argument("--unfreeze-epoch", type=int, default=10,
                       help="Epoch to unfreeze backbone in progressive unfreezing")

    # Advanced augmentation arguments
    parser.add_argument("--use-advanced-aug", action="store_true",
                       help="Use advanced augmentations (CutMix, etc.)")
    parser.add_argument("--cutmix-prob", type=float, default=0.5,
                       help="Probability of applying CutMix augmentation")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-mixed-precision", action="store_true")
    
    # Directory arguments
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--log-dir", default="./logs")
    
    # Mode arguments
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--checkpoint", help="Checkpoint path for evaluation")
    
    args = parser.parse_args()
    
    # Set random seeds
    pl.seed_everything(42)
    
    if args.mode == "train":
        model, trainer = train_model(args)
        logger.info("Training completed!")
        
        # Automatically evaluate best model
        best_checkpoint = Path(args.checkpoint_dir) / "multi_abnormality-*.ckpt"
        checkpoints = list(Path(args.checkpoint_dir).glob("multi_abnormality-*.ckpt"))
        if checkpoints:
            best_checkpoint = sorted(checkpoints)[-1]  # Latest checkpoint
            logger.info(f"Evaluating best model: {best_checkpoint}")
            evaluate_model(str(best_checkpoint), args.slice_dir, args.data_dir)
        
    elif args.mode == "evaluate":
        if not args.checkpoint:
            raise ValueError("Checkpoint path required for evaluation")
        evaluate_model(args.checkpoint, args.slice_dir, args.data_dir)

if __name__ == "__main__":
    main() 