#!/usr/bin/env python3
"""
Multi-Abnormality Classification for VLM3D Task 2
18-class binary classification for thoracic conditions from chest CT slices
"""

import os
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
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, 
    accuracy_score, average_precision_score, classification_report
)
from pathlib import Path
import argparse
import yaml
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTSliceDataset(Dataset):
    """Dataset class for loading 2D CT slices with multi-label abnormalities"""
    
    def __init__(self, 
                 slice_csv: str,
                 data_root: str,
                 transform=None,
                 augment: bool = False):
        
        self.df = pd.read_csv(slice_csv)
        self.data_root = Path(data_root)
        self.transform = transform
        self.augment = augment
        
        # 18 abnormality classes as defined in CT-RATE
        self.abnormality_classes = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Fracture', 'Hernia', 'Infiltration',
            'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
            'Support_Devices', 'Thickening', 'No_Finding'
        ]
        
        # Ensure all label columns exist
        for col in self.abnormality_classes:
            if col not in self.df.columns:
                self.df[col] = 0
                
        # Calculate class weights for imbalanced data
        self.class_weights = self._calculate_class_weights()
        
        logger.info(f"Loaded dataset with {len(self.df)} slices")
        logger.info(f"Class distribution: {self._get_class_distribution()}")
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data"""
        pos_counts = self.df[self.abnormality_classes].sum()
        neg_counts = len(self.df) - pos_counts
        
        # Use inverse frequency weighting
        weights = []
        for class_name in self.abnormality_classes:
            pos_count = pos_counts[class_name]
            neg_count = neg_counts[class_name]
            
            if pos_count == 0:
                weight = 1.0  # Default weight if no positive samples
            else:
                weight = neg_count / pos_count
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _get_class_distribution(self) -> Dict[str, float]:
        """Get the distribution of positive samples for each class"""
        distribution = {}
        for class_name in self.abnormality_classes:
            positive_ratio = self.df[class_name].mean()
            distribution[class_name] = positive_ratio
        return distribution
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load slice
        slice_path = self.data_root / row['file_path']
        slice_data = np.load(slice_path)
        
        # Convert to 3-channel image (RGB) for pretrained models
        if len(slice_data.shape) == 2:
            slice_data = np.stack([slice_data] * 3, axis=-1)
        
        # Apply transforms
        if self.transform:
            slice_data = self.transform(slice_data)
        
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
                 class_weights: Optional[torch.Tensor] = None,
                 dropout_rate: float = 0.3,
                 use_focal_loss: bool = True):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        
        # Build model
        self.backbone = self._build_backbone(model_name)
        self.classifier = self._build_classifier(dropout_rate)
        
        # Metrics storage
        self.validation_outputs = []
        self.test_outputs = []
        
        # Class names for logging
        self.class_names = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Fracture', 'Hernia', 'Infiltration',
            'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
            'Support_Devices', 'Thickening', 'No_Finding'
        ]
    
    def _build_backbone(self, model_name: str):
        """Build the backbone network"""
        if model_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            # Remove the final classification layer
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 2048
        elif model_name == "resnet101":
            backbone = models.resnet101(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 2048
        elif model_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=True)
            backbone.classifier = nn.Identity()
            self.feature_dim = 1280
        elif model_name == "densenet121":
            backbone = models.densenet121(pretrained=True)
            backbone.classifier = nn.Identity()
            self.feature_dim = 1024
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
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance"""
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
    
    def weighted_bce_loss(self, inputs, targets):
        """Weighted binary cross entropy loss"""
        if self.class_weights is not None:
            weights = self.class_weights.to(inputs.device)
            # Apply class weights
            pos_weights = weights.unsqueeze(0).expand_as(targets)
            loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weights)
        else:
            loss = F.binary_cross_entropy_with_logits(inputs, targets)
        return loss
    
    def compute_loss(self, logits, labels):
        """Compute the appropriate loss function"""
        if self.use_focal_loss:
            return self.focal_loss(logits, labels)
        else:
            return self.weighted_bce_loss(logits, labels)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        logits = self(images)
        loss = self.compute_loss(logits, labels)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        logits = self(images)
        loss = self.compute_loss(logits, labels)
        probs = torch.sigmoid(logits)
        
        # Store outputs for epoch-end metrics
        self.validation_outputs.append({
            'probs': probs.cpu(),
            'labels': labels.cpu(),
            'loss': loss.cpu()
        })
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate and log validation metrics"""
        if not self.validation_outputs:
            return
        
        # Concatenate all outputs
        all_probs = torch.cat([x['probs'] for x in self.validation_outputs], dim=0)
        all_labels = torch.cat([x['labels'] for x in self.validation_outputs], dim=0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_probs.numpy(), all_labels.numpy(), prefix='val')
        
        # Log metrics
        for metric_name, value in metrics.items():
            self.log(metric_name, value, on_epoch=True, prog_bar=False)
        
        # Clear outputs
        self.validation_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        logits = self(images)
        probs = torch.sigmoid(logits)
        
        self.test_outputs.append({
            'probs': probs.cpu(),
            'labels': labels.cpu(),
            'slice_ids': batch['slice_id'],
            'volume_names': batch['volume_name']
        })
        
        return probs
    
    def on_test_epoch_end(self):
        """Calculate and log test metrics"""
        if not self.test_outputs:
            return
        
        # Concatenate all outputs
        all_probs = torch.cat([x['probs'] for x in self.test_outputs], dim=0)
        all_labels = torch.cat([x['labels'] for x in self.test_outputs], dim=0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_probs.numpy(), all_labels.numpy(), prefix='test')
        
        # Log metrics
        for metric_name, value in metrics.items():
            self.log(metric_name, value, on_epoch=True)
        
        # Save detailed results
        self.save_test_results(all_probs.numpy(), all_labels.numpy())
        
        # Clear outputs
        self.test_outputs.clear()
    
    def calculate_metrics(self, probs: np.ndarray, labels: np.ndarray, prefix: str = '') -> Dict[str, float]:
        """Calculate comprehensive metrics for multi-label classification"""
        metrics = {}
        
        # Convert probabilities to binary predictions
        preds = (probs > 0.5).astype(int)
        
        # Overall metrics (macro and micro averages)
        try:
            # AUROC (macro and micro)
            auroc_macro = roc_auc_score(labels, probs, average='macro')
            auroc_micro = roc_auc_score(labels, probs, average='micro')
            metrics[f'{prefix}_auroc_macro'] = auroc_macro
            metrics[f'{prefix}_auroc_micro'] = auroc_micro
            
            # F1 Score
            f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
            f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
            metrics[f'{prefix}_f1_macro'] = f1_macro
            metrics[f'{prefix}_f1_micro'] = f1_micro
            
            # Precision
            precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
            precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
            metrics[f'{prefix}_precision_macro'] = precision_macro
            metrics[f'{prefix}_precision_micro'] = precision_micro
            
            # Recall
            recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
            recall_micro = recall_score(labels, preds, average='micro', zero_division=0)
            metrics[f'{prefix}_recall_macro'] = recall_macro
            metrics[f'{prefix}_recall_micro'] = recall_micro
            
            # Subset accuracy (exact match)
            subset_accuracy = accuracy_score(labels, preds)
            metrics[f'{prefix}_subset_accuracy'] = subset_accuracy
            
            # Sample-wise accuracy (at least one correct prediction per sample)
            sample_accuracy = np.mean([
                np.any(labels[i] == preds[i]) for i in range(len(labels))
            ])
            metrics[f'{prefix}_sample_accuracy'] = sample_accuracy
            
            # Average Precision Score
            ap_macro = average_precision_score(labels, probs, average='macro')
            ap_micro = average_precision_score(labels, probs, average='micro')
            metrics[f'{prefix}_ap_macro'] = ap_macro
            metrics[f'{prefix}_ap_micro'] = ap_micro
            
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
        
        return metrics
    
    def save_test_results(self, probs: np.ndarray, labels: np.ndarray):
        """Save detailed test results"""
        # Per-class metrics
        results = {
            'per_class_metrics': {},
            'confusion_matrices': {},
            'overall_metrics': self.calculate_metrics(probs, labels, prefix='test')
        }
        
        preds = (probs > 0.5).astype(int)
        
        for i, class_name in enumerate(self.class_names):
            class_labels = labels[:, i]
            class_probs = probs[:, i]
            class_preds = preds[:, i]
            
            if np.sum(class_labels) > 0:  # Only calculate if positive samples exist
                auroc = roc_auc_score(class_labels, class_probs)
                f1 = f1_score(class_labels, class_preds, zero_division=0)
                precision = precision_score(class_labels, class_preds, zero_division=0)
                recall = recall_score(class_labels, class_preds, zero_division=0)
                
                results['per_class_metrics'][class_name] = {
                    'auroc': float(auroc),
                    'f1': float(f1),
                    'precision': float(precision),
                    'recall': float(recall),
                    'support': int(np.sum(class_labels))
                }
        
        # Save results
        results_file = Path('./results') / 'test_results.json'
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {results_file}")
    
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
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test"""
    
    # Transforms
    train_transform = get_transforms(augment=True)
    val_test_transform = get_transforms(augment=False)
    
    # Datasets
    train_dataset = CTSliceDataset(
        slice_csv=config['data']['train_csv'],
        data_root=config['data']['data_root'],
        transform=train_transform,
        augment=True
    )
    
    val_dataset = CTSliceDataset(
        slice_csv=config['data']['val_csv'],
        data_root=config['data']['data_root'],
        transform=val_test_transform,
        augment=False
    )
    
    test_dataset = CTSliceDataset(
        slice_csv=config['data']['test_csv'],
        data_root=config['data']['data_root'],
        transform=val_test_transform,
        augment=False
    )
    
    # Create weighted sampler for training (handle class imbalance)
    if config['training']['use_weighted_sampling']:
        # Calculate sample weights based on label distribution
        label_matrix = train_dataset.df[train_dataset.abnormality_classes].values
        label_counts = np.sum(label_matrix, axis=1)  # Number of positive labels per sample
        
        # Give higher weight to samples with more rare combinations
        sample_weights = 1.0 / (label_counts + 1)  # +1 to avoid division by zero
        sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    else:
        sampler = None
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_weights

def train_model(config: Dict):
    """Train the multi-abnormality classification model"""
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(config)
    
    # Initialize model
    model = MultiAbnormalityModel(
        model_name=config['model']['backbone'],
        num_classes=18,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        class_weights=class_weights,
        dropout_rate=config['model']['dropout_rate'],
        use_focal_loss=config['training']['use_focal_loss']
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='multi_abnormality-{epoch:02d}-{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stopping_patience'],
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger_tb = TensorBoardLogger(
        save_dir=config['training']['log_dir'],
        name='multi_abnormality_classification'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        gpus=config['training']['gpus'] if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger_tb,
        deterministic=True,
        precision=16 if config['training']['use_mixed_precision'] else 32,
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches']
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    trainer.test(model, test_loader, ckpt_path='best')
    
    return model, trainer

def main():
    parser = argparse.ArgumentParser(description="Multi-Abnormality Classification Training")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--mode", choices=["train", "test", "inference"], default="train")
    parser.add_argument("--checkpoint", help="Checkpoint path for testing/inference")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    pl.seed_everything(config['training']['seed'])
    
    if args.mode == "train":
        model, trainer = train_model(config)
        logger.info("Training completed!")
        
    elif args.mode == "test":
        if not args.checkpoint:
            raise ValueError("Checkpoint path required for testing")
        
        # Load model from checkpoint
        model = MultiAbnormalityModel.load_from_checkpoint(args.checkpoint)
        
        # Create test data loader
        _, _, test_loader, _ = create_data_loaders(config)
        
        # Test
        trainer = pl.Trainer(gpus=config['training']['gpus'] if torch.cuda.is_available() else 0)
        trainer.test(model, test_loader)
        
    elif args.mode == "inference":
        # Implementation for inference mode would go here
        logger.info("Inference mode not yet implemented")

if __name__ == "__main__":
    main() 