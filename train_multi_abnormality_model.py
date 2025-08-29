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
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CTSliceDataset(Dataset):
    """Dataset for loading CT slices with multi-label abnormalities"""
    
    def __init__(self, 
                 slice_df: pd.DataFrame,
                 data_root: str,
                 multi_abnormality_df: pd.DataFrame = None,
                 transform=None,
                 augment: bool = False):
        
        self.slice_df = slice_df.copy()
        self.data_root = Path(data_root)
        self.transform = transform
        self.augment = augment
        
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
                 dropout_rate: float = 0.3):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        
        # Build model
        self.backbone = self._build_backbone(model_name)
        self.classifier = self._build_classifier(dropout_rate)
        
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
        """Build the backbone network"""
        if model_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 2048
        elif model_name == "resnet101":
            backbone = models.resnet101(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 2048
        elif model_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=True)
            # Remove the classifier to get features
            backbone = nn.Sequential(*list(backbone.children())[:-1])
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
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def compute_loss(self, logits, labels):
        """Compute focal loss for class imbalance"""
        # Focal loss parameters
        alpha = 0.25
        gamma = 2.0
        
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
        
        # Update metrics
        self.train_metrics['auroc'].update(probs, labels.int())
        self.train_metrics['f1'].update(probs, labels.int())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        logits = self(images)
        loss = self.compute_loss(logits, labels)
        probs = torch.sigmoid(logits)
        
        # Update metrics
        for name, metric in self.val_metrics.items():
            metric.update(probs, labels.int())
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'probs': probs, 'labels': labels}
    
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
                       slice_dir: str, batch_size: int = 32, num_workers: int = 4):
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
        augment=True
    )
    
    val_dataset = CTSliceDataset(
        slice_df=val_slices_df,
        data_root=slice_dir,
        multi_abnormality_df=multi_labels_df,
        transform=val_transform,
        augment=False
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
        args.slice_dir, args.batch_size, args.num_workers
    )
    
    # Initialize model
    model = MultiAbnormalityModel(
        model_name=args.model_name,
        num_classes=18,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        dropout_rate=args.dropout_rate
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
        deterministic=True
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
    
    # Calculate metrics
    metrics = {}
    
    # Overall metrics
    try:
        metrics['auroc_macro'] = roc_auc_score(all_labels, all_probs, average='macro')
        metrics['auroc_micro'] = roc_auc_score(all_labels, all_probs, average='micro')
        
        y_pred = (all_probs > 0.5).astype(int)
        metrics['f1_macro'] = f1_score(all_labels, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(all_labels, y_pred, average='micro', zero_division=0)
        metrics['precision_macro'] = precision_score(all_labels, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(all_labels, y_pred, average='macro', zero_division=0)
        metrics['accuracy'] = accuracy_score(all_labels, y_pred)
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
    
    # Print results
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name:15s}: {value:.4f}")
    
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