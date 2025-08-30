# VLM3D Task 2: Multi-Abnormality Classification - Complete Instructions

## Overview

This document provides comprehensive instructions for running the VLM3D Task 2 multi-abnormality classification pipeline. The system consists of two main scripts:

1. **`run_task2.py`** - Main runner script with high-level control and GPU management
2. **`train_multi_abnormality_model.py`** - Core training module with advanced model configurations

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Script 1: run_task2.py](#script-1-runtask2py)
- [Script 2: train_multi_abnormality_model.py](#script-2-train_multi_abnormality_modelpy)
- [GPU Configuration](#gpu-configuration)
- [Training Strategies](#training-strategies)
- [Advanced Configurations](#advanced-configurations)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (RTX 4060 recommended)
- **RAM**: Minimum 16GB, 32GB recommended
- **Storage**: At least 10GB free space for data and models

### Software
- **Python**: 3.8+ (3.9+ recommended)
- **CUDA**: 11.8+ (compatible with PyTorch 2.5.1)
- **Conda**: Anaconda or Miniconda for environment management

### Dependencies
- PyTorch 2.5.1+
- PyTorch Lightning 2.5.3+
- TorchVision
- Pandas, NumPy
- Scikit-learn
- TorchMetrics
- Matplotlib, Seaborn

## Quick Start

### 1. Basic Training (Auto GPU Selection)
```bash
conda run -n svenv python run_task2.py --loss-type focal --freeze-backbone --epochs 50
```

### 2. Force NVIDIA GPU Usage
```bash
conda run -n svenv python run_task2.py --gpu-device 1 --loss-type focal --freeze-backbone --epochs 50
```

### 3. PowerShell Helper Script
```powershell
.\run_with_nvidia_gpu.ps1
```

---

## Script 1: run_task2.py

### Purpose
`run_task2.py` is the main entry point that orchestrates the entire training pipeline. It handles:
- Environment validation
- Data structure verification
- GPU device selection
- Training execution
- Evaluation and prediction modes

### Basic Usage Structure
```bash
conda run -n ENV_NAME python run_task2.py [MODE] [OPTIONS]
```

### Mode Selection

#### 1. Training Mode (`--mode train`)
```bash
# Basic training
conda run -n svenv python run_task2.py --mode train --epochs 50

# Training with specific GPU
conda run -n svenv python run_task2.py --mode train --gpu-device 1 --epochs 50

# Training with advanced features
conda run -n svenv python run_task2.py --mode train \
    --gpu-device 1 \
    --loss-type focal \
    --freeze-backbone \
    --use-attention se \
    --epochs 100 \
    --batch-size 16
```

#### 2. Evaluation Mode (`--mode evaluate`)
```bash
# Evaluate with auto-detected checkpoint
conda run -n svenv python run_task2.py --mode evaluate

# Evaluate specific checkpoint
conda run -n svenv python run_task2.py --mode evaluate \
    --checkpoint ./checkpoints/multi_abnormality-epoch=50-val_loss=0.123.ckpt
```

#### 3. Both Training and Evaluation (`--mode both`) - DEFAULT
```bash
# Complete pipeline (train + evaluate)
conda run -n svenv python run_task2.py --mode both \
    --gpu-device 1 \
    --loss-type focal \
    --freeze-backbone \
    --epochs 50
```

#### 4. Prediction Mode (`--mode predict`)
```bash
# Predict on slices
conda run -n svenv python run_task2.py --mode predict \
    --predict-input slice1.npy slice2.npy slice3.npy \
    --predict-type slices

# Predict on volumes
conda run -n svenv python run_task2.py --mode predict \
    --predict-input volume1.nii.gz volume2.nii.gz \
    --predict-type volumes
```

### Core Arguments

#### Data Configuration
```bash
--data-dir ./ct_rate_data          # CT-RATE data directory
--slice-dir ./ct_rate_2d          # Extracted slices directory
```

#### Model Configuration
```bash
--model {resnet50,resnet101,efficientnet_b0}  # Model backbone
--batch-size 32                    # Batch size (adjust based on GPU memory)
--learning-rate 1e-4               # Learning rate
--epochs 100                       # Number of training epochs
```

#### GPU Configuration
```bash
--gpu-device 1                     # Force specific GPU (1 = NVIDIA RTX 4060)
# If not specified, auto-selects NVIDIA GPU over AMD
```

### Advanced Training Arguments

#### Backbone Control
```bash
--freeze-backbone                  # Freeze backbone layers during training
--progressive-unfreeze             # Gradually unfreeze backbone layers
--unfreeze-epoch 10               # Epoch to start unfreezing
```

#### Attention Mechanisms
```bash
--use-attention {none,se,cbam}    # Attention mechanism selection
# none: No attention (default)
# se: Squeeze-and-Excitation
# cbam: Convolutional Block Attention Module
```

#### Loss Functions
```bash
--loss-type {focal,bce,asl}       # Loss function selection
# focal: Focal Loss (default, good for imbalanced data)
# bce: Binary Cross Entropy
# asl: Asymmetric Loss (penalizes false positives more)
```

#### Augmentation Strategies
```bash
--use-advanced-aug                # Enable advanced augmentations
--cutmix-prob 0.5                 # CutMix probability
--use-multiscale                  # Multi-scale feature fusion
```

### Complete Example Commands

#### Example 1: Conservative Training (GPU 1, Focal Loss)
```bash
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --model resnet50 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --epochs 50 \
    --loss-type focal \
    --freeze-backbone \
    --early-stopping-patience 15
```

#### Example 2: Advanced Training (Attention + Augmentation)
```bash
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --model efficientnet_b0 \
    --batch-size 8 \
    --learning-rate 5e-5 \
    --epochs 100 \
    --loss-type asl \
    --use-attention cbam \
    --use-advanced-aug \
    --cutmix-prob 0.7 \
    --use-multiscale \
    --progressive-unfreeze \
    --unfreeze-epoch 20
```

#### Example 3: Quick Testing
```bash
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --epochs 5 \
    --batch-size 8 \
    --loss-type focal
```

---

## Script 2: train_multi_abnormality_model.py

### Purpose
`train_multi_abnormality_model.py` contains the core training logic, model architectures, and data processing. It's imported by `run_task2.py` and provides:

- Multi-abnormality classification models
- Advanced attention mechanisms
- Custom loss functions
- Data augmentation pipelines
- Training and evaluation loops

### Direct Usage (Advanced Users)
```bash
# Direct training (bypassing run_task2.py)
conda run -n svenv python train_multi_abnormality_model.py \
    --mode train \
    --model-name resnet50 \
    --batch-size 16 \
    --max-epochs 50 \
    --loss-type focal \
    --freeze-backbone
```

### Model Architecture Details

#### 1. Backbone Networks
```python
# ResNet50 (default)
--model-name resnet50              # 2048 features, ImageNet pretrained

# ResNet101
--model-name resnet101             # 2048 features, deeper architecture

# EfficientNet-B0
--model-name efficientnet_b0       # 1280 features, CT-CLIP compatible
```

#### 2. Attention Mechanisms

##### Squeeze-and-Excitation (SE)
```bash
--use-attention se
```
- **Purpose**: Channel-wise attention to highlight important features
- **Implementation**: Global average pooling + MLP + Sigmoid
- **Use Case**: General feature enhancement, good for medical imaging

##### Convolutional Block Attention Module (CBAM)
```bash
--use-attention cbam
```
- **Purpose**: Both channel and spatial attention
- **Implementation**: Channel attention + Spatial attention
- **Use Case**: Advanced feature refinement, better for complex patterns

#### 3. Loss Functions

##### Focal Loss (Default)
```bash
--loss-type focal
```
- **Purpose**: Handle class imbalance by focusing on hard examples
- **Parameters**: α=0.85, γ=4.0 (aggressive against over-prediction)
- **Use Case**: Imbalanced medical datasets, prevents over-prediction

##### Binary Cross Entropy (BCE)
```bash
--loss-type bce
```
- **Purpose**: Standard binary classification loss
- **Features**: Class weights support for imbalance handling
- **Use Case**: Balanced datasets, baseline comparison

##### Asymmetric Loss (ASL)
```bash
--loss-type asl
```
- **Purpose**: Different penalties for false positives vs false negatives
- **Parameters**: γ_neg=6, γ_pos=1, clip=0.05
- **Use Case**: Medical diagnosis where false positives are costly

### Advanced Training Features

#### 1. Progressive Unfreezing
```bash
--progressive-unfreeze             # Enable progressive unfreezing
--unfreeze-epoch 10               # Epoch to start unfreezing
```
**What it does**: 
- Starts with frozen backbone (faster initial training)
- Gradually unfreezes layers for fine-tuning
- Improves convergence and final performance

#### 2. Multi-Scale Feature Fusion
```bash
--use-multiscale                  # Enable multi-scale features
```
**What it does**:
- Combines features from different network depths
- Captures both low-level and high-level patterns
- Better for detecting abnormalities at different scales

#### 3. Advanced Augmentations
```bash
--use-advanced-aug                # Enable advanced augmentations
--cutmix-prob 0.5                 # CutMix probability
```
**Available Augmentations**:
- Random rotation (±10°)
- Random horizontal flip (50%)
- Random affine transformations
- Color jittering
- CutMix (when enabled)

### Data Processing Pipeline

#### 1. Dataset Structure
```
ct_rate_2d/
├── slices/
│   ├── train/                    # Training slice files (.npy)
│   └── valid/                    # Validation slice files (.npy)
└── splits/
    ├── train_slices.csv          # Training metadata
    └── valid_slices.csv          # Validation metadata
```

#### 2. Label Classes (18 Abnormality Types)
```python
abnormality_classes = [
    "Cardiomegaly", "Hiatal hernia", "Atelectasis", 
    "Pulmonary fibrotic sequela", "Peribronchial thickening",
    "Interlobular septal thickening", "Medical material",
    "Pericardial effusion", "Lymphadenopathy", "Lung nodule",
    "Pleural effusion", "Consolidation", "Lung opacity",
    "Mosaic attenuation pattern", "Bronchiectasis", "Emphysema",
    "Arterial wall calcification", "Coronary artery wall calcification"
]
```

#### 3. Data Loading Process
1. **Metadata Loading**: Read slice and label information
2. **Label Merging**: Match slices with multi-abnormality labels
3. **Class Weight Calculation**: Handle imbalanced data
4. **Transform Application**: Apply augmentations and normalization
5. **Batch Creation**: Create training/validation batches

### Training Configuration Options

#### 1. Optimizer Settings
```python
# AdamW Optimizer (default)
learning_rate: 1e-4
weight_decay: 1e-5

# Learning Rate Scheduler
scheduler: ReduceLROnPlateau
factor: 0.5
patience: 5
```

#### 2. Callbacks
```python
# Model Checkpointing
monitor: 'val_loss'
save_top_k: 3
save_last: True

# Early Stopping
patience: 10 (configurable)
monitor: 'val_loss'

# Learning Rate Monitoring
logging_interval: 'epoch'
```

#### 3. Mixed Precision
```bash
--use-mixed-precision             # Enable FP16 training
```
**Benefits**:
- Faster training (especially on RTX 4000 series)
- Lower memory usage
- Maintains accuracy

---

## GPU Configuration

### Automatic GPU Selection
The system automatically detects and prefers NVIDIA GPUs:

```python
# Auto-detection logic
for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i).lower()
    if 'nvidia' in gpu_name or 'rtx' in gpu_name or 'gtx' in gpu_name:
        selected_gpu = i
        break
```

### Manual GPU Selection
```bash
# Force GPU 1 (NVIDIA RTX 4060)
--gpu-device 1

# Force GPU 0 (if you want AMD)
--gpu-device 0
```

### GPU Memory Management
```bash
# Reduce batch size for memory issues
--batch-size 8                    # Small batch size
--batch-size 16                   # Medium batch size
--batch-size 32                   # Large batch size (default)

# Enable mixed precision for memory efficiency
--use-mixed-precision
```

---

## Training Strategies

### 1. Conservative Training (Recommended for Starters)
```bash
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --model resnet50 \
    --batch-size 16 \
    --epochs 50 \
    --loss-type focal \
    --freeze-backbone \
    --early-stopping-patience 15
```
**Use Case**: Initial experiments, baseline establishment
**Benefits**: Stable training, faster convergence, lower memory usage

### 2. Progressive Training
```bash
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --model resnet50 \
    --batch-size 16 \
    --epochs 100 \
    --loss-type focal \
    --progressive-unfreeze \
    --unfreeze-epoch 20
```
**Use Case**: Fine-tuning pretrained models
**Benefits**: Better final performance, controlled learning

### 3. Advanced Training
```bash
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --model efficientnet_b0 \
    --batch-size 8 \
    --epochs 100 \
    --loss-type asl \
    --use-attention cbam \
    --use-advanced-aug \
    --use-multiscale
```
**Use Case**: Maximum performance optimization
**Benefits**: Best possible results, advanced features
**Trade-offs**: Longer training time, higher memory usage

### 4. Quick Testing
```bash
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --epochs 5 \
    --batch-size 8 \
    --loss-type focal
```
**Use Case**: Code validation, quick experiments
**Benefits**: Fast feedback, minimal resource usage

---

## Advanced Configurations

### 1. Attention Mechanism Selection

#### No Attention (Default)
```bash
--use-attention none
```
**Best for**: Simple datasets, baseline models

#### Squeeze-and-Excitation
```bash
--use-attention se
```
**Best for**: General feature enhancement, medical imaging

#### CBAM
```bash
--use-attention cbam
```
**Best for**: Complex patterns, maximum performance

### 2. Loss Function Tuning

#### Focal Loss Parameters
```python
# Built-in aggressive settings for CT-CLIP
alpha = 0.85  # High alpha penalizes positive predictions
gamma = 4.0   # High gamma focuses on hard examples
```

#### Asymmetric Loss Parameters
```python
gamma_neg = 6    # Higher penalty for false positives
gamma_pos = 1    # Standard penalty for false negatives
clip = 0.05      # Clipping for numerical stability
```

### 3. Data Augmentation Pipeline

#### Basic Augmentations (Always Enabled)
- Random rotation (±10°)
- Random horizontal flip (50%)
- Random affine transformations
- Color jittering

#### Advanced Augmentations
```bash
--use-advanced-aug                # Enable CutMix, MixUp
--cutmix-prob 0.5                 # CutMix probability
```

### 4. Model Architecture Variations

#### ResNet Variants
```bash
--model resnet50                  # 25.6M parameters, balanced
--model resnet101                 # 44.6M parameters, deeper
```

#### EfficientNet
```bash
--model efficientnet_b0           # 5.3M parameters, efficient
# Compatible with CT-CLIP weights
```

---

## Monitoring and Logging

### 1. Training Progress
```bash
# View TensorBoard logs
tensorboard --logdir ./logs

# Check checkpoint directory
ls ./checkpoints/

# Monitor GPU usage
nvidia-smi
```

### 2. Log Files
```
logs/
└── multi_abnormality_classification/
    └── version_X/
        ├── events.out.tfevents.*
        └── hparams.yaml
```

### 3. Checkpoints
```
checkpoints/
├── multi_abnormality-epoch=01-val_loss=0.123.ckpt
├── multi_abnormality-epoch=02-val_loss=0.098.ckpt
└── last.ckpt
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Solution: Reduce batch size
--batch-size 8                    # Try smaller batch size
--use-mixed-precision             # Enable mixed precision
```

#### 2. Training Not Starting
```bash
# Check data structure
python run_task2.py --help        # Verify script runs
# Check required files exist
ls ./ct_rate_data/multi_abnormality_labels.csv
ls ./ct_rate_2d/splits/train_slices.csv
```

#### 3. GPU Not Being Used
```bash
# Force specific GPU
--gpu-device 1                    # Force NVIDIA GPU
# Check GPU detection
conda run -n svenv python -c "import torch; print(torch.cuda.device_count())"
```

#### 4. Poor Performance
```bash
# Try different loss functions
--loss-type asl                   # Asymmetric Loss
--loss-type bce                   # Binary Cross Entropy

# Enable attention
--use-attention cbam              # CBAM attention

# Progressive unfreezing
--progressive-unfreeze            # Gradual unfreezing
```

### Performance Optimization

#### 1. Memory Optimization
```bash
--batch-size 8                    # Smaller batches
--use-mixed-precision             # FP16 training
--num-workers 2                   # Reduce workers
```

#### 2. Speed Optimization
```bash
--use-mixed-precision             # FP16 training
--batch-size 32                   # Larger batches (if memory allows)
--num-workers 4                   # More workers
```

#### 3. Accuracy Optimization
```bash
--use-attention cbam              # Best attention mechanism
--use-advanced-aug                # Advanced augmentations
--use-multiscale                  # Multi-scale features
--progressive-unfreeze            # Progressive unfreezing
```

---

## Best Practices

### 1. Training Workflow
1. **Start Simple**: Use conservative settings first
2. **Validate**: Run short training to verify setup
3. **Optimize**: Gradually add advanced features
4. **Monitor**: Use TensorBoard for progress tracking
5. **Save**: Keep best checkpoints for comparison

### 2. Hyperparameter Tuning
1. **Learning Rate**: Start with 1e-4, adjust based on loss curve
2. **Batch Size**: Maximize within GPU memory constraints
3. **Epochs**: Use early stopping to prevent overfitting
4. **Loss Function**: Try focal loss first, then experiment with others

### 3. Model Selection
1. **ResNet50**: Good starting point, balanced performance
2. **EfficientNet-B0**: Best for CT-CLIP integration
3. **ResNet101**: Use when you need maximum performance

### 4. GPU Management
1. **Monitor Usage**: Use `nvidia-smi` to track GPU utilization
2. **Memory Management**: Adjust batch size based on available memory
3. **Mixed Precision**: Enable for RTX 4000 series GPUs

---

## Example Workflows

### Workflow 1: Baseline Establishment
```bash
# Step 1: Quick validation
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --epochs 5 \
    --batch-size 8

# Step 2: Baseline training
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --epochs 50 \
    --batch-size 16 \
    --loss-type focal \
    --freeze-backbone

# Step 3: Evaluation
conda run -n svenv python run_task2.py \
    --mode evaluate
```

### Workflow 2: Performance Optimization
```bash
# Step 1: Add attention
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --epochs 50 \
    --batch-size 16 \
    --loss-type focal \
    --use-attention se

# Step 2: Advanced features
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --epochs 50 \
    --batch-size 16 \
    --loss-type asl \
    --use-attention cbam \
    --use-advanced-aug

# Step 3: Progressive unfreezing
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --epochs 100 \
    --batch-size 16 \
    --loss-type asl \
    --use-attention cbam \
    --use-advanced-aug \
    --progressive-unfreeze \
    --unfreeze-epoch 20
```

### Workflow 3: Production Training
```bash
# Full production training
conda run -n svenv python run_task2.py \
    --mode both \
    --gpu-device 1 \
    --model efficientnet_b0 \
    --batch-size 8 \
    --epochs 200 \
    --loss-type asl \
    --use-attention cbam \
    --use-advanced-aug \
    --use-multiscale \
    --progressive-unfreeze \
    --unfreeze-epoch 30 \
    --early-stopping-patience 20
```

---

## Conclusion

This comprehensive guide covers all aspects of the VLM3D Task 2 training pipeline. The system is designed to be flexible and powerful, allowing users to:

1. **Start Simple**: Begin with basic configurations
2. **Scale Up**: Gradually add advanced features
3. **Optimize**: Fine-tune for maximum performance
4. **Monitor**: Track progress and results
5. **Deploy**: Use trained models for inference

For best results, start with conservative settings and gradually experiment with advanced features. Always monitor training progress and use early stopping to prevent overfitting.

### Quick Reference Commands

```bash
# Basic training
conda run -n svenv python run_task2.py --gpu-device 1 --epochs 50

# Advanced training
conda run -n svenv python run_task2.py --gpu-device 1 --use-attention cbam --use-advanced-aug --epochs 100

# Evaluation only
conda run -n svenv python run_task2.py --mode evaluate

# Prediction
conda run -n svenv python run_task2.py --mode predict --predict-input file1.npy --predict-type slices
```

For additional help or issues, refer to the troubleshooting section or check the GPU setup documentation.
