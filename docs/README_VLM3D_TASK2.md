# VLM3D Task 2: Multi-Abnormality Classification
## Complete Training, Evaluation & Prediction Pipeline

> **🎯 Objective**: Develop algorithms to output an 18-length binary vector indicating the presence of common thoracic conditions from volumetric chest CT scans.

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Quick Start](#quick-start)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation Pipeline](#evaluation-pipeline)
6. [Prediction/Inference Pipeline](#predictioninference-pipeline)
7. [Data Structure](#data-structure)
8. [Model Architecture](#model-architecture)
9. [Configuration Options](#configuration-options)
10. [Advanced Usage](#advanced-usage)
11. [Troubleshooting](#troubleshooting)
12. [Results & Metrics](#results--metrics)

---

## Overview

This repository provides a complete solution for **VLM3D Task 2: Multi-Abnormality Classification**. The system can:

- **Train** multi-label classifiers for 18 thoracic abnormalities
- **Evaluate** models using VLM3D challenge metrics (AUROC, F1, Precision, Recall, Accuracy)
- **Predict** abnormalities on new CT data (both 2D slices and 3D volumes)

### 🏥 18 Abnormality Classes

| Class | Description | Class | Description |
|-------|-------------|-------|-------------|
| **Atelectasis** | Lung collapse | **Mass** | Abnormal masses |
| **Cardiomegaly** | Enlarged heart | **Nodule** | Small lung nodules |
| **Consolidation** | Lung solidification | **Pleural_Thickening** | Thickened pleura |
| **Edema** | Fluid accumulation | **Pneumonia** | Lung infection |
| **Effusion** | Pleural fluid | **Pneumothorax** | Collapsed lung |
| **Emphysema** | Lung tissue damage | **Support_Devices** | Medical devices |
| **Fibrosis** | Lung scarring | **Thickening** | General thickening |
| **Fracture** | Bone fractures | **No_Finding** | No abnormalities |
| **Hernia** | Tissue displacement | **Infiltration** | Abnormal infiltration |

---

## Prerequisites & Setup

### ✅ Requirements
1. **Data prepared** using existing CT-RATE pipeline:
   ```bash
   python ct_rate_downloader.py --max-storage-gb 5 --download-volumes
   python 2d_slice_extractor.py
   ```

2. **Environment setup** (conda/mamba environment):
   ```bash
   ./setup_env.sh
   conda activate vlm3d_challenge
   ```

3. **Install Task 2 dependencies**:
   ```bash
   pip install pytorch-lightning torchmetrics scikit-multilearn imbalanced-learn
   ```

### 📁 Expected Data Structure
```
vlm3d-task-2/
├── ct_rate_data/                    # ✅ From ct_rate_downloader.py
│   ├── multi_abnormality_labels.csv
│   └── splits/
│       ├── train.csv
│       └── valid.csv
├── ct_rate_2d/                      # ✅ From 2d_slice_extractor.py
│   ├── splits/
│   │   ├── train_slices.csv
│   │   └── valid_slices.csv
│   └── slices/
│       ├── train/
│       └── valid/
└── Task 2 Files:                    # 🆕 New files for Task 2
    ├── train_multi_abnormality_model.py
    ├── predict_abnormalities.py
    ├── run_task2.py
    └── config_task2.yaml
```

---

## Quick Start

### 🚀 One-Command Training & Evaluation
```bash
# Complete pipeline: train + evaluate
python run_task2.py --mode both --epochs 30

# Quick test (5 epochs)
python run_task2.py --epochs 5
```

### 🔮 Quick Prediction on New Data
```bash
# Predict on slice files
python run_task2.py --mode predict --predict-input slice1.npy slice2.npy --predict-type slices

# Predict on volume files  
python run_task2.py --mode predict --predict-input volume1.nii.gz volume2.nii.gz --predict-type volumes
```

---

## Training Pipeline

### Basic Training
```bash
# Standard training
python run_task2.py --mode train --epochs 50

# With specific model
python run_task2.py --mode train --model efficientnet_b0 --epochs 30

# For limited GPU memory
python run_task2.py --mode train --batch-size 16 --epochs 30
```

### Advanced Training Options
```bash
# Custom hyperparameters
python run_task2.py --mode train \
    --model resnet101 \
    --batch-size 32 \
    --learning-rate 5e-5 \
    --epochs 100

# Direct training script (more control)
python train_multi_abnormality_model.py \
    --data-dir ./ct_rate_data \
    --slice-dir ./ct_rate_2d \
    --model-name resnet50 \
    --batch-size 32 \
    --max-epochs 50
```

### Training Process
1. **Data Loading**: Automatically loads your CT-RATE slice data and labels
2. **Model Initialization**: Creates multi-label classifier with pretrained backbone
3. **Training Loop**: Uses focal loss for class imbalance, early stopping, mixed precision
4. **Checkpointing**: Saves best models automatically
5. **Logging**: TensorBoard logs for monitoring

### Key Training Features
- ✅ **Multi-label Classification**: Handles multiple simultaneous abnormalities
- ✅ **Class Imbalance Handling**: Focal loss for rare conditions
- ✅ **Data Augmentation**: Medical image-specific transforms
- ✅ **Mixed Precision**: Memory efficient training
- ✅ **Early Stopping**: Prevents overfitting
- ✅ **Automatic Checkpointing**: Saves best models

---

## Evaluation Pipeline

### Basic Evaluation
```bash
# Evaluate best model automatically
python run_task2.py --mode evaluate

# Evaluate specific checkpoint
python run_task2.py --mode evaluate --checkpoint ./checkpoints/model.ckpt
```

### Comprehensive Evaluation
```bash
# Using direct evaluation script
python train_multi_abnormality_model.py \
    --mode evaluate \
    --checkpoint ./checkpoints/best_model.ckpt
```

### Evaluation Metrics

The model is evaluated using **VLM3D challenge metrics**:

| Metric | Description | Weight |
|--------|-------------|--------|
| **AUROC** | Area Under ROC Curve | Primary metric |
| **F1 Score** | Harmonic mean of precision/recall | High importance |
| **Precision** | Correct positive predictions | Medium importance |
| **Recall** | Found positive cases | Medium importance |
| **Accuracy** | Overall correctness | Lower importance |

### Output Files
- `./results/evaluation_results.json` - Detailed metrics
- `./logs/` - TensorBoard training logs
- `./checkpoints/` - Model weights

---

## Prediction/Inference Pipeline

After training, you can use the trained model to predict abnormalities on new CT data.

### 🔮 Predict on 2D Slices

```bash
# Single slice
python predict_abnormalities.py \
    --checkpoint ./checkpoints/best_model.ckpt \
    --input slice.npy \
    --data-type slices \
    --output predictions.json

# Multiple slices
python predict_abnormalities.py \
    --checkpoint ./checkpoints/best_model.ckpt \
    --input slice1.npy slice2.npy slice3.npy \
    --data-type slices

# Using run_task2.py
python run_task2.py --mode predict \
    --predict-input *.npy \
    --predict-type slices
```

### 🏥 Predict on 3D Volumes

```bash
# Single volume
python predict_abnormalities.py \
    --checkpoint ./checkpoints/best_model.ckpt \
    --input volume.nii.gz \
    --data-type volumes \
    --output volume_predictions.json

# Multiple volumes
python predict_abnormalities.py \
    --checkpoint ./checkpoints/best_model.ckpt \
    --input volume1.nii.gz volume2.nii.gz \
    --data-type volumes

# Using run_task2.py  
python run_task2.py --mode predict \
    --predict-input *.nii.gz \
    --predict-type volumes
```

### Prediction Output Format

The prediction script generates comprehensive JSON output:

```json
{
  "predictions": [
    {
      "file_path": "slice.npy",
      "probabilities": [0.1, 0.8, 0.3, ...],  // 18 values
      "predictions": [0, 1, 0, ...],            // Binary predictions
      "top_abnormalities": [
        {
          "class_name": "Cardiomegaly",
          "probability": 0.85,
          "predicted": true
        }
      ]
    }
  ],
  "summary": {
    "overall_statistics": {
      "positive_predictions": 3,
      "total_classes": 18
    },
    "top_detected_abnormalities": [...]
  }
}
```

### Volume-Level Predictions

For 3D volumes, the system:
1. **Extracts representative slices** (6 evenly spaced slices)
2. **Predicts on each slice** independently  
3. **Aggregates predictions**:
   - `max_probabilities`: Maximum across all slices
   - `mean_probabilities`: Average across all slices
4. **Provides both slice-level and volume-level results**

---

## Data Structure

### Input Data Requirements

#### For Training/Evaluation:
- **Slice metadata**: `./ct_rate_2d/splits/{train,valid}_slices.csv`
- **Multi-abnormality labels**: `./ct_rate_data/multi_abnormality_labels.csv`  
- **Slice files**: `./ct_rate_2d/slices/{train,valid}/*.npy`

#### For Prediction:
- **2D Slices**: `.npy` files (224x224 preprocessed slices)
- **3D Volumes**: `.nii.gz` files (raw CT volumes)

### Data Loading Process

1. **Slice Metadata Loading**: Reads CSV files with slice information
2. **Label Merging**: Matches volume names to multi-abnormality labels
3. **Image Loading**: Loads .npy slice files with proper preprocessing
4. **Transform Application**: Applies medical image-specific augmentations

---

## Model Architecture

### Core Components

```
Input: 224x224x3 CT slice
    ↓
Backbone CNN (ResNet50/101, EfficientNet-B0)
    ↓  
Global Average Pooling
    ↓
Classifier Head:
  - Linear(features → 512) + ReLU + Dropout
  - Linear(512 → 18) 
    ↓
Sigmoid Activation
    ↓
Output: 18 abnormality probabilities
```

### Supported Backbones
- **ResNet50** (default) - Good balance of speed/accuracy
- **ResNet101** - Higher accuracy, slower training
- **EfficientNet-B0** - Memory efficient, good for limited GPU

### Loss Function
- **Focal Loss** - Handles class imbalance by focusing on hard examples
- **Parameters**: α=0.25, γ=2.0

### Data Augmentation
- Random rotation (±10°)
- Random horizontal flip
- Random affine transformation
- Color jitter (brightness/contrast)
- ImageNet normalization

---

## Configuration Options

### Model Configuration (`config_task2.yaml`)

```yaml
# Model settings
model:
  backbone: "resnet50"        # resnet50, resnet101, efficientnet_b0
  dropout_rate: 0.3
  num_classes: 18

# Training settings  
training:
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-5
  max_epochs: 50
  early_stopping_patience: 10
  use_mixed_precision: true

# Data paths
data:
  ct_rate_dir: "./ct_rate_data"
  slice_dir: "./ct_rate_2d"
```

### Command Line Options

```bash
# Model options
--model {resnet50,resnet101,efficientnet_b0}
--batch-size INT          # Batch size (default: 32)
--learning-rate FLOAT     # Learning rate (default: 1e-4)
--epochs INT              # Number of epochs (default: 50)

# Data options
--data-dir PATH           # CT-RATE data directory
--slice-dir PATH          # Extracted slices directory

# Prediction options  
--predict-input FILE [FILE ...]  # Input files for prediction
--predict-type {slices,volumes}   # Type of input data
--checkpoint PATH                 # Model checkpoint path
```

---

## Advanced Usage

### Custom Training Script

For more control over training, use the direct training script:

```python
from train_multi_abnormality_model import train_model, MultiAbnormalityModel

# Custom training arguments
class Args:
    def __init__(self):
        self.data_dir = "./ct_rate_data"
        self.slice_dir = "./ct_rate_2d"
        self.model_name = "resnet50"
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.max_epochs = 50
        # ... other parameters

args = Args()
model, trainer = train_model(args)
```

### Custom Prediction Pipeline

```python
from predict_abnormalities import AbnormalityPredictor

# Initialize predictor
predictor = AbnormalityPredictor("./checkpoints/best_model.ckpt")

# Predict on slices
slice_paths = ["slice1.npy", "slice2.npy"]
results = predictor.predict_slices(slice_paths)

# Print results
predictor.print_summary(results)
predictor.save_predictions(results, "my_predictions.json")
```

### Hyperparameter Tuning

```bash
# Lower learning rate for fine-tuning
python run_task2.py --learning-rate 5e-5 --epochs 100

# Larger model for better accuracy
python run_task2.py --model resnet101 --batch-size 16

# Quick prototyping
python run_task2.py --epochs 5 --batch-size 16
```

### Ensemble Predictions

```python
# Train multiple models with different seeds/architectures
python run_task2.py --model resnet50 --epochs 50
python run_task2.py --model efficientnet_b0 --epochs 50

# Combine predictions (custom script needed)
```

---

## Troubleshooting

### Common Issues & Solutions

#### 🚨 "Missing dependency" error
```bash
conda activate vlm3d_challenge
pip install pytorch-lightning torchmetrics scikit-multilearn imbalanced-learn
```

#### 🚨 "Missing required files" error
```bash
# Re-run data preparation
python ct_rate_downloader.py --max-storage-gb 5 --download-volumes
python 2d_slice_extractor.py
```

#### 🚨 "CUDA out of memory" error
```bash
# Reduce batch size
python run_task2.py --batch-size 16

# Use gradient accumulation (custom implementation needed)
# Or use CPU training (slower)
python run_task2.py --device cpu
```

#### 🚨 "All abnormality prevalences are 0.0"
This indicates labels aren't properly loaded. Check:
1. `./ct_rate_data/multi_abnormality_labels.csv` exists
2. Volume names match between slice metadata and labels
3. Labels file has proper column names

#### 🚨 Model not converging
```bash
# Lower learning rate
python run_task2.py --learning-rate 5e-5

# More epochs
python run_task2.py --epochs 100

# Different model
python run_task2.py --model efficientnet_b0
```

#### 🚨 Prediction errors
```bash
# Check file paths exist
ls ./checkpoints/  # Verify checkpoint exists
ls slice.npy       # Verify input files exist

# Check file formats
file slice.npy     # Should be NumPy array
file volume.nii.gz # Should be NIfTI volume
```

### Debug Mode

```bash
# Enable verbose logging
export PYTHONPATH="."
python -u run_task2.py --mode train --epochs 1  # Quick test
```

### Performance Optimization

```bash
# Optimize for speed
python run_task2.py --batch-size 64 --num-workers 8

# Optimize for memory
python run_task2.py --batch-size 16 --use-mixed-precision

# Use different device
python predict_abnormalities.py --device cpu  # Force CPU
python predict_abnormalities.py --device cuda # Force GPU
```

---

## Results & Metrics

### Training Monitoring

#### TensorBoard Visualization
```bash
# Start TensorBoard
tensorboard --logdir ./logs

# Open in browser
http://localhost:6006
```

**Key metrics to monitor:**
- **Training/Validation Loss**: Should decrease steadily
- **AUROC**: Primary performance indicator (target: >0.8)
- **F1 Score**: Balance between precision/recall
- **Learning Rate**: Should decrease with ReduceLROnPlateau

#### Real-time Monitoring
```bash
# Watch training progress
tail -f logs/*/events.out.tfevents.*

# Monitor GPU usage
nvidia-smi -l 1
```

### Expected Performance

| Model | Training Time | AUROC | F1 Score | Memory Usage |
|-------|---------------|-------|----------|---------------|
| ResNet50 | ~2-4 hours | 0.75-0.85 | 0.60-0.75 | ~6GB GPU |
| ResNet101 | ~3-6 hours | 0.78-0.88 | 0.65-0.78 | ~8GB GPU |
| EfficientNet-B0 | ~2-3 hours | 0.77-0.87 | 0.63-0.76 | ~4GB GPU |

*Performance varies based on data size, hardware, and hyperparameters*

### Output Files Structure

```
vlm3d-task-2/
├── checkpoints/
│   ├── multi_abnormality-epoch=XX-val_loss=X.XXX.ckpt  # Best models
│   └── last.ckpt                                       # Latest checkpoint
├── logs/
│   └── multi_abnormality_classification/
│       └── version_X/                                  # TensorBoard logs
├── results/
│   ├── evaluation_results.json                        # Detailed metrics
│   └── predictions_slices.json                        # Prediction results
└── predictions_*.json                                 # Inference outputs
```

### Evaluation Results Format

```json
{
  "auroc_macro": 0.823,
  "auroc_micro": 0.834,
  "f1_macro": 0.671,
  "f1_micro": 0.698,
  "precision_macro": 0.702,
  "recall_macro": 0.645,
  "accuracy": 0.892
}
```

---

## 🎯 Summary

This comprehensive pipeline provides:

1. **🏋️ Training**: Multi-label classification with class imbalance handling
2. **📊 Evaluation**: VLM3D-compliant metrics and visualizations  
3. **🔮 Prediction**: Inference on new CT slices and volumes
4. **🔧 Flexibility**: Multiple models, hyperparameters, and usage modes
5. **📈 Monitoring**: TensorBoard integration and progress tracking

### Quick Commands Reference

```bash
# Complete pipeline
python run_task2.py --mode both --epochs 30

# Training only
python run_task2.py --mode train --model efficientnet_b0

# Evaluation only  
python run_task2.py --mode evaluate --checkpoint path/to/model.ckpt

# Prediction on slices
python run_task2.py --mode predict --predict-input *.npy --predict-type slices

# Prediction on volumes
python run_task2.py --mode predict --predict-input *.nii.gz --predict-type volumes

# Monitor training
tensorboard --logdir ./logs
```

---

**🚀 Ready to classify thoracic abnormalities? Start with:** `python run_task2.py --mode both --epochs 30`

For questions or issues, check the [Troubleshooting](#troubleshooting) section or examine the training logs in `./logs/`. 