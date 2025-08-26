# VLM3D Task 2: Multi-Abnormality Classification

## Overview

This repository implements a comprehensive solution for **VLM3D Task 2: Multi-Abnormality Classification**. The task involves developing algorithms to output an 18-length binary vector indicating the presence of common thoracic conditions from volumetric chest CT scans.

### Challenge Details
- **Input**: Volumetric chest CT scans
- **Output**: 18-length binary vector for thoracic abnormalities
- **Evaluation**: AUROC, F1, Precision, Recall, and Accuracy
- **Dataset**: CT-RATE dataset with 18 abnormality classes

## 📋 Abnormality Classes

The model predicts the presence of these 18 thoracic conditions:

1. **Atelectasis** - Lung collapse
2. **Cardiomegaly** - Enlarged heart
3. **Consolidation** - Lung tissue solidification
4. **Edema** - Fluid accumulation
5. **Effusion** - Pleural fluid
6. **Emphysema** - Lung tissue damage
7. **Fibrosis** - Lung scarring
8. **Fracture** - Bone fractures
9. **Hernia** - Tissue displacement
10. **Infiltration** - Abnormal tissue infiltration
11. **Mass** - Abnormal masses
12. **Nodule** - Small lung nodules
13. **Pleural_Thickening** - Thickened pleura
14. **Pneumonia** - Lung infection
15. **Pneumothorax** - Collapsed lung
16. **Support_Devices** - Medical devices
17. **Thickening** - General thickening
18. **No_Finding** - No abnormalities

## 🗂️ Project Structure

```
vlm3d-task-2/
├── ct_rate_downloader.py          # Download and prepare CT-RATE dataset
├── 2d_slice_extractor.py          # Extract 2D slices from 3D volumes
├── multi_abnormality_classifier.py # Main model implementation
├── train_multi_abnormality.py     # Training script
├── evaluate_model.py              # Comprehensive evaluation
├── config_multi_abnormality.yaml  # Configuration file
├── requirements_task2.txt         # Dependencies
├── setup_task2.sh                 # Environment setup
└── README_Task2.md                # This file

# Generated directories after running scripts:
├── ct_rate_data/                  # Raw CT data and metadata
│   ├── splits/                    # Train/val/test splits
│   └── ct_rate_volumes/           # Downloaded CT volumes
├── ct_rate_2d/                    # Extracted 2D slices
│   ├── slices/                    # Individual slice files (.npy)
│   └── splits/                    # Slice metadata CSVs
├── checkpoints/                   # Model checkpoints
├── logs/                          # Training logs
├── results/                       # Training results
└── evaluation_results/            # Evaluation outputs
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Run setup script
chmod +x setup_task2.sh
./setup_task2.sh

# Activate environment
source venv_task2/bin/activate
```

### 2. Data Preparation

```bash
# Step 1: Download CT-RATE dataset (adjust --max-storage-gb as needed)
python ct_rate_downloader.py --max-storage-gb 5 --download-volumes

# Step 2: Extract 2D slices from 3D volumes
python 2d_slice_extractor.py --strategy multi_slice --slices-per-volume 12
```

### 3. Model Training

```bash
# Analyze data distribution first
python train_multi_abnormality.py --data-analysis-only

# Create test split
python train_multi_abnormality.py --create-test-split

# Start training
python train_multi_abnormality.py
```

### 4. Model Evaluation

```bash
# Comprehensive evaluation
python evaluate_model.py --checkpoint ./checkpoints/best_model.ckpt
```

## 🔧 Configuration

### Key Configuration Options (`config_multi_abnormality.yaml`)

```yaml
# Model architecture
model:
  backbone: "resnet50"  # resnet50, resnet101, efficientnet_b0, densenet121
  dropout_rate: 0.3

# Training settings
training:
  batch_size: 32
  learning_rate: 1e-4
  max_epochs: 100
  use_focal_loss: true      # Better for imbalanced data
  use_weighted_sampling: true  # Handle class imbalance

# Data paths
data:
  train_csv: "./ct_rate_2d/splits/train_slices.csv"
  val_csv: "./ct_rate_2d/splits/valid_slices.csv"
  test_csv: "./ct_rate_2d/splits/test_slices.csv"
```

## 🏗️ Model Architecture

### Core Components

1. **Backbone**: Pre-trained CNN (ResNet50/101, EfficientNet, DenseNet)
2. **Classifier Head**: Multi-layer perceptron with dropout
3. **Output**: 18 sigmoid outputs for multi-label classification
4. **Loss Function**: Focal Loss or Weighted BCE for class imbalance

### Key Features

- **Multi-label Classification**: Handles multiple simultaneous abnormalities
- **Class Imbalance Handling**: Focal loss and weighted sampling
- **Data Augmentation**: Rotation, flipping, brightness/contrast adjustment
- **Mixed Precision Training**: Memory efficiency and faster training
- **Early Stopping**: Prevents overfitting

## 📊 Evaluation Metrics

The model is evaluated using multiple metrics as required by VLM3D:

### Primary Metrics
- **AUROC** (Area Under ROC Curve) - 30% weight
- **F1 Score** - 25% weight  
- **Precision** - 20% weight
- **Recall** - 15% weight
- **Accuracy** - 10% weight

### Additional Metrics
- **Average Precision (AP)**
- **Subset Accuracy** (exact match)
- **Sample Accuracy** (at least one correct)
- **Hamming Loss**
- **Jaccard Similarity**

### Per-Class Analysis
- Individual metrics for each of the 18 abnormalities
- Class prevalence vs. performance analysis
- ROC and Precision-Recall curves

## 📈 Advanced Usage

### Custom Model Training

```bash
# Train with different backbone
python train_multi_abnormality.py --config custom_config.yaml

# Resume from checkpoint
python train_multi_abnormality.py --resume-from-checkpoint ./checkpoints/last.ckpt
```

### Hyperparameter Tuning

Modify `config_multi_abnormality.yaml`:

```yaml
training:
  learning_rate: 5e-5     # Lower for fine-tuning
  batch_size: 16          # Smaller for limited memory
  use_focal_loss: false   # Try weighted BCE instead
  
model:
  backbone: "efficientnet_b0"  # Different architecture
  dropout_rate: 0.5           # Higher regularization
```

### Cross-Validation

```bash
# The evaluator supports stratified k-fold CV
python evaluate_model.py --checkpoint ./checkpoints/best.ckpt --cv-folds 5
```

## 🔍 Monitoring and Debugging

### TensorBoard Logs

```bash
# View training progress
tensorboard --logdir ./logs
```

### Key Metrics to Monitor
- **Training/Validation Loss**: Should decrease steadily
- **AUROC**: Primary performance indicator
- **Class-wise Performance**: Check for severely underperforming classes
- **Learning Rate**: Should decrease with ReduceLROnPlateau

### Common Issues and Solutions

1. **GPU Memory Issues**
   - Reduce batch size in config
   - Enable gradient accumulation
   - Use mixed precision training

2. **Poor Performance on Rare Classes**
   - Increase focal loss gamma parameter
   - Adjust class weights
   - Use oversampling techniques

3. **Overfitting**
   - Increase dropout rate
   - Add more data augmentation
   - Reduce model complexity

## 📋 Data Requirements

### Expected Data Structure

After running the data preparation scripts:

```
ct_rate_data/
├── splits/
│   ├── train.csv           # Training volume metadata
│   ├── valid.csv           # Validation volume metadata
│   └── test.csv            # Test volume metadata
└── ct_rate_volumes/
    ├── train/              # Training volumes (.nii.gz)
    └── valid/              # Validation volumes (.nii.gz)

ct_rate_2d/
├── splits/
│   ├── train_slices.csv    # Training slice metadata
│   ├── valid_slices.csv    # Validation slice metadata
│   └── test_slices.csv     # Test slice metadata
└── slices/
    ├── train/              # Training slices (.npy)
    ├── valid/              # Validation slices (.npy)
    └── test/               # Test slices (.npy)
```

### Slice Extraction Strategies

1. **Multi-slice** (default): 12 slices per volume, strategically sampled
2. **Best-slice**: Single best slice with highest diagnostic content
3. **Anatomical**: Slices at clinically relevant anatomical levels

## 🎯 Performance Optimization

### Training Speed
- Use GPU with CUDA support
- Enable mixed precision training
- Optimize num_workers for data loading
- Use pin_memory for faster GPU transfer

### Memory Efficiency
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing
- Process slices instead of full volumes

### Model Performance
- Experiment with different backbones
- Try ensemble methods
- Use test-time augmentation
- Fine-tune on specific abnormalities

## 📝 Results and Submission

### Generated Outputs

1. **Model Checkpoints**: `./checkpoints/`
2. **Training Logs**: `./logs/` (TensorBoard format)
3. **Evaluation Results**: `./evaluation_results/`
4. **Visualizations**: ROC curves, performance heatmaps
5. **Predictions**: `./predictions/` (for submission)

### Submission Format

The model generates predictions in the required format for VLM3D evaluation:
- 18-dimensional binary vectors
- Probability scores for each abnormality
- Volume-level and slice-level predictions

## 🤝 Contributing

Feel free to submit issues, feature requests, or improvements. Key areas for enhancement:
- Additional data augmentation techniques
- Novel architectures for multi-label classification
- Advanced class imbalance handling methods
- Ensemble and uncertainty quantification methods

## 📚 References

1. CT-RATE Dataset: [Link to paper/dataset]
2. VLM3D Challenge: [Challenge details]
3. Multi-label Classification: Relevant papers and techniques
4. Medical Image Analysis: Best practices and benchmarks

---

**Happy coding and good luck with the VLM3D challenge! 🏆** 