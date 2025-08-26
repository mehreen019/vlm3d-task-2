# VLM3D Task 2: Multi-Abnormality Classification - Quick Start

## Overview
This implements **Task 2: Multi-Abnormality Classification** for the VLM3D challenge. The task is to predict 18 different thoracic abnormalities from chest CT scans.

## Prerequisites ✅
You should have already completed:
1. ✅ Data download: `python ct_rate_downloader.py --max-storage-gb 5 --download-volumes`
2. ✅ Slice extraction: `python 2d_slice_extractor.py`
3. ✅ Environment setup: `./setup_env.sh` and activated the conda environment

## Quick Start 🚀

### Step 1: Install Additional Dependencies
```bash
# Activate your existing environment
conda activate vlm3d_challenge

# Install additional packages for Task 2
pip install pytorch-lightning torchmetrics scikit-multilearn imbalanced-learn
```

### Step 2: Run Training & Evaluation (All-in-One)
```bash
# Train and evaluate automatically
python run_task2.py --mode both --epochs 30
```

That's it! The script will:
- ✅ Check your data structure
- ✅ Train the multi-abnormality classification model  
- ✅ Automatically evaluate on the best checkpoint
- ✅ Save results and logs

## Command Options 🔧

### Basic Usage
```bash
# Train only
python run_task2.py --mode train

# Evaluate only (with specific checkpoint)
python run_task2.py --mode evaluate --checkpoint ./checkpoints/model.ckpt

# Custom model and parameters
python run_task2.py --model efficientnet_b0 --batch-size 16 --epochs 50
```

### Advanced Usage
```bash
# For limited GPU memory
python run_task2.py --batch-size 16 --epochs 30

# Different backbone
python run_task2.py --model resnet101

# Quick test run
python run_task2.py --epochs 5
```

## Expected Data Structure 📁
The script expects your data structure from the previous steps:
```
vlm3d-task-2/
├── ct_rate_data/
│   ├── multi_abnormality_labels.csv    # ✅ From downloader
│   └── splits/
│       ├── train.csv
│       └── valid.csv
├── ct_rate_2d/                         # ✅ From slice extractor  
│   ├── splits/
│   │   ├── train_slices.csv
│   │   └── valid_slices.csv
│   └── slices/
│       ├── train/
│       └── valid/
```

## Output Files 📊
After running, you'll get:
```
vlm3d-task-2/
├── checkpoints/           # Trained model weights
├── logs/                  # TensorBoard logs
└── results/              # Evaluation metrics (JSON)
```

## View Training Progress 📈
```bash
# Start TensorBoard
tensorboard --logdir ./logs

# Open in browser: http://localhost:6006
```

## The 18 Abnormality Classes 🏥
1. Atelectasis (lung collapse)
2. Cardiomegaly (enlarged heart)  
3. Consolidation (lung solidification)
4. Edema (fluid accumulation)
5. Effusion (pleural fluid)
6. Emphysema (lung tissue damage)
7. Fibrosis (lung scarring)
8. Fracture (bone fractures)
9. Hernia (tissue displacement)
10. Infiltration (abnormal infiltration)
11. Mass (abnormal masses)
12. Nodule (small nodules)
13. Pleural_Thickening (thickened pleura)
14. Pneumonia (lung infection)
15. Pneumothorax (collapsed lung)
16. Support_Devices (medical devices)
17. Thickening (general thickening)
18. No_Finding (no abnormalities)

## Key Features ⭐
- **Multi-label Classification**: Handles multiple simultaneous abnormalities
- **Class Imbalance Handling**: Focal loss for rare conditions
- **Medical Image Preprocessing**: Optimized for chest CT
- **Automatic Evaluation**: AUROC, F1, Precision, Recall, Accuracy
- **GPU Support**: Automatic detection with mixed precision
- **Progress Monitoring**: TensorBoard integration

## Evaluation Metrics 📊
The model is evaluated using VLM3D challenge metrics:
- **AUROC** (Area Under ROC Curve) - Primary metric
- **F1 Score** - Harmonic mean of precision/recall
- **Precision** - Correct positive predictions
- **Recall** - Found positive cases  
- **Accuracy** - Overall correctness

## Troubleshooting 🔧

### "Missing dependency" error
```bash
conda activate vlm3d_challenge
pip install pytorch-lightning torchmetrics
```

### "Missing required files" error
```bash
# Re-run the data preparation steps
python ct_rate_downloader.py --max-storage-gb 5 --download-volumes
python 2d_slice_extractor.py
```

### GPU out of memory
```bash
python run_task2.py --batch-size 16  # Reduce batch size
```

### All abnormality prevalences are 0.0
This happens if the multi-abnormality labels aren't properly loaded. The script will automatically handle this by looking for the labels in your CT-RATE data directory.

## Integration with Your Workflow 🔄
This builds directly on your existing pipeline:
1. Your **ct_rate_downloader.py** → Downloaded data
2. Your **2d_slice_extractor.py** → Extracted slices  
3. **run_task2.py** → Multi-abnormality classification

No need to change anything from your previous work!

---

**Ready to start? Just run:** `python run_task2.py --mode both --epochs 30` 🚀 