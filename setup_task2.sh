#!/bin/bash
# Setup script for VLM3D Task 2: Multi-Abnormality Classification
# This script sets up the environment for training and evaluating the model

set -e  # Exit on any error

echo "🚀 Setting up VLM3D Task 2: Multi-Abnormality Classification"
echo "============================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "🐍 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv_task2" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv_task2
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv_task2/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (with CUDA support if available)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "📚 Installing other dependencies..."
pip install -r requirements_task2.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p results
mkdir -p predictions
mkdir -p evaluation_results

# Verify installation
echo "✅ Verifying installation..."
python3 -c "
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import yaml
print('✅ All dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: source venv_task2/bin/activate"
echo "2. Ensure you have run the data downloader and slice extractor"
echo "3. Check your data structure:"
echo "   - ./ct_rate_data/splits/train.csv"
echo "   - ./ct_rate_data/splits/valid.csv" 
echo "   - ./ct_rate_2d/splits/train_slices.csv"
echo "   - ./ct_rate_2d/splits/valid_slices.csv"
echo "4. Run data analysis: python train_multi_abnormality.py --data-analysis-only"
echo "5. Start training: python train_multi_abnormality.py"
echo ""
echo "📖 Key files created:"
echo "   - multi_abnormality_classifier.py (main model implementation)"
echo "   - config_multi_abnormality.yaml (configuration)"
echo "   - train_multi_abnormality.py (training script)"
echo "   - evaluate_model.py (evaluation script)"
echo ""
echo "🔧 Usage examples:"
echo "   # Train model"
echo "   python train_multi_abnormality.py"
echo ""
echo "   # Evaluate model"
echo "   python evaluate_model.py --checkpoint ./checkpoints/best_model.ckpt"
echo ""
echo "   # Custom configuration"
echo "   python train_multi_abnormality.py --config custom_config.yaml" 