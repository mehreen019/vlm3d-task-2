#!/bin/bash
# Environment Setup Script for VLM3D Challenge Task 2
# This script sets up a complete Python environment for medical imaging and deep learning

set -e  # Exit on any error

echo "🏥 Setting up VLM3D Challenge environment..."

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "❌ Error: Neither conda nor mamba found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Create conda environment
ENV_NAME="vlm3d_challenge"
PYTHON_VERSION="3.9"

echo "📦 Creating conda environment: $ENV_NAME"
$CONDA_CMD create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install PyTorch (adjust CUDA version as needed)
echo "🔥 Installing PyTorch..."
# For CUDA 11.8 (adjust version based on your GPU)
$CONDA_CMD install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CPU-only (uncomment if you don't have GPU)
# $CONDA_CMD install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install medical imaging libraries
echo "🏥 Installing medical imaging libraries..."
pip install SimpleITK
pip install nibabel
pip install pydicom
pip install monai[all]

# Install deep learning and ML libraries
echo "🧠 Installing ML/DL libraries..."
pip install timm  # Pre-trained models
pip install transformers
pip install lightning  # PyTorch Lightning for clean training loops
pip install wandb  # For experiment tracking
pip install albumentations  # Data augmentation

# Install scientific computing libraries
echo "📊 Installing scientific libraries..."
pip install pandas
pip install numpy
pip install scikit-learn
pip install scipy
pip install matplotlib
pip install seaborn
pip install plotly
pip install opencv-python

# Install utilities
echo "🛠️ Installing utilities..."
pip install tqdm
pip install rich  # Beautiful terminal output
pip install typer  # CLI creation
pip install pydantic  # Data validation
pip install omegaconf  # Configuration management
pip install tensorboard

# Install Jupyter for experimentation
echo "📓 Installing Jupyter..."
pip install jupyter
pip install ipywidgets

# Install medical AI specific libraries
echo "🔬 Installing specialized libraries..."
pip install radiomics  # Feature extraction
pip install pyradiomics
pip install intensity-normalization  # CT intensity normalization

# Development tools
echo "🔧 Installing development tools..."
pip install black  # Code formatting
pip install isort  # Import sorting
pip install flake8  # Linting
pip install pytest  # Testing

# Install additional useful libraries
echo "📦 Installing additional libraries..."
pip install requests  # HTTP requests
pip install huggingface_hub  # For model sharing

echo "✅ Environment setup complete!"
echo ""
echo "📋 Summary of installed packages:"
echo "   - PyTorch with CUDA support"
echo "   - Medical imaging: SimpleITK, nibabel, MONAI"
echo "   - ML/DL: timm, transformers, lightning"
echo "   - Scientific: pandas, numpy, scikit-learn"
echo "   - Visualization: matplotlib, seaborn, plotly"
echo "   - Utilities: tqdm, wandb, rich"
echo "   - Development: black, isort, flake8"
echo ""
echo "🚀 To activate the environment, run:"
echo "   conda activate $ENV_NAME"
echo ""
echo "📝 Next steps:"
echo "   1. Run the CT-RATE downloader script"
echo "   2. Start with baseline model implementation"
echo "   3. Begin training and experimentation"