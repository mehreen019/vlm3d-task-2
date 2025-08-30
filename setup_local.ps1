# --- PyTorch + CUDA (adjust version if needed) ---
conda install -n svenv pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# (for CPU-only, instead use: conda install -n svenv pytorch torchvision torchaudio cpuonly -c pytorch -y)

# --- Medical imaging libraries ---
conda run -n svenv pip install SimpleITK
conda run -n svenv pip install nibabel
conda run -n svenv pip install pydicom
conda run -n svenv pip install "monai[all]"

# --- Deep learning & ML libraries ---
conda run -n svenv pip install timm
conda run -n svenv pip install transformers
conda run -n svenv pip install lightning
conda run -n svenv pip install wandb
conda run -n svenv pip install albumentations

# --- Scientific computing ---
conda run -n svenv pip install pandas
conda run -n svenv pip install numpy
conda run -n svenv pip install scikit-learn
conda run -n svenv pip install scipy
conda run -n svenv pip install matplotlib
conda run -n svenv pip install seaborn
conda run -n svenv pip install plotly
conda run -n svenv pip install opencv-python

# --- Utilities ---
conda run -n svenv pip install tqdm
conda run -n svenv pip install rich
conda run -n svenv pip install typer
conda run -n svenv pip install pydantic
conda run -n svenv pip install omegaconf
conda run -n svenv pip install tensorboard

# --- Jupyter ---
conda run -n svenv pip install jupyter
conda run -n svenv pip install ipywidgets

# --- Medical AI specific ---
conda run -n svenv pip install radiomics
conda run -n svenv pip install pyradiomics
conda run -n svenv pip install intensity-normalization

# --- Dev tools ---
conda run -n svenv pip install black
conda run -n svenv pip install isort
conda run -n svenv pip install flake8
conda run -n svenv pip install pytest

# --- Extra ---
conda run -n svenv pip install requests
conda run -n svenv pip install huggingface_hub

# --- Multi-label classification ---
conda run -n svenv pip install pytorch-lightning
conda run -n svenv pip install torchmetrics
conda run -n svenv pip install scikit-multilearn
conda run -n svenv pip install imbalanced-learn
