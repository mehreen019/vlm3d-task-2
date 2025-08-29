#!/usr/bin/env python3
"""
Download CT-CLIP Models for VLM3D Task 2 Integration
Uses huggingface_hub for proper authentication
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

def main():
    print("CT-CLIP Model Downloader")
    print("=" * 50)
    print("Downloading CT-CLIP pretrained models for your VLM3D Task 2\n")

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # HuggingFace dataset repo + model files
    repo_id = "ibrahimhamamci/CT-RATE"
    repo_type = "dataset"  # important: these are in a dataset repo

    models = {
        "ctclip_classfine.pt": {
            "filename": "models/CT-CLIP-Related/CT_LiPro_v2.pt",
            "description": "CT-CLIP ClassFine (Fastest - 0.5s inference, recommended)",
        },
        "ctclip_vocabfine.pt": {
            "filename": "models/CT-CLIP-Related/CT_VocabFine_v2.pt",
            "description": "CT-CLIP VocabFine (Balanced performance)",
        }
    }

    downloaded_models = []

    for local_name, model_info in models.items():
        print(f"📥 Downloading {local_name}: {model_info['description']}")

        try:
            filepath = hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=model_info["filename"],
                local_dir=str(models_dir)  # saves file inside models/
            )
            print(f"✅ Saved to: {filepath}\n")
            downloaded_models.append(local_name)
        except Exception as e:
            print(f"❌ Failed to download {local_name}: {e}\n")

    print("=" * 50)
    if downloaded_models:
        print(f"Successfully downloaded {len(downloaded_models)} models to {models_dir.absolute()}")
        for model in downloaded_models:
            path = models_dir / model
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f" - {model}: {size_mb:.2f} MB")
    else:
        print("No models were downloaded. Please check your Hugging Face login and URLs.")

    print("\nCT-CLIP Repository: https://github.com/ibrahimethemhamamci/CT-CLIP")

if __name__ == "__main__":
    main()
