#!/usr/bin/env python3
"""
Download CT-CLIP Models for VLM3D Task 2 Integration
Simple script to get CT-CLIP pretrained models
"""

import os
import urllib.request
from pathlib import Path

def download_file(url, filename, description=""):
    """Download file with progress"""
    print(f"⬇️ Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Saving to: {filename}")

    try:
        urllib.request.urlretrieve(url, filename)
        print("✅ Download complete!")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    print("🩻 CT-CLIP Model Downloader")
    print("=" * 50)
    print("Downloading CT-CLIP pretrained models for your VLM3D Task 2")
    print()

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # CT-CLIP model URLs (from official repository)
    models = {
        "ctclip_classfine.pt": {
            "url": "https://huggingface.co/ibrahimethemhamamci/CT-CLIP/resolve/main/ctclip_classfine.pt",
            "description": "CT-CLIP ClassFine (Fastest - 0.5s inference, recommended)",
            "size": "~500MB"
        },
        "ctclip_vocabfine.pt": {
            "url": "https://huggingface.co/ibrahimethemhamamci/CT-CLIP/resolve/main/ctclip_vocabfine.pt",
            "description": "CT-CLIP VocabFine (Balanced performance)",
            "size": "~500MB"
        }
    }

    print("📦 Available models:")
    for name, info in models.items():
        print(f"   • {name}: {info['description']} ({info['size']})")

    print(f"\n📁 Download directory: {models_dir.absolute()}")
    print()

    # Download models
    downloaded_models = []

    for model_name, model_info in models.items():
        filepath = models_dir / model_name

        if filepath.exists():
            print(f"✅ {model_name} already exists, skipping...")
            downloaded_models.append(model_name)
            continue

        success = download_file(
            model_info["url"],
            str(filepath),
            f"{model_name} - {model_info['description']}"
        )

        if success:
            downloaded_models.append(model_name)

        print()

    # Summary
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 50)

    if downloaded_models:
        print(f"✅ Successfully downloaded: {len(downloaded_models)} models")
        for model in downloaded_models:
            filepath = models_dir / model
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(".1f"
        print("
🎯 Ready to use!"        print("   Your models are in the 'models/' directory"
        print(f"   Total models available: {len(downloaded_models)}")

        print("
🚀 Next steps:"        print("1. Run: python integrate_ctclip.py"        print("2. Or modify your existing run_task2.py to use CT-CLIP"
    else:
        print("❌ No models downloaded successfully")
        print("🔧 Troubleshooting:")
        print("   • Check internet connection")
        print("   • Verify HuggingFace is accessible")
        print("   • Try downloading manually from the repository"

    print("
🔗 CT-CLIP Repository: https://github.com/ibrahimethemhamamci/CT-CLIP"
if __name__ == "__main__":
    main()
