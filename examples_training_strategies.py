#!/usr/bin/env python3
"""
Example Training Strategies for Enhanced VLM3D Task 2
Demonstrates different configurations for optimal performance
"""

import os
import subprocess

def run_command(cmd):
    """Run shell command and return success status"""
    print(f"🚀 Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        return False

# Strategy 1: Fast Training (Good Performance)
def strategy_fast_training():
    """Fast training with backbone freezing"""
    cmd = """
    python run_task2.py \
      --model efficientnet_b0 \
      --freeze-backbone \
      --use-attention se \
      --loss-type asl \
      --batch-size 64 \
      --learning-rate 2e-4 \
      --epochs 50
    """
    return run_command(cmd.strip())

# Strategy 2: Maximum Performance (Slower)
def strategy_max_performance():
    """Maximum performance with all features"""
    cmd = """
    python run_task2.py \
      --model efficientnet_b0 \
      --freeze-backbone \
      --use-attention cbam \
      --loss-type asl \
      --use-advanced-aug \
      --progressive-unfreeze \
      --unfreeze-epoch 8 \
      --batch-size 32 \
      --learning-rate 1e-4 \
      --epochs 100
    """
    return run_command(cmd.strip())

# Strategy 3: Ensemble Training
def strategy_ensemble():
    """Train multiple models for ensemble"""
    models = [
        {
            "name": "efficientnet_cbam_asl",
            "cmd": "python run_task2.py --model efficientnet_b0 --use-attention cbam --loss-type asl --epochs 80"
        },
        {
            "name": "resnet50_se_focal",
            "cmd": "python run_task2.py --model resnet50 --use-attention se --loss-type focal --epochs 80"
        },
        {
            "name": "resnet101_cbam_asl",
            "cmd": "python run_task2.py --model resnet101 --use-attention cbam --loss-type asl --epochs 80"
        }
    ]

    for model in models:
        print(f"\n🏗️ Training {model['name']}...")
        if not run_command(model['cmd']):
            print(f"⚠️ Failed to train {model['name']}, continuing...")

# Strategy 4: Ablation Study
def strategy_ablation_study():
    """Systematic comparison of different features"""
    configs = [
        ("baseline", "python run_task2.py --model resnet50 --epochs 30"),
        ("freeze_backbone", "python run_task2.py --model resnet50 --freeze-backbone --epochs 30"),
        ("attention_se", "python run_task2.py --model resnet50 --use-attention se --epochs 30"),
        ("attention_cbam", "python run_task2.py --model resnet50 --use-attention cbam --epochs 30"),
        ("loss_asl", "python run_task2.py --model resnet50 --loss-type asl --epochs 30"),
        ("advanced_aug", "python run_task2.py --model resnet50 --use-advanced-aug --epochs 30"),
        ("all_features", "python run_task2.py --model resnet50 --freeze-backbone --use-attention cbam --loss-type asl --use-advanced-aug --epochs 30")
    ]

    for name, cmd in configs:
        print(f"\n🔬 Testing {name}...")
        if not run_command(cmd):
            print(f"⚠️ Failed {name}, continuing...")

# Strategy 5: Memory-Constrained Training
def strategy_memory_constrained():
    """For systems with limited GPU memory"""
    cmd = """
    python run_task2.py \
      --model efficientnet_b0 \
      --freeze-backbone \
      --batch-size 16 \
      --use-attention se \
      --loss-type focal \
      --learning-rate 1e-4 \
      --epochs 50
    """
    return run_command(cmd.strip())

# Strategy 6: High-Performance Training
def strategy_high_performance():
    """For systems with ample resources"""
    cmd = """
    python run_task2.py \
      --model efficientnet_b0 \
      --freeze-backbone \
      --use-attention cbam \
      --loss-type asl \
      --use-advanced-aug \
      --progressive-unfreeze \
      --unfreeze-epoch 5 \
      --batch-size 128 \
      --learning-rate 3e-4 \
      --epochs 150 \
      --early-stopping-patience 20
    """
    return run_command(cmd.strip())

if __name__ == "__main__":
    print("🎯 Enhanced VLM3D Task 2 Training Strategies")
    print("=" * 50)

    # Choose your strategy
    strategies = {
        "1": ("Fast Training", strategy_fast_training),
        "2": ("Maximum Performance", strategy_max_performance),
        "3": ("Ensemble Training", strategy_ensemble),
        "4": ("Ablation Study", strategy_ablation_study),
        "5": ("Memory Constrained", strategy_memory_constrained),
        "6": ("High Performance", strategy_high_performance)
    }

    print("Available strategies:")
    for key, (name, _) in strategies.items():
        print(f"  {key}: {name}")

    choice = input("\nChoose strategy (1-6): ").strip()

    if choice in strategies:
        name, func = strategies[choice]
        print(f"\n🚀 Running {name}...")
        func()
    else:
        print("❌ Invalid choice. Please run with a specific strategy:")
        print("  python examples_training_strategies.py")
        print("Then choose 1-6 when prompted.")
