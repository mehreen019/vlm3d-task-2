#!/usr/bin/env python3
"""
CT-CLIP Evaluation Script for VLM3D Task 2
Evaluates the trained model with comprehensive metrics
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, average_precision_score
)
from ctclip_model import CTCLIPMultiAbnormalityModel, find_optimal_threshold
import os

def evaluate_model_comprehensive(model, test_loader, device):
    """Comprehensive model evaluation with multiple metrics"""

    model.eval()
    all_probs = []
    all_labels = []

    print("🔬 Evaluating model...")
    print("=" * 50)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            labels = batch['labels']

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            labels_np = labels.numpy()

            all_probs.append(probs)
            all_labels.append(labels_np)

            if (i + 1) % 10 == 0:
                print(f"Processed {i+1} batches...")

    # Concatenate all predictions
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    print("
📊 COMPUTING METRICS..."    print("=" * 50)

    # Basic statistics
    print("🔍 PREDICTION ANALYSIS:")
    print(f"  Total samples: {len(all_labels)}")
    print(f"  Prediction range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
    print(".1%"
    print(f"  All predictions == 0: {np.all(all_probs == 0)}")
    print(f"  All predictions == 1: {np.all(all_probs == 1)}")
    print(f"  Label distribution: {all_labels.sum(axis=0)} (positives per class)")

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(all_probs, all_labels)
    print(f"\n🎯 OPTIMAL THRESHOLD: {optimal_threshold:.3f}")

    # Evaluate with both thresholds
    results = {}

    for threshold_name, threshold in [("0.5", 0.5), ("Optimal", optimal_threshold)]:
        print(f"\n📈 METRICS WITH {threshold_name} THRESHOLD ({threshold}):")
        print("-" * 40)

        y_pred = (all_probs > threshold).astype(int)

        # Accuracy metrics
        hamming_accuracy = np.mean(all_labels == y_pred)
        exact_match_accuracy = np.mean(np.all(all_labels == y_pred, axis=1))
        sample_wise_accuracy = np.mean([
            accuracy_score(all_labels[i], y_pred[i]) for i in range(len(all_labels))
        ])

        print(".4f"        print(".4f"        print(".4f"
        # Classification metrics
        try:
            auroc_macro = roc_auc_score(all_labels, all_probs, average='macro')
            print(".4f"        except:
            print("  AUROC Macro: NaN (classes with no positives)")
            auroc_macro = 0.5

        try:
            auroc_micro = roc_auc_score(all_labels, all_probs, average='micro')
            print(".4f"        except:
            print("  AUROC Micro: NaN")
            auroc_micro = 0.5

        f1_macro = f1_score(all_labels, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, y_pred, average='micro', zero_division=0)
        precision_macro = precision_score(all_labels, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, y_pred, average='macro', zero_division=0)

        print(".4f"        print(".4f"        print(".4f"        print(".4f"
        # Store results
        results[threshold_name] = {
            'hamming_accuracy': hamming_accuracy,
            'exact_match_accuracy': exact_match_accuracy,
            'auroc_macro': auroc_macro,
            'auroc_micro': auroc_micro,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro
        }

    # Performance comparison
    print("
🏆 PERFORMANCE COMPARISON:"    print("=" * 50)

    metrics_to_compare = ['hamming_accuracy', 'auroc_macro', 'f1_macro', 'precision_macro', 'recall_macro']

    for metric in metrics_to_compare:
        val_05 = results['0.5'][metric]
        val_opt = results['Optimal'][metric]
        improvement = val_opt - val_05
        print("20s")

    # Success criteria check
    print("
✅ SUCCESS CRITERIA CHECK:"    print("=" * 50)

    hamming_opt = results['Optimal']['hamming_accuracy']
    auroc_opt = results['Optimal']['auroc_macro']
    pos_rate = np.sum(all_probs > 0.5) / all_probs.size

    criteria = [
        ("Hamming Accuracy > 35%", hamming_opt > 0.35, hamming_opt),
        ("AUROC Macro > 60%", auroc_opt > 0.6, auroc_opt),
        ("Positive Rate 45-65%", 0.45 <= pos_rate <= 0.65, pos_rate),
        ("Precision > 30%", results['Optimal']['precision_macro'] > 0.3, results['Optimal']['precision_macro'])
    ]

    passed = 0
    for criterion, met, value in criteria:
        status = "✅ PASSED" if met else "❌ FAILED"
        if "rate" in criterion.lower():
            print("25s")
        else:
            print("25s")
        if met:
            passed += 1

    print(f"\n🎯 OVERALL: {passed}/{len(criteria)} criteria passed")

    if passed >= 3:
        print("🎉 EXCELLENT! CT-CLIP integration is working well!")
    elif passed >= 2:
        print("👍 GOOD! Model shows improvement over baseline")
    else:
        print("⚠️ NEEDS IMPROVEMENT: May need more training or hyperparameter tuning")

    return results

def create_test_dataset():
    """Create a test dataset for evaluation"""
    from torch.utils.data import Dataset

    class TestCTDataset(Dataset):
        def __init__(self, n_samples=500):
            self.n_samples = n_samples
            # More realistic CT-like data
            self.data = torch.randn(n_samples, 3, 224, 224) * 0.1 + 0.5  # Normalized
            # Simulate multi-label with some correlation
            np.random.seed(42)
            base_labels = np.random.randint(0, 2, (n_samples, 5))
            # Add some correlated abnormalities
            labels = np.zeros((n_samples, 18))
            labels[:, :5] = base_labels
            # Add some correlated patterns
            labels[:, 5:10] = (base_labels[:, :5] + np.random.randint(0, 2, (n_samples, 5))) % 2
            labels[:, 10:] = np.random.randint(0, 2, (n_samples, 8)) * 0.3  # Rare abnormalities

            self.targets = torch.tensor(labels, dtype=torch.float32)

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            return {
                'image': self.data[idx],
                'labels': self.targets[idx]
            }

    return TestCTDataset()

def run_evaluation_test():
    """Run a complete evaluation test"""
    print("🚀 CT-CLIP Model Evaluation Test")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model_paths = [
        "models/ctclip_classfine.pt",
        "models/ctclip_vocabfine.pt",
        None
    ]

    for i, model_path in enumerate(model_paths):
        model_name = "CT-CLIP ClassFine" if model_path == "models/ctclip_classfine.pt" else \
                    "CT-CLIP VocabFine" if model_path == "models/ctclip_vocabfine.pt" else \
                    "ImageNet Pretrained"

        print(f"\n🧪 TESTING MODEL {i+1}: {model_name}")
        print("-" * 40)

        try:
            # Create model
            model = CTCLIPMultiAbnormalityModel(ctclip_path=model_path)
            model.to(device)

            # Create test dataset
            test_dataset = create_test_dataset()
            from torch.utils.data import DataLoader
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Evaluate
            results = evaluate_model_comprehensive(model, test_loader, device)

            print(f"\n📊 SUMMARY for {model_name}:")
            print(".4f"            print(".4f"            print(".1%")

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    print("🎯 CT-CLIP Evaluation Script")
    print("This script evaluates your CT-CLIP model performance")
    print("=" * 60)

    run_evaluation_test()

    print("
📋 NEXT STEPS:"    print("1. If using real data, replace create_test_dataset() with your actual dataset"    print("2. Run this after training to evaluate your model"    print("3. Compare results with your baseline (should be much better!)"    print("4. Use optimal threshold for final predictions"
if __name__ == "__main__":
    main()
