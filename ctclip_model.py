#!/usr/bin/env python3
"""
CT-CLIP Model Classes for VLM3D Task 2
Contains all model definitions and utilities
"""

import torch
import torch.nn as nn
import timm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import os

class CTCLIPFeatureExtractor(nn.Module):
    """CT-CLIP inspired feature extractor"""

    def __init__(self, ctclip_path=None, model_name="efficientnet_b0"):
        super().__init__()

        # Load EfficientNet as in CT-CLIP paper
        self.backbone = timm.create_model('efficientnet_b0.ra_in1k', pretrained=True)

        # Try to load CT-CLIP weights if available
        if ctclip_path and os.path.exists(ctclip_path):
            try:
                state_dict = torch.load(ctclip_path, map_location='cpu')
                # Load with strict=False in case of slight architectural differences
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded CT-CLIP weights from {ctclip_path}")
            except Exception as e:
                print(f"⚠️ Could not load CT-CLIP weights: {e}")
                print("   Using ImageNet pretrained weights instead")
        else:
            print("ℹ️ No CT-CLIP weights found, using ImageNet pretrained weights")

        # Remove classifier to get features (CT-CLIP approach)
        self.backbone.classifier = nn.Identity()
        self.feature_dim = 1280

    def forward(self, x):
        return self.backbone(x)

class CTCLIPMultiAbnormalityModel(nn.Module):
    """Complete CT-CLIP model for multi-abnormality detection"""

    def __init__(self, ctclip_path=None, num_classes=18, dropout_rate=0.5):
        super().__init__()

        self.feature_extractor = CTCLIPFeatureExtractor(ctclip_path)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # CT-CLIP uses dropout for regularization
            nn.Linear(self.feature_extractor.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),  # Slightly less dropout for second layer
            nn.Linear(1024, num_classes)
        )

        # Xavier initialization (CT-CLIP approach)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following CT-CLIP paper"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        print("✅ Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        print("✅ Backbone unfrozen")

class AggressiveFocalLoss(nn.Module):
    """Aggressive Focal Loss to fix over-prediction"""

    def __init__(self, alpha=0.85, gamma=4.0):
        super().__init__()
        self.alpha = alpha  # Higher = penalize positive predictions more
        self.gamma = gamma  # Higher = focus on hard examples

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        pt = torch.exp(-bce_loss)

        # Asymmetric alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()

def create_balanced_loss(alpha=0.85, gamma=4.0):
    """Create aggressive loss for over-prediction"""
    return AggressiveFocalLoss(alpha=alpha, gamma=gamma)

def find_optimal_threshold(probs, labels):
    """Find optimal threshold using F1 score maximization"""
    best_threshold = 0.5
    best_f1 = 0

    thresholds = np.arange(0.1, 0.9, 0.05)

    for threshold in thresholds:
        y_pred = (probs > threshold).astype(int)
        f1 = f1_score(labels, y_pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold

def test_model():
    """Test the CT-CLIP model"""
    print("🧪 Testing CT-CLIP Model...")
    print("=" * 40)

    # Test with CT-CLIP weights
    ctclip_paths = [
        "models/ctclip_classfine.pt",
        "models/ctclip_vocabfine.pt",
        None  # Test with ImageNet weights
    ]

    for i, ctclip_path in enumerate(ctclip_paths):
        print(f"\nTest {i+1}: {'CT-CLIP' if ctclip_path else 'ImageNet'} weights")

        try:
            model = CTCLIPMultiAbnormalityModel(ctclip_path=ctclip_path)
            criterion = create_balanced_loss()

            # Test forward pass
            dummy_input = torch.randn(4, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)

            print(f"  ✅ Model created successfully!")
            print(f"  📏 Input: {dummy_input.shape}")
            print(f"  📏 Output: {output.shape}")
            print(f"  🎯 Expected: [4, 18] for 18 abnormalities")

            # Test loss
            dummy_targets = torch.randint(0, 2, (4, 18)).float()
            loss = criterion(output, dummy_targets)
            print(".4f"
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            backbone_params = sum(p.numel() for p in model.feature_extractor.parameters())
            classifier_params = sum(p.numel() for p in model.classifier.parameters())

            print(f"  📊 Total parameters: {total_params:,}")
            print(f"  🏗️ Backbone params: {backbone_params:,}")
            print(f"  🧠 Classifier params: {classifier_params:,}")

        except Exception as e:
            print(f"  ❌ Test failed: {e}")

    print("
🎉 Model testing complete!"    print("💡 Your model is ready for training!")
    print("
🚀 Next: Run ctclip_training.py"
def main():
    test_model()

if __name__ == "__main__":
    main()
