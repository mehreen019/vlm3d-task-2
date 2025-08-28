#!/usr/bin/env python3
"""
Ensemble Methods and Test-Time Augmentation for CT Medical Image Classification
Implements advanced ensemble techniques for improved model performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pickle

logger = logging.getLogger(__name__)


class TestTimeAugmentation:
    """Test-Time Augmentation for CT scans"""
    
    def __init__(self, 
                 num_augmentations: int = 8,
                 rotation_angles: List[float] = [-10, -5, 0, 5, 10],
                 flip_horizontal: bool = True,
                 brightness_factors: List[float] = [0.9, 1.0, 1.1],
                 crop_ratios: List[float] = [0.95, 1.0]):
        
        self.num_augmentations = num_augmentations
        self.rotation_angles = rotation_angles
        self.flip_horizontal = flip_horizontal
        self.brightness_factors = brightness_factors
        self.crop_ratios = crop_ratios
        
        # Standard normalization for pretrained models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    def generate_augmentations(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Generate augmented versions of the input image"""
        augmented_images = []
        
        # Original image
        augmented_images.append(image)
        
        # Convert to PIL for easier manipulation
        if len(image.shape) == 3:
            image_pil = transforms.ToPILImage()(image)
        else:
            image_pil = transforms.ToPILImage()(image.unsqueeze(0))
        
        count = 1
        
        # Rotation augmentations
        for angle in self.rotation_angles:
            if count >= self.num_augmentations:
                break
            if angle != 0:
                rotated = transforms.functional.rotate(image_pil, angle)
                rotated_tensor = transforms.ToTensor()(rotated)
                if rotated_tensor.shape[0] == 1:
                    rotated_tensor = rotated_tensor.repeat(3, 1, 1)
                augmented_images.append(self.normalize(rotated_tensor))
                count += 1
        
        # Horizontal flip
        if self.flip_horizontal and count < self.num_augmentations:
            flipped = transforms.functional.hflip(image_pil)
            flipped_tensor = transforms.ToTensor()(flipped)
            if flipped_tensor.shape[0] == 1:
                flipped_tensor = flipped_tensor.repeat(3, 1, 1)
            augmented_images.append(self.normalize(flipped_tensor))
            count += 1
        
        # Brightness adjustments
        for brightness in self.brightness_factors:
            if count >= self.num_augmentations:
                break
            if brightness != 1.0:
                bright = transforms.functional.adjust_brightness(image_pil, brightness)
                bright_tensor = transforms.ToTensor()(bright)
                if bright_tensor.shape[0] == 1:
                    bright_tensor = bright_tensor.repeat(3, 1, 1)
                augmented_images.append(self.normalize(bright_tensor))
                count += 1
        
        # Center crops with different ratios
        for crop_ratio in self.crop_ratios:
            if count >= self.num_augmentations:
                break
            if crop_ratio != 1.0:
                w, h = image_pil.size
                new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
                cropped = transforms.functional.center_crop(image_pil, (new_h, new_w))
                resized = transforms.functional.resize(cropped, (h, w))
                resized_tensor = transforms.ToTensor()(resized)
                if resized_tensor.shape[0] == 1:
                    resized_tensor = resized_tensor.repeat(3, 1, 1)
                augmented_images.append(self.normalize(resized_tensor))
                count += 1
        
        # Pad to required number of augmentations if needed
        while len(augmented_images) < self.num_augmentations:
            augmented_images.append(augmented_images[0])
        
        return augmented_images[:self.num_augmentations]
    
    def aggregate_predictions(self, predictions: List[torch.Tensor], 
                            method: str = "mean") -> torch.Tensor:
        """Aggregate predictions from multiple augmentations"""
        stacked_preds = torch.stack(predictions, dim=0)
        
        if method == "mean":
            return torch.mean(stacked_preds, dim=0)
        elif method == "max":
            return torch.max(stacked_preds, dim=0)[0]
        elif method == "median":
            return torch.median(stacked_preds, dim=0)[0]
        elif method == "weighted_mean":
            # Give more weight to less augmented versions
            weights = torch.tensor([2.0] + [1.0] * (len(predictions) - 1))
            weights = weights / weights.sum()
            weights = weights.view(-1, 1, 1)  # Reshape for broadcasting
            return torch.sum(stacked_preds * weights, dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


class ModelEnsemble:
    """Ensemble of multiple trained models"""
    
    def __init__(self, model_paths: List[str], device: str = "cuda"):
        self.model_paths = model_paths
        self.device = device
        self.models = []
        self.load_models()
    
    def load_models(self):
        """Load all models from checkpoints"""
        for model_path in self.model_paths:
            try:
                # Import the enhanced model
                from train_enhanced_model import EnhancedMultiAbnormalityModel
                
                # Load model
                model = EnhancedMultiAbnormalityModel.load_from_checkpoint(model_path)
                model.eval()
                model.to(self.device)
                self.models.append(model)
                logger.info(f"Loaded model from {model_path}")
                
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Generate ensemble predictions"""
        if not self.models:
            raise RuntimeError("No models loaded")
        
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = torch.sigmoid(model(x))
                predictions.append(pred)
        
        # Simple average ensemble
        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        
        return ensemble_pred
    
    def predict_with_tta(self, x: torch.Tensor, tta: TestTimeAugmentation) -> torch.Tensor:
        """Generate ensemble predictions with test-time augmentation"""
        if not self.models:
            raise RuntimeError("No models loaded")
        
        batch_size = x.shape[0]
        all_predictions = []
        
        with torch.no_grad():
            for i in range(batch_size):
                image = x[i]
                
                # Generate augmented versions
                augmented_images = tta.generate_augmentations(image)
                
                # Get predictions from all models for all augmentations
                image_predictions = []
                
                for model in self.models:
                    model_aug_preds = []
                    
                    for aug_image in augmented_images:
                        aug_batch = aug_image.unsqueeze(0).to(self.device)
                        pred = torch.sigmoid(model(aug_batch))
                        model_aug_preds.append(pred.squeeze(0))
                    
                    # Aggregate TTA predictions for this model
                    model_tta_pred = tta.aggregate_predictions(model_aug_preds, method="weighted_mean")
                    image_predictions.append(model_tta_pred)
                
                # Ensemble the model predictions
                ensemble_pred = torch.stack(image_predictions, dim=0).mean(dim=0)
                all_predictions.append(ensemble_pred)
        
        return torch.stack(all_predictions, dim=0)


class AdaptiveEnsemble:
    """Adaptive ensemble that weights models based on confidence"""
    
    def __init__(self, model_paths: List[str], device: str = "cuda"):
        self.ensemble = ModelEnsemble(model_paths, device)
        self.model_weights = torch.ones(len(model_paths)) / len(model_paths)
        self.confidence_threshold = 0.7
    
    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions with confidence scores"""
        if not self.ensemble.models:
            raise RuntimeError("No models loaded")
        
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for model in self.ensemble.models:
                pred = torch.sigmoid(model(x))
                predictions.append(pred)
                
                # Calculate confidence as max probability minus entropy
                entropy = -torch.sum(pred * torch.log(pred + 1e-8), dim=1)
                max_prob = torch.max(pred, dim=1)[0]
                confidence = max_prob - entropy / np.log(pred.shape[1])
                confidences.append(confidence)
        
        predictions = torch.stack(predictions, dim=0)  # [num_models, batch_size, num_classes]
        confidences = torch.stack(confidences, dim=0)  # [num_models, batch_size]
        
        # Weight predictions by confidence
        conf_weights = F.softmax(confidences, dim=0)  # [num_models, batch_size]
        conf_weights = conf_weights.unsqueeze(-1)  # [num_models, batch_size, 1]
        
        weighted_pred = torch.sum(predictions * conf_weights, dim=0)
        avg_confidence = torch.mean(confidences, dim=0)
        
        return weighted_pred, avg_confidence
    
    def update_weights(self, validation_loader: DataLoader):
        """Update model weights based on validation performance"""
        model_scores = []
        
        for model in self.ensemble.models:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in validation_loader:
                    images = batch['image'].to(self.ensemble.device)
                    labels = batch['labels'].to(self.ensemble.device)
                    
                    outputs = torch.sigmoid(model(images))
                    predicted = (outputs > 0.5).float()
                    
                    correct += (predicted == labels).float().mean().item()
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            model_scores.append(accuracy)
        
        # Update weights based on performance
        scores_tensor = torch.tensor(model_scores)
        self.model_weights = F.softmax(scores_tensor / 0.1, dim=0)  # Temperature scaling
        
        logger.info(f"Updated model weights: {self.model_weights.tolist()}")


class BayesianEnsemble:
    """Bayesian ensemble with uncertainty quantification"""
    
    def __init__(self, model_paths: List[str], device: str = "cuda", num_samples: int = 100):
        self.model_paths = model_paths
        self.device = device
        self.num_samples = num_samples
        self.models = []
        self.load_models()
    
    def load_models(self):
        """Load models and enable dropout for uncertainty estimation"""
        for model_path in self.model_paths:
            try:
                from train_enhanced_model import EnhancedMultiAbnormalityModel
                
                model = EnhancedMultiAbnormalityModel.load_from_checkpoint(model_path)
                
                # Enable dropout for uncertainty estimation
                def enable_dropout(m):
                    if type(m) == nn.Dropout:
                        m.train()
                
                model.apply(enable_dropout)
                model.to(self.device)
                self.models.append(model)
                
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions with uncertainty estimates"""
        all_predictions = []
        
        for model in self.models:
            model_predictions = []
            
            with torch.no_grad():
                for _ in range(self.num_samples):
                    pred = torch.sigmoid(model(x))
                    model_predictions.append(pred)
            
            model_predictions = torch.stack(model_predictions, dim=0)
            all_predictions.append(model_predictions)
        
        # Combine all model samples
        all_samples = torch.cat(all_predictions, dim=0)  # [total_samples, batch_size, num_classes]
        
        # Calculate mean and uncertainty
        mean_pred = torch.mean(all_samples, dim=0)
        uncertainty = torch.std(all_samples, dim=0)
        
        return mean_pred, uncertainty


class EnsemblePredictor:
    """High-level interface for ensemble prediction"""
    
    def __init__(self, 
                 model_paths: List[str], 
                 ensemble_type: str = "simple",
                 device: str = "cuda",
                 use_tta: bool = True):
        
        self.device = device
        self.use_tta = use_tta
        self.ensemble_type = ensemble_type
        
        # Initialize TTA
        if use_tta:
            self.tta = TestTimeAugmentation(num_augmentations=8)
        
        # Initialize ensemble
        if ensemble_type == "simple":
            self.ensemble = ModelEnsemble(model_paths, device)
        elif ensemble_type == "adaptive":
            self.ensemble = AdaptiveEnsemble(model_paths, device)
        elif ensemble_type == "bayesian":
            self.ensemble = BayesianEnsemble(model_paths, device)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate ensemble predictions"""
        x = x.to(self.device)
        
        if self.ensemble_type == "simple":
            if self.use_tta:
                predictions = self.ensemble.predict_with_tta(x, self.tta)
                return {"predictions": predictions}
            else:
                predictions = self.ensemble.predict(x)
                return {"predictions": predictions}
        
        elif self.ensemble_type == "adaptive":
            predictions, confidence = self.ensemble.predict_with_confidence(x)
            return {"predictions": predictions, "confidence": confidence}
        
        elif self.ensemble_type == "bayesian":
            predictions, uncertainty = self.ensemble.predict_with_uncertainty(x)
            return {"predictions": predictions, "uncertainty": uncertainty}
    
    def save_ensemble_results(self, results: Dict, output_path: str):
        """Save ensemble prediction results"""
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Ensemble results saved to {output_path}")
    
    def load_ensemble_results(self, input_path: str) -> Dict:
        """Load ensemble prediction results"""
        with open(input_path, 'rb') as f:
            results = pickle.load(f)
        
        logger.info(f"Ensemble results loaded from {input_path}")
        return results


def create_ensemble_from_checkpoints(checkpoint_dir: str, 
                                   ensemble_type: str = "simple",
                                   max_models: int = 5) -> EnsemblePredictor:
    """Create ensemble from best checkpoints in directory"""
    checkpoint_path = Path(checkpoint_dir)
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_path.glob("*.ckpt"))
    
    if not checkpoint_files:
        raise RuntimeError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Select best checkpoints
    selected_checkpoints = [str(f) for f in checkpoint_files[:max_models]]
    
    logger.info(f"Creating {ensemble_type} ensemble from {len(selected_checkpoints)} models:")
    for i, ckpt in enumerate(selected_checkpoints):
        logger.info(f"  {i+1}. {Path(ckpt).name}")
    
    return EnsemblePredictor(
        model_paths=selected_checkpoints,
        ensemble_type=ensemble_type,
        use_tta=True
    )
