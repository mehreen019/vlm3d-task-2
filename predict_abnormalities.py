#!/usr/bin/env python3
"""
Multi-Abnormality Prediction/Inference Script
Load trained model and predict abnormalities on new CT data
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Union
from PIL import Image
import nibabel as nib

from train_multi_abnormality_model import MultiAbnormalityModel, get_transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionDataset(Dataset):
    """Dataset for prediction on new CT data"""
    
    def __init__(self, data_paths: List[str], transform=None, data_type: str = "slices"):
        """
        Args:
            data_paths: List of paths to slice files (.npy) or volume files (.nii.gz)
            transform: Image transforms
            data_type: "slices" for .npy files, "volumes" for .nii.gz files
        """
        self.data_paths = data_paths
        self.transform = transform
        self.data_type = data_type
        
        # Filter valid files
        valid_paths = []
        for path in data_paths:
            if Path(path).exists():
                valid_paths.append(path)
            else:
                logger.warning(f"File not found: {path}")
        
        self.data_paths = valid_paths
        logger.info(f"Loaded {len(self.data_paths)} files for prediction")
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        
        if self.data_type == "slices":
            # Load .npy slice file
            slice_data = np.load(file_path)
            
            # Ensure 2D
            if len(slice_data.shape) > 2:
                slice_data = slice_data.squeeze()
            
            # Convert to 3-channel for pretrained models
            if len(slice_data.shape) == 2:
                slice_data = np.stack([slice_data] * 3, axis=-1)
            
            # Normalize to [0, 1] if needed
            if slice_data.max() > 1.0:
                slice_data = slice_data / 255.0
            
            images = [slice_data]  # Single slice
            
        elif self.data_type == "volumes":
            # Load .nii.gz volume file and extract multiple slices
            volume = nib.load(file_path).get_fdata()
            
            # Reorient for axial slices
            if len(volume.shape) == 3:
                volume = np.transpose(volume, (2, 1, 0))
            
            # Extract representative slices (similar to slice extractor)
            images = self._extract_representative_slices(volume)
        
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")
        
        # Apply transforms to all images
        processed_images = []
        for img in images:
            if self.transform:
                # Convert to PIL for torchvision transforms
                img_uint8 = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_uint8)
                img_tensor = self.transform(img_pil)
            else:
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
            
            processed_images.append(img_tensor)
        
        return {
            'images': torch.stack(processed_images) if len(processed_images) > 1 else processed_images[0],
            'file_path': file_path,
            'num_slices': len(processed_images)
        }
    
    def _extract_representative_slices(self, volume: np.ndarray, num_slices: int = 6) -> List[np.ndarray]:
        """Extract representative slices from a volume"""
        depth = volume.shape[0]
        
        # Simple strategy: extract evenly spaced slices from middle 70% of volume
        start_idx = int(depth * 0.15)
        end_idx = int(depth * 0.85)
        
        if end_idx - start_idx < num_slices:
            # If volume is too small, take all slices
            slice_indices = list(range(start_idx, end_idx))
        else:
            # Extract evenly spaced slices
            slice_indices = np.linspace(start_idx, end_idx - 1, num_slices, dtype=int)
        
        slices = []
        for idx in slice_indices:
            slice_2d = volume[idx]
            
            # Apply basic preprocessing (similar to slice extractor)
            slice_2d = np.clip(slice_2d, -1000, 400)  # HU windowing
            slice_2d = (slice_2d + 1000) / 1400  # Normalize to [0, 1]
            
            # Convert to 3-channel
            slice_3d = np.stack([slice_2d] * 3, axis=-1)
            slices.append(slice_3d)
        
        return slices

class AbnormalityPredictor:
    """Main predictor class for abnormality detection"""
    
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on ("auto", "cpu", "cuda")
        """
        self.checkpoint_path = checkpoint_path
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Class names
        self.class_names =  [
            "Cardiomegaly", "Hiatal hernia", "Atelectasis", "Pulmonary fibrotic sequela",
            "Peribronchial thickening", "Interlobular septal thickening", "Medical material",
            "Pericardial effusion", "Lymphadenopathy", "Lung nodule", "Pleural effusion",
            "Consolidation", "Lung opacity", "Mosaic attenuation pattern", "Bronchiectasis",
            "Emphysema", "Arterial wall calcification", "Coronary artery wall calcification"
        ]
        
        logger.info(f"Loaded model on device: {self.device}")
    
    def _load_model(self) -> MultiAbnormalityModel:
        """Load model from checkpoint"""
        try:
            model = MultiAbnormalityModel.load_from_checkpoint(self.checkpoint_path)
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_slices(self, slice_paths: List[str], batch_size: int = 32) -> Dict:
        """
        Predict abnormalities for a list of slice files (.npy)
        
        Args:
            slice_paths: List of paths to .npy slice files
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with predictions and metadata
        """
        # Create dataset and dataloader
        transform = get_transforms(augment=False)
        dataset = PredictionDataset(slice_paths, transform=transform, data_type="slices")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Generate predictions
        all_probs = []
        all_file_paths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                images = batch['images'].to(self.device)
                file_paths = batch['file_path']
                
                logits = self.model(images)
                probs = torch.sigmoid(logits)
                
                all_probs.append(probs.cpu().numpy())
                all_file_paths.extend(file_paths)
        
        # Combine results
        all_probs = np.vstack(all_probs)
        
        # Create results
        results = {
            'predictions': [],
            'summary': self._create_summary(all_probs),
            'metadata': {
                'num_slices': len(slice_paths),
                'class_names': self.class_names,
                'model_checkpoint': self.checkpoint_path
            }
        }
        
        # Add per-slice predictions
        for i, (prob_vector, file_path) in enumerate(zip(all_probs, all_file_paths)):
            slice_result = {
                'file_path': file_path,
                'slice_id': Path(file_path).stem,
                'probabilities': prob_vector.tolist(),
                'predictions': (prob_vector > 0.5).astype(int).tolist(),
                'top_abnormalities': self._get_top_predictions(prob_vector)
            }
            results['predictions'].append(slice_result)
        
        return results
    
    def predict_volumes(self, volume_paths: List[str], batch_size: int = 16) -> Dict:
        """
        Predict abnormalities for a list of volume files (.nii.gz)
        
        Args:
            volume_paths: List of paths to .nii.gz volume files
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with predictions and metadata
        """
        transform = get_transforms(augment=False)
        dataset = PredictionDataset(volume_paths, transform=transform, data_type="volumes")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)  # Process one volume at a time
        
        results = {
            'predictions': [],
            'metadata': {
                'num_volumes': len(volume_paths),
                'class_names': self.class_names,
                'model_checkpoint': self.checkpoint_path
            }
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting volumes"):
                images = batch['images'][0]  # Remove batch dimension
                file_path = batch['file_path'][0]
                num_slices = batch['num_slices'][0].item()
                
                # Process slices in batches
                slice_probs = []
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i+batch_size].to(self.device)
                    logits = self.model(batch_images)
                    probs = torch.sigmoid(logits)
                    slice_probs.append(probs.cpu().numpy())
                
                # Combine slice predictions
                volume_probs = np.vstack(slice_probs)
                
                # Aggregate to volume-level prediction (max probability across slices)
                volume_max_probs = np.max(volume_probs, axis=0)
                volume_mean_probs = np.mean(volume_probs, axis=0)
                
                volume_result = {
                    'file_path': file_path,
                    'volume_id': Path(file_path).stem,
                    'num_slices_processed': num_slices,
                    'max_probabilities': volume_max_probs.tolist(),
                    'mean_probabilities': volume_mean_probs.tolist(),
                    'max_predictions': (volume_max_probs > 0.5).astype(int).tolist(),
                    'mean_predictions': (volume_mean_probs > 0.5).astype(int).tolist(),
                    'top_abnormalities_max': self._get_top_predictions(volume_max_probs),
                    'top_abnormalities_mean': self._get_top_predictions(volume_mean_probs),
                    'slice_predictions': volume_probs.tolist()
                }
                
                results['predictions'].append(volume_result)
        
        # Add summary across all volumes
        if results['predictions']:
            all_max_probs = np.array([pred['max_probabilities'] for pred in results['predictions']])
            results['summary'] = self._create_summary(all_max_probs)
        
        return results
    
    def _get_top_predictions(self, prob_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Get top K predictions with class names and probabilities"""
        top_indices = np.argsort(prob_vector)[::-1][:top_k]
        
        top_predictions = []
        for idx in top_indices:
            if prob_vector[idx] > 0.1:  # Only include if probability > 10%
                top_predictions.append({
                    'class_name': self.class_names[idx],
                    'probability': float(prob_vector[idx]),
                    'predicted': bool(prob_vector[idx] > 0.5)
                })
        
        return top_predictions
    
    def _create_summary(self, all_probs: np.ndarray) -> Dict:
        """Create summary statistics across all predictions"""
        mean_probs = np.mean(all_probs, axis=0)
        max_probs = np.max(all_probs, axis=0)
        
        summary = {
            'overall_statistics': {
                'mean_probabilities': mean_probs.tolist(),
                'max_probabilities': max_probs.tolist(),
                'positive_predictions': (mean_probs > 0.5).sum().item(),
                'total_classes': len(self.class_names)
            },
            'top_detected_abnormalities': self._get_top_predictions(mean_probs),
            'class_statistics': {}
        }
        
        # Per-class statistics
        for i, class_name in enumerate(self.class_names):
            class_probs = all_probs[:, i]
            summary['class_statistics'][class_name] = {
                'mean_probability': float(np.mean(class_probs)),
                'max_probability': float(np.max(class_probs)),
                'detection_rate': float((class_probs > 0.5).mean()),
                'positive_cases': int((class_probs > 0.5).sum())
            }
        
        return summary
    
    def save_predictions(self, results: Dict, output_path: str):
        """Save predictions to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Predictions saved to: {output_path}")
    
    def print_summary(self, results: Dict):
        """Print a human-readable summary of predictions"""
        logger.info("=" * 60)
        logger.info("ABNORMALITY PREDICTION SUMMARY")
        logger.info("=" * 60)
        
        if 'summary' in results:
            summary = results['summary']
            overall = summary['overall_statistics']
            
            logger.info(f"Total predictions: {len(results['predictions'])}")
            logger.info(f"Positive abnormalities detected: {overall['positive_predictions']}/{overall['total_classes']}")
            
            logger.info("\nTop detected abnormalities:")
            for pred in summary['top_detected_abnormalities'][:5]:
                logger.info(f"  {pred['class_name']:18s}: {pred['probability']:.3f} ({'✓' if pred['predicted'] else '✗'})")
            
            logger.info("\nPer-class detection rates:")
            for class_name, stats in summary['class_statistics'].items():
                if stats['detection_rate'] > 0:
                    logger.info(f"  {class_name:18s}: {stats['detection_rate']:.1%} ({stats['positive_cases']} cases)")

def main():
    parser = argparse.ArgumentParser(description="Predict abnormalities using trained model")
    
    # Required arguments
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input", required=True, nargs="+", help="Input files (slices or volumes)")
    
    # Optional arguments
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--data-type", choices=["slices", "volumes"], default="slices",
                       help="Type of input data")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device for inference")
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = AbnormalityPredictor(args.checkpoint, device=args.device)
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        return
    
    # Run predictions
    logger.info(f"Running predictions on {len(args.input)} files...")
    
    if args.data_type == "slices":
        results = predictor.predict_slices(args.input, batch_size=args.batch_size)
    elif args.data_type == "volumes":
        results = predictor.predict_volumes(args.input, batch_size=args.batch_size)
    
    # Print summary
    predictor.print_summary(results)
    
    # Save results
    if args.output:
        predictor.save_predictions(results, args.output)
    else:
        # Default output path
        output_path = f"predictions_{args.data_type}.json"
        predictor.save_predictions(results, output_path)

if __name__ == "__main__":
    main() 