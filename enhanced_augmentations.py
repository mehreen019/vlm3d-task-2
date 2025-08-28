#!/usr/bin/env python3
"""
Enhanced Data Augmentation Techniques for CT Medical Images
Implements novel augmentation strategies specifically designed for CT scans
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, Optional
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from scipy import ndimage


class CTSpecificAugmentations:
    """CT-specific augmentation techniques that preserve medical relevance"""
    
    def __init__(self):
        self.anatomical_masks = self._create_anatomical_masks()
    
    def _create_anatomical_masks(self):
        """Create anatomical region masks for targeted augmentation"""
        # Simple anatomical regions for 224x224 images
        masks = {}
        
        # Lung region mask (approximate)
        lung_mask = np.zeros((224, 224), dtype=np.float32)
        lung_mask[40:180, 20:100] = 1.0  # Left lung
        lung_mask[40:180, 124:204] = 1.0  # Right lung
        masks['lungs'] = lung_mask
        
        # Heart region mask (approximate)
        heart_mask = np.zeros((224, 224), dtype=np.float32)
        heart_mask[80:160, 90:140] = 1.0
        masks['heart'] = heart_mask
        
        # Spine region mask (approximate)
        spine_mask = np.zeros((224, 224), dtype=np.float32)
        spine_mask[60:180, 105:119] = 1.0
        masks['spine'] = spine_mask
        
        return masks
    
    def random_lung_texture_augmentation(self, image: np.ndarray, prob: float = 0.3) -> np.ndarray:
        """Add realistic lung texture variations"""
        if random.random() > prob:
            return image
            
        # Create subtle texture patterns
        h, w = image.shape[:2]
        texture = np.random.normal(0, 5, (h//4, w//4))
        texture = cv2.resize(texture, (w, h))
        
        # Apply only to lung regions
        lung_mask = cv2.resize(self.anatomical_masks['lungs'], (w, h))
        
        if len(image.shape) == 3:
            texture = np.stack([texture] * image.shape[2], axis=-1)
            lung_mask = np.stack([lung_mask] * image.shape[2], axis=-1)
        
        augmented = image + texture * lung_mask * 0.1
        return np.clip(augmented, 0, 255).astype(image.dtype)
    
    def simulate_breathing_artifact(self, image: np.ndarray, prob: float = 0.2) -> np.ndarray:
        """Simulate breathing motion artifacts"""
        if random.random() > prob:
            return image
            
        # Create subtle motion blur in lung regions
        kernel_size = random.choice([3, 5])
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1/kernel_size  # Horizontal motion
        
        # Apply motion blur
        if len(image.shape) == 3:
            blurred = cv2.filter2D(image, -1, kernel)
        else:
            blurred = cv2.filter2D(image, -1, kernel)
        
        # Blend with original using lung mask
        lung_mask = cv2.resize(self.anatomical_masks['lungs'], image.shape[:2][::-1])
        if len(image.shape) == 3:
            lung_mask = np.stack([lung_mask] * image.shape[2], axis=-1)
        
        result = image * (1 - lung_mask * 0.3) + blurred * (lung_mask * 0.3)
        return result.astype(image.dtype)
    
    def add_realistic_noise(self, image: np.ndarray, prob: float = 0.4) -> np.ndarray:
        """Add CT scanner-specific noise patterns"""
        if random.random() > prob:
            return image
            
        # Quantum noise (Poisson-like)
        quantum_noise = np.random.poisson(image/10) * 10 - image
        quantum_noise = np.clip(quantum_noise, -20, 20)
        
        # Electronic noise (Gaussian)
        electronic_noise = np.random.normal(0, 2, image.shape)
        
        # Combine noises
        total_noise = quantum_noise * 0.3 + electronic_noise * 0.7
        
        result = image + total_noise
        return np.clip(result, 0, 255).astype(image.dtype)


class MixupCutmix:
    """Implementation of Mixup and CutMix for medical images"""
    
    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0, prob: float = 0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation"""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation"""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        _, _, H, W = x.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling for cut position
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup or cutmix randomly"""
        if random.random() > self.prob:
            return x, y, y, 1.0
        
        if random.random() < 0.5:
            return self.mixup_data(x, y)
        else:
            return self.cutmix_data(x, y)


class AdvancedCTAugmentations:
    """Combined advanced augmentation pipeline for CT scans"""
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 zoom_range: Tuple[float, float] = (0.9, 1.1),
                 brightness_range: float = 0.1,
                 contrast_range: Tuple[float, float] = (0.9, 1.1),
                 elastic_deformation: bool = True,
                 ct_specific_aug: bool = True):
        
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.elastic_deformation = elastic_deformation
        
        self.ct_augmentations = CTSpecificAugmentations() if ct_specific_aug else None
    
    def elastic_transform(self, image: np.ndarray, alpha: float = 20, sigma: float = 3) -> np.ndarray:
        """Apply elastic deformation"""
        random_state = np.random.RandomState(None)
        shape = image.shape[:2]
        
        dx = ndimage.gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        ) * alpha
        dy = ndimage.gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        ) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = ndimage.map_coordinates(
                    image[:, :, i], indices, order=1, mode='reflect'
                ).reshape(shape)
        else:
            result = ndimage.map_coordinates(
                image, indices, order=1, mode='reflect'
            ).reshape(shape)
        
        return result
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply the full augmentation pipeline"""
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure float format for processing
        original_dtype = image.dtype
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
        
        # Apply CT-specific augmentations
        if self.ct_augmentations:
            image = self.ct_augmentations.random_lung_texture_augmentation(image)
            image = self.ct_augmentations.simulate_breathing_artifact(image)
            image = self.ct_augmentations.add_realistic_noise(image)
        
        # Apply elastic deformation
        if self.elastic_deformation and random.random() < 0.3:
            image = self.elastic_transform(image)
        
        # Standard geometric augmentations
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            if len(image.shape) == 3:
                # For RGB images
                image_pil = Image.fromarray(image.astype(np.uint8))
                image_pil = TF.rotate(image_pil, angle)
                image = np.array(image_pil).astype(np.float32)
            else:
                # For grayscale
                center = (image.shape[1]//2, image.shape[0]//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Zoom augmentation
        if random.random() < 0.3:
            zoom_factor = random.uniform(*self.zoom_range)
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            
            if len(image.shape) == 3:
                resized = cv2.resize(image, (new_w, new_h))
            else:
                resized = cv2.resize(image, (new_w, new_h))
            
            # Crop or pad to original size
            if zoom_factor > 1:  # Crop
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = resized[start_h:start_h+h, start_w:start_w+w]
            else:  # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                if len(image.shape) == 3:
                    image = np.pad(resized, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0, 0)), mode='reflect')
                else:
                    image = np.pad(resized, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='reflect')
        
        # Brightness adjustment
        if random.random() < 0.4:
            brightness_factor = random.uniform(1-self.brightness_range, 1+self.brightness_range)
            image = image * brightness_factor
        
        # Contrast adjustment
        if random.random() < 0.4:
            contrast_factor = random.uniform(*self.contrast_range)
            mean_val = np.mean(image)
            image = (image - mean_val) * contrast_factor + mean_val
        
        # Clip values and restore original dtype
        image = np.clip(image, 0, 255)
        if original_dtype == np.uint8:
            image = image.astype(np.uint8)
        
        return image


class MixupCutmixLoss(nn.Module):
    """Loss function for mixup/cutmix training"""
    
    def __init__(self, base_loss_fn: nn.Module):
        super().__init__()
        self.base_loss_fn = base_loss_fn
    
    def forward(self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Compute mixed loss"""
        return lam * self.base_loss_fn(pred, y_a) + (1 - lam) * self.base_loss_fn(pred, y_b)
