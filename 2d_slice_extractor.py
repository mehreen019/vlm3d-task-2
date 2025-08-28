#!/usr/bin/env python3
"""
2D Slice Extraction from 3D CT Volumes for VLM3D Challenge
Converts 3D CT volumes to 2D slices with intelligent sampling strategies
Fixed version for CT-RATE dataset with proper data handling
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib  # Better for .nii.gz files than SimpleITK
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import argparse
import logging
from skimage import transform
import cv2
from scipy import ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTSliceExtractor:
    def __init__(self, 
                 data_dir: str = "./ct_rate_data",
                 output_dir: str = "./ct_rate_2d",
                 slice_strategy: str = "multi_slice",
                 slices_per_volume: int = 12):  # Increased for better coverage
        
        self.data_dir = Path(data_dir).resolve()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.slice_strategy = slice_strategy
        self.slices_per_volume = slices_per_volume
        
        # Optimized chest CT preprocessing parameters
        self.hu_window = (-1000, 400)  # Standard chest window
        self.target_size = (224, 224)  # More VLM-friendly size, reduces storage
        
        # CT-RATE specific abnormalities

        # 18 abnormalities
        self.abnormality_classes = [
            "Cardiomegaly", "Hiatal hernia", "Atelectasis", "Pulmonary fibrotic sequela",
            "Peribronchial thickening", "Interlobular septal thickening", "Medical material",
            "Pericardial effusion", "Lymphadenopathy", "Lung nodule", "Pleural effusion",
            "Consolidation", "Lung opacity", "Mosaic attenuation pattern", "Bronchiectasis",
            "Emphysema", "Arterial wall calcification", "Coronary artery wall calcification"
        ]

        logger.info(f"Initialized slice extractor: {slice_strategy} strategy, "
                   f"{slices_per_volume} slices per volume")
    
    def load_ct_volume(self, ct_path: Path) -> Tuple[np.ndarray, Dict]:
        """Load CT volume using nibabel (better for .nii.gz)"""
        try:
            # Load with nibabel
            img = nib.load(str(ct_path))
            volume = img.get_fdata()
            
            # Ensure proper orientation (typically: sagittal, coronal, axial)
            # For chest CT, we want axial slices (last dimension)
            if len(volume.shape) == 3:
                # Reorient if needed - typically we want axial as first dimension
                # Standard nibabel loading gives (x, y, z), we want (z, y, x) for axial slices
                volume = np.transpose(volume, (2, 1, 0))
            
            metadata = {
                'original_shape': volume.shape,
                'header': img.header,
                'affine': img.affine,
                'voxel_size': img.header.get_zooms() if hasattr(img.header, 'get_zooms') else None
            }
            
            return volume.astype(np.float32), metadata
        
        except Exception as e:
            logger.error(f"Failed to load {ct_path}: {e}")
            return None, None
    
    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Apply comprehensive CT preprocessing"""
        # Apply HU windowing
        volume = np.clip(volume, self.hu_window[0], self.hu_window[1])
        
        # Normalize to [0, 255] for better visualization and model compatibility
        volume = ((volume - self.hu_window[0]) / (self.hu_window[1] - self.hu_window[0]) * 255)
        
        return volume.astype(np.float32)
    
    def find_chest_region(self, volume: np.ndarray) -> Tuple[int, int]:
        """Find the chest region with improved anatomical detection"""
        depth, height, width = volume.shape
        
        # Method 1: Statistical approach
        slice_stats = []
        for i in range(depth):
            slice_2d = volume[i]
            # Calculate multiple metrics
            mean_val = np.mean(slice_2d)
            std_val = np.std(slice_2d)
            # Count non-background pixels (assuming background is very low intensity)
            non_bg_pixels = np.sum(slice_2d > np.percentile(slice_2d, 10))
            
            slice_stats.append({
                'index': i,
                'mean': mean_val,
                'std': std_val,
                'content': non_bg_pixels,
                'score': std_val * non_bg_pixels  # Combined score
            })
        
        slice_stats = sorted(slice_stats, key=lambda x: x['score'], reverse=True)
        
        # Method 2: Find continuous chest region
        # Look for slices with high anatomical content
        high_content_indices = [s['index'] for s in slice_stats[:int(depth * 0.7)]]
        high_content_indices.sort()
        
        if len(high_content_indices) < 10:  # Fallback
            start_slice = max(0, int(depth * 0.15))
            end_slice = min(depth - 1, int(depth * 0.85))
        else:
            # Find the largest continuous region
            gaps = np.diff(high_content_indices)
            large_gaps = np.where(gaps > 5)[0]  # Gaps larger than 5 slices
            
            if len(large_gaps) == 0:
                start_slice = high_content_indices[0]
                end_slice = high_content_indices[-1]
            else:
                # Take the largest continuous segment
                segments = []
                prev_gap = -1
                for gap_idx in np.append(large_gaps, len(high_content_indices) - 1):
                    segment = high_content_indices[prev_gap + 1:gap_idx + 1]
                    if len(segment) > 0:
                        segments.append((segment[0], segment[-1], len(segment)))
                    prev_gap = gap_idx
                
                # Choose longest segment
                best_segment = max(segments, key=lambda x: x[2])
                start_slice, end_slice = best_segment[0], best_segment[1]
        
        # Ensure reasonable bounds
        start_slice = max(0, start_slice)
        end_slice = min(depth - 1, end_slice)
        
        logger.debug(f"Chest region: slices {start_slice}-{end_slice} out of {depth}")
        
        return start_slice, end_slice
    
    def sample_slices_multi_slice(self, volume: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Enhanced multi-slice sampling with better distribution"""
        start_slice, end_slice = self.find_chest_region(volume)
        chest_depth = end_slice - start_slice + 1
        
        if chest_depth <= self.slices_per_volume:
            # Use all available slices in chest region
            slice_indices = list(range(start_slice, end_slice + 1))
        else:
            # Strategic sampling across chest region
            # Sample more densely in middle region where most pathology occurs
            positions = []
            
            # Upper chest (20% of slices)
            upper_count = max(1, int(self.slices_per_volume * 0.2))
            upper_positions = np.linspace(0.0, 0.3, upper_count, endpoint=False)
            positions.extend(upper_positions)
            
            # Middle chest (60% of slices) - most important region
            middle_count = max(2, int(self.slices_per_volume * 0.6))
            middle_positions = np.linspace(0.25, 0.75, middle_count)
            positions.extend(middle_positions)
            
            # Lower chest (20% of slices)
            lower_count = self.slices_per_volume - upper_count - middle_count
            if lower_count > 0:
                lower_positions = np.linspace(0.7, 1.0, lower_count, endpoint=True)
                positions.extend(lower_positions)
            
            # Convert positions to slice indices
            positions = sorted(set(positions))  # Remove duplicates and sort
            slice_indices = [int(start_slice + pos * chest_depth) for pos in positions]
            slice_indices = [max(start_slice, min(end_slice, idx)) for idx in slice_indices]
            slice_indices = sorted(list(set(slice_indices)))  # Remove duplicates
        
        # Extract slices
        sampled_slices = []
        for idx in slice_indices:
            if 0 <= idx < volume.shape[0]:
                slice_2d = volume[idx]
                sampled_slices.append((idx, slice_2d))
        
        return sampled_slices
    
    def sample_slices_best_slice(self, volume: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Sample the single best slice with highest diagnostic content"""
        start_slice, end_slice = self.find_chest_region(volume)
        
        best_score = -1
        best_idx = start_slice + (end_slice - start_slice) // 2
        
        for idx in range(start_slice, end_slice + 1):
            slice_2d = volume[idx]
            # Combined scoring: structure + contrast + content
            std_score = np.std(slice_2d)
            edge_score = np.std(cv2.Laplacian(slice_2d.astype(np.uint8), cv2.CV_64F))
            content_score = np.sum(slice_2d > np.percentile(slice_2d, 25))
            
            combined_score = std_score * 0.4 + edge_score * 0.3 + content_score * 0.3
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx
        
        return [(best_idx, volume[best_idx])]
    
    def sample_slices_anatomical(self, volume: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Sample slices at clinically relevant anatomical levels"""
        start_slice, end_slice = self.find_chest_region(volume)
        chest_depth = end_slice - start_slice
        
        # Anatomical levels based on chest CT interpretation
        anatomical_levels = [
            0.1,   # Upper mediastinum
            0.25,  # Aortic arch level
            0.4,   # Pulmonary artery level
            0.55,  # Heart level
            0.7,   # Lower lobe level
            0.85   # Diaphragm level
        ]
        
        sampled_slices = []
        for level in anatomical_levels:
            slice_idx = int(start_slice + level * chest_depth)
            slice_idx = max(start_slice, min(end_slice, slice_idx))
            sampled_slices.append((slice_idx, volume[slice_idx]))
        
        return sampled_slices
    
    def enhance_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Apply enhancement to improve slice quality"""
        # Adaptive histogram equalization for better contrast
        slice_uint8 = (slice_2d * 255 / slice_2d.max()).astype(np.uint8) if slice_2d.max() > 0 else slice_2d.astype(np.uint8)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(slice_uint8)
        
        # Convert back to float32 and normalize
        enhanced = enhanced.astype(np.float32) / 255.0
        
        return enhanced
    
    def resize_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Resize slice with proper anti-aliasing"""
        # Use high-quality resizing
        resized = transform.resize(
            slice_2d, 
            self.target_size, 
            preserve_range=True, 
            anti_aliasing=True,
            order=1  # Bilinear interpolation
        )
        return resized.astype(np.float32)
    
    def extract_slices_from_volume(self, volume_name: str, volume_data: pd.Series) -> List[Dict]:
        """Extract slices from a single CT volume"""
        # Build the actual file path based on your downloader structure
        split = volume_data['split']  # 'train' or 'valid'
        
        # Your downloader saves files as: ct_rate_volumes/{split}/{volume_name}
        ct_path = self.data_dir / "ct_rate_volumes" / split / volume_name
        ct_path = Path(ct_path).absolute()

        ct_path = Path(str(ct_path))  # force to string first
        print("Checking:", ct_path)
        print("Absolute:", ct_path.absolute())
        print("Exists:", ct_path.exists())
        print("Is file:", ct_path.is_file())
        

        print("Looking for:", ct_path.absolute(), " also known as: ", ct_path)
        
        if not ct_path.absolute().exists():
            logger.warning(f"CT file not found: {ct_path.absolute()}")
            return []
        
        # Load volume
        volume, metadata = self.load_ct_volume(ct_path.absolute())
        if volume is None:
            return []
        
        # Preprocess
        volume = self.preprocess_volume(volume)
        
        # Sample slices based on strategy
        if self.slice_strategy == "multi_slice":
            sampled_slices = self.sample_slices_multi_slice(volume)
        elif self.slice_strategy == "best_slice":
            sampled_slices = self.sample_slices_best_slice(volume)
        elif self.slice_strategy == "anatomical":
            sampled_slices = self.sample_slices_anatomical(volume)
        else:
            raise ValueError(f"Unknown slice strategy: {self.slice_strategy}")
        
        # Extract abnormality labels from the row
        labels = {}
        for abnormality in self.abnormality_classes:
            if abnormality in volume_data.index:
                labels[abnormality] = int(volume_data[abnormality])
            else:
                labels[abnormality] = 0  # Default to 0 if missing
        
        # Process each sampled slice
        slice_data = []
        for slice_idx, slice_2d in sampled_slices:
            # Enhance and resize slice
            enhanced_slice = self.enhance_slice(slice_2d)
            resized_slice = self.resize_slice(enhanced_slice)
            
            # Create slice ID
            slice_id = f"{volume_name}_slice_{slice_idx:03d}"
            
            # Save slice as .npy file
            slice_file = self.output_dir / "slices" / split / f"{slice_id}.npy"
            slice_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(slice_file, resized_slice)
            
            # Create slice metadata
            slice_info = {
                'slice_id': slice_id,
                'volume_name': volume_name,
                'slice_index': slice_idx,
                'relative_position': slice_idx / volume.shape[0] if volume.shape[0] > 0 else 0.0,
                'file_path': str(slice_file.relative_to(self.output_dir)),
                'slice_shape': list(resized_slice.shape),
                'original_volume_shape': list(volume.shape),
                'split': split,
                **labels  # Include all abnormality labels
            }
            
            slice_data.append(slice_info)
        
        return slice_data
    
    def process_split(self, split_name: str) -> pd.DataFrame:
        """Process all volumes in a data split"""
        logger.info(f"Processing {split_name} split...")
        
        # Load split CSV
        split_file = self.data_dir / "splits" / f"{split_name}.csv"
        print(f"Loading split file: {split_file}")
        if not split_file.exists():
            logger.error(f"Split file not found: {split_file}")
            return pd.DataFrame()
        
        split_df = pd.read_csv(split_file)
        logger.info(f"Found {len(split_df)} volumes in {split_name} split")
        
        # Process each volume
        all_slice_data = []
        failed_count = 0
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Extracting {split_name} slices"):
            volume_name = row['VolumeName']
            
            try:
                # Extract slices
                slice_data = self.extract_slices_from_volume(volume_name, row)
                all_slice_data.extend(slice_data)
            except Exception as e:
                logger.error(f"Failed to process {volume_name}: {e}")
                failed_count += 1
                continue
        
        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} volumes in {split_name}")
        
        # Create DataFrame
        slice_df = pd.DataFrame(all_slice_data)
        
        if slice_df.empty:
            logger.error(f"No slices extracted for {split_name}")
            return slice_df
        
        # Save slice metadata
        output_file = self.output_dir / "splits" / f"{split_name}_slices.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        slice_df.to_csv(output_file, index=False)
        
        logger.info(f"Processed {len(slice_df)} slices from {len(split_df)} volumes in {split_name}")
        
        return slice_df
    
    def analyze_slice_distribution(self, slice_df: pd.DataFrame) -> Dict:
        """Analyze the distribution of extracted slices"""
        if slice_df.empty:
            return {}
        
        # Abnormality prevalences
        prevalences = {}
        for abnormality in self.abnormality_classes:
            if abnormality in slice_df.columns:
                prevalences[abnormality] = float(slice_df[abnormality].mean())
        
        # Slice position distribution
        positions = slice_df['relative_position'].describe().to_dict()
        
        # Multi-label statistics
        label_matrix = slice_df[self.abnormality_classes].values
        avg_abnormalities = float(np.mean(np.sum(label_matrix, axis=1)))
        
        stats = {
            'total_slices': int(len(slice_df)),
            'unique_volumes': int(slice_df['volume_name'].nunique()),
            'slices_per_volume': float(len(slice_df) / slice_df['volume_name'].nunique()),
            'abnormality_prevalences': prevalences,
            'slice_position_stats': {k: float(v) for k, v in positions.items()},
            'avg_abnormalities_per_slice': avg_abnormalities
        }
        
        return stats
    
    def create_visualization_samples(self, slice_df: pd.DataFrame, n_samples: int = 16):
        """Create visualization of sample slices"""
        if slice_df.empty:
            logger.warning("No slices to visualize")
            return
        
        logger.info(f"Creating visualization of {n_samples} sample slices...")
        
        # Sample slices with diverse abnormalities
        sample_slices = slice_df.sample(n=min(n_samples, len(slice_df)))
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        for i, (_, row) in enumerate(sample_slices.iterrows()):
            if i >= len(axes):
                break
                
            # Load slice
            slice_path = self.output_dir / row['file_path']
            if slice_path.exists():
                slice_data = np.load(slice_path)
                
                # Find present abnormalities
                abnormalities = [abnormality for abnormality in self.abnormality_classes 
                               if row.get(abnormality, 0) == 1][:3]  # Show max 3
                abnormality_str = ", ".join(abnormalities) if abnormalities else "No_Finding"
                
                # Plot
                axes[i].imshow(slice_data, cmap='gray', vmin=0, vmax=1)
                axes[i].set_title(f"{row['slice_id'][:20]}\nPos: {row['relative_position']:.2f}\n{abnormality_str}", 
                                fontsize=8)
                axes[i].axis('off')
        
        # Remove unused subplots
        for i in range(len(sample_slices), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_slices_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {self.output_dir / 'sample_slices_visualization.png'}")

def main():
    parser = argparse.ArgumentParser(description="Extract 2D slices from 3D CT volumes")
    parser.add_argument("--data-dir", default="./ct_rate_data", help="CT-RATE data directory")
    parser.add_argument("--output-dir", default="./ct_rate_2d", help="Output directory for 2D slices")
    parser.add_argument("--strategy", choices=["multi_slice", "best_slice", "anatomical"], 
                       default="multi_slice", help="Slice sampling strategy")
    parser.add_argument("--slices-per-volume", type=int, default=12, help="Number of slices per volume")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"], help="Data splits to process")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = CTSliceExtractor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        slice_strategy=args.strategy,
        slices_per_volume=args.slices_per_volume
    )
    
    # Process each split
    all_stats = {}
    for split in args.splits:
        slice_df = extractor.process_split(split)
        if not slice_df.empty:
            stats = extractor.analyze_slice_distribution(slice_df)
            all_stats[split] = stats
            
            # Create visualization for training split
            if split == "train":
                extractor.create_visualization_samples(slice_df)
    
    # Save overall statistics
    if all_stats:
        stats_file = extractor.output_dir / "slice_extraction_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
    
        # Print summary
        logger.info("=" * 60)
        logger.info("🎉 2D Slice Extraction Complete!")
        logger.info("=" * 60)
        
        total_slices = sum(stats['total_slices'] for stats in all_stats.values())
        total_volumes = sum(stats['unique_volumes'] for stats in all_stats.values())
        
        logger.info(f"📊 Summary:")
        logger.info(f"   Strategy: {args.strategy}")
        logger.info(f"   Total slices extracted: {total_slices}")
        logger.info(f"   From volumes: {total_volumes}")
        logger.info(f"   Average slices per volume: {total_slices/total_volumes:.1f}")
        
        for split, stats in all_stats.items():
            logger.info(f"   {split}: {stats['total_slices']} slices from {stats['unique_volumes']} volumes")
        
        logger.info(f"💾 Output directory: {args.output_dir}")
        logger.info(f"📈 Statistics saved to: {stats_file}")
    else:
        logger.error("No data processed successfully!")

if __name__ == "__main__":
    main()