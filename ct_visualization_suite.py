#!/usr/bin/env python3
"""
CT-RATE Visualization Suite
Comprehensive visualization tools for 3D CT volumes and 2D slices
Designed to work with ct_rate_downloader.py and 2d_slice_extractor.py outputs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from pathlib import Path
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import ndimage
from skimage import measure, filters
import cv2

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTVisualizationSuite:
    def __init__(self, 
                 data_dir: str = "./ct_rate_data",
                 slice_dir: str = "./ct_rate_2d",
                 output_dir: str = "./visualizations"):
        
        self.data_dir = Path(data_dir)
        self.slice_dir = Path(slice_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CT-RATE abnormalities
        self.abnormality_classes = [
            "Cardiomegaly", "Hiatal hernia", "Atelectasis", "Pulmonary fibrotic sequela",
            "Peribronchial thickening", "Interlobular septal thickening", "Medical material",
            "Pericardial effusion", "Lymphadenopathy", "Lung nodule", "Pleural effusion",
            "Consolidation", "Lung opacity", "Mosaic attenuation pattern", "Bronchiectasis",
            "Emphysema", "Arterial wall calcification", "Coronary artery wall calcification"
        ]
        
        # Color palette for abnormalities
        self.color_palette = sns.color_palette("husl", len(self.abnormality_classes))
        self.abnormality_colors = dict(zip(self.abnormality_classes, self.color_palette))
        
        logger.info(f"Initialized CT Visualization Suite")
        logger.info(f"Data dir: {self.data_dir}")
        logger.info(f"Slice dir: {self.slice_dir}")
        logger.info(f"Output dir: {self.output_dir}")

    def load_metadata(self) -> pd.DataFrame:
        """Load merged metadata from downloader"""
        metadata_file = self.data_dir / "metadata.csv"
        if metadata_file.exists():
            return pd.read_csv(metadata_file)
        else:
            logger.error(f"Metadata file not found: {metadata_file}")
            return pd.DataFrame()

    def load_labels(self) -> pd.DataFrame:
        """Load multi-abnormality labels"""
        labels_file = self.data_dir / "multi_abnormality_labels.csv"
        if labels_file.exists():
            return pd.read_csv(labels_file)
        else:
            logger.error(f"Labels file not found: {labels_file}")
            return pd.DataFrame()

    def load_slice_data(self, split: str = "train") -> pd.DataFrame:
        """Load 2D slice metadata"""
        slice_file = self.slice_dir / "splits" / f"{split}_slices.csv"
        if slice_file.exists():
            return pd.read_csv(slice_file)
        else:
            logger.error(f"Slice file not found: {slice_file}")
            return pd.DataFrame()

    def visualize_dataset_overview(self, metadata_df: pd.DataFrame, labels_df: pd.DataFrame):
        """Create comprehensive dataset overview visualizations"""
        logger.info("Creating dataset overview visualizations...")
        
        # Merge metadata and labels
        df = pd.merge(metadata_df, labels_df, on=["VolumeName", "split"], how="inner")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("CT-RATE Dataset Overview", fontsize=16, fontweight='bold')
        
        # 1. Split distribution
        split_counts = df['split'].value_counts()
        axes[0, 0].pie(split_counts.values, labels=split_counts.index, autopct='%1.1f%%', 
                       colors=['skyblue', 'lightcoral'])
        axes[0, 0].set_title(f'Dataset Split Distribution\n(Total: {len(df)} volumes)')
        
        # 2. Abnormality prevalence
        prevalences = []
        for abnormality in self.abnormality_classes:
            if abnormality in df.columns:
                prevalences.append(df[abnormality].mean())
            else:
                prevalences.append(0)
        
        sorted_indices = np.argsort(prevalences)[::-1]
        sorted_abnormalities = [self.abnormality_classes[i] for i in sorted_indices]
        sorted_prevalences = [prevalences[i] for i in sorted_indices]
        
        y_pos = np.arange(len(sorted_abnormalities))
        bars = axes[0, 1].barh(y_pos, sorted_prevalences, 
                               color=[self.abnormality_colors[abn] for abn in sorted_abnormalities])
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels([abn.replace(' ', '\n') for abn in sorted_abnormalities], fontsize=8)
        axes[0, 1].set_xlabel('Prevalence')
        axes[0, 1].set_title('Abnormality Prevalence')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Number of abnormalities per volume distribution
        label_cols = [col for col in self.abnormality_classes if col in df.columns]
        n_abnormalities = df[label_cols].sum(axis=1)
        axes[0, 2].hist(n_abnormalities, bins=range(0, max(n_abnormalities)+2), 
                        alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_xlabel('Number of Abnormalities')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title(f'Distribution of Abnormalities per Volume\n(Mean: {n_abnormalities.mean():.2f})')
        axes[0, 2].grid(alpha=0.3)
        
        # 4. Volume metadata distribution (if available)
        if 'SliceThickness' in df.columns:
            axes[1, 0].hist(df['SliceThickness'].dropna(), bins=30, alpha=0.7, color='orange')
            axes[1, 0].set_xlabel('Slice Thickness (mm)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Slice Thickness Distribution')
            axes[1, 0].grid(alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Slice Thickness\ndata not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 5. Co-occurrence heatmap (top abnormalities)
        top_abnormalities = sorted_abnormalities[:10]  # Top 10 most prevalent
        cooccurrence_matrix = np.zeros((len(top_abnormalities), len(top_abnormalities)))
        
        for i, abn1 in enumerate(top_abnormalities):
            for j, abn2 in enumerate(top_abnormalities):
                if abn1 in df.columns and abn2 in df.columns:
                    cooccurrence_matrix[i, j] = ((df[abn1] == 1) & (df[abn2] == 1)).sum()
        
        im = axes[1, 1].imshow(cooccurrence_matrix, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_xticks(range(len(top_abnormalities)))
        axes[1, 1].set_yticks(range(len(top_abnormalities)))
        axes[1, 1].set_xticklabels([abn[:15] for abn in top_abnormalities], rotation=45, ha='right', fontsize=8)
        axes[1, 1].set_yticklabels([abn[:15] for abn in top_abnormalities], fontsize=8)
        axes[1, 1].set_title('Abnormality Co-occurrence Matrix')
        plt.colorbar(im, ax=axes[1, 1])
        
        # 6. Split comparison
        split_comparison = []
        for split in df['split'].unique():
            split_data = df[df['split'] == split]
            for abnormality in top_abnormalities[:8]:  # Top 8 for visibility
                if abnormality in split_data.columns:
                    split_comparison.append({
                        'Split': split,
                        'Abnormality': abnormality,
                        'Prevalence': split_data[abnormality].mean()
                    })
        
        if split_comparison:
            comparison_df = pd.DataFrame(split_comparison)
            pivot_df = comparison_df.pivot(index='Abnormality', columns='Split', values='Prevalence')
            pivot_df.plot(kind='bar', ax=axes[1, 2], width=0.8)
            axes[1, 2].set_title('Abnormality Prevalence by Split')
            axes[1, 2].set_xlabel('Abnormality')
            axes[1, 2].set_ylabel('Prevalence')
            axes[1, 2].legend()
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "dataset_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dataset overview saved to {self.output_dir / 'dataset_overview.png'}")

    def visualize_3d_volume_analysis(self, metadata_df: pd.DataFrame, labels_df: pd.DataFrame, n_samples: int = 6):
        """Create 3D volume analysis visualizations"""
        logger.info("Creating 3D volume analysis...")
        
        df = pd.merge(metadata_df, labels_df, on=["VolumeName", "split"], how="inner")
        
        # Sample volumes with different abnormality patterns
        sample_volumes = self._select_representative_volumes(df, n_samples)
        
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle("3D CT Volume Analysis - Multi-Planar Views", fontsize=16, fontweight='bold')
        
        for i, (_, volume_data) in enumerate(sample_volumes.iterrows()):
            volume_name = volume_data['VolumeName']
            split = volume_data['split']
            
            # Load volume
            volume_path = self.data_dir / "ct_rate_volumes" / split / volume_name
            if volume_path.exists():
                volume, _ = self._load_volume(volume_path)
                if volume is not None:
                    self._plot_multiplanar_view(volume, axes[i], volume_name, volume_data)
                else:
                    self._plot_placeholder(axes[i], f"Failed to load: {volume_name}")
            else:
                self._plot_placeholder(axes[i], f"File not found: {volume_name}")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "3d_volume_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"3D volume analysis saved to {self.output_dir / '3d_volume_analysis.png'}")

    def visualize_2d_slice_analysis(self, slice_df: pd.DataFrame):
        """Create 2D slice analysis visualizations"""
        logger.info("Creating 2D slice analysis...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle("2D Slice Analysis", fontsize=16, fontweight='bold')
        
        # 1. Slice position distribution
        axes[0, 0].hist(slice_df['relative_position'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Relative Position in Volume')
        axes[0, 0].set_ylabel('Number of Slices')
        axes[0, 0].set_title('Distribution of Slice Positions')
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Slices per volume distribution
        slices_per_volume = slice_df.groupby('volume_name').size()
        axes[0, 1].hist(slices_per_volume, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('Number of Slices per Volume')
        axes[0, 1].set_ylabel('Number of Volumes')
        axes[0, 1].set_title(f'Slices per Volume Distribution\n(Mean: {slices_per_volume.mean():.1f})')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Abnormality prevalence in slices
        slice_prevalences = []
        for abnormality in self.abnormality_classes:
            if abnormality in slice_df.columns:
                slice_prevalences.append(slice_df[abnormality].mean())
            else:
                slice_prevalences.append(0)
        
        sorted_indices = np.argsort(slice_prevalences)[::-1][:12]  # Top 12
        top_abnormalities = [self.abnormality_classes[i] for i in sorted_indices]
        top_prevalences = [slice_prevalences[i] for i in sorted_indices]
        
        bars = axes[0, 2].bar(range(len(top_abnormalities)), top_prevalences,
                             color=[self.abnormality_colors[abn] for abn in top_abnormalities])
        axes[0, 2].set_xticks(range(len(top_abnormalities)))
        axes[0, 2].set_xticklabels([abn[:12] for abn in top_abnormalities], rotation=45, ha='right')
        axes[0, 2].set_ylabel('Prevalence in Slices')
        axes[0, 2].set_title('Top Abnormalities in 2D Slices')
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # 4-6. Sample slices with different abnormality counts
        slice_abnormality_counts = slice_df[self.abnormality_classes].sum(axis=1)
        
        # No abnormalities
        no_abn_slices = slice_df[slice_abnormality_counts == 0].sample(n=min(3, len(slice_df[slice_abnormality_counts == 0])))
        self._plot_sample_slices(no_abn_slices, axes[1, :], "No Abnormalities")
        
        # Multiple abnormalities
        multi_abn_slices = slice_df[slice_abnormality_counts >= 3].sample(n=min(3, len(slice_df[slice_abnormality_counts >= 3])))
        if len(multi_abn_slices) > 0:
            self._plot_sample_slices(multi_abn_slices, axes[2, :], "Multiple Abnormalities")
        else:
            for ax in axes[2, :]:
                ax.text(0.5, 0.5, 'No samples with\n≥3 abnormalities', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "2d_slice_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"2D slice analysis saved to {self.output_dir / '2d_slice_analysis.png'}")

    def create_interactive_abnormality_explorer(self, labels_df: pd.DataFrame):
        """Create interactive abnormality co-occurrence visualization using Plotly"""
        logger.info("Creating interactive abnormality explorer...")
        
        # Calculate co-occurrence matrix
        available_abnormalities = [abn for abn in self.abnormality_classes if abn in labels_df.columns]
        cooccurrence_data = []
        
        for abn1 in available_abnormalities:
            for abn2 in available_abnormalities:
                cooccurrence = ((labels_df[abn1] == 1) & (labels_df[abn2] == 1)).sum()
                cooccurrence_data.append({
                    'Abnormality_1': abn1,
                    'Abnormality_2': abn2,
                    'Co_occurrence': cooccurrence,
                    'Normalized': cooccurrence / len(labels_df) if len(labels_df) > 0 else 0
                })
        
        cooccurrence_df = pd.DataFrame(cooccurrence_data)
        
        # Create heatmap
        fig = px.imshow(
            cooccurrence_df.pivot(index='Abnormality_1', columns='Abnormality_2', values='Normalized'),
            title="Interactive Abnormality Co-occurrence Heatmap",
            labels=dict(color="Normalized Co-occurrence"),
            aspect="auto"
        )
        
        fig.update_layout(
            title_x=0.5,
            xaxis_title="Abnormality 2",
            yaxis_title="Abnormality 1",
            height=800,
            width=800
        )
        
        # Save interactive plot
        fig.write_html(self.output_dir / "interactive_abnormality_explorer.html")
        
        logger.info(f"Interactive explorer saved to {self.output_dir / 'interactive_abnormality_explorer.html'}")

    def create_statistical_summary_report(self, metadata_df: pd.DataFrame, labels_df: pd.DataFrame, slice_df: pd.DataFrame = None):
        """Create comprehensive statistical summary"""
        logger.info("Creating statistical summary report...")
        
        df = pd.merge(metadata_df, labels_df, on=["VolumeName", "split"], how="inner")
        
        # Prepare summary statistics
        stats = {
            'dataset_overview': {
                'total_volumes': len(df),
                'train_volumes': len(df[df['split'] == 'train']),
                'valid_volumes': len(df[df['split'] == 'valid']),
            },
            'abnormality_statistics': {},
            'volume_characteristics': {}
        }
        
        # Abnormality statistics
        label_cols = [col for col in self.abnormality_classes if col in df.columns]
        for abnormality in label_cols:
            stats['abnormality_statistics'][abnormality] = {
                'total_cases': int(df[abnormality].sum()),
                'prevalence': float(df[abnormality].mean()),
                'train_cases': int(df[df['split'] == 'train'][abnormality].sum()),
                'valid_cases': int(df[df['split'] == 'valid'][abnormality].sum())
            }
        
        # Multi-label statistics
        n_abnormalities = df[label_cols].sum(axis=1)
        stats['multi_label_stats'] = {
            'avg_abnormalities_per_volume': float(n_abnormalities.mean()),
            'max_abnormalities_per_volume': int(n_abnormalities.max()),
            'volumes_with_no_abnormalities': int((n_abnormalities == 0).sum()),
            'volumes_with_multiple_abnormalities': int((n_abnormalities > 1).sum())
        }
        
        # Add slice statistics if available
        if slice_df is not None and not slice_df.empty:
            slice_n_abnormalities = slice_df[label_cols].sum(axis=1)
            stats['slice_statistics'] = {
                'total_slices': len(slice_df),
                'avg_slices_per_volume': float(len(slice_df) / slice_df['volume_name'].nunique()),
                'avg_abnormalities_per_slice': float(slice_n_abnormalities.mean()),
                'slices_with_no_abnormalities': int((slice_n_abnormalities == 0).sum()),
                'slices_with_multiple_abnormalities': int((slice_n_abnormalities > 1).sum())
            }
        
        # Save statistics
        with open(self.output_dir / "statistical_summary.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create summary visualization
        self._create_summary_plots(stats, df, slice_df)
        
        logger.info(f"Statistical summary saved to {self.output_dir / 'statistical_summary.json'}")

    def _create_summary_plots(self, stats: dict, df: pd.DataFrame, slice_df: pd.DataFrame = None):
        """Create summary plots from statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Statistical Summary Report", fontsize=16, fontweight='bold')
        
        # Volume distribution
        volume_counts = [stats['dataset_overview']['train_volumes'], stats['dataset_overview']['valid_volumes']]
        axes[0, 0].pie(volume_counts, labels=['Train', 'Valid'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title(f"Volume Distribution\n(Total: {stats['dataset_overview']['total_volumes']})")
        
        # Top abnormalities by prevalence
        abn_stats = stats['abnormality_statistics']
        if abn_stats:
            top_abnormalities = sorted(abn_stats.items(), key=lambda x: x[1]['prevalence'], reverse=True)[:10]
            abnormalities, prevalences = zip(*[(abn, data['prevalence']) for abn, data in top_abnormalities])
            
            y_pos = np.arange(len(abnormalities))
            axes[0, 1].barh(y_pos, prevalences, color='skyblue')
            axes[0, 1].set_yticks(y_pos)
            axes[0, 1].set_yticklabels([abn[:20] for abn in abnormalities], fontsize=9)
            axes[0, 1].set_xlabel('Prevalence')
            axes[0, 1].set_title('Top 10 Abnormalities by Prevalence')
            axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Multi-label distribution
        if 'multi_label_stats' in stats:
            ml_stats = stats['multi_label_stats']
            categories = ['No Abnormalities', 'Single Abnormality', 'Multiple Abnormalities']
            counts = [
                ml_stats['volumes_with_no_abnormalities'],
                stats['dataset_overview']['total_volumes'] - ml_stats['volumes_with_no_abnormalities'] - ml_stats['volumes_with_multiple_abnormalities'],
                ml_stats['volumes_with_multiple_abnormalities']
            ]
            axes[1, 0].bar(categories, counts, color=['lightgreen', 'orange', 'salmon'])
            axes[1, 0].set_ylabel('Number of Volumes')
            axes[1, 0].set_title('Multi-label Distribution')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Slice vs Volume comparison (if slice data available)
        if slice_df is not None and 'slice_statistics' in stats:
            comparison_data = {
                'Average Abnormalities': [
                    stats['multi_label_stats']['avg_abnormalities_per_volume'],
                    stats['slice_statistics']['avg_abnormalities_per_slice']
                ],
                'No Abnormality Rate': [
                    stats['multi_label_stats']['volumes_with_no_abnormalities'] / stats['dataset_overview']['total_volumes'],
                    stats['slice_statistics']['slices_with_no_abnormalities'] / stats['slice_statistics']['total_slices']
                ]
            }
            
            x = np.arange(len(comparison_data))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, [comparison_data['Average Abnormalities'][0], comparison_data['No Abnormality Rate'][0]], 
                          width, label='Volumes', color='skyblue')
            axes[1, 1].bar(x + width/2, [comparison_data['Average Abnormalities'][1], comparison_data['No Abnormality Rate'][1]], 
                          width, label='Slices', color='lightcoral')
            
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title('Volume vs Slice Comparison')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(['Avg Abnormalities', 'No Abnormality Rate'])
            axes[1, 1].legend()
            axes[1, 1].grid(axis='y', alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Slice data not available\nfor comparison', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "summary_plots.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _select_representative_volumes(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Select representative volumes with different abnormality patterns"""
        label_cols = [col for col in self.abnormality_classes if col in df.columns]
        n_abnormalities = df[label_cols].sum(axis=1)
        
        # Try to get diverse samples
        samples = []
        
        # Normal cases (no abnormalities)
        normal_cases = df[n_abnormalities == 0]
        if len(normal_cases) > 0:
            samples.append(normal_cases.sample(1))
        
        # Single abnormality cases
        single_abn_cases = df[n_abnormalities == 1]
        if len(single_abn_cases) > 0:
            samples.append(single_abn_cases.sample(1))
        
        # Multiple abnormality cases
        multi_abn_cases = df[n_abnormalities > 1]
        if len(multi_abn_cases) > 0:
            samples.append(multi_abn_cases.sample(min(n_samples - len(samples), len(multi_abn_cases))))
        
        # Fill remaining slots randomly
        remaining_slots = n_samples - sum(len(s) for s in samples)
        if remaining_slots > 0:
            remaining_df = df.drop(pd.concat(samples).index if samples else [])
            if len(remaining_df) > 0:
                samples.append(remaining_df.sample(min(remaining_slots, len(remaining_df))))
        
        return pd.concat(samples) if samples else df.sample(min(n_samples, len(df)))

    def _load_volume(self, volume_path: Path) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """Load CT volume"""
        try:
            img = nib.load(str(volume_path))
            volume = img.get_fdata()
            if len(volume.shape) == 3:
                volume = np.transpose(volume, (2, 1, 0))  # Reorient for axial slices
            return volume.astype(np.float32), {'shape': volume.shape}
        except Exception as e:
            logger.warning(f"Failed to load volume {volume_path}: {e}")
            return None, None

    def _plot_multiplanar_view(self, volume: np.ndarray, axes, volume_name: str, volume_data: pd.Series):
        """Plot multiplanar view of CT volume"""
        depth, height, width = volume.shape
        
        # Apply windowing
        windowed_volume = np.clip(volume, -1000, 400)
        windowed_volume = (windowed_volume + 1000) / 1400  # Normalize to 0-1
        
        # Axial slice (middle)
        axial_slice = windowed_volume[depth // 2]
        axes[0].imshow(axial_slice, cmap='gray', aspect='auto')
        axes[0].set_title(f'Axial\n{volume_name[:20]}')
        axes[0].axis('off')
        
        # Coronal slice (middle)
        coronal_slice = windowed_volume[:, height // 2, :]
        axes[1].imshow(coronal_slice, cmap='gray', aspect='auto')
        axes[1].set_title('Coronal')
        axes[1].axis('off')
        
        # Sagittal slice (middle)
        sagittal_slice = windowed_volume[:, :, width // 2]
        axes[2].imshow(sagittal_slice, cmap='gray', aspect='auto')
        axes[2].set_title('Sagittal')
        axes[2].axis('off')
        
        # Abnormality summary
        abnormalities = []
        for abnormality in self.abnormality_classes:
            if abnormality in volume_data.index and volume_data[abnormality] == 1:
                abnormalities.append(abnormality)
        
        abn_text = "Abnormalities:\n" + "\n".join(abnormalities[:3]) if abnormalities else "No abnormalities"
        if len(abnormalities) > 3:
            abn_text += f"\n+{len(abnormalities)-3} more"
        
        axes[3].text(0.1, 0.9, abn_text, transform=axes[3].transAxes, fontsize=8, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[3].set_xlim(0, 1)
        axes[3].set_ylim(0, 1)
        axes[3].axis('off')

    def _plot_sample_slices(self, sample_slices: pd.DataFrame, axes, title_prefix: str):
        """Plot sample 2D slices"""
        for i, (_, row) in enumerate(sample_slices.iterrows()):
            if i >= len(axes):
                break
            
            slice_path = self.slice_dir / row['file_path']
            if slice_path.exists():
                try:
                    slice_data = np.load(slice_path)
                    axes[i].imshow(slice_data, cmap='gray', vmin=0, vmax=1)
                    
                    # Find abnormalities
                    abnormalities = [abn for abn in self.abnormality_classes 
                                   if abn in row.index and row[abn] == 1]
                    abn_text = ", ".join(abnormalities[:2]) if abnormalities else "Normal"
                    if len(abnormalities) > 2:
                        abn_text += f" +{len(abnormalities)-2}"
                    
                    axes[i].set_title(f"{title_prefix}\n{abn_text}", fontsize=9)
                    axes[i].axis('off')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Error loading\n{row["slice_id"]}', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_xticks([])
                    axes[i].set_yticks([])
            else:
                axes[i].text(0.5, 0.5, 'File not found', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_xticks([])
                axes[i].set_yticks([])

    def _plot_placeholder(self, axes, message: str):
        """Plot placeholder for missing data"""
        for ax in axes:
            ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

def main():
    parser = argparse.ArgumentParser(description="CT-RATE Visualization Suite")
    parser.add_argument("--data-dir", default="./ct_rate_data", help="CT-RATE data directory")
    parser.add_argument("--slice-dir", default="./ct_rate_2d", help="2D slices directory")
    parser.add_argument("--output-dir", default="./visualizations", help="Output directory for visualizations")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"], help="Data splits to analyze")
    parser.add_argument("--skip-3d", action="store_true", help="Skip 3D volume visualizations")
    parser.add_argument("--skip-2d", action="store_true", help="Skip 2D slice visualizations")
    parser.add_argument("--n-volume-samples", type=int, default=6, help="Number of volume samples for 3D analysis")
    
    args = parser.parse_args()
    
    # Initialize visualization suite
    viz_suite = CTVisualizationSuite(
        data_dir=args.data_dir,
        slice_dir=args.slice_dir,
        output_dir=args.output_dir
    )
    
    # Load data
    logger.info("Loading data...")
    metadata_df = viz_suite.load_metadata()
    labels_df = viz_suite.load_labels()
    
    if metadata_df.empty or labels_df.empty:
        logger.error("Failed to load required data files. Please run ct_rate_downloader.py first.")
        return
    
    # Create visualizations
    logger.info("Creating dataset overview...")
    viz_suite.visualize_dataset_overview(metadata_df, labels_df)
    
    if not args.skip_3d:
        logger.info("Creating 3D volume analysis...")
        viz_suite.visualize_3d_volume_analysis(metadata_df, labels_df, args.n_volume_samples)
    
    if not args.skip_2d:
        # Load slice data for each split
        for split in args.splits:
            logger.info(f"Creating 2D slice analysis for {split}...")
            slice_df = viz_suite.load_slice_data(split)
            if not slice_df.empty:
                viz_suite.visualize_2d_slice_analysis(slice_df)
                break  # Just use first available split for slice analysis
    
    # Create interactive visualizations
    logger.info("Creating interactive abnormality explorer...")
    viz_suite.create_interactive_abnormality_explorer(labels_df)
    
    # Create statistical summary
    logger.info("Creating statistical summary...")
    slice_df = None
    if not args.skip_2d:
        for split in args.splits:
            slice_df = viz_suite.load_slice_data(split)
            if not slice_df.empty:
                break
    
    viz_suite.create_statistical_summary_report(metadata_df, labels_df, slice_df)
    
    logger.info("=" * 60)
    logger.info("🎨 CT Visualization Suite Complete!")
    logger.info("=" * 60)
    logger.info(f"📊 Visualizations saved to: {args.output_dir}")
    logger.info(f"📈 Files created:")
    logger.info(f"   - dataset_overview.png")
    if not args.skip_3d:
        logger.info(f"   - 3d_volume_analysis.png")
    if not args.skip_2d:
        logger.info(f"   - 2d_slice_analysis.png")
    logger.info(f"   - interactive_abnormality_explorer.html")
    logger.info(f"   - statistical_summary.json")
    logger.info(f"   - summary_plots.png")

if __name__ == "__main__":
    main()