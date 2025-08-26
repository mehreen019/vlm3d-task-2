#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for Multi-Abnormality Classification
VLM3D Task 2: Detailed evaluation with all required metrics
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import argparse
import yaml
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from multi_abnormality_classifier import MultiAbnormalityModel, create_data_loaders, CTSliceDataset, get_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.abnormality_classes = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Fracture', 'Hernia', 'Infiltration',
            'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
            'Support_Devices', 'Thickening', 'No_Finding'
        ]
        
        self.results_dir = Path('./evaluation_results')
        self.results_dir.mkdir(exist_ok=True)
    
    def load_model(self, checkpoint_path: str) -> MultiAbnormalityModel:
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from: {checkpoint_path}")
        model = MultiAbnormalityModel.load_from_checkpoint(checkpoint_path)
        model.eval()
        return model
    
    def predict_on_dataset(self, model: MultiAbnormalityModel, dataloader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate predictions on a dataset"""
        model.eval()
        all_probs = []
        all_labels = []
        all_slice_ids = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating predictions"):
                images = batch['image']
                labels = batch['labels']
                slice_ids = batch['slice_id']
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    images = images.cuda()
                    model = model.cuda()
                
                logits = model(images)
                probs = torch.sigmoid(logits)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
                all_slice_ids.extend(slice_ids)
        
        return np.vstack(all_probs), np.vstack(all_labels), all_slice_ids
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                      threshold: float = 0.5) -> Dict:
        """Calculate all required metrics for VLM3D evaluation"""
        y_pred = (y_prob > threshold).astype(int)
        
        metrics = {}
        
        # Overall metrics (macro and micro averages)
        try:
            # AUROC
            auroc_macro = roc_auc_score(y_true, y_prob, average='macro')
            auroc_micro = roc_auc_score(y_true, y_prob, average='micro')
            metrics['auroc_macro'] = auroc_macro
            metrics['auroc_micro'] = auroc_micro
            
            # F1 Score
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
            metrics['f1_macro'] = f1_macro
            metrics['f1_micro'] = f1_micro
            
            # Precision
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
            metrics['precision_macro'] = precision_macro
            metrics['precision_micro'] = precision_micro
            
            # Recall
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
            metrics['recall_macro'] = recall_macro
            metrics['recall_micro'] = recall_micro
            
            # Accuracy variants
            subset_accuracy = accuracy_score(y_true, y_pred)  # Exact match
            metrics['subset_accuracy'] = subset_accuracy
            
            # Sample-wise accuracy
            sample_accuracy = np.mean([
                np.any(y_true[i] == y_pred[i]) for i in range(len(y_true))
            ])
            metrics['sample_accuracy'] = sample_accuracy
            
            # Average Precision Score (AP)
            ap_macro = average_precision_score(y_true, y_prob, average='macro')
            ap_micro = average_precision_score(y_true, y_prob, average='micro')
            metrics['ap_macro'] = ap_macro
            metrics['ap_micro'] = ap_micro
            
            # Hamming loss (multi-label specific)
            hamming_loss = np.mean(y_true != y_pred)
            metrics['hamming_loss'] = hamming_loss
            
            # Jaccard similarity (multi-label specific)
            jaccard_scores = []
            for i in range(len(y_true)):
                intersection = np.sum(y_true[i] & y_pred[i])
                union = np.sum(y_true[i] | y_pred[i])
                if union == 0:
                    jaccard_scores.append(1.0)  # Both empty sets
                else:
                    jaccard_scores.append(intersection / union)
            metrics['jaccard_score'] = np.mean(jaccard_scores)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
        
        return metrics
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                   threshold: float = 0.5) -> Dict:
        """Calculate per-class metrics"""
        y_pred = (y_prob > threshold).astype(int)
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.abnormality_classes):
            class_true = y_true[:, i]
            class_prob = y_prob[:, i]
            class_pred = y_pred[:, i]
            
            if np.sum(class_true) > 0:  # Only if positive samples exist
                try:
                    auroc = roc_auc_score(class_true, class_prob)
                    f1 = f1_score(class_true, class_pred, zero_division=0)
                    precision = precision_score(class_true, class_pred, zero_division=0)
                    recall = recall_score(class_true, class_pred, zero_division=0)
                    ap = average_precision_score(class_true, class_prob)
                    
                    per_class_metrics[class_name] = {
                        'auroc': float(auroc),
                        'f1': float(f1),
                        'precision': float(precision),
                        'recall': float(recall),
                        'ap': float(ap),
                        'support': int(np.sum(class_true)),
                        'prevalence': float(np.mean(class_true))
                    }
                except Exception as e:
                    logger.warning(f"Error calculating metrics for {class_name}: {e}")
                    per_class_metrics[class_name] = {
                        'auroc': 0.0, 'f1': 0.0, 'precision': 0.0, 
                        'recall': 0.0, 'ap': 0.0,
                        'support': int(np.sum(class_true)),
                        'prevalence': float(np.mean(class_true))
                    }
            else:
                per_class_metrics[class_name] = {
                    'auroc': float('nan'), 'f1': 0.0, 'precision': 0.0, 
                    'recall': 0.0, 'ap': float('nan'),
                    'support': 0,
                    'prevalence': 0.0
                }
        
        return per_class_metrics
    
    def calculate_vlm3d_score(self, metrics: Dict) -> float:
        """Calculate VLM3D challenge score with weighted metrics"""
        weights = self.config['evaluation']['metrics']
        
        score = (
            metrics.get('auroc_macro', 0) * weights['auroc_weight'] +
            metrics.get('f1_macro', 0) * weights['f1_weight'] +
            metrics.get('precision_macro', 0) * weights['precision_weight'] +
            metrics.get('recall_macro', 0) * weights['recall_weight'] +
            metrics.get('subset_accuracy', 0) * weights['accuracy_weight']
        )
        
        return score
    
    def create_visualizations(self, y_true: np.ndarray, y_prob: np.ndarray, 
                            per_class_metrics: Dict):
        """Create comprehensive visualizations"""
        
        # 1. ROC Curves for each class
        plt.figure(figsize=(20, 15))
        
        # Plot ROC curves in subplots
        n_classes = len(self.abnormality_classes)
        n_rows = 5
        n_cols = 4
        
        for i, class_name in enumerate(self.abnormality_classes):
            plt.subplot(n_rows, n_cols, i + 1)
            
            class_true = y_true[:, i]
            class_prob = y_prob[:, i]
            
            if np.sum(class_true) > 0:
                fpr, tpr, _ = roc_curve(class_true, class_prob)
                auroc = per_class_metrics[class_name]['auroc']
                plt.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{class_name}')
                plt.legend(loc="lower right")
            else:
                plt.text(0.5, 0.5, 'No positive samples', ha='center', va='center')
                plt.title(f'{class_name}')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'roc_curves_per_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precision-Recall Curves
        plt.figure(figsize=(20, 15))
        
        for i, class_name in enumerate(self.abnormality_classes):
            plt.subplot(n_rows, n_cols, i + 1)
            
            class_true = y_true[:, i]
            class_prob = y_prob[:, i]
            
            if np.sum(class_true) > 0:
                precision, recall, _ = precision_recall_curve(class_true, class_prob)
                ap = per_class_metrics[class_name]['ap']
                plt.plot(recall, precision, label=f'AP = {ap:.3f}')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'{class_name}')
                plt.legend(loc="lower left")
            else:
                plt.text(0.5, 0.5, 'No positive samples', ha='center', va='center')
                plt.title(f'{class_name}')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'pr_curves_per_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance summary heatmap
        metrics_matrix = []
        metric_names = ['AUROC', 'F1', 'Precision', 'Recall', 'AP']
        
        for class_name in self.abnormality_classes:
            class_metrics = per_class_metrics[class_name]
            row = [
                class_metrics.get('auroc', 0),
                class_metrics.get('f1', 0),
                class_metrics.get('precision', 0),
                class_metrics.get('recall', 0),
                class_metrics.get('ap', 0)
            ]
            metrics_matrix.append(row)
        
        plt.figure(figsize=(8, 12))
        sns.heatmap(
            metrics_matrix,
            xticklabels=metric_names,
            yticklabels=self.abnormality_classes,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Score'}
        )
        plt.title('Per-Class Performance Metrics')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Class prevalence vs Performance
        prevalences = [per_class_metrics[class_name]['prevalence'] for class_name in self.abnormality_classes]
        aurocs = [per_class_metrics[class_name].get('auroc', 0) for class_name in self.abnormality_classes]
        
        plt.figure(figsize=(12, 8))
        plt.scatter(prevalences, aurocs, alpha=0.7, s=100)
        
        for i, class_name in enumerate(self.abnormality_classes):
            plt.annotate(class_name, (prevalences[i], aurocs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Class Prevalence')
        plt.ylabel('AUROC')
        plt.title('Class Prevalence vs AUROC Performance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'prevalence_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.results_dir}")
    
    def evaluate_model(self, checkpoint_path: str) -> Dict:
        """Complete model evaluation"""
        logger.info("Starting comprehensive model evaluation...")
        
        # Load model
        model = self.load_model(checkpoint_path)
        
        # Create data loaders
        _, val_loader, test_loader, _ = create_data_loaders(self.config)
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_probs, val_labels, val_ids = self.predict_on_dataset(model, val_loader)
        
        # Evaluate on test set (if available)
        test_probs, test_labels, test_ids = None, None, None
        if Path(self.config['data']['test_csv']).exists():
            logger.info("Evaluating on test set...")
            test_probs, test_labels, test_ids = self.predict_on_dataset(model, test_loader)
        
        # Calculate metrics
        val_metrics = self.calculate_comprehensive_metrics(val_labels, val_probs)
        val_per_class = self.calculate_per_class_metrics(val_labels, val_probs)
        val_vlm3d_score = self.calculate_vlm3d_score(val_metrics)
        
        results = {
            'validation': {
                'overall_metrics': val_metrics,
                'per_class_metrics': val_per_class,
                'vlm3d_score': val_vlm3d_score,
                'n_samples': len(val_labels)
            }
        }
        
        if test_probs is not None:
            test_metrics = self.calculate_comprehensive_metrics(test_labels, test_probs)
            test_per_class = self.calculate_per_class_metrics(test_labels, test_probs)
            test_vlm3d_score = self.calculate_vlm3d_score(test_metrics)
            
            results['test'] = {
                'overall_metrics': test_metrics,
                'per_class_metrics': test_per_class,
                'vlm3d_score': test_vlm3d_score,
                'n_samples': len(test_labels)
            }
        
        # Create visualizations
        primary_labels = test_labels if test_labels is not None else val_labels
        primary_probs = test_probs if test_probs is not None else val_probs
        primary_per_class = results.get('test', results['validation'])['per_class_metrics']
        
        self.create_visualizations(primary_labels, primary_probs, primary_per_class)
        
        # Save detailed results
        results_file = self.results_dir / 'comprehensive_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation complete! Results saved to {results_file}")
        
        return results
    
    def print_results_summary(self, results: Dict):
        """Print a summary of evaluation results"""
        logger.info("=" * 80)
        logger.info("📊 EVALUATION RESULTS SUMMARY")
        logger.info("=" * 80)
        
        for split_name, split_results in results.items():
            logger.info(f"\n{split_name.upper()} SET RESULTS:")
            logger.info("-" * 40)
            
            overall = split_results['overall_metrics']
            vlm3d_score = split_results['vlm3d_score']
            n_samples = split_results['n_samples']
            
            logger.info(f"Samples: {n_samples}")
            logger.info(f"VLM3D Score: {vlm3d_score:.4f}")
            logger.info(f"")
            logger.info(f"Overall Metrics (Macro/Micro):")
            logger.info(f"  AUROC:     {overall.get('auroc_macro', 0):.4f} / {overall.get('auroc_micro', 0):.4f}")
            logger.info(f"  F1:        {overall.get('f1_macro', 0):.4f} / {overall.get('f1_micro', 0):.4f}")
            logger.info(f"  Precision: {overall.get('precision_macro', 0):.4f} / {overall.get('precision_micro', 0):.4f}")
            logger.info(f"  Recall:    {overall.get('recall_macro', 0):.4f} / {overall.get('recall_micro', 0):.4f}")
            logger.info(f"  Accuracy:  {overall.get('subset_accuracy', 0):.4f} (subset) / {overall.get('sample_accuracy', 0):.4f} (sample)")
            
            # Top performing classes
            per_class = split_results['per_class_metrics']
            auroc_scores = {k: v.get('auroc', 0) for k, v in per_class.items() 
                           if not np.isnan(v.get('auroc', 0))}
            
            if auroc_scores:
                top_classes = sorted(auroc_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"\nTop 5 Classes (by AUROC):")
                for class_name, auroc in top_classes:
                    logger.info(f"  {class_name}: {auroc:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Abnormality Classification Model")
    parser.add_argument("--config", 
                       default="config_multi_abnormality.yaml",
                       help="Configuration file path")
    parser.add_argument("--checkpoint", required=True, 
                       help="Model checkpoint path")
    parser.add_argument("--output-dir", 
                       default="./evaluation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    evaluator.results_dir = Path(args.output_dir)
    evaluator.results_dir.mkdir(exist_ok=True)
    
    # Run evaluation
    results = evaluator.evaluate_model(args.checkpoint)
    
    # Print summary
    evaluator.print_results_summary(results)
    
    logger.info(f"\n🎉 Evaluation complete!")
    logger.info(f"📁 Results saved to: {evaluator.results_dir}")
    logger.info(f"📊 Visualizations available in: {evaluator.results_dir}")

if __name__ == "__main__":
    main() 