"""
Comprehensive visualization and analysis module for thesis results.

Generates publication-ready plots:
1. ANN training curves (loss & accuracy) - SimpleCNN, VGG16, ResNet18
2. Surrogate gradient functions vs membrane potential
3. Accuracy vs T time-steps (convergence analysis)
4. Spike count comparison (VGG vs ResNet)
5. Convergence speed metrics
6. Confusion matrices
7. Precision, Recall, F1-score tables

All plots saved to results/ directory with high resolution (300 DPI).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    precision_score, recall_score, f1_score
)
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

from snn_model import rate_code_input
from utils import accuracy as compute_accuracy


class ThesisVisualizer:
    """
    Central class for all thesis visualizations and analysis.
    """
    
    def __init__(self, output_dir: str = "./results"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save all figures and tables
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style for publication-ready plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_training_curves(self, histories: Dict[str, Dict], datasets: List[str],
                            filename: str = "training_curves.png") -> None:
        """
        Plot training and validation loss/accuracy for multiple models.
        
        Args:
            histories: Dict of {model_name: {train_loss, train_acc, val_loss, val_acc}}
            datasets: List of dataset names for the title
            filename: Output filename
        """
        print("Generating training curves...")
        
        fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 5*len(datasets)))
        if len(datasets) == 1:
            axes = axes.reshape(1, -1)
        
        colors = {'SimpleCNN': '#2E86AB', 'VGG16': '#A23B72', 'ResNet18': '#F18F01'}
        
        for idx, dataset in enumerate(datasets):
            # Loss
            ax_loss = axes[idx, 0]
            ax_acc = axes[idx, 1]
            
            for model_name, history in histories.items():
                if history:
                    ax_loss.plot(
                        history.get('train_loss', []),
                        label=f"{model_name} (train)",
                        marker='o',
                        markersize=3,
                        alpha=0.7,
                        color=colors.get(model_name, '#000000')
                    )
                    ax_loss.plot(
                        history.get('val_loss', []),
                        label=f"{model_name} (val)",
                        linestyle='--',
                        marker='s',
                        markersize=3,
                        alpha=0.7,
                        color=colors.get(model_name, '#000000')
                    )
                    
                    # Accuracy
                    ax_acc.plot(
                        history.get('train_acc', []),
                        label=f"{model_name} (train)",
                        marker='o',
                        markersize=3,
                        alpha=0.7,
                        color=colors.get(model_name, '#000000')
                    )
                    ax_acc.plot(
                        history.get('val_acc', []),
                        label=f"{model_name} (val)",
                        linestyle='--',
                        marker='s',
                        markersize=3,
                        alpha=0.7,
                        color=colors.get(model_name, '#000000')
                    )
            
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_title(f'{dataset}: Training Loss')
            ax_loss.legend(loc='best', fontsize=8)
            ax_loss.grid(True, alpha=0.3)
            
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.set_title(f'{dataset}: Training Accuracy')
            ax_acc.legend(loc='best', fontsize=8)
            ax_acc.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")
    
    def plot_surrogate_gradients(self, filename: str = "surrogate_gradients.png") -> None:
        """
        Plot surrogate gradient functions vs membrane potential.
        
        Shows how STDB approximates non-differentiable spike function.
        """
        print("Generating surrogate gradient plots...")
        
        # Create membrane potential range
        u_mem = np.linspace(-2, 2, 1000)
        v_th = 1.0
        
        # Surrogate functions
        def sigmoid_surrogate(u, vth=1.0, beta=5.0):
            x = beta * (u - vth)
            return beta / (1 + np.abs(x))
        
        def exponential_surrogate(u, vth=1.0, beta=2.0):
            return beta * np.exp(-beta * np.abs(u - vth))
        
        def true_spike(u, vth=1.0):
            return (u > vth).astype(float)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Surrogate functions
        ax = axes[0]
        ax.plot(u_mem, sigmoid_surrogate(u_mem), label='Sigmoid Surrogate (β=5)', linewidth=2.5)
        ax.plot(u_mem, exponential_surrogate(u_mem), label='Exponential Surrogate (β=2)', linewidth=2.5)
        ax.axvline(v_th, color='red', linestyle='--', label='Threshold V_th', linewidth=2)
        ax.fill_between(u_mem, 0, true_spike(u_mem), alpha=0.2, label='True Spike (binary)')
        ax.set_xlabel('Membrane Potential u_mem')
        ax.set_ylabel('Gradient / Spike')
        ax.set_title('Surrogate Gradient Functions for STDB')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Plot 2: Comparison of different beta values
        ax = axes[1]
        for beta in [1.0, 2.0, 5.0, 10.0]:
            ax.plot(u_mem, sigmoid_surrogate(u_mem, beta=beta), 
                   label=f'Sigmoid (β={beta})', linewidth=2.5)
        ax.axvline(v_th, color='red', linestyle='--', label='Threshold V_th', linewidth=2)
        ax.set_xlabel('Membrane Potential u_mem')
        ax.set_ylabel('Gradient')
        ax.set_title('Effect of Sharpness Parameter β')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")
    
    def plot_accuracy_vs_timesteps(self, results: Dict[str, List[float]],
                                  filename: str = "accuracy_vs_timesteps.png") -> None:
        """
        Plot SNN accuracy vs number of time steps (T).
        
        Shows convergence behavior: how quickly SNN accuracy stabilizes.
        
        Args:
            results: Dict of {model_name: [acc_T50, acc_T100, acc_T200, acc_T500, ...]}
            filename: Output filename
        """
        print("Generating accuracy vs T plots...")
        
        time_steps = [50, 100, 200, 300, 500]
        colors = {'SimpleCNN': '#2E86AB', 'VGG16': '#A23B72', 'ResNet18': '#F18F01'}
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Accuracy vs T for all models
        ax = axes[0]
        for model_name, accuracies in results.items():
            ax.plot(time_steps[:len(accuracies)], accuracies, 
                   marker='o', linewidth=2.5, markersize=8,
                   label=model_name, color=colors.get(model_name, '#000000'))
        
        ax.set_xlabel('Time Steps (T)')
        ax.set_ylabel('Accuracy')
        ax.set_title('SNN Accuracy Convergence vs Time Steps')
        ax.set_xscale('log')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, which='both')
        
        # Plot 2: Convergence speed (% improvement from T=50)
        ax = axes[1]
        for model_name, accuracies in results.items():
            if len(accuracies) > 0:
                baseline = accuracies[0]
                improvement = [(acc - baseline) * 100 for acc in accuracies]
                ax.plot(time_steps[:len(accuracies)], improvement,
                       marker='s', linewidth=2.5, markersize=8,
                       label=model_name, color=colors.get(model_name, '#000000'))
        
        ax.set_xlabel('Time Steps (T)')
        ax.set_ylabel('Accuracy Improvement from T=50 (%)')
        ax.set_title('SNN Convergence Speed')
        ax.set_xscale('log')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")
    
    def plot_spike_counts(self, spike_data: Dict[str, Dict[str, float]],
                         filename: str = "spike_counts.png") -> None:
        """
        Plot spike count statistics for different architectures.
        
        Args:
            spike_data: Dict of {model_name: {layer_name: spike_count}}
            filename: Output filename
        """
        print("Generating spike count plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models = list(spike_data.keys())
        layer_names = list(spike_data[models[0]].keys()) if models else []
        
        # Plot 1: Spike counts per layer
        ax = axes[0]
        x = np.arange(len(layer_names))
        width = 0.25
        
        colors = {'SimpleCNN': '#2E86AB', 'VGG16': '#A23B72', 'ResNet18': '#F18F01'}
        
        for idx, model_name in enumerate(models):
            counts = [spike_data[model_name].get(layer, 0) for layer in layer_names]
            ax.bar(x + idx*width, counts, width, label=model_name,
                  color=colors.get(model_name, '#000000'), alpha=0.8)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Total Spike Count')
        ax.set_title('Spike Counts by Layer')
        ax.set_xticks(x + width)
        ax.set_xticklabels(layer_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Firing rates (normalized)
        ax = axes[1]
        firing_rates = {}
        for model_name in models:
            max_spikes = max(spike_data[model_name].values())
            firing_rates[model_name] = {
                layer: spike_data[model_name][layer] / max_spikes 
                for layer in layer_names
            }
        
        for idx, model_name in enumerate(models):
            rates = [firing_rates[model_name].get(layer, 0) for layer in layer_names]
            ax.bar(x + idx*width, rates, width, label=model_name,
                  color=colors.get(model_name, '#000000'), alpha=0.8)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Normalized Firing Rate')
        ax.set_title('Normalized Firing Rates')
        ax.set_xticks(x + width)
        ax.set_xticklabels(layer_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")
    
    def plot_convergence_speed(self, convergence_data: Dict[str, Dict],
                              filename: str = "convergence_speed.png") -> None:
        """
        Plot convergence speed metrics for ANN training.
        
        Args:
            convergence_data: Dict of {model_name: {epochs_to_90, epochs_to_95, best_epoch, ...}}
            filename: Output filename
        """
        print("Generating convergence speed analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(convergence_data.keys())
        colors = {'SimpleCNN': '#2E86AB', 'VGG16': '#A23B72', 'ResNet18': '#F18F01'}
        
        # Plot 1: Epochs to reach 90% accuracy
        ax = axes[0, 0]
        epochs_90 = [convergence_data[m].get('epochs_to_90', 0) for m in models]
        bars1 = ax.bar(models, epochs_90, color=[colors.get(m, '#000000') for m in models], alpha=0.8)
        ax.set_ylabel('Epochs')
        ax.set_title('Epochs to Reach 90% Accuracy')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars1, epochs_90):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{int(val)}', ha='center', va='bottom')
        
        # Plot 2: Epochs to reach 95% accuracy
        ax = axes[0, 1]
        epochs_95 = [convergence_data[m].get('epochs_to_95', 0) for m in models]
        bars2 = ax.bar(models, epochs_95, color=[colors.get(m, '#000000') for m in models], alpha=0.8)
        ax.set_ylabel('Epochs')
        ax.set_title('Epochs to Reach 95% Accuracy')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars2, epochs_95):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{int(val)}', ha='center', va='bottom')
        
        # Plot 3: Best validation accuracy
        ax = axes[1, 0]
        best_accs = [convergence_data[m].get('best_acc', 0) for m in models]
        bars3 = ax.bar(models, best_accs, color=[colors.get(m, '#000000') for m in models], alpha=0.8)
        ax.set_ylabel('Accuracy')
        ax.set_title('Best Validation Accuracy')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars3, best_accs):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}', ha='center', va='bottom')
        
        # Plot 4: Training time comparison
        ax = axes[1, 1]
        train_times = [convergence_data[m].get('training_time_hours', 0) for m in models]
        bars4 = ax.bar(models, train_times, color=[colors.get(m, '#000000') for m in models], alpha=0.8)
        ax.set_ylabel('Time (hours)')
        ax.set_title('Training Time on GPU')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars4, train_times):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}h', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str], model_name: str,
                             dataset_name: str,
                             filename: str = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: List of class names
            model_name: Name of model (for title)
            dataset_name: Name of dataset (for title)
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            filename = f"confusion_matrix_{model_name}_{dataset_name}.png"
        
        print(f"Generating confusion matrix: {model_name} on {dataset_name}...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix: {model_name} on {dataset_name}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")
    
    def save_metrics_table(self, y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str], model_name: str,
                          dataset_name: str,
                          filename: str = None) -> pd.DataFrame:
        """
        Compute and save precision, recall, F1-score table.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: List of class names
            model_name: Name of model
            dataset_name: Name of dataset
            filename: Output filename (auto-generated if None)
            
        Returns:
            DataFrame with metrics
        """
        if filename is None:
            filename = f"metrics_{model_name}_{dataset_name}.csv"
        
        print(f"Computing metrics: {model_name} on {dataset_name}...")
        
        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(class_names)), zero_division=0
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'Class': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # Add macro and weighted averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        # Append summary rows
        df = pd.concat([
            df,
            pd.DataFrame({
                'Class': ['Macro Average', 'Weighted Average'],
                'Precision': [macro_precision, weighted_precision],
                'Recall': [macro_recall, weighted_recall],
                'F1-Score': [macro_f1, weighted_f1],
                'Support': [support.sum(), support.sum()]
            })
        ], ignore_index=True)
        
        # Save to CSV
        df.to_csv(self.output_dir / filename, index=False)
        print(f"✓ Saved: {filename}")
        
        return df
    
    def create_metrics_heatmap(self, metrics_data: Dict[str, pd.DataFrame],
                              filename: str = "metrics_heatmap.png") -> None:
        """
        Create heatmap comparing metrics across models and datasets.
        
        Args:
            metrics_data: Dict of {f"{model}_{dataset}": df}
            filename: Output filename
        """
        print("Generating metrics heatmap...")
        
        # Extract F1-scores for heatmap
        f1_data = {}
        for key, df in metrics_data.items():
            model, dataset = key.rsplit('_', 1)
            if model not in f1_data:
                f1_data[model] = {}
            # Get macro F1 (last row)
            macro_f1 = df.iloc[-2]['F1-Score']  # Macro average row
            f1_data[model][dataset] = macro_f1
        
        # Create DataFrame
        heatmap_df = pd.DataFrame(f1_data).T
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax,
                   cbar_kws={'label': 'F1-Score'}, vmin=0.7, vmax=1.0)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Model')
        ax.set_title('F1-Score Comparison: Models vs Datasets')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")
    
    def create_summary_report(self, all_results: Dict, filename: str = "thesis_summary.json") -> None:
        """
        Save comprehensive summary report in JSON.
        
        Args:
            all_results: Dictionary with all experimental results
            filename: Output filename
        """
        print("Creating summary report...")
        
        # Convert tensors to float for JSON serialization
        results_serializable = {}
        for key, value in all_results.items():
            if isinstance(value, dict):
                results_serializable[key] = {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                                            for k, v in value.items()}
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                results_serializable[key] = float(value)
            else:
                results_serializable[key] = value
        
        with open(self.output_dir / filename, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"✓ Saved: {filename}")


def print_metrics_table(df: pd.DataFrame, title: str = "Metrics Table") -> None:
    """
    Pretty print metrics table to console.
    
    Args:
        df: DataFrame with metrics
        title: Table title
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")
