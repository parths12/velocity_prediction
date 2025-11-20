#!/usr/bin/env python3
"""
NGSIM Model Evaluation Script
==============================
Evaluates your SUMO-trained iTransformer model on NGSIM real-world data.

Author: Parth Shinde
Date: November 2025

This script:
1. Loads your trained model (SUMO weights)
2. Loads NGSIM processed data
3. Performs zero-shot evaluation (no fine-tuning)
4. Computes metrics (RMSE, MAE, MAPE)
5. Visualizes predictions vs actuals
6. Measures domain gap
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class NGSIMEvaluator:
    """Evaluates model on NGSIM test data."""

    def __init__(self, model_path: str, ngsim_data_path: str, device: str = 'cuda'):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model weights (.pth file)
            ngsim_data_path: Path to NGSIM processed data folder
            device: 'cuda' or 'cpu'
        """
        self.model_path = Path(model_path)
        self.ngsim_path = Path(ngsim_data_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"[INFO] NGSIM Evaluator initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.model_path}")
        print(f"   Data: {self.ngsim_path}")

        # Load model
        self.model = self.load_model()

        # Load data
        self.test_data = self.load_ngsim_data('ngsim_test.pkl')
        self.metadata = self.load_ngsim_data('ngsim_metadata.pkl')

        print(f"   ✓ Loaded {len(self.test_data)} test sequences")

    def load_model(self):
        """Load trained model from checkpoint."""
        print(f"\n[LOAD] Loading model...")

        # Import your model architecture (adjust import as needed)
        # Note: Model was trained with 34 features and 60 lookback, but we'll adapt NGSIM data
        try:
            from velocity_prediction import iTransformer
            # Load with original SUMO architecture
            model = iTransformer(
                num_variates=34,  # Original SUMO model uses 34 features
                lookback_len=60,  # Original SUMO model uses 60 lookback steps
                dim=128,
                depth=4,
                heads=8,
                dim_head=32,
                pred_length=20
            )
        except ImportError:
            print("   ❌ Could not import iTransformer from velocity_prediction.py")
            print("   Make sure velocity_prediction.py is in the current directory")
            raise

        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        print(f"   [OK] Model loaded successfully")
        return model

    def load_ngsim_data(self, filename: str):
        """Load NGSIM processed data."""
        filepath = self.ngsim_path / filename
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data

    def adapt_ngsim_to_sumo_format(self, ngsim_input):
        """
        Adapt NGSIM input (300, 26) to SUMO format (60, 34).
        
        Args:
            ngsim_input: NGSIM input array (300, 26)
            
        Returns:
            SUMO format array (60, 34)
        """
        # Downsample from 300 to 60 steps (sample every 5th frame)
        downsampled = ngsim_input[::5]  # (60, 26)
        
        # Pad from 26 to 34 features (add zeros for missing features)
        # SUMO has: ego_speed, ego_accel, ego_lane_pos, ego_lane + 6 vehicles * 5 features each
        # NGSIM has: 26 features (ego_velocity + 12 surrounding vehicles * 2 + lane)
        # We'll pad with zeros for the missing features
        padded = np.zeros((60, 34))
        padded[:, :26] = downsampled  # Copy NGSIM features
        
        return padded

    def prepare_batch(self, sequences: List[Dict], batch_size: int = 32):
        """
        Prepare batches for evaluation.

        Args:
            sequences: List of sequence dictionaries
            batch_size: Batch size

        Yields:
            Batches of (inputs, targets)
        """
        n_sequences = len(sequences)

        for i in range(0, n_sequences, batch_size):
            batch_seq = sequences[i:i + batch_size]

            # Adapt NGSIM format to SUMO format
            adapted_inputs = []
            for seq in batch_seq:
                adapted = self.adapt_ngsim_to_sumo_format(seq['input'])
                adapted_inputs.append(adapted)
            
            # Stack inputs and outputs
            inputs = torch.FloatTensor(adapted_inputs)
            targets = torch.FloatTensor([seq['output'] for seq in batch_seq])

            yield inputs.to(self.device), targets.to(self.device)

    def evaluate(self, batch_size: int = 32) -> Dict:
        """
        Evaluate model on NGSIM test set.

        Args:
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with metrics
        """
        print(f"\n[EVAL] Evaluating on NGSIM test set...")

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.prepare_batch(self.test_data, batch_size)):
                # Forward pass
                predictions = self.model(inputs)

                # Store predictions and targets
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    print(f"   Batch {batch_idx + 1}/{len(self.test_data)//batch_size}...", end='\r')

        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute metrics
        results = self.compute_metrics(all_predictions, all_targets)

        # Store for visualization
        self.predictions = all_predictions
        self.targets = all_targets

        print(f"\n   [OK] Evaluation complete")
        return results

    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """
        Compute evaluation metrics.

        Args:
            predictions: Predicted velocities (n_samples, 20)
            targets: Actual velocities (n_samples, 20)

        Returns:
            Dictionary with metrics
        """
        # Overall metrics
        mse = mean_squared_error(targets.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets.flatten(), predictions.flatten())

        # Avoid division by zero in MAPE
        mask = targets.flatten() != 0
        mape = np.mean(np.abs((targets.flatten()[mask] - predictions.flatten()[mask]) / 
                              targets.flatten()[mask])) * 100

        # Aggregate accuracy (like your SUMO evaluation)
        mean_pred = predictions.mean(axis=1)
        mean_target = targets.mean(axis=1)
        aggregate_mape = np.mean(np.abs((mean_target - mean_pred) / mean_target)) * 100
        aggregate_accuracy = 100 - aggregate_mape

        # Per-timestep metrics
        per_timestep_rmse = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
        per_timestep_mae = np.mean(np.abs(predictions - targets), axis=0)

        results = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'aggregate_accuracy': aggregate_accuracy,
            'per_timestep_rmse': per_timestep_rmse,
            'per_timestep_mae': per_timestep_mae,
            'n_samples': len(predictions)
        }

        return results

    def print_results(self, results: Dict, sumo_rmse: float = 0.356):
        """
        Print evaluation results.

        Args:
            results: Results dictionary from evaluate()
            sumo_rmse: Your SUMO test RMSE for comparison
        """
        print("\n" + "="*70)
        print("NGSIM EVALUATION RESULTS")
        print("="*70)

        print(f"\n[METRICS] Overall Metrics:")
        print(f"   RMSE: {results['rmse']:.4f} m/s")
        print(f"   MAE: {results['mae']:.4f} m/s")
        print(f"   MAPE: {results['mape']:.2f}%")
        print(f"   Aggregate Accuracy: {results['aggregate_accuracy']:.2f}%")
        print(f"   Samples: {results['n_samples']}")

        # Domain gap analysis
        domain_gap = results['rmse'] - sumo_rmse
        domain_gap_pct = (domain_gap / sumo_rmse) * 100

        print(f"\n[DOMAIN] Domain Gap Analysis:")
        print(f"   SUMO RMSE: {sumo_rmse:.4f} m/s")
        print(f"   NGSIM RMSE: {results['rmse']:.4f} m/s")
        print(f"   Domain Gap: {domain_gap:.4f} m/s ({domain_gap_pct:+.1f}%)")

        if domain_gap < 0.2:
            print(f"   [OK] Excellent! Small domain gap - model generalizes well")
        elif domain_gap < 0.3:
            print(f"   [OK] Good domain gap - acceptable generalization")
        elif domain_gap < 0.5:
            print(f"   [WARN] Moderate domain gap - fine-tuning recommended")
        else:
            print(f"   [ERROR] Large domain gap - fine-tuning required")

        # Per-timestep summary
        print(f"\n[TIMESTEP] Per-Timestep RMSE:")
        print(f"   1-5s: {results['per_timestep_rmse'][:5].mean():.4f} m/s")
        print(f"   6-10s: {results['per_timestep_rmse'][5:10].mean():.4f} m/s")
        print(f"   11-15s: {results['per_timestep_rmse'][10:15].mean():.4f} m/s")
        print(f"   16-20s: {results['per_timestep_rmse'][15:].mean():.4f} m/s")

        print("\n" + "="*70)

    def visualize_predictions(self, n_samples: int = 5, save_path: str = 'ngsim_evaluation_results'):
        """
        Visualize predictions vs actuals.

        Args:
            n_samples: Number of sample trajectories to plot
            save_path: Directory to save figures
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[VIZ] Creating visualizations...")

        # 1. Sample trajectories
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
        if n_samples == 1:
            axes = [axes]

        time_steps = np.arange(1, 21)
        indices = np.random.choice(len(self.predictions), n_samples, replace=False)

        for i, idx in enumerate(indices):
            ax = axes[i]

            pred = self.predictions[idx]
            actual = self.targets[idx]

            ax.plot(time_steps, actual, 'o-', color='green', 
                   label='Actual', linewidth=2, markersize=6, alpha=0.8)
            ax.plot(time_steps, pred, 's-', color='blue', 
                   label='Predicted', linewidth=2, markersize=4, alpha=0.8)

            ax.set_xlabel('Prediction Horizon (seconds)', fontsize=11)
            ax.set_ylabel('Velocity (m/s)', fontsize=11)
            ax.set_title(f'Sample {i+1}: NGSIM Trajectory Prediction', fontsize=12, weight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'ngsim_sample_predictions.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved sample predictions: {save_dir}/ngsim_sample_predictions.png")
        plt.close()

        # 2. Error vs prediction horizon
        per_timestep_rmse = np.sqrt(np.mean((self.predictions - self.targets) ** 2, axis=0))
        per_timestep_mae = np.mean(np.abs(self.predictions - self.targets), axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(time_steps, per_timestep_rmse, 'o-', color='red', 
                linewidth=2, markersize=6, label='RMSE')
        ax1.set_xlabel('Prediction Horizon (seconds)', fontsize=12)
        ax1.set_ylabel('RMSE (m/s)', fontsize=12)
        ax1.set_title('RMSE vs Prediction Horizon (NGSIM)', fontsize=13, weight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        ax2.plot(time_steps, per_timestep_mae, 'o-', color='orange', 
                linewidth=2, markersize=6, label='MAE')
        ax2.set_xlabel('Prediction Horizon (seconds)', fontsize=12)
        ax2.set_ylabel('MAE (m/s)', fontsize=12)
        ax2.set_title('MAE vs Prediction Horizon (NGSIM)', fontsize=13, weight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(save_dir / 'ngsim_error_vs_horizon.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved error plot: {save_dir}/ngsim_error_vs_horizon.png")
        plt.close()

        # 3. Prediction scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Sample points for clearer visualization
        sample_indices = np.random.choice(self.predictions.size, min(5000, self.predictions.size), replace=False)
        pred_flat = self.predictions.flatten()[sample_indices]
        actual_flat = self.targets.flatten()[sample_indices]

        ax.scatter(actual_flat, pred_flat, alpha=0.3, s=10)
        ax.plot([actual_flat.min(), actual_flat.max()], 
               [actual_flat.min(), actual_flat.max()], 
               'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Velocity (m/s)', fontsize=12)
        ax.set_ylabel('Predicted Velocity (m/s)', fontsize=12)
        ax.set_title('Predicted vs Actual Velocities (NGSIM)', fontsize=13, weight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.savefig(save_dir / 'ngsim_scatter_plot.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved scatter plot: {save_dir}/ngsim_scatter_plot.png")
        plt.close()

        print(f"\n[OK] All visualizations saved to: {save_dir}/")

    def save_results(self, results: Dict, save_path: str = 'ngsim_evaluation_results'):
        """
        Save evaluation results to file.

        Args:
            results: Results dictionary
            save_path: Directory to save results
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save as pickle
        with open(save_dir / 'ngsim_results.pkl', 'wb') as f:
            pickle.dump(results, f)

        # Save as text
        with open(save_dir / 'ngsim_results.txt', 'w') as f:
            f.write("NGSIM EVALUATION RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"RMSE: {results['rmse']:.4f} m/s\n")
            f.write(f"MAE: {results['mae']:.4f} m/s\n")
            f.write(f"MAPE: {results['mape']:.2f}%\n")
            f.write(f"Aggregate Accuracy: {results['aggregate_accuracy']:.2f}%\n")
            f.write(f"\nPer-Timestep RMSE:\n")
            for i, rmse in enumerate(results['per_timestep_rmse'], 1):
                f.write(f"  {i}s: {rmse:.4f} m/s\n")

        print(f"\n[SAVE] Results saved to: {save_dir}/")

def main():
    """Main evaluation pipeline."""

    print("="*70)
    print("NGSIM MODEL EVALUATION - ZERO-SHOT")
    print("="*70)

    # Configuration
    MODEL_PATH = 'ngsim_finetuned_model.pth'  # Path to fine-tuned model
    NGSIM_DATA_PATH = './ngsim_processed'  # Path to processed NGSIM data
    RESULTS_PATH = './ngsim_evaluation_results'
    SUMO_RMSE = 0.356  # Your SUMO test RMSE for comparison

    # Initialize evaluator
    evaluator = NGSIMEvaluator(
        model_path=MODEL_PATH,
        ngsim_data_path=NGSIM_DATA_PATH,
        device='cuda'  # Change to 'cpu' if no GPU
    )

    # Evaluate
    results = evaluator.evaluate(batch_size=32)

    # Print results
    evaluator.print_results(results, sumo_rmse=SUMO_RMSE)

    # Visualize
    evaluator.visualize_predictions(n_samples=10, save_path=RESULTS_PATH)

    # Save results
    evaluator.save_results(results, save_path=RESULTS_PATH)

    print("\n" + "="*70)
    print("🎉 EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n📦 Results saved to: {RESULTS_PATH}/")
    print("   - ngsim_results.pkl")
    print("   - ngsim_results.txt")
    print("   - ngsim_sample_predictions.png")
    print("   - ngsim_error_vs_horizon.png")
    print("   - ngsim_scatter_plot.png")
    print("\n🚀 Next step: If domain gap is large, run ngsim_transfer_learning.py for fine-tuning!")

if __name__ == '__main__':
    main()
