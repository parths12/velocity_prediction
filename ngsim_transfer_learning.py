#!/usr/bin/env python3
"""
NGSIM Transfer Learning Script
===============================
Fine-tunes your SUMO-trained model on NGSIM real-world data.

Author: Parth Shinde
Date: November 2025

This script:
1. Loads your SUMO-trained model
2. Freezes early layers (keeps learned patterns)
3. Fine-tunes on NGSIM training data
4. Evaluates on NGSIM test set
5. Compares before/after fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class NGSIMDataset(Dataset):
    """PyTorch Dataset for NGSIM sequences."""

    def __init__(self, sequences: List[Dict]):
        self.sequences = sequences

    def adapt_ngsim_to_sumo_format(self, ngsim_input):
        """Adapt NGSIM input (300, 26) to SUMO format (60, 34)."""
        # Downsample from 300 to 60 steps (sample every 5th frame)
        downsampled = ngsim_input[::5]  # (60, 26)
        
        # Pad from 26 to 34 features
        padded = np.zeros((60, 34))
        padded[:, :26] = downsampled  # Copy NGSIM features
        
        return padded

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Adapt NGSIM format to SUMO format
        adapted_input = self.adapt_ngsim_to_sumo_format(seq['input'])
        return (
            torch.FloatTensor(adapted_input),   # (60, 34)
            torch.FloatTensor(seq['output'])   # (20,)
        )

class TransferLearner:
    """Transfer learning for NGSIM fine-tuning."""

    def __init__(self, model_path: str, ngsim_data_path: str, device: str = 'cuda'):
        """
        Initialize transfer learner.

        Args:
            model_path: Path to SUMO-trained model
            ngsim_data_path: Path to NGSIM processed data
            device: 'cuda' or 'cpu'
        """
        self.model_path = Path(model_path)
        self.ngsim_path = Path(ngsim_data_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"[INFO] Transfer Learning initialized")
        print(f"   Device: {self.device}")
        print(f"   Base model: {self.model_path}")
        print(f"   Data: {self.ngsim_path}")

        # Load model
        self.model = self.load_model()

        # Load data
        self.train_data = self.load_data('ngsim_train.pkl')
        self.test_data = self.load_data('ngsim_test.pkl')

        print(f"   [OK] Loaded {len(self.train_data)} training sequences")
        print(f"   [OK] Loaded {len(self.test_data)} test sequences")

    def load_model(self):
        """Load SUMO-trained model."""
        print(f"\n[LOAD] Loading base model...")

        try:
            from velocity_prediction import iTransformer
            # Load with original SUMO architecture (will adapt NGSIM data)
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
            print("   ❌ Could not import iTransformer")
            raise

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)

        print(f"   [OK] Model loaded")
        return model

    def load_data(self, filename: str):
        """Load NGSIM data."""
        with open(self.ngsim_path / filename, 'rb') as f:
            return pickle.load(f)

    def freeze_layers(self, freeze_embedding: bool = True, freeze_early_layers: bool = True):
        """
        Freeze early layers to preserve SUMO-learned patterns.

        Args:
            freeze_embedding: Freeze embedding layer
            freeze_early_layers: Freeze first transformer layer
        """
        print(f"\n[FREEZE] Freezing layers...")

        frozen_params = 0
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += param.numel()

            # Freeze embedding
            if freeze_embedding and 'embedding' in name:
                param.requires_grad = False
                frozen_params += param.numel()
                print(f"   [OK] Froze: {name}")

            # Freeze first transformer layer
            elif freeze_early_layers and ('encoder' in name and 'layers.0' in name):
                param.requires_grad = False
                frozen_params += param.numel()
                print(f"   [OK] Froze: {name}")

        trainable_params = total_params - frozen_params
        print(f"\n   Total params: {total_params:,}")
        print(f"   Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate model on data loader.

        Args:
            data_loader: DataLoader with test data

        Returns:
            RMSE
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(inputs)

                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        rmse = np.sqrt(mean_squared_error(all_targets.flatten(), all_preds.flatten()))
        return rmse

    def fine_tune(self, epochs: int = 20, batch_size: int = 32, 
                 learning_rate: float = 1e-4, save_path: str = 'ngsim_finetuned_model.pth'):
        """
        Fine-tune model on NGSIM training data.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate (lower than SUMO training)
            save_path: Path to save fine-tuned model
        """
        print(f"\n[TRAIN] Starting fine-tuning...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")

        # Create data loaders
        train_dataset = NGSIMDataset(self.train_data)
        test_dataset = NGSIMDataset(self.test_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer (only for unfrozen parameters)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )

        criterion = nn.MSELoss()

        # Evaluate before fine-tuning
        print(f"\n[EVAL] Evaluating before fine-tuning...")
        before_rmse = self.evaluate(test_loader)
        print(f"   RMSE (before): {before_rmse:.4f} m/s")

        # Training loop
        best_rmse = float('inf')
        train_losses = []
        test_rmses = []

        print(f"\n[TRAIN] Training...")
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                predictions = self.model(inputs)
                loss = criterion(predictions, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Average loss
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Evaluate on test set
            test_rmse = self.evaluate(test_loader)
            test_rmses.append(test_rmse)

            print(f"   Epoch {epoch+1}/{epochs}: ", end='')
            print(f"Train Loss = {avg_loss:.4f}, Test RMSE = {test_rmse:.4f} m/s")

            # Save best model
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_loss,
                    'test_rmse': test_rmse
                }, save_path)
                print(f"      [OK] Best model saved (RMSE: {best_rmse:.4f})")

        # Evaluate after fine-tuning
        print(f"\n[EVAL] Evaluating after fine-tuning...")
        after_rmse = self.evaluate(test_loader)
        print(f"   RMSE (after): {after_rmse:.4f} m/s")

        # Summary
        improvement = before_rmse - after_rmse
        improvement_pct = (improvement / before_rmse) * 100

        print(f"\n[OK] Fine-tuning complete!")
        print(f"   RMSE before: {before_rmse:.4f} m/s")
        print(f"   RMSE after: {after_rmse:.4f} m/s")
        print(f"   Improvement: {improvement:.4f} m/s ({improvement_pct:+.1f}%)")
        print(f"   Best RMSE: {best_rmse:.4f} m/s")
        print(f"   Model saved: {save_path}")

        # Plot training curves
        self.plot_training_curves(train_losses, test_rmses, save_path='ngsim_transfer_learning_curves.png')

        return {
            'before_rmse': before_rmse,
            'after_rmse': after_rmse,
            'best_rmse': best_rmse,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'train_losses': train_losses,
            'test_rmses': test_rmses
        }

    def plot_training_curves(self, train_losses: List[float], test_rmses: List[float], 
                            save_path: str = 'training_curves.png'):
        """Plot training loss and test RMSE curves."""
        epochs = np.arange(1, len(train_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, train_losses, 'o-', color='blue', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss (MSE)', fontsize=12)
        ax1.set_title('Training Loss vs Epoch', fontsize=13, weight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, test_rmses, 'o-', color='red', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Test RMSE (m/s)', fontsize=12)
        ax2.set_title('Test RMSE vs Epoch', fontsize=13, weight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[SAVE] Training curves saved: {save_path}")
        plt.close()

def main():
    """Main transfer learning pipeline."""

    print("="*70)
    print("NGSIM TRANSFER LEARNING")
    print("="*70)

    # Configuration
    BASE_MODEL_PATH = 'combined_velocity_prediction_model.pth'  # Your SUMO-trained model
    NGSIM_DATA_PATH = './ngsim_processed'
    FINETUNED_MODEL_PATH = 'ngsim_finetuned_model.pth'

    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4  # 10x lower than SUMO training (1e-3)

    # Initialize transfer learner
    learner = TransferLearner(
        model_path=BASE_MODEL_PATH,
        ngsim_data_path=NGSIM_DATA_PATH,
        device='cuda'
    )

    # Freeze early layers (preserve SUMO knowledge)
    learner.freeze_layers(
        freeze_embedding=True,      # Keep embedding learned from SUMO
        freeze_early_layers=True    # Keep first transformer layer
    )

    # Fine-tune
    results = learner.fine_tune(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_path=FINETUNED_MODEL_PATH
    )

    print("\n" + "="*70)
    print("🎉 TRANSFER LEARNING COMPLETE!")
    print("="*70)
    print(f"\n📦 Fine-tuned model saved: {FINETUNED_MODEL_PATH}")
    print(f"📊 Training curves saved: ngsim_transfer_learning_curves.png")
    print("\n🚀 Next step: Re-run ngsim_evaluation.py with fine-tuned model!")
    print(f"   Update MODEL_PATH = '{FINETUNED_MODEL_PATH}' in ngsim_evaluation.py")

if __name__ == '__main__':
    main()
