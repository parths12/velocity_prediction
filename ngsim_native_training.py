#!/usr/bin/env python3
"""
NGSIM Improved Training Script with Hyperparameter Tuning & Data Augmentation
===============================================================================
Enhanced version targeting RMSE < 2.0 m/s

Improvements:
1. Larger model (d_model=256, 3 layers)
2. Lower learning rate for stability  
3. Data augmentation (Gaussian noise, scaling, time masking)
4. Longer training (200 epochs)
5. Better regularization

Author: Parth Shinde
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Data Augmentation Functions
# ============================================================================

class DataAugmentation:
    """Data augmentation for time-series data."""
    
    def __init__(self, noise_std=0.01, scale_range=(0.95, 1.05), mask_prob=0.1):
        """
        Args:
            noise_std: Standard deviation for Gaussian noise
            scale_range: Range for random scaling
            mask_prob: Probability of masking each timestep
        """
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.mask_prob = mask_prob
    
    def add_gaussian_noise(self, x):
        """Add Gaussian noise to input."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def random_scaling(self, x):
        """Apply random scaling to input."""
        scale = torch.FloatTensor(1).uniform_(*self.scale_range).item()
        return x * scale
    
    def time_masking(self, x):
        """Randomly mask timesteps."""
        mask = torch.rand(x.size(0), 1) > self.mask_prob
        mask = mask.expand_as(x)
        return x * mask.float()
    
    def augment(self, x, training=True):
        """
        Apply augmentation (only during training).
        
        Args:
            x: Input tensor (seq_len, num_features)
            training: Whether in training mode
        """
        if not training:
            return x
        
        # Randomly apply augmentations
        if torch.rand(1).item() < 0.5:
            x = self.add_gaussian_noise(x)
        if torch.rand(1).item() < 0.3:
            x = self.random_scaling(x)
        if torch.rand(1).item() < 0.2:
            x = self.time_masking(x)
        
        return x


# ============================================================================
# Improved Model Architecture
# ============================================================================

class ImprovedNGSIMiTransformer(nn.Module):
    """
    Improved iTransformer with larger capacity for better performance.
    
    Key improvements:
    - Larger d_model (256 vs 128)
    - More layers (3 vs 2)
    - Better dropout strategy
    - Residual connections in prediction head
    """
    
    def __init__(
        self,
        num_features: int = 26,
        seq_length: int = 300,
        d_model: int = 256,              # Increased from 128
        nhead: int = 4,
        num_layers: int = 3,             # Increased from 2
        prediction_horizon: int = 20,
        dropout: float = 0.15            # Slightly increased dropout
    ):
        super(ImprovedNGSIMiTransformer, self).__init__()
        
        self.num_features = num_features
        self.seq_length = seq_length
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        
        # Feature embedding (per-variate)
        self.feature_embedding = nn.Sequential(
            nn.Linear(seq_length, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_features, d_model) * 0.02
        )
        
        # Transformer encoder layers with layer normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Improved prediction head with residual connections
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * num_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout/2),
            
            nn.Linear(128, prediction_horizon)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_length, num_features)
        
        Returns:
            predictions: (batch, prediction_horizon)
        """
        batch_size = x.size(0)
        
        # Transpose to (batch, num_features, seq_length)
        x = x.transpose(1, 2)
        
        # Embed each feature independently
        x = self.feature_embedding(x)  # (batch, num_features, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Flatten for prediction
        x = x.reshape(batch_size, -1)
        
        # Predict future velocities
        predictions = self.prediction_head(x)
        
        return predictions


# ============================================================================
# Improved Dataset with Augmentation
# ============================================================================

class AugmentedNGSIMDataset(Dataset):
    """Dataset with data augmentation."""
    
    def __init__(self, sequences: List[Dict], augmentation=None, training=True):
        """
        Args:
            sequences: List of dicts with 'input' and 'output'
            augmentation: DataAugmentation object
            training: Whether in training mode
        """
        self.sequences = sequences
        self.augmentation = augmentation
        self.training = training
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.FloatTensor(seq['input'])  # (300, 26)
        y = torch.FloatTensor(seq['output']) # (20,)
        
        # Apply augmentation if available
        if self.augmentation is not None:
            x = self.augmentation.augment(x, training=self.training)
        
        return x, y


# ============================================================================
# Improved Training Manager
# ============================================================================

class ImprovedNGSIMTrainer:
    """Enhanced trainer with better optimization strategies."""
    
    def __init__(
        self,
        ngsim_data_path: str = './ngsim_processed',
        model_save_path: str = './models',
        device: str = 'cuda'
    ):
        self.ngsim_path = Path(ngsim_data_path)
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*70)
        print("IMPROVED NGSIM TRAINER - OPTIMIZED FOR RMSE < 2.0")
        print("="*70)
        print(f"\n[INITIALIZATION]:")
        print(f"   Device: {self.device}")
        print(f"   Data path: {self.ngsim_path}")
        print(f"   Model save path: {self.model_save_path}")
        
        # Load data
        self.train_data = self.load_data('ngsim_train.pkl')
        self.val_data = self.load_data('ngsim_val.pkl')
        self.test_data = self.load_data('ngsim_test.pkl')
        self.metadata = self.load_data('ngsim_metadata.pkl')
        
        # Create validation set if missing
        if len(self.val_data) == 0 and len(self.train_data) > 0:
            print("\n⚠️  Creating validation set from training data...")
            val_size = max(1, int(len(self.train_data) * 0.1))
            indices = np.random.permutation(len(self.train_data))
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            
            original_train = self.train_data
            self.val_data = [original_train[i] for i in val_indices]
            self.train_data = [original_train[i] for i in train_indices]
        
        print(f"\n📊 Data loaded:")
        print(f"   Train: {len(self.train_data)} sequences")
        print(f"   Val: {len(self.val_data)} sequences")
        print(f"   Test: {len(self.test_data)} sequences")
        
        # Extract dimensions
        if len(self.train_data) > 0:
            sample = self.train_data[0]
            self.seq_length = sample['input'].shape[0]
            self.num_features = sample['input'].shape[1]
            self.prediction_horizon = len(sample['output'])
            
            print(f"\n📐 Dimensions:")
            print(f"   Input: ({self.seq_length}, {self.num_features})")
            print(f"   Output: ({self.prediction_horizon},)")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.augmentation = DataAugmentation(
            noise_std=0.01,
            scale_range=(0.95, 1.05),
            mask_prob=0.1
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'learning_rate': []
        }
    
    def load_data(self, filename: str):
        """Load preprocessed data."""
        filepath = self.ngsim_path / filename
        if not filepath.exists():
            print(f"   ⚠️  {filename} not found")
            return []
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def initialize_model(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.15
    ):
        """Initialize improved model."""
        print(f"\n🏗️  Building improved model...")
        
        self.model = ImprovedNGSIMiTransformer(
            num_features=self.num_features,
            seq_length=self.seq_length,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            prediction_horizon=self.prediction_horizon,
            dropout=dropout
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   ✅ Model created")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model capacity: ~{total_params / 1e6:.2f}M parameters")
    
    def setup_training(
        self,
        learning_rate: float = 5e-4,     # Lower than before
        weight_decay: float = 1e-4,      # Stronger regularization
        warmup_epochs: int = 5
    ):
        """Setup optimizer with warmup."""
        print(f"\n⚙️  Setting up training...")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (200 - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda
        )
        
        print(f"   ✅ Optimizer: AdamW")
        print(f"      - Learning rate: {learning_rate}")
        print(f"      - Weight decay: {weight_decay}")
        print(f"      - Warmup epochs: {warmup_epochs}")
        print(f"   ✅ Scheduler: Cosine Annealing with Warmup")
    
    def train_epoch(self, data_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch with gradient clipping."""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions = self.model(inputs)
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"   Batch {batch_idx+1}/{len(data_loader)}: Loss={loss.item():.4f}", 
                      end='\r')
        
        avg_loss = epoch_loss / len(data_loader)
        return avg_loss
    
    def validate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Validate on validation set."""
        if len(data_loader) == 0:
            return 0.0, float('inf')
        
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)
                
                val_loss += loss.item()
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = val_loss / len(data_loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        rmse = np.sqrt(mean_squared_error(all_targets.flatten(), all_preds.flatten()))
        
        return avg_loss, rmse
    
    def train(
        self,
        epochs: int = 200,               # Increased from 100
        batch_size: int = 16,            # Smaller batch for better gradients
        early_stopping_patience: int = 25,
        save_best: bool = True
    ):
        """Main training loop with improvements."""
        print(f"\n{'='*70}")
        print(f"STARTING IMPROVED TRAINING")
        print(f"{'='*70}")
        print(f"\n📋 Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Early stopping patience: {early_stopping_patience}")
        print(f"   Data augmentation: ✅ Enabled")
        
        # Create augmented datasets
        train_dataset = AugmentedNGSIMDataset(
            self.train_data,
            augmentation=self.augmentation,
            training=True
        )
        val_dataset = AugmentedNGSIMDataset(
            self.val_data,
            augmentation=None,  # No augmentation for validation
            training=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"\n🚀 Training started...")
        best_val_rmse = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_rmse = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_rmse'].append(val_rmse)
            self.history['learning_rate'].append(current_lr)
            
            print(f"\n   📊 Results:")
            print(f"      Train Loss: {train_loss:.4f}")
            print(f"      Val Loss: {val_loss:.4f}")
            print(f"      Val RMSE: {val_rmse:.4f} m/s")
            print(f"      Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
                
                if save_best:
                    self.model_save_path.mkdir(parents=True, exist_ok=True)
                    model_path = self.model_save_path / 'ngsim_improved_best.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_rmse': val_rmse,
                        'val_loss': val_loss,
                        'history': self.history
                    }, model_path)
                    print(f"      ✅ Best model saved! (RMSE: {val_rmse:.4f})")
            else:
                patience_counter += 1
                print(f"      ⏳ Patience: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered")
                print(f"   Best val RMSE: {best_val_rmse:.4f} m/s")
                break
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"\n   🎯 Best Val RMSE: {best_val_rmse:.4f} m/s")
        print(f"   📅 Total Epochs: {epoch+1}")
        
        return best_val_rmse
    
    def evaluate_test(self, model_path: str = None) -> Dict:
        """Evaluate on test set."""
        print(f"\n{'='*70}")
        print(f"EVALUATING ON TEST SET")
        print(f"{'='*70}")
        
        # Load best model
        if model_path:
            print(f"\n📥 Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)  # Using weights_only=False to allow loading numpy arrays
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Model loaded")
        
        # Create test dataset (no augmentation)
        test_dataset = AugmentedNGSIMDataset(
            self.test_data,
            augmentation=None,
            training=False
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Evaluate
        self.model.eval()
        all_preds = []
        all_targets = []
        
        print(f"\n🔍 Running inference...")
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(inputs)
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"   Batch {batch_idx+1}/{len(test_loader)}...", end='\r')
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        rmse = np.sqrt(mean_squared_error(all_targets.flatten(), all_preds.flatten()))
        mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
        
        mask = all_targets.flatten() != 0
        mape = np.mean(np.abs((all_targets.flatten()[mask] - all_preds.flatten()[mask]) / 
                              all_targets.flatten()[mask])) * 100
        
        per_timestep_rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2, axis=0))
        per_timestep_mae = np.mean(np.abs(all_preds - all_targets), axis=0)
        
        mean_pred = all_preds.mean(axis=1)
        mean_target = all_targets.mean(axis=1)
        aggregate_mape = np.mean(np.abs((mean_target - mean_pred) / mean_target)) * 100
        aggregate_accuracy = 100 - aggregate_mape
        
        results = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'aggregate_accuracy': aggregate_accuracy,
            'per_timestep_rmse': per_timestep_rmse,
            'per_timestep_mae': per_timestep_mae,
            'predictions': all_preds,
            'targets': all_targets
        }
        
        # Print results
        print(f"\n\n{'='*70}")
        print(f"📊 TEST RESULTS")
        print(f"{'='*70}")
        print(f"\n   Overall Metrics:")
        print(f"      RMSE: {rmse:.4f} m/s")
        print(f"      MAE: {mae:.4f} m/s")
        print(f"      MAPE: {mape:.2f}%")
        print(f"      Aggregate Accuracy: {aggregate_accuracy:.2f}%")
        
        print(f"\n   📈 Per-Horizon RMSE:")
        print(f"      1-5s:   {per_timestep_rmse[:5].mean():.4f} m/s")
        print(f"      6-10s:  {per_timestep_rmse[5:10].mean():.4f} m/s")
        print(f"      11-15s: {per_timestep_rmse[10:15].mean():.4f} m/s")
        print(f"      16-20s: {per_timestep_rmse[15:].mean():.4f} m/s")
        
        # Target check
        if rmse < 2.0:
            print(f"\n   ✅ SUCCESS! Achieved target RMSE < 2.0 m/s!")
        elif rmse < 3.0:
            print(f"\n   ✓ Great improvement! Close to publication target.")
        else:
            print(f"\n   ⚠️  More tuning needed for publication target.")
        
        return results
    
    def plot_training_history(self, save_path: str = 'ngsim_improved_training.png'):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = np.arange(1, len(self.history['train_loss']) + 1)
        
        # Training loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'o-', 
                       label='Train Loss', linewidth=2, markersize=3)
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss (MSE)', fontsize=11)
        axes[0, 0].set_title('Training Loss', fontsize=13, weight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Validation loss
        axes[0, 1].plot(epochs, self.history['val_loss'], 'o-', color='orange', 
                       label='Val Loss', linewidth=2, markersize=3)
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('Loss (MSE)', fontsize=11)
        axes[0, 1].set_title('Validation Loss', fontsize=13, weight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Validation RMSE
        axes[1, 0].plot(epochs, self.history['val_rmse'], 'o-', color='red', 
                       label='Val RMSE', linewidth=2, markersize=3)
        axes[1, 0].axhline(y=2.0, color='green', linestyle='--', 
                          label='Target (2.0)', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('RMSE (m/s)', fontsize=11)
        axes[1, 0].set_title('Validation RMSE', fontsize=13, weight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Learning rate
        axes[1, 1].plot(epochs, self.history['learning_rate'], 'o-', color='green', 
                       label='Learning Rate', linewidth=2, markersize=3)
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=11)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=13, weight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Training history saved: {save_path}")
        plt.close()


def main():
    """Main training pipeline with improvements."""
    
    print("\n" + "="*70)
    print("NGSIM IMPROVED TRAINING - TARGET RMSE < 2.0 M/S")
    print("="*70)
    print("\n[KEY IMPROVEMENTS]")
    print("   - Larger model (d_model=256, 3 layers)")
    print("   - Lower learning rate (5e-4)")
    print("   - Data augmentation (noise, scaling, masking)")
    print("   - Cosine annealing with warmup")
    print("   - Longer training (200 epochs)")
    print("   - Better regularization")
    
    # Configuration
    NGSIM_DATA_PATH = './ngsim_processed'
    MODEL_SAVE_PATH = './models'
    
    # Improved hyperparameters
    D_MODEL = 256              # Increased from 128
    NHEAD = 4
    NUM_LAYERS = 3             # Increased from 2
    DROPOUT = 0.15             # Increased from 0.1
    LEARNING_RATE = 5e-4       # Decreased from 1e-3
    WEIGHT_DECAY = 1e-4        # Increased from 1e-5
    EPOCHS = 200               # Increased from 100
    BATCH_SIZE = 16            # Decreased from 32
    EARLY_STOPPING_PATIENCE = 25
    WARMUP_EPOCHS = 5
    
    # Initialize trainer
    trainer = ImprovedNGSIMTrainer(
        ngsim_data_path=NGSIM_DATA_PATH,
        model_save_path=MODEL_SAVE_PATH,
        device='cuda'
    )
    
    # Initialize model
    trainer.initialize_model(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    # Setup training
    trainer.setup_training(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS
    )
    
    # Train
    best_val_rmse = trainer.train(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        save_best=True
    )
    
    # Plot training history
    trainer.plot_training_history('ngsim_improved_training_history.png')
    
    # Evaluate on test set
    test_results = trainer.evaluate_test(
        model_path=MODEL_SAVE_PATH + '/ngsim_improved_best.pth'
    )
    
    # Save results
    results_path = Path('./ngsim_improved_results.json')
    with open(results_path, 'w') as f:
        json_results = {
            'rmse': float(test_results['rmse']),
            'mae': float(test_results['mae']),
            'mape': float(test_results['mape']),
            'aggregate_accuracy': float(test_results['aggregate_accuracy']),
            'per_timestep_rmse': test_results['per_timestep_rmse'].tolist(),
            'per_timestep_mae': test_results['per_timestep_mae'].tolist()
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"ALL DONE!")
    print(f"{'='*70}")
    print(f"\n📦 Outputs:")
    print(f"   Best model: {MODEL_SAVE_PATH}/ngsim_improved_best.pth")
    print(f"   Training history: ngsim_improved_training_history.png")
    print(f"   Test results: ngsim_improved_results.json")
    
    print(f"\n📈 Improvement Summary:")
    print(f"   Previous RMSE: 3.14 m/s")
    print(f"   New RMSE: {test_results['rmse']:.2f} m/s")
    improvement = ((3.14 - test_results['rmse']) / 3.14) * 100
    print(f"   Improvement: {improvement:.1f}%")
    
    if test_results['rmse'] < 2.0:
        print(f"\n🎉 SUCCESS! Publication-ready results achieved!")
    elif test_results['rmse'] < 2.5:
        print(f"\n✓ Excellent results! Very close to publication target.")
    else:
        print(f"\n⚠️  Consider more training data or further tuning.")


if __name__ == '__main__':
    main()