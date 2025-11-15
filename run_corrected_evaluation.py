#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run the corrected model evaluation
Uses evaluate_model_CORRECTED.py to properly align predictions
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from existing modules
from velocity_prediction import iTransformer, preprocess_data, TimeSeriesDataset
from predict import load_model
from evaluate_model_CORRECTED import plot_predictions_correctly

# Non-interactive plotting
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_test_loader_and_actuals(csv_path, lookback_len=60, pred_length=20, batch_size=64):
    """
    Create test loader and extract actual velocities from CSV
    
    Args:
        csv_path: Path to CSV file
        lookback_len: Lookback window size
        pred_length: Prediction horizon
        batch_size: Batch size for DataLoader
        
    Returns:
        test_loader: DataLoader for test data
        actual_velocities: Numpy array of actual velocities (original scale)
        X_scaled: Scaled data array
        scaler: Scaler used for preprocessing
    """
    print(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract actual velocities BEFORE preprocessing (original scale)
    # The velocity is in the 'ego_speed' column
    if 'ego_speed' in df.columns:
        actual_velocities = df['ego_speed'].values
    else:
        # If column name is different, try to find velocity column
        print("Warning: 'ego_speed' column not found. Looking for velocity column...")
        if 'velocity' in df.columns:
            actual_velocities = df['velocity'].values
        else:
            raise ValueError("Could not find velocity column in CSV")
    
    print(f"  Found {len(actual_velocities)} time steps")
    print(f"  Velocity range: {actual_velocities.min():.2f} - {actual_velocities.max():.2f} m/s")
    
    # Preprocess data (this will scale it)
    print("Preprocessing data...")
    X_scaled, scaler = preprocess_data(df)
    print(f"  Scaled data shape: {X_scaled.shape}")
    
    # Create dataset
    dataset = TimeSeriesDataset(X_scaled, lookback_len, pred_length)
    print(f"  Dataset size: {len(dataset)} samples")
    
    # Create test loader (use all data for evaluation)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader, actual_velocities, X_scaled, scaler


def evaluate_model_with_inverse_transform(model, test_loader, actual_velocities, scaler, 
                                         num_features, traffic_density_name="Test", 
                                         lookback_len=60, pred_horizon=20):
    """
    Wrapper around evaluate_model that handles inverse transformation of predictions
    """
    model.eval()
    
    # Initialize predictions array with SAME LENGTH as actual data
    predictions_aligned = np.zeros(len(actual_velocities))
    
    all_predictions = []
    all_targets = []
    
    print(f"\nEvaluating on {traffic_density_name}...")
    print(f"  Actual velocities length: {len(actual_velocities)}")
    print(f"  Lookback window: {lookback_len}s")
    print(f"  Prediction horizon: {pred_horizon}s")
    
    velocity_idx = 0
    
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            X_batch = X_batch.float()
            y_batch = y_batch.float()
            
            # Get predictions from model (in scaled space)
            predictions_scaled = model(X_batch)  # Shape: (batch_size, pred_horizon)
            
            batch_size = X_batch.shape[0]
            
            # Process each sample in batch
            for sample_idx in range(batch_size):
                pred_start_idx = velocity_idx + lookback_len
                
                # Inverse transform predictions
                pred_scaled = predictions_scaled[sample_idx].cpu().numpy()  # (pred_horizon,)
                # Create dummy array for inverse transform
                dummy = np.zeros((pred_horizon, num_features))
                dummy[:, 0] = pred_scaled  # Velocity is first column
                pred_inv = scaler.inverse_transform(dummy)[:, 0]  # Get velocity column
                
                # Store predictions at correct positions
                for h in range(pred_horizon):
                    actual_idx = pred_start_idx + h
                    if actual_idx < len(predictions_aligned):
                        predictions_aligned[actual_idx] = pred_inv[h]
                
                # Also inverse transform targets for metrics
                target_scaled = y_batch[sample_idx].cpu().numpy()
                dummy_target = np.zeros((pred_horizon, num_features))
                dummy_target[:, 0] = target_scaled
                target_inv = scaler.inverse_transform(dummy_target)[:, 0]
                
                all_predictions.extend(pred_inv)
                all_targets.extend(target_inv)
                
                velocity_idx += 1
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    if len(all_predictions) > 0:
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        mape = np.mean(np.abs((all_targets - all_predictions) / (np.abs(all_targets) + 1e-6))) * 100
        
        # R² score
        ss_res = np.sum((all_targets - all_predictions) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        mse = rmse = mae = mape = r2 = 0
    
    # Print metrics
    print(f"\n  Results:")
    print(f"    MSE:  {mse:.6f}")
    print(f"    RMSE: {rmse:.6f} m/s")
    print(f"    MAE:  {mae:.6f} m/s")
    print(f"    MAPE: {mape:.2f}%")
    print(f"    R²:   {r2:.4f}")
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }
    
    return metrics, predictions_aligned


def main():
    """Main function to run corrected evaluation"""
    
    # Configuration
    model_path = "combined_velocity_prediction_model.pth"
    data_path = "combined_simulation_data.csv"
    lookback_len = 60
    pred_length = 20
    
    print("="*70)
    print("CORRECTED MODEL EVALUATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Lookback: {lookback_len}s, Prediction: {pred_length}s")
    print("="*70)
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file {model_path} not found!")
        print("Available model files:")
        for f in os.listdir('.'):
            if f.endswith('.pth'):
                print(f"  - {f}")
        return
    
    if not os.path.exists(data_path):
        print(f"\nERROR: Data file {data_path} not found!")
        print("Available CSV files:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                print(f"  - {f}")
        return
    
    # Load data and create test loader
    try:
        test_loader, actual_velocities, X_scaled, scaler = create_test_loader_and_actuals(
            data_path, lookback_len, pred_length
        )
    except Exception as e:
        print(f"\nERROR creating test loader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    try:
        num_variates = X_scaled.shape[1]
        model = load_model(model_path, num_variates, lookback_len, pred_length)
        print(f"  Model loaded successfully")
        print(f"  Number of features: {num_variates}")
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run evaluation with inverse transformation
    print("\n" + "="*70)
    print("RUNNING CORRECTED EVALUATION")
    print("="*70)
    
    try:
        # Evaluate model with inverse transformation
        metrics, predictions_aligned = evaluate_model_with_inverse_transform(
            model=model,
            test_loader=test_loader,
            actual_velocities=actual_velocities,
            scaler=scaler,
            num_features=num_variates,
            traffic_density_name="Combined Data",
            lookback_len=lookback_len,
            pred_horizon=pred_length
        )
        
        # Plot results with correct alignment
        print("\nGenerating visualization...")
        plot_predictions_correctly(
            actual_velocities=actual_velocities,
            predictions_aligned=predictions_aligned,
            traffic_density_name="Combined Data",
            lookback_len=lookback_len
        )
        
        # Print summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"MSE:  {metrics['mse']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f} m/s")
        print(f"MAE:  {metrics['mae']:.6f} m/s")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"R²:   {metrics['r2']:.4f}")
        print("="*70)
        
        print("\n✓ Evaluation completed successfully!")
        print("✓ Visualization saved as: velocity_prediction_combined_data.png")
        
    except Exception as e:
        print(f"\nERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

