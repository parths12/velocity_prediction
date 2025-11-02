#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse

# Import functions from velocity_prediction.py
from velocity_prediction import iTransformer, preprocess_data
from predict import load_model

def evaluate_model(model_path, data_path, lookback_len=60, pred_length=20):
    """
    Evaluate the model's performance on the given data.
    
    Args:
        model_path: Path to the saved model
        data_path: Path to the data CSV file
        lookback_len: Number of time steps to look back
        pred_length: Number of time steps to predict
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Preprocess data
    X, scaler = preprocess_data(df)
    
    # Load model
    model = load_model(model_path, X.shape[1], lookback_len, pred_length)
    
    # Create dataset for evaluation
    # We'll use a sliding window approach to make predictions and compare with actual values
    predictions = []
    actuals = []
    
    for i in range(len(X) - lookback_len - pred_length + 1):
        # Get input sequence
        input_seq = X[i:i+lookback_len]
        # Get actual future values (only velocity)
        actual_future = X[i+lookback_len:i+lookback_len+pred_length, 0]
        
        # Make prediction
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0)  # Add batch dimension
        model.eval()
        with torch.no_grad():
            pred = model(input_tensor).numpy()[0]
        
        # Store prediction and actual values
        predictions.append(pred)
        actuals.append(actual_future)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse transform to get actual velocity values
    # Create dummy arrays with the same shape as the original data
    pred_dummy = np.zeros((len(predictions), pred_length, X.shape[1]))
    actual_dummy = np.zeros((len(actuals), pred_length, X.shape[1]))
    
    # Put the predicted/actual velocity in the first column
    for i in range(len(predictions)):
        pred_dummy[i, :, 0] = predictions[i]
        actual_dummy[i, :, 0] = actuals[i]
    
    # Reshape for inverse transform
    pred_dummy_reshaped = pred_dummy.reshape(-1, X.shape[1])
    actual_dummy_reshaped = actual_dummy.reshape(-1, X.shape[1])
    
    # Inverse transform
    pred_inv = scaler.inverse_transform(pred_dummy_reshaped)[:, 0].reshape(-1, pred_length)
    actual_inv = scaler.inverse_transform(actual_dummy_reshaped)[:, 0].reshape(-1, pred_length)
    
    # Calculate metrics
    # For each prediction horizon (1s, 2s, ..., pred_length)
    metrics = {
        'mse': [],
        'rmse': [],
        'mae': [],
        'mape': []
    }
    
    for t in range(pred_length):
        # Get predictions and actuals for this time step
        pred_t = pred_inv[:, t]
        actual_t = actual_inv[:, t]
        
        # Calculate metrics
        mse = mean_squared_error(actual_t, pred_t)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_t, pred_t)
        
        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        mask = actual_t != 0
        mape = np.mean(np.abs((actual_t[mask] - pred_t[mask]) / actual_t[mask])) * 100
        
        # Store metrics
        metrics['mse'].append(mse)
        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        metrics['mape'].append(mape)
    
    # Calculate average metrics across all time steps
    avg_metrics = {
        'avg_mse': np.mean(metrics['mse']),
        'avg_rmse': np.mean(metrics['rmse']),
        'avg_mae': np.mean(metrics['mae']),
        'avg_mape': np.mean(metrics['mape'])
    }
    
    # Combine metrics
    all_metrics = {**metrics, **avg_metrics}
    
    return all_metrics, pred_inv, actual_inv

def plot_metrics(metrics, output_file='model_evaluation.png'):
    """
    Plot evaluation metrics.
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_file: Path to save the plot
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot MSE
    axs[0, 0].plot(range(1, len(metrics['mse'])+1), metrics['mse'], 'b-', marker='o')
    axs[0, 0].set_title('Mean Squared Error (MSE)')
    axs[0, 0].set_xlabel('Prediction Horizon (s)')
    axs[0, 0].set_ylabel('MSE')
    axs[0, 0].grid(True)
    
    # Plot RMSE
    axs[0, 1].plot(range(1, len(metrics['rmse'])+1), metrics['rmse'], 'r-', marker='o')
    axs[0, 1].set_title('Root Mean Squared Error (RMSE)')
    axs[0, 1].set_xlabel('Prediction Horizon (s)')
    axs[0, 1].set_ylabel('RMSE (m/s)')
    axs[0, 1].grid(True)
    
    # Plot MAE
    axs[1, 0].plot(range(1, len(metrics['mae'])+1), metrics['mae'], 'g-', marker='o')
    axs[1, 0].set_title('Mean Absolute Error (MAE)')
    axs[1, 0].set_xlabel('Prediction Horizon (s)')
    axs[1, 0].set_ylabel('MAE (m/s)')
    axs[1, 0].grid(True)
    
    # Plot MAPE
    axs[1, 1].plot(range(1, len(metrics['mape'])+1), metrics['mape'], 'm-', marker='o')
    axs[1, 1].set_title('Mean Absolute Percentage Error (MAPE)')
    axs[1, 1].set_xlabel('Prediction Horizon (s)')
    axs[1, 1].set_ylabel('MAPE (%)')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved metrics plot to {output_file}")
    
    plt.show()

def plot_predictions(predictions, actuals, num_samples=5, output_file='prediction_samples.png'):
    """
    Plot sample predictions vs actual values.
    
    Args:
        predictions: Array of predictions
        actuals: Array of actual values
        num_samples: Number of samples to plot
        output_file: Path to save the plot
    """
    # Select random samples
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    
    # Create figure with subplots
    fig, axs = plt.subplots(len(indices), 1, figsize=(12, 4*len(indices)))
    
    # If only one sample, make axs iterable
    if len(indices) == 1:
        axs = [axs]
    
    # Plot each sample
    for i, idx in enumerate(indices):
        pred = predictions[idx]
        actual = actuals[idx]
        
        # Plot
        axs[i].plot(range(1, len(pred)+1), pred, 'r-', marker='o', label='Predicted')
        axs[i].plot(range(1, len(actual)+1), actual, 'b-', marker='x', label='Actual')
        axs[i].set_title(f'Sample {i+1}')
        axs[i].set_xlabel('Prediction Horizon (s)')
        axs[i].set_ylabel('Velocity (m/s)')
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved prediction samples plot to {output_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--model', type=str, default='velocity_prediction_model.pth', help='Path to the model file')
    parser.add_argument('--data', type=str, default='simulation_data.csv', help='Path to the data file')
    parser.add_argument('--lookback', type=int, default=60, help='Number of time steps to look back')
    parser.add_argument('--horizon', type=int, default=20, help='Number of time steps to predict')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found.")
        return
    
    if not os.path.exists(args.data):
        print(f"Data file {args.data} not found.")
        return
    
    # Evaluate model
    print(f"Evaluating model {args.model} on data {args.data}...")
    metrics, predictions, actuals = evaluate_model(args.model, args.data, args.lookback, args.horizon)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Average MSE: {metrics['avg_mse']:.4f}")
    print(f"Average RMSE: {metrics['avg_rmse']:.4f} m/s")
    print(f"Average MAE: {metrics['avg_mae']:.4f} m/s")
    print(f"Average MAPE: {metrics['avg_mape']:.2f}%")
    
    # Print metrics for each prediction horizon
    print("\nMetrics by Prediction Horizon:")
    for t in range(args.horizon):
        print(f"t+{t+1}s: MSE={metrics['mse'][t]:.4f}, RMSE={metrics['rmse'][t]:.4f} m/s, MAE={metrics['mae'][t]:.4f} m/s, MAPE={metrics['mape'][t]:.2f}%")
    
    # Plot metrics
    plot_metrics(metrics)
    
    # Plot sample predictions
    plot_predictions(predictions, actuals)

if __name__ == "__main__":
    main() 