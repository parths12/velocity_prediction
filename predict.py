#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import the iTransformer model from velocity_prediction.py
from velocity_prediction import iTransformer, preprocess_data, predict_velocity

def load_model(model_path, num_variates, lookback_len, pred_length):
    """
    Load a trained iTransformer model.
    
    Args:
        model_path: Path to the saved model
        num_variates: Number of variates in the data
        lookback_len: Number of time steps to look back
        pred_length: Number of time steps to predict
        
    Returns:
        Loaded model
    """
    # Initialize model with the same parameters as during training
    model = iTransformer(
        num_variates=num_variates,
        lookback_len=lookback_len,
        dim=128,
        depth=4,
        heads=8,
        dim_head=32,
        pred_length=pred_length
    )
    
    # Load saved weights
    model.load_state_dict(torch.load(model_path))
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def main():
    # Configuration
    model_path = "velocity_prediction_model.pth"
    data_path = "simulation_data.csv"
    lookback_len = 60  # 60 seconds of historical data
    pred_length = 20  # Predict 20 seconds into the future
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Preprocess data
    print("Preprocessing data...")
    X, scaler = preprocess_data(df)
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, X.shape[1], lookback_len, pred_length)
    
    # Predict future velocity
    print("Predicting future velocity...")
    predicted_velocity = predict_velocity(model, X, scaler, lookback_len, pred_length)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot historical velocity
    actual_velocity = df['ego_speed'].values
    plt.plot(range(len(actual_velocity)), actual_velocity, label='Historical Velocity')
    
    # Plot predicted velocity
    time_pred = range(len(actual_velocity), len(actual_velocity) + pred_length)
    plt.plot(time_pred, predicted_velocity, label='Predicted Velocity', color='red')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Vehicle Velocity Prediction')
    plt.legend()
    plt.grid(True)
    plt.savefig('velocity_prediction_new.png')
    plt.show()
    
    print(f"Predicted velocity for the next {pred_length} seconds:")
    for i, vel in enumerate(predicted_velocity):
        print(f"t+{i+1}s: {vel:.2f} m/s")

if __name__ == "__main__":
    main() 