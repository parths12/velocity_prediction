#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
import time
import glob
import pandas as pd
from velocity_prediction import preprocess_data, train_model, TimeSeriesDataset, evaluate_model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

def run_command(command, description):
    """
    Run a command and print its output.
    
    Args:
        command: Command to run
        description: Description of the command
    """
    print(f"\n{'='*80}")
    print(f"Running {description}...")
    print(f"{'='*80}\n")
    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\nError running {description}. Return code: {process.returncode}")
        return False
    
    return True

def combine_simulation_data():
    # Find all simulation_data_*.csv files
    files = glob.glob('simulation_data_*.csv')
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    # Ensure all required columns exist
    required_cols = ['ego_speed', 'ego_accel', 'ego_lane', 'ego_lane_pos']
    for i in range(1, 7):
        required_cols.extend([
            f'veh{i}_distance', f'veh{i}_speed', f'veh{i}_accel', f'veh{i}_is_ahead', f'veh{i}_rel_speed'])
    for col in required_cols:
        if col not in combined_df.columns:
            combined_df[col] = 0
    # Reorder columns to match expected order
    other_cols = [c for c in combined_df.columns if c not in required_cols]
    combined_df = combined_df[required_cols + other_cols]
    combined_df.to_csv('combined_simulation_data.csv', index=False)
    print(f"Combined {len(files)} files into combined_simulation_data.csv with {len(combined_df)} rows.")
    return combined_df

def main():
    # Step 1: Combine data
    df = combine_simulation_data()
    # Step 2: Preprocess
    print("Preprocessing data...")
    X, scaler = preprocess_data(df)
    # Step 3: Train
    print("Training model...")
    lookback_len = 60
    pred_length = 20
    dataset = TimeSeriesDataset(X, lookback_len, pred_length)
    model = train_model(X, lookback_len, pred_length)
    # Step 4: Evaluate
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=32)
    criterion = nn.MSELoss()
    metrics = evaluate_model(model, test_loader, criterion)
    print("\nModel Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"Accuracy (within 5% error): {metrics['accuracy']:.2f}%")
    print(f"Average Inference Time: {metrics['inference_time_ms']:.2f} ms per batch")
    # Save model
    torch.save(model.state_dict(), "velocity_prediction_model.pth")
    print("Saved model to velocity_prediction_model.pth")

if __name__ == "__main__":
    main() 