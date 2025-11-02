#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import random
import xml.etree.ElementTree as ET

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

# iTransformer model implementation
class iTransformer(nn.Module):
    def __init__(self, num_variates, lookback_len, dim, depth, heads, dim_head, pred_length):
        super().__init__()
        self.num_variates = num_variates
        self.lookback_len = lookback_len
        self.pred_length = pred_length
        
        # Embedding for each variate (time series)
        self.variate_embedding = nn.Linear(lookback_len, dim)
        
        # Positional encoding for variates
        self.variate_pos_embedding = nn.Parameter(torch.randn(1, num_variates, dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Prediction head
        self.pred_head = nn.Linear(dim, pred_length)
        
    def forward(self, x):
        # x shape: [batch_size, lookback_len, num_variates]
        batch_size = x.shape[0]
        
        # Transpose to [batch_size, num_variates, lookback_len]
        x = x.transpose(1, 2)
        
        # Embed each variate sequence
        x = self.variate_embedding(x)  # [batch_size, num_variates, dim]
        
        # Add positional encoding
        x = x + self.variate_pos_embedding
        
        # Pass through transformer
        x = self.transformer_encoder(x)  # [batch_size, num_variates, dim]
        
        # Predict future values for each variate
        predictions = self.pred_head(x)  # [batch_size, num_variates, pred_length]
        
        # For velocity prediction, we only need the first variate (our vehicle's velocity)
        velocity_pred = predictions[:, 0, :]  # [batch_size, pred_length]
        
        return velocity_pred

# Dataset class for time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback_len, pred_length):
        self.data = data
        self.lookback_len = lookback_len
        self.pred_length = pred_length
        
    def __len__(self):
        return len(self.data) - self.lookback_len - self.pred_length + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.lookback_len]
        y = self.data[idx+self.lookback_len:idx+self.lookback_len+self.pred_length, 0]  # Only predict velocity
        return torch.FloatTensor(x), torch.FloatTensor(y)

# Function to collect data from SUMO simulation
def collect_simulation_data(sumocfg_file, ego_vehicle_id, num_steps, step_length=1.0):
    """
    Collect data from SUMO simulation for the ego vehicle and surrounding vehicles.
    
    Args:
        sumocfg_file: Path to SUMO configuration file
        ego_vehicle_id: ID of the ego vehicle
        num_steps: Number of simulation steps to run
        step_length: Length of each simulation step in seconds
        
    Returns:
        DataFrame with collected data
    """
    print(f"Starting data collection with config: {sumocfg_file}")
    print(f"Looking for ego vehicle with ID: {ego_vehicle_id}")
    print(f"Will run for {num_steps} steps with step length {step_length}s")
    
    # Check if config file exists
    if not os.path.exists(sumocfg_file):
        print(f"ERROR: Config file {sumocfg_file} not found!")
        return pd.DataFrame()
    
    # Start SUMO
    try:
        print("Checking for SUMO binary...")
        sumo_binary = checkBinary('sumo')
        print(f"Found SUMO binary at: {sumo_binary}")
        
        print(f"Starting SUMO with command: {sumo_binary} -c {sumocfg_file} --step-length {step_length}")
        traci.start([sumo_binary, "-c", sumocfg_file, "--step-length", str(step_length)])
        print("SUMO started successfully")
    except Exception as e:
        print(f"ERROR starting SUMO: {e}")
        return pd.DataFrame()
    
    # Data collection
    data = []
    ego_found = False
    
    print(f"Starting simulation loop for {num_steps} steps...")
    for step in range(num_steps):
        if step % 100 == 0:
            print(f"Simulation step {step}/{num_steps}")
        
        try:
            traci.simulationStep()
        except Exception as e:
            print(f"ERROR during simulation step {step}: {e}")
            break
        
        # Get all vehicles in the simulation
        try:
            vehicles = traci.vehicle.getIDList()
            if step % 100 == 0:
                print(f"Vehicles in simulation at step {step}: {vehicles}")
        except Exception as e:
            print(f"ERROR getting vehicle list at step {step}: {e}")
            continue
        
        # Skip if ego vehicle is not in the simulation yet
        if ego_vehicle_id not in vehicles:
            if step % 100 == 0:
                print(f"Ego vehicle {ego_vehicle_id} not found at step {step}")
            continue
        
        if not ego_found:
            print(f"Ego vehicle {ego_vehicle_id} found at step {step}")
            ego_found = True
        
        # Get ego vehicle data
        try:
            ego_speed = traci.vehicle.getSpeed(ego_vehicle_id)
            ego_accel = traci.vehicle.getAcceleration(ego_vehicle_id)
            ego_pos = traci.vehicle.getPosition(ego_vehicle_id)
            ego_lane = traci.vehicle.getLaneID(ego_vehicle_id)
            ego_lane_pos = traci.vehicle.getLanePosition(ego_vehicle_id)
            
            if step % 100 == 0:
                print(f"Ego data at step {step}: speed={ego_speed}, accel={ego_accel}, pos={ego_pos}")
        except Exception as e:
            print(f"ERROR getting ego vehicle data at step {step}: {e}")
            continue
        
        # Get surrounding vehicles data
        surrounding_vehicles = []
        for veh_id in vehicles:
            if veh_id != ego_vehicle_id:
                try:
                    # Calculate distance to ego vehicle
                    veh_pos = traci.vehicle.getPosition(veh_id)
                    dx = veh_pos[0] - ego_pos[0]
                    dy = veh_pos[1] - ego_pos[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Only consider vehicles within 100m
                    if distance <= 100:
                        veh_speed = traci.vehicle.getSpeed(veh_id)
                        veh_accel = traci.vehicle.getAcceleration(veh_id)
                        veh_lane = traci.vehicle.getLaneID(veh_id)
                        
                        # Determine if vehicle is ahead or behind
                        is_ahead = 1 if traci.vehicle.getLanePosition(veh_id) > ego_lane_pos else 0
                        
                        # Calculate relative speed
                        rel_speed = veh_speed - ego_speed
                        
                        surrounding_vehicles.append({
                            'id': veh_id,
                            'distance': distance,
                            'speed': veh_speed,
                            'accel': veh_accel,
                            'lane': veh_lane,
                            'is_ahead': is_ahead,
                            'rel_speed': rel_speed
                        })
                except Exception as e:
                    print(f"ERROR processing surrounding vehicle {veh_id} at step {step}: {e}")
                    continue
        
        if step % 100 == 0:
            print(f"Found {len(surrounding_vehicles)} surrounding vehicles within 100m at step {step}")
        
        # Sort surrounding vehicles by distance
        surrounding_vehicles.sort(key=lambda x: x['distance'])
        
        # Take up to 6 closest vehicles (3 ahead, 3 behind)
        ahead_vehicles = [v for v in surrounding_vehicles if v['is_ahead'] == 1][:3]
        behind_vehicles = [v for v in surrounding_vehicles if v['is_ahead'] == 0][:3]
        
        # Pad if less than 3 vehicles ahead or behind
        while len(ahead_vehicles) < 3:
            ahead_vehicles.append({
                'id': 'none',
                'distance': 100,
                'speed': 0,
                'accel': 0,
                'lane': 'none',
                'is_ahead': 1
            })
        
        while len(behind_vehicles) < 3:
            behind_vehicles.append({
                'id': 'none',
                'distance': 100,
                'speed': 0,
                'accel': 0,
                'lane': 'none',
                'is_ahead': 0
            })
        
        # Combine all data
        step_data = {
            'time': step * step_length,
            'ego_speed': ego_speed,
            'ego_accel': ego_accel,
            'ego_lane': ego_lane,
            'ego_lane_pos': ego_lane_pos
        }
        
        # Add surrounding vehicles data
        for i, veh in enumerate(ahead_vehicles + behind_vehicles):
            step_data[f'veh{i+1}_distance'] = veh['distance']
            step_data[f'veh{i+1}_speed'] = veh['speed']
            step_data[f'veh{i+1}_accel'] = veh['accel']
            step_data[f'veh{i+1}_is_ahead'] = veh['is_ahead']
            step_data[f'veh{i+1}_rel_speed'] = veh.get('rel_speed', 0)
        
        data.append(step_data)
    
    # Close SUMO
    try:
        traci.close()
        print("SUMO simulation closed successfully")
    except Exception as e:
        print(f"ERROR closing SUMO: {e}")
    
    # Create DataFrame
    print(f"Collected {len(data)} data points")
    if len(data) == 0:
        print("WARNING: No data collected!")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    return df

# Function to preprocess data for the model
def preprocess_data(df):
    """
    Preprocess the collected data for the model.
    
    Args:
        df: DataFrame with collected data
        
    Returns:
        Preprocessed data as numpy array
    """
    # Select features
    features = ['ego_speed', 'ego_accel', 'ego_lane_pos']
    
    # If ego_lane is categorical, convert to numeric (hash or label encoding)
    if 'ego_lane' in df.columns:
        if df['ego_lane'].dtype == object:
            df['ego_lane'] = df['ego_lane'].astype('category').cat.codes
        features.append('ego_lane')
    
    # Add surrounding vehicles features
    for i in range(1, 7):  # 6 surrounding vehicles
        features.extend([
            f'veh{i}_distance', f'veh{i}_speed', f'veh{i}_accel', f'veh{i}_is_ahead', f'veh{i}_rel_speed'])
    
    # Extract features
    X = df[features].values
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

# Function to train the model
def train_model(X, lookback_len=60, pred_length=20, epochs=50, batch_size=32):
    """
    Train the iTransformer model.
    
    Args:
        X: Preprocessed data
        lookback_len: Number of time steps to look back
        pred_length: Number of time steps to predict
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model
    """
    # Create dataset
    dataset = TimeSeriesDataset(X, lookback_len, pred_length)
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = iTransformer(
        num_variates=X.shape[1],
        lookback_len=lookback_len,
        dim=128,
        depth=4,
        heads=8,
        dim_head=32,
        pred_length=pred_length
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
    
    return model

def evaluate_model(model, test_loader, criterion):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained iTransformer model
        test_loader: DataLoader with test data
        criterion: Loss function
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            predictions.extend(outputs.numpy())
            actuals.extend(batch_y.numpy())
    
    # Calculate metrics
    mse = total_loss / len(test_loader)
    rmse = np.sqrt(mse)
    
    # Calculate accuracy (within 5% error margin)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    accuracy = np.mean(np.abs(predictions - actuals) / actuals < 0.05) * 100
    
    # Calculate inference time
    start_time = time.time()
    for batch_x, _ in test_loader:
        _ = model(batch_x)
    inference_time = (time.time() - start_time) / len(test_loader) * 1000  # ms per batch
    
    return {
        'mse': mse,
        'rmse': rmse,
        'accuracy': accuracy,
        'inference_time_ms': inference_time
    }

# Function to predict future velocity
def predict_velocity(model, X, scaler, lookback_len=60, pred_length=20):
    """
    Predict future velocity using the trained model.
    
    Args:
        model: Trained iTransformer model
        X: Preprocessed data
        scaler: Scaler used for preprocessing
        lookback_len: Number of time steps to look back
        pred_length: Number of time steps to predict
        
    Returns:
        Predicted velocity values
    """
    # Get the last lookback_len time steps
    last_sequence = X[-lookback_len:]
    
    # Convert to tensor
    last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(last_sequence)
    
    # Convert prediction to numpy
    prediction = prediction.numpy()[0]
    
    # Inverse transform to get actual velocity values
    # Create a dummy array with the same shape as the original data
    dummy = np.zeros((pred_length, X.shape[1]))
    # Put the predicted velocity in the first column (ego_speed)
    dummy[:, 0] = prediction
    # Inverse transform
    prediction_inv = scaler.inverse_transform(dummy)[:, 0]
    
    return prediction_inv

def modify_routes_file(input_file, output_file, flow_factor):
    """
    Modify the routes file to change traffic density.
    
    Args:
        input_file: Original routes file
        output_file: Modified routes file
        flow_factor: Factor to multiply flow rates (vehsPerHour)
    """
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Modify flow rates
    for flow in root.findall(".//flow"):
        current_rate = float(flow.get("vehsPerHour"))
        new_rate = max(1, int(current_rate * flow_factor))
        flow.set("vehsPerHour", str(new_rate))
    
    # Write modified file
    tree.write(output_file)

def run_simulation_with_traffic_density(density_factor, simulation_id):
    """
    Run a simulation with a specific traffic density.
    
    Args:
        density_factor: Factor to multiply flow rates
        simulation_id: Identifier for this simulation
        
    Returns:
        DataFrame with collected data
    """
    # Create modified routes file
    original_routes = "routes.rou.xml"
    modified_routes = f"routes_{simulation_id}.rou.xml"
    modify_routes_file(original_routes, modified_routes, density_factor)
    
    # Create modified config file
    original_config = "simulation.sumocfg"
    modified_config = f"simulation_{simulation_id}.sumocfg"
    
    tree = ET.parse(original_config)
    root = tree.getroot()
    
    # Update route file reference
    for route_files in root.findall(".//route-files"):
        route_files.set("value", modified_routes)
    
    # Write modified config file
    tree.write(modified_config)
    
    # Run simulation
    print(f"Running simulation {simulation_id} with density factor {density_factor}...")
    df = collect_simulation_data(modified_config, "ego", 1000)
    
    # Add metadata
    df['simulation_id'] = simulation_id
    df['density_factor'] = density_factor
    
    return df

def main():
    # Configuration
    sumocfg_file = "simulation.sumocfg"  # Path to our SUMO config file
    
    try:
        # Run simulation and collect data
        print("Running simulation...")
        df = run_simulation_with_traffic_density(1.0, "original")
        
        # Preprocess data
        print("Preprocessing data...")
        X, scaler = preprocess_data(df)
        
        # Create dataset
        lookback_len = 60  # 60 seconds of historical data
        pred_length = 20   # Predict 20 seconds into the future
        dataset = TimeSeriesDataset(X, lookback_len, pred_length)
        
        # Train model
        print("Training model...")
        model = train_model(X, lookback_len, pred_length)
        
        # Create test dataset and loader
        test_size = int(0.2 * len(dataset))
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Evaluate model
        criterion = nn.MSELoss()
        metrics = evaluate_model(model, test_loader, criterion)
        
        print("\nModel Evaluation Metrics:")
        print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        print(f"Accuracy (within 5% error): {metrics['accuracy']:.2f}%")
        print(f"Average Inference Time: {metrics['inference_time_ms']:.2f} ms per batch")
        
    except Exception as e:
        print(f"ERROR training model: {e}")
        return
    
    # Save model
    try:
        print("\nSaving model...")
        torch.save(model.state_dict(), "velocity_prediction_model.pth")
        print(f"Saved model to velocity_prediction_model.pth")
    except Exception as e:
        print(f"ERROR saving model: {e}")

if __name__ == "__main__":
    main() 