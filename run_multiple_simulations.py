#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import random
import time

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

# Import functions from velocity_prediction.py
from velocity_prediction import collect_simulation_data, preprocess_data, train_model, predict_velocity, iTransformer

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
    # Define different traffic densities
    traffic_densities = [
        (0.5, "very_light"),  # 50% of original flow rates
        (1.0, "light"),       # Original flow rates
        (1.5, "moderate"),    # 150% of original flow rates
        (2.0, "heavy"),       # 200% of original flow rates
        (3.0, "very_heavy")   # 300% of original flow rates
    ]
    
    num_runs_per_density = 3  # Number of simulations per density
    all_data = []
    
    for density_factor, density_name in traffic_densities:
        for run in range(num_runs_per_density):
            # Use a different simulation_id for each run
            simulation_id = f"{density_name}_{run}_{int(time.time())}"
            # Optionally, set a random seed for SUMO (if supported in your config)
            # os.environ['SUMO_RAND_SEED'] = str(random.randint(0, 100000))
            df = run_simulation_with_traffic_density(density_factor, simulation_id)
            df['traffic_density'] = density_name
            df['run'] = run
            all_data.append(df)
            # Save individual simulation data
            df.to_csv(f"simulation_data_{density_name}_run{run}.csv", index=False)
            print(f"Saved data for {density_name} traffic, run {run} to simulation_data_{density_name}_run{run}.csv")
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv("combined_simulation_data.csv", index=False)
    print("Saved combined data to combined_simulation_data.csv")
    # Preprocess combined data
    X, scaler = preprocess_data(combined_df)
    # Train model on combined data
    lookback_len = 60
    pred_length = 20
    model = train_model(X, lookback_len, pred_length)
    # Save model
    torch.save(model.state_dict(), "combined_velocity_prediction_model.pth")
    print("Saved model to combined_velocity_prediction_model.pth")
    # Make predictions for each traffic density (using first run as example)
    plt.figure(figsize=(15, 10))
    for i, (_, density_name) in enumerate(traffic_densities):
        df = pd.read_csv(f"simulation_data_{density_name}_run0.csv")
        X_density, _ = preprocess_data(df)
        predicted_velocity = predict_velocity(model, X_density, scaler, lookback_len, pred_length)
        plt.subplot(len(traffic_densities), 1, i+1)
        actual_velocity = df['ego_speed'].values
        plt.plot(range(len(actual_velocity)), actual_velocity, label='Historical')
        time_pred = range(len(actual_velocity), len(actual_velocity) + pred_length)
        plt.plot(time_pred, predicted_velocity, label='Predicted', color='red')
        plt.title(f'Traffic Density: {density_name.replace("_", " ").title()}')
        plt.ylabel('Velocity (m/s)')
        if i == len(traffic_densities) - 1:
            plt.xlabel('Time (s)')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('velocity_predictions_by_density.png')
    plt.show()
    # Clean up temporary files
    for density_factor, density_name in traffic_densities:
        for run in range(num_runs_per_density):
            simulation_id = f"{density_name}_{run}_{int(time.time())}"
            try:
                os.remove(f"routes_{simulation_id}.rou.xml")
                os.remove(f"simulation_{simulation_id}.sumocfg")
            except:
                pass

if __name__ == "__main__":
    main() 
    