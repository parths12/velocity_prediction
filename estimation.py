#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import torch
import time
from collections import deque
import threading
import queue

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from velocity_prediction import preprocess_data, iTransformer, predict_velocity

class RealTimeEstimator:
    def __init__(self, model_path="combined_velocity_prediction_model.pth", lookback_len=60, pred_length=20):
        """
        Initialize the real-time estimator.
        
        Args:
            model_path: Path to the trained model
            lookback_len: Number of historical time steps to use for prediction
            pred_length: Number of future time steps to predict
        """
        self.lookback_len = lookback_len
        self.pred_length = pred_length
        self.history_buffer = deque(maxlen=lookback_len)
        self.prediction_queue = queue.Queue()
        self.running = False
        
        # Load the model
        try:
            # Initialize model with correct parameters
            num_variates = 26  # Number of features
            dim = 128  # Embedding dimension (changed from 64 to 128)
            depth = 4  # Number of transformer layers
            heads = 8  # Number of attention heads
            dim_head = 32  # Dimension of each attention head
            
            self.model = iTransformer(
                num_variates=num_variates,
                lookback_len=lookback_len,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                pred_length=pred_length
            )
            
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Initialize scaler
        self.scaler = None
    
    def update_measurements(self, ego_speed, ego_accel, surrounding_vehicles):
        """
        Update the measurement history with new data.
        
        Args:
            ego_speed: Current speed of ego vehicle
            ego_accel: Current acceleration of ego vehicle
            surrounding_vehicles: List of dictionaries containing data about surrounding vehicles
        """
        # Create a data point with features in the same order as velocity_prediction.py
        data_point = {
            'ego_speed': ego_speed,
            'ego_accel': ego_accel
        }
        
        # Add surrounding vehicles data in the same order as velocity_prediction.py
        for i, veh in enumerate(surrounding_vehicles[:6]):  # Take up to 6 closest vehicles
            data_point[f'veh{i+1}_distance'] = veh.get('distance', 100.0)
            data_point[f'veh{i+1}_speed'] = veh.get('speed', 0.0)
            data_point[f'veh{i+1}_accel'] = veh.get('accel', 0.0)
            data_point[f'veh{i+1}_is_ahead'] = veh.get('is_ahead', 0)
        
        # Pad if less than 6 vehicles
        for i in range(len(surrounding_vehicles), 6):
            data_point[f'veh{i+1}_distance'] = 100.0
            data_point[f'veh{i+1}_speed'] = 0.0
            data_point[f'veh{i+1}_accel'] = 0.0
            data_point[f'veh{i+1}_is_ahead'] = 0
        
        # Add to history buffer
        self.history_buffer.append(data_point)
    
    def get_prediction(self):
        """
        Get the latest prediction from the queue.
        
        Returns:
            Latest velocity prediction or None if no prediction available
        """
        try:
            return self.prediction_queue.get_nowait()
        except queue.Empty:
            return None
    
    def predict_thread(self):
        """
        Thread function that continuously makes predictions based on the history buffer.
        """
        while self.running:
            if len(self.history_buffer) >= self.lookback_len:
                try:
                    # Convert history buffer to DataFrame
                    df = pd.DataFrame(list(self.history_buffer))
                    
                    # Preprocess data using the same function as velocity_prediction.py
                    if self.scaler is None:
                        X, self.scaler = preprocess_data(df)
                    else:
                        X = self.scaler.transform(df.values)
                    
                    # Make prediction using the same function as velocity_prediction.py
                    predicted_velocity = predict_velocity(self.model, X, self.scaler, self.lookback_len, self.pred_length)
                    
                    # Put prediction in queue
                    self.prediction_queue.put(predicted_velocity)
                except Exception as e:
                    print(f"Error making prediction: {e}")
            
            # Sleep to avoid consuming too much CPU
            time.sleep(0.1)
    
    def start(self):
        """Start the prediction thread."""
        self.running = True
        self.prediction_thread = threading.Thread(target=self.predict_thread)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
    
    def stop(self):
        """Stop the prediction thread."""
        self.running = False
        if hasattr(self, 'prediction_thread'):
            self.prediction_thread.join()

def main():
    """
    Main function to demonstrate usage.
    """
    # Initialize SUMO connection
    if 'SUMO_HOME' not in os.environ:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    
    try:
        traci.start(["sumo-gui", "-c", "simulation.sumocfg"])
    except Exception as e:
        print(f"Error starting SUMO: {e}")
        sys.exit(1)
    
    # Initialize estimator
    estimator = RealTimeEstimator()
    estimator.start()
    
    try:
        step = 0
        while step < 1000:  # Run for 1000 steps
            traci.simulationStep()
            
            # Get ego vehicle data
            try:
                ego_id = "ego"
                if ego_id in traci.vehicle.getIDList():
                    ego_speed = traci.vehicle.getSpeed(ego_id)
                    ego_accel = traci.vehicle.getAcceleration(ego_id)
                    ego_pos = traci.vehicle.getPosition(ego_id)
                    ego_lane = traci.vehicle.getLaneID(ego_id)
                    ego_lane_pos = traci.vehicle.getLanePosition(ego_id)
                    
                    # Get surrounding vehicles
                    surrounding_vehicles = []
                    for veh_id in traci.vehicle.getIDList():
                        if veh_id != ego_id:
                            veh_pos = traci.vehicle.getPosition(veh_id)
                            dx = veh_pos[0] - ego_pos[0]
                            dy = veh_pos[1] - ego_pos[1]
                            distance = np.sqrt(dx**2 + dy**2)
                            
                            if distance <= 100:  # Only consider vehicles within 100m
                                surrounding_vehicles.append({
                                    'id': veh_id,
                                    'distance': distance,
                                    'speed': traci.vehicle.getSpeed(veh_id),
                                    'accel': traci.vehicle.getAcceleration(veh_id),
                                    'is_ahead': 1 if traci.vehicle.getLanePosition(veh_id) > ego_lane_pos else 0
                                })
                    
                    # Sort by distance
                    surrounding_vehicles.sort(key=lambda x: x['distance'])
                    
                    # Update estimator
                    estimator.update_measurements(ego_speed, ego_accel, surrounding_vehicles)
                    
                    # Get prediction
                    prediction = estimator.get_prediction()
                    if prediction is not None:
                        print(f"\nStep {step}:")
                        print(f"Current speed: {ego_speed:.2f} m/s")
                        print("Predicted speeds for next 20 seconds:")
                        for i, speed in enumerate(prediction):
                            print(f"t+{i+1}s: {speed:.2f} m/s")
            
            except Exception as e:
                print(f"Error processing step {step}: {e}")
            
            step += 1
    
    finally:
        estimator.stop()
        traci.close()

if __name__ == "__main__":
    main()
