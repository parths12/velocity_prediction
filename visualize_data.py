#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def visualize_single_simulation(data_file):
    """
    Visualize data from a single simulation.
    
    Args:
        data_file: Path to the CSV file with simulation data
    """
    # Load data
    df = pd.read_csv(data_file)
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Ego vehicle speed over time
    axs[0].plot(df['time'], df['ego_speed'], 'b-', linewidth=2)
    axs[0].set_title('Ego Vehicle Speed Over Time')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Speed (m/s)')
    axs[0].grid(True)
    
    # Plot 2: Ego vehicle acceleration over time
    axs[1].plot(df['time'], df['ego_accel'], 'r-', linewidth=2)
    axs[1].set_title('Ego Vehicle Acceleration Over Time')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Acceleration (m/s²)')
    axs[1].grid(True)
    
    # Plot 3: Distance to surrounding vehicles over time
    # Get all columns with 'distance' in the name
    distance_cols = [col for col in df.columns if 'distance' in col]
    
    for col in distance_cols:
        # Extract vehicle number from column name
        veh_num = col.split('_')[0]
        # Get corresponding 'is_ahead' column
        is_ahead_col = f"{veh_num}_is_ahead"
        
        if is_ahead_col in df.columns:
            # Check if this vehicle is ahead or behind
            is_ahead = df[is_ahead_col].iloc[0]
            label = f"{veh_num} ({'ahead' if is_ahead else 'behind'})"
            axs[2].plot(df['time'], df[col], label=label)
    
    axs[2].set_title('Distance to Surrounding Vehicles Over Time')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Distance (m)')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.splitext(data_file)[0] + '_visualization.png'
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")
    
    plt.show()

def visualize_multiple_simulations(data_files):
    """
    Compare data from multiple simulations.
    
    Args:
        data_files: List of paths to CSV files with simulation data
    """
    # Load all data
    dfs = []
    for file in data_files:
        df = pd.read_csv(file)
        # Extract simulation name from file name
        sim_name = os.path.splitext(os.path.basename(file))[0].replace('simulation_data_', '')
        df['simulation'] = sim_name
        dfs.append(df)
    
    # Combine all data
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Ego vehicle speed comparison
    sns.lineplot(data=combined_df, x='time', y='ego_speed', hue='simulation', ax=axs[0])
    axs[0].set_title('Ego Vehicle Speed Comparison')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Speed (m/s)')
    axs[0].grid(True)
    
    # Plot 2: Ego vehicle acceleration comparison
    sns.lineplot(data=combined_df, x='time', y='ego_accel', hue='simulation', ax=axs[1])
    axs[1].set_title('Ego Vehicle Acceleration Comparison')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Acceleration (m/s²)')
    axs[1].grid(True)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('simulation_comparison.png')
    print("Saved comparison visualization to simulation_comparison.png")
    
    plt.show()

def visualize_traffic_density_impact():
    """
    Visualize the impact of traffic density on ego vehicle speed.
    """
    # Check if combined data exists
    if not os.path.exists('combined_simulation_data.csv'):
        print("Combined simulation data not found. Please run run_multiple_simulations.py first.")
        return
    
    # Load combined data
    df = pd.read_csv('combined_simulation_data.csv')
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Average speed by traffic density
    avg_speed = df.groupby('traffic_density')['ego_speed'].mean().reset_index()
    # Sort by density factor
    density_order = ['very_light', 'light', 'moderate', 'heavy', 'very_heavy']
    avg_speed['traffic_density'] = pd.Categorical(avg_speed['traffic_density'], categories=density_order, ordered=True)
    avg_speed = avg_speed.sort_values('traffic_density')
    
    sns.barplot(data=avg_speed, x='traffic_density', y='ego_speed', ax=axs[0])
    axs[0].set_title('Average Ego Vehicle Speed by Traffic Density')
    axs[0].set_xlabel('Traffic Density')
    axs[0].set_ylabel('Average Speed (m/s)')
    axs[0].grid(True)
    
    # Plot 2: Speed distribution by traffic density
    sns.boxplot(data=df, x='traffic_density', y='ego_speed', order=density_order, ax=axs[1])
    axs[1].set_title('Ego Vehicle Speed Distribution by Traffic Density')
    axs[1].set_xlabel('Traffic Density')
    axs[1].set_ylabel('Speed (m/s)')
    axs[1].grid(True)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('traffic_density_impact.png')
    print("Saved traffic density impact visualization to traffic_density_impact.png")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize simulation data')
    parser.add_argument('--file', type=str, help='Path to a single simulation data file')
    parser.add_argument('--files', nargs='+', help='Paths to multiple simulation data files for comparison')
    parser.add_argument('--density-impact', action='store_true', help='Visualize impact of traffic density')
    
    args = parser.parse_args()
    
    if args.file:
        visualize_single_simulation(args.file)
    elif args.files:
        visualize_multiple_simulations(args.files)
    elif args.density_impact:
        visualize_traffic_density_impact()
    else:
        # Default: look for simulation_data.csv
        if os.path.exists('simulation_data.csv'):
            visualize_single_simulation('simulation_data.csv')
        else:
            print("No simulation data file specified and simulation_data.csv not found.")
            print("Please specify a file with --file or run velocity_prediction.py first.")

if __name__ == "__main__":
    main() 