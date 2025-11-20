#!/usr/bin/env python3
"""
NGSIM Data Preprocessing Script
================================
Converts NGSIM trajectory data to your 26-feature format for iTransformer training.

Author: Parth Shinde
Date: November 2025

This script:
1. Loads NGSIM US-101/I-80 trajectory data
2. Extracts 26 features (ego + 13 surrounding vehicles)
3. Creates 60-second lookback sequences
4. Prepares train/validation/test splits
5. Saves processed data for model training
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class NGSIMPreprocessor:
    """Preprocesses NGSIM trajectory data for velocity prediction."""

    def __init__(self, ngsim_folder: str):
        """
        Initialize preprocessor.

        Args:
            ngsim_folder: Path to NGSIM_dataset folder containing trajectory files
        """
        self.ngsim_folder = Path(ngsim_folder)
        self.data = None
        self.processed_trajectories = []
        self.frame_cache = {}  # Cache for frame lookups

        # Feature names (matching your SUMO setup)
        self.feature_names = [
            'ego_velocity',
            'front_velocity', 'front_distance',
            'front_left_velocity', 'front_left_distance',
            'front_right_velocity', 'front_right_distance',
            'front_front_velocity', 'front_front_distance',
            'rear_velocity', 'rear_distance',
            'rear_left_velocity', 'rear_left_distance',
            'rear_right_velocity', 'rear_right_distance',
            'rear_rear_velocity', 'rear_rear_distance',
            'left_velocity', 'left_distance',
            'left_front_velocity', 'left_front_distance',
            'right_velocity', 'right_distance',
            'right_front_velocity', 'right_front_distance',
            'current_lane'
        ]

        # NGSIM column names (from data dictionary)
        self.ngsim_columns = [
            'Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time',
            'Local_X', 'Local_Y', 'Global_X', 'Global_Y',
            'v_Length', 'v_Width', 'v_Class',
            'v_Vel', 'v_Acc', 'Lane_ID',
            'Preceding', 'Following', 'Space_Headway', 'Time_Headway'
        ]

        # Conversion factors
        self.FT_TO_M = 0.3048  # feet to meters
        self.FT_SEC_TO_M_SEC = 0.3048  # feet/sec to m/sec

        print(f"[INFO] NGSIM Preprocessor initialized")
        print(f"   Looking for data in: {self.ngsim_folder}")

    def load_trajectory_file(self, filename: str) -> pd.DataFrame:
        """
        Load NGSIM trajectory file.

        Args:
            filename: Name of trajectory file (e.g., 'trajectories-0750-0805.txt')

        Returns:
            DataFrame with trajectory data
        """
        filepath = self.ngsim_folder / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"\n[LOAD] Loading {filename}...")

        # Read CSV (space or tab delimited)
        df = pd.read_csv(filepath, delim_whitespace=True, header=None,
                         names=self.ngsim_columns)

        # Convert units to metric
        df['v_Vel'] = df['v_Vel'] * self.FT_SEC_TO_M_SEC  # ft/s → m/s
        df['Local_X'] = df['Local_X'] * self.FT_TO_M  # ft → m
        df['Local_Y'] = df['Local_Y'] * self.FT_TO_M  # ft → m
        df['Space_Headway'] = df['Space_Headway'] * self.FT_TO_M  # ft → m

        print(f"   [OK] Loaded {len(df)} data points")
        print(f"   [OK] {df['Vehicle_ID'].nunique()} unique vehicles")
        print(f"   [OK] Time range: {df['Frame_ID'].min()/10:.1f}s to {df['Frame_ID'].max()/10:.1f}s")

        return df

    def get_surrounding_vehicles(self, df: pd.DataFrame, vehicle_id: int, 
                                frame_id: int, ego_lane: int) -> Dict:
        """
        Extract surrounding vehicles at given timestep.

        Args:
            df: Full trajectory DataFrame
            vehicle_id: Ego vehicle ID
            frame_id: Current frame
            ego_lane: Ego vehicle's lane

        Returns:
            Dictionary with surrounding vehicle info
        """
        # Use cached frame data if available
        if frame_id in self.frame_cache:
            frame_df = self.frame_cache[frame_id]
        else:
            # Get all vehicles in current frame (optimized with index)
            frame_df = df[df['Frame_ID'] == frame_id].copy()
            self.frame_cache[frame_id] = frame_df

        # Get ego vehicle
        ego = frame_df[frame_df['Vehicle_ID'] == vehicle_id]
        if len(ego) == 0:
            return None

        ego_y = ego['Local_Y'].values[0]
        ego_x = ego['Local_X'].values[0]

        # Initialize surrounding vehicles dictionary
        surrounding = {
            'front': None, 'front_left': None, 'front_right': None,
            'front_front': None, 'rear': None, 'rear_left': None,
            'rear_right': None, 'rear_rear': None, 'left': None,
            'left_front': None, 'right': None, 'right_front': None
        }

        # Helper function to find closest vehicle
        def find_closest(lane: int, direction: str) -> Optional[pd.Series]:
            lane_vehicles = frame_df[frame_df['Lane_ID'] == lane]

            if direction == 'front':
                candidates = lane_vehicles[lane_vehicles['Local_Y'] > ego_y]
                if len(candidates) > 0:
                    return candidates.loc[candidates['Local_Y'].idxmin()]
            else:  # rear
                candidates = lane_vehicles[lane_vehicles['Local_Y'] < ego_y]
                if len(candidates) > 0:
                    return candidates.loc[candidates['Local_Y'].idxmax()]

            return None

        # Same lane (front, rear)
        front = find_closest(ego_lane, 'front')
        if front is not None:
            surrounding['front'] = front
            # Front-front vehicle
            front_y = front['Local_Y']
            front_front_candidates = frame_df[
                (frame_df['Lane_ID'] == ego_lane) & 
                (frame_df['Local_Y'] > front_y)
            ]
            if len(front_front_candidates) > 0:
                surrounding['front_front'] = front_front_candidates.loc[
                    front_front_candidates['Local_Y'].idxmin()
                ]

        rear = find_closest(ego_lane, 'rear')
        if rear is not None:
            surrounding['rear'] = rear
            # Rear-rear vehicle
            rear_y = rear['Local_Y']
            rear_rear_candidates = frame_df[
                (frame_df['Lane_ID'] == ego_lane) & 
                (frame_df['Local_Y'] < rear_y)
            ]
            if len(rear_rear_candidates) > 0:
                surrounding['rear_rear'] = rear_rear_candidates.loc[
                    rear_rear_candidates['Local_Y'].idxmax()
                ]

        # Left lane
        if ego_lane > 1:  # Has left lane
            left_lane = ego_lane - 1
            left = find_closest(left_lane, 'front')  # Closest in left lane
            if left is not None:
                surrounding['left'] = left if abs(left['Local_Y'] - ego_y) < 50 else None

            left_front = find_closest(left_lane, 'front')
            if left_front is not None and left_front['Local_Y'] > ego_y:
                surrounding['left_front'] = left_front

            left_rear = find_closest(left_lane, 'rear')
            if left_rear is not None and left_rear['Local_Y'] < ego_y:
                surrounding['rear_left'] = left_rear

        # Right lane
        if ego_lane < 5:  # Has right lane (NGSIM US-101 has lanes 1-5)
            right_lane = ego_lane + 1
            right = find_closest(right_lane, 'front')
            if right is not None:
                surrounding['right'] = right if abs(right['Local_Y'] - ego_y) < 50 else None

            right_front = find_closest(right_lane, 'front')
            if right_front is not None and right_front['Local_Y'] > ego_y:
                surrounding['right_front'] = right_front

            right_rear = find_closest(right_lane, 'rear')
            if right_rear is not None and right_rear['Local_Y'] < ego_y:
                surrounding['rear_right'] = right_rear

        return surrounding

    def extract_features(self, df: pd.DataFrame, vehicle_id: int, 
                        frame_id: int) -> Optional[np.ndarray]:
        """
        Extract 26 features for a single timestep.

        Args:
            df: Full trajectory DataFrame
            vehicle_id: Ego vehicle ID
            frame_id: Current frame

        Returns:
            26-element feature vector or None if invalid
        """
        # Get ego vehicle
        ego = df[(df['Vehicle_ID'] == vehicle_id) & (df['Frame_ID'] == frame_id)]
        if len(ego) == 0:
            return None

        ego = ego.iloc[0]
        ego_lane = ego['Lane_ID']
        ego_y = ego['Local_Y']

        # Initialize features with default values (0 or -1 for missing)
        features = np.zeros(26)

        # Feature 0: ego velocity
        features[0] = ego['v_Vel']

        # Get surrounding vehicles
        surrounding = self.get_surrounding_vehicles(df, vehicle_id, frame_id, ego_lane)

        if surrounding is None:
            return None

        # Helper to extract velocity and distance
        def extract_veh_info(vehicle: Optional[pd.Series]) -> Tuple[float, float]:
            if vehicle is None:
                return (0.0, 100.0)  # Default: no vehicle, far away
            else:
                vel = vehicle['v_Vel']
                dist = abs(vehicle['Local_Y'] - ego_y)
                return (vel, dist)

        # Extract features for all surrounding vehicles
        positions = ['front', 'front_left', 'front_right', 'front_front',
                    'rear', 'rear_left', 'rear_right', 'rear_rear',
                    'left', 'left_front', 'right', 'right_front']

        feature_idx = 1
        for pos in positions:
            vel, dist = extract_veh_info(surrounding[pos])
            features[feature_idx] = vel
            features[feature_idx + 1] = dist
            feature_idx += 2

        # Feature 25: current lane (normalized)
        features[25] = ego_lane / 5.0  # Normalize to [0, 1]

        return features

    def create_sequences(self, df: pd.DataFrame, 
                        lookback_steps: int = 300,
                        prediction_steps: int = 20,
                        max_vehicles: int = 1000) -> List[Dict]:
        """
        Create input-output sequences for model training.

        Args:
            df: Trajectory DataFrame
            lookback_steps: Number of past timesteps (60s = 600 steps at 0.1s)
            prediction_steps: Number of future timesteps to predict (20s = 20 steps at 1s)
            max_vehicles: Maximum number of vehicles to process (for subset testing)

        Returns:
            List of sequences, each containing input and output
        """
        sequences = []
        vehicle_ids = df['Vehicle_ID'].unique()
        
        # Limit to subset for faster processing
        if max_vehicles and len(vehicle_ids) > max_vehicles:
            print(f"\n[WARN] Limiting to first {max_vehicles} vehicles (out of {len(vehicle_ids)}) for faster processing")
            vehicle_ids = vehicle_ids[:max_vehicles]
        
        # Pre-index dataframe by Frame_ID for faster lookups
        print(f"\n[INDEX] Indexing dataframe by Frame_ID...")
        df_indexed = df.set_index('Frame_ID', drop=False)
        print(f"   [OK] Indexed {len(df_indexed)} rows")

        print(f"\n[SEQ] Creating sequences...")
        print(f"   Lookback: {lookback_steps} steps ({lookback_steps/10}s)"
              f" | Prediction: {prediction_steps} steps ({prediction_steps}s)")
        print(f"   Processing {len(vehicle_ids)} vehicles...")

        # Pre-cache all unique frames
        unique_frames = df['Frame_ID'].unique()
        print(f"   Pre-caching {len(unique_frames)} frames...")
        for frame_id in unique_frames:
            self.frame_cache[frame_id] = df[df['Frame_ID'] == frame_id].copy()
        print(f"   [OK] Frame cache ready")

        for i, vehicle_id in enumerate(vehicle_ids):
            if (i + 1) % 100 == 0:
                print(f"   Processing vehicle {i+1}/{len(vehicle_ids)}...", end='\r')

            # Get vehicle's trajectory
            vehicle_df = df[df['Vehicle_ID'] == vehicle_id].sort_values('Frame_ID')

            # Need at least lookback frames + prediction frames (prediction sampled every 10 frames = 1s intervals)
            min_frames_needed = lookback_steps + (prediction_steps * 10)
            if len(vehicle_df) < min_frames_needed:
                if i < 10:  # Debug first 10
                    print(f"   Vehicle {vehicle_id}: {len(vehicle_df)} frames (need {min_frames_needed}) - SKIPPED")
                continue  # Skip short trajectories
            
            # Debug: print first few vehicles' frame counts
            if i < 5:
                print(f"   Vehicle {vehicle_id}: {len(vehicle_df)} frames (need {min_frames_needed}) - PROCESSING")

            frames = vehicle_df['Frame_ID'].values

            # Slide window through trajectory (larger step for faster processing)
            step_size = 200  # Increased step size for faster processing
            for start_idx in range(0, len(vehicle_df) - min_frames_needed, step_size):
                # Extract lookback features (every 0.1s = every frame)
                lookback_features = []
                valid_sequence = True
                
                for j in range(start_idx, start_idx + lookback_steps):
                    frame_id = frames[j]
                    features = self.extract_features(df, vehicle_id, frame_id)
                    if features is None:
                        valid_sequence = False
                        if i < 3 and start_idx == 0:  # Debug first sequence of first 3 vehicles
                            print(f"      [DEBUG] Failed to extract features at frame {frame_id}")
                        break
                    lookback_features.append(features)

                if not valid_sequence or len(lookback_features) != lookback_steps:
                    continue  # Skip incomplete sequences

                # Extract prediction targets (every 1s = every 10 frames)
                prediction_velocities = []
                for k in range(prediction_steps):
                    pred_idx = start_idx + lookback_steps + (k * 10)  # Sample every 10 frames
                    if pred_idx >= len(vehicle_df):
                        valid_sequence = False
                        break
                    pred_frame = frames[pred_idx]
                    pred_ego = vehicle_df[vehicle_df['Frame_ID'] == pred_frame]
                    if len(pred_ego) > 0:
                        prediction_velocities.append(pred_ego.iloc[0]['v_Vel'])
                    else:
                        valid_sequence = False
                        break

                if not valid_sequence or len(prediction_velocities) != prediction_steps:
                    continue  # Skip incomplete predictions

                # Store sequence
                sequences.append({
                    'vehicle_id': vehicle_id,
                    'start_frame': frames[start_idx],
                    'input': np.array(lookback_features),  # Shape: (600, 26)
                    'output': np.array(prediction_velocities)  # Shape: (20,)
                })

        print(f"\n   [OK] Created {len(sequences)} valid sequences")
        return sequences

    def split_data(self, sequences: List[Dict], 
                   train_ratio: float = 0.1,
                   val_ratio: float = 0.0,
                   test_ratio: float = 0.9) -> Tuple[List, List, List]:
        """
        Split sequences into train/val/test sets.

        Args:
            sequences: List of sequence dictionaries
            train_ratio: Fraction for training (default 0.1 for fine-tuning)
            val_ratio: Fraction for validation (default 0.0)
            test_ratio: Fraction for testing (default 0.9)

        Returns:
            train_sequences, val_sequences, test_sequences
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        np.random.shuffle(sequences)

        n_total = len(sequences)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_seq = sequences[:n_train]
        val_seq = sequences[n_train:n_train + n_val]
        test_seq = sequences[n_train + n_val:]

        print(f"\n[SPLIT] Data split:")
        print(f"   Train: {len(train_seq)} ({len(train_seq)/n_total*100:.1f}%)")
        print(f"   Val: {len(val_seq)} ({len(val_seq)/n_total*100:.1f}%)")
        print(f"   Test: {len(test_seq)} ({len(test_seq)/n_total*100:.1f}%)")

        return train_seq, val_seq, test_seq

    def save_processed_data(self, train, val, test, output_dir: str):
        """
        Save processed sequences to disk.

        Args:
            train, val, test: Sequence lists
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n[SAVE] Saving processed data to {output_path}...")

        # Save as pickle
        with open(output_path / 'ngsim_train.pkl', 'wb') as f:
            pickle.dump(train, f)
        print(f"   [OK] Saved train data: {len(train)} sequences")

        with open(output_path / 'ngsim_val.pkl', 'wb') as f:
            pickle.dump(val, f)
        print(f"   [OK] Saved validation data: {len(val)} sequences")

        with open(output_path / 'ngsim_test.pkl', 'wb') as f:
            pickle.dump(test, f)
        print(f"   [OK] Saved test data: {len(test)} sequences")

        # Save metadata
        metadata = {
            'n_features': 26,
            'lookback_steps': self.lookback_steps if hasattr(self, 'lookback_steps') else 300,
            'prediction_steps': self.prediction_steps if hasattr(self, 'prediction_steps') else 20,
            'feature_names': self.feature_names,
            'n_train': len(train),
            'n_val': len(val),
            'n_test': len(test)
        }

        with open(output_path / 'ngsim_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print(f"   [OK] Saved metadata")

        print(f"\n[OK] Preprocessing complete!")

def main():
    """Main preprocessing pipeline."""

    print("="*70)
    print("NGSIM DATA PREPROCESSING")
    print("="*70)

    # Configuration
    NGSIM_FOLDER = './NGSIM_dataset'  # Adjust to your path
    OUTPUT_DIR = './ngsim_processed'
    TRAJECTORY_FILES = [
        'trajectories-0750am-0805am.txt',  # Adjust filenames based on your data
        # Add more files if needed:
        # 'trajectories-0805-0820.txt',
        # 'trajectories-0820-0835.txt',
    ]
    
    # Adjust for NGSIM data (vehicles have ~400-500 frames, so reduce requirements)
    LOOKBACK_STEPS = 300  # 30 seconds instead of 60 (300 frames at 0.1s intervals)
    PREDICTION_STEPS = 20  # Keep 20 seconds

    # Initialize preprocessor
    preprocessor = NGSIMPreprocessor(NGSIM_FOLDER)

    # Load and process each file
    all_sequences = []
    MAX_VEHICLES = 1000  # Process subset for faster testing
    
    for filename in TRAJECTORY_FILES:
        try:
            # Load trajectory file
            df = preprocessor.load_trajectory_file(filename)

            # Create sequences (with subset limit and adjusted steps)
            sequences = preprocessor.create_sequences(df, 
                                                     lookback_steps=LOOKBACK_STEPS,
                                                     prediction_steps=PREDICTION_STEPS,
                                                     max_vehicles=MAX_VEHICLES)
            all_sequences.extend(sequences)

        except FileNotFoundError as e:
            print(f"   [WARN] {e}")
            continue
        except Exception as e:
            print(f"   [ERROR] Error processing {filename}: {e}")
            continue

    if len(all_sequences) == 0:
        print("\n[ERROR] No sequences created! Check your data files.")
        return

    # Split data
    train, val, test = preprocessor.split_data(
        all_sequences,
        train_ratio=0.1,   # 10% for fine-tuning
        val_ratio=0.0,     # 0% for validation (not needed for testing only)
        test_ratio=0.9     # 90% for testing
    )

    # Save processed data
    preprocessor.save_processed_data(train, val, test, OUTPUT_DIR)

    print("\n" + "="*70)
    print("[SUCCESS] PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\n[OUTPUT] Output files saved to: {OUTPUT_DIR}/")
    print("   - ngsim_train.pkl")
    print("   - ngsim_val.pkl")
    print("   - ngsim_test.pkl")
    print("   - ngsim_metadata.pkl")
    print("\n[NEXT] Next step: Run ngsim_evaluation.py to test your model!")

if __name__ == '__main__':
    main()
