 # Vehicle Velocity Prediction and Estimation System

## Overview

This report describes a comprehensive system for predicting and estimating vehicle velocities in a traffic simulation environment. The system consists of two main components:

1. **Velocity Prediction System** (`velocity_prediction.py`): A machine learning-based system that predicts future vehicle velocities using historical data.
2. **Real-time Estimation System** (`estimation.py`): A real-time system that continuously estimates and predicts vehicle velocities during simulation.

## System Architecture

### 1. Velocity Prediction System

The velocity prediction system is built around an iTransformer model and consists of the following components:

#### 1.1 Data Collection
- Uses SUMO (Simulation of Urban MObility) for traffic simulation
- Collects data for:
  - Ego vehicle (speed, acceleration, position)
  - Surrounding vehicles (up to 6 closest vehicles within 100m)
  - Features include: distance, speed, acceleration, and relative position

#### 1.2 iTransformer Model
- Architecture:
  - Input: Time series data with 26 features
  - Embedding dimension: 128
  - 4 transformer layers
  - 8 attention heads
  - 32-dimensional attention heads
  - Lookback length: 60 time steps
  - Prediction length: 20 time steps

#### 1.3 Training Process
- Data preprocessing:
  - Standard scaling of features
  - Time series windowing
  - 80/20 train-validation split
- Training parameters:
  - Batch size: 32
  - Epochs: 50
  - Loss function: Mean Squared Error
  - Optimizer: Adam with learning rate 0.001

#### 1.4 Prediction Pipeline
1. Collect simulation data
2. Preprocess data
3. Train model
4. Make predictions
5. Visualize results

### 2. Real-time Estimation System

The real-time estimation system provides continuous velocity predictions during simulation:

#### 2.1 RealTimeEstimator Class
- Components:
  - History buffer (deque with maxlen=60)
  - Prediction queue
  - iTransformer model
  - Data scaler

#### 2.2 Real-time Processing
- Data collection:
  - Updates measurements every simulation step
  - Tracks ego vehicle and surrounding vehicles
  - Maintains sliding window of historical data

#### 2.3 Prediction Thread
- Runs in background
- Makes predictions when sufficient history is available
- Updates predictions every 0.1 seconds
- Handles data preprocessing and model inference

#### 2.4 Main Loop
1. Initialize SUMO connection
2. Start prediction thread
3. For each simulation step:
   - Get ego vehicle data
   - Get surrounding vehicle data
   - Update measurements
   - Get and display predictions

## Technical Details

### Data Features
1. Ego Vehicle:
   - Speed
   - Acceleration

2. Surrounding Vehicles (up to 6):
   - Distance
   - Speed
   - Acceleration
   - Relative position (ahead/behind)

### Model Architecture
```python
iTransformer(
    num_variates=26,      # Number of features
    lookback_len=60,      # Historical time steps
    dim=128,             # Embedding dimension
    depth=4,             # Number of transformer layers
    heads=8,             # Number of attention heads
    dim_head=32,         # Dimension of each attention head
    pred_length=20       # Prediction horizon
)
```

### Performance Metrics
- Training loss: MSE
- Validation loss: MSE
- Prediction horizon: 20 seconds
- Update frequency: 0.1 seconds

## Usage

### Velocity Prediction
```python
# Train and predict
python velocity_prediction.py
```

### Real-time Estimation
```python
# Run real-time estimation
python estimation.py
```

## Integration

The two systems are designed to work together:
1. `velocity_prediction.py` trains the model on historical data
2. `estimation.py` uses the trained model for real-time predictions
3. Both systems use the same data preprocessing and model architecture

## Future Improvements

1. Model Enhancements:
   - Add more sophisticated attention mechanisms
   - Implement ensemble methods
   - Add uncertainty estimation

2. System Features:
   - Add more vehicle features
   - Implement adaptive prediction horizons
   - Add visualization tools

3. Performance:
   - Optimize real-time processing
   - Add parallel processing
   - Implement model compression

## Conclusion

The system provides a robust framework for vehicle velocity prediction and estimation in traffic simulations. The combination of offline training and real-time estimation allows for both accurate predictions and practical deployment in simulation environments. 