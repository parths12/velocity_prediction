# Vehicle Velocity Prediction with SUMO and iTransformer

This project uses SUMO (Simulation of Urban MObility) to simulate traffic scenarios and collects data about a vehicle and its surroundings. It then uses an iTransformer model to predict the vehicle's velocity for the next 20 seconds.

## Requirements

- Python 3.7+
- SUMO 1.8.0+
- PyTorch 1.8.0+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- seaborn

## Installation

1. Install SUMO following the instructions at [SUMO Installation Guide](https://sumo.dlr.de/docs/Installing/index.html).

2. Set the SUMO_HOME environment variable:
   - On Windows: `set SUMO_HOME=C:\path\to\sumo`
   - On Linux/Mac: `export SUMO_HOME=/path/to/sumo`

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

- `velocity_prediction.py`: Main script for data collection, model training, and prediction
- `predict.py`: Script to load a trained model and make predictions
- `run_multiple_simulations.py`: Script to run multiple simulations with different traffic conditions
- `visualize_data.py`: Script to visualize the collected data from simulations
- `evaluate_model.py`: Script to evaluate the model's performance using various metrics
- `run_pipeline.py`: Script to run the entire pipeline from simulation to evaluation
- `simulation.sumocfg`: SUMO configuration file
- `network.net.xml`: SUMO network file with a 3-lane highway
- `routes.rou.xml`: SUMO routes file with ego vehicle and traffic
- `gui-settings.xml`: SUMO GUI settings

## Usage

### Running the Complete Pipeline

The easiest way to run the entire system is to use the pipeline script:

1. Run the pipeline with a single simulation:
   ```
   python run_pipeline.py
   ```

2. Run the pipeline with multiple simulations:
   ```
   python run_pipeline.py --multiple-simulations
   ```

3. Skip the simulation step (use existing data):
   ```
   python run_pipeline.py --skip-simulation
   ```

4. Skip the training step (use existing model):
   ```
   python run_pipeline.py --skip-training
   ```

5. Specify custom model and data files:
   ```
   python run_pipeline.py --model custom_model.pth --data custom_data.csv
   ```

### Running Individual Components

If you prefer to run each component separately, you can use the following scripts:

#### Running a Single Simulation and Training the Model

1. Run the main script:
   ```
   python velocity_prediction.py
   ```

   This will:
   - Run the SUMO simulation
   - Collect data about the ego vehicle and surrounding vehicles
   - Preprocess the data
   - Train the iTransformer model
   - Save the model and data
   - Make predictions for the next 20 seconds
   - Generate a plot of the results

#### Running Multiple Simulations with Different Traffic Conditions

1. Run the multiple simulations script:
   ```
   python run_multiple_simulations.py
   ```

   This will:
   - Run multiple SUMO simulations with different traffic densities
   - Collect data from each simulation
   - Combine the data from all simulations
   - Train a model on the combined data
   - Make predictions for each traffic density
   - Generate plots of the results

#### Making Predictions with a Trained Model

1. After training, you can use the prediction script:
   ```
   python predict.py
   ```

   This will:
   - Load the saved model and data
   - Make predictions for the next 20 seconds
   - Generate a plot of the results

#### Visualizing Simulation Data

1. To visualize data from a single simulation:
   ```
   python visualize_data.py --file simulation_data.csv
   ```

2. To compare data from multiple simulations:
   ```
   python visualize_data.py --files simulation_data_very_light.csv simulation_data_light.csv simulation_data_heavy.csv
   ```

3. To visualize the impact of traffic density on vehicle speed:
   ```
   python visualize_data.py --density-impact
   ```

#### Evaluating Model Performance

1. To evaluate a trained model's performance:
   ```
   python evaluate_model.py --model velocity_prediction_model.pth --data simulation_data.csv
   ```

   This will:
   - Load the model and data
   - Make predictions and compare with actual values
   - Calculate metrics like MSE, RMSE, MAE, and MAPE
   - Generate plots of the metrics and sample predictions

2. You can also specify custom lookback and prediction horizons:
   ```
   python evaluate_model.py --model velocity_prediction_model.pth --data simulation_data.csv --lookback 60 --horizon 20
   ```

## How It Works

1. **Data Collection**: The system runs a SUMO simulation and collects data about:
   - Ego vehicle: speed, acceleration, lane position
   - Surrounding vehicles: distance, speed, acceleration, relative position

2. **Preprocessing**: The collected data is normalized and prepared for the model.

3. **Model Training**: An iTransformer model is trained on the preprocessed data. The iTransformer is a variant of the Transformer architecture specifically designed for time series forecasting.

4. **Prediction**: The trained model predicts the ego vehicle's velocity for the next 20 seconds.

5. **Evaluation**: The model's performance is evaluated using metrics like MSE, RMSE, MAE, and MAPE.

## Customization

- Modify `routes.rou.xml` to change traffic conditions
- Adjust model parameters in `velocity_prediction.py`
- Change the prediction horizon by modifying the `pred_length` parameter
- Add more traffic density scenarios in `run_multiple_simulations.py`
- Create custom visualizations in `visualize_data.py`
- Add additional evaluation metrics in `evaluate_model.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 