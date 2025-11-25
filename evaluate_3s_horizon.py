import torch
import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent directory to path to import model definition
sys.path.append(str(Path(__file__).parent))
from ngsim_native_training import NGSIMTransformer  # Assuming this is where your model is defined

def load_model(model_path, device='cuda'):
    """Load the trained model."""
    # Initialize model with the same architecture as during training
    model = NGSIMTransformer(
        d_model=256,  # Match your training configuration
        nhead=4,      # Match your training configuration
        num_layers=3, # Match your training configuration
        dropout=0.1   # Match your training configuration
    )
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_test_data(data_path='./ngsim_processed/ngsim_test.pkl'):
    """Load test data."""
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)
    return test_data

def evaluate_3s_horizon():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and data
    print("Loading model...")
    model = load_model('./models/ngsim_native_best.pth', device)
    
    print("Loading test data...")
    test_data = load_test_data()
    
    # Assuming test_data is a dictionary with 'X' and 'y' keys
    test_inputs = torch.FloatTensor(test_data['X']).to(device)
    test_targets = torch.FloatTensor(test_data['y']).to(device)
    
    print("Running inference...")
    with torch.no_grad():
        predictions = model(test_inputs)
    
    # Extract first 3 seconds (assuming each timestep is 1 second)
    # Note: Adjust the slicing based on your actual data structure
    pred_3s = predictions[:, :3]  # First 3 timesteps
    target_3s = test_targets[:, :3]
    
    # Calculate RMSE for 3-second horizon
    mse = torch.mean((pred_3s - target_3s) ** 2)
    rmse_3s = torch.sqrt(mse).item()
    
    print("\n" + "="*50)
    print(f"3-second Horizon Evaluation")
    print("="*50)
    print(f"McMaster 3s RMSE: 0.321 m/s")
    print(f"Your 3s RMSE: {rmse_3s:.4f} m/s")
    print("="*50)
    
    # Additional metrics for 3-second horizon
    mae_3s = torch.mean(torch.abs(pred_3s - target_3s)).item()
    mape_3s = 100 * torch.mean(torch.abs((pred_3s - target_3s) / (target_3s + 1e-6))).item()
    
    print(f"\nAdditional 3-second Metrics:")
    print(f"MAE: {mae_3s:.4f} m/s")
    print(f"MAPE: {mape_3s:.2f}%")
    print("\nNote: Lower values are better for all metrics.")

if __name__ == "__main__":
    evaluate_3s_horizon()
