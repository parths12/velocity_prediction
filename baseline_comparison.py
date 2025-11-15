"""
baseline_comparison.py
Compares iTransformer against LSTM and Standard Transformer baselines
Quantifies performance improvement of feature-wise attention approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class LSTMBaseline(nn.Module):
    """Standard LSTM for velocity prediction (temporal baseline)"""

    def __init__(self, input_size=26, hidden_size=64, num_layers=2, pred_length=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, pred_length)
        )

    def forward(self, x):
        # x: (batch, time, features)
        lstm_out, (h, c) = self.lstm(x)  # lstm_out: (batch, time, hidden)

        # Take last timestep
        last_output = lstm_out[:, -1, :]  # (batch, hidden)

        # Predict
        predictions = self.fc(last_output)  # (batch, pred_length)

        return predictions


class StandardTransformer(nn.Module):
    """Standard Transformer (temporal attention baseline)"""

    def __init__(self, input_size=26, d_model=128, nhead=8, num_layers=4, pred_length=20):
        super().__init__()
        self.pred_length = pred_length

        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 60, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, pred_length)

    def forward(self, x):
        # x: (batch, time, features)
        x = self.embedding(x)  # (batch, time, d_model)
        x = x + self.pos_encoding

        x = self.transformer(x)  # (batch, time, d_model)

        # Take last timestep
        x = x[:, -1, :]  # (batch, d_model)

        predictions = self.fc(x)  # (batch, pred_length)

        return predictions


def train_model(model, train_loader, val_loader, model_name, epochs=50, lr=0.001):
    """Train any model and track losses"""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    print(f"\nTraining {model_name}...")
    print("-" * 50)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.float()
            y_batch = y_batch.float()

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.float()
                y_batch = y_batch.float()

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    print(f"[OK] {model_name} training complete!")
    return train_losses, val_losses


def evaluate_model(model, test_loader, model_name):
    """Evaluate model and return metrics"""

    model.eval()
    predictions_list = []
    targets_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.float()
            y_batch = y_batch.float()

            predictions = model(X_batch)
            predictions_list.append(predictions.numpy())
            targets_list.append(y_batch.numpy())

    predictions = np.vstack(predictions_list)
    targets = np.vstack(targets_list)

    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-6))) * 100

    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        'name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


def compare_models(lstm_results, transformer_results, itransformer_results):
    """Compare and visualize model performance"""

    print("\n" + "="*70)
    print("BASELINE COMPARISON RESULTS")
    print("="*70)

    models = [lstm_results, transformer_results, itransformer_results]
    names = ['LSTM', 'Standard Transformer', 'iTransformer']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    print("\nPerformance Metrics Comparison:")
    print("-" * 70)
    print(f"{'Model':<25} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'MAPE':<10}")
    print("-" * 70)

    for model_result, name in zip(models, names):
        print(f"{name:<25} {model_result['mse']:<12.6f} {model_result['rmse']:<12.6f} "
              f"{model_result['mae']:<12.6f} {model_result['mape']:<10.2f}%")

    # Calculate improvements
    lstm_mse = lstm_results['mse']
    transformer_mse = transformer_results['mse']
    itransformer_mse = itransformer_results['mse']

    improvement_vs_lstm = ((lstm_mse - itransformer_mse) / lstm_mse) * 100
    improvement_vs_transformer = ((transformer_mse - itransformer_mse) / transformer_mse) * 100

    print("\n" + "-" * 70)
    print(f"iTransformer Improvement vs LSTM:                    {improvement_vs_lstm:+.1f}%")
    print(f"iTransformer Improvement vs Standard Transformer:   {improvement_vs_transformer:+.1f}%")
    print("-" * 70)

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSE comparison
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']
    values = [
        [lstm_results['mse'], transformer_results['mse'], itransformer_results['mse']],
        [lstm_results['rmse'], transformer_results['rmse'], itransformer_results['rmse']],
        [lstm_results['mae'], transformer_results['mae'], itransformer_results['mae']],
        [lstm_results['mape'], transformer_results['mape'], itransformer_results['mape']]
    ]

    for idx, (ax, metric, vals) in enumerate(zip(axes.flat, metrics, values)):
        bars = ax.bar(names, vals, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Saved: baseline_comparison.png")
    plt.show()

    return improvement_vs_lstm, improvement_vs_transformer


if __name__ == "__main__":
    print("="*70)
    print("BASELINE COMPARISON MODULE")
    print("="*70)
    print("""
\nUsage Instructions:

1. Train LSTM baseline:
    lstm_model = LSTMBaseline()
    train_model(lstm_model, train_loader, val_loader, "LSTM", epochs=50)
    lstm_results = evaluate_model(lstm_model, test_loader, "LSTM")

2. Train Standard Transformer baseline:
    transformer_model = StandardTransformer()
    train_model(transformer_model, train_loader, val_loader, "Standard Transformer", epochs=50)
    transformer_results = evaluate_model(transformer_model, test_loader, "Transformer")

3. Train iTransformer:
    itransformer_model = iTransformer()
    train_model(itransformer_model, train_loader, val_loader, "iTransformer", epochs=50)
    itransformer_results = evaluate_model(itransformer_model, test_loader, "iTransformer")

4. Compare:
    improve_lstm, improve_tf = compare_models(
        lstm_results, transformer_results, itransformer_results
    )

This provides:
- Quantitative performance comparison
- Visualization of improvements
- Justification for feature-wise attention design
    """)
