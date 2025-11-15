"""
uncertainty_quantification.py
Adds uncertainty quantification to iTransformer
Predicts both velocity AND confidence intervals
Critical for safety-critical autonomous driving
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


class UncertaintyiTransformer(nn.Module):
    """iTransformer with Bayesian uncertainty estimation"""

    def __init__(self, num_variates=26, lookback_len=60, dim=128, depth=4, 
                 heads=8, dim_head=16, pred_length=20):
        super().__init__()

        self.num_variates = num_variates
        self.lookback_len = lookback_len
        self.pred_length = pred_length

        # Embedding
        self.variate_embedding = nn.Linear(lookback_len, dim)
        self.variate_pos_embedding = nn.Parameter(torch.randn(1, num_variates, dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Dual prediction heads
        self.mean_head = nn.Linear(dim, pred_length)      # Predict mean velocity
        self.variance_head = nn.Linear(dim, pred_length)  # Predict uncertainty (log variance)

    def forward(self, x):
        """Forward pass returning both predictions and uncertainties"""
        # x: (batch, time, features)
        x = x.transpose(1, 2)  # (batch, features, time)

        # Embedding
        x = self.variate_embedding(x)  # (batch, features, dim)
        x = x + self.variate_pos_embedding

        # Transformer
        x = self.transformer_encoder(x)

        # Dual heads
        mean_pred = self.mean_head(x[:, 0, :])           # (batch, pred_length)
        log_variance = self.variance_head(x[:, 0, :])    # (batch, pred_length)
        variance = torch.exp(log_variance)               # Ensure positive

        return mean_pred, variance


def negative_log_likelihood_loss(mean, variance, target):
    """
    Negative log-likelihood loss for Gaussian distribution
    Combines prediction accuracy with confidence calibration
    """
    precision = 1.0 / (variance + 1e-6)
    loss = 0.5 * torch.log(variance) + 0.5 * precision * (target - mean) ** 2
    return loss.mean()


def train_uncertainty_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train uncertainty-aware iTransformer"""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    print("\n" + "="*70)
    print("TRAINING UNCERTAINTY-AWARE iTransformer")
    print("="*70)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.float()
            y_batch = y_batch.float()

            optimizer.zero_grad()

            mean_pred, variance = model(X_batch)
            loss = negative_log_likelihood_loss(mean_pred, variance, y_batch)

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

                mean_pred, variance = model(X_batch)
                loss = negative_log_likelihood_loss(mean_pred, variance, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print(f"\n[OK] Training complete!")
    return train_losses, val_losses


def evaluate_uncertainty(model, test_loader):
    """Evaluate uncertainty calibration"""

    print("\n" + "="*70)
    print("UNCERTAINTY EVALUATION")
    print("="*70)

    model.eval()
    all_means = []
    all_vars = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.float()
            y_batch = y_batch.float()

            mean_pred, variance = model(X_batch)

            all_means.append(mean_pred.numpy())
            all_vars.append(variance.numpy())
            all_targets.append(y_batch.numpy())

    means = np.vstack(all_means)
    variances = np.vstack(all_vars)
    targets = np.vstack(all_targets)

    # Calculate metrics
    mse = np.mean((means - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(means - targets))

    # Uncertainty calibration: what % of errors fall within confidence interval?
    std_devs = np.sqrt(variances)
    errors = np.abs(means - targets)

    # Check different confidence levels
    print("\nPrediction Accuracy:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f} m/s")
    print(f"  MAE:  {mae:.6f} m/s")

    print("\nUncertainty Calibration (% of errors within bounds):")
    for confidence_level in [68, 95, 99]:  # 1σ, 2σ, 3σ
        multiplier = (confidence_level / 68.0)  # σ units
        within_bounds = (errors < multiplier * std_devs).mean() * 100
        print(f"  {confidence_level}% confidence: {within_bounds:.1f}% of predictions within bounds")

    # Average predicted uncertainty
    avg_predicted_uncertainty = std_devs.mean()
    print(f"\nAverage Predicted Uncertainty: {avg_predicted_uncertainty:.6f} m/s")

    return means, variances, targets


def visualize_predictions_with_uncertainty(means, variances, targets, num_samples=10):
    """Visualize predictions with confidence intervals"""

    std_devs = np.sqrt(variances)

    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()

    for idx in range(min(num_samples, len(means))):
        ax = axes[idx]

        time_steps = np.arange(20)

        # Plot actual
        ax.plot(time_steps, targets[idx], 'g-', linewidth=2.5, label='Actual', marker='o')

        # Plot mean prediction
        ax.plot(time_steps, means[idx], 'b-', linewidth=2.5, label='Predicted', marker='s')

        # Plot confidence interval (±1σ)
        ax.fill_between(
            time_steps,
            means[idx] - std_devs[idx],
            means[idx] + std_devs[idx],
            alpha=0.3,
            label='±1σ Uncertainty'
        )

        # Plot ±2σ
        ax.fill_between(
            time_steps,
            means[idx] - 2*std_devs[idx],
            means[idx] + 2*std_devs[idx],
            alpha=0.15,
            label='±2σ Uncertainty'
        )

        ax.set_title(f'Sample {idx+1}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Velocity (m/s)')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig('predictions_with_uncertainty.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Saved: predictions_with_uncertainty.png")
    plt.show()


def plot_uncertainty_metrics(means, variances, targets):
    """Visualize uncertainty metrics"""

    std_devs = np.sqrt(variances)
    errors = np.abs(means - targets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Error vs Uncertainty
    axes[0].scatter(std_devs.flatten(), errors.flatten(), alpha=0.3, s=10)
    axes[0].set_xlabel('Predicted Uncertainty (σ)')
    axes[0].set_ylabel('Absolute Error')
    axes[0].set_title('Error vs Predicted Uncertainty')
    axes[0].grid(True, alpha=0.3)

    # Correlation
    correlation = np.corrcoef(std_devs.flatten(), errors.flatten())[0, 1]
    axes[0].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Calibration curve
    confidence_levels = np.linspace(10, 99, 20)
    within_bounds_pct = []

    for conf in confidence_levels:
        multiplier = conf / 68.0  # Convert to σ units
        within = (errors < multiplier * std_devs).mean() * 100
        within_bounds_pct.append(within)

    axes[1].plot(confidence_levels, within_bounds_pct, 'b-', linewidth=2, label='Model')
    axes[1].plot([10, 99], [10, 99], 'r--', linewidth=2, label='Ideal Calibration')
    axes[1].set_xlabel('Expected Coverage (%)')
    axes[1].set_ylabel('Observed Coverage (%)')
    axes[1].set_title('Uncertainty Calibration Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([10, 100])
    axes[1].set_ylim([10, 100])

    plt.tight_layout()
    plt.savefig('uncertainty_metrics.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: uncertainty_metrics.png")
    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("UNCERTAINTY QUANTIFICATION MODULE")
    print("="*70)
    print("""
\nUsage Instructions:

1. Create model:
    model = UncertaintyiTransformer()

2. Train with uncertainty:
    train_losses, val_losses = train_uncertainty_model(
        model, train_loader, val_loader, epochs=50
    )

3. Evaluate:
    means, variances, targets = evaluate_uncertainty(model, test_loader)

4. Visualize:
    visualize_predictions_with_uncertainty(means, variances, targets)
    plot_uncertainty_metrics(means, variances, targets)

This enables:
- Confidence intervals for predictions
- Detection of uncertain scenarios
- Safety-critical decision making
- Model calibration assessment
    """)
