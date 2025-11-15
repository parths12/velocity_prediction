#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# Non-interactive plotting
os.environ.setdefault('MPLBACKEND', 'Agg')

from velocity_prediction import (
    iTransformer,
    preprocess_data,
    TimeSeriesDataset,
)

from baseline_comparison import (
    LSTMBaseline,
    StandardTransformer,
    train_model as train_baseline_model,
    evaluate_model as eval_baseline_model,
    compare_models,
)

from attention_visualization import (
    visualize_attention_heatmap,
    analyze_feature_importance,
    visualize_ego_attention,
)

from uncertainty_quantification import (
    UncertaintyiTransformer,
    train_uncertainty_model,
    evaluate_uncertainty,
    visualize_predictions_with_uncertainty,
    plot_uncertainty_metrics,
)

import pandas as pd


def build_loaders_from_csv(csv_path: str, lookback_len: int = 60, pred_length: int = 20, batch_size: int = 64):
    df = pd.read_csv(csv_path)
    X, _ = preprocess_data(df)

    dataset = TimeSeriesDataset(X, lookback_len, pred_length)

    if len(dataset) < 100:
        raise RuntimeError("Not enough samples to build loaders. Generate more combined data.")

    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, X


def load_itransformer(num_features: int, lookback_len: int = 60, pred_length: int = 20, weight_path: str = None):
    model = iTransformer(
        num_variates=num_features,
        lookback_len=lookback_len,
        dim=128,
        depth=4,
        heads=8,
        dim_head=32,
        pred_length=pred_length,
    )
    if weight_path and os.path.exists(weight_path):
        state = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state)
    return model


def main():
    lookback_len = 60
    pred_length = 20
    combined_csv = 'combined_simulation_data.csv'

    if not os.path.exists(combined_csv):
        raise FileNotFoundError("combined_simulation_data.csv not found. Run run_multiple_simulations.py first.")

    # Build data loaders
    train_loader, val_loader, test_loader, X_all = build_loaders_from_csv(
        combined_csv, lookback_len, pred_length, batch_size=64
    )
    num_features = X_all.shape[1]

    # 1) Baselines vs iTransformer
    lstm = LSTMBaseline(input_size=num_features, hidden_size=64, num_layers=2, pred_length=pred_length)
    tfm = StandardTransformer(input_size=num_features, d_model=128, nhead=8, num_layers=4, pred_length=pred_length)
    itr = load_itransformer(num_features, lookback_len, pred_length, 'combined_velocity_prediction_model.pth')

    # Train baselines briefly to keep runtime reasonable
    train_baseline_model(lstm, train_loader, val_loader, "LSTM", epochs=10, lr=1e-3)
    train_baseline_model(tfm, train_loader, val_loader, "Standard Transformer", epochs=10, lr=1e-3)

    lstm_results = eval_baseline_model(lstm, test_loader, "LSTM")
    tfm_results = eval_baseline_model(tfm, test_loader, "Transformer")
    itr_results = eval_baseline_model(itr, test_loader, "iTransformer")

    compare_models(lstm_results, tfm_results, itr_results)

    # 2) Attention visualizations from iTransformer
    # Grab a small batch from test loader
    test_batch = next(iter(test_loader))[0][:32]  # (batch, time, features)
    visualize_attention_heatmap(itr, test_batch, layer_idx=0, head_idx=0)
    analyze_feature_importance(itr, test_batch)
    visualize_ego_attention(itr, test_batch)

    # 3) Uncertainty quantification (brief training)
    u_model = UncertaintyiTransformer(num_variates=num_features, lookback_len=lookback_len, pred_length=pred_length)
    train_uncertainty_model(u_model, train_loader, val_loader, epochs=5, lr=1e-3)
    means, variances, targets = evaluate_uncertainty(u_model, test_loader)
    visualize_predictions_with_uncertainty(means, variances, targets)
    plot_uncertainty_metrics(means, variances, targets)

    print("\nAll future work modules executed. Outputs saved as PNGs in the project root.")


if __name__ == '__main__':
    main()


