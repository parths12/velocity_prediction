"""
attention_visualization.py
Visualizes learned attention weights from iTransformer model
Shows which traffic features the model focuses on for predictions
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def extract_attention_weights(model, input_data, layer_idx=0, head_idx=0):
    """Extract attention weights from a specific transformer layer/head.

    This directly calls the layer's MultiheadAttention with need_weights=True
    to ensure weights are returned (hooks are unreliable across torch versions).
    """
    model.eval()

    # Transpose input to feature-first format and embed
    x = input_data.transpose(1, 2)  # [batch, features, time]
    x = model.variate_embedding(x)  # [batch, features, dim]
    x = x + model.variate_pos_embedding

    # Select layer
    try:
        layer = model.transformer_encoder.layers[layer_idx]
    except Exception:
        return None

    # Call self-attention with weights requested
    with torch.no_grad():
        attn_out, attn_weights = layer.self_attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )

    if attn_weights is None:
        return None

    # attn_weights: [batch, heads, seq, seq]
    weights = attn_weights[0, head_idx].cpu().numpy()
    return weights


def visualize_attention_heatmap(model, input_data, layer_idx=0, head_idx=0, feature_names=None):
    """Visualize attention weights as heatmap"""
    attention_weights = extract_attention_weights(model, input_data, layer_idx, head_idx)

    if attention_weights is None:
        print(f"Could not extract attention weights for layer {layer_idx}")
        return

    if feature_names is None:
        n = input_data.shape[2]
        feature_names = [f'feat_{i}' for i in range(n)]

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        attention_weights,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        square=True
    )
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Attended Features')
    plt.ylabel('Query Features')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'attention_layer{layer_idx}_head{head_idx}.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: attention_layer{layer_idx}_head{head_idx}.png")
    plt.show()


def analyze_feature_importance(model, input_data, feature_names=None):
    """Analyze which features receive most attention"""
    if feature_names is None:
        n = input_data.shape[2]
        feature_names = [f'feat_{i}' for i in range(n)]

    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    n = input_data.shape[2]
    feature_importance = np.zeros(n)

    # Average attention across all layers and heads
    for layer in range(4):
        for head in range(8):
            weights = extract_attention_weights(model, input_data, layer, head)
            if weights is not None:
                feature_importance += weights.sum(axis=0)

    # Normalize
    feature_importance = feature_importance / (4 * 8)

    # Sort and display
    sorted_indices = np.argsort(feature_importance)[::-1]

    print("\nTop 10 Most Important Features (by attention):")
    for rank, idx in enumerate(sorted_indices[:10], 1):
        print(f"{rank:2d}. {feature_names[idx]:20s}: {feature_importance[idx]:.6f}")

    # Visualization
    plt.figure(figsize=(12, 6))
    top_indices = sorted_indices[:15]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_indices)))
    plt.barh(range(len(top_indices)), feature_importance[top_indices], color=colors)
    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
    plt.xlabel('Average Attention Weight')
    plt.title('Top 15 Most Important Features (by Attention)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: feature_importance.png")
    plt.show()


def visualize_ego_attention(model, input_data):
    """Visualize which features ego_speed attends to most"""
    print("\n" + "="*60)
    print("EGO VEHICLE ATTENTION ANALYSIS")
    print("="*60)

    n = input_data.shape[2]
    ego_attention = np.zeros(n)
    feature_names = [f'feat_{i}' for i in range(n)]

    # Analyze attention for ego_speed (feature 0)
    for layer in range(4):
        for head in range(8):
            weights = extract_attention_weights(model, input_data, layer, head)
            if weights is not None:
                # ego_speed is first feature (index 0), see what it attends to
                ego_attention += weights[0]  # First row of attention matrix

    # Normalize
    ego_attention = ego_attention / (4 * 8)

    # Sort
    sorted_indices = np.argsort(ego_attention)[::-1]

    print("\nEgo Velocity Attention Distribution:")
    print("(What features does ego velocity pay attention to?)")
    for rank, idx in enumerate(sorted_indices[:12], 1):
        print(f"{rank:2d}. {feature_names[idx]:20s}: {ego_attention[idx]:.6f}")

    # Visualization
    plt.figure(figsize=(12, 7))
    colors = ['#FF6B6B' if i == 0 else '#4ECDC4' for i in sorted_indices[:15]]
    plt.barh(range(len(sorted_indices[:15])), ego_attention[sorted_indices[:15]], color=colors)
    plt.yticks(range(len(sorted_indices[:15])), [feature_names[i] for i in sorted_indices[:15]])
    plt.xlabel('Attention Weight')
    plt.title('What Does Ego Velocity Attend To? (Feature Relationships)')
    plt.tight_layout()
    plt.savefig('ego_attention.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: ego_attention.png")
    plt.show()


if __name__ == "__main__":
    print("="*60)
    print("ATTENTION VISUALIZATION MODULE")
    print("="*60)
    print("""
\nUsage Instructions:
1. Load your trained model and test data
2. Call visualization functions with your model

Example:
    import torch
    from attention_visualization import *

    # Load model
    model = torch.load('velocity_prediction_model.pth')

    # Load or create test data
    input_data = torch.randn(32, 60, 26)  # batch, time, features

    # Generate visualizations
    visualize_attention_heatmap(model, input_data, layer_idx=0, head_idx=0)
    analyze_feature_importance(model, input_data)
    visualize_ego_attention(model, input_data)
    """)
