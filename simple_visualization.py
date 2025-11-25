import os
import numpy as np
from matplotlib import pyplot as plt

# Create output directory
output_dir = "visualization_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Data from the comparison report
methods = ['SUMO (Baseline)', 'NGSIM (Adapter)', 'NGSIM (Native)']
rmse = [0.356, 11.410, 2.368]
mae = [None, 10.86, 1.703]
accuracy = [96.5, 16.21, 83.16]

horizons = ['1-5s', '6-10s', '11-15s', '16-20s']
horizon_rmses = [1.5793, 2.1026, 2.4897, 3.0342]

# 1. Performance Comparison
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(methods))

# Plot RMSE
plt.bar(index, rmse, bar_width, label='RMSE (m/s)')

# Plot MAE (skip None values)
mae_values = [m if m is not None else 0 for m in mae]
plt.bar(index + bar_width, mae_values, bar_width, label='MAE (m/s)')

# Add accuracy as text
for i, acc in enumerate(accuracy):
    plt.text(i, max(rmse) * 1.1, f'{acc:.1f}%', ha='center', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('Method')
plt.ylabel('Error (m/s)')
plt.title('Model Performance Comparison')
plt.xticks(index + bar_width/2, methods)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/performance_comparison.png', dpi=150)
plt.close()

# 2. Horizon Analysis
plt.figure(figsize=(10, 6))
bars = plt.bar(horizons, horizon_rmses, color='skyblue')

# Add value labels
for bar, rmse_val in zip(bars, horizon_rmses):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{rmse_val:.2f}', ha='center')

# Add McMaster baseline
plt.axhline(y=0.321, color='r', linestyle='--', label='McMaster Baseline (3s)')
plt.text(0.5, 0.4, 'McMaster Baseline (0.32 m/s)', color='r')

plt.xlabel('Prediction Horizon')
plt.ylabel('RMSE (m/s)')
plt.title('Prediction Error by Horizon')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/horizon_analysis.png', dpi=150)
plt.close()

# 3. Improvement Comparison
plt.figure(figsize=(12, 6))

# RMSE Improvement
plt.subplot(1, 2, 1)
rmse_data = [11.410, 2.368]
plt.bar(['Adapter', 'Native'], rmse_data, color=['lightcoral', 'lightgreen'])
plt.title('RMSE Comparison')
plt.ylabel('RMSE (m/s)')
plt.text(0.5, 7, f'79.2% Reduction', ha='center', 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# Accuracy Improvement
plt.subplot(1, 2, 2)
acc_data = [16.21, 83.16]
plt.bar(['Adapter', 'Native'], acc_data, color=['lightcoral', 'lightgreen'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.text(0.5, 50, f'+66.95% Improvement', ha='center',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.suptitle('Adapter vs Native NGSIM Performance')
plt.tight_layout()
plt.savefig(f'{output_dir}/improvement_comparison.png', dpi=150)
plt.close()

print(f"✅ Visualizations saved to: {output_dir}/")
print("\nGenerated files:")
for f in os.listdir(output_dir):
    if f.endswith('.png'):
        print(f"- {f}")
