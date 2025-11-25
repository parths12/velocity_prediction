import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from matplotlib.ticker import MaxNLocator

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")

# Create output directory
output_dir = Path("visualization_results")
output_dir.mkdir(exist_ok=True)

def load_comparison_data():
    """Load data from the comparison report."""
    data = {
        'methods': ['SUMO (Baseline)', 'NGSIM (Adapter)', 'NGSIM (Native)'],
        'rmse': [0.356, 11.410, 2.368],
        'mae': [None, 10.86, 1.703],
        'accuracy': [96.5, 16.21, 83.16]
    }
    
    horizons = ['1-5s', '6-10s', '11-15s', '16-20s']
    horizon_rmses = [1.5793, 2.1026, 2.4897, 3.0342]
    
    return data, horizons, horizon_rmses

def plot_performance_comparison(data):
    """Plot performance comparison between different methods."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter out None values for MAE
    mae_data = [x for x in data['mae'] if x is not None]
    mae_labels = [l for l, v in zip(data['methods'], data['mae']) if v is not None]
    
    # Plot RMSE and MAE
    x = np.arange(len(data['methods']))
    width = 0.3
    
    # RMSE bars
    bars1 = ax.bar(x - width/2, data['rmse'], width, label='RMSE (m/s)', 
                  color=sns.color_palette()[0])
    
    # MAE bars (only for methods with MAE data)
    bars2 = ax.bar(x[1:] + width/2, mae_data, width, label='MAE (m/s)', 
                  color=sns.color_palette()[1])
    
    # Accuracy line
    ax2 = ax.twinx()
    line = ax2.plot(x, data['accuracy'], 'o-', color=sns.color_palette()[2], 
                   linewidth=2, markersize=8, label='Accuracy (%)')
    
    # Add value labels
    for i, v in enumerate(data['rmse']):
        ax.text(i - width/2, v + 0.2, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(mae_data):
        ax.text(i + 1 + width/2, v + 0.2, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(data['accuracy']):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # Customize the plot
    ax.set_xticks(x)
    ax.set_xticklabels(data['methods'])
    ax.set_ylabel('Error (m/s)')
    ax2.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, max(data['rmse']) * 1.2)
    ax2.set_ylim(0, 110)
    
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.title('Model Performance Comparison', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_horizon_analysis(horizons, rmses):
    """Plot RMSE across different prediction horizons."""
    plt.figure(figsize=(12, 6))
    
    # Plot RMSE vs horizon
    x = np.arange(len(horizons))
    bars = plt.bar(x, rmses, color=sns.color_palette("Blues", len(horizons)))
    
    # Add value labels
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{rmse:.2f}', ha='center', va='bottom')
    
    # Add McMaster baseline
    plt.axhline(y=0.321, color='r', linestyle='--', label='McMaster Baseline (3s)')
    plt.text(len(horizons)-0.5, 0.35, 'McMaster Baseline (0.32 m/s)', color='r')
    
    # Customize the plot
    plt.xticks(x, horizons)
    plt.xlabel('Prediction Horizon')
    plt.ylabel('RMSE (m/s)')
    plt.title('Prediction Error Across Different Horizons', pad=20)
    plt.ylim(0, max(rmses) * 1.15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'horizon_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_improvement_comparison():
    """Plot the improvement from Adapter to Native NGSIM."""
    improvements = {
        'RMSE': {
            'Adapter': 11.410,
            'Native': 2.368,
            'Improvement': 9.042,
            'Improvement %': 79.2
        },
        'Accuracy': {
            'Adapter': 16.21,
            'Native': 83.16,
            'Improvement': 66.95,
            'Improvement %': 413.0
        }
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # RMSE Improvement
    ax1.bar(['Adapter', 'Native'], 
            [improvements['RMSE']['Adapter'], improvements['RMSE']['Native']],
            color=[sns.color_palette()[3], sns.color_palette()[0]])
    ax1.set_ylabel('RMSE (m/s)')
    ax1.set_title('RMSE Improvement')
    ax1.text(0.5, 0.95, f"{improvements['RMSE']['Improvement %']}% Reduction", 
             transform=ax1.transAxes, ha='center', va='top', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Accuracy Improvement
    ax2.bar(['Adapter', 'Native'], 
            [improvements['Accuracy']['Adapter'], improvements['Accuracy']['Native']],
            color=[sns.color_palette()[3], sns.color_palette()[2]])
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Improvement')
    ax2.text(0.5, 0.95, f"+{improvements['Accuracy']['Improvement']:.1f}% Improvement", 
             transform=ax2.transAxes, ha='center', va='top', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.suptitle('Improvement from Adapter to Native NGSIM', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_visualization():
    """Create a summary visualization with all key metrics."""
    data, horizons, horizon_rmses = load_comparison_data()
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    
    # Performance comparison subplot
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(data['methods']))
    width = 0.35
    
    # RMSE bars
    bars1 = ax1.bar(x - width/2, data['rmse'], width, label='RMSE (m/s)')
    
    # MAE bars (only for methods with MAE data)
    mae_data = [x for x in data['mae'] if x is not None]
    bars2 = ax1.bar(x[1:] + width/2, mae_data, width, label='MAE (m/s)')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.2f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Accuracy line
    ax1b = ax1.twinx()
    line, = ax1b.plot(x, data['accuracy'], 'o-', color='green', 
                     linewidth=2, markersize=8, label='Accuracy (%)')
    
    # Customize subplot 1
    ax1.set_xticks(x)
    ax1.set_xticklabels(data['methods'])
    ax1.set_ylabel('Error (m/s)')
    ax1.set_ylim(0, max(data['rmse']) * 1.3)
    ax1b.set_ylabel('Accuracy (%)')
    ax1b.set_ylim(0, 110)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.set_title('Model Performance Comparison', pad=20)
    
    # Horizon analysis subplot
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(horizons))
    bars = ax2.bar(x, horizon_rmses, color=sns.color_palette("Blues", len(horizons)))
    
    # Add value labels
    for bar, rmse in zip(bars, horizon_rmses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{rmse:.2f}', ha='center', va='bottom')
    
    # Add McMaster baseline
    ax2.axhline(y=0.321, color='r', linestyle='--', label='McMaster Baseline (3s)')
    
    # Customize subplot 2
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons)
    ax2.set_xlabel('Prediction Horizon')
    ax2.set_ylabel('RMSE (m/s)')
    ax2.set_ylim(0, max(horizon_rmses) * 1.2)
    ax2.legend()
    ax2.set_title('Prediction Error by Horizon', pad=20)
    
    # Improvement comparison subplot
    ax3 = fig.add_subplot(gs[1, 1])
    
    # RMSE improvement
    rmse_improve = [11.410, 2.368]
    acc_improve = [16.21, 83.16]
    
    ax3b = ax3.twinx()
    
    # Plot RMSE improvement
    bars_rmse = ax3.bar([0], [rmse_improve[0]], width=0.4, color=sns.color_palette()[0], 
                       label='RMSE (m/s)')
    bars_rmse2 = ax3.bar([0.5], [rmse_improve[1]], width=0.4, color=sns.color_palette()[1])
    
    # Plot accuracy improvement
    bars_acc = ax3b.bar([1], [acc_improve[0]], width=0.4, color=sns.color_palette()[2], 
                       alpha=0.7, label='Accuracy (%)')
    bars_acc2 = ax3b.bar([1.5], [acc_improve[1]], width=0.4, 
                        color=sns.color_palette()[3], alpha=0.7)
    
    # Add value labels
    for bar, val in zip(bars_rmse + bars_rmse2, rmse_improve):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2, 
                f'{val:.2f}', ha='center')
    
    for bar, val in zip(bars_acc + bars_acc2, acc_improve):
        height = bar.get_height()
        ax3b.text(bar.get_x() + bar.get_width()/2., height + 1, 
                 f'{val:.1f}%', ha='center')
    
    # Customize subplot 3
    ax3.set_xticks([0.25, 1.25])
    ax3.set_xticklabels(['RMSE', 'Accuracy'])
    ax3.set_ylabel('RMSE (m/s)')
    ax3b.set_ylabel('Accuracy (%)')
    
    # Add legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, ['RMSE (m/s)', 'Accuracy (%)'], loc='upper center')
    
    ax3.set_title('Adapter vs Native NGSIM', pad=20)
    
    # Add improvement arrows and text
    ax3.annotate('', xy=(0.5, 5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='<->', color='red', linewidth=2))
    ax3.text(0.5, 3, '79.2%\nReduction', ha='center', color='red')
    
    ax3b.annotate('', xy=(1.5, 50), xytext=(1.5, 20),
                 arrowprops=dict(arrowstyle='<->', color='green', linewidth=2))
    ax3b.text(1.5, 35, '+66.95%\nImprovement', ha='center', color='green')
    
    # Adjust layout and save
    plt.suptitle('NGSIM Model Performance Analysis', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("🚀 Generating visualizations...")
    
    # Load data
    data, horizons, horizon_rmses = load_comparison_data()
    
    # Generate individual plots
    plot_performance_comparison(data)
    plot_horizon_analysis(horizons, horizon_rmses)
    plot_improvement_comparison()
    
    # Generate summary visualization
    create_summary_visualization()
    
    print(f"✅ Visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    for f in output_dir.glob('*.png'):
        print(f"- {f.name}")

if __name__ == "__main__":
    main()
