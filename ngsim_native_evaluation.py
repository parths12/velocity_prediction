#!/usr/bin/env python3
"""
NGSIM Native Evaluation Script
===============================
Comprehensive evaluation after native NGSIM training.

Author: Parth Shinde
Date: November 2025
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict

class NGSIMNativeEvaluator:
    """Evaluate natively-trained NGSIM model."""

    def __init__(
        self,
        model_path: str = './models/ngsim_native_best.pth',
        ngsim_data_path: str = './ngsim_processed',
        results_path: str = './ngsim_native_evaluation_results',
        device: str = 'cuda'
    ):
        self.model_path = Path(model_path)
        self.ngsim_path = Path(ngsim_data_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"📊 NGSIM Native Evaluator")
        print(f"   Model: {self.model_path}")
        print(f"   Device: {self.device}")

    def load_results(self, results_file: str = './ngsim_native_results.json'):
        """Load saved results."""
        with open(results_file, 'r') as f:
            return json.load(f)

    def create_comparison_report(
        self,
        native_results: Dict,
        adapter_rmse: float = 11.41,
        sumo_rmse: float = 0.356
    ):
        """Create comprehensive comparison report."""

        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    NGSIM NATIVE TRAINING RESULTS                     ║
╚══════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 PERFORMANCE COMPARISON

┌──────────────────┬─────────────┬──────────────┬──────────────────┐
│ Method           │ RMSE (m/s)  │ MAE (m/s)    │ Accuracy (%)     │
├──────────────────┼─────────────┼──────────────┼──────────────────┤
│ SUMO (Baseline)  │ {sumo_rmse:>11.3f} │ N/A          │ 96.5%            │
│ NGSIM (Adapter)  │ {adapter_rmse:>11.3f} │ 10.86        │ 16.21%           │
│ NGSIM (Native)   │ {native_results['rmse']:>11.3f} │ {native_results['mae']:<12.3f} │ {native_results['aggregate_accuracy']:<15.2f}%  │
└──────────────────┴─────────────┴──────────────┴──────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ KEY IMPROVEMENTS

Adapter → Native NGSIM:
  • RMSE Improvement: {adapter_rmse - native_results['rmse']:.3f} m/s 
    ({((adapter_rmse - native_results['rmse'])/adapter_rmse * 100):.1f}% reduction)

  • Accuracy Gain: {native_results['aggregate_accuracy'] - 16.21:.2f}% 
    (from 16.21% to {native_results['aggregate_accuracy']:.2f}%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 PER-HORIZON RMSE (20 seconds)

"""

        # Add per-timestep breakdown
        per_timestep_rmse = native_results['per_timestep_rmse']

        report += "┌─────────┬──────────────┐\n"
        report += "│ Horizon │ RMSE (m/s)   │\n"
        report += "├─────────┼──────────────┤\n"

        for i in range(0, 20, 5):
            if i+5 <= len(per_timestep_rmse):
                avg_rmse = np.mean(per_timestep_rmse[i:i+5])
                report += f"│ {i+1:>2}-{i+5:>2}s  │ {avg_rmse:>12.4f} │\n"

        report += "└─────────┴──────────────┘\n\n"

        # Add McMaster comparison
        report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        report += "🎯 COMPARISON TO MCMASTER PAPER\n\n"

        # Extract 3s RMSE
        rmse_3s = np.mean(per_timestep_rmse[:3])
        mcmaster_rmse = 0.321

        report += f"McMaster (3s horizon):     {mcmaster_rmse:.3f} m/s\n"
        report += f"Your Model (3s horizon):   {rmse_3s:.3f} m/s\n"

        if rmse_3s < mcmaster_rmse:
            report += f"\n✅ SUCCESS! You BEAT McMaster by {mcmaster_rmse - rmse_3s:.3f} m/s!\n"
        elif rmse_3s < mcmaster_rmse * 1.5:
            report += f"\n✓ Competitive! Within 50% of McMaster.\n"
        else:
            report += f"\n⚠️  Gap: {rmse_3s - mcmaster_rmse:.3f} m/s. Further tuning needed.\n"

        report += "\n" + "="*70 + "\n"
        report += "\n📝 CONCLUSION\n\n"

        if native_results['rmse'] < 2.0:
            report += "✅ EXCELLENT! Native training achieved publication-quality results.\n"
            report += "   Ready for conference submission.\n"
        elif native_results['rmse'] < 5.0:
            report += "✓ GOOD! Significant improvement over adapter approach.\n"
            report += "   Consider hyperparameter tuning for further gains.\n"
        else:
            report += ("⚠️  Moderate improvement. Check:" 
                      "\n   - Data preprocessing quality"
                      "\n   - Model architecture settings"
                      "\n   - Training hyperparameters\n")

        report += "\n" + "="*70 + "\n"

        # Save report with UTF-8 encoding
        report_path = self.results_path / 'comparison_report.txt'
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(report)
            print(f"\n💾 Report saved: {report_path}")
        except Exception as e:
            print(f"⚠️  Error saving report: {e}")
            # Fallback: Save without special characters
            clean_report = report.encode('ascii', 'ignore').decode('ascii')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(clean_report)
            print("⚠️  Saved report with ASCII characters only due to encoding issues")

        return report

    def visualize_improvements(
        self,
        native_results: Dict,
        adapter_rmse: float = 11.41,
        adapter_mae: float = 10.86
    ):
        """Create visualization comparing adapter vs native."""

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # RMSE comparison
        methods = ['NGSIM\n(Adapter)', 'NGSIM\n(Native)']
        rmse_values = [adapter_rmse, native_results['rmse']]
        colors = ['red', 'green']

        bars = axes[0].bar(methods, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        axes[0].set_ylabel('RMSE (m/s)', fontsize=12, weight='bold')
        axes[0].set_title('RMSE Comparison', fontsize=14, weight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, rmse_values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}',
                        ha='center', va='bottom', fontsize=11, weight='bold')

        # Add improvement annotation
        improvement = ((adapter_rmse - native_results['rmse']) / adapter_rmse) * 100
        axes[0].text(0.5, max(rmse_values) * 0.5,
                    f'{improvement:.1f}% Improvement',
                    ha='center', fontsize=13, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        # MAE comparison
        mae_values = [adapter_mae, native_results['mae']]
        bars = axes[1].bar(methods, mae_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        axes[1].set_ylabel('MAE (m/s)', fontsize=12, weight='bold')
        axes[1].set_title('MAE Comparison', fontsize=14, weight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, value in zip(bars, mae_values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}',
                        ha='center', va='bottom', fontsize=11, weight='bold')

        plt.tight_layout()
        save_path = self.results_path / 'improvement_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Improvement visualization saved: {save_path}")
        plt.close()

    def visualize_per_horizon(self, native_results: Dict):
        """Visualize per-horizon RMSE."""

        per_timestep_rmse = np.array(native_results['per_timestep_rmse'])
        per_timestep_mae = np.array(native_results['per_timestep_mae'])
        horizons = np.arange(1, len(per_timestep_rmse) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # RMSE vs horizon
        ax1.plot(horizons, per_timestep_rmse, 'o-', color='red', 
                linewidth=2, markersize=6, label='RMSE')
        ax1.set_xlabel('Prediction Horizon (seconds)', fontsize=12)
        ax1.set_ylabel('RMSE (m/s)', fontsize=12)
        ax1.set_title('RMSE vs Prediction Horizon (Native NGSIM)', fontsize=13, weight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # Highlight 3s mark (McMaster comparison)
        ax1.axvline(x=3, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='3s (McMaster)')
        rmse_3s = np.mean(per_timestep_rmse[:3])
        ax1.text(3, max(per_timestep_rmse) * 0.9, 
                f'3s RMSE: {rmse_3s:.3f}',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # MAE vs horizon
        ax2.plot(horizons, per_timestep_mae, 'o-', color='orange', 
                linewidth=2, markersize=6, label='MAE')
        ax2.set_xlabel('Prediction Horizon (seconds)', fontsize=12)
        ax2.set_ylabel('MAE (m/s)', fontsize=12)
        ax2.set_title('MAE vs Prediction Horizon (Native NGSIM)', fontsize=13, weight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        save_path = self.results_path / 'per_horizon_errors.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Per-horizon errors saved: {save_path}")
        plt.close()

def main():
    """Main evaluation pipeline."""

    print("="*70)
    print("NGSIM NATIVE EVALUATION")
    print("="*70)

    # Initialize evaluator
    evaluator = NGSIMNativeEvaluator(
        model_path='./models/ngsim_native_best.pth',
        ngsim_data_path='./ngsim_processed',
        results_path='./ngsim_native_evaluation_results',
        device='cuda'
    )

    # Load results
    print(f"\n📥 Loading results...")
    native_results = evaluator.load_results('./ngsim_native_results.json')
    print(f"   ✓ Results loaded")

    # Create comparison report
    print(f"\n📝 Creating comparison report...")
    evaluator.create_comparison_report(
        native_results=native_results,
        adapter_rmse=11.41,  # Your previous adapter result
        sumo_rmse=0.356       # Your SUMO result
    )

    # Visualize improvements
    print(f"\n📊 Creating visualizations...")
    evaluator.visualize_improvements(
        native_results=native_results,
        adapter_rmse=11.41,
        adapter_mae=10.86
    )

    evaluator.visualize_per_horizon(native_results)

    print(f"\n" + "="*70)
    print(f"EVALUATION COMPLETE!")
    print(f"="*70)
    print(f"\n📦 Check results in: ./ngsim_native_evaluation_results/")

if __name__ == '__main__':
    main()
