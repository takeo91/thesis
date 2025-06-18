"""
Visualize Metric Comparison

This script visualizes the comparison of different similarity metrics for the per-sensor
membership function approach, based on the results of the RQ2 experiment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import argparse

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Define metrics data
metrics_data = {
    "metric_name": ["Jaccard", "Dice", "Cosine"],
    "accuracy": [0.9412, 0.9412, 0.2353],
    "balanced_accuracy": [0.9697, 0.9697, 0.3333],
    "macro_f1": [0.9175, 0.9175, 0.1270],
    "computation_time": [4.56, 4.52, 4.45]
}

def create_performance_comparison(output_dir: Path):
    """Create performance comparison visualizations."""
    print(f"Creating visualizations in {output_dir}")
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_data)
    print(f"DataFrame created with shape: {df.shape}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot accuracy
    sns.barplot(x="metric_name", y="accuracy", data=df, ax=axes[0])
    axes[0].set_title("Accuracy by Similarity Metric", fontsize=14)
    axes[0].set_xlabel("Similarity Metric")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.0)
    
    # Plot balanced accuracy
    sns.barplot(x="metric_name", y="balanced_accuracy", data=df, ax=axes[1])
    axes[1].set_title("Balanced Accuracy by Similarity Metric", fontsize=14)
    axes[1].set_xlabel("Similarity Metric")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].set_ylim(0, 1.0)
    
    # Plot macro F1
    sns.barplot(x="metric_name", y="macro_f1", data=df, ax=axes[2])
    axes[2].set_title("Macro F1 by Similarity Metric", fontsize=14)
    axes[2].set_xlabel("Similarity Metric")
    axes[2].set_ylabel("Macro F1")
    axes[2].set_ylim(0, 1.0)
    
    # Add values on top of bars
    for ax in axes:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / "metric_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance comparison to {output_path}")
    
    # Create computation time comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x="metric_name", y="computation_time", data=df)
    plt.title("Computation Time by Similarity Metric", fontsize=14)
    plt.xlabel("Similarity Metric")
    plt.ylabel("Computation Time (seconds)")
    
    # Add values on top of bars
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():.2f}s', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / "metric_computation_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved computation time comparison to {output_path}")
    
    # Create radar chart for comprehensive comparison
    plt.figure(figsize=(10, 8))
    
    # Prepare data for radar chart
    metrics = ["Accuracy", "Balanced Accuracy", "Macro F1"]
    
    # Number of variables
    N = len(metrics)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], size=10)
    plt.ylim(0, 1)
    
    # Plot each similarity metric
    for i, metric in enumerate(df["metric_name"]):
        values = df.loc[i, ["accuracy", "balanced_accuracy", "macro_f1"]].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=metric)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Similarity Metrics Comparison", size=16)
    
    plt.tight_layout()
    output_path = output_dir / "metric_radar_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved radar comparison to {output_path}")
    
    # Create combined metrics visualization
    plt.figure(figsize=(12, 8))
    
    # Melt the dataframe for easier plotting
    df_melt = pd.melt(df, id_vars=["metric_name"], 
                      value_vars=["accuracy", "balanced_accuracy", "macro_f1"],
                      var_name="Metric", value_name="Value")
    
    # Create grouped bar chart
    sns.barplot(x="metric_name", y="Value", hue="Metric", data=df_melt)
    plt.title("Performance Metrics by Similarity Metric", fontsize=14)
    plt.xlabel("Similarity Metric")
    plt.ylabel("Value")
    plt.ylim(0, 1.0)
    plt.legend(title="Metric")
    
    plt.tight_layout()
    output_path = output_dir / "metric_combined_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined comparison to {output_path}")
    
    # Save data to CSV
    output_path = output_dir / "metric_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved data to {output_path}")
    
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize similarity metric comparison")
    parser.add_argument("--output_dir", type=str, default="results/metric_comparison",
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    print(f"Starting visualization with output directory: {args.output_dir}")
    create_performance_comparison(Path(args.output_dir))
    print("Visualization complete!") 