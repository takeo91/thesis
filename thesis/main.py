"""
Main entry point for the thesis project.

This script provides examples of how to use the thesis modules
for analyzing sensor data and computing fuzzy similarity metrics.
"""

import argparse
import os

from thesis.analysis import (
    process_opportunity_dataset,
    run_experiment,
)


def run_opportunity_dataset_example():
    """Run an example analysis on the Opportunity dataset."""
    print("Loading Opportunity dataset...")
    
    # Process the dataset
    processor = process_opportunity_dataset()
    
    # Get a summary of the data
    summary = processor.get_data_summary()
    print("\nDataset Summary:")
    for key, value in summary.items():
        if key != "label_activities":
            print(f"  {key}: {value}")
    
    print("\nActivity Labels:")
    for activity_type, labels in summary["label_activities"].items():
        print(f"  {activity_type}: {labels}")
    
    # Plot Back Accelerometer data during Walking
    print("\nPlotting Back Accelerometer data during Walking...")
    processor.plot_sensor_data(
        sensor_type="Accelerometer",
        body_part="BACK",
        measurement_type="acc",
        axis="X",
        activity_filter={"Locomotion": "Walk"},
        save_path="results/back_acc_walking.png"
    )
    
    print("Example completed. Plot saved to results/back_acc_walking.png")


def run_similarity_experiment():
    """Run an experiment to compare different similarity metrics."""
    print("Running similarity metrics experiment...")
    
    # Run the experiment with default settings
    results_df = run_experiment(
        specific_cases=[1, 2, 3],  # Run only specific test cases
        sigma_options=[0.1, 0.2, "r0.1"],  # Use specific sigma values
        n_jobs=4,  # Use 4 parallel processes (adjust based on your system)
        save_plots=True
    )
    
    # Print summary of results
    print("\nResults Summary:")
    print(f"Completed {len(results_df)} parameter combinations")
    
    # Calculate mean scores grouped by case and normalization
    summary = results_df.groupby(['Case', 'Normalized'])[
        [col for col in results_df.columns if col.startswith('Sim_')]
    ].mean()
    
    print("\nMean similarity scores by case and normalization:")
    print(summary[['Sim_Jaccard', 'Sim_Dice', 'Sim_Cosine', 'Sim_Pearson']].round(4))
    
    print("\nExperiment completed. Results saved to results/similarity_results.csv")
    print("Plots saved to results/plots/ directory")


def main():
    """Main function to parse arguments and run selected examples."""
    parser = argparse.ArgumentParser(description='Run thesis examples')
    parser.add_argument('--dataset', action='store_true', help='Run Opportunity dataset example')
    parser.add_argument('--experiment', action='store_true', help='Run similarity metrics experiment')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs("results/plots", exist_ok=True)
    
    if args.dataset:
        run_opportunity_dataset_example()
    elif args.experiment:
        run_similarity_experiment()
    else:
        # If no arguments are provided, show help
        parser.print_help()


if __name__ == "__main__":
    main()
