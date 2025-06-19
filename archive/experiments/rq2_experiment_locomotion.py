"""
RQ2 Main Experiment Controller: Discriminative Power Assessment

This module implements the main experiment controller for Research Question 2,
evaluating the discriminative power of fuzzy similarity metrics for sensor-based
activity classification using the Opportunity and PAMAP2 datasets.

Key Features:
- Automated experiment orchestration across multiple datasets
- Configurable window sizes and overlap ratios
- Statistical analysis integration
- Progress tracking and result persistence
- Publication-ready output generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
import time
from datetime import datetime
import json
import pickle
from dataclasses import dataclass, asdict
import logging

# Import thesis modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from thesis.data import (
    create_opportunity_dataset, create_pamap2_dataset,
    WindowConfig, create_multiple_window_configs
)
from thesis.exp.activity_classification import (
    ClassificationConfig, ClassificationResults,
    run_activity_classification_experiment, save_classification_results
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RQ2ExperimentConfig:
    """Configuration for RQ2 experiments."""
    datasets: List[str] = None
    window_sizes: List[int] = None
    overlap_ratios: List[float] = None
    target_activities: Dict[str, List[str]] = None
    min_samples_per_activity: int = 100
    sensor_selection: Dict[str, Dict] = None
    quick_test: bool = False
    output_dir: str = "results/rq2_classification/locomotion"
    opportunity_label_type: str = "Locomotion"  # Changed default to Locomotion
    min_samples_per_class: int = 5  # Reduced from default 10
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["opportunity", "pamap2"]
        
        if self.window_sizes is None:
            self.window_sizes = [64, 128, 256]  # Added smaller window size
        
        if self.overlap_ratios is None:
            self.overlap_ratios = [0.5, 0.7, 0.9]  # Added higher overlap ratio
        
        # Define target activities for different label types
        opportunity_activities_by_type = {
            "ML_Both_Arms": [
                "Open Door 1", "Open Door 2", "Close Door 1", "Close Door 2",
                "Open Fridge", "Close Fridge", "Toggle Switch"
            ],
            "Locomotion": [
                "Stand", "Walk", "Sit", "Lie"
            ],
            "HL_Activity": [
                "Relaxing", "Coffee time", "Early morning", "Cleanup", 
                "Sandwich time"
            ],
            "LL_Left_Arm": [
                "reach", "open", "close", "release", "move"
            ],
            "LL_Right_Arm": [
                "reach", "open", "close", "release", "move", "sip", "bite"
            ]
        }
        
        if self.target_activities is None:
            self.target_activities = {
                "opportunity": opportunity_activities_by_type.get(
                    self.opportunity_label_type, 
                    opportunity_activities_by_type["Locomotion"]  # Changed default fallback
                ),
                "pamap2": [
                    "walking", "running", "cycling", "sitting", "standing",
                    "ascending_stairs", "descending_stairs"
                ]
            }
        
        if self.sensor_selection is None:
            self.sensor_selection = {
                "opportunity": {
                    "sensor_types": ["IMU", "Accelerometer"],
                    "body_parts": ["RightLowerArm", "LeftLowerArm", "Back"],
                    "axes": ["X", "Y", "Z"]
                },
                "pamap2": {
                    "sensor_locations": ["hand", "chest", "ankle"],
                    "sensor_types": ["accelerometer", "gyroscope"],
                    "axes": ["x", "y", "z"]
                }
            }


class RQ2Experiment:
    """Main controller for RQ2 discriminative power experiments."""
    
    def __init__(self, config: RQ2ExperimentConfig):
        self.config = config
        self.results = {}
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment metadata
        self.metadata = {
            "experiment_id": f"rq2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "config": asdict(config),
            "start_time": datetime.now().isoformat()
        }
        
        logger.info(f"🚀 RQ2 Experiment initialized: {self.metadata['experiment_id']}")
    
    def load_and_preprocess_dataset(
        self, 
        dataset_name: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[int, str]]:
        """
        Load and preprocess dataset for activity classification.
        
        Args:
            dataset_name: Name of dataset ("opportunity" or "pamap2")
        
        Returns:
            Tuple of (data, labels, activity_names)
        """
        logger.info(f"📊 Loading {dataset_name} dataset...")
        
        if dataset_name.lower() == "opportunity":
            # Load Opportunity dataset
            dataset = create_opportunity_dataset()
            df = dataset.df
            
            # Select relevant sensors using MultiIndex
            idx = pd.IndexSlice
            sensor_config = self.config.sensor_selection["opportunity"]
            
            # Get sensor data using MultiIndex selection
            sensor_data_list = []
            for sensor_type in sensor_config["sensor_types"]:
                for body_part in sensor_config["body_parts"]:
                    for axis in sensor_config["axes"]:
                        try:
                            # Try to get data for this combination
                            sensor_col = df.loc[:, idx[sensor_type, body_part, :, axis]]
                            if not sensor_col.empty:
                                sensor_data_list.append(sensor_col)
                        except KeyError:
                            continue
            
            # Combine sensor data
            if sensor_data_list:
                # Concatenate all sensor columns
                sensor_df = pd.concat(sensor_data_list, axis=1)
                # Remove any duplicate columns
                sensor_df = sensor_df.loc[:, ~sensor_df.columns.duplicated()]
                data = sensor_df.values
            else:
                # Fallback: get all numeric sensor data
                sensor_mask = df.columns.get_level_values('SensorType').isin(['Accelerometer', 'Gyroscope', 'Magnetometer', 'IMU'])
                data = df.loc[:, sensor_mask].values
            
            # Get activity labels - try different label types
            # First check what label types are available
            label_types = df.columns[df.columns.get_level_values('SensorType') == 'Label']
            logger.info(f"   Available label types: {label_types.get_level_values('BodyPart').unique()}")
            
            # Use the specified label type from config
            label_type = self.config.opportunity_label_type
            try:
                labels = df.loc[:, idx["Label", label_type, "Label", "N/A"]].values
                logger.info(f"   Using {label_type} labels")
            except KeyError:
                logger.warning(f"   Label type '{label_type}' not found, trying fallbacks...")
                # Fallback order
                fallback_order = ["Locomotion", "ML_Both_Arms", "HL_Activity"]  # Changed fallback order
                for fallback_type in fallback_order:
                    try:
                        labels = df.loc[:, idx["Label", fallback_type, "Label", "N/A"]].values
                        logger.info(f"   Using fallback label type: {fallback_type}")
                        break
                    except KeyError:
                        continue
                else:
                    raise ValueError("No valid label type found in dataset")
            
            # Filter target activities
            target_activities = self.config.target_activities["opportunity"]
            
            # Handle NaN and unknown labels
            labels = np.where(pd.isna(labels), "Unknown", labels)
            
            # Log available activities
            unique_labels = np.unique(labels)
            logger.info(f"   Available activities in dataset: {unique_labels[:20]}")
            
            # Create activity mask
            activity_mask = np.isin(labels, target_activities)
            
            # Apply mask to both data and labels
            if activity_mask.any():
                # Ensure mask is 1D
                activity_mask = activity_mask.flatten()
                data = data[activity_mask]
                labels = labels[activity_mask]
                logger.info(f"   Filtered to {len(labels)} samples with target activities")
            else:
                logger.warning(f"No matching activities found for {target_activities}")
                logger.warning(f"Using all non-Unknown activities instead")
                # Use all non-unknown activities as fallback
                activity_mask = (labels != "Unknown").flatten()
                data = data[activity_mask]
                labels = labels[activity_mask]
            
        elif dataset_name.lower() == "pamap2":
            # Load PAMAP2 dataset
            dataset = create_pamap2_dataset()
            df = dataset.df
            
            # Select relevant sensors using MultiIndex
            idx = pd.IndexSlice
            sensor_config = self.config.sensor_selection["pamap2"]
            
            # Get sensor data using MultiIndex selection
            sensor_data_list = []
            for sensor_location in sensor_config["sensor_locations"]:
                for sensor_type in sensor_config["sensor_types"]:
                    for axis in sensor_config["axes"]:
                        try:
                            # Try to get data for this combination
                            sensor_col = df.loc[:, idx[sensor_location, sensor_type, axis]]
                            if not sensor_col.empty:
                                sensor_data_list.append(sensor_col)
                        except KeyError:
                            continue
            
            # Combine sensor data
            if sensor_data_list:
                # Concatenate all sensor columns
                sensor_df = pd.concat(sensor_data_list, axis=1)
                # Remove any duplicate columns
                sensor_df = sensor_df.loc[:, ~sensor_df.columns.duplicated()]
                data = sensor_df.values
            else:
                # Fallback: get all numeric sensor data
                non_label_cols = df.columns[~df.columns.get_level_values('SensorType').isin(['Label'])]
                data = df.loc[:, non_label_cols].values
            
            # Get activity labels
            try:
                labels = df.loc[:, idx["Label", "Activity", "Name", "N/A"]].values
            except KeyError:
                raise ValueError("Activity labels not found in PAMAP2 dataset")
            
            # Filter target activities
            target_activities = self.config.target_activities["pamap2"]
            
            # Handle NaN labels
            labels = np.where(pd.isna(labels), "other", labels)
            
            # Create activity mask
            activity_mask = np.isin(labels, target_activities)
            
            # Apply mask to both data and labels
            if activity_mask.any():
                # Ensure mask is 1D
                activity_mask = activity_mask.flatten()
                data = data[activity_mask]
                labels = labels[activity_mask]
                logger.info(f"   Filtered to {len(labels)} samples with target activities")
            else:
                logger.warning(f"No matching activities found for {target_activities}")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Encode labels as integers
        unique_labels = np.unique(labels)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        encoded_labels = np.array([label_mapping[label] for label in labels.flatten()])
        
        # Create inverse mapping for reporting
        inverse_mapping = {i: label for label, i in label_mapping.items()}
        
        # Log dataset summary
        logger.info(f"   ✅ Loaded {len(labels)} samples, {data.shape[1]} features")
        logger.info(f"   ✅ Activities: {len(unique_labels)} classes")
        logger.info(f"   ✅ Label encoding: {label_mapping}")
        
        # Log class distribution
        for label, idx in label_mapping.items():
            count = np.sum(encoded_labels == idx)
            logger.info(f"       - {label}: {count} samples")
        
        return data, encoded_labels, unique_labels, inverse_mapping
    
    def run_dataset_experiments(
        self, 
        dataset_name: str
    ) -> Dict[str, ClassificationResults]:
        """
        Run experiments for a single dataset.
        
        Args:
            dataset_name: Name of dataset to run experiments on
        
        Returns:
            Dictionary of experiment results
        """
        logger.info("============================================================")
        logger.info(f"🔬 Running experiments on {dataset_name.upper()} dataset")
        logger.info("============================================================")
        
        # Load and preprocess dataset
        data, labels, activity_names, label_mapping = self.load_and_preprocess_dataset(dataset_name)
        
        # Create classification configuration
        classification_config = ClassificationConfig(
            window_sizes=self.config.window_sizes,
            overlap_ratios=self.config.overlap_ratios,
            min_samples_per_class=self.config.min_samples_per_class,  # Use the reduced min_samples_per_class
            similarity_normalization=True,
            ndg_kernel_type="epanechnikov",
            ndg_sigma_method="adaptive"
        )
        
        # Run classification experiments
        experiment_name = f"rq2_{dataset_name.lower()}"
        results_list = run_activity_classification_experiment(
            data=data,
            labels=labels,
            config=classification_config,
            experiment_name=experiment_name
        )
        
        # Save results
        if results_list:
            output_dir = self.output_dir / dataset_name.lower()
            output_dir.mkdir(parents=True, exist_ok=True)
            save_classification_results(results_list, output_dir)
            logger.info(f"💾 Results saved to {output_dir}")
        else:
            logger.warning("⚠️ No valid results to save")
        
        # Convert to dictionary for easier access
        results_dict = {
            f"{r.window_config.window_size}_{r.window_config.overlap_ratio}": r
            for r in results_list
        }
        
        return results_dict
    
    def run_all_experiments(self) -> None:
        """Run experiments for all configured datasets."""
        logger.info("\n🚀 Starting RQ2 Experiments")
        logger.info(f"   Datasets: {self.config.datasets}")
        logger.info(f"   Window sizes: {self.config.window_sizes}")
        logger.info(f"   Overlap ratios: {self.config.overlap_ratios}")
        logger.info("")
        
        start_time = time.time()
        
        # Run experiments for each dataset
        for dataset_name in self.config.datasets:
            try:
                self.results[dataset_name] = self.run_dataset_experiments(dataset_name)
            except Exception as e:
                logger.error(f"❌ Error in {dataset_name} experiments: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Update metadata with end time and duration
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["total_duration_seconds"] = time.time() - start_time
        
        # Save results and create summary report
        self.save_results()
        self.create_summary_report()
    
    def save_results(self) -> None:
        """Save experiment metadata and results."""
        # Save metadata
        metadata_path = self.output_dir / "rq2_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save all results in a single file
        all_results_path = self.output_dir / "rq2_all_results.pkl"
        with open(all_results_path, "wb") as f:
            pickle.dump(self.results, f)
        
        logger.info(f"💾 Experiment metadata saved to {metadata_path}")
        logger.info(f"💾 All results saved to {all_results_path}")
    
    def create_summary_report(self) -> None:
        """Create comprehensive summary report of all experiments."""
        # Create summary DataFrame
        summary_rows = []
        
        for dataset_name, dataset_results in self.results.items():
            for config_key, result in dataset_results.items():
                window_size = result.window_config.window_size
                overlap_ratio = result.window_config.overlap_ratio
                
                # Extract performance metrics for each similarity metric
                for metric_name, metrics in result.performance_metrics.items():
                    row = {
                        "dataset": dataset_name,
                        "window_size": window_size,
                        "overlap_ratio": overlap_ratio,
                        "metric_name": metric_name,
                        "macro_f1": metrics.get("macro_f1", 0),
                        "balanced_accuracy": metrics.get("balanced_accuracy", 0),
                        "accuracy": metrics.get("accuracy", 0),
                        "n_windows": result.windowed_data.n_windows,
                        "n_classes": len(np.unique(result.windowed_data.labels)),
                        "computation_time": result.computation_time.get(metric_name, 0)
                    }
                    summary_rows.append(row)
        
        if not summary_rows:
            logger.warning("⚠️ No results to summarize")
            return
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Save comprehensive summary as CSV
        summary_path = self.output_dir / "rq2_comprehensive_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"📊 Comprehensive summary saved to {summary_path}")
        
        # Run statistical analysis
        try:
            from thesis.exp.rq2_statistical_analysis import run_statistical_analysis
            stat_results = run_statistical_analysis(summary_df, self.output_dir)
            
            # Save statistical results
            stat_path = self.output_dir / "rq2_statistical_analysis.pkl"
            with open(stat_path, "wb") as f:
                pickle.dump(stat_results, f)
            logger.info(f"📊 Statistical analysis saved to {stat_path}")
            
        except Exception as e:
            logger.error(f"❌ Error in statistical analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Create markdown report
        self.create_markdown_report(summary_df)
    
    def create_markdown_report(self, summary_df: pd.DataFrame) -> None:
        """Create a markdown summary report."""
        report_path = self.output_dir / "rq2_summary_report.md"
        
        with open(report_path, "w") as f:
            f.write("# RQ2 Experiment Summary Report\n\n")
            f.write(f"**Experiment ID**: {self.metadata['experiment_id']}\n")
            f.write(f"**Date**: {self.metadata['start_time']}\n")
            f.write(f"**Duration**: {self.metadata.get('total_duration_seconds', 0):.1f} seconds\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Datasets**: {', '.join(self.config.datasets)}\n")
            f.write(f"- **Window sizes**: {self.config.window_sizes}\n")
            f.write(f"- **Overlap ratios**: {self.config.overlap_ratios}\n")
            f.write(f"- **Total experiments**: {len(summary_df)}\n\n")
            
            if len(summary_df) == 0:
                f.write("## No Results\n\n")
                f.write("No experiments completed successfully. Check the logs for errors.\n")
                logger.warning("No results to report - all experiments failed")
                return
            
            f.write("## Top Performing Metrics\n\n")
            
            # Top metrics overall
            top_overall = summary_df.nlargest(10, 'macro_f1')
            f.write("### Overall Top 10 Metrics\n\n")
            f.write("| Dataset | Window | Overlap | Metric | Macro F1 | Balanced Acc |\n")
            f.write("|---------|--------|---------|--------|----------|-------------|\n")
            
            for _, row in top_overall.iterrows():
                f.write(f"| {row['dataset']} | {row['window_size']} | {row['overlap_ratio']:.1f} | "
                       f"{row['metric_name']} | {row['macro_f1']:.3f} | {row['balanced_accuracy']:.3f} |\n")
            
            # Top metrics per dataset
            for dataset in self.config.datasets:
                dataset_df = summary_df[summary_df['dataset'] == dataset]
                if len(dataset_df) > 0:
                    f.write(f"\n### Top 5 Metrics - {dataset.upper()}\n\n")
                    top_dataset = dataset_df.nlargest(5, 'macro_f1')
                    
                    f.write("| Window | Overlap | Metric | Macro F1 | Balanced Acc |\n")
                    f.write("|--------|---------|--------|----------|-------------|\n")
                    
                    for _, row in top_dataset.iterrows():
                        f.write(f"| {row['window_size']} | {row['overlap_ratio']:.1f} | "
                               f"{row['metric_name']} | {row['macro_f1']:.3f} | {row['balanced_accuracy']:.3f} |\n")
            
            f.write("\n## Metric Performance Summary\n\n")
            
            # Average performance per metric across all configurations
            metric_avg = summary_df.groupby('metric_name')['macro_f1'].agg(['mean', 'std', 'count'])
            metric_avg = metric_avg.sort_values('mean', ascending=False).head(15)
            
            f.write("### Top 15 Metrics by Average Performance\n\n")
            f.write("| Metric | Mean F1 | Std Dev | Count |\n")
            f.write("|--------|---------|---------|-------|\n")
            
            for metric_name, row in metric_avg.iterrows():
                f.write(f"| {metric_name} | {row['mean']:.3f} | {row['std']:.3f} | {int(row['count'])} |\n")
                
            # Add statistical analysis summary if available
            stat_report_path = self.output_dir / "rq2_statistical_report.md"
            if stat_report_path.exists():
                f.write("\n## Statistical Analysis Summary\n\n")
                f.write("A comprehensive statistical analysis has been performed to evaluate the discriminative power\n")
                f.write("of similarity metrics. For full details, see [Statistical Report](rq2_statistical_report.md).\n\n")
                
                # Read key findings from statistical report
                try:
                    with open(stat_report_path, "r") as stat_f:
                        lines = stat_f.readlines()
                        
                        # Extract Friedman test results
                        friedman_section = False
                        friedman_lines = []
                        
                        for line in lines:
                            if "Overall Statistical Significance" in line:
                                friedman_section = True
                                continue
                            elif friedman_section and line.strip() == "":
                                break
                            elif friedman_section:
                                friedman_lines.append(line.strip())
                        
                        if friedman_lines:
                            f.write("### Key Statistical Findings\n\n")
                            for line in friedman_lines:
                                f.write(f"{line}\n")
                            f.write("\n")
                    
                        # Extract top metrics from statistical analysis
                        f.write("### Statistically Significant Top Metrics\n\n")
                        f.write("The following metrics showed statistically significant discriminative power:\n\n")
                        
                        rankings_found = False
                        for i, line in enumerate(lines):
                            if "Top 10 Similarity Metrics by Discriminative Power" in line:
                                # Extract the table header and first few rows
                                if i+3 < len(lines) and "---" in lines[i+2]:
                                    f.write(lines[i+1])  # Table header
                                    f.write(lines[i+2])  # Table separator
                                    
                                    # Add top 5 rows
                                    for j in range(i+3, min(i+8, len(lines))):
                                        if lines[j].strip():
                                            f.write(lines[j])
                                    
                                    rankings_found = True
                                    break
                                    
                        if not rankings_found:
                            f.write("See the full statistical report for details.\n")
                            
                except Exception as e:
                    f.write("See the full statistical report for details.\n")
                
                # Add links to visualizations
                f.write("\n### Visualizations\n\n")
                f.write("The following visualizations have been generated:\n\n")
                f.write("- [Metric Performance Heatmap](rq2_performance_heatmap.png)\n")
                f.write("- [Statistical Significance Matrix](rq2_statistical_significance.png)\n")
                f.write("- [Activity-Specific Performance](rq2_activity_performance.png)\n")
            
            # Add conclusion
            f.write("\n## Conclusion\n\n")
            best_metric = metric_avg.index[0]
            best_f1 = metric_avg.iloc[0]['mean']
            
            f.write(f"The analysis demonstrates that **{best_metric}** is the top-performing similarity metric ")
            f.write(f"with an average macro F1 score of {best_f1:.3f} across all experimental configurations. ")
            
            # Add conclusion based on statistical analysis
            stat_analysis_path = self.output_dir / "rq2_statistical_analysis.pkl"
            if stat_analysis_path.exists():
                try:
                    with open(stat_analysis_path, "rb") as f_stat:
                        stat_results = pickle.load(f_stat)
                        
                    if stat_results['friedman_result']['reject_h0']:
                        f.write("Statistical analysis confirms significant differences in discriminative power ")
                        f.write("between similarity metrics (p < 0.05), supporting hypothesis H2.\n")
                    else:
                        f.write("Statistical analysis does not show significant differences between all metrics, ")
                        f.write("suggesting that multiple similarity metrics may be suitable for activity classification.\n")
                except Exception:
                    f.write("See the statistical report for detailed analysis of the results.\n")
            else:
                f.write("See the full report for detailed analysis of the results.\n")
        
        logger.info(f"📝 Markdown report saved to {report_path}")


def main():
    """Main entry point for RQ2 experiments."""
    # Create experiment configuration
    config = RQ2ExperimentConfig(
        datasets=["opportunity"],  # Focus only on Opportunity
        window_sizes=[64, 128, 256],  # Added smaller window size
        overlap_ratios=[0.5, 0.7, 0.9],  # Added higher overlap ratio
        opportunity_label_type="Locomotion",  # Use Locomotion labels
        min_samples_per_class=5,  # Reduced from default 10
        quick_test=False  # Run full experiment
    )
    
    # Create and run experiment
    experiment = RQ2Experiment(config)
    experiment.run_all_experiments()


def quick_test():
    """Quick test with reduced data for debugging."""
    config = RQ2ExperimentConfig(
        datasets=["opportunity"],  # Just one dataset
        window_sizes=[64, 128],  # Added smaller window size
        overlap_ratios=[0.5, 0.9],  # Test extreme overlap values
        opportunity_label_type="Locomotion",  # Use Locomotion labels
        min_samples_per_class=5,  # Reduced minimum samples
        quick_test=True  # Use subset of data
    )
    
    experiment = RQ2Experiment(config)
    experiment.run_all_experiments()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RQ2 experiments")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        main()