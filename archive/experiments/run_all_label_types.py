"""
Run RQ2 experiments on all label types from the Opportunity dataset.

This script runs the activity classification experiments using different
label hierarchies (Locomotion, HL_Activity, ML_Both_Arms, LL_Left_Arm, LL_Right_Arm)
to compare discriminative power across different activity granularities.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from thesis.exp.rq2_experiment import RQ2Experiment, RQ2ExperimentConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_experiments_for_label_type(label_type: str, quick_test: bool = False):
    """Run RQ2 experiments for a specific label type."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiments for label type: {label_type}")
    logger.info(f"{'='*60}")
    
    # Create configuration for this label type
    config = RQ2ExperimentConfig(
        datasets=["opportunity"],  # Only Opportunity has multiple label types
        window_sizes=[128, 256] if not quick_test else [128],
        overlap_ratios=[0.5, 0.7] if not quick_test else [0.5],
        opportunity_label_type=label_type,
        quick_test=quick_test,
        output_dir=f"results/rq2_classification/label_comparison/{label_type}"
    )
    
    # Run experiment
    experiment = RQ2Experiment(config)
    experiment.run_all_experiments()
    
    return experiment


def main(quick_test: bool = False):
    """Run experiments on all label types."""
    logger.info("🚀 Starting RQ2 experiments on all Opportunity label types")
    
    # Define all label types to test
    label_types = [
        "Locomotion",      # Basic movement (Stand, Walk, Sit, Lie)
        "HL_Activity",     # High-level activities (Coffee time, Cleanup, etc.)
        "ML_Both_Arms",    # Mid-level both arms (Open Door, Close Fridge, etc.)
        "LL_Left_Arm",     # Low-level left arm (reach, open, close, etc.)
        "LL_Right_Arm",    # Low-level right arm (reach, open, close, etc.)
    ]
    
    # Run experiments for each label type
    results = {}
    for label_type in label_types:
        try:
            experiment = run_experiments_for_label_type(label_type, quick_test)
            results[label_type] = experiment
            logger.info(f"✅ Completed experiments for {label_type}")
        except Exception as e:
            logger.error(f"❌ Failed experiments for {label_type}: {e}")
            results[label_type] = None
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY OF ALL LABEL TYPE EXPERIMENTS")
    logger.info("="*60)
    
    successful = [lt for lt, exp in results.items() if exp is not None]
    failed = [lt for lt, exp in results.items() if exp is None]
    
    logger.info(f"✅ Successful: {len(successful)} - {', '.join(successful)}")
    if failed:
        logger.info(f"❌ Failed: {len(failed)} - {', '.join(failed)}")
    
    logger.info("\n🎉 All experiments completed!")
    logger.info(f"Results saved in: results/rq2_classification/label_comparison/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RQ2 experiments on all label types")
    parser.add_argument("--quick", action="store_true", help="Run quick test with reduced data")
    parser.add_argument("--label-type", type=str, help="Run only specific label type")
    args = parser.parse_args()
    
    if args.label_type:
        # Run single label type
        run_experiments_for_label_type(args.label_type, args.quick)
    else:
        # Run all label types
        main(args.quick) 