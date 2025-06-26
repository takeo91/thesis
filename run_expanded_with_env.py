#!/usr/bin/env python3
"""
Wrapper to run expanded metrics experiment with correct environment variables.
"""

import os
import sys
from pathlib import Path

def main():
    # Set the correct data path
    thesis_data_path = Path(__file__).parent.parent / "Data"
    os.environ["THESIS_DATA"] = str(thesis_data_path.absolute())
    
    print(f"🔧 Setting THESIS_DATA={os.environ['THESIS_DATA']}")
    
    # Import and run the main experiment
    from run_expanded_metrics_experiment import run_expanded_metrics_experiment
    
    success = run_expanded_metrics_experiment()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()