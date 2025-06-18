# Fuzzy Similarity Metrics for Sensor Data

This repository contains the implementation and experiments for the thesis "Development and comparison of fuzzy similarity correlation metrics for sensor data in health application and assisted living environments".

## Overview

This thesis investigates fuzzy similarity metrics for sensor data analysis in health applications and assisted living environments. It focuses on developing and comparing various similarity metrics, with a special emphasis on a novel per-sensor membership function approach.

## Key Features

- **Similarity Metrics**: Implementation of various similarity metrics for sensor data
- **Per-Sensor Approach**: Novel approach using one membership function per sensor
- **Activity Classification**: Experiments on activity recognition using the Opportunity dataset
- **Visualization**: Tools for visualizing and comparing metric performance

## Repository Structure

```
thesis/
├── data/                # Data loading and preprocessing
├── exp/                 # Experiments
├── fuzzy/               # Fuzzy logic and similarity metrics
├── docs/                # Documentation
└── results/             # Experiment results
```

## Documentation

Comprehensive documentation is available in the `docs` directory:

- [Documentation Index](docs/README.md)
- [Per-Sensor Approach](docs/per_sensor_approach/README.md)
- [Similarity Metrics](docs/metrics/README.md)
- [RQ2 Experiments](docs/rq2/README.md)
- [Code Documentation](docs/code/README.md)
- [Time Series Windowing](docs/windowing/README.md)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/thesis.git
   cd thesis
   ```

2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running Experiments

#### Per-Sensor Quick Test

```
python -m thesis.exp.per_sensor_quick_test --n_samples_per_class 50 --window_size 32 --n_jobs 2
```

#### RQ2 Per-Sensor Experiment

```
python -m thesis.exp.rq2_per_sensor_experiment --max_samples 300 --window_sizes 32 --overlap_ratios 0.5 --min_samples_per_class 2 --similarity_metrics jaccard,dice,cosine --n_jobs 2
```

## Results

The per-sensor membership function approach consistently outperforms the traditional single-membership approach:

| Test | Traditional Approach (F1) | Per-Sensor Approach (F1) | Improvement |
|------|---------------------------|--------------------------|------------|
| Small-Scale | 0.3333 | 1.0000 | +0.6667 |
| Medium-Scale | 0.3750 | 0.7619 | +0.3869 |
| RQ2 Experiment | N/A | 1.0000 | N/A |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Opportunity dataset for providing the sensor data
- The fuzzy logic community for their foundational work
