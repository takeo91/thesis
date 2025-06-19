import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_nemenyi_friedman


def perform_rq2_statistical_analysis(results_df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive statistical analysis for RQ2.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing experimental results with columns:
        - 'metric_name': Name of the similarity metric
        - 'dataset': Dataset name (e.g., 'opportunity', 'pamap2')
        - 'window_size': Window size in samples
        - 'overlap_ratio' or 'overlap': Overlap fraction
        - 'macro_f1': Macro-averaged F1 score
        - 'balanced_accuracy': Balanced accuracy
        - 'per_class_f1': Dictionary of per-class F1 scores
    
    Returns:
    --------
    Dict with statistical analysis results:
        - 'friedman_result': Friedman test results
        - 'nemenyi_result': Nemenyi post-hoc test results
        - 'metric_rankings': Ranked metrics DataFrame
        - 'effect_sizes': Effect size calculations
        - 'confidence_intervals': Confidence intervals
    """
    # Handle different column naming conventions
    if 'overlap_ratio' in results_df.columns and 'overlap' not in results_df.columns:
        results_df = results_df.rename(columns={'overlap_ratio': 'overlap'})
    
    # Ensure required columns exist
    required_columns = ['dataset', 'window_size', 'overlap', 'metric_name', 'macro_f1']
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Prepare data for analysis
    # Pivot to get metrics as columns and experiments as rows
    pivot_df = results_df.pivot_table(
        index=['dataset', 'window_size', 'overlap'],
        columns='metric_name',
        values='macro_f1'
    ).reset_index()
    
    # Extract just the metric columns for the Friedman test
    metric_columns = [col for col in pivot_df.columns 
                      if col not in ['dataset', 'window_size', 'overlap']]
    
    # Check if we have enough data for statistical analysis
    if len(pivot_df) < 2 or len(metric_columns) < 2:
        # Not enough data for statistical tests
        dummy_result = {
            'statistic': 0.0,
            'p_value': 1.0,
            'reject_h0': False,
            'note': "Insufficient data for statistical analysis"
        }
        
        # Create a simple ranking based on mean performance
        metric_means = results_df.groupby('metric_name')['macro_f1'].mean().reset_index()
        metric_means['rank'] = range(1, len(metric_means) + 1)
        
        return {
            'friedman_result': dummy_result,
            'nemenyi_result': pd.DataFrame(),
            'metric_rankings': metric_means,
            'effect_sizes': pd.DataFrame(),
            'confidence_intervals': pd.DataFrame(),
            'activity_analysis': {'note': "Insufficient data"}
        }
    
    # 1. Friedman test for overall comparison
    friedman_result = friedman_test(pivot_df[metric_columns])
    
    # 2. Nemenyi post-hoc test for pairwise comparisons
    nemenyi_result = nemenyi_test(pivot_df[metric_columns])
    
    # 3. Calculate effect sizes
    effect_sizes = calculate_effect_sizes(results_df)
    
    # 4. Calculate confidence intervals
    confidence_intervals = calculate_confidence_intervals(results_df)
    
    # 5. Activity-specific analysis
    activity_analysis = analyze_per_activity_performance(results_df)
    
    # 6. Rank metrics
    metric_rankings = rank_similarity_metrics(results_df)
    
    return {
        'friedman_result': friedman_result,
        'nemenyi_result': nemenyi_result,
        'metric_rankings': metric_rankings,
        'effect_sizes': effect_sizes,
        'confidence_intervals': confidence_intervals,
        'activity_analysis': activity_analysis
    }


def friedman_test(data: pd.DataFrame) -> Dict:
    """
    Perform Friedman test on the data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with metrics as columns
    
    Returns:
    --------
    Dict with test results:
        - 'statistic': Friedman test statistic
        - 'p_value': p-value
        - 'reject_h0': Whether to reject null hypothesis (p < 0.05)
    """
    # Convert to numpy array
    data_array = data.values
    
    # Perform Friedman test
    statistic, p_value = stats.friedmanchisquare(*[data_array[:, i] for i in range(data_array.shape[1])])
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'reject_h0': p_value < 0.05
    }


def nemenyi_test(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform Nemenyi post-hoc test following Friedman test.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with metrics as columns
    
    Returns:
    --------
    pd.DataFrame
        Pairwise p-values matrix
    """
    # Convert to numpy array
    data_array = data.values
    
    # Perform Nemenyi test
    posthoc_matrix = posthoc_nemenyi_friedman(data_array)
    
    # Convert to DataFrame with metric names
    posthoc_df = pd.DataFrame(
        posthoc_matrix,
        index=data.columns,
        columns=data.columns
    )
    
    return posthoc_df


def calculate_effect_sizes(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate effect sizes (Cohen's d, Cliff's delta) for pairwise metric comparisons.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing experimental results
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with effect sizes for each metric pair
    """
    metrics = results_df['metric_name'].unique()
    effect_sizes_data = []
    
    for i, metric1 in enumerate(metrics):
        for metric2 in enumerate(metrics[i+1:], i+1):
            metric2_name = metric2[1]
            
            # Get data for both metrics
            data1 = results_df[results_df['metric_name'] == metric1]['macro_f1'].values
            data2 = results_df[results_df['metric_name'] == metric2_name]['macro_f1'].values
            
            # Calculate Cohen's d
            cohens_d = calculate_cohens_d(data1, data2)
            
            # Calculate Cliff's delta
            cliffs_delta = calculate_cliffs_delta(data1, data2)
            
            effect_sizes_data.append({
                'metric1': metric1,
                'metric2': metric2_name,
                'cohens_d': cohens_d,
                'cliffs_delta': cliffs_delta,
                'effect_magnitude': interpret_effect_size(cohens_d)
            })
    
    return pd.DataFrame(effect_sizes_data)


def calculate_cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Parameters:
    -----------
    data1, data2 : np.ndarray
        Arrays of values to compare
    
    Returns:
    --------
    float
        Cohen's d effect size
    """
    # Calculate means
    mean1, mean2 = np.mean(data1), np.mean(data2)
    
    # Calculate pooled standard deviation
    n1, n2 = len(data1), len(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    return d


def calculate_cliffs_delta(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate Cliff's delta effect size (non-parametric).
    
    Parameters:
    -----------
    data1, data2 : np.ndarray
        Arrays of values to compare
    
    Returns:
    --------
    float
        Cliff's delta effect size
    """
    # Count dominance
    dominance = 0
    for x in data1:
        for y in data2:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1
    
    # Normalize
    return dominance / (len(data1) * len(data2))


def interpret_effect_size(d: float) -> str:
    """
    Interpret the magnitude of Cohen's d effect size.
    
    Parameters:
    -----------
    d : float
        Cohen's d value
    
    Returns:
    --------
    str
        Interpretation of effect size magnitude
    """
    d = abs(d)  # Use absolute value for interpretation
    
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def calculate_confidence_intervals(results_df: pd.DataFrame, 
                                  confidence: float = 0.95) -> pd.DataFrame:
    """
    Calculate confidence intervals for each metric.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing experimental results
    confidence : float
        Confidence level (default: 0.95 for 95% CI)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with confidence intervals for each metric
    """
    metrics = results_df['metric_name'].unique()
    ci_data = []
    
    for metric in metrics:
        # Get data for this metric
        data = results_df[results_df['metric_name'] == metric]['macro_f1'].values
        
        # Calculate mean and standard error
        mean = np.mean(data)
        se = stats.sem(data)
        
        # Calculate confidence interval
        ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=se)
        
        ci_data.append({
            'metric_name': metric,
            'mean': mean,
            'lower_ci': ci[0],
            'upper_ci': ci[1],
            'ci_width': ci[1] - ci[0]
        })
    
    return pd.DataFrame(ci_data)


def analyze_per_activity_performance(results_df: pd.DataFrame) -> Dict:
    """
    Analyze performance of metrics per activity class.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing experimental results with per_class_f1 column
        
    Returns:
    --------
    Dict
        Dictionary with activity-specific analysis
    """
    # Check if per_class_f1 column exists
    if 'per_class_f1' not in results_df.columns:
        return {"error": "No per-class F1 scores found in the data"}
    
    # First, extract and normalize the per-class F1 scores
    activities = set()
    valid_rows = 0
    
    for _, row in results_df.iterrows():
        if isinstance(row.get('per_class_f1'), dict) and row['per_class_f1']:
            activities.update(row['per_class_f1'].keys())
            valid_rows += 1
    
    if not activities or valid_rows == 0:
        return {"error": "No valid per-class F1 scores found in the data"}
    
    activities = sorted(list(activities))
    
    # Create a DataFrame for activity-specific analysis
    activity_data = []
    
    for _, row in results_df.iterrows():
        if not isinstance(row.get('per_class_f1'), dict) or not row['per_class_f1']:
            continue
            
        metric = row['metric_name']
        dataset = row['dataset']
        window_size = row['window_size']
        overlap = row.get('overlap', row.get('overlap_ratio', 0.0))
        
        for activity, f1 in row['per_class_f1'].items():
            activity_data.append({
                'metric_name': metric,
                'dataset': dataset,
                'window_size': window_size,
                'overlap': overlap,
                'activity': activity,
                'f1_score': f1
            })
    
    if not activity_data:
        return {"error": "No per-class F1 scores found in the data"}
    
    activity_df = pd.DataFrame(activity_data)
    
    # Calculate best metric per activity
    best_metrics_per_activity = activity_df.groupby('activity')['f1_score'].mean().reset_index()
    best_metrics_per_activity = best_metrics_per_activity.sort_values('f1_score', ascending=False)
    
    # Calculate activity difficulty (average F1 across all metrics)
    activity_difficulty = activity_df.groupby('activity')['f1_score'].mean().sort_values()
    
    return {
        'activity_df': activity_df,
        'best_metrics_per_activity': best_metrics_per_activity,
        'activity_difficulty': activity_difficulty
    }


def rank_similarity_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank similarity metrics by discriminative power.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing experimental results
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with ranked metrics
    """
    # Calculate average performance metrics per similarity metric
    metric_performance = results_df.groupby('metric_name').agg({
        'macro_f1': ['mean', 'std', 'min', 'max'],
        'balanced_accuracy': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Flatten multi-level columns
    metric_performance.columns = [
        '_'.join(col).strip('_') for col in metric_performance.columns.values
    ]
    
    # Calculate consistency across datasets
    dataset_consistency = calculate_dataset_consistency(results_df)
    
    # Calculate configuration consistency
    config_consistency = calculate_configuration_consistency(results_df)
    
    # Merge all metrics
    metric_rankings = pd.merge(
        metric_performance, 
        dataset_consistency,
        on='metric_name'
    )
    
    metric_rankings = pd.merge(
        metric_rankings,
        config_consistency,
        on='metric_name'
    )
    
    # Rank metrics by macro_f1_mean
    metric_rankings = metric_rankings.sort_values('macro_f1_mean', ascending=False)
    
    # Add rank column
    metric_rankings['rank'] = range(1, len(metric_rankings) + 1)
    
    return metric_rankings


def calculate_dataset_consistency(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate consistency of metrics across datasets.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing experimental results
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with dataset consistency metrics
    """
    # Calculate average performance per metric and dataset
    dataset_perf = results_df.groupby(['metric_name', 'dataset'])['macro_f1'].mean().reset_index()
    
    # Pivot to get datasets as columns
    dataset_pivot = dataset_perf.pivot(index='metric_name', columns='dataset', values='macro_f1')
    
    # Calculate standard deviation across datasets
    dataset_pivot['dataset_std'] = dataset_pivot.std(axis=1)
    dataset_pivot['dataset_range'] = dataset_pivot.max(axis=1) - dataset_pivot.min(axis=1)
    
    # Calculate correlation between dataset rankings
    datasets = results_df['dataset'].unique()
    if len(datasets) >= 2:
        # Calculate Spearman correlation between dataset rankings
        dataset_rankings = {}
        for dataset in datasets:
            dataset_data = results_df[results_df['dataset'] == dataset]
            avg_perf = dataset_data.groupby('metric_name')['macro_f1'].mean()
            dataset_rankings[dataset] = avg_perf.rank(ascending=False)
        
        # Calculate average Spearman correlation
        spearman_corrs = []
        for i, dataset1 in enumerate(datasets):
            for dataset2 in datasets[i+1:]:
                corr, _ = stats.spearmanr(dataset_rankings[dataset1], dataset_rankings[dataset2])
                spearman_corrs.append(corr)
        
        dataset_pivot['dataset_rank_correlation'] = np.mean(spearman_corrs)
    else:
        dataset_pivot['dataset_rank_correlation'] = 1.0  # Only one dataset
    
    # Reset index to get metric_name as a column
    result = dataset_pivot[['dataset_std', 'dataset_range', 'dataset_rank_correlation']].reset_index()
    
    return result


def calculate_configuration_consistency(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate consistency of metrics across window sizes and overlaps.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing experimental results
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with configuration consistency metrics
    """
    # Group by metric and configuration
    config_perf = results_df.groupby(['metric_name', 'window_size', 'overlap'])['macro_f1'].mean().reset_index()
    
    # Calculate statistics per metric
    config_stats = config_perf.groupby('metric_name').agg({
        'macro_f1': ['std', lambda x: max(x) - min(x)]
    }).reset_index()
    
    # Flatten multi-level columns
    config_stats.columns = ['metric_name', 'config_std', 'config_range']
    
    return config_stats


def plot_metric_performance_heatmap(results: Dict, output_path: str = None) -> None:
    """
    Generate a heatmap visualization of metric performance.
    
    Parameters:
    -----------
    results : Dict
        Results from perform_rq2_statistical_analysis
    output_path : str, optional
        Path to save the plot
    """
    # Check if metric_rankings exists and is valid
    if 'metric_rankings' not in results or not isinstance(results['metric_rankings'], pd.DataFrame) or results['metric_rankings'].empty:
        print("Error: No valid metric rankings available")
        # Create a simple error plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Error: No valid metric rankings available", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return
    
    # Get metric rankings
    metric_rankings = results['metric_rankings']
    
    # Select top 15 metrics for better visualization (or all if less than 15)
    top_count = min(15, len(metric_rankings))
    top_metrics = metric_rankings.head(top_count)
    
    # Check for required columns
    required_columns = ['metric_name']
    heatmap_columns = []
    
    # Check for performance columns
    if 'macro_f1_mean' in top_metrics.columns:
        heatmap_columns.append('macro_f1_mean')
    elif 'mean' in top_metrics.columns:
        heatmap_columns.append('mean')
    
    if 'balanced_accuracy_mean' in top_metrics.columns:
        heatmap_columns.append('balanced_accuracy_mean')
    
    if 'dataset_std' in top_metrics.columns:
        heatmap_columns.append('dataset_std')
    
    if 'config_std' in top_metrics.columns:
        heatmap_columns.append('config_std')
    
    if not heatmap_columns:
        print("Error: No performance metrics available for heatmap")
        # Create a simple error plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Error: No performance metrics available for heatmap", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap data
    try:
        heatmap_data = top_metrics[heatmap_columns].set_index(top_metrics['metric_name'])
        
        # Rename columns for better readability
        column_mapping = {
            'macro_f1_mean': 'Macro F1',
            'mean': 'Mean F1',
            'balanced_accuracy_mean': 'Balanced Acc',
            'dataset_std': 'Dataset Var',
            'config_std': 'Config Var'
        }
        
        heatmap_data.columns = [column_mapping.get(col, col) for col in heatmap_data.columns]
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f', 
                    linewidths=.5, cbar_kws={'label': 'Value'})
        
        plt.title(f'Top {top_count} Similarity Metrics Performance')
        plt.tight_layout()
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        plt.clf()
        plt.text(0.5, 0.5, f"Error creating heatmap: {e}", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_statistical_significance(results: Dict, output_path: str = None) -> None:
    """
    Generate a visualization of statistical significance between metrics.
    
    Parameters:
    -----------
    results : Dict
        Results from perform_rq2_statistical_analysis
    output_path : str, optional
        Path to save the plot
    """
    # Check if nemenyi_result exists and is valid
    if 'nemenyi_result' not in results or not isinstance(results['nemenyi_result'], pd.DataFrame) or results['nemenyi_result'].empty:
        print("Error: No valid Nemenyi test results available")
        # Create a simple error plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Error: No valid Nemenyi test results available", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return
    
    # Get Nemenyi test results
    nemenyi_result = results['nemenyi_result']
    
    # Get top metrics
    try:
        if 'metric_rankings' in results and not results['metric_rankings'].empty:
            top_count = min(15, len(results['metric_rankings']))
            top_metrics = results['metric_rankings'].head(top_count)['metric_name'].tolist()
        else:
            # Use all metrics from Nemenyi result
            top_metrics = nemenyi_result.index.tolist()
            top_count = min(15, len(top_metrics))
            top_metrics = top_metrics[:top_count]
    except Exception as e:
        print(f"Error getting top metrics: {e}")
        # Create a simple error plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error getting top metrics: {e}", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return
    
    # Filter Nemenyi results for top metrics
    try:
        filtered_nemenyi = nemenyi_result.loc[top_metrics, top_metrics]
        
        # Create significance heatmap
        plt.figure(figsize=(12, 10))
        
        # Create mask for non-significant differences (p >= 0.05)
        mask = filtered_nemenyi >= 0.05
        
        # Create heatmap
        sns.heatmap(filtered_nemenyi, annot=True, cmap='coolwarm_r', fmt='.3f',
                    linewidths=.5, mask=mask, cbar_kws={'label': 'p-value'})
        
        plt.title(f'Statistical Significance Between Top {len(top_metrics)} Metrics (p < 0.05)')
        plt.tight_layout()
        
    except Exception as e:
        print(f"Error creating significance heatmap: {e}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error creating significance heatmap: {e}", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_activity_performance(results: Dict, output_path: str = None) -> None:
    """
    Generate a visualization of per-activity performance.
    
    Parameters:
    -----------
    results : Dict
        Results from perform_rq2_statistical_analysis
    output_path : str, optional
        Path to save the plot
    """
    # Get activity analysis results
    activity_analysis = results.get('activity_analysis', {})
    
    if not activity_analysis or 'error' in activity_analysis or 'note' in activity_analysis:
        error_msg = activity_analysis.get('error', activity_analysis.get('note', "No activity data available"))
        print(f"Error: {error_msg}")
        # Create a simple error plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error: {error_msg}", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return
    
    # Check if activity_df exists in the analysis
    if 'activity_df' not in activity_analysis:
        print("Error: No activity_df in activity analysis")
        # Create a simple error plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Error: No activity data available", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return
    
    activity_df = activity_analysis['activity_df']
    
    # Check if we have data to plot
    if activity_df.empty:
        print("Error: No activity data available")
        # Create a simple error plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Error: No activity data available", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return
    
    # Get top 5 metrics
    try:
        top_metrics = results['metric_rankings'].head(5)['metric_name'].tolist()
    except (KeyError, AttributeError):
        # If metric_rankings is not available, use the top 5 from activity_df
        top_metrics = activity_df.groupby('metric_name')['f1_score'].mean().nlargest(5).index.tolist()
    
    # Filter for top metrics
    filtered_activity = activity_df[activity_df['metric_name'].isin(top_metrics)]
    
    if filtered_activity.empty:
        print("Error: No data for top metrics")
        # Create a simple error plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Error: No data for top metrics", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return
    
    # Create grouped bar chart
    plt.figure(figsize=(14, 8))
    
    # Plot
    sns.barplot(x='activity', y='f1_score', hue='metric_name', data=filtered_activity)
    
    plt.title('Per-Activity Performance of Top 5 Metrics')
    plt.xlabel('Activity')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def generate_summary_report(results: Dict, output_path: str) -> None:
    """
    Generate a comprehensive summary report of the statistical analysis.
    
    Parameters:
    -----------
    results : Dict
        Results from perform_rq2_statistical_analysis
    output_path : str
        Path to save the report
    """
    # Check if we have valid results
    if not results or not isinstance(results, dict):
        with open(output_path, "w") as f:
            f.write("# RQ2 Statistical Analysis Summary Report\n\n")
            f.write("**Error**: No valid statistical results available.\n")
        return
    
    # Extract results
    friedman_result = results.get('friedman_result', {})
    metric_rankings = results.get('metric_rankings', pd.DataFrame())
    
    # Create report
    report = [
        "# RQ2 Statistical Analysis Summary Report\n\n",
    ]
    
    # Check if we have enough data for statistical analysis
    if 'note' in friedman_result and friedman_result['note'] == "Insufficient data for statistical analysis":
        report.append("## Limited Data Available\n\n")
        report.append("There was insufficient data for a comprehensive statistical analysis. ")
        report.append("This could be due to limited experimental configurations or a small number of metrics.\n\n")
        
        # Add basic metric rankings if available
        if not metric_rankings.empty and 'metric_name' in metric_rankings.columns:
            report.append("## Metric Rankings (Limited Data)\n\n")
            
            if 'mean' in metric_rankings.columns:
                top_metrics = metric_rankings.sort_values('mean', ascending=False).head(10)
                report.append("| Rank | Metric | Mean F1 |\n")
                report.append("|------|--------|--------|\n")
                
                for i, (_, row) in enumerate(top_metrics.iterrows(), 1):
                    report.append(f"| {i} | {row['metric_name']} | {row['mean']:.4f} |\n")
            else:
                # Just list the metrics
                report.append("Available metrics (no performance data):\n\n")
                for i, metric in enumerate(metric_rankings['metric_name'].tolist(), 1):
                    report.append(f"{i}. {metric}\n")
        
        with open(output_path, "w") as f:
            f.write("".join(report))
        return
    
    # Standard report with statistical analysis
    report.extend([
        "## Overall Statistical Significance\n\n",
        f"Friedman Test Statistic: {friedman_result.get('statistic', 'N/A'):.2f}\n",
        f"p-value: {friedman_result.get('p_value', 'N/A'):.6f}\n",
        f"Null Hypothesis Rejected: {friedman_result.get('reject_h0', 'N/A')}\n\n",
        "## Top 10 Similarity Metrics by Discriminative Power\n\n"
    ])
    
    # Add top 10 metrics table
    if not metric_rankings.empty and len(metric_rankings) > 0:
        top_10 = metric_rankings.head(min(10, len(metric_rankings)))
        
        # Check for required columns
        if 'metric_name' in top_10.columns:
            # Determine which columns to use
            f1_col = 'macro_f1_mean' if 'macro_f1_mean' in top_10.columns else 'mean' if 'mean' in top_10.columns else None
            acc_col = 'balanced_accuracy_mean' if 'balanced_accuracy_mean' in top_10.columns else None
            consistency_col = 'dataset_std' if 'dataset_std' in top_10.columns else None
            
            # Create header based on available columns
            header = "| Rank | Metric |"
            separator = "|------|--------|"
            
            if f1_col:
                header += " Macro F1 |"
                separator += "----------|"
            
            if acc_col:
                header += " Balanced Accuracy |"
                separator += "------------------|"
            
            if consistency_col:
                header += " Dataset Consistency |"
                separator += "---------------------|"
            
            report.append(header + "\n")
            report.append(separator + "\n")
            
            for i, row in enumerate(top_10.itertuples(), 1):
                line = f"| {i} | {getattr(row, 'metric_name')} |"
                
                if f1_col:
                    f1_value = getattr(row, f1_col) if hasattr(row, f1_col) else 'N/A'
                    if isinstance(f1_value, (int, float)):
                        line += f" {f1_value:.4f} |"
                    else:
                        line += f" {f1_value} |"
                
                if acc_col:
                    acc_value = getattr(row, acc_col) if hasattr(row, acc_col) else 'N/A'
                    if isinstance(acc_value, (int, float)):
                        line += f" {acc_value:.4f} |"
                    else:
                        line += f" {acc_value} |"
                
                if consistency_col:
                    consistency_value = getattr(row, consistency_col) if hasattr(row, consistency_col) else 'N/A'
                    if isinstance(consistency_value, (int, float)):
                        line += f" {consistency_value:.4f} |"
                    else:
                        line += f" {consistency_value} |"
                
                report.append(line + "\n")
        else:
            report.append("No metric ranking data available.\n")
    else:
        report.append("No metric ranking data available.\n")
    
    report.append("\n## Effect Size Analysis\n\n")
    
    # Add effect sizes for top 5 metrics
    effect_sizes = results.get('effect_sizes', pd.DataFrame())
    
    if not effect_sizes.empty and len(effect_sizes) > 0:
        # Get top 5 metrics
        if not metric_rankings.empty and 'metric_name' in metric_rankings.columns:
            top_5_metrics = metric_rankings.head(min(5, len(metric_rankings)))['metric_name'].tolist()
            
            # Filter effect sizes for top 5 metrics
            top_effect_sizes = effect_sizes[
                (effect_sizes['metric1'].isin(top_5_metrics)) & 
                (effect_sizes['metric2'].isin(top_5_metrics))
            ]
            
            if not top_effect_sizes.empty:
                report.append("Effect sizes between top metrics:\n\n")
                report.append("| Metric 1 | Metric 2 | Cohen's d | Cliff's delta | Effect Magnitude |\n")
                report.append("|----------|----------|-----------|--------------|------------------|\n")
                
                for _, row in top_effect_sizes.iterrows():
                    report.append(
                        f"| {row['metric1']} | {row['metric2']} | "
                        f"{row['cohens_d']:.4f} | {row['cliffs_delta']:.4f} | "
                        f"{row['effect_magnitude']} |\n"
                    )
            else:
                report.append("No effect size data available for top metrics.\n")
        else:
            report.append("No metric ranking data available for effect size analysis.\n")
    else:
        report.append("No effect size data available.\n")
    
    report.append("\n## Conclusion\n\n")
    
    # Add conclusion based on results
    if friedman_result.get('reject_h0', False):
        # Get top metric name
        top_metric = "the top-performing metric" 
        if not metric_rankings.empty and 'metric_name' in metric_rankings.columns:
            top_metric = metric_rankings.iloc[0]['metric_name'] if len(metric_rankings) > 0 else "the top-performing metric"
        
        report.append(
            "The statistical analysis confirms that there are significant differences "
            "in the discriminative power of different similarity metrics for activity classification. "
            f"{top_metric} "
            "demonstrates superior discriminative power, which is statistically significant "
            "compared to lower-ranked metrics."
        )
    else:
        report.append(
            "The statistical analysis does not provide strong evidence for significant differences "
            "in the discriminative power of the tested similarity metrics. "
            "This suggests that the choice of similarity metric may not be critical "
            "for the activity classification task, and other factors may have more influence."
        )
    
    # Write report to file
    with open(output_path, 'w') as f:
        f.write(''.join(report))


def main():
    """
    Example usage of the statistical analysis module.
    """
    # This would be called from rq2_experiment.py with actual results
    print("RQ2 Statistical Analysis Module")
    print("Run this module through the main experiment controller: rq2_experiment.py")


if __name__ == "__main__":
    main() 