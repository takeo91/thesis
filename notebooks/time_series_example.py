# Cell 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Thesis package imports
from thesis.fuzzy.membership import compute_membership_function
from thesis.fuzzy.similarity import (
    similarity_jaccard,
    similarity_cosine,
    similarity_dice,
    similarity_overlap_coefficient
)

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)

# Cell 2: Generate Sample Time Series Data
np.random.seed(42)

# Time vector
t = np.linspace(0, 10, 1000)

# Series 1: Sine wave with noise
series1 = np.sin(t) + 0.2 * np.random.randn(len(t))

# Series 2: Similar to Series 1 but with phase shift
series2 = np.sin(t + 0.5) + 0.2 * np.random.randn(len(t))

# Series 3: Different pattern (cosine with different frequency)
series3 = np.cos(2*t) + 0.2 * np.random.randn(len(t))

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(t, series1, label='Series 1')
plt.plot(t, series2, label='Series 2')
plt.plot(t, series3, label='Series 3')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Sample Time Series Data')
plt.legend()
plt.show()

# Cell 3: Compute Fuzzy Membership Functions
# Domain for membership functions
x_min = min(np.min(series1), np.min(series2), np.min(series3))
x_max = max(np.max(series1), np.max(series2), np.max(series3))
x_range = x_max - x_min
x_values = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 500)

# Compute membership functions
_, mu_s1, sigma1 = compute_membership_function(series1, x_values, sigma='r0.1')
_, mu_s2, sigma2 = compute_membership_function(series2, x_values, sigma='r0.1')
_, mu_s3, sigma3 = compute_membership_function(series3, x_values, sigma='r0.1')

# Plot membership functions
plt.figure(figsize=(12, 6))
plt.plot(x_values, mu_s1, label=f'Series 1 (σ={sigma1:.3f})')
plt.plot(x_values, mu_s2, label=f'Series 2 (σ={sigma2:.3f})')
plt.plot(x_values, mu_s3, label=f'Series 3 (σ={sigma3:.3f})')
plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.title('Fuzzy Membership Functions')
plt.legend()
plt.show()

# Cell 4: Compute Similarity Metrics
# Calculate similarities between pairs
similarity_metrics = {
    'Series 1-2': {
        'Jaccard': similarity_jaccard(mu_s1, mu_s2),
        'Cosine': similarity_cosine(mu_s1, mu_s2),
        'Dice': similarity_dice(mu_s1, mu_s2),
        'Overlap': similarity_overlap_coefficient(mu_s1, mu_s2)
    },
    'Series 1-3': {
        'Jaccard': similarity_jaccard(mu_s1, mu_s3),
        'Cosine': similarity_cosine(mu_s1, mu_s3),
        'Dice': similarity_dice(mu_s1, mu_s3),
        'Overlap': similarity_overlap_coefficient(mu_s1, mu_s3)
    },
    'Series 2-3': {
        'Jaccard': similarity_jaccard(mu_s2, mu_s3),
        'Cosine': similarity_cosine(mu_s2, mu_s3),
        'Dice': similarity_dice(mu_s2, mu_s3),
        'Overlap': similarity_overlap_coefficient(mu_s2, mu_s3)
    }
}

# Convert to DataFrame for easier display
similarity_df = pd.DataFrame({
    'Series Pair': list(similarity_metrics.keys()),
    'Jaccard': [d['Jaccard'] for d in similarity_metrics.values()],
    'Cosine': [d['Cosine'] for d in similarity_metrics.values()],
    'Dice': [d['Dice'] for d in similarity_metrics.values()],
    'Overlap': [d['Overlap'] for d in similarity_metrics.values()]
})

print(similarity_df)

# Cell 5: Visualize Similarity Results
# Melt the DataFrame for easier plotting
similarity_melted = pd.melt(similarity_df, id_vars=['Series Pair'], 
                            var_name='Similarity Metric', value_name='Similarity')

# Plot similarity metrics
plt.figure(figsize=(12, 6))
sns.barplot(x='Series Pair', y='Similarity', hue='Similarity Metric', data=similarity_melted)
plt.title('Comparison of Fuzzy Similarity Metrics')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Cell 6: Experiment with Different Sigma Values
# Test different sigma values
sigma_values = ['r0.05', 'r0.1', 'r0.2', 'r0.3']

# Compute membership functions for Series 1 with different sigma values
plt.figure(figsize=(12, 6))
for sigma in sigma_values:
    _, mu, sigma_val = compute_membership_function(series1, x_values, sigma=sigma)
    plt.plot(x_values, mu, label=f'σ={sigma} ({sigma_val:.3f})')

plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.title('Effect of Sigma on Membership Function (Series 1)')
plt.legend()
plt.show()

# Cell 7: Sigma Value Impact on Similarity
# Compute Jaccard similarity with different sigma values
jaccard_results = []

for sigma in sigma_values:
    # Compute membership functions
    _, mu_s1_sigma, _ = compute_membership_function(series1, x_values, sigma=sigma)
    _, mu_s2_sigma, _ = compute_membership_function(series2, x_values, sigma=sigma)
    _, mu_s3_sigma, _ = compute_membership_function(series3, x_values, sigma=sigma)
    
    # Compute similarity
    jaccard_s1_s2 = similarity_jaccard(mu_s1_sigma, mu_s2_sigma)
    jaccard_s1_s3 = similarity_jaccard(mu_s1_sigma, mu_s3_sigma)
    
    jaccard_results.append({
        'Sigma': sigma,
        'Series 1-2': jaccard_s1_s2,
        'Series 1-3': jaccard_s1_s3,
        'Ratio': jaccard_s1_s2 / max(jaccard_s1_s3, 1e-9)
    })

jaccard_df = pd.DataFrame(jaccard_results)
print(jaccard_df)

# Plot the effect of sigma on Jaccard similarity
plt.figure(figsize=(12, 6))
plt.plot(range(len(sigma_values)), jaccard_df['Series 1-2'], 'o-', label='Series 1-2')
plt.plot(range(len(sigma_values)), jaccard_df['Series 1-3'], 'o-', label='Series 1-3')
plt.plot(range(len(sigma_values)), jaccard_df['Ratio'], 'o-', label='Ratio (1-2 / 1-3)')
plt.xticks(range(len(sigma_values)), sigma_values)
plt.xlabel('Sigma Value')
plt.ylabel('Jaccard Similarity')
plt.title('Effect of Sigma on Jaccard Similarity')
plt.legend()
plt.grid(True)
plt.show()

# Cell 8: Conclusions
print("Key Findings:")
print("1. The similarity between Series 1 and 2 (phase-shifted versions of the same signal) is higher")
print("   than between Series 1 and 3 (different patterns) across all metrics.")
print("2. Sigma value affects the shape of membership functions and resulting similarity values.")
print("3. Selecting an appropriate sigma value is crucial for distinguishing between")
print("   similar and dissimilar time series.")
print("4. Different similarity metrics emphasize different aspects of similarity,")
print("   with some being more sensitive to pattern differences than others.") 