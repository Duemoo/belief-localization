import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Load data
base = '/mnt/sda/hoyeon/belief-localization/results/'
df_rome = pd.read_csv(f'{base}/gpt2-medium_ROME_outputs_cf_editing_sweep_ws-[1]_layer-all_n50.csv')
df_noise = pd.read_csv(f'{base}/gpt2-medium_ROME_NOISE-scaled_random_outputs_cf_editing_sweep_ws-[1]_layer-all_n50.csv')

print("Data shapes:")
print(f"ROME: {df_rome.shape}, NOISE: {df_noise.shape}")

# Find available metrics
available_metrics = []
candidate_metrics = [
    'post_neighborhood_success', 'neighborhood_prob_diff', 'paraphrase_prob_diff',
    'essence_ppl_diff', 'post_essence_ppl', 'pre_essence_ppl', 'essence_score',
    'post_rewrite_success', 'rewrite_prob_diff'  # Include these for context
]

for metric in candidate_metrics:
    if metric in df_rome.columns and metric in df_noise.columns:
        available_metrics.append(metric)

print(f"\nAvailable metrics: {available_metrics}")

# Core analysis
print("\n" + "="*60)
print("ROME vs NOISE COMPARISON")
print("="*60)

for metric in available_metrics:
    rome_vals = df_rome[metric].dropna()
    noise_vals = df_noise[metric].dropna()
    
    rome_mean = rome_vals.mean()
    noise_mean = noise_vals.mean()
    diff = abs(rome_mean - noise_mean)
    
    # Statistical test
    if len(rome_vals) > 1 and len(noise_vals) > 1:
        t_stat, p_val = ttest_ind(rome_vals, noise_vals)
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    else:
        p_val = float('nan')
        significance = ""
    
    print(f"{metric:25s}: ROME={rome_mean:6.3f}, NOISE={noise_mean:6.3f}, |diff|={diff:6.3f}, p={p_val:6.4f} {significance}")

# NEH Assessment
print("\n" + "="*60)
print("NEH ASSESSMENT")
print("="*60)

if 'post_neighborhood_success' in available_metrics:
    rome_neighbor = df_rome['post_neighborhood_success'].mean()
    noise_neighbor = df_noise['post_neighborhood_success'].mean()
    
    if noise_neighbor > rome_neighbor + 0.05:  # 5% threshold
        print("🔥 STRONG NEH EVIDENCE: Noise preserves neighbors BETTER than ROME!")
    elif abs(noise_neighbor - rome_neighbor) < 0.05:
        print("✅ MODERATE NEH EVIDENCE: Similar neighborhood preservation")
    else:
        print("❌ NEH NOT SUPPORTED: ROME preserves neighbors better")

print(f"\nROME neighborhood success: {rome_neighbor:.3f}")
print(f"Noise neighborhood success: {noise_neighbor:.3f}")
print(f"Difference: {noise_neighbor - rome_neighbor:.3f} (positive = noise better)")