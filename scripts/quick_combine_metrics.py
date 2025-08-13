#!/usr/bin/env python3
"""
Quick Metrics Combination Script - One-liner version
Run this to quickly combine all metrics from all folders
"""

import pandas as pd
import os
import glob

if __name__ == "__main__":
    # Find all individual metrics files (exclude combined ones)
    pattern = "results/**/*_metrics_comparison.csv"
    all_files = glob.glob(pattern, recursive=True)
    metrics_files = [f for f in all_files if not any(exclude in f.lower() for exclude in ['combined', 'all_metrics', 'aggregated'])]
    
    print(f"🔍 Found {len(metrics_files)} metrics files")
    
    # Load and combine all metrics
    all_metrics = []
    for file_path in metrics_files:
        df = pd.read_csv(file_path)
        # Extract dataset and column from path
        path_parts = file_path.split('/')
        if len(path_parts) >= 4:
            df['Dataset'] = path_parts[1]
            df['Column'] = path_parts[2]
        all_metrics.append(df)
    
    # Combine and save
    combined = pd.concat(all_metrics, ignore_index=True)
    combined = combined.sort_values(['Dataset', 'Column', 'Method'])
    
    output_path = "results/all_metrics_comparison.csv"
    combined.to_csv(output_path, index=False)
    
    print(f"✅ Combined {len(combined)} methods from {len(metrics_files)} files")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Summary: {combined['Dataset'].nunique()} datasets, {combined['Column'].nunique()} columns")
