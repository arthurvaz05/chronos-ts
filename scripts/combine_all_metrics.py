#!/usr/bin/env python3
"""
Comprehensive Metrics Aggregation Script
Automatically finds and combines all metrics comparison files from all result folders
"""

import pandas as pd
import os
import glob
from pathlib import Path

def find_all_metrics_files(results_dir="results"):
    """
    Find all metrics comparison CSV files in the results directory structure
    """
    print("🔍 Searching for metrics comparison files...")
    
    # Pattern to match metrics comparison files
    pattern = f"{results_dir}/**/*_metrics_comparison.csv"
    all_files = glob.glob(pattern, recursive=True)
    
    # Filter out combined/aggregated files to avoid duplicates
    metrics_files = [f for f in all_files if not any(exclude in f.lower() for exclude in ['combined', 'all_metrics', 'aggregated'])]
    
    if not metrics_files:
        print("⚠️  No individual metrics comparison files found!")
        return []
    
    print(f"📁 Found {len(metrics_files)} individual metrics comparison files:")
    for file in metrics_files:
        print(f"   - {file}")
    
    return metrics_files

def extract_dataset_info(file_path):
    """
    Extract dataset and column information from file path
    Example: results/climate/usa_clima/climate_usa_clima_metrics_comparison.csv
    Returns: ('climate', 'usa_clima')
    """
    path_parts = Path(file_path).parts
    
    # Look for the pattern: results/dataset_name/column_name/filename
    if len(path_parts) >= 4:
        dataset_name = path_parts[1]  # e.g., 'climate'
        column_name = path_parts[2]   # e.g., 'usa_clima'
        return dataset_name, column_name
    
    return "unknown", "unknown"

def load_and_process_metrics(file_path):
    """
    Load a metrics CSV file and add dataset/column information
    """
    try:
        # Load the metrics file
        df = pd.read_csv(file_path)
        
        # Extract dataset and column info from path
        dataset_name, column_name = extract_dataset_info(file_path)
        
        # Add metadata columns
        df['Dataset'] = dataset_name
        df['Column'] = column_name
        df['Source_File'] = os.path.basename(file_path)
        
        print(f"✅ Loaded: {dataset_name}/{column_name} ({len(df)} methods)")
        return df
        
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def combine_all_metrics(results_dir="results"):
    """
    Main function to combine all metrics from all folders
    """
    print("🚀 Starting comprehensive metrics aggregation...")
    print("=" * 80)
    
    # Find all metrics files
    metrics_files = find_all_metrics_files(results_dir)
    
    if not metrics_files:
        return None
    
    print("\n📊 Loading and processing metrics files...")
    
    # Load and process each metrics file
    all_metrics = []
    successful_loads = 0
    
    for file_path in metrics_files:
        df = load_and_process_metrics(file_path)
        if df is not None:
            all_metrics.append(df)
            successful_loads += 1
    
    if not all_metrics:
        print("❌ No metrics files could be loaded successfully!")
        return None
    
    print(f"\n✅ Successfully loaded {successful_loads}/{len(metrics_files)} files")
    
    # Combine all metrics
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    
    # Reorder columns for better readability
    column_order = ['Dataset', 'Column', 'Method', 'RMSE', 'MAE', 'MAPE (%)', 'MASE', 'SMAPE (%)', 'Correlation', 'Source_File']
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in combined_metrics.columns]
    combined_metrics = combined_metrics[existing_columns]
    
    # Sort by Dataset, Column, and Method for better organization
    combined_metrics = combined_metrics.sort_values(['Dataset', 'Column', 'Method'])
    
    return combined_metrics

def save_and_display_results(combined_metrics, output_dir="results"):
    """
    Save the combined metrics and display summary
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save combined metrics
    output_path = f"{output_dir}/all_metrics_comparison.csv"
    combined_metrics.to_csv(output_path, index=False)
    
    print("\n📊 Combined Metrics Summary:")
    print("=" * 80)
    
    # Display summary statistics
    print(f"📈 Total Methods Evaluated: {len(combined_metrics)}")
    print(f"🌍 Total Datasets: {combined_metrics['Dataset'].nunique()}")
    print(f"📊 Total Columns: {combined_metrics['Column'].nunique()}")
    
    # Show unique datasets and columns
    print(f"\n📁 Datasets Found:")
    for dataset in sorted(combined_metrics['Dataset'].unique()):
        dataset_data = combined_metrics[combined_metrics['Dataset'] == dataset]
        columns = sorted(dataset_data['Column'].unique())
        print(f"   - {dataset}: {', '.join(columns)}")
    
    # Show methods used
    print(f"\n🔧 Methods Evaluated:")
    for method in sorted(combined_metrics['Method'].unique()):
        print(f"   - {method}")
    
    # Display the combined table
    print(f"\n📋 Combined Metrics Table:")
    print("=" * 80)
    print(combined_metrics.to_string(index=False))
    print("=" * 80)
    
    print(f"\n✅ Combined metrics saved to: {output_path}")
    
    return output_path

def main():
    """
    Main execution function
    """
    print("🎯 Comprehensive Metrics Aggregation Tool")
    print("=" * 80)
    
    # Combine all metrics
    combined_metrics = combine_all_metrics()
    
    if combined_metrics is None:
        print("❌ Failed to combine metrics. Exiting.")
        return
    
    # Save and display results
    output_path = save_and_display_results(combined_metrics)
    
    print(f"\n🎉 Aggregation completed successfully!")
    print(f"📁 Output file: {output_path}")
    print(f"📊 Total records: {len(combined_metrics)}")

if __name__ == "__main__":
    main()
