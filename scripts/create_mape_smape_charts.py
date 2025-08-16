#!/usr/bin/env python3
"""
Create MAPE and SMAPE Charts from Combined Metrics
Visualizes the Mean Absolute Percentage Error and Symmetric Mean Absolute Percentage Error
across different datasets, columns, and methods
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_combined_metrics(file_path="results/all_metrics_comparison.csv"):
    """
    Load the combined metrics data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def create_mape_smape_comparison_chart(df, output_dir="figures"):
    """
    Create a comprehensive comparison chart of MAPE vs SMAPE
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    # No main title - keeping only subplot titles
    
    # 1. Overall MAPE vs SMAPE scatter plot
    ax1 = axes[0, 0]
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        ax1.scatter(method_data['MAPE (%)'], method_data['SMAPE (%)'], 
                   label=method, alpha=0.7, s=60)
    
    ax1.set_xlabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('SMAPE (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall MAPE vs SMAPE by Method', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add diagonal line for reference
    max_val = max(df['MAPE (%)'].max(), df['SMAPE (%)'].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='MAPE = SMAPE')
    
    # 2. Box plot by Method
    ax2 = axes[0, 1]
    df_melted = df.melt(value_vars=['MAPE (%)', 'SMAPE (%)'], 
                        id_vars=['Method'], var_name='Metric', value_name='Value')
    sns.boxplot(data=df_melted, x='Method', y='Value', hue='Metric', ax=ax2)
    ax2.set_title('MAPE and SMAPE Distribution by Method', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error Percentage (%)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Metric')
    
    # 3. Heatmap by Dataset and Method
    ax3 = axes[1, 0]
    pivot_data = df.groupby(['Dataset', 'Method'])['MAPE (%)'].mean().unstack()
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax3, cbar_kws={'label': 'MAPE (%)'})
    ax3.set_title('Average MAPE by Dataset and Method', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    
    # 4. Heatmap by Dataset and Method for SMAPE
    ax4 = axes[1, 1]
    pivot_data_smape = df.groupby(['Dataset', 'Method'])['SMAPE (%)'].mean().unstack()
    sns.heatmap(pivot_data_smape, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax4, cbar_kws={'label': 'SMAPE (%)'})
    ax4.set_title('Average SMAPE by Dataset and Method', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # Save the chart
    output_path = f"{output_dir}/mape_smape_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison chart to: {output_path}")
    
    return fig

def create_dataset_specific_charts(df, output_dir="figures"):
    """
    Create detailed charts for each dataset
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    for dataset in df['Dataset'].unique():
        dataset_data = df[df['Dataset'] == dataset]
        
        # Create figure for this dataset
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        # No main title - keeping only subplot titles
        
        # 1. MAPE by Column and Method
        ax1 = axes[0, 0]
        pivot_mape = dataset_data.pivot_table(values='MAPE (%)', 
                                            index='Column', columns='Method', aggfunc='mean')
        sns.heatmap(pivot_mape, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax1, 
                   cbar_kws={'label': 'MAPE (%)'})
        ax1.set_title(f'MAPE by Column and Method', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Column', fontsize=12, fontweight='bold')
        
        # 2. SMAPE by Column and Method
        ax2 = axes[0, 1]
        pivot_smape = dataset_data.pivot_table(values='SMAPE (%)', 
                                             index='Column', columns='Method', aggfunc='mean')
        sns.heatmap(pivot_smape, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax2, 
                   cbar_kws={'label': 'SMAPE (%)'})
        ax2.set_title(f'SMAPE by Column and Method', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Column', fontsize=12, fontweight='bold')
        
        # 3. Box plot of MAPE by Method
        ax3 = axes[1, 0]
        sns.boxplot(data=dataset_data, x='Method', y='MAPE (%)', ax=ax3)
        ax3.set_title(f'MAPE Distribution by Method', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax3.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Box plot of SMAPE by Method
        ax4 = axes[1, 1]
        sns.boxplot(data=dataset_data, x='Method', y='SMAPE (%)', ax=ax4)
        ax4.set_title(f'SMAPE Distribution by Method', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax4.set_ylabel('SMAPE (%)', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)
        
        # Save the chart
        output_path = f"{output_dir}/{dataset}_mape_smape_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved {dataset} analysis chart to: {output_path}")
        
        plt.close()

def create_method_comparison_chart(df, output_dir="figures"):
    """
    Create a focused comparison chart showing method performance
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Calculate average metrics by method
    method_stats = df.groupby('Method')[['MAPE (%)', 'SMAPE (%)']].agg(['mean', 'std']).round(3)
    
    # Create the comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # No main title - keeping only subplot titles
    
    # MAPE comparison
    methods = method_stats.index
    mape_means = method_stats[('MAPE (%)', 'mean')]
    mape_stds = method_stats[('MAPE (%)', 'std')]
    
    bars1 = ax1.bar(methods, mape_means, yerr=mape_stds, capsize=5, 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax1.set_title('Average MAPE by Method', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars1, mape_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{mean_val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # SMAPE comparison
    smape_means = method_stats[('SMAPE (%)', 'mean')]
    smape_stds = method_stats[('SMAPE (%)', 'std')]
    
    bars2 = ax2.bar(methods, smape_means, yerr=smape_stds, capsize=5,
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax2.set_title('Average SMAPE by Method', fontsize=14, fontweight='bold')
    ax2.set_ylabel('SMAPE (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars2, smape_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{mean_val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)
    
    # Save the chart
    output_path = f"{output_dir}/method_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved method comparison chart to: {output_path}")
    
    return fig

def create_summary_statistics(df):
    """
    Print summary statistics for MAPE and SMAPE
    """
    print("\n📊 MAPE and SMAPE Summary Statistics:")
    print("=" * 60)
    
    # Overall statistics
    print(f"\n🌍 Overall Statistics:")
    print(f"   MAPE - Mean: {df['MAPE (%)'].mean():.2f}%, Std: {df['MAPE (%)'].std():.2f}%")
    print(f"   SMAPE - Mean: {df['SMAPE (%)'].mean():.2f}%, Std: {df['SMAPE (%)'].std():.2f}%")
    
    # By method
    print(f"\n🔧 By Method:")
    method_stats = df.groupby('Method')[['MAPE (%)', 'SMAPE (%)']].agg(['mean', 'std']).round(3)
    print(method_stats)
    
    # By dataset
    print(f"\n📁 By Dataset:")
    dataset_stats = df.groupby('Dataset')[['MAPE (%)', 'SMAPE (%)']].agg(['mean', 'std']).round(3)
    print(dataset_stats)
    
    # Best and worst performers
    print(f"\n🏆 Best Performers (Lowest MAPE):")
    best_mape = df.nsmallest(5, 'MAPE (%)')[['Dataset', 'Column', 'Method', 'MAPE (%)']]
    print(best_mape.to_string(index=False))
    
    print(f"\n❌ Worst Performers (Highest MAPE):")
    worst_mape = df.nlargest(5, 'MAPE (%)')[['Dataset', 'Column', 'Method', 'MAPE (%)']]
    print(worst_mape.to_string(index=False))

def main():
    """
    Main execution function
    """
    print("🎯 Creating MAPE and SMAPE Charts from Combined Metrics")
    print("=" * 60)
    
    # Load the combined metrics
    df = load_combined_metrics()
    if df is None:
        return
    
    # Create output directory
    output_dir = "figures"
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n📊 Creating charts...")
    
    # Create comprehensive comparison chart
    create_mape_smape_comparison_chart(df, output_dir)
    
    # Create dataset-specific charts
    create_dataset_specific_charts(df, output_dir)
    
    # Create method comparison chart
    create_method_comparison_chart(df, output_dir)
    
    # Print summary statistics
    create_summary_statistics(df)
    
    print(f"\n🎉 All charts created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total records analyzed: {len(df)}")

if __name__ == "__main__":
    main()
