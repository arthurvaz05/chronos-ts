#!/usr/bin/env python3
"""
Script to create a comprehensive plot of the predictions vs actual values.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

def plot_predictions(predictions_file):
    """
    Create a comprehensive plot of predictions vs actual values.
    
    Args:
        predictions_file: Path to the predictions CSV file
    """
    # Read the predictions data
    df = pd.read_csv(predictions_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Separate actual and predicted data
    actual_data = df[df['type'] == 'actual'].copy()
    predicted_data = df[df['type'] == 'predicted'].copy()
    
    # Sort by timestamp
    actual_data = actual_data.sort_values('timestamp')
    predicted_data = predicted_data.sort_values('timestamp')
    
    # Create the plot
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Time Series with Predictions
    ax1.plot(actual_data['timestamp'], actual_data['value'], 
             label='Actual Values', marker='o', linewidth=2, markersize=4, color='blue')
    ax1.plot(predicted_data['timestamp'], predicted_data['value'], 
             label='Predicted Values', marker='x', linewidth=2, markersize=6, color='red')
    
    # Add a vertical line to separate actual from predicted
    if len(actual_data) > 0 and len(predicted_data) > 0:
        last_actual_time = actual_data['timestamp'].max()
        ax1.axvline(x=last_actual_time, color='green', linestyle='--', alpha=0.7, 
                    label='Prediction Start')
    
    ax1.set_title('Climate Time Series: Actual vs Predicted Values', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Climate Value', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Scatter plot of actual vs predicted (for overlapping time periods)
    # Find overlapping timestamps
    common_times = set(actual_data['timestamp']) & set(predicted_data['timestamp'])
    
    if common_times:
        # Get values for common timestamps
        actual_common = actual_data[actual_data['timestamp'].isin(common_times)]
        predicted_common = predicted_data[predicted_data['timestamp'].isin(common_times)]
        
        # Merge on timestamp
        comparison = pd.merge(actual_common, predicted_common, on='timestamp', suffixes=('_actual', '_predicted'))
        
        ax2.scatter(comparison['value_actual'], comparison['value_predicted'], 
                   alpha=0.7, color='purple', s=50)
        
        # Add diagonal line for perfect predictions
        min_val = min(comparison['value_actual'].min(), comparison['value_predicted'].min())
        max_val = max(comparison['value_actual'].max(), comparison['value_predicted'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        ax2.set_title('Actual vs Predicted Values (Overlapping Periods)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Actual Values', fontsize=12)
        ax2.set_ylabel('Predicted Values', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Calculate correlation and RMSE
        correlation = comparison['value_actual'].corr(comparison['value_predicted'])
        rmse = np.sqrt(((comparison['value_actual'] - comparison['value_predicted']) ** 2).mean())
        
        # Add text box with metrics
        metrics_text = f'Correlation: {correlation:.3f}\nRMSE: {rmse:.3f}'
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No overlapping time periods\nfor comparison', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=14)
        ax2.set_title('Actual vs Predicted Values (No Overlap)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = predictions_file.replace('.csv', '_analysis_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Prediction analysis plot saved to: {output_file}")
    
    # Show some statistics
    print(f"\n📈 Prediction Statistics:")
    print(f"   Actual data points: {len(actual_data)}")
    print(f"   Predicted data points: {len(predicted_data)}")
    print(f"   Time range: {actual_data['timestamp'].min()} to {predicted_data['timestamp'].max()}")
    
    if len(actual_data) > 0:
        print(f"   Actual value range: {actual_data['value'].min():.3f} to {actual_data['value'].max():.3f}")
    if len(predicted_data) > 0:
        print(f"   Predicted value range: {predicted_data['value'].min():.3f} to {predicted_data['value'].max():.3f}")
    
    plt.show()

def main():
    """Main function to plot predictions."""
    # Path to the predictions file
    predictions_file = "results/climate/usa_clima/climate_usa_clima_predictions.csv"
    
    try:
        plot_predictions(predictions_file)
    except FileNotFoundError:
        print(f"❌ Error: Predictions file not found at {predictions_file}")
        print("Please run the training and evaluation pipeline first.")
    except Exception as e:
        print(f"❌ Error creating plot: {e}")

if __name__ == "__main__":
    main()
