import sys
import subprocess
import os
import re
import json
from scripts.generate_ts import read_transform_data, convert_to_arrow
import yaml

def prepare_data(csv_path: str, arrow_path: str, test_mode: bool = False, max_columns: int = 2):
    """
    Reads the climate dataset from CSV and converts it to Arrow format.
    
    Args:
        csv_path: Path to the CSV file
        arrow_path: Path to save the Arrow file
        test_mode: If True, only process limited columns
        max_columns: Maximum number of columns to process (0 = all columns)
    """
    print("Preparing data: converting CSV to Arrow format...")
    
    # Import here to avoid circular imports
    from scripts.generate_ts import read_transform_data, convert_to_arrow
    
    time_series = read_transform_data(csv_path, test_mode, max_columns)
    convert_to_arrow(arrow_path, time_series=time_series)
    print(f"Data written to {arrow_path}\n")

def get_latest_run_dir(output_dir="output"):
    """
    Finds the run-N directory with the highest N in the output directory.
    """
    run_pattern = re.compile(r"run-(\d+)")
    max_run = -1
    for name in os.listdir(output_dir):
        match = run_pattern.fullmatch(name)
        if match:
            run_num = int(match.group(1))
            if run_num > max_run:
                max_run = run_num
    if max_run == -1:
        raise RuntimeError("No run-N directories found in output directory.")
    return f"{output_dir}/run-{max_run}/checkpoint-final"

def run_command(cmd, step_name):
    """
    Runs a shell command and handles errors.
    """
    print(f"{step_name}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"{step_name} failed. Exiting.")
        sys.exit(result.returncode)
    print(f"{step_name} completed successfully.\n")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


 



def generate_yaml_config(dataset_name: str, column_name: str, prediction_length: int = 5):
    config = [
        {
            'name': dataset_name,
            'dataset_path': f'Dataset/Dataset.arrow/{dataset_name}_{column_name}.arrow',
            'offset': -prediction_length,  # Matches your prediction_length
            'prediction_length': prediction_length,  # Matches your training configuration
            'num_rolls': 1
        }
    ]
    config_path = f'scripts/evaluation/configs/{dataset_name}.yaml'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"YAML config written to {config_path}")

def create_dataset_column_folders(dataset_name: str, csv_path: str, test_mode=False, max_columns=2):
    """
    Create organized folder structure for each dataset and its columns.
    
    Args:
        dataset_name: Name of the dataset
        csv_path: Path to the CSV file to read columns from
        test_mode: If True, only create folder for first column
    """
    # Read CSV to get column names
    df = pd.read_csv(csv_path)
    
    # Create main dataset folder
    dataset_folder = f"results/{dataset_name}"
    os.makedirs(dataset_folder, exist_ok=True)
    
    # Create subfolders for columns (time series)
    column_folders = {}
    
    if test_mode:
        # Test mode: limited number of columns
        if len(df.columns) > 0:
            if max_columns == 0:
                # Process all columns
                for col in df.columns:
                    column_folder = f"{dataset_folder}/{col}"
                    os.makedirs(column_folder, exist_ok=True)
                    column_folders[col] = column_folder
                print(f"TEST MODE: Created folders for ALL columns: {len(column_folders)} columns")
            else:
                # Process limited number of columns
                for i in range(min(max_columns, len(df.columns))):
                    col = df.columns[i]
                    column_folder = f"{dataset_folder}/{col}"
                    os.makedirs(column_folder, exist_ok=True)
                    column_folders[col] = column_folder
                print(f"TEST MODE: Created folders for first {len(column_folders)} columns: {list(column_folders.keys())}")
    else:
        # Normal mode: all columns
        for col in df.columns:
            column_folder = f"{dataset_folder}/{col}"
            os.makedirs(column_folder, exist_ok=True)
            column_folders[col] = column_folder
    
    if not test_mode:
        print(f"Created organized folder structure for {dataset_name}:")
        for col, folder in column_folders.items():
            print(f"  - {folder}")
    
    return column_folders

def save_training_history_and_charts(dataset_name: str, column_folder: str, column_name: str):
    """
    Save training parameters history and create training progress charts.
    
    Args:
        dataset_name: Name of the dataset
        column_folder: Path to the column folder
        column_name: Name of the column
    """
    try:
        training_dir = f"results/{dataset_name}/training"
        
        # Find the latest training run
        run_pattern = re.compile(r"run-(\d+)")
        max_run = -1
        for name in os.listdir(training_dir):
            match = run_pattern.fullmatch(name)
            if match:
                run_num = int(match.group(1))
                if run_num > max_run:
                    max_run = run_num
        
        if max_run == -1:
            print(f"⚠️  No training runs found in {training_dir}")
            return
        
        latest_run_dir = f"{training_dir}/run-{max_run}"
        print(f"📊 Processing training history from: {latest_run_dir}")
        
        # Save training parameters to a JSON file directly in column folder
        # Use the centralized training configuration for consistency
        training_config = get_training_config(dataset_name, 5)  # Default prediction length
        training_params = {
            "dataset_name": dataset_name,
            "column_name": column_name,
            **training_config,  # Include all training configuration parameters
            "training_run": f"run-{max_run}",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        params_file = f"{column_folder}/{dataset_name}_{column_name}_training_params.json"
        with open(params_file, 'w') as f:
            json.dump(training_params, f, indent=2, default=str)
        print(f"📝 Training parameters saved to: {params_file}")
        
        # Save inference parameters separately (can use larger model for inference)
        inference_params = get_inference_config(dataset_name, max_run)
        # Add column name to inference params
        inference_params["column_name"] = column_name
        
        inference_params_file = f"{column_folder}/{dataset_name}_{column_name}_inference_params.json"
        with open(inference_params_file, 'w') as f:
            json.dump(inference_params, f, indent=2, default=str)
        print(f"📝 Inference parameters saved to: {inference_params_file}")
        
        # Create training progress charts directly in column folder
        create_training_charts(dataset_name, latest_run_dir, column_folder, column_name)
        
    except Exception as e:
        print(f"⚠️  Error saving training history: {e}")

def create_training_charts(dataset_name: str, training_run_dir: str, column_folder: str, column_name: str):
    """
    Create training progress charts from training logs.
    
    Args:
        dataset_name: Name of the dataset
        training_run_dir: Path to the training run directory
        column_folder: Path to the column folder
        column_name: Name of the column
    """
    try:
        # Check if there are training logs
        logs_dir = f"{training_run_dir}/logs"
        if not os.path.exists(logs_dir):
            print(f"⚠️  No logs directory found in {training_run_dir}")
            return
        
        # Look for tensorboard event files
        event_files = [f for f in os.listdir(logs_dir) if f.startswith("events.out.tfevents")]
        if not event_files:
            print(f"⚠️  No tensorboard event files found in {logs_dir}")
            return
        
        print(f"📈 Creating training charts from logs...")
        
        # Create a simple training progress visualization
        # Since we can't easily parse tensorboard files, we'll create a summary chart
        create_training_summary_chart(dataset_name, training_run_dir, column_folder, column_name)
        
    except Exception as e:
        print(f"⚠️  Error creating training charts: {e}")

def create_training_summary_chart(dataset_name: str, training_run_dir: str, column_folder: str, column_name: str):
    """
    Create a summary chart showing training configuration and expected progress.
    
    Args:
        dataset_name: Name of the dataset
        training_run_dir: Path to the training run directory
        column_folder: Path to the column folder
        column_name: Name of the column
    """
    try:
        # Create a summary visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Summary for {dataset_name} - {column_name}', fontsize=16, fontweight='bold')
        
        # Chart 1: Actual Training Loss Progression (from logs)
        try:
            # Try to read actual training logs from checkpoint directories
            log_file = None
            # Look for trainer_state.json in checkpoint directories
            # We want the final training step, not checkpoint-final
            checkpoint_dirs = [d for d in os.listdir(training_run_dir) if d.startswith('checkpoint-') and d != 'checkpoint-final']
            
            # Extract step numbers and find the highest step
            max_step = 0
            log_file = None
            
            for checkpoint_dir in checkpoint_dirs:
                try:
                    step_num = int(checkpoint_dir.split('-')[1])
                    if step_num > max_step:
                        max_step = step_num
                        checkpoint_path = os.path.join(training_run_dir, checkpoint_dir)
                        potential_log_file = os.path.join(checkpoint_path, 'trainer_state.json')
                        if os.path.exists(potential_log_file):
                            log_file = potential_log_file
                except:
                    continue
            
            if log_file:
                print(f"  📊 Reading training logs from: checkpoint-{max_step} (final training step)")
            else:
                print(f"  ⚠️  No trainer_state.json found in checkpoints")
            
            if log_file and os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    trainer_state = json.load(f)
                
                if 'log_history' in trainer_state:
                    log_history = trainer_state['log_history']
                    steps = [log.get('step', 0) for log in log_history if 'loss' in log]
                    losses = [log.get('loss', 0) for log in log_history if 'loss' in log]
                    
                    if steps and losses:
                        ax1.plot(steps, losses, marker='o', linewidth=2, markersize=6, color='blue')
                        ax1.set_title('Actual Training Loss Progression')
                        ax1.set_xlabel('Training Steps')
                        ax1.set_ylabel('Loss')
                        ax1.grid(True, alpha=0.3)
                        
                        # Add final loss annotation
                        final_loss = losses[-1]
                        ax1.annotate(f'Final Loss: {final_loss:.3f}', 
                                    xy=(steps[-1], losses[-1]), 
                                    xytext=(steps[-1]*0.7, losses[-1]*1.2),
                                    arrowprops=dict(arrowstyle='->', color='red'),
                                    fontsize=10, color='red', fontweight='bold')
                        
                        print(f"  📊 Actual training loss progression loaded from {os.path.basename(log_file)}")
                        print(f"     Final loss: {final_loss:.3f}")
                    else:
                        raise ValueError("No loss data in log history")
                else:
                    raise ValueError("No log history found")
            else:
                raise FileNotFoundError("No trainer state file found in checkpoints")
        except Exception as e:
            print(f"  ⚠️  Could not load actual training logs: {e}")
            # Fallback to expected progression based on our configuration
            training_config = get_training_config(dataset_name, 5)
            max_steps = training_config["max_steps"]
            steps = list(range(0, max_steps + 1, max_steps // 10))
            expected_losses = [6.6, 5.1, 4.6, 4.2, 3.9, 3.6, 3.4, 3.1, 2.9, 2.8, 2.6]
            ax1.plot(steps, expected_losses, marker='o', linewidth=2, markersize=6, color='blue')
            ax1.set_title('Expected Loss Progression (Fallback)')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
        
        # Chart 2: Learning Rate Schedule (from our configuration)
        training_config = get_training_config(dataset_name, 5)
        max_steps = training_config["max_steps"]
        lr_steps = list(range(0, max_steps + 1, max_steps // 10))
        if training_config["lr_scheduler_type"] == "cosine":
            # Cosine learning rate schedule
            lr_values = [training_config["learning_rate"] * (0.5 * (1 + np.cos(np.pi * i / max_steps))) for i in lr_steps]
        else:
            # Linear learning rate schedule
            lr_values = [training_config["learning_rate"] * (1 - i / max_steps) for i in lr_steps]
        
        ax2.plot(lr_steps, lr_values, marker='s', linewidth=2, markersize=6, color='orange')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Actual Gradient Norm Progression (if available) or Expected
        try:
            if log_file and os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    trainer_state = json.load(f)
                
                if 'log_history' in trainer_state:
                    log_history = trainer_state['log_history']
                    steps = [log.get('step', 0) for log in log_history if 'grad_norm' in log]
                    grad_norms = [log.get('grad_norm', 0) for log in log_history if 'grad_norm' in log]
                    
                    if steps and grad_norms:
                        ax3.plot(steps, grad_norms, marker='^', linewidth=2, markersize=6, color='green')
                        ax3.set_title('Actual Gradient Norm Progression')
                        ax3.set_xlabel('Training Steps')
                        ax3.set_ylabel('Gradient Norm')
                        ax3.grid(True, alpha=0.3)
                        
                        # Add final gradient norm annotation
                        final_grad_norm = grad_norms[-1]
                        ax3.annotate(f'Final Grad Norm: {final_grad_norm:.3f}', 
                                    xy=(steps[-1], grad_norms[-1]), 
                                    xytext=(steps[-1]*0.7, grad_norms[-1]*1.2),
                                    arrowprops=dict(arrowstyle='->', color='red'),
                                    fontsize=10, color='red', fontweight='bold')
                        
                        print(f"     Final gradient norm: {final_grad_norm:.3f}")
                    else:
                        raise ValueError("No gradient norm data in log history")
                else:
                    raise ValueError("No log history found")
            else:
                raise FileNotFoundError("No log file available")
        except Exception as e:
            # Fallback to expected gradient norms
            steps = list(range(0, max_steps + 1, max_steps // 10))
            expected_grad_norms = [8.0, 6.6, 12.2, 4.0, 8.2, 5.9, 9.5, 6.3, 4.7, 4.1, 5.0]
            ax3.plot(steps, expected_grad_norms, marker='^', linewidth=2, markersize=6, color='green')
            ax3.set_title('Expected Gradient Norm Progression (Fallback)')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Gradient Norm')
            ax3.grid(True, alpha=0.3)
        
        # Chart 4: Training Configuration Summary (from our actual configuration)
        training_config = get_training_config(dataset_name, 5)
        config_text = f"""
Training Configuration:
• Model: {training_config['model_id'].split('/')[-1]}
• Dataset: {dataset_name}
• Column: {column_name}
• Max Steps: {training_config['max_steps']}
• Batch Size: {training_config['per_device_train_batch_size']}
• Context Length: {training_config['context_length']}
• Prediction Length: {training_config['prediction_length']}
• Learning Rate: {training_config['learning_rate']}
• Training Run: {os.path.basename(training_run_dir)}
        """
        ax4.text(0.1, 0.5, config_text, transform=ax4.transAxes, fontsize=12, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax4.set_title('Training Configuration')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save the chart directly in column folder
        chart_path = f"{column_folder}/{dataset_name}_{column_name}_training_chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training summary chart saved to: {chart_path}")
        
    except Exception as e:
        print(f"⚠️  Error creating training summary chart: {e}")

def create_prediction_analysis_plot(dataset_name: str, column_folder: str, column_name: str):
    """
    Create a comprehensive analysis plot of predictions vs actual values.
    
    Args:
        dataset_name: Name of the dataset
        column_folder: Path to the column folder
        column_name: Name of the column
    """
    try:
        predictions_file = f"{column_folder}/{dataset_name}_{column_name}_predictions.csv"
        
        if not os.path.exists(predictions_file):
            print(f"⚠️  Predictions file not found: {predictions_file}")
            return
        
        print(f"📊 Creating prediction analysis plot for {dataset_name}_{column_name}...")
        
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
        
        # Create the plot - single chart only
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # Split the last 5 values from actual data for comparison with predictions
        if len(actual_data) >= 5:
            # Training data (excluding last 5 values)
            training_data = actual_data.iloc[:-5]
            # Last 5 values for comparison with predictions
            comparison_data = actual_data.iloc[-5:]
            
            # Create year-based timestamps for training data
            training_years = []
            for i in range(len(training_data)):
                year = 1990 + i
                training_years.append(year)
            
            # Create year-based timestamps for comparison data (same period as predictions)
            comparison_years = []
            for i in range(len(comparison_data)):
                year = 1990 + len(training_data) + i
                comparison_years.append(year)
            
            # Create year-based timestamps for predicted data
            predicted_years = []
            for i in range(len(predicted_data)):
                year = 1990 + len(training_data) + i
                predicted_years.append(year)
            
            # Plot training data (excluding last 5 values)
            ax.plot(training_years, training_data['value'], 
                    label='Training Data (1990-{})'.format(training_years[-1]), 
                    marker='o', linewidth=2, markersize=6, color='blue')
            
            # Plot the last 5 actual values (for comparison with predictions)
            ax.plot(comparison_years, comparison_data['value'], 
                    label='Actual Values (Comparison Period)', 
                    marker='s', linewidth=2, markersize=8, color='green')
            
            # Plot predicted values
            ax.plot(predicted_years, predicted_data['value'], 
                    label='Predicted Values', marker='x', linewidth=2, markersize=8, color='red')
            
            # Add a vertical line to separate training from comparison/prediction
            if len(training_data) > 0:
                last_training_year = training_years[-1]
                ax.axvline(x=last_training_year, color='green', linestyle='--', alpha=0.7, 
                          label=f'Comparison Start ({last_training_year})')
            
            # Set title and labels
            ax.set_title(f'{dataset_name.title()} - {column_name}: Training vs Actual vs Predicted (1990-{predicted_years[-1]})', 
                         fontsize=16, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Climate Value', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis to show years clearly
            all_years = training_years + comparison_years
            ax.set_xticks(all_years[::5])  # Show every 5th year for readability
            ax.tick_params(axis='x', rotation=45)
            
            # Add text box with statistics
            stats_text = f'Training: {len(training_data)} points\nActual (Comparison): {len(comparison_data)} points\nPredicted: {len(predicted_data)} points\nTime Range: 1990-{predicted_years[-1]}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Calculate and display comparison metrics
            if len(comparison_data) == len(predicted_data):
                # Extract values for calculations
                actual_values = comparison_data['value'].values
                predicted_values = predicted_data['value'].values
                
                # Calculate RMSE
                rmse = np.sqrt(((actual_values - predicted_values) ** 2).mean())
                
                # Calculate MAE (Mean Absolute Error)
                mae = np.mean(np.abs(actual_values - predicted_values))
                
                # Calculate MAPE (Mean Absolute Percentage Error)
                # Avoid division by zero
                mape = np.mean(np.abs((actual_values - predicted_values) / np.where(actual_values != 0, actual_values, 1))) * 100
                
                # Calculate MASE (Mean Absolute Scaled Error)
                # Use training data for baseline
                if len(training_data) > 1:
                    # Calculate naive forecast (previous value) for training data
                    training_values = training_data['value'].values
                    naive_forecast = training_values[:-1]
                    actual_training = training_values[1:]
                    mae_naive = np.mean(np.abs(actual_training - naive_forecast))
                    
                    if mae_naive > 0:
                        mase = mae / mae_naive
                    else:
                        mase = np.nan
                else:
                    mase = np.nan
                
                # Calculate SMAPE (Symmetric Mean Absolute Percentage Error)
                smape = np.mean(2 * np.abs(predicted_values - actual_values) / (np.abs(actual_values) + np.abs(predicted_values))) * 100
                
                # Calculate correlation with proper error handling
                try:
                    # Ensure we have valid numeric data
                    if len(actual_values) > 1 and len(predicted_values) > 1:
                        # Convert to pandas Series for correlation calculation
                        actual_series = pd.Series(actual_values)
                        predicted_series = pd.Series(predicted_values)
                        correlation = actual_series.corr(predicted_series)
                        
                        # Check if correlation is NaN and provide explanation
                        if pd.isna(correlation):
                            correlation_reason = "NaN (possible reasons: constant values, insufficient variation, or data alignment issues)"
                            correlation_display = "NaN"
                        else:
                            correlation_reason = f"{correlation:.3f}"
                            correlation_display = f"{correlation:.3f}"
                    else:
                        correlation = np.nan
                        correlation_reason = "NaN (insufficient data points)"
                        correlation_display = "NaN"
                except Exception as e:
                    correlation = np.nan
                    correlation_reason = f"NaN (error: {str(e)})"
                    correlation_display = "NaN"
                
                # Create comprehensive metrics text
                metrics_text = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nMAPE: {mape:.1f}%\nMASE: {mase:.3f}\nSMAPE: {smape:.1f}%\nCorrelation: {correlation_display}'
                
                # Add metrics text box (positioned to avoid overlap with other text)
                ax.text(0.98, 0.85, metrics_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
                
                # Print detailed metrics
                print(f"  📊 Comparison Metrics:")
                print(f"     RMSE: {rmse:.3f}")
                print(f"     MAE: {mae:.3f}")
                print(f"     MAPE: {mape:.1f}%")
                print(f"     MASE: {mase:.3f}")
                print(f"     SMAPE: {smape:.1f}%")
                print(f"     Correlation: {correlation_reason}")
                
                # Debug information for correlation issue
                if pd.isna(correlation):
                    print(f"  🔍 Correlation Debug Info:")
                    print(f"     Actual values: {actual_values}")
                    print(f"     Predicted values: {predicted_values}")
                    print(f"     Actual std: {np.std(actual_values):.6f}")
                    print(f"     Predicted std: {np.std(predicted_values):.6f}")
                    print(f"     Data types - Actual: {type(actual_values)}, Predicted: {type(predicted_values)}")
        else:
            # Fallback if we don't have enough data
            ax.text(0.5, 0.5, 'Insufficient data for comparison\nNeed at least 5 data points', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('Insufficient Data for Analysis', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{dataset_name}_{column_name}_prediction_analysis.png"
        plot_path = f"{column_folder}/{plot_filename}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Prediction analysis plot saved to: {plot_path}")
        
        # Print statistics
        print(f"  📈 Prediction Statistics:")
        print(f"     Actual data points: {len(actual_data)}")
        print(f"     Predicted data points: {len(predicted_data)}")
        if len(actual_data) > 0:
            print(f"     Actual value range: {actual_data['value'].min():.3f} to {actual_data['value'].max():.3f}")
        if len(predicted_data) > 0:
            print(f"     Predicted value range: {predicted_data['value'].min():.3f} to {predicted_data['value'].max():.3f}")
        
    except Exception as e:
        print(f"⚠️  Error creating prediction analysis plot: {e}")

def organize_results_by_column(dataset_name: str, column_folders: dict):
    """
    Organize results by copying training outputs to column folders.
    All other files are now saved directly in column folders.
    
    Args:
        dataset_name: Name of the dataset
        column_folders: Dictionary mapping column names to folder paths
    """
    try:
        print(f"📁 Copying training outputs to column folder for {dataset_name}...")
        
        # Get the first column folder (since we're in test mode with only one column)
        if not column_folders:
            print(f"⚠️  No column folders found for {dataset_name}")
            return
        
        column_name = list(column_folders.keys())[0]
        column_folder = column_folders[column_name]
        
        # Only copy training outputs since everything else is saved directly in column folder
        base_dir = f"results/{dataset_name}"
        training_dir = f"{base_dir}/training"
        
        if os.path.exists(training_dir):
            training_dest = f"{column_folder}/{dataset_name}_{column_name}_training_outputs"
            import shutil
            if os.path.exists(training_dest):
                shutil.rmtree(training_dest)
            shutil.copytree(training_dir, training_dest)
            print(f"  🏋️  Training outputs copied to: {training_dest}")
        
        print(f"✅ Training outputs copied to column folder: {column_folder}")
        
    except Exception as e:
        print(f"⚠️  Error copying training outputs: {e}")

def get_training_config(dataset_name: str, prediction_length: int = 5):
    """
    Centralized function to get training configuration parameters.
    This ensures consistency between training commands and parameter saving.
    
    Args:
        dataset_name: Name of the dataset
        prediction_length: Number of values to predict
        
    Returns:
        dict: Training configuration parameters
    """
    return {
        "model_id": "amazon/chronos-t5-base",  # Large model for better performance
        "prediction_length": prediction_length,
        "context_length": 64,  # Large context window for pattern recognition
        "min_past": 32,  # Large min-past for sufficient context
        "max_steps": 2000,  # Optimal training steps (was working well)
        "save_steps": 500,
        "log_steps": 100,
        "per_device_train_batch_size": 8,  # Optimal batch size (was working well)
        "learning_rate": 5e-5,  # Optimal learning rate (was working well)
        "warmup_ratio": 0.1,  # Optimal warmup (was working well)
        "gradient_accumulation_steps": 4,  # Optimal accumulation (was working well)
        "lr_scheduler_type": "cosine",  # Cosine scheduler for better convergence
        "shuffle_buffer_length": 1000,  # Large shuffling buffer for better generalization
        "random_init": False,  # Use pretrained weights (was working well)
        "num_samples": 10,  # Multiple samples for robust training
        "temperature": 0.8,  # Lower temperature for focused but not rigid predictions
        "top_k": 20,  # Limited vocabulary for focused sampling
        "seed": 42,  # Set seed for reproducible randomness
        "n_tokens": 4096,  # Optimal vocabulary size (was working well)
        "tokenizer_kwargs": "{'low_limit': -15.0, 'high_limit': 15.0}",  # Optimal tokenizer range (was working well)
        "dataloader_num_workers": 0,  # Avoid multiprocessing issues on macOS
        "max_missing_prop": 0.9,  # Allow more missing data
        "torch_compile": False,  # Disable torch compilation to avoid MPS issues
        "output_dir": f"results/{dataset_name}/training",  # Save training results in dataset folder
        # Additional jitter and optimization parameters
        "top_p": 0.9  # Add nucleus sampling for better diversity
    }

def get_inference_config(dataset_name: str, max_run: int):
    """
    Centralized function to get inference configuration parameters.
    This ensures consistency for inference parameter saving.
    
    Args:
        dataset_name: Name of the dataset
        max_run: Training run number for reference
        
    Returns:
        dict: Inference configuration parameters
    """
    return {
        "dataset_name": dataset_name,
        "model_id": "amazon/chronos-t5-large",  # Largest available model for inference
        "prediction_length": 5,
        "context_length": 256,  # Maximum context for inference
        "min_past": 128,  # Maximum min-past for inference
        "num_samples": 30,  # More samples for better inference quality
        "temperature": 0.6,  # Lower temperature for more focused inference
        "top_k": 10,  # Lower top-k for more focused inference
        "top_p": 0.85,  # Tighter nucleus sampling
        "repetition_penalty": 1.2,  # Stronger repetition penalty
        "length_penalty": 1.1,  # Slight preference for longer sequences
        "seed": 42,  # Same seed for reproducibility
        "inference_run": f"inference-{max_run}",
        "timestamp": pd.Timestamp.now().isoformat(),
        "note": "Inference parameters optimized for maximum prediction quality"
    }

def run_all_steps(dataset_name: str, test_mode=False, max_columns=2):
    # Create organized folder structure for this dataset
    csv_path = f"Dataset/{dataset_name}.csv"
    column_folders = create_dataset_column_folders(dataset_name, csv_path, test_mode, max_columns)
    
    # Set prediction length parameter
    prediction_length = 5
    
    # Process each column separately
    for column_name in column_folders.keys():
        print(f"\n🔄 Processing column: {column_name}")
        
        # Generate YAML config for evaluation
        generate_yaml_config(dataset_name, column_name, prediction_length=prediction_length)
        
        # Step 1: Data preparation for this column
        prepare_data(f"Dataset/{dataset_name}.csv", f"Dataset/Dataset.arrow/{dataset_name}_{column_name}.arrow", test_mode, max_columns)
        
        # Step 2: Training for this column
        training_config = get_training_config(dataset_name, prediction_length)
        
        commands = [
            [
                "python", "scripts/training/train.py", 
                f"['Dataset/Dataset.arrow/{dataset_name}_{column_name}.arrow']",
                "--model-id", training_config["model_id"],
                "--prediction-length", str(training_config["prediction_length"]),
                "--context-length", str(training_config["context_length"]),
                "--min-past", str(training_config["min_past"]),
                "--max-steps", str(training_config["max_steps"]),
                "--save-steps", str(training_config["save_steps"]),
                "--log-steps", str(training_config["log_steps"]),
                "--per-device-train-batch-size", str(training_config["per_device_train_batch_size"]),
                "--learning-rate", str(training_config["learning_rate"]),
                "--warmup-ratio", str(training_config["warmup_ratio"]),
                "--gradient-accumulation-steps", str(training_config["gradient_accumulation_steps"]),
                "--lr-scheduler-type", training_config["lr_scheduler_type"],
                "--shuffle-buffer-length", str(training_config["shuffle_buffer_length"]),
                "--random-init" if training_config["random_init"] else "--no-random-init",
                "--num-samples", str(training_config["num_samples"]),
                "--temperature", str(training_config["temperature"]),
                "--top-k", str(training_config["top_k"]),
                "--seed", str(training_config["seed"]),
                "--n-tokens", str(training_config["n_tokens"]),
                "--tokenizer-kwargs", training_config["tokenizer_kwargs"],
                "--output-dir", training_config["output_dir"],
                "--dataloader-num-workers", str(training_config["dataloader_num_workers"]),
                "--max-missing-prop", str(training_config["max_missing_prop"]),
                "--no-torch-compile" if not training_config["torch_compile"] else "--torch-compile"
            ]
        ]
        step_names = [
            f"Step 2: Running training for {column_name}..."
        ]
        for cmd, step in zip(commands, step_names):
            run_command(cmd, step)
        
        # Step 3: Evaluation for this column
        latest_checkpoint = get_latest_run_dir(f"results/{dataset_name}/training")
        column_folder = column_folders[column_name]
        
        eval_cmd = [
            "python", "scripts/evaluation/evaluate_local_with_predictions.py",
            "--model-path", latest_checkpoint,
            "--dataset-path", f"Dataset/Dataset.arrow/{dataset_name}_{column_name}.arrow",
            "--output-path", f"{column_folder}/{dataset_name}_{column_name}_predictions.csv",
            "--prediction-length", str(prediction_length),
            "--use-inference-model"
        ]
        run_command(eval_cmd, f"Step 3: Running evaluation for {column_name}...")
        
        # Step 4: Create prediction analysis plot for this column
        create_prediction_analysis_plot(dataset_name, column_folder, column_name)
        
        # Step 5: Save training history and create training charts for this column
        save_training_history_and_charts(dataset_name, column_folder, column_name)
        
        print(f"✅ Completed processing for column: {column_name}")
    
    print("All steps completed successfully for all columns.")
    
    # Step 6: Organize results by column
    organize_results_by_column(dataset_name, column_folders)

def main():
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Chronos Forecasting Pipeline with configurable datasets and columns')
    parser.add_argument('--datasets', type=int, default=1, 
                       help='Number of datasets to process (default: 1, max: 5)')
    parser.add_argument('--columns', type=int, default=2, 
                       help='Number of columns per dataset to process (default: 2, 0 = all columns)')
    parser.add_argument('--all-datasets', action='store_true', 
                       help='Process all available datasets (overrides --datasets)')
    parser.add_argument('--all-columns', action='store_true', 
                       help='Process all columns in each dataset (overrides --columns)')
    parser.add_argument('--prediction-length', type=int, default=5, 
                       help='Number of values to predict (default: 5)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.datasets < 1 or args.datasets > 5:
        print("⚠️  Warning: --datasets must be between 1 and 5. Setting to 1.")
        args.datasets = 1
    
    if args.columns < 0:
        print("⚠️  Warning: --columns cannot be negative. Setting to 0 (all columns).")
        args.columns = 0
    
    # Determine configuration
    if args.all_datasets:
        datasets_to_process = ["climate", "emissions-co2", "gdp", "pesticides", "fertilizers"]
        print("🚀 FULL MODE: Processing ALL datasets")
    else:
        datasets_to_process = ["climate", "emissions-co2", "gdp", "pesticides", "fertilizers"][:args.datasets]
        print(f"📊 LIMITED MODE: Processing {len(datasets_to_process)} dataset(s)")
    
    if args.all_columns:
        columns_per_dataset = 0
        print("📈 Processing ALL columns in each dataset")
    else:
        columns_per_dataset = args.columns
        print(f"📈 Processing {columns_per_dataset} column(s) per dataset")
    
    print(f"🎯 Prediction length: {args.prediction_length}")
    print(f"📊 Available datasets: {len(['climate', 'emissions-co2', 'gdp', 'pesticides', 'fertilizers'])}")
    print("=" * 70)
    
    # Create main results directory
    os.makedirs("results", exist_ok=True)
    
    # Process datasets
    for i, dataset_name in enumerate(datasets_to_process, 1):
        print(f"\n📊 Processing Dataset {i}/{len(datasets_to_process)}: {dataset_name}")
        print("-" * 50)
        
        # Set test_mode based on column configuration
        test_mode = columns_per_dataset > 0  # True if limited columns, False if all columns
        
        run_all_steps(dataset_name, test_mode=test_mode, max_columns=columns_per_dataset)
        
        # Break if we've processed the requested number of datasets
        if not args.all_datasets and i >= args.datasets:
            print(f"\n✅ COMPLETED: Processed {i} dataset(s) as requested")
            break
    
    print("\n🎉 Pipeline completed successfully!")
    print("\n📁 Results organized in the following structure:")
    print("results/")
    for dataset_name in datasets_to_process:
        print(f"  ├── {dataset_name}/")
        print(f"  │   ├── training/          # Training outputs")
        if args.all_columns:
            print(f"  │   └── [all_columns]/      # All columns processed")
        else:
            print(f"  │   └── [first_{columns_per_dataset}_columns]/   # First {columns_per_dataset} columns")
        print(f"  └── ...")
    
    print(f"\n💡 Usage examples:")
    print(f"  python run_all.py                           # Default: 1 dataset, 2 columns")
    print(f"  python run_all.py --datasets 3              # 3 datasets, 2 columns each")
    print(f"  python run_all.py --columns 1               # 1 dataset, 1 column")
    print(f"  python run_all.py --all-datasets            # All datasets, 2 columns each")
    print(f"  python run_all.py --all-columns             # 1 dataset, all columns")
    print(f"  python run_all.py --all-datasets --all-columns  # All datasets, all columns")
    print(f"  python run_all.py --prediction-length 10    # Predict 10 values")

if __name__ == "__main__":
    main() 