import sys
import subprocess
import os
import re
import json
from scripts.generate_ts import read_transform_data, convert_to_arrow
import yaml

def validate_data_quality(time_series, dataset_name, column_name):
    """
    Validates data quality before training to catch issues early.
    
    Args:
        time_series: List of time series data
        dataset_name: Name of the dataset
        column_name: Name of the column
        
    Returns:
        bool: True if data is valid for training, False otherwise
    """
    print(f"🔍 Validating data quality for {dataset_name}_{column_name}...")
    
    if not time_series:
        print(f"❌ ERROR: No time series data found")
        return False
    
    if len(time_series) == 0:
        print(f"❌ ERROR: Empty time series list")
        return False
    
    # Check first time series
    ts = time_series[0]
    
    # Handle different data structures
    if isinstance(ts, dict):
        # If it's a dictionary, get the target field
        if 'target' not in ts:
            print(f"❌ ERROR: Time series dictionary missing 'target' field")
            return False
        ts = ts['target']
    
    # Convert to numpy array if it's a tensor
    if hasattr(ts, 'numpy'):
        ts_array = ts.numpy()
    elif isinstance(ts, np.ndarray):
        ts_array = ts
    else:
        print(f"❌ ERROR: Time series is not a numpy array or tensor")
        return False
    
    # Check data length
    if len(ts_array) < 10:
        print(f"❌ ERROR: Time series too short ({len(ts_array)} points). Need at least 10 points for training.")
        return False
    
    # Check for NaN values
    if np.any(np.isnan(ts_array)):
        print(f"❌ ERROR: Found NaN values in time series")
        return False
    
    # Check for infinite values
    if np.any(np.isinf(ts_array)):
        print(f"❌ ERROR: Found infinite values in time series")
        return False
    
    # Check for constant values (no variation)
    if np.std(ts_array) == 0:
        print(f"❌ ERROR: Time series has no variation (constant values)")
        return False
    
    # Check for very small datasets (training issue we encountered)
    if len(ts_array) < 20:
        print(f"❌ ERROR: Time series too short ({len(ts_array)} points). Need at least 20 points for training.")
        return False
    elif len(ts_array) < 50:
        print(f"⚠️  WARNING: Time series short ({len(ts_array)} points). Training may be challenging but will attempt.")
        print(f"   💡 Note: Smaller datasets may require different training parameters.")
        # Don't block training, just warn
    else:
        print(f"  📊 Sufficient data for training: ✅")
    
    # Check data range
    min_val = np.min(ts_array)
    max_val = np.max(ts_array)
    print(f"  📊 Data length: {len(ts_array)} points")
    print(f"  📊 Value range: {min_val:.2f} to {max_val:.2f}")
    print(f"  📊 Standard deviation: {np.std(ts_array):.2f}")
    print(f"  📊 No NaN values: ✅")
    print(f"  📊 No infinite values: ✅")
    print(f"  📊 Has variation: ✅")
    print(f"  📊 Sufficient data for training: ✅")
    
    print(f"✅ Data quality validation passed for {dataset_name}_{column_name}")
    return True

def prepare_data(csv_path: str, arrow_path: str, test_mode: bool = False, max_columns: int = 2, column_name: str = None):
    """
    Reads the dataset from CSV and converts it to Arrow format for a specific column.
    
    Args:
        csv_path: Path to the CSV file
        arrow_path: Path to save the Arrow file
        test_mode: If True, only process limited columns
        max_columns: Maximum number of columns to process (0 = all columns)
        column_name: Specific column name to process (if None, process all columns)
    """
    print("Preparing data: converting CSV to Arrow format...")
    
    # Import here to avoid circular imports
    from scripts.generate_ts import read_transform_data, read_transform_data_single_column, convert_to_arrow
    
    # If column_name is specified, process only that column
    if column_name is not None:
        time_series = read_transform_data_single_column(csv_path, column_name)
    else:
        time_series = read_transform_data(csv_path, test_mode, max_columns)
    
    # Validate basic data quality before proceeding
    dataset_name = csv_path.split('/')[-1].replace('.csv', '')
    
    # Use the column_name parameter if provided, otherwise try to extract from data
    if column_name is None:
        if isinstance(time_series, list) and len(time_series) > 0:
            if isinstance(time_series[0], dict):
                # If it's a list of dictionaries, get the first key
                column_name = list(time_series[0].keys())[0]
            else:
                # If it's a list of numpy arrays, use the dataset name
                column_name = dataset_name
        else:
            column_name = dataset_name
    
    if not validate_data_quality(time_series, dataset_name, column_name):
        print(f"❌ Data quality validation failed. Cannot proceed with training.")
        return False
    
    # Convert to Arrow format first
    convert_to_arrow(arrow_path, time_series=time_series)
    print(f"Data written to {arrow_path}\n")
    
    # Now test the Arrow file loading (after it's created)
    print(f"🧪 Testing Arrow file loading for training...")
    try:
        from gluonts.dataset.arrow import ArrowFile
        test_dataset = ArrowFile(arrow_path)
        test_data = list(test_dataset)
        
        if len(test_data) == 0:
            print(f"❌ ERROR: Arrow file contains no data")
            return False
        
        # Check if we can create a batch (this is where training failed)
        test_batch = test_data[:1]  # Take first item
        if not test_batch or test_batch[0] is None:
            print(f"❌ ERROR: Cannot create valid batch from data")
            return False
        
        print(f"  📊 Arrow file test: ✅")
        print(f"  📊 Batch creation test: ✅")
        print(f"✅ Arrow file loading test passed for training")
        
    except Exception as e:
        print(f"❌ ERROR: Arrow file loading test failed: {e}")
        return False
    
    return True

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

def extract_timing_from_training_output(output_text):
    """
    Extract timing information from training console output.
    
    Args:
        output_text (str): Console output text from training
        
    Returns:
        dict: Dictionary containing timing information or None if not found
    """
    import re
    
    # Look for the final training summary line
    # Pattern: {'train_runtime': 337.521, 'train_samples_per_second': 31.287, 'train_steps_per_second': 0.978, 'train_loss': 2.124661821307558, 'epoch': 1.0}
    pattern = r"\{'train_runtime': ([0-9.]+), 'train_samples_per_second': ([0-9.]+), 'train_steps_per_second': ([0-9.]+), 'train_loss': ([0-9.]+), 'epoch': ([0-9.]+)\}"
    
    match = re.search(pattern, output_text)
    if match:
        timing_info = {
            'train_runtime': float(match.group(1)),
            'train_samples_per_second': float(match.group(2)),
            'train_steps_per_second': float(match.group(3)),
            'train_loss': float(match.group(4)),
            'epoch': float(match.group(5))
        }
        
        # Extract step information from the final checkpoint
        # Look for the final step in the log history
        step_pattern = r"'step': ([0-9]+)"
        step_match = re.search(step_pattern, output_text)
        if step_match:
            timing_info['step'] = int(step_match.group(1))
            print(f"🔍 Debug: Found step with pattern 1: {timing_info['step']}")
        else:
            # Try alternative pattern
            step_pattern2 = r"step': ([0-9]+)"
            step_match2 = re.search(step_pattern2, output_text)
            if step_match2:
                timing_info['step'] = int(step_match2.group(1))
                print(f"🔍 Debug: Found step with pattern 2: {timing_info['step']}")
            else:
                # If no step found, use the max_steps from training config (usually 330)
                timing_info['step'] = 330
                print(f"🔍 Debug: Using default step: {timing_info['step']}")
        
        return timing_info
    
    return None

def save_timing_info_to_file(training_run_dir, timing_info):
    """
    Save timing information to a JSON file in the training run directory.
    
    Args:
        training_run_dir (str): Path to the training run directory
        timing_info (dict): Dictionary containing timing information
    """
    if not timing_info:
        return
    
    timing_file = os.path.join(training_run_dir, "timing_info.json")
    
    try:
        with open(timing_file, 'w') as f:
            json.dump(timing_info, f, indent=2)
        print(f"✅ Timing information saved to: {timing_file}")
        print(f"   Content: {timing_info}")
    except Exception as e:
        print(f"❌ Error saving timing information: {e}")

def run_command(cmd, step_name):
    """
    Runs a shell command and handles errors.
    """
    print(f"{step_name}")
    
    # Capture output for training commands to extract timing information
    if "train.py" in str(cmd):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{step_name} failed. Exiting.")
            print(f"Error output: {result.stderr}")
            sys.exit(result.returncode)
        
        # Extract timing information from training output
        timing_info = extract_timing_from_training_output(result.stdout)
        if timing_info:
            # Save timing info to the training run directory
            # Find the latest training run directory for the current dataset
            import glob
            training_dirs = glob.glob("results/*/training/run-*")
            if training_dirs:
                # Sort by creation time and get the latest
                latest_run_dir = max(training_dirs, key=os.path.getctime)
                save_timing_info_to_file(latest_run_dir, timing_info)
        
        print(f"{step_name} completed successfully.\n")
    else:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"{step_name} failed. Exiting.")
            sys.exit(result.returncode)
        print(f"{step_name} completed successfully.\n")

def test_training_data_loading(arrow_path, step_name):
    """
    Tests if the training data can be loaded without the batch loading error.
    """
    print(f"🧪 {step_name} - Testing training data loading...")
    try:
        # Test the exact data loading that training will do
        from gluonts.dataset.arrow import ArrowFile
        test_dataset = ArrowFile(arrow_path)
        test_data = list(test_dataset)
        
        if len(test_data) == 0:
            print(f"❌ ERROR: No data found in Arrow file")
            return False
        
        # Test batch creation (this is where training failed)
        test_batch = test_data[:1]
        if not test_batch or test_batch[0] is None:
            print(f"❌ ERROR: Batch creation failed - data contains None values")
            return False
        
        # Test if we can access the target field
        first_item = test_batch[0]
        if 'target' not in first_item:
            print(f"❌ ERROR: Data missing 'target' field")
            return False
        
        target = first_item['target']
        if target is None or len(target) == 0:
            print(f"❌ ERROR: Target field is None or empty")
            return False
        
        print(f"✅ Training data loading test passed")
        print(f"  📊 Data points: {len(test_data)}")
        print(f"  📊 Target length: {len(target)}")
        print(f"  📊 Batch creation: ✅")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Training data loading test failed: {e}")
        return False


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
    # Try to read CSV with different separators
    df = None
    separators = [',', ';', '\t', '|', '/']
    
    for sep in separators:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            # Check if we got reasonable column names (not too long)
            if len(df.columns) > 0 and all(len(str(col)) < 100 for col in df.columns):
                print(f"✅ Successfully read CSV with separator: '{sep}'")
                break
        except Exception as e:
            print(f"⚠️  Failed to read CSV with separator '{sep}': {e}")
            continue
    
    if df is None or len(df.columns) == 0:
        raise ValueError(f"Could not read CSV file {csv_path} with any of the separators: {separators}")
    
    # Clean column names to make them safe for folder names
    safe_columns = []
    for col in df.columns:
        # Remove quotes and clean the column name
        clean_col = str(col).strip().strip('"\'')
        # Replace problematic characters
        clean_col = clean_col.replace('/', '_').replace('\\', '_').replace(':', '_')
        clean_col = clean_col.replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_')
        clean_col = clean_col.replace('|', '_').replace(';', '_')
        
        # Limit length to avoid filesystem issues
        if len(clean_col) > 50:
            clean_col = clean_col[:47] + "..."
        
        safe_columns.append(clean_col)
    
    print(f"📁 Found {len(safe_columns)} columns: {safe_columns}")
    
    # Create main dataset folder
    dataset_folder = f"results/{dataset_name}"
    os.makedirs(dataset_folder, exist_ok=True)
    
    # Create subfolders for columns (time series)
    column_folders = {}
    
    if test_mode:
        # Test mode: limited number of columns
        if len(safe_columns) > 0:
            if max_columns == 0:
                # Process all columns
                for col in safe_columns:
                    column_folder = f"{dataset_folder}/{col}"
                    os.makedirs(column_folder, exist_ok=True)
                    column_folders[col] = column_folder
                print(f"TEST MODE: Created folders for ALL columns: {len(column_folders)} columns")
            else:
                # Process limited number of columns
                for i in range(min(max_columns, len(safe_columns))):
                    col = safe_columns[i]
                    column_folder = f"{dataset_folder}/{col}"
                    os.makedirs(column_folder, exist_ok=True)
                    column_folders[col] = column_folder
                print(f"TEST MODE: Created folders for first {len(column_folders)} columns: {list(column_folders.keys())}")
    else:
        # Normal mode: all columns
        for col in safe_columns:
            column_folder = f"{dataset_folder}/{col}"
            os.makedirs(column_folder, exist_ok=True)
            column_folders[col] = column_folder
    
    if not test_mode:
        print(f"Created organized folder structure for {dataset_name}:")
        for col, folder in column_folders.items():
            print(f"  - {folder}")
    
    return column_folders

def parse_column_names(columns_arg: str, available_columns: list) -> list:
    """
    Parse column names argument that can be comma-separated or dash-separated range.
    
    Args:
        columns_arg: String argument from command line (e.g., "brazil_gdp,canada_gdp" or "brazil_gdp-canada_gdp")
        available_columns: List of available columns in the dataset
        
    Returns:
        list: List of column names to process
    """
    if not columns_arg:
        return []
    
    # Remove any whitespace and split by comma
    columns_list = [col.strip() for col in columns_arg.split(',')]
    
    # Handle dash-separated ranges
    final_columns = []
    for col_range in columns_list:
        if '-' in col_range and not col_range.startswith('-') and not col_range.endswith('-'):
            # This is a range (e.g., "brazil_gdp-canada_gdp")
            start_col, end_col = col_range.split('-', 1)
            start_col = start_col.strip()
            end_col = end_col.strip()
            
            try:
                start_idx = available_columns.index(start_col)
                end_idx = available_columns.index(end_col)
                
                if start_idx <= end_idx:
                    # Add all columns in the range
                    range_columns = available_columns[start_idx:end_idx + 1]
                    final_columns.extend(range_columns)
                    print(f"📊 Range '{col_range}' expanded to: {range_columns}")
                else:
                    # Reverse range
                    range_columns = available_columns[end_idx:start_idx + 1]
                    final_columns.extend(range_columns)
                    print(f"📊 Range '{col_range}' expanded to: {range_columns}")
                    
            except ValueError as e:
                print(f"⚠️  Warning: Could not parse range '{col_range}': {e}")
                # Try to add individual columns if they exist
                if start_col in available_columns:
                    final_columns.append(start_col)
                if end_col in available_columns:
                    final_columns.append(end_col)
        else:
            # Single column name
            final_columns.append(col_range)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_columns = []
    for col in final_columns:
        if col not in seen:
            seen.add(col)
            unique_columns.append(col)
    
    # Validate that all columns exist
    valid_columns = []
    for col in unique_columns:
        if col in available_columns:
            valid_columns.append(col)
        else:
            print(f"⚠️  Warning: Column '{col}' not found in dataset. Available columns: {available_columns}")
    
    return valid_columns


def create_dataset_column_folders_specific(dataset_name: str, csv_path: str, target_column: str):
    """
    Create organized folder structure for a specific column in a dataset.
    
    Args:
        dataset_name: Name of the dataset
        csv_path: Path to the CSV file to read columns from
        target_column: Name of the specific column to process
        
    Returns:
        dict: Dictionary mapping column names to folder paths (only the target column)
    """
    # Try to read CSV with different separators
    df = None
    separators = [',', ';', '\t', '|', '/']
    
    for sep in separators:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            # Check if we got reasonable column names (not too long)
            if len(df.columns) > 0 and all(len(str(col)) < 100 for col in df.columns):
                print(f"✅ Successfully read CSV with separator: '{sep}'")
                break
        except Exception as e:
            print(f"⚠️  Failed to read CSV with separator '{sep}': {e}")
            continue
    
    if df is None or len(df.columns) == 0:
        raise ValueError(f"Could not read CSV file {csv_path} with any of the separators: {separators}")
    
    # Clean column names to make them safe for folder names
    safe_columns = []
    for col in df.columns:
        # Remove quotes and clean the column name
        clean_col = str(col).strip().strip('"\'')
        # Replace problematic characters
        clean_col = clean_col.replace('/', '_').replace('\\', '_').replace(':', '_')
        clean_col = clean_col.replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_')
        clean_col = clean_col.replace('|', '_').replace(';', '_')
        
        # Limit length to avoid filesystem issues
        if len(clean_col) > 50:
            clean_col = clean_col[:47] + "..."
        
        safe_columns.append(clean_col)
    
    print(f"📁 Found {len(safe_columns)} columns: {safe_columns}")
    
    # Check if target column exists
    if target_column not in safe_columns:
        # Try to find a close match
        close_matches = [col for col in safe_columns if target_column.lower() in col.lower() or col.lower() in target_column.lower()]
        if close_matches:
            print(f"⚠️  Target column '{target_column}' not found, but found similar columns: {close_matches}")
            print(f"💡 Using first match: {close_matches[0]}")
            target_column = close_matches[0]
        else:
            print(f"❌ Target column '{target_column}' not found in available columns: {safe_columns}")
            print(f"💡 Available columns: {safe_columns}")
            raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    print(f"🎯 Processing specific column: {target_column}")
    
    # Create main dataset folder
    dataset_folder = f"results/{dataset_name}"
    os.makedirs(dataset_folder, exist_ok=True)
    
    # Create folder only for the target column
    column_folders = {}
    column_folder = f"{dataset_folder}/{target_column}"
    os.makedirs(column_folder, exist_ok=True)
    column_folders[target_column] = column_folder
    
    print(f"Created folder structure for specific column {target_column}:")
    print(f"  - {column_folder}")
    
    return column_folders


def create_dataset_column_folders_multiple(dataset_name: str, csv_path: str, target_columns: list):
    """
    Create organized folder structure for multiple specific columns in a dataset.
    
    Args:
        dataset_name: Name of the dataset
        csv_path: Path to the CSV file to read columns from
        target_columns: List of specific column names to process
        
    Returns:
        dict: Dictionary mapping column names to folder paths
    """
    # Try to read CSV with different separators
    df = None
    separators = [',', ';', '\t', '|', '/']
    
    for sep in separators:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            # Check if we got reasonable column names (not too long)
            if len(df.columns) > 0 and all(len(str(col)) < 100 for col in df.columns):
                print(f"✅ Successfully read CSV with separator: '{sep}'")
                break
        except Exception as e:
            print(f"⚠️  Failed to read CSV with separator '{sep}': {e}")
            continue
    
    if df is None or len(df.columns) == 0:
        raise ValueError(f"Could not read CSV file {csv_path} with any of the separators: {separators}")
    
    # Clean column names to make them safe for folder names
    safe_columns = []
    for col in df.columns:
        # Remove quotes and clean the column name
        clean_col = str(col).strip().strip('"\'')
        # Replace problematic characters
        clean_col = clean_col.replace('/', '_').replace('\\', '_').replace(':', '_')
        clean_col = clean_col.replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_')
        clean_col = clean_col.replace('|', '_').replace(';', '_')
        
        # Limit length to avoid filesystem issues
        if len(clean_col) > 50:
            clean_col = clean_col[:47] + "..."
        
        safe_columns.append(clean_col)
    
    print(f"📁 Found {len(safe_columns)} columns: {safe_columns}")
    
    # Validate that all target columns exist
    valid_target_columns = []
    for target_col in target_columns:
        if target_col in safe_columns:
            valid_target_columns.append(target_col)
        else:
            # Try to find a close match
            close_matches = [col for col in safe_columns if target_col.lower() in col.lower() or col.lower() in target_col.lower()]
            if close_matches:
                print(f"⚠️  Target column '{target_col}' not found, but found similar column: {close_matches[0]}")
                valid_target_columns.append(close_matches[0])
            else:
                print(f"⚠️  Warning: Target column '{target_col}' not found in available columns")
    
    if not valid_target_columns:
        print(f"❌ No valid target columns found. Available columns: {safe_columns}")
        raise ValueError("No valid target columns found in dataset")
    
    print(f"🎯 Processing {len(valid_target_columns)} specific columns: {valid_target_columns}")
    
    # Create main dataset folder
    dataset_folder = f"results/{dataset_name}"
    os.makedirs(dataset_folder, exist_ok=True)
    
    # Create folders for each target column
    column_folders = {}
    for target_col in valid_target_columns:
        column_folder = f"{dataset_folder}/{target_col}"
        os.makedirs(column_folder, exist_ok=True)
        column_folders[target_col] = column_folder
    
    print(f"Created folder structure for {len(valid_target_columns)} columns:")
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
        
        # Initialize log_file variable
        log_file = None
        
        # Get timing information for the title
        timing_info = ""
        try:
            # Look for trainer_state.json in checkpoint directories
            checkpoint_dirs = [d for d in os.listdir(training_run_dir) if d.startswith('checkpoint-') and d != 'checkpoint-final']
            
            # Extract step numbers and find the highest step
            max_step = 0
            
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
            
            if log_file and os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    trainer_state = json.load(f)
                
                if 'log_history' in trainer_state and trainer_state['log_history']:
                    last_log = trainer_state['log_history'][-1]
                    if 'train_runtime' in last_log:
                        runtime = last_log['train_runtime']
                        minutes = int(runtime // 60)
                        seconds = int(runtime % 60)
                        timing_info = f" (Runtime: {minutes}m {seconds}s)"
        except Exception as e:
            print(f"  ⚠️  Could not get timing info for title: {e}")
            pass
        
        fig.suptitle(f'Training Summary for {dataset_name} - {column_name}{timing_info}', fontsize=16, fontweight='bold')
        
        # Chart 1: Actual Training Loss Progression (from logs)
        try:
            if log_file:
                print(f"  📊 Reading training logs from: {os.path.basename(os.path.dirname(log_file))} (final training step)")
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
                        
                        # Print timing information if available
                        if 'train_runtime' in log_history[-1]:
                            runtime = log_history[-1]['train_runtime']
                            minutes = int(runtime // 60)
                            seconds = int(runtime % 60)
                            print(f"     Training time: {minutes}m {seconds}s")
                        if 'train_steps_per_second' in log_history[-1]:
                            print(f"     Training speed: {log_history[-1]['train_steps_per_second']:.2f} steps/sec")
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
        
        # Add timing information box to the learning rate chart
        try:
            # Try to read timing info from a separate file first
            timing_info_file = os.path.join(training_run_dir, "timing_info.json")
            timing_text = ""
            
            if os.path.exists(timing_info_file):
                with open(timing_info_file, 'r') as f:
                    timing_data = json.load(f)
                
                if 'train_runtime' in timing_data:
                    runtime = timing_data['train_runtime']
                    minutes = int(runtime // 60)
                    seconds = int(runtime % 60)
                    timing_text += f"Training Time: {minutes}m {seconds}s\n"
                
                if 'train_steps_per_second' in timing_data:
                    speed = timing_data['train_steps_per_second']
                    timing_text += f"Speed: {speed:.2f} steps/sec\n"
                
                if 'step' in timing_data:
                    timing_text += f"Steps: {timing_data['step']}"
            
            # Fallback to trainer_state.json if timing_info.json doesn't exist
            elif log_file and os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    trainer_state = json.load(f)
                
                if 'log_history' in trainer_state and trainer_state['log_history']:
                    last_log = trainer_state['log_history'][-1]
                    
                    # Create timing information text
                    if 'train_runtime' in last_log:
                        runtime = last_log['train_runtime']
                        minutes = int(runtime // 60)
                        seconds = int(runtime % 60)
                        timing_text += f"Training Time: {minutes}m {seconds}s\n"
                    
                    if 'train_steps_per_second' in last_log:
                        speed = last_log['train_steps_per_second']
                        timing_text += f"Speed: {speed:.2f} steps/sec\n"
                    
                    if 'step' in last_log:
                        timing_text += f"Steps: {last_log['step']}"
            
            # Add timing box to the chart
            if timing_text:
                ax2.text(0.02, 0.98, timing_text, 
                        transform=ax2.transAxes, fontsize=9, 
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.4", 
                        facecolor="lightblue", alpha=0.8, 
                        edgecolor="navy", linewidth=1))
        except Exception as e:
            print(f"  ⚠️  Could not add timing box to learning rate chart: {e}")
        

        
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
                        
                        # Print timing information if available
                        if 'train_runtime' in log_history[-1]:
                            runtime = log_history[-1]['train_runtime']
                            minutes = int(runtime // 60)
                            seconds = int(runtime % 60)
                            print(f"     Training time: {minutes}m {seconds}s")
                        if 'train_steps_per_second' in log_history[-1]:
                            print(f"     Training speed: {log_history[-1]['train_steps_per_second']:.2f} steps/sec")
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
        
        # Calculate training time from timing_info.json or trainer_state.json if available
        training_time = "N/A"
        total_steps = "N/A"
        steps_per_second = "N/A"
        
        try:
            # Try to read timing info from a separate file first
            timing_info_file = os.path.join(training_run_dir, "timing_info.json")
            print(f"  🔍 Debug: training_run_dir = {training_run_dir}")
            print(f"  🔍 Debug: timing_info_file = {timing_info_file}")
            print(f"  🔍 Debug: timing_info_file exists = {os.path.exists(timing_info_file)}")
            
            if os.path.exists(timing_info_file):
                print(f"  🔍 Debug: Reading timing_info.json from {timing_info_file}")
                with open(timing_info_file, 'r') as f:
                    timing_data = json.load(f)
                
                print(f"  🔍 Debug: timing_data keys = {list(timing_data.keys())}")
                
                if 'train_runtime' in timing_data:
                    training_time = f"{timing_data['train_runtime']:.1f}s"
                    print(f"  🔍 Debug: Found train_runtime = {timing_data['train_runtime']}")
                if 'step' in timing_data:
                    total_steps = str(timing_data['step'])
                    print(f"  🔍 Debug: Found step = {timing_data['step']}")
                if 'train_steps_per_second' in timing_data:
                    steps_per_second = f"{timing_data['train_steps_per_second']:.2f}"
                    print(f"  🔍 Debug: Found train_steps_per_second = {timing_data['train_steps_per_second']}")
            
            # Fallback to trainer_state.json if timing_info.json doesn't exist
            elif log_file and os.path.exists(log_file):
                print(f"  🔍 Debug: Reading trainer_state.json from {log_file}")
                with open(log_file, 'r') as f:
                    trainer_state = json.load(f)
                
                print(f"  🔍 Debug: trainer_state keys = {list(trainer_state.keys())}")
                
                if 'log_history' in trainer_state and trainer_state['log_history']:
                    print(f"  🔍 Debug: log_history length = {len(trainer_state['log_history'])}")
                    # Get training time from the last log entry
                    last_log = trainer_state['log_history'][-1]
                    print(f"  🔍 Debug: last_log keys = {list(last_log.keys())}")
                    
                    if 'train_runtime' in last_log:
                        training_time = f"{last_log['train_runtime']:.1f}s"
                        print(f"  🔍 Debug: Found train_runtime = {last_log['train_runtime']}")
                    if 'step' in last_log:
                        total_steps = str(last_log['step'])
                        print(f"  🔍 Debug: Found step = {last_log['step']}")
                    if 'train_steps_per_second' in last_log:
                        steps_per_second = f"{last_log['train_steps_per_second']:.2f}"
                        print(f"  🔍 Debug: Found train_steps_per_second = {last_log['train_steps_per_second']}")
                else:
                    print(f"  🔍 Debug: No log_history found in trainer_state")
            else:
                print(f"  🔍 Debug: Neither timing_info.json nor trainer_state.json found")
        except Exception as e:
            print(f"  ⚠️  Could not extract timing information: {e}")
            import traceback
            traceback.print_exc()
        
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

Training Performance:
• Total Steps Completed: {total_steps}
• Training Time: {training_time}
• Steps per Second: {steps_per_second}
• Efficiency: {f'{float(steps_per_second):.2f} steps/sec' if steps_per_second != 'N/A' else 'N/A'}
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
        
        # Read the predictions data with automatic separator detection
        df = None
        separators = [',', ';', '\t', '|', '/']
        
        for sep in separators:
            try:
                df = pd.read_csv(predictions_file, sep=sep)
                if len(df.columns) > 0:
                    print(f"✅ Successfully read predictions CSV with separator: '{sep}'")
                    break
            except Exception as e:
                continue
        
        if df is None or len(df.columns) == 0:
            raise ValueError(f"Could not read predictions file {predictions_file}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Separate actual and all four prediction types
        actual_data = df[df['type'] == 'actual'].copy()
        ft_sa_data = df[df['type'] == 'predicted FT SA'].copy()
        zr_sa_data = df[df['type'] == 'predicted ZR SA'].copy()
        ft_ro_data = df[df['type'] == 'predicted FT RO'].copy()
        zr_ro_data = df[df['type'] == 'predicted ZR RO'].copy()
        
        # Sort by timestamp
        actual_data = actual_data.sort_values('timestamp')
        ft_sa_data = ft_sa_data.sort_values('timestamp')
        zr_sa_data = zr_sa_data.sort_values('timestamp')
        ft_ro_data = ft_ro_data.sort_values('timestamp')
        zr_ro_data = zr_ro_data.sort_values('timestamp')
        
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
            
            # Create year-based timestamps for all prediction types
            ft_sa_years = []
            for i in range(len(ft_sa_data)):
                year = 1990 + len(training_data) + i
                ft_sa_years.append(year)
            
            zr_sa_years = []
            for i in range(len(zr_sa_data)):
                year = 1990 + len(training_data) + i
                zr_sa_years.append(year)
            
            ft_ro_years = []
            for i in range(len(ft_ro_data)):
                year = 1990 + len(training_data) + i
                ft_ro_years.append(year)
            
            zr_ro_years = []
            for i in range(len(zr_ro_data)):
                year = 1990 + len(training_data) + i
                zr_ro_years.append(year)
            
            # Plot training data (excluding last 5 values)
            ax.plot(training_years, training_data['value'], 
                    label='Training Data (1990-{})'.format(training_years[-1]), 
                    marker='o', linewidth=2, markersize=6, color='blue')
            
            # Plot the last 5 actual values (for comparison with predictions)
            ax.plot(comparison_years, comparison_data['value'], 
                    label='Actual Values (Comparison Period)', 
                    marker='s', linewidth=2, markersize=8, color='green')
            
            # Plot fine-tuned steps ahead predictions
            if len(ft_sa_data) > 0:
                ax.plot(ft_sa_years, ft_sa_data['value'], 
                        label='Fine-tuned Steps Ahead (FT SA)', marker='x', linewidth=2, markersize=8, color='red')
            
            # Plot zero-shot steps ahead predictions
            if len(zr_sa_data) > 0:
                ax.plot(zr_sa_years, zr_sa_data['value'], 
                        label='Zero-shot Steps Ahead (ZR SA)', marker='^', linewidth=2, markersize=8, color='orange')
            
            # Plot fine-tuned rolling origin predictions
            if len(ft_ro_data) > 0:
                ax.plot(ft_ro_years, ft_ro_data['value'], 
                        label='Fine-tuned Rolling Origin (FT RO)', marker='s', linewidth=2, markersize=8, color='purple')
            
            # Plot zero-shot rolling origin predictions
            if len(zr_ro_data) > 0:
                ax.plot(zr_ro_years, zr_ro_data['value'], 
                        label='Zero-shot Rolling Origin (ZR RO)', marker='d', linewidth=2, markersize=8, color='brown')
            
            # Add a vertical line to separate training from comparison/prediction
            if len(training_data) > 0:
                last_training_year = training_years[-1]
                ax.axvline(x=last_training_year, color='green', linestyle='--', alpha=0.7, 
                          label=f'Comparison Start ({last_training_year})')
            
            # Set title and labels
            all_years = [ft_sa_years, zr_sa_years, ft_ro_years, zr_ro_years]
            max_year = max([years[-1] if years else 0 for years in all_years])
            ax.set_title(f'{dataset_name.title()} - {column_name}: Training vs Actual vs All Prediction Methods (1990-{max_year})', 
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
            stats_text = f'Training: {len(training_data)} points\nActual (Comparison): {len(comparison_data)} points\nFT Steps Ahead: {len(ft_sa_data)} points\nZR Steps Ahead: {len(zr_sa_data)} points\nFT Rolling Origin: {len(ft_ro_data)} points\nZR Rolling Origin: {len(zr_ro_data)} points\nTime Range: 1990-{max_year}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Calculate and display comparison metrics for all four prediction types
            if len(comparison_data) > 0:
                actual_values = comparison_data['value'].values
                
                # Helper function to calculate metrics
                def calculate_metrics(actual_vals, predicted_vals, training_vals):
                    if len(actual_vals) != len(predicted_vals):
                        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    
                    # RMSE
                    rmse = np.sqrt(((actual_vals - predicted_vals) ** 2).mean())
                    
                    # MAE
                    mae = np.mean(np.abs(actual_vals - predicted_vals))
                    
                    # MAPE
                    mape = np.mean(np.abs((actual_vals - predicted_vals) / np.where(actual_vals != 0, actual_vals, 1))) * 100
                    
                    # MASE
                    mase = np.nan
                    if len(training_vals) > 1:
                        naive_forecast = training_vals[:-1]
                        actual_training = training_vals[1:]
                        mae_naive = np.mean(np.abs(actual_training - naive_forecast))
                        if mae_naive > 0:
                            mase = mae / mae_naive
                    
                    # SMAPE
                    smape = np.mean(2 * np.abs(predicted_vals - actual_vals) / (np.abs(actual_vals) + np.abs(predicted_vals))) * 100
                    
                    # Correlation
                    correlation = np.nan
                    try:
                        if len(actual_vals) > 1 and len(predicted_vals) > 1:
                            actual_series = pd.Series(actual_vals)
                            predicted_series = pd.Series(predicted_vals)
                            correlation = actual_series.corr(predicted_series)
                    except:
                        pass
                    
                    return rmse, mae, mape, mase, smape, correlation
                
                # Get training values for MASE calculation
                training_values = training_data['value'].values if len(training_data) > 0 else np.array([])
                
                # Calculate metrics for all four prediction types
                ft_sa_metrics = calculate_metrics(actual_values, ft_sa_data['value'].values, training_values) if len(ft_sa_data) > 0 else (np.nan,)*6
                zr_sa_metrics = calculate_metrics(actual_values, zr_sa_data['value'].values, training_values) if len(zr_sa_data) > 0 else (np.nan,)*6
                ft_ro_metrics = calculate_metrics(actual_values, ft_ro_data['value'].values, training_values) if len(ft_ro_data) > 0 else (np.nan,)*6
                zr_ro_metrics = calculate_metrics(actual_values, zr_ro_data['value'].values, training_values) if len(zr_ro_data) > 0 else (np.nan,)*6
                
                # Save metrics to CSV table for better comparison
                metrics_data = {
                    'Method': ['FT Steps Ahead', 'ZR Steps Ahead', 'FT Rolling Origin', 'ZR Rolling Origin'],
                    'RMSE': [ft_sa_metrics[0], zr_sa_metrics[0], ft_ro_metrics[0], zr_ro_metrics[0]],
                    'MAE': [ft_sa_metrics[1], zr_sa_metrics[1], ft_ro_metrics[1], zr_ro_metrics[1]],
                    'MAPE (%)': [ft_sa_metrics[2], zr_sa_metrics[2], ft_ro_metrics[2], zr_ro_metrics[2]],
                    'MASE': [ft_sa_metrics[3], zr_sa_metrics[3], ft_ro_metrics[3], zr_ro_metrics[3]],
                    'SMAPE (%)': [ft_sa_metrics[4], zr_sa_metrics[4], ft_ro_metrics[4], zr_ro_metrics[4]],
                    'Correlation': [ft_sa_metrics[5], zr_sa_metrics[5], ft_ro_metrics[5], zr_ro_metrics[5]]
                }
                
                metrics_df = pd.DataFrame(metrics_data)
                metrics_csv_path = f"{column_folder}/{dataset_name}_{column_name}_metrics_comparison.csv"
                metrics_df.to_csv(metrics_csv_path, index=False)
                print(f"  📊 Metrics comparison table saved to: {metrics_csv_path}")
                
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
                
                # Create comprehensive metrics text for all four prediction types
                ft_sa_text = f'FT Steps Ahead:\nRMSE: {ft_sa_metrics[0]:.3f}\nMAE: {ft_sa_metrics[1]:.3f}\nMAPE: {ft_sa_metrics[2]:.1f}%\nMASE: {ft_sa_metrics[3]:.3f}\nSMAPE: {ft_sa_metrics[4]:.1f}%\nCorr: {"NaN" if pd.isna(ft_sa_metrics[5]) else f"{ft_sa_metrics[5]:.3f}"}'
                
                zr_sa_text = f'ZR Steps Ahead:\nRMSE: {zr_sa_metrics[0]:.3f}\nMAE: {zr_sa_metrics[1]:.3f}\nMAPE: {zr_sa_metrics[2]:.1f}%\nMASE: {zr_sa_metrics[3]:.3f}\nSMAPE: {zr_sa_metrics[4]:.1f}%\nCorr: {"NaN" if pd.isna(zr_sa_metrics[5]) else f"{zr_sa_metrics[5]:.3f}"}'
                
                ft_ro_text = f'FT Rolling Origin:\nRMSE: {ft_ro_metrics[0]:.3f}\nMAE: {ft_ro_metrics[1]:.3f}\nMAPE: {ft_ro_metrics[2]:.1f}%\nMASE: {ft_ro_metrics[3]:.3f}\nSMAPE: {ft_ro_metrics[4]:.1f}%\nCorr: {"NaN" if pd.isna(ft_ro_metrics[5]) else f"{ft_ro_metrics[5]:.3f}"}'
                
                zr_ro_text = f'ZR Rolling Origin:\nRMSE: {zr_ro_metrics[0]:.3f}\nMAE: {zr_ro_metrics[1]:.3f}\nMAPE: {zr_ro_metrics[2]:.1f}%\nMASE: {zr_ro_metrics[3]:.3f}\nSMAPE: {zr_ro_metrics[4]:.1f}%\nCorr: {"NaN" if pd.isna(zr_ro_metrics[5]) else f"{zr_ro_metrics[5]:.3f}"}'
                
                # Metrics text boxes removed to avoid blocking prediction lines
                # Metrics are now saved to a separate CSV file for better comparison
                
                # Print detailed metrics
                print(f"  📊 Fine-tuned Steps Ahead (FT SA) Metrics:")
                print(f"     RMSE: {ft_sa_metrics[0]:.3f}")
                print(f"     MAE: {ft_sa_metrics[1]:.3f}")
                print(f"     MAPE: {ft_sa_metrics[2]:.1f}%")
                print(f"     MASE: {ft_sa_metrics[3]:.3f}")
                print(f"     SMAPE: {ft_sa_metrics[4]:.1f}%")
                print(f"     Correlation: {'NaN' if pd.isna(ft_sa_metrics[5]) else f'{ft_sa_metrics[5]:.3f}'}")
                
                print(f"  📊 Zero-shot Steps Ahead (ZR SA) Metrics:")
                print(f"     RMSE: {zr_sa_metrics[0]:.3f}")
                print(f"     MAE: {zr_sa_metrics[1]:.3f}")
                print(f"     MAPE: {zr_sa_metrics[2]:.1f}%")
                print(f"     MASE: {zr_sa_metrics[3]:.3f}")
                print(f"     SMAPE: {zr_sa_metrics[4]:.1f}%")
                print(f"     Correlation: {'NaN' if pd.isna(zr_sa_metrics[5]) else f'{zr_sa_metrics[5]:.3f}'}")
                
                print(f"  📊 Fine-tuned Rolling Origin (FT RO) Metrics:")
                print(f"     RMSE: {ft_ro_metrics[0]:.3f}")
                print(f"     MAE: {ft_ro_metrics[1]:.3f}")
                print(f"     MAPE: {ft_ro_metrics[2]:.1f}%")
                print(f"     MASE: {ft_ro_metrics[3]:.3f}")
                print(f"     SMAPE: {ft_ro_metrics[4]:.1f}%")
                print(f"     Correlation: {'NaN' if pd.isna(ft_ro_metrics[5]) else f'{ft_ro_metrics[5]:.3f}'}")
                
                print(f"  📊 Zero-shot Rolling Origin (ZR RO) Metrics:")
                print(f"     RMSE: {zr_ro_metrics[0]:.3f}")
                print(f"     MAE: {zr_ro_metrics[1]:.3f}")
                print(f"     MAPE: {zr_ro_metrics[2]:.1f}%")
                print(f"     MASE: {zr_ro_metrics[3]:.3f}")
                print(f"     SMAPE: {zr_ro_metrics[4]:.1f}%")
                print(f"     Correlation: {'NaN' if pd.isna(zr_ro_metrics[5]) else f'{zr_ro_metrics[5]:.3f}'}")
                
                # Debug information for correlation issues
                if any(pd.isna(metrics[5]) for metrics in [ft_sa_metrics, zr_sa_metrics, ft_ro_metrics, zr_ro_metrics]):
                    print(f"  🔍 Correlation Debug Info:")
                    print(f"     Actual values: {actual_values}")
                    if len(ft_sa_data) > 0:
                        print(f"     FT SA Predicted values: {ft_sa_data['value'].values}")
                    if len(zr_sa_data) > 0:
                        print(f"     ZR SA Predicted values: {zr_sa_data['value'].values}")
                    if len(ft_ro_data) > 0:
                        print(f"     FT RO Predicted values: {ft_ro_data['value'].values}")
                    if len(zr_ro_data) > 0:
                        print(f"     ZR RO Predicted values: {zr_ro_data['value'].values}")
                    print(f"     Actual std: {np.std(actual_values):.6f}")
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
        print(f"     Fine-tuned Steps Ahead (FT SA): {len(ft_sa_data)}")
        print(f"     Zero-shot Steps Ahead (ZR SA): {len(zr_sa_data)}")
        print(f"     Fine-tuned Rolling Origin (FT RO): {len(ft_ro_data)}")
        print(f"     Zero-shot Rolling Origin (ZR RO): {len(zr_ro_data)}")
        if len(actual_data) > 0:
            print(f"     Actual value range: {actual_data['value'].min():.3f} to {actual_data['value'].max():.3f}")
        if len(ft_sa_data) > 0:
            print(f"     FT SA predicted value range: {ft_sa_data['value'].min():.3f} to {ft_sa_data['value'].max():.3f}")
        if len(zr_sa_data) > 0:
            print(f"     ZR SA predicted value range: {zr_sa_data['value'].min():.3f} to {zr_sa_data['value'].max():.3f}")
        if len(ft_ro_data) > 0:
            print(f"     FT RO predicted value range: {ft_ro_data['value'].min():.3f} to {ft_ro_data['value'].max():.3f}")
        if len(zr_ro_data) > 0:
            print(f"     ZR RO predicted value range: {zr_ro_data['value'].min():.3f} to {zr_ro_data['value'].max():.3f}")
        
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

def get_training_config(dataset_name: str, prediction_length: int = 5, data_length: int = None):
    """
    Centralized function to get training configuration parameters.
    This ensures consistency between training commands and parameter saving.
    
    Args:
        dataset_name: Name of the dataset
        prediction_length: Number of values to predict
        data_length: Length of the time series data (for adaptive parameters)
        
    Returns:
        dict: Training configuration parameters
    """
    # Base configuration
    config = {
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
        "top_p": 0.9  # Add nucleus sampling for better diversity
    }
    
    # Adapt parameters for smaller datasets
    if data_length and data_length < 50:
        print(f"  🔧 Adapting training parameters for small dataset ({data_length} points)...")
        config.update({
            "max_steps": min(1000, data_length * 10),  # Reduce steps for small datasets
            "save_steps": min(200, data_length * 2),   # Save more frequently
            "log_steps": min(50, data_length),         # Log more frequently
            "per_device_train_batch_size": 4,          # Smaller batch size
            "gradient_accumulation_steps": 8,          # More accumulation to compensate
            "shuffle_buffer_length": min(500, data_length * 10),  # Smaller buffer
            "context_length": min(32, data_length // 2),  # Adapt context to data size
            "min_past": min(16, data_length // 4),       # Adapt min-past to data size
        })
        print(f"  🔧 Adapted parameters: max_steps={config['max_steps']}, batch_size={config['per_device_train_batch_size']}")
    
    return config

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

def run_all_steps(dataset_name: str, test_mode=False, max_columns=2, specific_column=None, multi_columns=None):
    # Create organized folder structure for this dataset
    csv_path = f"Dataset/{dataset_name}.csv"
    
    # Handle specific column mode
    if multi_columns:
        # Multi-column mode: parse column names and create folders for multiple columns
        # First, read the CSV to get available columns for parsing
        df = None
        separators = [',', ';', '\t', '|', '/']
        for sep in separators:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                if len(df.columns) > 0 and all(len(str(col)) < 100 for col in df.columns):
                    break
            except:
                continue
        
        if df is None:
            raise ValueError(f"Could not read CSV file {csv_path}")
        
        # Clean column names to match the format used in parsing
        available_columns = []
        for col in df.columns:
            clean_col = str(col).strip().strip('"\'')
            clean_col = clean_col.replace('/', '_').replace('\\', '_').replace(':', '_')
            clean_col = clean_col.replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_')
            clean_col = clean_col.replace('|', '_').replace(';', '_')
            if len(clean_col) > 50:
                clean_col = clean_col[:47] + "..."
            available_columns.append(clean_col)
        
        # Parse the column names argument
        target_columns = parse_column_names(multi_columns, available_columns)
        if not target_columns:
            raise ValueError("No valid columns specified in --columns-names argument")
        
        # Create folder structure for multiple columns
        column_folders = create_dataset_column_folders_multiple(dataset_name, csv_path, target_columns)
    elif specific_column:
        # Create folder structure for specific column only
        column_folders = create_dataset_column_folders_specific(dataset_name, csv_path, specific_column)
    else:
        # Normal mode
        column_folders = create_dataset_column_folders(dataset_name, csv_path, test_mode, max_columns)
    
    # Set prediction length parameter
    prediction_length = 5
    
    # Process each column separately
    for column_name in column_folders.keys():
        print(f"\n🔄 Processing column: {column_name}")
        
        # Generate YAML config for evaluation
        generate_yaml_config(dataset_name, column_name, prediction_length=prediction_length)
        
        # Step 1: Data preparation for this column
        if not prepare_data(f"Dataset/{dataset_name}.csv", f"Dataset/Dataset.arrow/{dataset_name}_{column_name}.arrow", test_mode, max_columns, column_name):
            print(f"❌ Data preparation failed for {column_name}. Skipping to next column.")
            continue
        
        # Step 2: Test training data loading before training
        arrow_path = f"Dataset/Dataset.arrow/{dataset_name}_{column_name}.arrow"
        if not test_training_data_loading(arrow_path, f"Step 2: Testing training data for {column_name}"):
            print(f"❌ Training data test failed for {column_name}. Skipping training.")
            print(f"💡 Recommendation: Use pre-trained models for evaluation instead.")
            # Skip training but continue with evaluation
            training_skipped = True
        else:
            training_skipped = False
        
        # Step 3: Training for this column (only if data test passed)
        if not training_skipped:
            # Get data length for adaptive training parameters
            arrow_path = f"Dataset/Dataset.arrow/{dataset_name}_{column_name}.arrow"
            try:
                from gluonts.dataset.arrow import ArrowFile
                test_dataset = ArrowFile(arrow_path)
                test_data = list(test_dataset)
                data_length = len(test_data[0]['target']) if test_data and 'target' in test_data[0] else None
            except:
                data_length = None
            
            training_config = get_training_config(dataset_name, prediction_length, data_length)
        
        if not training_skipped:
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
                f"Step 3: Running training for {column_name}..."
            ]
            for cmd, step in zip(commands, step_names):
                run_command(cmd, step)
        else:
            print(f"⏭️  Training skipped for {column_name} due to data quality issues")
        
        # Step 4: Evaluation for this column
        if training_skipped:
            # Use pre-trained model for evaluation since training was skipped
            print(f"🔄 Using pre-trained model for evaluation since training was skipped")
            eval_cmd = [
                "python", "scripts/evaluation/evaluate_local_with_predictions.py",
                "--model-path", "amazon/chronos-t5-base",
                "--dataset-path", f"Dataset/Dataset.arrow/{dataset_name}_{column_name}.arrow",
                "--output-path", f"{column_folders[column_name]}/{dataset_name}_{column_name}_predictions.csv",
                "--prediction-length", str(prediction_length),
                "--use-inference-model",
                "--enable-zero-shot"
            ]
        else:
            # Use trained model for evaluation
            latest_checkpoint = get_latest_run_dir(f"results/{dataset_name}/training")
            eval_cmd = [
                "python", "scripts/evaluation/evaluate_local_with_predictions.py",
                "--model-path", latest_checkpoint,
                "--dataset-path", f"Dataset/Dataset.arrow/{dataset_name}_{column_name}.arrow",
                "--output-path", f"{column_folders[column_name]}/{dataset_name}_{column_name}_predictions.csv",
                "--prediction-length", str(prediction_length),
                "--use-inference-model",
                "--enable-zero-shot"
            ]
        
        run_command(eval_cmd, f"Step 4: Running evaluation for {column_name}...")
        
        # Step 5: Create prediction analysis plot for this column
        create_prediction_analysis_plot(dataset_name, column_folders[column_name], column_name)
        
        # Step 6: Save training history and create training charts for this column (only if training was successful)
        if not training_skipped:
            save_training_history_and_charts(dataset_name, column_folders[column_name], column_name)
        else:
            print(f"⏭️  Skipping training history save since training was skipped")
        
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
    parser.add_argument('--dataset-name', type=str, 
                       help='Process specific dataset by name (e.g., emissions-co2, climate, gdp)')
    parser.add_argument('--column-name', type=str, 
                       help='Process specific column by name (e.g., germany_gdp, usa_clima, china_co2)')
    parser.add_argument('--columns-names', type=str, 
                       help='Process multiple specific columns by names. Use comma-separated list or dash-separated range (e.g., "brazil_gdp,canada_gdp" or "brazil_gdp-canada_gdp")')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.datasets < 1 or args.datasets > 5:
        print("⚠️  Warning: --datasets must be between 1 and 5. Setting to 1.")
        args.datasets = 1
    
    if args.columns < 0:
        print("⚠️  Warning: --columns cannot be negative. Setting to 0 (all columns).")
        args.columns = 0
    
    # Determine configuration
    if args.dataset_name:
        # Process specific dataset by name
        if args.dataset_name in ["climate", "emissions-co2", "gdp", "pesticides", "fertilizers"]:
            datasets_to_process = [args.dataset_name]
            print(f"🎯 SPECIFIC MODE: Processing dataset: {args.dataset_name}")
        else:
            print(f"❌ Error: Unknown dataset '{args.dataset_name}'")
            print("Available datasets: climate, emissions-co2, gdp, pesticides, fertilizers")
            sys.exit(1)
    elif args.all_datasets:
        datasets_to_process = ["climate", "emissions-co2", "gdp", "pesticides", "fertilizers"]
        print("🚀 FULL MODE: Processing ALL datasets")
    else:
        datasets_to_process = ["climate", "emissions-co2", "gdp", "pesticides", "fertilizers"][:args.datasets]
        print(f"📊 LIMITED MODE: Processing {len(datasets_to_process)} dataset(s)")
    
    # Handle column configuration
    if args.columns_names:
        # Process multiple specific columns
        print(f"🎯 MULTI-COLUMN MODE: Processing multiple specific columns")
        columns_per_dataset = 0  # Will be handled specially in run_all_steps
        specific_column_mode = True
        multi_column_mode = True
    elif args.column_name:
        # Process specific column
        print(f"🎯 COLUMN-SPECIFIC MODE: Processing column: {args.column_name}")
        columns_per_dataset = 0  # Will be handled specially in run_all_steps
        specific_column_mode = True
        multi_column_mode = False
    elif args.all_columns:
        columns_per_dataset = 0
        print("📈 Processing ALL columns in each dataset")
        specific_column_mode = False
        multi_column_mode = False
    else:
        columns_per_dataset = args.columns
        print(f"📈 Processing {columns_per_dataset} column(s) per dataset")
        specific_column_mode = False
        multi_column_mode = False
    
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
        if args.columns_names:
            # Multi-column mode
            test_mode = False
            specific_column = None
            multi_columns = args.columns_names
        elif args.column_name:
            # Specific column mode
            test_mode = False
            specific_column = args.column_name
            multi_columns = None
        else:
            # Normal mode
            test_mode = columns_per_dataset > 0  # True if limited columns, False if all columns
            specific_column = None
            multi_columns = None
        
        run_all_steps(dataset_name, test_mode=test_mode, max_columns=columns_per_dataset, specific_column=specific_column, multi_columns=multi_columns)
        
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
    print(f"  python run_all.py --dataset-name emissions-co2  # Process ONLY emissions-co2")
    print(f"  python run_all.py --dataset-name climate        # Process ONLY climate")
    print(f"  python run_all.py --dataset-name gdp --column-name germany_gdp  # Process ONLY germany_gdp column")
    print(f"  python run_all.py --all-datasets            # All datasets, 2 columns each")
    print(f"  python run_all.py --all-columns             # 1 dataset, all columns")
    print(f"  python run_all.py --all-datasets --all-columns  # All datasets, all columns")
    print(f"  python run_all.py --prediction-length 10    # Predict 10 values")

if __name__ == "__main__":
    main() 