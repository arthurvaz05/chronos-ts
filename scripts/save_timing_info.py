#!/usr/bin/env python3
"""
Script to save timing information from training runs.
This script extracts timing information from the console output and saves it to a JSON file.
"""

import json
import os
import sys
import re
from pathlib import Path

def extract_timing_from_console_output(output_text):
    """
    Extract timing information from console output text.
    
    Args:
        output_text (str): Console output text from training
        
    Returns:
        dict: Dictionary containing timing information
    """
    timing_info = {}
    
    # Look for the final training summary line
    # Pattern: {'train_runtime': 330.8523, 'train_samples_per_second': 31.918, 'train_steps_per_second': 0.997, 'train_loss': 2.1241879087505917, 'epoch': 1.0}
    pattern = r"\{'train_runtime': ([0-9.]+), 'train_samples_per_second': ([0-9.]+), 'train_steps_per_second': ([0-9.]+), 'train_loss': ([0-9.]+), 'epoch': ([0-9.]+)\}"
    
    match = re.search(pattern, output_text)
    if match:
        timing_info['train_runtime'] = float(match.group(1))
        timing_info['train_samples_per_second'] = float(match.group(2))
        timing_info['train_steps_per_second'] = float(match.group(3))
        timing_info['train_loss'] = float(match.group(4))
        timing_info['epoch'] = float(match.group(5))
        
        # Extract step information from the final checkpoint
        step_pattern = r"step': ([0-9]+)"
        step_match = re.search(step_pattern, output_text)
        if step_match:
            timing_info['step'] = int(step_match.group(1))
    
    return timing_info

def save_timing_info(training_run_dir, timing_info):
    """
    Save timing information to a JSON file in the training run directory.
    
    Args:
        training_run_dir (str): Path to the training run directory
        timing_info (dict): Dictionary containing timing information
    """
    if not timing_info:
        print("⚠️  No timing information to save")
        return
    
    timing_file = os.path.join(training_run_dir, "timing_info.json")
    
    try:
        with open(timing_file, 'w') as f:
            json.dump(timing_info, f, indent=2)
        print(f"✅ Timing information saved to: {timing_file}")
        print(f"   Content: {timing_info}")
    except Exception as e:
        print(f"❌ Error saving timing information: {e}")

def main():
    """
    Main function to extract and save timing information.
    """
    if len(sys.argv) < 3:
        print("Usage: python save_timing_info.py <training_run_dir> <console_output_file>")
        print("Example: python save_timing_info.py results/pesticides/training/run-7 console_output.txt")
        sys.exit(1)
    
    training_run_dir = sys.argv[1]
    console_output_file = sys.argv[2]
    
    if not os.path.exists(training_run_dir):
        print(f"❌ Training run directory does not exist: {training_run_dir}")
        sys.exit(1)
    
    if not os.path.exists(console_output_file):
        print(f"❌ Console output file does not exist: {console_output_file}")
        sys.exit(1)
    
    # Read console output
    try:
        with open(console_output_file, 'r') as f:
            console_output = f.read()
    except Exception as e:
        print(f"❌ Error reading console output file: {e}")
        sys.exit(1)
    
    # Extract timing information
    timing_info = extract_timing_from_console_output(console_output)
    
    if timing_info:
        # Save timing information
        save_timing_info(training_run_dir, timing_info)
    else:
        print("⚠️  No timing information found in console output")
        print("   Make sure the console output contains the final training summary line")

if __name__ == "__main__":
    main()
