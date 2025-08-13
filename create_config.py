#!/usr/bin/env python3
"""
Script to create training configuration files for Chronos models.
"""

import argparse
import yaml
from pathlib import Path


def create_training_config(datasets, output_path):
    """
    Create a training configuration file for Chronos models.
    
    Args:
        datasets: List of dataset paths
        output_path: Path where to save the config file
    """
    config = {
        "model": {
            "name": "chronos-t5-tiny",
            "model_id": "amazon/chronos-t5-tiny"
        },
        "data": {
            "train_datasets": datasets,
            "validation_datasets": datasets,  # Using same data for validation
            "prediction_length": 5,
            "context_length": 20,
            "num_series": 1000,  # Limit for training
            "max_length": 1024
        },
        "training": {
            "epochs": 10,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "save_steps": 500,
            "eval_steps": 500,
            "logging_steps": 100
        },
        "output": {
            "output_dir": "output",
            "save_total_limit": 3
        }
    }
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Training configuration saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create training configuration for Chronos models")
    parser.add_argument("--output", required=True, help="Output path for the config file")
    parser.add_argument("--datasets", nargs="+", required=True, help="List of dataset paths")
    
    args = parser.parse_args()
    
    create_training_config(args.datasets, args.output)


if __name__ == "__main__":
    main()
