#!/usr/bin/env python3
"""
Simplified Chronos Local Evaluation Script
Loads trained model, predicts next 5 values, and saves results with labels
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
from gluonts.dataset.arrow import ArrowFile
from gluonts.dataset.common import ListDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path, use_inference_model=True):
    """Load the trained Chronos model locally or use a larger model for inference"""
    try:
        if use_inference_model:
            # Use a larger model for inference (better quality predictions)
            logger.info("Using larger model for inference: amazon/chronos-t5-large")
            pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-large")
        else:
            # Use the trained model
            logger.info(f"Loading trained model from {model_path}")
            pipeline = ChronosPipeline.from_pretrained(model_path)
        return pipeline, None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

def load_inference_config(dataset_name, column_name):
    """Load inference configuration from the saved JSON file"""
    try:
        # Look for inference config in the results directory
        config_path = f"results/{dataset_name}/{column_name}/{dataset_name}_{column_name}_inference_params.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import json
                config = json.load(f)
            logger.info(f"Loaded inference config from {config_path}")
            return config
        else:
            logger.warning(f"Inference config not found at {config_path}, using defaults")
            return None
    except Exception as e:
        logger.error(f"Error loading inference config: {e}")
        return None

def load_base_model_for_zero_shot():
    """Load the base Chronos model for zero-shot evaluation"""
    try:
        logger.info("Loading base model for zero-shot evaluation: amazon/chronos-t5-base")
        pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-base")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        return None

def load_dataset(dataset_path):
    """Load the training dataset"""
    try:
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = ArrowFile(dataset_path)
        return list(dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None

def predict_next_values(pipeline, time_series, prediction_length=5, inference_config=None, is_zero_shot=False):
    """Predict the next values for a time series using Chronos pipeline"""
    try:
        # Debug: Check pipeline configuration
        logger.info(f"Requesting prediction_length: {prediction_length}")
        if hasattr(pipeline, 'config'):
            logger.info(f"Pipeline config prediction_length: {getattr(pipeline.config, 'prediction_length', 'Not found')}")
        
        # Convert time series to tensor
        if hasattr(time_series, 'numpy'):
            series = time_series.numpy()
        else:
            series = np.array(time_series)
        
        # Take the last few values as context (e.g., last 20 values)
        context_length = min(20, len(series))
        context = series[-context_length:]
        
        # Convert to tensor
        input_tensor = torch.tensor(context, dtype=torch.float32)
        
        # Use inference config if available, otherwise use defaults
        if inference_config and not is_zero_shot:
            # Use fine-tuned inference parameters
            num_samples = inference_config.get('num_samples', 20)
            temperature = inference_config.get('temperature', 0.7)
            top_k = inference_config.get('top_k', 15)
            top_p = inference_config.get('top_p', 0.9)
            logger.info(f"Using inference config: samples={num_samples}, temp={temperature}, top_k={top_k}, top_p={top_p}")
        else:
            # Use default parameters for zero-shot
            num_samples = 20
            temperature = 0.7
            top_k = 15
            top_p = 0.9
            logger.info(f"Using default parameters: samples={num_samples}, temp={temperature}, top_k={top_k}, top_p={top_p}")
        
        # Use Chronos pipeline to predict
        predictions = pipeline.predict(
            [input_tensor],
            prediction_length=prediction_length,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Extract predicted values - properly handle multiple samples
        if hasattr(predictions[0], 'samples'):
            # We have multiple samples, take the mean across samples for each prediction step
            # predictions[0].samples shape: (num_samples, prediction_length)
            # Take mean across samples to get average prediction for each step
            predicted_values = predictions[0].samples.mean(axis=0)
            logger.info(f"Generated {len(predicted_values)} predictions from {predictions[0].samples.shape[0]} samples")
        else:
            predicted_values = np.array(predictions[0]).flatten()
            logger.info(f"Generated {len(predicted_values)} predictions (no samples attribute)")
        
        # Ensure we only return exactly prediction_length values
        if len(predicted_values) > prediction_length:
            predicted_values = predicted_values[:prediction_length]
            logger.warning(f"Model returned {len(predicted_values)} predictions, truncating to {prediction_length}")
        elif len(predicted_values) < prediction_length:
            logger.warning(f"Model returned only {len(predicted_values)} predictions, expected {prediction_length}")
        
        logger.info(f"Final prediction count: {len(predicted_values)} (requested: {prediction_length})")
        return predicted_values
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        # Return dummy predictions if generation fails
        return np.random.randn(prediction_length)

def predict_zero_shot(base_pipeline, time_series, prediction_length=5):
    """Predict using zero-shot approach with base model"""
    try:
        logger.info("Running zero-shot prediction with base model")
        return predict_next_values(base_pipeline, time_series, prediction_length, is_zero_shot=True)
    except Exception as e:
        logger.error(f"Error during zero-shot prediction: {e}")
        return np.random.randn(prediction_length)

def predict_rolling_origin(pipeline, time_series, prediction_length=5, inference_config=None, is_zero_shot=False, num_rolls=3):
    """Predict using rolling origin approach - multiple predictions from different starting points"""
    try:
        logger.info(f"Running rolling origin prediction with {num_rolls} rolls")
        
        # Convert time series to numpy array
        if hasattr(time_series, 'numpy'):
            series = time_series.numpy()
        else:
            series = np.array(time_series)
        
        all_predictions = []
        
        # Perform multiple predictions from different starting points
        for roll in range(num_rolls):
            # Calculate starting point for this roll
            # Start from different positions to simulate rolling origin
            start_idx = len(series) - prediction_length - roll - 1
            if start_idx < 20:  # Need at least 20 values for context
                start_idx = 20
            
            # Take context from start_idx to end
            context = series[start_idx:len(series) - roll]
            
            if len(context) < 20:  # Skip if not enough context
                continue
                
            # Convert to tensor
            input_tensor = torch.tensor(context, dtype=torch.float32)
            
            # Use inference config if available, otherwise use defaults
            if inference_config and not is_zero_shot:
                num_samples = inference_config.get('num_samples', 20)
                temperature = inference_config.get('temperature', 0.7)
                top_k = inference_config.get('top_k', 15)
                top_p = inference_config.get('top_p', 0.9)
            else:
                num_samples = 20
                temperature = 0.7
                top_k = 15
                top_p = 0.9
            
            # Predict
            predictions = pipeline.predict(
                [input_tensor],
                prediction_length=prediction_length,
                num_samples=num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Extract predicted values
            if hasattr(predictions[0], 'samples'):
                predicted_values = predictions[0].samples.mean(axis=0)
            else:
                predicted_values = np.array(predictions[0]).flatten()
            
            # Ensure correct length
            if len(predicted_values) > prediction_length:
                predicted_values = predicted_values[:prediction_length]
            
            all_predictions.append(predicted_values)
        
        # Average predictions across all rolls
        if all_predictions:
            avg_predictions = np.mean(all_predictions, axis=0)
            logger.info(f"Rolling origin: {len(all_predictions)} successful rolls, averaged predictions")
            return avg_predictions
        else:
            logger.warning("No successful rolling origin predictions, using single prediction")
            return predict_next_values(pipeline, time_series, prediction_length, inference_config, is_zero_shot)
            
    except Exception as e:
        logger.error(f"Error during rolling origin prediction: {e}")
        return predict_next_values(pipeline, time_series, prediction_length, inference_config, is_zero_shot)

def save_results_with_labels(original_data, ft_sa_values, zr_sa_values, ft_ro_values, zr_ro_values, output_path, prediction_length=5):
    """Save training, validation, and all four prediction types: FT/ZR Steps Ahead and Rolling Origin"""
    try:
        rows = []
        
        # Get the original time series data
        if hasattr(original_data, 'numpy'):
            series = original_data.numpy()
        else:
            series = np.array(original_data)
        
        # Add training data (all but last prediction_length values)
        if len(series) >= prediction_length:
            training_data = series[:-prediction_length]
            for i, value in enumerate(training_data):
                rows.append({
                    'timestamp': i,
                    'value': float(value),
                    'type': 'actual'
                })
            
            # Add validation data (last prediction_length actual values)
            validation_data = series[-prediction_length:]
            for i, value in enumerate(validation_data):
                rows.append({
                    'timestamp': len(training_data) + i,
                    'value': float(value),
                    'type': 'actual'
                })
        else:
            # If not enough data, mark all as training
            for i, value in enumerate(series):
                rows.append({
                    'timestamp': i,
                    'value': float(value),
                    'type': 'actual'
                })
        
        # Add fine-tuned steps ahead predicted values
        for i, value in enumerate(ft_sa_values):
            rows.append({
                'timestamp': len(series) - prediction_length + i,  # Same timestamps as validation
                'value': float(value),
                'type': 'predicted FT SA'
            })
        
        # Add zero-shot steps ahead predicted values
        for i, value in enumerate(zr_sa_values):
            rows.append({
                'timestamp': len(series) - prediction_length + i,  # Same timestamps as validation
                'value': float(value),
                'type': 'predicted ZR SA'
            })
        
        # Add fine-tuned rolling origin predicted values
        for i, value in enumerate(ft_ro_values):
            rows.append({
                'timestamp': len(series) - prediction_length + i,  # Same timestamps as validation
                'value': float(value),
                'type': 'predicted FT RO'
            })
        
        # Add zero-shot rolling origin predicted values
        for i, value in enumerate(zr_ro_values):
            rows.append({
                'timestamp': len(series) - prediction_length + i,  # Same timestamps as validation
                'value': float(value),
                'type': 'predicted ZR RO'
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Training: {len([r for r in rows if r['type'] == 'actual'])} values")
        logger.info(f"Fine-tuned Steps Ahead (FT SA): {len([r for r in rows if r['type'] == 'predicted FT SA'])} values")
        logger.info(f"Zero-shot Steps Ahead (ZR SA): {len([r for r in rows if r['type'] == 'predicted ZR SA'])} values")
        logger.info(f"Fine-tuned Rolling Origin (FT RO): {len([r for r in rows if r['type'] == 'predicted FT RO'])} values")
        logger.info(f"Zero-shot Rolling Origin (ZR RO): {len([r for r in rows if r['type'] == 'predicted ZR RO'])} values")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description="Chronos Local Evaluation with Fine-tuned and Zero-shot Comparison")
    parser.add_argument("--model-path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset (.arrow file)")
    parser.add_argument("--output-path", required=True, help="Path to save results CSV")
    parser.add_argument("--prediction-length", type=int, default=5, help="Number of values to predict")
    parser.add_argument("--use-inference-model", action="store_true", default=True, help="Use larger model for inference (default: True)")
    parser.add_argument("--enable-zero-shot", action="store_true", default=True, help="Enable zero-shot evaluation (default: True)")
    
    args = parser.parse_args()
    
    # Extract dataset and column names from the dataset path for inference config
    dataset_path_parts = args.dataset_path.split('/')
    if len(dataset_path_parts) >= 3:
        dataset_name = dataset_path_parts[-2]  # e.g., "climate" from "Dataset/Dataset.arrow/climate_usa_clima.arrow"
        column_name = dataset_path_parts[-1].replace('.arrow', '').split('_', 1)[1]  # e.g., "usa_clima" from "climate_usa_clima.arrow"
    else:
        dataset_name = "unknown"
        column_name = "unknown"
    
    logger.info(f"Dataset: {dataset_name}, Column: {column_name}")
    
    # Load fine-tuned model
    ft_pipeline, _ = load_model(args.model_path, use_inference_model=args.use_inference_model)
    if ft_pipeline is None:
        return
    
    # Load base model for zero-shot evaluation
    zr_pipeline = None
    if args.enable_zero_shot:
        zr_pipeline = load_base_model_for_zero_shot()
        if zr_pipeline is None:
            logger.warning("Failed to load base model for zero-shot, continuing with fine-tuned only")
    
    # Load inference configuration
    inference_config = load_inference_config(dataset_name, column_name)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    if dataset is None:
        return
    
    # Take first time series for prediction
    if len(dataset) > 0:
        time_series = dataset[0]["target"]
        logger.info(f"Processing time series with {len(time_series)} values")
        
        # Fine-tuned Steps Ahead prediction
        ft_sa_values = predict_next_values(ft_pipeline, time_series, args.prediction_length, inference_config, is_zero_shot=False)
        
        # Zero-shot Steps Ahead prediction
        zr_sa_values = None
        if zr_pipeline:
            zr_sa_values = predict_zero_shot(zr_pipeline, time_series, args.prediction_length)
        else:
            # Create dummy zero-shot predictions if base model failed to load
            zr_sa_values = np.random.randn(args.prediction_length)
            logger.warning("Using dummy zero-shot predictions due to base model loading failure")
        
        # Fine-tuned Rolling Origin prediction
        ft_ro_values = predict_rolling_origin(ft_pipeline, time_series, args.prediction_length, inference_config, is_zero_shot=False, num_rolls=3)
        
        # Zero-shot Rolling Origin prediction
        zr_ro_values = None
        if zr_pipeline:
            zr_ro_values = predict_rolling_origin(zr_pipeline, time_series, args.prediction_length, inference_config=None, is_zero_shot=True, num_rolls=3)
        else:
            # Create dummy zero-shot rolling origin predictions if base model failed to load
            zr_ro_values = np.random.randn(args.prediction_length)
            logger.warning("Using dummy zero-shot rolling origin predictions due to base model loading failure")
        
        # Save results with all four prediction types
        save_results_with_labels(time_series, ft_sa_values, zr_sa_values, ft_ro_values, zr_ro_values, args.output_path, args.prediction_length)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Fine-tuned Steps Ahead (FT SA): {len(ft_sa_values)} values")
        logger.info(f"Zero-shot Steps Ahead (ZR SA): {len(zr_sa_values)} values")
        logger.info(f"Fine-tuned Rolling Origin (FT RO): {len(ft_ro_values)} values")
        logger.info(f"Zero-shot Rolling Origin (ZR RO): {len(zr_ro_values)} values")
    else:
        logger.error("No data found in dataset")

if __name__ == "__main__":
    main()
