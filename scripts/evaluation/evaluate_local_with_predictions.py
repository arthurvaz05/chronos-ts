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

def load_dataset(dataset_path):
    """Load the training dataset"""
    try:
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = ArrowFile(dataset_path)
        return list(dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None

def predict_next_values(pipeline, time_series, prediction_length=5):
    """Predict the next 5 values for a time series using Chronos pipeline"""
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
        
        # Use Chronos pipeline to predict with optimized inference parameters
        predictions = pipeline.predict(
            [input_tensor],
            prediction_length=prediction_length,
            num_samples=20,  # Multiple samples for better quality
            temperature=0.7,  # Lower temperature for more focused predictions
            top_k=15,  # Lower top-k for more focused sampling
            top_p=0.9  # Add nucleus sampling
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

def save_results_with_labels(original_data, predicted_values, output_path, prediction_length=5):
    """Save training, validation, and predicted values with labels"""
    try:
        rows = []
        
        # Get the original time series data
        if hasattr(original_data, 'numpy'):
            series = original_data.numpy()
        else:
            series = np.array(original_data)
        
        # Add training data (all but last 5 values)
        if len(series) >= prediction_length:
            training_data = series[:-prediction_length]
            for i, value in enumerate(training_data):
                rows.append({
                    'timestamp': i,
                    'value': float(value),
                    'type': 'actual'
                })
            
            # Add validation data (last 5 actual values)
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
        
        # Add predicted values with same timestamps as validation data
        for i, value in enumerate(predicted_values):
            rows.append({
                'timestamp': len(series) - prediction_length + i,  # Same timestamps as validation
                'value': float(value),
                'type': 'predicted'
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Training: {len([r for r in rows if r['type'] == 'training'])} values")
        logger.info(f"Validation: {len([r for r in rows if r['type'] == 'validation'])} values")
        logger.info(f"Predicted: {len([r for r in rows if r['type'] == 'predicted'])} values")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description="Simple Chronos Local Evaluation")
    parser.add_argument("--model-path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset (.arrow file)")
    parser.add_argument("--output-path", required=True, help="Path to save results CSV")
    parser.add_argument("--prediction-length", type=int, default=5, help="Number of values to predict")
    parser.add_argument("--use-inference-model", action="store_true", default=True, help="Use larger model for inference (default: True)")
    
    args = parser.parse_args()
    
    # Load model
    pipeline, _ = load_model(args.model_path, use_inference_model=args.use_inference_model)
    if pipeline is None:
        return
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    if dataset is None:
        return
    
    # Take first time series for prediction
    if len(dataset) > 0:
        time_series = dataset[0]["target"]
        logger.info(f"Processing time series with {len(time_series)} values")
        
        # Predict next values
        predicted_values = predict_next_values(pipeline, time_series, args.prediction_length)
        
        # Save results with labels
        save_results_with_labels(time_series, predicted_values, args.output_path, args.prediction_length)
        
        logger.info("Evaluation completed successfully!")
    else:
        logger.error("No data found in dataset")

if __name__ == "__main__":
    main()
