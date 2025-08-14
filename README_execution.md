# Chronos Forecasting - Execution & Evaluation Guide

This guide provides comprehensive instructions for running all evaluation workflows and using the local evaluation script with predictions.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Standard Evaluation (Paper Benchmarks)](#standard-evaluation-paper-benchmarks)
4. [Local Evaluation with Predictions](#local-evaluation-with-predictions)
5. [Complete Evaluation Workflow](#complete-evaluation-workflow)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

## 🚀 Prerequisites

Before running evaluations, ensure you have:

- Python 3.8+ installed
- CUDA-compatible GPU (recommended for faster inference)
- Access to trained Chronos models or pre-trained models from HuggingFace
- Time series datasets in GluonTS Arrow format

## 📦 Installation

### For Standard Evaluation
```bash
pip install "chronos-forecasting[evaluation] @ git+https://github.com/amazon-science/chronos-forecasting.git"
```

### For Training and Local Evaluation
```bash
# Clone the repository
git clone https://github.com/amazon-science/chronos-forecasting.git
cd chronos-forecasting

# Install in editable mode with all dependencies
pip install --editable ".[training,evaluation]"
```

## 📊 Standard Evaluation (Paper Benchmarks)

The standard evaluation computes WQL and MASE metrics for in-domain and zero-shot benchmarks as reported in the paper.

### Quick Start

```bash
# In-domain evaluation
python scripts/evaluation/evaluate.py scripts/evaluation/configs/in-domain.yaml \
    evaluation/results/chronos-t5-small-in-domain.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20

# Zero-shot evaluation
python scripts/evaluation/evaluate.py scripts/evaluation/configs/zero-shot.yaml \
    evaluation/results/chronos-t5-small-zero-shot.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20
```

### Available Models

| Model | Parameters | Use Case |
|-------|------------|----------|
| `amazon/chronos-t5-tiny` | 8M | Quick testing, limited resources |
| `amazon/chronos-t5-mini` | 20M | Development and testing |
| `amazon/chronos-t5-small` | 46M | **Recommended for evaluation** |
| `amazon/chronos-t5-base` | 200M | High accuracy, more resources |
| `amazon/chronos-t5-large` | 710M | Best accuracy, significant resources |

### Configuration Files

The evaluation uses YAML configuration files located in `scripts/evaluation/configs/`:

- `in-domain.yaml`: Datasets seen during training
- `zero-shot.yaml`: Datasets not seen during training
- `climate.yaml`: Climate-specific datasets
- `emissions-co2.yaml`: CO2 emissions datasets

### Aggregating Results

After running evaluations, compute aggregated relative scores:

```python
import pandas as pd
from scipy.stats import gmean

def agg_relative_score(model_df: pd.DataFrame, baseline_df: pd.DataFrame):
    relative_score = model_df.drop("model", axis="columns") / baseline_df.drop("model", axis="columns")
    return relative_score.agg(gmean)

# Load results
result_df = pd.read_csv("evaluation/results/chronos-t5-small-in-domain.csv").set_index("dataset")
baseline_df = pd.read_csv("evaluation/results/seasonal-naive-in-domain.csv").set_index("dataset")

# Compute aggregated scores
agg_score_df = agg_relative_score(result_df, baseline_df)
print(agg_score_df)
```

## 🔍 Local Evaluation with Predictions

The `evaluate_local_with_predictions.py` script provides comprehensive local evaluation comparing fine-tuned models against zero-shot baselines.

### Key Features

- **Fine-tuned vs Zero-shot Comparison**: Evaluates your trained model against base Chronos
- **Multiple Prediction Strategies**: Steps Ahead (SA) and Rolling Origin (RO) approaches
- **Comprehensive Output**: Clear labels for all data types and predictions
- **Automatic Configuration**: Loads saved inference parameters from training runs

### 🚀 **Quick Start - Execute from Terminal**

#### Basic Evaluation (Climate USA Model)
```bash
# Navigate to your project directory
cd chronos-forecasting

# Run the evaluation script
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5
```

#### What This Command Does:
1. **Loads your fine-tuned model** from the specified checkpoint
2. **Loads the dataset** from the Arrow file
3. **Makes predictions** using both fine-tuned and zero-shot approaches
4. **Saves results** to a CSV file with clear labels
5. **Compares performance** between your model and baseline

### 📋 **Terminal Execution Examples**

#### 1. **Basic Climate Model Evaluation**
```bash
# Evaluate Climate USA model
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5
```

#### 2. **Custom Prediction Length**
```bash
# Predict next 10 values instead of 5
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions_10steps.csv" \
    --prediction-length 10
```

#### 3. **Use Trained Model Directly**
```bash
# Use actual trained model instead of inference model (saves memory)
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5 \
    --no-use-inference-model
```

#### 4. **Faster Execution (No Zero-shot)**
```bash
# Only evaluate fine-tuned model (faster execution)
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5 \
    --no-enable-zero-shot
```

#### 5. **Evaluate Multiple Models in Sequence**
```bash
# Create a simple loop to evaluate multiple models
for model in "usa_clima" "china_clima"; do
    echo "Evaluating $model model..."
    python scripts/evaluation/evaluate_local_with_predictions.py \
        --model-path "results/climate/$model/climate_${model}_training_outputs/run-0/checkpoint-final" \
        --dataset-path "Dataset/climate_${model}.arrow" \
        --output-path "evaluation/results/climate_${model}_predictions.csv" \
        --prediction-length 5
done
```

### Command Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--model-path` | Path to trained model checkpoint | - | ✅ |
| `--dataset-path` | Path to dataset (.arrow file) | - | ✅ |
| `--output-path` | Path to save results CSV | - | ✅ |
| `--prediction-length` | Number of values to predict | 5 | ❌ |
| `--use-inference-model` | Use larger model for inference | True | ❌ |
| `--enable-zero-shot` | Enable zero-shot evaluation | True | ❌ |

### Usage Examples

#### 1. Climate Dataset Evaluation
```bash
# Basic climate model evaluation
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5
```

#### 2. Custom Prediction Length
```bash
# Predict next 10 values
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions_10steps.csv" \
    --prediction-length 10
```

#### 3. Use Trained Model Directly
```bash
# Use actual trained model instead of inference model
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5 \
    --no-use-inference-model
```

#### 4. Disable Zero-shot Comparison
```bash
# Only evaluate fine-tuned model (faster execution)
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5 \
    --no-enable-zero-shot
```

### Output Format

The script generates a CSV file with the following structure:

| timestamp | value | type |
|-----------|-------|------|
| 0 | 1.23 | actual |
| 1 | 1.45 | actual |
| ... | ... | ... |
| 95 | 2.34 | actual |
| 96 | 2.56 | predicted FT SA |
| 97 | 2.78 | predicted FT SA |
| 98 | 2.91 | predicted FT SA |
| 99 | 3.12 | predicted FT SA |
| 100 | 3.34 | predicted FT SA |

### 🎯 **What Happens When You Run the Script**

#### **Terminal Output Example:**
```bash
$ python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5

INFO:__main__:Dataset: climate, Column: usa_clima
INFO:__main__:Using larger model for inference: amazon/chronos-t5-large
INFO:__main__:Loading base model for zero-shot evaluation: amazon/chronos-t5-base
INFO:__main__:Loading dataset from Dataset/climate_usa_clima.arrow
INFO:__main__:Processing time series with 100 values
INFO:__main__:Requesting prediction_length: 5
INFO:__main__:Using inference config: samples=20, temp=0.7, top_k=15, top_p=0.9
INFO:__main__:Generated 5 predictions from 20 samples
INFO:__main__:Running zero-shot prediction with base model
INFO:__main__:Running rolling origin prediction with 3 rolls
INFO:__main__:Rolling origin: 3 successful rolls, averaged predictions
INFO:__main__:Results saved to evaluation/results/climate_usa_clima_predictions.csv
INFO:__main__:Training: 95 values
INFO:__main__:Fine-tuned Steps Ahead (FT SA): 5 values
INFO:__main__:Zero-shot Steps Ahead (ZR SA): 5 values
INFO:__main__:Fine-tuned Rolling Origin (FT RO): 5 values
INFO:__main__:Zero-shot Rolling Origin (ZR RO): 5 values
INFO:__main__:Evaluation completed successfully!
```

#### **What Each Step Does:**
1. **Model Loading**: Loads your fine-tuned model and base Chronos model
2. **Dataset Loading**: Loads the time series data from Arrow file
3. **Fine-tuned Prediction**: Makes predictions using your trained model
4. **Zero-shot Prediction**: Makes predictions using base Chronos model
5. **Rolling Origin**: Creates multiple predictions from different starting points
6. **Results Saving**: Saves everything to a CSV file with clear labels

### Data Types Explained

- **`actual`**: Training and validation data (ground truth)
- **`predicted FT SA`**: Fine-tuned Steps Ahead predictions
- **`predicted ZR SA`**: Zero-shot Steps Ahead predictions
- **`predicted FT RO`**: Fine-tuned Rolling Origin predictions
- **`predicted ZR RO`**: Zero-shot Rolling Origin predictions

## 🔄 Complete Evaluation Workflow

### Step 1: Prepare Your Environment
```bash
# Clone and setup
git clone https://github.com/amazon-science/chronos-forecasting.git
cd chronos-forecasting
pip install --editable ".[training,evaluation]"

# Verify installation
python -c "from chronos import ChronosPipeline; print('Chronos installed successfully!')"
```

### Step 2: Run Standard Evaluations
```bash
# Create results directory
mkdir -p evaluation/results

# Run in-domain evaluation
python scripts/evaluation/evaluate.py scripts/evaluation/configs/in-domain.yaml \
    evaluation/results/chronos-t5-small-in-domain.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20

# Run zero-shot evaluation
python scripts/evaluation/evaluate.py scripts/evaluation/configs/zero-shot.yaml \
    evaluation/results/chronos-t5-small-zero-shot.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20
```

### Step 3: Run Local Evaluations
```bash
# Evaluate your fine-tuned models
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5

# Repeat for other models/datasets
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/china_clima/climate_china_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_china_clima.arrow" \
    --output-path "evaluation/results/climate_china_clima_predictions.csv" \
    --prediction-length 5
```

### Step 4: Aggregate and Analyze Results
```bash
# Run aggregation script
python scripts/evaluation/agg-relative-score.py

# Or manually aggregate using Python
python -c "
import pandas as pd
from scipy.stats import gmean

# Load your results
result_df = pd.read_csv('evaluation/results/chronos-t5-small-in-domain.csv').set_index('dataset')
baseline_df = pd.read_csv('evaluation/results/seasonal-naive-in-domain.csv').set_index('dataset')

# Compute aggregated scores
agg_score_df = result_df.drop('model', axis='columns') / baseline_df.drop('model', axis='columns')
agg_score_df = agg_score_df.agg(gmean)

print('Aggregated: Relative Scores:')
print(agg_score_df)
"
```

## 🚀 How to Run All Evaluation Files

### Overview of Available Evaluation Scripts

The evaluation system consists of several interconnected scripts that work together to provide comprehensive model assessment:

| Script | Purpose | Output | Use Case |
|--------|---------|---------|----------|
| `evaluate.py` | Standard paper benchmarks | CSV with WQL/MASE metrics | Reproducing paper results |
| `evaluate_local_with_predictions.py` | Local model evaluation | CSV with predictions | Fine-tuned model assessment |
| `agg-relative-score.py` | Result aggregation | Aggregated scores | Performance comparison |

### 🎯 **Quick Terminal Commands to Run Everything**

#### **Option A: Run All Evaluations with One Command**
```bash
# Create and run the master script
cat > run_all.sh << 'EOF'
#!/bin/bash
echo "🚀 Running Complete Chronos Evaluation Pipeline"

# Create directories
mkdir -p evaluation/results

# 1. Standard evaluations
echo "📊 Running standard evaluations..."
python scripts/evaluation/evaluate.py scripts/evaluation/configs/in-domain.yaml \
    evaluation/results/chronos-t5-small-in-domain.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 --device=cuda:0 --num-samples 20

python scripts/evaluation/evaluate.py scripts/evaluation/configs/zero-shot.yaml \
    evaluation/results/chronos-t5-small-zero-shot.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 --device=cuda:0 --num-samples 20

# 2. Local evaluations
echo "🔍 Running local evaluations..."
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5

# 3. Aggregate results
echo "📊 Aggregating results..."
python scripts/evaluation/agg-relative-score.py

echo "✅ All evaluations completed!"
EOF

# Make executable and run
chmod +x run_all.sh
./run_all.sh
```

#### **Option B: Run Step by Step from Terminal**
```bash
# Step 1: Standard evaluations
echo "Step 1: Running standard evaluations..."
python scripts/evaluation/evaluate.py scripts/evaluation/configs/in-domain.yaml \
    evaluation/results/chronos-t5-small-in-domain.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 --device=cuda:0 --num-samples 20

python scripts/evaluation/evaluate.py scripts/evaluation/configs/zero-shot.yaml \
    evaluation/results/chronos-t5-small-zero-shot.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 --device=cuda:0 --num-samples 20

# Step 2: Local evaluations
echo "Step 2: Running local evaluations..."
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5

# Step 3: Aggregate results
echo "Step 3: Aggregating results..."
python scripts/evaluation/agg-relative-score.py
```

#### **Option C: One-liner for Quick Testing**
```bash
# Quick test with just local evaluation
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/quick_test.csv" \
    --prediction-length 5 --no-enable-zero-shot
```

### 🎯 **Option 1: Run Everything at Once (Recommended)**

Create a master script to run all evaluations automatically:

```bash
#!/bin/bash
# run_all_evaluations.sh

echo "🚀 Starting Complete Chronos Evaluation Pipeline"
echo "=================================================="

# Step 1: Create directories
mkdir -p evaluation/results
mkdir -p scripts/evaluation/results

# Step 2: Run standard evaluations
echo " benchmarking..."
 
echo "Running in-domain evaluation..."
python scripts/evaluation/evaluate.py scripts/evaluation/configs/in-domain.yaml \
    evaluation/results/chronos-t5-small-in-domain.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20

echo "Running zero-shot evaluation..."
python scripts/evaluation/evaluate.py scripts/evaluation/configs/zero-shot.yaml \
    evaluation/results/chronos-t5-small-zero-shot.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20

# Step 3: Run local evaluations for all available models
echo "🔍 Running local evaluations..."

# Climate USA model
if [ -d "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" ]; then
    echo "Evaluating Climate USA model..."
    python scripts/evaluation/evaluate_local_with_predictions.py \
        --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
        --dataset-path "Dataset/climate_usa_clima.arrow" \
        --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
        --prediction-length 5
else
    echo "⚠️  Climate USA model not found, skipping..."
fi

# Climate China model
if [ -d "results/climate/china_clima/climate_china_clima_training_outputs/run-0/checkpoint-final" ]; then
    echo "Evaluating Climate China model..."
    python scripts/evaluation/evaluate_local_with_predictions.py \
        --model-path "results/climate/china_clima/climate_china_clima_training_outputs/run-0/checkpoint-final" \
        --dataset-path "Dataset/climate_china_clima.arrow" \
        --output-path "evaluation/results/climate_china_clima_predictions.csv" \
        --prediction-length 5
else
    echo "⚠️  Climate China model not found, skipping..."
fi

# Step 4: Aggregate results
echo "📊 Aggregating results..."
python scripts/evaluation/agg-relative-score.py

echo "✅ All evaluations completed!"
echo "📁 Results saved in: evaluation/results/"
echo "📁 Local predictions saved in: scripts/evaluation/results/"
```

**Usage:**
```bash
# Make executable and run
chmod +x run_all_evaluations.sh
./run_all_evaluations.sh
```

### 🎯 **Option 2: Python Script for Complete Evaluation**

Create a Python script for more control and error handling:

```python
# run_all_evaluations.py
#!/usr/bin/env python3
"""
Master script to run all Chronos evaluations
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_model_exists(model_path):
    """Check if a model checkpoint exists"""
    return os.path.exists(model_path)

def main():
    print("🚀 Starting Complete Chronos Evaluation Pipeline")
    print("=" * 60)
    
    # Create directories
    os.makedirs("evaluation/results", exist_ok=True)
    os.makedirs("scripts/evaluation/results", exist_ok=True)
    
    # Step 1: Standard evaluations
    print("\n📊 Step 1: Running Standard Evaluations")
    print("-" * 40)
    
    # In-domain evaluation
    success = run_command(
        'python scripts/evaluation/evaluate.py scripts/evaluation/configs/in-domain.yaml '
        'evaluation/results/chronos-t5-small-in-domain.csv '
        '--chronos-model-id "amazon/chronos-t5-small" '
        '--batch-size=32 --device=cuda:0 --num-samples 20',
        "In-domain evaluation"
    )
    
    if not success:
        print("⚠️  Continuing with other evaluations...")
    
    # Zero-shot evaluation
    success = run_command(
        'python scripts/evaluation/evaluate.py scripts/evaluation/configs/zero-shot.yaml '
        'evaluation/results/chronos-t5-small-zero-shot.csv '
        '--chronos-model-id "amazon/chronos-t5-small" '
        '--batch-size=32 --device=cuda:0 --num-samples 20',
        "Zero-shot evaluation"
    )
    
    # Step 2: Local evaluations
    print("\n🔍 Step 2: Running Local Evaluations")
    print("-" * 40)
    
    # Define models to evaluate
    models_to_evaluate = [
        {
            "name": "Climate USA",
            "model_path": "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final",
            "dataset_path": "Dataset/climate_usa_clima.arrow",
            "output_path": "evaluation/results/climate_usa_clima_predictions.csv"
        },
        {
            "name": "Climate China",
            "model_path": "results/climate/china_clima/climate_china_clima_training_outputs/run-0/checkpoint-final",
            "dataset_path": "Dataset/climate_china_clima.arrow",
            "output_path": "evaluation/results/climate_china_clima_predictions.csv"
        }
    ]
    
    for model in models_to_evaluate:
        if check_model_exists(model["model_path"]):
            print(f"🔍 Evaluating {model['name']} model...")
            success = run_command(
                f'python scripts/evaluation/evaluate_local_with_predictions.py '
                f'--model-path "{model["model_path"]}" '
                f'--dataset-path "{model["dataset_path"]}" '
                f'--output-path "{model["output_path"]}" '
                f'--prediction-length 5',
                f"{model['name']} evaluation"
            )
        else:
            print(f"⚠️  {model['name']} model not found at {model['model_path']}, skipping...")
    
    # Step 3: Aggregate results
    print("\n📊 Step 3: Aggregating Results")
    print("-" * 40)
    
    success = run_command(
        'python scripts/evaluation/agg-relative-score.py',
        "Results aggregation"
    )
    
    print("\n" + "=" * 60)
    print("🎉 Evaluation Pipeline Completed!")
    print("📁 Results saved in: evaluation/results/")
    print("📁 Local predictions saved in: scripts/evaluation/results/")
    
    # List generated files
    print("\n📋 Generated Files:")
    results_dir = Path("evaluation/results")
    if results_dir.exists():
        for file in results_dir.glob("*.csv"):
            print(f"  - {file}")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python run_all_evaluations.py
```

### 🎯 **Option 3: Individual Commands (Manual Control)**

Run each evaluation step individually for maximum control:

```bash
# 1. Standard evaluations
echo "Running standard evaluations..."
python scripts/evaluation/evaluate.py scripts/evaluation/configs/in-domain.yaml \
    evaluation/results/chronos-t5-small-in-domain.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 --device=cuda:0 --num-samples 20

python scripts/evaluation/evaluate.py scripts/evaluation/configs/zero-shot.yaml \
    evaluation/results/chronos-t5-small-zero-shot.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 --device=cuda:0 --num-samples 20

# 2. Local evaluations
echo "Running local evaluations..."
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5

# 3. Aggregate results
echo "Aggregating results..."
python scripts/evaluation/agg-relative-score.py
```

### 🔧 **Configuration and Customization**

#### Environment Variables
```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Set batch size for memory management
export CHRONOS_BATCH_SIZE=16

# Set number of samples
export CHRONOS_NUM_SAMPLES=20
```

#### Custom Model Evaluation
```bash
# Add your own model to the evaluation pipeline
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/your_model/checkpoint-final" \
    --dataset-path "Dataset/your_dataset.arrow" \
    --output-path "evaluation/results/your_model_predictions.csv" \
    --prediction-length 10 \
    --no-enable-zero-shot  # Faster execution
```

### 📊 **Monitoring and Debugging**

#### Check Progress
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check log files
tail -f evaluation/results/*.log

# Monitor disk space
df -h
```

#### Common Issues and Solutions
```bash
# If CUDA out of memory
export CUDA_VISIBLE_DEVICES=0
python scripts/evaluation/evaluate.py ... --batch-size=16

# If model loading fails
ls -la results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final/

# If dataset not found
ls -la Dataset/
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Errors
```bash
# Error: "Failed to load model"
# Solution: Check model path and ensure checkpoint exists
ls -la results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final/
```

#### 2. CUDA Out of Memory
```bash
# Solution: Reduce batch size or use CPU
python scripts/evaluation/evaluate.py ... --batch-size=16 --device=cpu
```

#### 3. Dataset Format Issues
```bash
# Error: "Failed to load dataset"
# Solution: Ensure dataset is in GluonTS Arrow format
python -c "
from gluonts.dataset.arrow import ArrowFile
dataset = ArrowFile('Dataset/climate_usa_clima.arrow')
print(f'Dataset loaded successfully with {len(list(dataset))} series')
"
```

#### 4. Missing Dependencies
```bash
# Install missing packages
pip install torch torchvision torchaudio
pip install gluonts
pip install datasets
pip install scipy
```

### Performance Optimization

#### GPU Usage
```bash
# Check GPU availability
nvidia-smi

# Force GPU usage
export CUDA_VISIBLE_DEVICES=0
python scripts/evaluation/evaluate_local_with_predictions.py ...
```

#### Memory Management
```bash
# Use smaller models for limited memory
--chronos-model-id "amazon/chronos-t5-tiny"

# Reduce prediction length
--prediction-length 3

# Disable zero-shot for faster execution
--no-enable-zero-shot
```

## 🚀 Advanced Usage

### Batch Processing Multiple Models
```bash
#!/bin/bash
# batch_evaluate.sh

MODELS=(
    "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final"
    "results/climate/usa_clima/climate_usa_clima_training_outputs/run-1/checkpoint-final"
    "results/climate/china_clima/climate_china_clima_training_outputs/run-0/checkpoint-final"
)

DATASETS=(
    "Dataset/climate_usa_clima.arrow"
    "Dataset/climate_usa_clima.arrow"
    "Dataset/climate_china_clima.arrow"
)

for i in "${!MODELS[@]}"; do
    echo "Evaluating model ${i+1}/${#MODELS[@]}"
    python scripts/evaluation/evaluate_local_with_predictions.py \
        --model-path "${MODELS[$i]}" \
        --dataset-path "${DATASETS[$i]}" \
        --output-path "evaluation/results/model_${i+1}_predictions.csv" \
        --prediction-length 5
done
```

### Custom Evaluation Metrics
```python
# custom_evaluation.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions(results_path):
    df = pd.read_csv(results_path)
    
    # Separate actual and predicted values
    actual = df[df['type'] == 'actual']['value'].tail(5).values
    ft_sa = df[df['type'] == 'predicted FT SA']['value'].values
    zr_sa = df[df['type'] == 'predicted ZR SA']['value'].values
    
    # Calculate metrics
    ft_rmse = np.sqrt(mean_squared_error(actual, ft_sa))
    zr_rmse = np.sqrt(mean_squared_error(actual, zr_sa))
    
    ft_mae = mean_absolute_error(actual, ft_sa)
    zr_mae = mean_absolute_error(actual, zr_sa)
    
    print(f"Fine-tuned RMSE: {ft_rmse:.4f}")
    print(f"Zero-shot RMSE: {zr_rmse:.4f}")
    print(f"Fine-tuned MAE: {ft_mae:.4f}")
    print(f"Zero-shot MAE: {zr_mae:.4f}")
    
    return {
        'ft_rmse': ft_rmse,
        'zr_rmse': zr_rmse,
        'ft_mae': ft_mae,
        'zr_mae': zr_mae
    }

# Usage
if __name__ == "__main__":
    metrics = evaluate_predictions("evaluation/results/climate_usa_clima_predictions.csv")
```

### Integration with Jupyter Notebooks
```python
# evaluation_notebook.ipynb
import pandas as pd
import matplotlib.pyplot as plt
from scripts.evaluation.evaluate_local_with_predictions import *

# Load and analyze results
df = pd.read_csv("evaluation/results/climate_usa_clima_predictions.csv")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df[df['type'] == 'actual']['value'], label='Actual', linewidth=2)
plt.plot(df[df['type'] == 'predicted FT SA']['value'], label='Fine-tuned SA', marker='o')
plt.plot(df[df['type'] == 'predicted ZR SA']['value'], label='Zero-shot SA', marker='s')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Climate Model Predictions Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

## 📚 Additional Resources

- [Main Chronos Documentation](../README.md)
- [Training Guide](../scripts/README.md)
- [Model Architecture Details](../docs/)
- [Paper: Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)

## 🤝 Support

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review the [GitHub Issues](https://github.com/amazon-science/chronos-forecasting/issues)
3. Create a new issue with detailed error information
4. Include your system configuration and error logs

---

**Happy Forecasting! 🚀**
